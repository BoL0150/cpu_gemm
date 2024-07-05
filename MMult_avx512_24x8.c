#include <immintrin.h>
/* Create macros so that the matrices are stored in column-major order */
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]
#define kc 240
#define mc 240
#define n_max 2000

/* Routine for computing C = A * B + C */

void AddDot( int, double *, int, double *, double * );
void AddDot24x8(int k, double* A, int lda, double* B, int ldb, double* C, int ldc);
void InnerKernel(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc, int need_pack_b);
void PackMatrixA(int k, double *a, int lda, double *dest);
void PackMatrixB(int k, double *b, int ldb, double *dest);

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  for (int p = 0; p < k; p += kc) {
    int pb = (k - p < kc) ? k - p : kc;
    for (int i = 0; i < m; i += mc) {
      int ib = (m - i < mc) ? m - i : mc;
      // 由于A的多个分块都与B的同一个分块相乘，所以B的每个分块只需要在第一次运算时被pack
      InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc, (i == 0));
    }
  }
}
void InnerKernel(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc, int need_pack_b) {
  // 由于需要从A中load8个数据，所以A数组需要对齐；而B数组只需要广播，所以B数组不需要对齐
  double packedA[m * k] __attribute__((aligned(64)));
  static double packedB[kc * n_max];
  for (int j=0; j<n; j += 8){        /* Loop over the columns of C */
    if (need_pack_b) {
      PackMatrixB(k, &B(0, j), ldb, &packedB[j * k]);
    }
    for (int i=0; i<m; i += 24){        /* Loop over the rows of C */
      if (j == 0) {
        PackMatrixA(k, &A(i, 0), lda, &packedA[i * k]);
      }
      // 对最外层循环进行了步长为4的循环展开，也就是在每个内层循环中，
      // A的每个行向量要与四个B的列向量进行内积
      AddDot24x8(k, &packedA[i * k], lda, &packedB[j *k], ldb, &C(i, j), ldc);
    }
  }
}
void PackMatrixA(int k, double *a, int lda, double *dest) {
  // 将24*k的矩阵（k个列之间内存不连续）打包成k个列之间内存连续存储
  for (int j = 0; j < k; j++) {
    double* a_ptr = &A(0, j);
    *(dest + 0) = *(a_ptr + 0);
    *(dest + 1) = *(a_ptr + 1);
    *(dest + 2) = *(a_ptr + 2);
    *(dest + 3) = *(a_ptr + 3);
    *(dest + 4) = *(a_ptr + 4);
    *(dest + 5) = *(a_ptr + 5);
    *(dest + 6) = *(a_ptr + 6);
    *(dest + 7) = *(a_ptr + 7);
    *(dest + 8) = *(a_ptr + 8);
    *(dest + 9) = *(a_ptr + 9);
    *(dest + 10) = *(a_ptr + 10);
    *(dest + 11) = *(a_ptr + 11);
    *(dest + 12) = *(a_ptr + 12);
    *(dest + 13) = *(a_ptr + 13);
    *(dest + 14) = *(a_ptr + 14);
    *(dest + 15) = *(a_ptr + 15);
    *(dest + 16) = *(a_ptr + 16);
    *(dest + 17) = *(a_ptr + 17);
    *(dest + 18) = *(a_ptr + 18);
    *(dest + 19) = *(a_ptr + 19);
    *(dest + 20) = *(a_ptr + 20);
    *(dest + 21) = *(a_ptr + 21);
    *(dest + 22) = *(a_ptr + 22);
    *(dest + 23) = *(a_ptr + 23);
    dest += 24;
  }
}
void PackMatrixB(int k, double *b, int ldb, double *dest) {
  double* b_i0_ptr = &B(0, 0), 
        * b_i1_ptr = &B(0, 1),
        * b_i2_ptr = &B(0, 2),
        * b_i3_ptr = &B(0, 3),
        * b_i4_ptr = &B(0, 4),
        * b_i5_ptr = &B(0, 5),
        * b_i6_ptr = &B(0, 6),
        * b_i7_ptr = &B(0, 7);
  // 将k*8的矩阵（k个行之间内存不连续）打包成k个行之间内存连续存储
  for (int i = 0; i < k; i++) {
    *(dest+0) = *b_i0_ptr++;
    *(dest+1) = *b_i1_ptr++;
    *(dest+2) = *b_i2_ptr++;
    *(dest+3) = *b_i3_ptr++;
    *(dest+4) = *b_i4_ptr++;
    *(dest+5) = *b_i5_ptr++;
    *(dest+6) = *b_i6_ptr++;
    *(dest+7) = *b_i7_ptr++;
    dest += 8;
  }
}
void AddDot24x8(int k, double* a, int lda, double* b, int ldb, double* c, int ldc) {
  // C
  __m512d c00_to_70 = _mm512_setzero_pd(),
          c80_to_150 = _mm512_setzero_pd(),
          c160_to_230 = _mm512_setzero_pd(),

          c01_to_71 = _mm512_setzero_pd(),
          c81_to_151 = _mm512_setzero_pd(),
          c161_to_231 = _mm512_setzero_pd(),

          c02_to_72 = _mm512_setzero_pd(),
          c82_to_152 = _mm512_setzero_pd(),
          c162_to_232 = _mm512_setzero_pd(),

          c03_to_73 = _mm512_setzero_pd(),
          c83_to_153 = _mm512_setzero_pd(),
          c163_to_233 = _mm512_setzero_pd(),

          c04_to_74 = _mm512_setzero_pd(),
          c84_to_154 = _mm512_setzero_pd(),
          c164_to_234 = _mm512_setzero_pd(),

          c05_to_75 = _mm512_setzero_pd(),
          c85_to_155 = _mm512_setzero_pd(),
          c165_to_235 = _mm512_setzero_pd(),

          c06_to_76 = _mm512_setzero_pd(),
          c86_to_156 = _mm512_setzero_pd(),
          c166_to_236 = _mm512_setzero_pd(),

          c07_to_77 = _mm512_setzero_pd(),
          c87_to_157 = _mm512_setzero_pd(),
          c167_to_237 = _mm512_setzero_pd();
          
  // B
  __m512d bp0, bp1, bp2, bp3, bp4, bp5, bp6, bp7;
  // A
  __m512d a0p_to_7p, a8p_to_15p, a16p_to_23p;
  for (int p = 0; p < k; p++) {
    // a_0p = A(0, p);
    // a_1p = A(1, p);
    // a_2p = A(2, p);
    // a_3p = A(3, p);
    a0p_to_7p = _mm512_load_pd(a);
    a8p_to_15p = _mm512_load_pd(a + 8);
    a16p_to_23p = _mm512_load_pd(a + 16);
    a += 24;
    
    bp0 = _mm512_set1_pd(*b);
    bp1 = _mm512_set1_pd(*(b + 1));
    bp2 = _mm512_set1_pd(*(b + 2));
    bp3 = _mm512_set1_pd(*(b + 3));
    bp4 = _mm512_set1_pd(*(b + 4));
    bp5 = _mm512_set1_pd(*(b + 5));
    bp6 = _mm512_set1_pd(*(b + 6));
    bp7 = _mm512_set1_pd(*(b + 7));
    b += 8;

    c00_to_70 += a0p_to_7p * bp0;
    c80_to_150 += a8p_to_15p * bp0;
    c160_to_230 += a16p_to_23p * bp0;

    c01_to_71 += a0p_to_7p * bp1;
    c81_to_151 += a8p_to_15p * bp1;
    c161_to_231 += a16p_to_23p * bp1;

    c02_to_72 += a0p_to_7p * bp2;
    c82_to_152 += a8p_to_15p * bp2;
    c162_to_232 += a16p_to_23p * bp2;

    c03_to_73 += a0p_to_7p * bp3;
    c83_to_153 += a8p_to_15p * bp3;
    c163_to_233 += a16p_to_23p * bp3;

    c04_to_74 += a0p_to_7p * bp4;
    c84_to_154 += a8p_to_15p * bp4;
    c164_to_234 += a16p_to_23p * bp4;

    c05_to_75 += a0p_to_7p * bp5;
    c85_to_155 += a8p_to_15p * bp5;
    c165_to_235 += a16p_to_23p * bp5;

    c06_to_76 += a0p_to_7p * bp6;
    c86_to_156 += a8p_to_15p * bp6;
    c166_to_236 += a16p_to_23p * bp6;

    c07_to_77 += a0p_to_7p * bp7;
    c87_to_157 += a8p_to_15p * bp7;
    c167_to_237 += a16p_to_23p * bp7;

  }
  // _mm_store_pd(&C(0, 0), c00_10);
  // _mm_store_pd(&C(2, 0), c20_30);
  // _mm_store_pd(&C(0, 1), c01_11);
  // _mm_store_pd(&C(2, 1), c21_31);
  // _mm_store_pd(&C(0, 2), c02_12);
  // _mm_store_pd(&C(2, 2), c22_32);
  // _mm_store_pd(&C(0, 3), c03_13);
  // _mm_store_pd(&C(2, 3), c23_33);
  // 注意，不能使用上面的写法！因为每个AddDot4*4函数是对C中的4 * 4的方块进行累加，
  // 而上面的写法是直接将其覆盖。而simd指令集没有累加的指令，所以我们只能先将向量寄存器中
  // 的数值保存到数组中，然后再将数组中的值累加到C中
  // 并且由于store指令要求16字节对齐，所以声明临时数组时要对齐
    double C_incr[192] __attribute__((aligned(64)));
    _mm512_store_pd(C_incr + 0, c00_to_70);
    _mm512_store_pd(C_incr + 8, c80_to_150);
    _mm512_store_pd(C_incr + 16, c160_to_230);

    _mm512_store_pd(C_incr + 24, c01_to_71);
    _mm512_store_pd(C_incr + 32, c81_to_151);
    _mm512_store_pd(C_incr + 40, c161_to_231);

    _mm512_store_pd(C_incr + 48, c02_to_72);
    _mm512_store_pd(C_incr + 56, c82_to_152);
    _mm512_store_pd(C_incr + 64, c162_to_232);

    _mm512_store_pd(C_incr + 72, c03_to_73);
    _mm512_store_pd(C_incr + 80, c83_to_153);
    _mm512_store_pd(C_incr + 88, c163_to_233);

    _mm512_store_pd(C_incr + 96, c04_to_74);
    _mm512_store_pd(C_incr + 104, c84_to_154);
    _mm512_store_pd(C_incr + 112, c164_to_234);

    _mm512_store_pd(C_incr + 120, c05_to_75);
    _mm512_store_pd(C_incr + 128, c85_to_155);
    _mm512_store_pd(C_incr + 136, c165_to_235);

    _mm512_store_pd(C_incr + 144, c06_to_76);
    _mm512_store_pd(C_incr + 152, c86_to_156);
    _mm512_store_pd(C_incr + 160, c166_to_236);

    _mm512_store_pd(C_incr + 168, c07_to_77);
    _mm512_store_pd(C_incr + 176, c87_to_157);
    _mm512_store_pd(C_incr + 184, c167_to_237);

    for (int j = 0; j < 8; j++) {
        C(0, j) += C_incr[j * 24 + 0];
        C(1, j) += C_incr[j * 24 + 1];
        C(2, j) += C_incr[j * 24 + 2];
        C(3, j) += C_incr[j * 24 + 3];
        C(4, j) += C_incr[j * 24 + 4];
        C(5, j) += C_incr[j * 24 + 5];
        C(6, j) += C_incr[j * 24 + 6];
        C(7, j) += C_incr[j * 24 + 7];
        C(8, j) += C_incr[j * 24 + 8];
        C(9, j) += C_incr[j * 24 + 9];
        C(10, j) += C_incr[j * 24 + 10];
        C(11, j) += C_incr[j * 24 + 11];
        C(12, j) += C_incr[j * 24 + 12];
        C(13, j) += C_incr[j * 24 + 13];
        C(14, j) += C_incr[j * 24 + 14];
        C(15, j) += C_incr[j * 24 + 15];
        C(16, j) += C_incr[j * 24 + 16];
        C(17, j) += C_incr[j * 24 + 17];
        C(18, j) += C_incr[j * 24 + 18];
        C(19, j) += C_incr[j * 24 + 19];
        C(20, j) += C_incr[j * 24 + 20];
        C(21, j) += C_incr[j * 24 + 21];
        C(22, j) += C_incr[j * 24 + 22];
        C(23, j) += C_incr[j * 24 + 23];
    }
}

