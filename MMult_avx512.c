#include <immintrin.h>
/* Create macros so that the matrices are stored in column-major order */
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]
#define kc 128
#define mc 256
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
    for (int i=0; i<m; i += 8){        /* Loop over the rows of C */
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
  // 将4*k的矩阵（k个列之间内存不连续）打包成k个列之间内存连续存储
  for (int j = 0; j < k; j++) {
    *dest++ = A(0, j);
    *dest++ = A(1, j);
    *dest++ = A(2, j);
    *dest++ = A(3, j);
    *dest++ = A(4, j);
    *dest++ = A(5, j);
    *dest++ = A(6, j);
    *dest++ = A(7, j);
  }
}
void PackMatrixB(int k, double *b, int ldb, double *dest) {
  // 将k*4的矩阵（k个行之间内存不连续）打包成k个行之间内存连续存储
  for (int i = 0; i < k; i++) {
    *dest++ = B(i, 0);
    *dest++ = B(i, 1);
    *dest++ = B(i, 2);
    *dest++ = B(i, 3);
    *dest++ = B(i, 4);
    *dest++ = B(i, 5);
    *dest++ = B(i, 6);
    *dest++ = B(i, 7);
  }
}
void AddDot24x8(int k, double* a, int lda, double* b, int ldb, double* c, int ldc) {
  // register double c_00 = 0.0, c_01 = 0.0, c_02 = 0.0, c_03 = 0.0, a_0p;
  // register double c_10 = 0.0, c_11 = 0.0, c_12 = 0.0, c_13 = 0.0, a_1p;
  // register double c_20 = 0.0, c_21 = 0.0, c_22 = 0.0, c_23 = 0.0, a_2p;
  // register double c_30 = 0.0, c_31 = 0.0, c_32 = 0.0, c_33 = 0.0, a_3p;

  // C
  __m512d c00_to_70 = _mm512_setzero_pd(),
          c01_to_71 = _mm512_setzero_pd(),
          c02_to_72 = _mm512_setzero_pd(),
          c03_to_73 = _mm512_setzero_pd(),
          c04_to_74 = _mm512_setzero_pd(),
          c05_to_75 = _mm512_setzero_pd(),
          c06_to_76 = _mm512_setzero_pd(),
          c07_to_77 = _mm512_setzero_pd();
          
  // B
  __m512d bp0, bp1, bp2, bp3, bp4, bp5, bp6, bp7;
  // A
  __m512d a0p_to_7p;
  for (int p = 0; p < k; p++) {
    // a_0p = A(0, p);
    // a_1p = A(1, p);
    // a_2p = A(2, p);
    // a_3p = A(3, p);
    a0p_to_7p = _mm512_load_pd(a);
    a += 8;
    
    bp0 = _mm512_set1_pd(*b);
    bp1 = _mm512_set1_pd(*(b + 1));
    bp2 = _mm512_set1_pd(*(b + 2));
    bp3 = _mm512_set1_pd(*(b + 3));
    bp4 = _mm512_set1_pd(*(b + 4));
    bp5 = _mm512_set1_pd(*(b + 5));
    bp6 = _mm512_set1_pd(*(b + 6));
    bp7 = _mm512_set1_pd(*(b + 7));
    b += 8;
    // // 共享了对A的访问，重复利用A
    c00_to_70 += a0p_to_7p * bp0;
    c01_to_71 += a0p_to_7p * bp1;
    c02_to_72 += a0p_to_7p * bp2;
    c03_to_73 += a0p_to_7p * bp3;
    c04_to_74 += a0p_to_7p * bp4;
    c05_to_75 += a0p_to_7p * bp5;
    c06_to_76 += a0p_to_7p * bp6;
    c07_to_77 += a0p_to_7p * bp7;
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
    double C_incr[64] __attribute__((aligned(64)));
    _mm512_store_pd(C_incr + 0, c00_to_70);
    _mm512_store_pd(C_incr + 8, c01_to_71);
    _mm512_store_pd(C_incr + 16, c02_to_72);
    _mm512_store_pd(C_incr + 24, c03_to_73);
    _mm512_store_pd(C_incr + 32, c04_to_74);
    _mm512_store_pd(C_incr + 40, c05_to_75);
    _mm512_store_pd(C_incr + 48, c06_to_76);
    _mm512_store_pd(C_incr + 56, c07_to_77);

    for (int j = 0; j < 8; j++) {
        C(0, j) += C_incr[j * 8 + 0];
        C(1, j) += C_incr[j * 8 + 1];
        C(2, j) += C_incr[j * 8 + 2];
        C(3, j) += C_incr[j * 8 + 3];
        C(4, j) += C_incr[j * 8 + 4];
        C(5, j) += C_incr[j * 8 + 5];
        C(6, j) += C_incr[j * 8 + 6];
        C(7, j) += C_incr[j * 8 + 7];
    }
}

