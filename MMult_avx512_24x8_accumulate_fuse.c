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
    _mm512_storeu_pd(dest + 0, _mm512_loadu_pd(a_ptr + 0));
    _mm512_storeu_pd(dest + 8, _mm512_loadu_pd(a_ptr + 8));
    _mm512_storeu_pd(dest + 16, _mm512_loadu_pd(a_ptr + 16));
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
  __m512d bp0;
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

    bp0 = _mm512_set1_pd(*(b + 0));
    c00_to_70 += a0p_to_7p * bp0;
    c80_to_150 += a8p_to_15p * bp0;
    c160_to_230 += a16p_to_23p * bp0;

    bp0 = _mm512_set1_pd(*(b + 1));
    c01_to_71 += a0p_to_7p * bp0;
    c81_to_151 += a8p_to_15p * bp0;
    c161_to_231 += a16p_to_23p * bp0;

    bp0 = _mm512_set1_pd(*(b + 2));
    c02_to_72 += a0p_to_7p * bp0;
    c82_to_152 += a8p_to_15p * bp0;
    c162_to_232 += a16p_to_23p * bp0;

    bp0 = _mm512_set1_pd(*(b + 3));
    c03_to_73 += a0p_to_7p * bp0;
    c83_to_153 += a8p_to_15p * bp0;
    c163_to_233 += a16p_to_23p * bp0;

    bp0 = _mm512_set1_pd(*(b + 4));
    c04_to_74 += a0p_to_7p * bp0;
    c84_to_154 += a8p_to_15p * bp0;
    c164_to_234 += a16p_to_23p * bp0;

    bp0 = _mm512_set1_pd(*(b + 5));
    c05_to_75 += a0p_to_7p * bp0;
    c85_to_155 += a8p_to_15p * bp0;
    c165_to_235 += a16p_to_23p * bp0;

    bp0 = _mm512_set1_pd(*(b + 6));
    c06_to_76 += a0p_to_7p * bp0;
    c86_to_156 += a8p_to_15p * bp0;
    c166_to_236 += a16p_to_23p * bp0;

    bp0 = _mm512_set1_pd(*(b + 7));
    c07_to_77 += a0p_to_7p * bp0;
    c87_to_157 += a8p_to_15p * bp0;
    c167_to_237 += a16p_to_23p * bp0;

    b += 8;
  }
    _mm512_storeu_pd(&C(0 , 0), _mm512_add_pd(_mm512_loadu_pd(&C(0 , 0)), c00_to_70));
    _mm512_storeu_pd(&C(8 , 0), _mm512_add_pd(_mm512_loadu_pd(&C(8 , 0)), c80_to_150));
    _mm512_storeu_pd(&C(16, 0), _mm512_add_pd(_mm512_loadu_pd(&C(16, 0)), c160_to_230));

    _mm512_storeu_pd(&C(0 , 1), _mm512_add_pd(_mm512_loadu_pd(&C(0 , 1)), c01_to_71));
    _mm512_storeu_pd(&C(8 , 1), _mm512_add_pd(_mm512_loadu_pd(&C(8 , 1)), c81_to_151));
    _mm512_storeu_pd(&C(16, 1), _mm512_add_pd(_mm512_loadu_pd(&C(16, 1)), c161_to_231));

    _mm512_storeu_pd(&C(0 , 2), _mm512_add_pd(_mm512_loadu_pd(&C(0 , 2)), c02_to_72));
    _mm512_storeu_pd(&C(8 , 2), _mm512_add_pd(_mm512_loadu_pd(&C(8 , 2)), c82_to_152));
    _mm512_storeu_pd(&C(16, 2), _mm512_add_pd(_mm512_loadu_pd(&C(16, 2)), c162_to_232));

    _mm512_storeu_pd(&C(0 , 3), _mm512_add_pd(_mm512_loadu_pd(&C(0 , 3)), c03_to_73));
    _mm512_storeu_pd(&C(8 , 3), _mm512_add_pd(_mm512_loadu_pd(&C(8 , 3)), c83_to_153));
    _mm512_storeu_pd(&C(16, 3), _mm512_add_pd(_mm512_loadu_pd(&C(16, 3)), c163_to_233));

    _mm512_storeu_pd(&C(0 , 4), _mm512_add_pd(_mm512_loadu_pd(&C(0 , 4)), c04_to_74));
    _mm512_storeu_pd(&C(8 , 4), _mm512_add_pd(_mm512_loadu_pd(&C(8 , 4)), c84_to_154));
    _mm512_storeu_pd(&C(16, 4), _mm512_add_pd(_mm512_loadu_pd(&C(16, 4)), c164_to_234));

    _mm512_storeu_pd(&C(0 , 5), _mm512_add_pd(_mm512_loadu_pd(&C(0 , 5)), c05_to_75));
    _mm512_storeu_pd(&C(8 , 5), _mm512_add_pd(_mm512_loadu_pd(&C(8 , 5)), c85_to_155));
    _mm512_storeu_pd(&C(16, 5), _mm512_add_pd(_mm512_loadu_pd(&C(16, 5)), c165_to_235));

    _mm512_storeu_pd(&C(0 , 6), _mm512_add_pd(_mm512_loadu_pd(&C(0 , 6)), c06_to_76));
    _mm512_storeu_pd(&C(8 , 6), _mm512_add_pd(_mm512_loadu_pd(&C(8 , 6)), c86_to_156));
    _mm512_storeu_pd(&C(16, 6), _mm512_add_pd(_mm512_loadu_pd(&C(16, 6)), c166_to_236));

    _mm512_storeu_pd(&C(0 , 7), _mm512_add_pd(_mm512_loadu_pd(&C(0 , 7)), c07_to_77));
    _mm512_storeu_pd(&C(8 , 7), _mm512_add_pd(_mm512_loadu_pd(&C(8 , 7)), c87_to_157));
    _mm512_storeu_pd(&C(16, 7), _mm512_add_pd(_mm512_loadu_pd(&C(16, 7)), c167_to_237));

}

