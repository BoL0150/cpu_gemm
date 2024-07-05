#include <immintrin.h>
/* Create macros so that the matrices are stored in column-major order */
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot( int, double *, int, double *, double * );
void AddDot24x8(int k, double* A, int lda, double* B, int ldb, double* C, int ldc);

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+= 4 ){        /* Loop over the columns of C */
    for ( i=0; i<m; i+= 4 ){        /* Loop over the rows of C */
      // 对最外层循环进行了步长为4的循环展开，也就是在每个内层循环中，
      // A的每个行向量要与四个B的列向量进行内积
      AddDot24x8(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}
// 将这四个函数调用内联到AddDot1x4中，然后将这四个for循环合并
// 再将四个AddDot1x4内联到AddDot4x4中。相当于是四个长为K的行向量乘以四个长为K的列向量，对k进行遍历，
// 那么在一个循环内部就是长为四的列向量外积长为四的行向量，得到C中4 * 4的一块
// 然后每个循环都对这个4 * 4的块进行累加
void AddDot24x8(int k, double* a, int lda, double* b, int ldb, double* c, int ldc) {
  // register double c_00 = 0.0, c_01 = 0.0, c_02 = 0.0, c_03 = 0.0, a_0p;
  // register double c_10 = 0.0, c_11 = 0.0, c_12 = 0.0, c_13 = 0.0, a_1p;
  // register double c_20 = 0.0, c_21 = 0.0, c_22 = 0.0, c_23 = 0.0, a_2p;
  // register double c_30 = 0.0, c_31 = 0.0, c_32 = 0.0, c_33 = 0.0, a_3p;

  // C
  __m128d c00_10 = _mm_setzero_pd(), c20_30 = _mm_setzero_pd(), 
          c01_11 = _mm_setzero_pd(), c21_31 = _mm_setzero_pd(), 
          c02_12 = _mm_setzero_pd(), c22_32 = _mm_setzero_pd(), 
          c03_13 = _mm_setzero_pd(), c23_33 = _mm_setzero_pd();
  // B
  __m128d bp0, bp1, bp2, bp3;
  // A
  __m128d a0p_1p, a2p_3p;
  for (int p = 0; p < k; p++) {
    // a_0p = A(0, p);
    // a_1p = A(1, p);
    // a_2p = A(2, p);
    // a_3p = A(3, p);
    a0p_1p = _mm_load_pd(&A(0, p));
    a2p_3p = _mm_load_pd(&A(2, p));
    
    bp0 = _mm_loaddup_pd(&B(p, 0));
    bp1 = _mm_loaddup_pd(&B(p, 1));
    bp2 = _mm_loaddup_pd(&B(p, 2));
    bp3 = _mm_loaddup_pd(&B(p, 3));
    // // 共享了对A的访问，重复利用A

    c00_10 += a0p_1p * bp0;
    c20_30 += a2p_3p * bp0;
    c01_11 += a0p_1p * bp1;
    c21_31 += a2p_3p * bp1;
    c02_12 += a0p_1p * bp2;
    c22_32 += a2p_3p * bp2;
    c03_13 += a0p_1p * bp3;
    c23_33 += a2p_3p * bp3;
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
    double C_incr[16] __attribute__((aligned(64)));
    _mm_store_pd(C_incr + 0, c00_10);
    _mm_store_pd(C_incr + 2, c20_30);
    _mm_store_pd(C_incr + 4, c01_11);
    _mm_store_pd(C_incr + 6, c21_31);
    _mm_store_pd(C_incr + 8, c02_12);
    _mm_store_pd(C_incr + 10,c22_32);
    _mm_store_pd(C_incr + 12,c03_13);
    _mm_store_pd(C_incr + 14,c23_33);

    for (int j = 0; j < 4; j++) {
        C(0, j) += C_incr[j * 4 + 0];
        C(1, j) += C_incr[j * 4 + 1];
        C(2, j) += C_incr[j * 4 + 2];
        C(3, j) += C_incr[j * 4 + 3];
    }
}
