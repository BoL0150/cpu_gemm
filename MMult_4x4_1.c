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
  register double c_00 = 0.0, c_01 = 0.0, c_02 = 0.0, c_03 = 0.0, a_0p;
  register double c_10 = 0.0, c_11 = 0.0, c_12 = 0.0, c_13 = 0.0, a_1p;
  register double c_20 = 0.0, c_21 = 0.0, c_22 = 0.0, c_23 = 0.0, a_2p;
  register double c_30 = 0.0, c_31 = 0.0, c_32 = 0.0, c_33 = 0.0, a_3p;

  for (int p = 0; p < k; p++) {
    a_0p = A(0, p);
    // 共享了对A的访问，重复利用A
    c_00 += a_0p * B(p, 0);
    c_01 += a_0p * B(p, 1);
    c_02 += a_0p * B(p, 2);
    c_03 += a_0p * B(p, 3);

    a_1p = A(1, p);
    c_10 += a_1p * B(p, 0);
    c_11 += a_1p * B(p, 1);
    c_12 += a_1p * B(p, 2);
    c_13 += a_1p * B(p, 3);

    a_2p = A(2, p);
    c_20 += a_2p * B(p, 0);
    c_21 += a_2p * B(p, 1);
    c_22 += a_2p * B(p, 2);
    c_23 += a_2p * B(p, 3);

    a_3p = A(3, p);
    c_30 += a_3p * B(p, 0);
    c_31 += a_3p * B(p, 1);
    c_32 += a_3p * B(p, 2);
    c_33 += a_3p * B(p, 3);
  }
  C(0,0) += c_00;
  C(0,1) += c_01;
  C(0,2) += c_02;
  C(0,3) += c_03;

  C(1,0) += c_10;
  C(1,1) += c_11;
  C(1,2) += c_12;
  C(1,3) += c_13;

  C(2,0) += c_20;
  C(2,1) += c_21;
  C(2,2) += c_22;
  C(2,3) += c_23;

  C(3,0) += c_30;
  C(3,1) += c_31;
  C(3,2) += c_32;
  C(3,3) += c_33;
}
/* Create macro to let X( i ) equal the ith element of x */

#define X(i) x[ (i)*incx ]

// x是行向量，y是列向量
void AddDot( int k, double *x, int incx,  double *y, double *gamma )
{
  int p;
  // 由于X是行向量，而矩阵的存储方式又是列优先，所以访问行向量的相邻元素需要每次隔一个步长，这个步长就是矩阵的
  // lead dimension，对列优先存储来说是矩阵的行数，对行优先存储来说是矩阵的列数
  for ( p=0; p<k; p++ ){
    *gamma += X( p ) * y[ p ];     
  }
}
