/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot( int, double *, int, double *, double * );
void AddDot1x4(int k, double* A, int lda, double* B, int ldb, double* C, int ldc);

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+= 4 ){        /* Loop over the columns of C */
    for ( i=0; i<m; i+= 1 ){        /* Loop over the rows of C */
      // 对最外层循环进行了步长为4的循环展开，也就是在每个内层循环中，
      // A的每个行向量要与四个B的列向量进行内积
      AddDot1x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}
//将这四个函数调用内联到AddDot1x4中，然后将这四个for循环合并
void AddDot1x4(int k, double* a, int lda, double* b, int ldb, double* c, int ldc) {
  // AddDot(k, &A(0, 0), lda, &B(0, 0), &C(0, 0));
  // AddDot(k, &A(0, 0), lda, &B(0, 1), &C(0, 1));
  // AddDot(k, &A(0, 0), lda, &B(0, 2), &C(0, 2));
  // AddDot(k, &A(0, 0), lda, &B(0, 3), &C(0, 3));
  // 由于C(0, 0)到C(0, 3)要重复访问，而A(0, p)在每个循环内部要重复访问，
  // 所以可以使用寄存器缓存C和A的数据
  register double c_00 = 0.0, c_01 = 0.0, c_02 = 0.0, c_03 = 0.0, a_0p;
  for (int p = 0; p < k; p++) {
    a_0p = A(0, p);
    // 共享了对A的访问，重复利用A
    c_00 += a_0p * B(p, 0);
    c_01 += a_0p * B(p, 1);
    c_02 += a_0p * B(p, 2);
    c_03 += a_0p * B(p, 3);
  }
  C(0,0) += c_00;
  C(0,1) += c_01;
  C(0,2) += c_02;
  C(0,3) += c_03;
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
