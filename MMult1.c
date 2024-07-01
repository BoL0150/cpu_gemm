/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot( int, double *, int, double *, double * );

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=1 ){        /* Loop over the columns of C */
    for ( i=0; i<m; i+=1 ){        /* Loop over the rows of C */
      // C(i,j)等于A的行向量乘以B的列向量
      // 将一个行向量与另一个列向量的内积运算封装为AddDot函数
      AddDot( k, &A( i,0 ), lda, &B( 0,j ), &C( i,j ) );
    }
  }
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
