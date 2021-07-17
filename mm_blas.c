/*
 * @(#)File:           mm_blas.c
 * @(#)Last changed:   220200611 15:51:00 
 * @(#)Purpose:        Matrix Multiply in C using BLAS
 * @(#)Author:         Murilo Boratto - murilo.boratto@fieb.org.br
 * @(#)Usage:         
 * @(*) Hotocompile:   gcc mm_blas.c -o mm_blas -lblas -lgfortran -fopenmp
 *
 * @(*) Hotoexecute:  ./mm_blas <size problem>
 *                    ./mm_blas     16
 *                    */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern void  dgemm_(char *transa, char *transb, int  *m, int *n, int  *k, double *alpha, double *a, int  *lda, double *b, int  *ldb, double *beta, double *c, int *ldc);
 
void mm_blas(double *A, double *B, double *C, int size){

 char transa ='N';
 char transb ='N';
  
 double alpha = 1.;
 double beta =  0.;
  
 int m = size;
 int n = size; 
 int k = size;
  
 int lda = size;
 int ldb = size;
 int ldc = size;


 dgemm_(&transa, 
 	      &transb, 
 	      &m, 
 	      &n, 
 	      &k, 
 	      &alpha, 
 	      A, 
 	      &lda, 
 	      B, 
 	      &ldb, 
 	      &beta, 
 	      C, 
 	      &ldc);

}/*mm_blas*/


void mm(double *A, double *B, double *C, int n){

int i, j, k;

 for (i = 0; i < n; i++) 
  for (j = 0; j < n; j++)
    for (k = 0; k < n; k++) 
      C[i*n+j]+=A[i*n+k]*B[k*n+j];

}/*mm*/

void fill_matrix(double *A, int n){

  int i,j;
 
  for(i = 0; i < n; i++)
    for(j = 0; j < n; j++)
      A[i*n+j] = rand()%(10-1)*1;
  
}/*fill_matrix*/


void print_matrix(double *A, int n){

  int i,j;

  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++)
      printf("%1.2f\t", A[i*n+j]);
    printf("\n");
  }

  printf("\n");

}/*print_matrix*/



int main(int argc, char **argv){

 int n = atoi(argv[1]);  
 int i, j, k;
 double t1, t2;

 double  *A = (double *) malloc (sizeof(double) * n * n);
 double  *B = (double *) malloc (sizeof(double) * n * n);
 double  *C = (double *) malloc (sizeof(double) * n * n);

 fill_matrix(A, n);
 fill_matrix(B, n);

 t1 = omp_get_wtime();

   mm_blas(A, B, C, n);

 t2 = omp_get_wtime();

 printf("%d\t%f\n", n, t2-t1);

 //print_matrix(A, n);
 //print_matrix(B, n);
 //print_matrix(C, n);

   return 0;
 
 }/*main*/
 
 
 
