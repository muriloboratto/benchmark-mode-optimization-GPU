/*
@(#)File:           mm_cublas.cu
@(#)Last changed:   220200611 15:51:00 
@(#)Purpose:        Matrix Multiply in C using CUBLAS API
@(#)Author:         Murilo Boratto - murilo.boratto@fieb.org.br
@(#)Usage:         
@(*) Hotocompile:   nvcc mm_cublas.cu -o mm_cublas -Xcompiler -fopenmp -lcublas

@(*) Hotoexecute:  ./mm_cublas <size problem>
                   ./mm_cublas     16
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "math.h"
#include "cublas_v2.h"


void mm_cublas(double *A_host, double *B_host, double *C_host, int size){

  double alpha = 1.;
  double beta =  0.;
  int m = size;
  int n = size; 
  int k = size;
  int lda = size;
  int ldb = size;
  int ldc = size;
            
  double *A_device;
  double *B_device;
  double *C_device;
  
  cudaMalloc((void**)&A_device, size * size * sizeof(double) ); 
  cudaMalloc((void**)&B_device, size * size * sizeof(double) ); 
  cudaMalloc((void**)&C_device, size * size * sizeof(double) ); 

  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSetMatrix(size, size, sizeof(double), A_host, size, A_device, size);
  cublasSetMatrix(size, size, sizeof(double), B_host, size, B_device, size);
  cublasSetMatrix(size, size, sizeof(double), C_host, size, C_device, size);
  
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A_device, lda, B_device, ldb, &beta, C_device, ldc);
 
  cublasGetMatrix(size, size, sizeof(double), C_device, size, C_host, size);

  cudaFree(A_device);
  cudaFree(B_device);
  cudaFree(C_device);
  
  cublasDestroy(handle);
   
}/*mm_cublas*/


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
 double t1, t2;

 double  *A = (double *) malloc (sizeof(double) * n * n);
 double  *B = (double *) malloc (sizeof(double) * n * n);
 double  *C = (double *) malloc (sizeof(double) * n * n);

 fill_matrix(A, n);
 fill_matrix(B, n);

 
 t1 = omp_get_wtime();

   mm_cublas(A, B, C, n); 

 t2 = omp_get_wtime();

 printf("%d\t%f\n", n, t2-t1);

 //print_matrix(A, n);
 //print_matrix(B, n);
 //print_matrix(C, n);

 return 0;

}/*main*/



