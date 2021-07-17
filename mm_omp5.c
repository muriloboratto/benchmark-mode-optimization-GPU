/*
 *  @(#)File:           mm_omp5.c
 *  @(#)Last changed:   220200520 09:05:00 
 *  @(#)Purpose:        Matrix Multiply in OpenMP5 on GPU
 *  @(#)Author:         Murilo Boratto - murilo.boratto@fieb.org.br
 *  @(#)Usage:         
 *  @(*) Hotocompile:   clang mm_omp5.c -o mm_omp5 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
 *  @(*) Hotoexecute:   ./mm_omp5 <size problem>
 *                      ./mm_omp5     16                     
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


void mm_omp5(double *A, double *B, double *C, int n){

int i, j, k;

#pragma omp target data map(to:A[:n*n], B[:n*n], n) map(from:C[:n*n])
 #pragma omp target teams distribute parallel for simd
   for(i = 0; i < n; i++) 
    for(j = 0; j < n; j++)
      for(k = 0; k < n; k++) 
        C[i*n+j] += A[i*n+k] * B[k*n+j];

}/*mm_omp5*/

void fill_matrix(double *A, int n){

  int i,j;
 
  for(i = 0; i < n; i++)
    for(j = 0; j < n; j++)
      A[i*n+j] = rand()%(10-1)*1;
  
}/*fill_matriz*/


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

 fill_matrix(A,n);
 fill_matrix(B,n);

t1 =  omp_get_wtime();

  mm_omp5(A, B, C, n);
 
 t2 =  omp_get_wtime();
 
 printf("%d\t%f\n", n, t2-t1);

   //print_matrix(A,n);
   //print_matrix(B,n);
   //print_matrix(C,n);
    
   return 0;
      
 }/*main*/
            
            
