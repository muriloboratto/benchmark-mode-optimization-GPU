/*
 * @(#)File:           mm.c
 * @(#)Last changed:   220200611 15:51:00 
 * @(#)Purpose:        Matrix Multiply in C
 * @(#)Author:         Murilo Boratto - murilo.boratto@fieb.org.br
 * @(#)Usage:         
 * @(*) Hotocompile:   gcc mm.c -o mm -fopenmp
 *
 * @(*) Hotoexecute:  ./mm <size problem>
 *                    ./mm     16
 *                    */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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

   mm(A, B, C, n);

 t2 = omp_get_wtime();

 printf("%d\t%f\n", n, t2-t1);

 //print_matrix(A, n);
 //print_matrix(B, n);
 //print_matrix(C, n);
    
  return 0;
      
 }/*main*/
   
