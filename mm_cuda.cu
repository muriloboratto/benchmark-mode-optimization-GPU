/*
@(#)File:           mm_cuda.cu
@(#)Last changed:   220200520 09:05:00 
@(#)Purpose:        Matrix Multiply in CUDA with GRID 2D BLOCKSIZE 2D
@(#)Author:         Murilo Boratto - murilo.boratto@fieb.org.br
@(#)Usage:         
@(*) Hotocompile:   nvcc mm_cuda.cu -o mm_cuda -Xcompiler -fopenmp
@(*) Hotoexecute:  ./mm_cuda <size problem>
                   ./mm_cuda 16 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

__global__ void kernel(double *A, double *B, double *C, int n) {
  
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

if(i < n && j < n)
    for( int k = 0; k < n; k++) 
       C[i*n+j] += A[i*n+k] * B[k*n+j];

}/*mult_matrix*/
 

void mult_matrix_cpu(double *A, double *B, double *C, int n) {

 for(int i = 0; i < n; i++) 
     for(int j = 0; j < n; j++)
        for(int k = 0; k < n; k++) 
           C[i*n+j]+=A[i*n+k]*B[k*n+j];
        
}/*mult_matrix_cpu*/

void fill_matrix(double *A, int n){
 
  for(int i=0; i < n; i++)
    for(int j=0; j < n; j++)
      A[i*n+j] = rand()%(10-1)*1;
   
}/*inicializa_matriz*/


void print_matrix(double *A, int n){

  for(int i=0; i < n; i++){
    for(int j=0; j < n; j++)
      printf("%1.2f\t", A[i*n+j]);
    printf("\n");
  }

  printf("\n");

}/*print_matrix*/


int main(int argc, char **argv){

    int n = atoi(argv[1]);
    int sizeblock = 16;
    double t1, t2;

    /*Host*/
    double *A_host=(double *) malloc (n * n * sizeof(double));
    double *B_host=(double *) malloc (n * n * sizeof(double));
    double *C_host=(double *) malloc (n * n * sizeof(double));
        
    fill_matrix(A_host, n);
    fill_matrix(B_host, n);
      
    //print_matrix(A_host,n);
    //print_matrix(B_host,n);

    /*Device*/
    double *A_device;
    double *B_device;
    double *C_device;
	
    cudaMalloc((void**)&A_device, n * n * sizeof(double) ); 
    cudaMalloc((void**)&B_device, n * n * sizeof(double) ); 
    cudaMalloc((void**)&C_device, n * n * sizeof(double) ); 

    t1 = omp_get_wtime();

    cudaMemcpy(A_device, A_host, n * n * sizeof(double), cudaMemcpyHostToDevice ); 
    cudaMemcpy(B_device, B_host, n * n * sizeof(double), cudaMemcpyHostToDevice ); 
	
    /*Computational GRID: (Grid: 2D Block: 2D)*/
    dim3 dimGrid ( (int) ceil( (float) n / sizeblock), (int) ceil( (float) n / sizeblock) );
    dim3 dimBlock( sizeblock, sizeblock);  
   
              kernel<<< dimGrid,dimBlock >>>(A_device, B_device, C_device, n);        
   
    cudaMemcpy(C_host, C_device, n * n * sizeof(double), cudaMemcpyDeviceToHost ); 

    t2 = omp_get_wtime();

    printf("%d\t%f\n", n, t2-t1);
    
    //print_matrix(C_host, n );

    cudaFree(A_device );
    cudaFree(B_device );
    cudaFree(C_device );
  
    return 0;

}/*main*/

