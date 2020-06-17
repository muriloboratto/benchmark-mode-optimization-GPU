# Benchmark Matrix multiply on GPU Environment

The aim of this benchmark is learn how to analyse modules for C using the API CUDA, OPENMP5 and OPENACC. 
A practical example to see how it can be used and to see a real example of the speed gains. 
The results are impressive for the effort and performance on the supercomputacional environment.

----
## Command Line Arguments

Example:  bash START.sh <supercomputer> [[[--comparison file] | [--help]]

  (required) Specifies the name of supercomputer (word) will be execute

     <supercomputer> - airis | ogbon

     file - mm_blas  | mm_cublas 

     $ bash START.sh ogbon --comparison mm_blas

----
## How to execute?

> bash START.sh ogbon --comparison mm_blas


----
## Hierachy
	         |--------------------------|             |------------|
                | TIME | SPEEDUP | MEMORY  |--has-a----->|   RESULTS  |
                |--------------------------|             |------------|
                           ^                                ^  ^
                          /                                 |  |
                       has-a                                |  |
                        /                                   |  |
                       /           |--------------|         |  |
                      /            |    PLOTS     |-is-a----|  |
                     /             |--------------|            |
                    /                                        is-a 
                   /                                           |
    |----------------|                                  |------------|
    |     OBJECT     |--has-a-------------------------->| PROFILING  |
    |----------------|                                  |------------|
   
                      
                    

----
## Codes

----
### Sequential

~~~c++
void mm(double *A, double *B, double *C, int n){

 for(int i = 0; i < n; i++) 
  for(int j = 0; j < n; j++)
    for(int k = 0; k < n; k++) 
      C[i*n+j]+=A[i*n+k]*B[k*n+j];

}
~~~

----
### BLAS

~~~c++
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

 dgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

}
~~~

----
### CUBLAS

~~~c++
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
   
}
~~~

----
### OpenMP 5

~~~c++
void mm_omp5(double *A, double *B, double *C, int n){

int i, j, k;

#pragma omp target data map(to:A[:n*n], B[:n*n], n) map(from:C[:n*n])
 #pragma omp target teams distribute parallel for private(i,j,k)
   for(i = 0; i < n; i++) 
    for(j = 0; j < n; j++)
      for(k = 0; k < n; k++) 
        C[i*n+j] += A[i*n+k] * B[k*n+j];

}
~~~

### OpenACC

~~~c++
void mm_openacc(double *A, double *B, double *C, int n){

int i, j, k;

#pragma acc data present_or_copyin(A[:n*n], B[:n*n], n) copyout(C[:n*n])
 #pragma acc parallel 
   #pragma acc loop
     for(i = 0; i < n; i++)
      for(j = 0; j < n; j++)
        for(k = 0; k < n; k++)
          C[i*n+j] += A[i*n+k] * B[k*n+j];

}
~~~

### CUDA

~~~c++
__global__ void kernel(double *A, double *B, double *C, int n) {
  
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

if(i < n && j < n)
    for( int k = 0; k < n; k++) 
       C[i*n+j] += A[i*n+k] * B[k*n+j];

}
~~~

----
## Acknowledgements

- [1] rai.bizerra@fieb.org.br - Raí Bizerra
- [2] silvano.junior@fieb.org.br - Silvano Júnior



