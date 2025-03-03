#define LOAD_SIZE 32
#include <iostream>
#include <cuda_runtime.h>

__global__ void prefixsum_kernel(float* A,float* C,int N) {
    int tdx = threadIdx.x;
    int i = 2 * blockDim.x * blockIdx.x + tdx;

    __shared__ float S_A[LOAD_SIZE];

    if (i<N){
        S_A[tdx] = A[i];
    }
    if (i+blockDim.x < N) {
        S_A[tdx + blockDim.x] = A[i+blockDim.x];
    }
    __syncthreads();

    for(int i=1;i<=blockDim.x;i++){
        __syncthreads();
        int j = i * 2 * (tdx+1) -1;
        if (j<LOAD_SIZE) {
            S_A[j] += S_A[j-i];
        }
    }
    __syncthreads();

    for(int i=LOAD_SIZE/4;i>=1;i/=2) {
        __syncthreads();
        int j = i * 2 * (tdx+1) -1 ;
        if (j<LOAD_SIZE - i){
            S_A[j+i] += S_A[j];
        }
        __syncthreads();
    }

    if (i<N){
        C[i] = S_A[tdx];
    }
    if (i<N-blockDim.x){
        C[i+blockDim.x] = S_A[tdx + blockDim.x];
    }
    __syncthreads();
}