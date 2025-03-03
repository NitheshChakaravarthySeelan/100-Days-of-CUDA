#include <cuda_runtime.h>
#include <iostream>

__global__ void LeakyReluKernel(const float *A, float *B,float slope, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N){
        if (A[index] >0){
            B[index] = A[index];
        }
        else{
            B[index] = A[index] * slope;
        }
    }
}

void CudaLeakyRelu(float *A,float *B,float slope,int N){
    int ThreadPerBlock = 256;
    int BlocksPerGrid = (N+ThreadPerBlock-1) / ThreadPerBlock;
    LeakyReluKernel<<<BlocksPerGrid,ThreadPerBlock>>>(A,B,slope,N);
}