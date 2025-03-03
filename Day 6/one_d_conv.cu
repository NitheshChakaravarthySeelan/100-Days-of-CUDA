#include <iostream>
#include <cuda_runtime.h>
#define Mask_width 5

__constant__ float M[Mask_width]; // Conv filter
__global__ float void conv_kernel(const float* A, const float* C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i<n){
        float result = 0.0;
        for (int k = -Mask_width/2; k<=Mask_width/2;k++){
            if (i+k >= 0 && i+k <n) {
                result += A[i+k] * M[k+Mask_width/2];
            }
            C[i] = result;
        }
    }
}

int main() {
    int n = 10;
    float A[n],C[n];
    float d_M[Mask_width];

    for (int i=0; i<Mask_width;i++) {
        d_M[i] = i;
    }
    for (int i=0;i<n;i++) {
        A[i] = i;
    }

    float* d_A;
    float* d_C;
    cudaMalloc((void**)&d_A, n * sizeof(float));
    cudaMalloc((void**)&d_C, n * sizeof(float));

    cudaMemcpy(d_A,A,n*sizeof(float),cudaMemcpyHostToDevice);
    checkCudaError("Failed to copy input data to device");

    cudaMemcpyToSymbol(M,d_M,Mask_width*sizeof(float));
    checkCudaError("Failed to copy mask to device");

    dim3 blockSize(256);
    dim3 dimGrid((n + blockSize.x - 1) / blockSize.x);
    oned_conv_kernel<<<dimGrid, blockSize>>>(d_A, d_C, n);
    cudaDeviceSynchronize();
    checkCudaError("Kernel execution failed");
}