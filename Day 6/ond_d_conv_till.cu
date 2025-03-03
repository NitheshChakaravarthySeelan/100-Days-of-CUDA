#include <iostream>
#include <cuda_runtime.h>
#define Mask_width 5

__constant__ float M[Mask_width];
__global__ void conv_kernel_tilling(const float* A, float* C, int n) {
    int tdx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tdx;

    __shared__ float S_A[32 + Mask_width - 1];

    if (i<n) {
        S_A[tdx + Mask_width/2] = A[i];
    }

    if (tdx < Mask_width/2){
        int left_idx = blockDim.x * blockIdx.x - (Mask_width/2) + tdx;
        if (left_idx >= 0) {
            S_A[tdx] = A[left_idx];
        }
        else {
            S_A[tdx] = 0.0f;
        }
    }

    if (tdx < Mask_width/2) {
        int right_idx = blockIdx.x * blockDim.x + blockDim.x + threadIdx;
        if (right_idx < n){
            S_A[tdx + blockDim.x + Mask_width/2] = A[right_idx];
        }
        else {
            S_A[tdx + blockDim.x + Mask_width/2] = 0.0f;
        }
    }
    __syncthreads();

    if (i<n) { 
        float result = 0.0f;
        for (int k = 0; k<Mask_width;k++){
            int idx = tdx + k;
            if ((i+k-Mask_width/2) >= 0 && (i+k-Mask_width/2)<n){
                result += S_A[idx] * M[k];
            }
        }
        C[i] = result;
    }
}