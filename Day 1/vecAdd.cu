#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void vecAddKernel(float* A,float* B,float* C,int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* B, float* C, int n)
{
    int size = n * sizeof(float);
    float *d_A, *d_B , *d_C;

    cudaMalloc((void **) &d_A,size);
    cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_B,size);
    cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_C,size);
    vecAddKernel <<< ceil(n/256.0), 256>>>(d_A,d_B,d_C,n);

    cudaDeviceSynchronize();

    cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    const int n = 1000;

    float *h_A = new float[n];
    float *h_B = new float[n];
    float *h_C = new float[n];

    for (int i=0; i<n; i++){
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2*i);
    }

    vecAdd(h_A,h_B,h_C,n);

    std::cout << "Sample results from vector addition:" << std::endl;
    for (int i=0;i<10;i++){
        std::cout << h_A[i] << "+" << h_B[i] << "=" << h_C[i] << std::endl;
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;

}
