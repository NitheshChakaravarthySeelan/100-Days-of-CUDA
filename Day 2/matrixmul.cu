#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C,int N)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < N){
        float sum = 0.0f;
        for (int k=0;k<N;k++){
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(){
    int N = 3;
    int size = N * N * sizeof(float);

    float h_A[] = { 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f };

    float h_B[] = { 9.0f, 8.0f, 7.0f,
            6.0f, 5.0f, 4.0f,
            3.0f, 2.0f, 1.0f };

    float h_C[9];

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMalloc((void**)&d_C,size);

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,(N+threadsPerBlock.y -1) / threadsPerBlock.y );

    matrixMultiplyKernel<<<numBlocks,threadsPerBlock>>>(d_A,d_B,d_C,N);
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);

    printf("result matrix is ");
    for (int i=0;i<N;i++)
    {
        for (int j=0;j<N;j++){
            printf("%.2f\t",h_C[i*N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}