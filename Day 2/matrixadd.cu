#include <stdio.h>
#include <cuda_runtime.h>

__global__ void MatrixAdd(const float* A, const float* B, float* C, int rows,int cols)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i<rows && j<cols)
    {
        C[i*cols+j] = A[i * cols +j] + B[i * cols + j];
    }
}

int main() {
    int rows = 3;
    int cols = 3;
    int size = rows * cols * sizeof(float);

    float h_A[] = {1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f};

    float h_B[] = {9.0f, 8.0f, 7.0f,
            6.0f, 5.0f, 4.0f,
            3.0f, 2.0f, 1.0f}; 

    float h_C[9];

    float *d_A, *d_B, *d_C;
    
    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMalloc((void**)&d_C,size);

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,(rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    MatrixAdd <<< numBlocks , threadsPerBlock>>>(d_A,d_B,d_C,rows,cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);

    printf("Result Matrix C (A+B):\n");
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            printf("%.2f",h_C[i*cols + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}