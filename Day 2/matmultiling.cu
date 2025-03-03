#define Tile_Width 16
#include <stdio.h>
#include <iostream>
#include <cmath>

__global__ void matmulkernel(float* M, float* N, float* P, int Width)

{
    __shared__ float ds_M[Tile_Width][Tile_Width];
    __shared__ float ds_N[Tile_Width][Tile_Width];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y*Tile_Width + ty;   // Row of P element
    int col = blockIdx.x*Tile_Width + tx;   // Column of P element

    float Pvalue = 0;

    for (int ph = 0; ph < Width/Tile_Width; ++ph) {
        ds_M[ty][tx] = M[row*Width + ph*Tile_Width + tx];
        ds_N[ty][tx] = N[(ph*Tile_Width + ty)*Width + col];
        __syncthreads();

        for (int i = 0; i < Tile_Width; ++i) {
            Pvalue += ds_M[ty][i] * ds_N[i][tx];
        }
        __syncthreads();
    }
    P[row*Width + col] = Pvalue;
}
int main()
{
    int Width = 1024;  // For example, a 1024 x 1024 matrix
    int size = Width * Width * sizeof(float);

    // Allocate host memory for matrices
    float* h_M = (float*)malloc(size);
    float* h_N = (float*)malloc(size);
    float* h_P = (float*)malloc(size);

    // Initialize matrices M and N (for example, fill with some values)
    for (int i = 0; i < Width * Width; i++) {
        h_M[i] = 1.0f; // or some other value
        h_N[i] = 1.0f; // for simplicity
    }

    // Allocate device memory
    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    // Copy matrices from host to device
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    // Setup execution configuration.
    dim3 dimBlock(Tile_Width, Tile_Width);
    dim3 dimGrid((Width + Tile_Width - 1) / Tile_Width, (Width + Tile_Width - 1) / Tile_Width);

    // Launch the kernel
    matmulkernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // (Optional) Verify or print part of the output here

    // Clean up memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}