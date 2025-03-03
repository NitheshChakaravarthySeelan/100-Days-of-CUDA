#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define Filter_radius 1
#define Filter_width (2 * Filter_radius +1)
#define Tile_width 16

__constant__ float F_c[Filter_width][Filter_width];
__global__ void conv2d_tilled(const float* N, float* P, int width, int height){
    int outrow = blockIdx.y * Tile_width + threadIdx.y;
    int outcol = blockIdx.x * Tile_width + threadIdx.x;

    extern __shared__ float N_ds[];

    int sharedwidth = Tile_width + 2 * Filter_radius;

    int sharedrow = blockIdx.y * Tile_width - Filter_radius;
    int sharedcol = blockIdx.x * Tile_width - Filter_radius;

    int tdx = threadIdx.x;
    int tdy = threadIdx.y;

    for (int i = tdy;i<sharedwidth;i+=Tile_width){
        for ( int j= tdx;j<sharedwidth;i+=Tile_width){
            int globalRow = sharedrow + i;
            int globalCol = sharedcol + j;

            if (globalRow >=0 && globalRow < height && globalCol >= 0 && globalCol < width) {
                N_ds[i*sharedwidth + j] = N[globalRow * width + globalCol];
            }
            else {
                N_ds[i * sharedwidth + j] = 0.0f;
            }
        }
        __syncthreads();

        if (outrow < height && outcol < width){
            float output = 0.0f;
            int sharedrowidx = threadIdx.y + Filter_radius;
            int sharedcolidx = threadIdx.x + Filter_radius;

            for (int i=0;i<Filter_width;i++){
                for (int j=0; j<Filter_width;j++){
                    output += F_c[i][j] * N_ds[(sharedrowidx - Filter_radius + i) * sharedwidth + (sharedcolidx - Filter_radius + j)];
                }
            }
            P[outrow * width + outcol] = output;
        }
    }
}

int main() {
    int width = 512;
    int height = 512;
    int size = width * height * sizeof(float);

    float* h_N = (float*)malloc(size);
    float* h_P = (float*)malloc(size);

    for (int i=0; i<width * height; i++) {
        h_N[i] = 1.0f;
    }

    float h_F[Filter_width * Filter_width] = {
        1,2,1,
        2,4,2,
        1,2,1
    };

    cudaMemcpyToSymbol(F_c,h_F,sizeof(float)*Filter_width*Filter_width);

    float *d_N, *d_P;
    cudaMalloc(&d_N,size);
    cudaMalloc(&d_P,size);

    cudaMemcpy(d_N,h_N,size,cudaMemcpyHostToDevice);
    dim3 dimBlock(Tile_width,Tile_width);
    dim3 dimGrid((width + Tile_width - 1) / Tile_width, (height + Tile_width - 1) / Tile_width);

    int sharedMemSize = (Tile_width + 2 * Filter_radius) * (Tile_width + 2 * Filter_radius) * sizeof(float);

    conv2d_tilled<<<dimGrid,dimBlock,sharedMemSize>>>(d_N,d_P,width,height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost);

    std::cout << "output (first 5 elements):";
    for (int i=0;i<5;i++){
        std::cout << h_P[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_N);
    cudaFree(d_P);
    free(h_N);
    free(h_P);

    return 0;
}