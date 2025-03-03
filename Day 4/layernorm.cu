#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void layerNormKernel(const float *d_in, float *d_out, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        extern __shared__ float sdata[];
        float *row_data = sdata;

        for (int col = threadIdx.y;col < cols;col += blockDim.y)
        {
            row_data[col] = d_in[row * cols + col];
        }
        __syncthreads();

        float mean = 0.0f;
        for (int col=0;col<cols;col++)
        {
            mean += row_data[col];
        }
        mean /= cols;

        float var = 0.0f;
        for (int col= 0;col<cols;col++)
        {
            float diff = row_data[col] - mean;
            var += diff * diff;
        }
        var /= cols;

        float stddev = sqrtf(var + 1e-7);

        for (int col = threadIdx.y;col<cols;col+= blockDim.y)
        {
            d_out[row * cols + col] = (row_data[col] - mean) / stddev;
        }
    }
}

int main()
{
    int rows = 10, cols = 10;
    float *h_in,*h_out;
    float *d_in,*d_out;

    h_in = (float*)malloc(rows * cols * sizeof(float));
    h_out = (float*)malloc(rows * cols * sizeof(float));

    for (int i =0;i<rows;i++)
    {
        for (int j=0;j<cols;j++)
        {
            h_in[i*cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    cudaMalloc((void**)&d_in,rows * cols * sizeof(float));
    cudaMalloc((void**)&d_out,rows * cols * sizeof(float));

    cudaMemcpy(d_in,h_in,rows * cols * sizeof(float),cudaMemcpyHostToDevice);

    int blocksize = 256;
    int gridsize = (rows + blocksize - 1) / blocksize;

    size_t shared_memeory_size = cols * sizeof(float);

    layerNormKernel<<<gridsize,blocksize,shared_memeory_size>>>(d_in,d_out,rows,cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out,d_out,rows * cols * sizeof(float),cudaMemcpyDeviceToHost);

    for (int i=0;i<rows;i++)
    {
        for (int j=0;j<cols;j++)
        {
            printf("%f ",h_out[i*cols + j]);
        }
        printf("\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in); 
    free(h_out);

    return 0;
}