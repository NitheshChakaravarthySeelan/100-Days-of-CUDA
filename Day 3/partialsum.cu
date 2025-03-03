#include <stdio.h>
#include <cuda_runtime.h>

__global__ void partialSumKernel(const float *d_in, float *d_out, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float mySum = 0.0f;
    if (index < n)
    {
        mySum = d_in[index];
    }

    if (index + blockDim.x < n)
    {
        mySum += d_in[index + blockDim.x];
    }
    sdata[tid] = mySum;
    __syncthreads();

    for (int s = blockDim.x /2 ; s>0; s>>= 1)
    {
        if (tid<s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

int main()
{
    int n = 1024;
    int size = n * sizeof(float);
    float *h_in = (float*)malloc(size);
    for (int i=0;i<n;i++)
    {
        h_in[i] = 1.0f;
    }

    float *d_in , *d_out;
    cudaMalloc((void**)&d_in,size);
    cudaMemcpy(d_in,h_in,size,cudaMemcpyHostToDevice);

    int blockSize = 256; // Number of threads per block
    int numBlocks = (n + blockSize * 2 - 1) / (blockSize * 2);
    cudaMalloc((void**)&d_out,numBlocks * sizeof(float));

    // blockSize * sizeof(float) - is the size of the shared memory
    partialSumKernel<<<numBlocks,blockSize,blockSize * sizeof(float)>>>(d_in,d_out,n);
    cudaDeviceSynchronize();

    float *h_out = (float*)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_out,d_out,numBlocks * sizeof(float),cudaMemcpyDeviceToHost);

    printf("Partial Sums for each block:\n");
    for (int i=0;i<numBlocks;i++)
    {
        printf("Block %d: %.2f\n", i, h_out[i]);
    }

    float totSum = 0.0f;
    for (int i = 0; i < numBlocks; i++)
    {
        totSum += h_out[i];
    }
    printf("Total Sum: %.2f\n", totSum);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
