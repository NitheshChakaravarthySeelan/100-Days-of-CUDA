#include <cuda_runtime.h>
#include <iostream>

#define SEQ_LEN 64
#define HEAD_DIM 64
#define BATCH_SIZE 1
#define NUM_HEADS 1

#define BLOCK_COL 32
#define BLOCK_ROW 32 

#define TOTAL_COL ((SEQ_LEN + BLOCK_COL -1) / BLOCK_COL)
#define TOTAL_ROW ((SEQ_LEN + BLOCK_ROW -1) / BLOCK_ROW)

__global__ void flashAttentionKernel(const float *Q, const float *K, const float *V, float *O, float *m, float *l, const int seq_len, const int head_dim){
    int thread_idx = threadIdx.x;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int qvk_offset = (batch_idx * gridDim.y * seq_len * head_dim) + (head_idx * seq_len * head_dim);
    int lm_offset = (batch_idx * gridDim.y * seq_len) + (head_idx * seq_len);

    extern __shared__ float shared_memory[];
    float *Q_tile = shared_memory;
    float *K_tile = &shared_memory[BLOCK_ROW * HEAD_DIM];
    float *V_tile = &shared_memory[2 * BLOCK_ROW * HEAD_DIM];
    float *S_tile = &shared_memory[3 * BLOCK_ROW * HEAD_DIM];

    for (int j=0;j<TOTAL_COL;++j){
        for (int i=0;i<BLOCK_COL;++i){
            int global_idx = j*BLOCK_COL + i;
            if (global_idx < seq_len)
            {
                for (int d= 0;d<head_dim;++d){
                    K_tile[i*head_dim+d] = K[qvk_offset + global_idx * head_dim +d];
                    V_tile[i*head_dim+d] = V[qvk_offset + global_idx * head_dim + d];
                }
            }
        }
        __syncthreads();

        for (int i =0;i<TOTAL_ROW;++i){
            for (int r =0;r<BLOCK_ROW;++r){
                int global_idx = i*BLOCK_ROW + r;
                if (global_idx < seq_len){
                    for (int d=0;d<head_dim;++d){
                        Q_tile[r*head_dim+d] = Q[qvk_offset + global_idx * head_dim + d];
                    }
                }
            }
            __syncthreads();

            float row_max = -1e20;
            for (int r=0;r<BLOCK_ROW;++r){
                for (int c=0;c<BLOCK_COL;++c){
                    float score = 0.0f;
                    for (int d=0;d<head_dim;++d){
                        score += Q_tile[r*head_dim+d] * K_tile[c*head_dim+d];
                    }
                    score *= rsqrtf((float)head_dim);
                    S_tile[r*BLOCK_COL + c] = expf(score);
                    row_max = fmaxf(row_max,score);
                }
            }

            for (int r=0;r<BLOCK_ROW;++r){
                float row_sum = 0;
                for (int c=0;c<BLOCK_COL;++c){
                    row_sum += S_tile[r*BLOCK_COL +c];
                }
                for (int d=0;d<head_dim;++d){
                    float weighted_sum = 0.0f;
                    for (int c=0;c<BLOCK_COL;++c){
                        weighted_sum += S_tile[r*BLOCK_COL+c] * V_tile[c*head_dim+d];
                    }
                O[qvk_offset+r*head_dim+d] = weighted_sum /row_sum;
                }
            }
        }
        __syncthreads();
    }
}

void initializeRandom(float *data,int size)
{
    for (int i=0;i<size;++i){
        data[i] = (float)rand() / RAND_MAX;
    }
}

int main()
{
    int matrixSize = BATCH_SIZE * NUM_HEADS * SEQ_LEN * HEAD_DIM;
    int vectorSize = BATCH_SIZE * NUM_HEADS * SEQ_LEN;

    float *Q = new float[matrixSize];
    float *K = new float[matrixSize];
    float *V = new float[matrixSize];
    float *O = new float[matrixSize];

    initializeRandom(Q,matrixSize);
    initializeRandom(K,matrixSize);
    initializeRandom(V,matrixSize);

    float *dQ,*dK,*dV,*dO;
    cudaMalloc(&dQ,matrixSize * sizeof(float));
    cudaMalloc(&dK,matrixSize * sizeof(float));
    cudaMalloc(&dV,matrixSize * sizeof(float));
    cudaMalloc(&dO,matrixSize * sizeof(float));

    cudaMemcpy(dQ,Q,matrixSize * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dK,K,matrixSize * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dV,V,matrixSize * sizeof(float),cudaMemcpyHostToDevice);

    int sharedMemorySize = (3 * BLOCK_ROW * HEAD_DIM * sizeof(float)) + (BLOCK_ROW * BLOCK_COL * sizeof(float));

    dim3 gridSize(BATCH_SIZE,NUM_HEADS);
    dim3 blockSize(BLOCK_COL);

    flashAttentionKernel<<<gridSize,blockSize,sharedMemorySize>>>(dQ,dK,dV,dO,nullptr,nullptr,SEQ_LEN,HEAD_DIM);
    cudaDeviceSynchronize();

    std::cout <<"First output value :" << O << std::endl;

    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] O;
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);

    return 0;
}