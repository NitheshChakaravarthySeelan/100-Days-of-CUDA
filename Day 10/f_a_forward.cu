#include <iostream>
#include <cuda_runtime.h>
#include <math_constants.h>

// Sequence and embedding settings
#define SEQ_LEN 2       // Sequence Length (N)
#define EMBED_DIM 2     // Embedding Dimension (d)
#define BATCH_SIZE 1    // Number of batches
#define NUM_HEADS 1     // Number of attention heads

// Tiling factors
#define BLOCK_COL 32
#define BLOCK_ROW 32

// Compute number of tiles along rows & columns
#define TOTAL_COLS ((SEQ_LEN + BLOCK_COL - 1) / BLOCK_COL)
#define TOTAL_ROWS ((SEQ_LEN + BLOCK_ROW - 1) / BLOCK_ROW)

// CUDA Kernel for FlashAttention Forward Pass
__global__ void flashAttentionKernel(const float *Q, const float *K, const float *V,
                                     float *O, float *m, float *l,
                                     const int seq_len, const int embed_dim)
(
    int thread_idx = threadIdx.x;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int qvk_offset = (batch_idx * gridDim.y * seq_len * embed_dim) + (head_idx * seq_len * embed_dim);

    extern __shared__ float shared_memory[];
    float *Q_tile = shared_memory;
    float *K_tile = &shared_memory[BLOCK_ROW * EMBED_DIM];
    float *V_tile = &shared_memory[2 * BLOCK_ROW * EMBED_DIM];
    float *S_tile = &shared_memory[3 * BLOCK_ROW * EMBED_DIM];

    for (int r=0;r<BLOCK_ROW;++r){
        int globalRow = batch_idx * BLOCK_ROW + r;
        if (globalRow < seq_len) {
            for (int c=0;c<EMBED_DIM;++c){
                Q_tile[r*EMBED_DIM+c] = Q[qvk_offset + globalRow * EMBED_DIM + c];
                K_tile[r*EMBED_DIM+c] = K[qvk_offset + globalRow * EMBED_DIM + c];
                V_tile[r*EMBED_DIM+c] = V[qvk_offset + globalRow * EMBED_DIM + c];
            }
        }
    }
    __syncthreads();

    for (int r=0;r<BLOCK_ROW;++r){
        for (int c=0;c<BLOCK_ROW;++c){
            int globalRow = batch_idx * BLOCK_ROW + r;
            int globalCol = head_idx * BLOCK_COL + c;
            if (globalRow < seq_len && globalCol < seq_len){
                float score = 0.0f;
                for (int k=0;k<EMBED_DIM;++k){
                    score += Q_tile[r*EMBED_DIM+k] * K_tile[c*EMBED_DIM+k];
                }
                score *= rsqrtf((float)EMBED_DIM)
                S_tile[r*EMBED_DIM+c] = expf(score);
            }
        }
    }
    __syncthreads();

    for (int r=0;r<BLOCK_ROW;++r){
        for (int c=0;c<BLOCK_COL,++c){
            float sum_score = 0.0f;
            for (int k=0;k<EMBED_DIM;++k){
                sum_score += S_tile[r*EMBED_DIM+k] * V_tile[k*BLOCK_COL+c];
            }
            O[qvk_offset+r*EMBED_DIM+c] = sum_score;
        }
    }
)