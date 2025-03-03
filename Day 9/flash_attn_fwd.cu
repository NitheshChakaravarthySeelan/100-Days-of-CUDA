#include <iostream>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>

#define M 32        // SRAM shared memory size
#define N 32        // Sequence length
#define d_model 32  // Embedding dimension

// Compute block sizes
#define block_col ((int)ceil(M / (4.0 * d_model))) // Number of key-value vectors in each tile
#define block_row (min((int)ceil(M / (4.0 * d_model)), d_model)) // Number of query vectors in each tile
#define Total_row ((int)ceil(N / (float)block_row)) // Total tiles along query dimension
#define Total_col ((int)ceil(N / (float)block_col)) // Total tiles along key-value dimension

__global__ void flashAttentionKernel(
    const float *Q, const float *K, const float *V,
    float *O, float *m, float *l)
{
    int thread_idx = threadIdx.x;

    // Define tile sizes
    const int KV_block = block_col * d_model;
    const int QO_block = block_col * d_model;

    // Allocate shared memory for faster memory access
    __shared__ float Q_tile[QO_block]; 
    __shared__ float K_tile[KV_block];
    __shared__ float V_tile[KV_block];
    __shared__ float O_tile[QO_block];
    __shared__ float l_tile[block_row];
    __shared__ float m_tile[block_row];

    float S[block_row * block_col];  // Score matrix
    float P[block_row * block_col];  // Probability matrix

    // Loop over Key/Value tiles
    for (int j = 0; j < Total_col; ++j)
    {
        // Load Key & Value tile from HBM (Global Memory) to Shared Memory
        for (int p = 0; p < block_col; ++p) {
            for (int k = 0; k < d_model; ++k) {
                K_tile[p * d_model + k] = K[j * KV_block + p * d_model + k];
                V_tile[p * d_model + k] = V[j * KV_block + p * d_model + k];
            }
        }
        __syncthreads();

        // Loop over Query tiles
        for (int i = 0; i < Total_row; ++i)
        {
            // Load Query tile from HBM (Global Memory) to Shared Memory
            for (int p = 0; p < block_row; ++p) {
                for (int k = 0; k < d_model; ++k) {
                    Q_tile[p * d_model + k] = Q[i * QO_block + p * d_model + k];
                }
                l_tile[p] = l[i * block_row + p];
                m_tile[p] = m[i * block_row + p];
            }
            __syncthreads();

            // Compute Attention Scores **Sij = Qi * Kj'**
            float row_max = -1e20;
            for (int p = 0; p < block_col; ++p) {
                float score = 0;
                for (int k = 0; k < d_model; ++k) {
                    score += Q_tile[thread_idx * d_model + k] * K_tile[p * d_model + k];
                }
                S[thread_idx * block_col + p] = score;  // Store score
                if (score > row_max) row_max = score;
            }
            m_tile[thread_idx] = row_max; // Store max score for row
            __syncthreads();

            // Compute Softmax probabilities **Pij = exp(Sij - max)**
            float row_sum = 0;
            for (int p = 0; p < block_col; ++p) {
                P[thread_idx * block_col + p] = expf(S[thread_idx * block_col + p] - row_max);
                row_sum += P[thread_idx * block_col + p];
            }
            l_tile[thread_idx] += row_sum;
            __syncthreads();

            // Normalize and compute Output **Oij = Pij * Vj**
            for (int k = 0; k < d_model; ++k) {
                float weighted_sum = 0;
                for (int p = 0; p < block_col; ++p) {
                    weighted_sum += P[thread_idx * block_col + p] * V_tile[p * d_model + k];
                }
                O_tile[thread_idx * d_model + k] = weighted_sum / row_sum;
            }
        }
    }
}
