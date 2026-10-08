"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // TODO: implement elementwise c[i] = a[i] + b[i]

    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) c[i]=a[i]+b[i];
    
    
    }

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    // TODO: multiply each element of a by scalar in place
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = a[i] * scalar;
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    // TODO: replace each a[i] with expf(a[i])
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) a[i]=expf(a[i]);
}

# Step 4 - row_max
__global__ void row_max(const float* matrix, float* out, int rows, int cols) {

    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows) {

        float m = -1e38f; 

        for (int c = 0; c < cols; c++) {

            float val = matrix[r * cols + c];
            
            if (val > m) {
                m = val;
            }
        }

        out[r] = m;
    }
}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
 
    int r = blockIdx.x;
    if (r >= rows) return;  

    extern __shared__ float sdata[];  

    int tid = threadIdx.x;
    int stride = blockDim.x;

    float partial_sum = 0.0f; 
    for (int c = tid; c < cols; c += stride) {
        partial_sum += matrix[r * cols + c];
    }
    sdata[tid] = partial_sum;  

    //CRITICAL: Wait for ALL threads to finish first step
    __syncthreads();


    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];  
        }
 
        __syncthreads();
    }

    if (tid == 0) {
        out[r] = sdata[0];  
    }
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {

    float sum = 0.0f;

 
    for (int i = 0; i < n; i++) {

        sum += a[i] * b[i];
    }

    return sum;
}

# Step 7 - matmul
__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;  
    int col = blockIdx.x * blockDim.x + threadIdx.x;  

    if (row < m && col < n) {         

        float sum = 0.0f;

        for (int l = 0; l < k; l++) {
            sum += a[row * k + l]    
                 * b[l * n + col];   
        }

        c[row * n + col] = sum;     
    }
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {

    int c = blockIdx.x * blockDim.x + threadIdx.x; 
    int r = blockIdx.y * blockDim.y + threadIdx.y;  

    if (c < cols && r < rows) {
 
        out[c * rows + r] = in[r * cols + c];
    }
}

# Step 9 - qk_scores
__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {

    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i < seq_len && j < seq_len) {
        

        const float* q_row_i = &q[i * head_dim];
        const float* k_row_j = &k[j * head_dim];
        

        float raw_score = dot_product(q_row_i, k_row_j, head_dim);
        

        scores[i * seq_len + j] = raw_score / sqrtf((float)head_dim);
    }
}

# Step 10 - softmax_rows
__global__ void softmax_rows(float* matrix, int rows, int cols) {
  
    int r = blockIdx.x;
    if (r >= rows) return; 

    extern __shared__ float sdata[]; 
    int tid = threadIdx.x;

    float* row_ptr = &matrix[r * cols];


    float local_max = -1e38f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float val = row_ptr[c];
        if (val > local_max) local_max = val;
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    float row_max = sdata[0]; 
    __syncthreads(); 

 
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        // x_i = exp(x_i - max)
        float val = expf(row_ptr[c] - row_max);
        row_ptr[c] = val;  
        local_sum += val;   
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float row_sum = sdata[0]; 
    __syncthreads();

    for (int c = tid; c < cols; c += blockDim.x) {
        row_ptr[c] /= row_sum;
    }
}

# Step 11 - pv_matmul
__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {

    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int d = blockIdx.x * blockDim.x + threadIdx.x;


    if (i < seq_len && d < head_dim) {
        float sum = 0.0f;

        for (int j = 0; j < seq_len; j++) {
 
            float p_val = p[i * seq_len + j];
    
            float v_val = v[j * head_dim + d];
            
            sum += p_val * v_val;
        }
  
        out[i * head_dim + d] = sum;
    }
}

# Step 12 - naive_attention
void naive_attention(const float* d_q, const float* d_k, const float* d_v, float* d_out, int seq_len, int head_dim) {
 
    float* d_scores = nullptr;
    size_t scores_size = seq_len * seq_len * sizeof(float);
    
    cudaMalloc((void**)&d_scores, scores_size);

    dim3 block(16, 16); 

    dim3 grid_scores(
        (seq_len + block.x - 1) / block.x, 
        (seq_len + block.y - 1) / block.y  
    );

    dim3 grid_output(
        (head_dim + block.x - 1) / block.x, 
        (seq_len + block.y - 1) / block.y   
    );


    qk_scores<<<grid_scores, block>>>(d_q, d_k, d_scores, seq_len, head_dim);

    int threads_per_row_block = 256;
    int shared_mem_size = threads_per_row_block * sizeof(float);
    softmax_rows<<<seq_len, threads_per_row_block, shared_mem_size>>>(d_scores, seq_len, seq_len);

    pv_matmul<<<grid_output, block>>>(d_scores, d_v, d_out, seq_len, head_dim);

    cudaFree(d_scores);
}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    // TODO: return the running max of old_max and new_val
    return fmaxf(old_max, new_val);
}

# Step 14 - correction_factor
__device__ float correction_factor(float old_max, float new_max) {
    // TODO: return the scalar used to rescale running statistics
    return expf(old_max - new_max);
}

# Step 15 - update_running_sum
__device__ float update_running_sum(float old_sum, float correction, float block_sum) {

    float rescaled_sum = old_sum * correction;
 
    return rescaled_sum + block_sum;
}

# Step 16 - rescale_output
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
 
    for (int d = 0; d < head_dim; d++) {
        out_row[d] *= correction;
    }
}

# Step 17 - load_tile
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    
    int total_elements = tile_rows * tile_cols;

    for (int i = thread_id; i < total_elements; i += num_threads) {
  
        int r = i / tile_cols;  
        int c = i % tile_cols; 

        int global_r = src_row_start + r;
        int global_c = src_col_start + c;

        if (global_r < src_rows && global_c < src_cols) {
 
            shared_dst[i] = src[global_r * src_cols + global_c];
        } else {
  
            shared_dst[i] = 0.0f;
        }
    }
}

# Step 18 - tile_scores
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {

    int total_scores = tile_q * tile_k;

   
    for (int idx = thread_id; idx < total_scores; idx += num_threads) {

        int i = idx / tile_k; 
        int j = idx % tile_k; 

   
        const float* q_row = &q_tile[i * head_dim];
        const float* k_row = &k_tile[j * head_dim];

  
        float raw_score = dot_product(q_row, k_row, head_dim);

        s_tile[idx] = raw_score * scale;
    }
}

# Step 19 - tile_rowmax
__device__ void tile_rowmax(const float* s_tile, float* row_max_out, int tile_q, int tile_k, int thread_id, int num_threads) {

    for (int r = thread_id; r < tile_q; r += num_threads) {
        
        float m = -1e38f; 
        
        for (int c = 0; c < tile_k; c++) {
            float val = s_tile[r * tile_k + c];
            if (val > m) {
                m = val;
            }
        }

        row_max_out[r] = m;
    }


    __syncthreads();
}

# Step 20 - tile_exp
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    
    int total_elements = tile_q * tile_k;

    for (int i = thread_id; i < total_elements; i += num_threads) {
        
   
        int r = i / tile_k;

        float m = row_max[r];

        s_tile[i] = expf(s_tile[i] - m);
    }
}

# Step 21 - tile_rowsum
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float sum = 0.0f;
        
   
        int row_offset = r * tile_k;

    
        for (int c = 0; c < tile_k; c++) {
            sum += p_tile[row_offset + c];
        }
        

        row_sum_out[r] = sum;
    }
}

# Step 22 - accumulate_pv
__device__ void accumulate_pv(const float* p_tile, const float* v_tile, 
                              float* out_acc, int tile_q, int tile_k, int head_dim, 
                              int thread_id, int num_threads) {
    

    int total_elements = tile_q * head_dim;

    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        int r = idx / head_dim;
        int d = idx % head_dim;

        const float* p_row = &p_tile[r * tile_k];
        
        float local_sum = 0.0f;

        for (int j = 0; j < tile_k; j++) {
            local_sum += p_row[j] * v_tile[j * head_dim + d];
        }

        out_acc[idx] += local_sum;
    }
}

# Step 23 - flash_attention_kernel
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
  
    extern __shared__ char smem[];
    
    int q_tile_size = tile_q * head_dim * sizeof(float);
    int k_tile_size = tile_k * head_dim * sizeof(float);
    int v_tile_size = tile_k * head_dim * sizeof(float);
    int s_tile_size = tile_q * tile_k * sizeof(float);
    int stats_size  = tile_q * sizeof(float); 

    float* q_shared = (float*)smem;
    float* k_shared = (float*)(smem + q_tile_size);
    float* v_shared = (float*)(smem + q_tile_size + k_tile_size);
    float* s_tile   = (float*)(smem + q_tile_size + k_tile_size + v_tile_size);
    float* o_shared = (float*)(smem + q_tile_size + k_tile_size + v_tile_size + s_tile_size);
    float* m_shared = (float*)(smem + q_tile_size + k_tile_size + v_tile_size + s_tile_size + tile_q * head_dim * sizeof(float));
    float* l_shared = m_shared + tile_q;
    float* row_max  = l_shared + tile_q;
    float* row_sum  = row_max + tile_q;

    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;
    
    int q_block_start = blockIdx.x * tile_q;

  
    load_tile(q, q_shared, q_block_start, 0, seq_len, head_dim, tile_q, head_dim, thread_id, num_threads);

    for (int r = thread_id; r < tile_q; r += num_threads) {
        m_shared[r] = -1e38f;
        l_shared[r] = 0.0f;
    }
    for (int i = thread_id; i < tile_q * head_dim; i += num_threads) {
        o_shared[i] = 0.0f;
    }
    __syncthreads();
    for (int kv_start = 0; kv_start < seq_len; kv_start += tile_k) {

        load_tile(k, k_shared, kv_start, 0, seq_len, head_dim, tile_k, head_dim, thread_id, num_threads);
        load_tile(v, v_shared, kv_start, 0, seq_len, head_dim, tile_k, head_dim, thread_id, num_threads);
        __syncthreads();

        tile_scores(q_shared, k_shared, s_tile, tile_q, tile_k, head_dim, scale, thread_id, num_threads);
        __syncthreads();

        tile_rowmax(s_tile, row_max, tile_q, tile_k, thread_id, num_threads);
        __syncthreads();

      
        for (int r = thread_id; r < tile_q; r += num_threads) {
            float tile_max = row_max[r];
            float old_max = m_shared[r];
            
            float new_max = online_max(old_max, tile_max);
            float corr = correction_factor(old_max, new_max);

            m_shared[r] = new_max;
            rescale_output(&o_shared[r * head_dim], head_dim, corr);
   
            row_max[r] = corr; 
        }
        __syncthreads();

        tile_exp(s_tile, m_shared, tile_q, tile_k, thread_id, num_threads);
        tile_rowsum(s_tile, row_sum, tile_q, tile_k, thread_id, num_threads);
        __syncthreads();

        for (int r = thread_id; r < tile_q; r += num_threads) {
            l_shared[r] = update_running_sum(l_shared[r], row_max[r], row_sum[r]);
        }

        accumulate_pv(s_tile, v_shared, o_shared, tile_q, tile_k, head_dim, thread_id, num_threads);
        __syncthreads(); 
    }

    for (int i = thread_id; i < tile_q * head_dim; i += num_threads) {
        int r = i / head_dim;
        int d = i % head_dim;
        int global_r = q_block_start + r;
        
        if (global_r < seq_len && l_shared[r] > 1e-10f) {
            out[global_r * head_dim + d] = o_shared[i] / l_shared[r];
        } else if (global_r < seq_len) {
            out[global_r * head_dim + d] = 0.0f;
        }
    }
}

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

