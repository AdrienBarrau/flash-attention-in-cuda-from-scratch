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
    // Simply iterate through the entire output accumulator for this query 
    // and multiply by the scalar correction factor.
    for (int d = 0; d < head_dim; d++) {
        out_row[d] *= correction;
    }
}

# Step 17 - load_tile (not yet solved)
# TODO: implement

# Step 18 - tile_scores (not yet solved)
# TODO: implement

# Step 19 - tile_rowmax (not yet solved)
# TODO: implement

# Step 20 - tile_exp (not yet solved)
# TODO: implement

# Step 21 - tile_rowsum (not yet solved)
# TODO: implement

# Step 22 - accumulate_pv (not yet solved)
# TODO: implement

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

