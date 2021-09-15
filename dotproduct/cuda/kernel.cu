#include <math.h>
#include <common.h>


const int THREADS_PER_BLOCK = 256;


__global__ void cudaDotProduct(float *x,  float *y, float *out, int n)
{
    // Shared among all threads within a single block.
    __shared__ float cache[THREADS_PER_BLOCK];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    out[tid] = x[tid] + y[tid];
    if (tid < n) {
        cache[cacheIndex] = x[tid] * y[tid];
    }

    // Perform reduction of values stored in the cache.
    // Complexity: O(log2(n))
    int offset = blockDim.x / 2;
    while (offset != 0) {
        if (cacheIndex < offset) {
            cache[cacheIndex] += cache[cacheIndex + offset];
        }
        // Make sure that every thread has finished writing into the cache.
        __syncthreads();
        offset /= 2;
    }

    // Store the reduction result in the global output array.
    if (cacheIndex == 0) {
        out[blockIdx.x] = cache[0];
    }
}


float *DEV_A = NULL, *DEV_B = NULL, *DEV_OUT = NULL;
float *HOST_OUT = NULL;
int current_n_bytes = -1;


float dotproduct(float *a, float *b, int n) {
    int n_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int n_bytes = n * sizeof(float);
    int n_bytes_out = n_blocks * sizeof(float);
    // Allocate memory on GPU if needed
    if (current_n_bytes != n_bytes) {
        // Free already the allocated memory if it actually was
        if (DEV_A != NULL || DEV_B != NULL || DEV_OUT != NULL) {
            CUDA_CHECK(cudaFree(DEV_A));
            CUDA_CHECK(cudaFree(DEV_B));
            CUDA_CHECK(cudaFree(DEV_OUT));
            delete[] HOST_OUT;
        }
        printf("Allocating memory on GPU. Old current_n_bytes=%d", current_n_bytes);
        current_n_bytes = n_bytes;
        printf("New current_n_bytes=%d", current_n_bytes);
        CUDA_CHECK(cudaMalloc(&DEV_A, n_bytes));
        CUDA_CHECK(cudaMalloc(&DEV_B, n_bytes));
        CUDA_CHECK(cudaMalloc(&DEV_OUT, n_bytes_out));
        // Allocate memory for the output
        HOST_OUT = new float[n_blocks];
    }

    // Run the dotproduct
    CUDA_CHECK(cudaMemcpy(DEV_A, a, n_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(DEV_B, b, n_bytes, cudaMemcpyHostToDevice));
    cudaDotProduct<<<n_blocks, THREADS_PER_BLOCK>>>(DEV_A, DEV_B, DEV_OUT, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(HOST_OUT, DEV_OUT, n_bytes_out, cudaMemcpyDeviceToHost));

    // Perform final reduction
    float result = 0;
    for (int i = 0; i < n; i++) {
        result += HOST_OUT[i];
    }
    return result;
}
