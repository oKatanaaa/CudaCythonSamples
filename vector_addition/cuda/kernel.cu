#include <iostream>
#include <math.h>
#include <common.h>

int THREADS_PER_BLOCK = 256;


__global__ void cudaAdd(float *x,  float *y, float *out) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = x[tid] + y[tid];
}


float *dev_a = NULL, *dev_b = NULL, *dev_out = NULL;
int current_n_bytes = -1;


void vec_add(float *a, float *b, float *out, int n) {
    //printf("Received vector with length n=%d", n);
    // Allocate memory on GPU
    int n_bytes = n * sizeof(float);
    if (current_n_bytes != n_bytes) {
        // Free already allocated memory if it actually was
        if (dev_a != NULL) {
            CUDA_CHECK(cudaFree(dev_a));
            CUDA_CHECK(cudaFree(dev_b));
            CUDA_CHECK(cudaFree(dev_out));
        }
        printf("Allocating memory on GPU. Old current_n_bytes=%d", current_n_bytes);
        current_n_bytes = n_bytes;
        printf("New current_n_bytes=%d", current_n_bytes);
        CUDA_CHECK(cudaMalloc(&dev_a, n_bytes));
        CUDA_CHECK(cudaMalloc(&dev_b, n_bytes));
        CUDA_CHECK(cudaMalloc(&dev_out, n_bytes));
    }
    
    CUDA_CHECK(cudaMemcpy(dev_a, a, n_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, b, n_bytes, cudaMemcpyHostToDevice));
    int n_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaAdd<<<n_blocks, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, dev_out, n_bytes, cudaMemcpyDeviceToHost));
}
