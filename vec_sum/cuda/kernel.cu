#include <iostream>
#include <math.h>
#include <common.h>

const unsigned int THREADS_PER_BLOCK = 256;


template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >=  8) sdata[tid] += sdata[tid + 4];
    if (blockSize >=  4) sdata[tid] += sdata[tid + 2];
    if (blockSize >=  2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void cudaSum(float *g_idata, float *g_odata)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    // First level of reduction
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];

    __syncthreads();
    // It is a fucking madness to optimize code like this
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads();
    }
    // Unroll last 6 iterations
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


float *dev_idata = NULL, *dev_odata = NULL, *host_odata = NULL;
int current_n_bytes = -1;

float vec_sum(float *a, int n) {
    // --- MEMORY ALLOCATION
    int n_bytes_in = n * sizeof(float);
    int n_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK / 2;
    int n_bytes_out = n_blocks * sizeof(float);
    if (current_n_bytes != n_bytes_in) {
        if (dev_idata != NULL) {
            CUDA_CHECK(cudaFree(dev_idata));
            delete[] host_odata;
        }
        printf("Allocating memory on GPU. Old current_n_bytes=%d", current_n_bytes);
        current_n_bytes = n_bytes_in;
        printf("New current_n_bytes=%d", current_n_bytes);
        CUDA_CHECK(cudaMalloc(&dev_idata, n_bytes_in));
        CUDA_CHECK(cudaMalloc(&dev_odata, n_bytes_out));
        host_odata = new float[n_bytes_out];
    }
    
    CUDA_CHECK(cudaMemcpy(dev_idata, a, n_bytes_in, cudaMemcpyHostToDevice));

    int n_bytes_smem = THREADS_PER_BLOCK * sizeof(float);
    cudaSum<THREADS_PER_BLOCK><<< n_blocks, THREADS_PER_BLOCK, n_bytes_smem >>>(dev_idata, dev_odata);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_odata, dev_odata, n_bytes_out, cudaMemcpyDeviceToHost));

    // Final reduction
    float out = 0.0;
    for(int i = 0; i < n_blocks; i++) {
        out += host_odata[i];
    }
    return out;
}
