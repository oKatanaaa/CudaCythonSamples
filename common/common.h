#ifndef CYTHON_CUDA_SAMPLES_COMMON_H
#define CYTHON_CUDA_SAMPLES_COMMON_H

#include <cuda.h>
#include <iostream>


#define rnd(x) (x * rand() / RAND_MAX)


void CUDA_CHECK(cudaError_t call_resp) {
    const cudaError_t error = call_resp;
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
        fprintf(stderr, "code: %d, reason: %s\n", error,
                cudaGetErrorString(error));
    }
}

#endif
