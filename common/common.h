#ifndef CYTHON_CUDA_SAMPLES_COMMON_H
#define CYTHON_CUDA_SAMPLES_COMMON_H
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>


#define rnd(x) (x * rand() / RAND_MAX)


inline void CUDA_CHECK(cudaError_t error) {
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
        fprintf(stderr, "code: %d, reason: %s\n", error,
                cudaGetErrorString(error));
    }
}

#endif
