#ifndef CYTHON_CUDA_SAMPLES_COMMON_H
#define CYTHON_CUDA_SAMPLES_COMMON_H
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>


#define rnd(x) (x * rand() / RAND_MAX)


static void CudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: %s:%d, ", file, line);
        fprintf(stderr, "code: %d, reason: %s\n", error,
                cudaGetErrorString(error));
        exit( EXIT_FAILURE );
    }
}


#define CUDA_CHECK( err ) (CudaCheck( err, __FILE__, __LINE__ ))


#endif
