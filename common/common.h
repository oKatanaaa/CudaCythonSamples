#include <iostream>

void CUDA_CHECK(cudaError_t call_resp) {
    const cudaError_t error = call_resp;
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
        fprintf(stderr, "code: %d, reason: %s\n", error,
                cudaGetErrorString(error));
    }
}
