#include <iostream>
#include <cublas_v2.h>
#include <math.h>
#include <common.h>

float *DEV_A = NULL, *DEV_B = NULL, *DEV_C = NULL;
int A_DIM1 = -1, A_DIM2 = -1, B_DIM2 = -1;
cublasHandle_t HANDLE = NULL;


float* allocate_floatmat_device(int n, int m) {
    size_t mat_size = n * m * sizeof(float);
    float *dev_mat;
    CUDA_CHECK(cudaMalloc(&dev_mat, mat_size));
    return dev_mat;
}

void gpu_matmul(float *a, float *b, float *out, int a_dim1, int a_dim2, int b_dim2) {
    // Allocate buffers on device
    if ((A_DIM1 != a_dim1) || (A_DIM2 != a_dim2) || (B_DIM2 != b_dim2)) {
        if (DEV_A != NULL || DEV_B != NULL || DEV_C != NULL) {
            CUDA_CHECK(cudaFree(DEV_A));
            CUDA_CHECK(cudaFree(DEV_B));
            CUDA_CHECK(cudaFree(DEV_C));
        }
        DEV_A = allocate_floatmat_device(a_dim1, a_dim2);
        DEV_B = allocate_floatmat_device(a_dim2, b_dim2);
        DEV_C = allocate_floatmat_device(a_dim1, b_dim2);
    }
    
    // Transfer the data to the device
    size_t size_a = a_dim1 * a_dim2 * sizeof(float);
    size_t size_b = a_dim2 * b_dim2 * sizeof(float);
    size_t size_c = a_dim1 * b_dim2 * sizeof(float);
    cudaMemcpy(DEV_A, a, size_a, cudaMemcpyHostToDevice); 
    cudaMemcpy(DEV_B, b, size_b, cudaMemcpyHostToDevice);

    if (HANDLE == NULL) {
        cublasCreate(&HANDLE);
    }

    // CUBLAS_OP_N means "do not transpose"
    float alpha = 1.0;
    float beta = 0.0;
    // Result of this multiplication is C = alpha * A * B + beta * C
    cublasSgemm(
        HANDLE, CUBLAS_OP_N, CUBLAS_OP_N,
        a_dim1, b_dim2, a_dim2,             // mats dimensionality info
        &alpha,
        DEV_A, a_dim1,                      // actual DEV_A buffer info
        DEV_B, a_dim2,                      // actual DEV_B buffer info
        &beta,
        DEV_C, a_dim1                       // actual DEV_C buffer info
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    // Transfer the data to the host
    CUDA_CHECK(cudaMemcpy(out, DEV_C, size_c, cudaMemcpyDeviceToHost));
}
