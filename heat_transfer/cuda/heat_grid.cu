#include "heat_grid.h"
#include "common.h"


void HeatGrid::init_grid(int w_, int h_) {
    printf("Initializing heat grid...\n");
    w = w_;
    h = h_;
    int n_bytes_float = w * h * sizeof(float);
    float_grid_n_bytes = n_bytes_float;
    CUDA_CHECK(cudaMalloc((void**)&dev_inSrc, n_bytes_float));
    CUDA_CHECK(cudaMalloc((void**)&dev_outSrc, n_bytes_float));
    CUDA_CHECK(cudaMalloc((void**)&dev_constSrc, n_bytes_float));
    host_outSrc = new float[w * h];

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    totalTime = 0.0;
    frames = 0.0;
    printf("Successfully initialized heat grid.\n");
}

void HeatGrid::init_heaters(float *heat_grid) {
    CUDA_CHECK(cudaMemcpy(dev_constSrc, heat_grid, float_grid_n_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_inSrc, heat_grid, float_grid_n_bytes, cudaMemcpyHostToDevice));
    printf("Successfully initialized dev_constSrc.\n");
}

void HeatGrid::measure_elapsed_time() {
    float elapsed_time;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    totalTime += elapsed_time;
}

void HeatGrid::destroy_grid() {
    CUDA_CHECK(cudaFree(dev_inSrc));
    CUDA_CHECK(cudaFree(dev_outSrc));
    CUDA_CHECK(cudaFree(dev_constSrc));

    delete[] host_outSrc;
}


