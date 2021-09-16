#include "heat_grid.h"
#include "common.h"


void HeatGrid::init_grid(int w_, int h_) {
    printf("Initializing heat grid...\n");
    w = w_;
    h = h_;
    int n_bytes_float = w * h * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&dev_inSrc, n_bytes_float));
    CUDA_CHECK(cudaMalloc((void**)&dev_outSrc, n_bytes_float));
    CUDA_CHECK(cudaMalloc((void**)&dev_constSrc, n_bytes_float));

    heat_img_size = w * h * sizeof(unsigned char);
    CUDA_CHECK(cudaMalloc((void**)&dev_heat_img, heat_img_size));
    heat_img = new unsigned char[w * h];
    if (heat_img == NULL) {
        printf("Unable to initialize heat_img.\n");
    }

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    totalTime = 0.0;
    frames = 0.0;
    printf("Successfully initialized heat grid.\n");
}

void HeatGrid::init_heaters(float *heat_grid) {
    CUDA_CHECK(cudaMemcpy(dev_constSrc, heat_grid, w * h * sizeof(float), cudaMemcpyHostToDevice));
    printf("Successfully initialized dev_constSrc.\n");
}

void HeatGrid::measure_elapsed_time() {
    float elapsed_time;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    totalTime += elapsed_time;
}

void HeatGrid::destroy_grid() {
    CUDA_CHECK(cudaFree(dev_heat_img));
    CUDA_CHECK(cudaFree(dev_inSrc));
    CUDA_CHECK(cudaFree(dev_outSrc));
    CUDA_CHECK(cudaFree(dev_constSrc));

    delete[] heat_img;
}


