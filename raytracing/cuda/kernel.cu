#include <math.h>
#include "common.h"
#include "sphere.cu"


const int N_SPHERES = 32;
__constant__ Sphere DEV_SPHERES[N_SPHERES];


__global__ void cudaRayTracing(unsigned char *img, int w, int h)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > w || y > h) {
        return;
    }
    int img_offset = x + y * blockDim.x * gridDim.x;

    float ox = (x - w / 2);
    float oy = (y - h / 2);
    // Red, Blue, Green
    float r = 0.0, g = 0.0, b = 0.0;
    float max_dist = -INF;

    // Determine which object is hit by the ray and get its color
    for (int i = 0; i < N_SPHERES; i++) {
        float intensity;
        float dist = DEV_SPHERES[i].hit(ox, oy, &intensity);
        if (dist > max_dist) {
            r = DEV_SPHERES[i].r * intensity;
            g = DEV_SPHERES[i].g * intensity;
            b = DEV_SPHERES[i].b * intensity;
        }
    }

    // Save the color data
    img[img_offset * 3 + 0] = (int) (r * 255);
    img[img_offset * 3 + 1] = (int) (g * 255);
    img[img_offset * 3 + 2] = (int) (b * 255);
}


unsigned char *DEV_IMG = NULL;
int current_n_bytes = -1;


void trace_spheres(unsigned char *img, int w, int h) {
    int n_bytes = w * h * 3 * sizeof(unsigned char);
    // Allocate memory on GPU if needed
    if (current_n_bytes != n_bytes) {
        // Free already the allocated memory if it actually was
        if (DEV_IMG != NULL) {
            CUDA_CHECK(cudaFree(DEV_IMG));
        }
        printf("Allocating memory on GPU. Old current_n_bytes=%d", current_n_bytes);
        current_n_bytes = n_bytes;
        printf("New current_n_bytes=%d", current_n_bytes);
        CUDA_CHECK(cudaMalloc(&DEV_IMG, n_bytes));
    }

    // Generate spheres, copy them into the constant memory and free the allocated memory
    Sphere* spheres = generate_random_spheres(N_SPHERES, w, h);
    CUDA_CHECK(cudaMemcpyToSymbol(DEV_SPHERES, spheres, sizeof(Sphere) * N_SPHERES));
    delete[] spheres;

    dim3 grids((w + 16 - 1) / 16, (h + 16 - 1) / 16);
    dim3 threads(16, 16);
    // Run the raytracing
    cudaRayTracing<<<grids, threads>>>(DEV_IMG, w, h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(img, DEV_IMG, n_bytes, cudaMemcpyDeviceToHost));
}
