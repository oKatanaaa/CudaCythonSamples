#include "common.h"
#include "kernel.cu"
#include "heat_grid.h"


void evolve_heat(HeatGrid *grid, float speed, int time_steps) {
    int x_blocks = (grid->w + 16 - 1) / 16;
    int y_blocks = (grid->h + 16 - 1) / 16;
    dim3 blocks(x_blocks, y_blocks);
    dim3 threads(16, 16);

    // Start heat evolution
    CUDA_CHECK(cudaEventRecord(grid->start, 0));
    bool dstOut = true;
    for (int i = 0; i < time_steps * 2; i++) {
        float *in, *out;
        if (dstOut) {
            in = grid->dev_inSrc;
            out = grid->dev_outSrc;
        } else {
            in = grid->dev_outSrc;
            out = grid->dev_inSrc;
        }
        // Copy information about heaters in the grid
        copy_const_kernel<<<blocks, threads>>>(in, grid->w, grid->h);
        blend_kernel<<<blocks, threads>>>(out, dstOut, speed, grid->w, grid->h);
        dstOut = !dstOut;
    }
    float_to_color<<<blocks, threads>>>(grid->dev_heat_img, grid->dev_inSrc, grid->w, grid->h);
    CUDA_CHECK(cudaMemcpy(grid->heat_img, grid->dev_heat_img, grid->heat_img_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(grid->stop, 0));
    CUDA_CHECK(cudaEventSynchronize(grid->stop));
    grid->measure_elapsed_time();
    grid->frames += time_steps;
}


void bind_grid(HeatGrid* grid) {
    size_t null_offset = 0;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaBindTexture2D(
        &null_offset,
        texConstSrc,                            // The texture ptr
        grid->dev_constSrc, desc, grid->w, grid->h,   // Info about the buffer to be binded
        sizeof(float) * grid->w                // Pitch? Not sure what it means
    ));
    CUDA_CHECK(cudaBindTexture2D(
        &null_offset,
        texIn,                                  // The texture ptr
        grid->dev_inSrc, desc, grid->w, grid->h,      // Info about the buffer to be binded
        sizeof(float) * grid->w                // Pitch? Not sure what it means
    ));
    CUDA_CHECK(cudaBindTexture2D(
        &null_offset,
        texOut,                                 // The texture ptr
        grid->dev_outSrc, desc, grid->w, grid->h,     // Info about the buffer to be binded
        sizeof(float) * grid->w                // Pitch? Not sure what it means
    ));
    printf("Successfully binded grid.\n");
}


void unbind_grid() {
    CUDA_CHECK(cudaUnbindTexture(texIn));
    CUDA_CHECK(cudaUnbindTexture(texOut));
    CUDA_CHECK(cudaUnbindTexture(texConstSrc));
}
