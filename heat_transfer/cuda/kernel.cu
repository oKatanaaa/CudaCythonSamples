#include <cuda.h>


texture<float, 2> texConstSrc;
texture<float, 2> texIn;
texture<float, 2> texOut;


__global__ void copy_const_kernel(float *input_grid, int w, int h) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * w;
    if (x >= w || y >= h) return;

    float c = tex2D(texConstSrc, x, y);
    if (c != 0) input_grid[offset] = c;
    //input_grid[offset] = 1.0;
}


/**
* Performs heat redistribution in the grid.
*
* @param dst The buffer to write the result in.
* @param dstOut Indicates which texture memory to use (texIn or texOut) for
* taking the initial temperature values.
* @param w Width of the grid.
* @param h Height of the grid.
*/
__global__ void blend_kernel(float *dst, bool dstOut, float speed, int w, int h) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * w;
    if (x >= w || y >= h) return;

    // `c` stands for center
    float t, l, c, r, b;
    if (dstOut) {
        t = tex2D(texIn, x, y - 1);
        l = tex2D(texIn, x - 1, y);
        c = tex2D(texIn, x, y);
        r = tex2D(texIn, x + 1, y);
        b = tex2D(texIn, x, y + 1);
    } else {
        t = tex2D(texOut, x, y - 1);
        l = tex2D(texOut, x - 1, y);
        c = tex2D(texOut, x, y);
        r = tex2D(texOut, x + 1, y);
        b = tex2D(texOut, x, y + 1);
    }
    dst[offset] = c + speed * (t + b + r + l - 4 * c);
}

