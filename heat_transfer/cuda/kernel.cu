#include <cuda.h>


texture<float, 2> texConstSrc;
texture<float, 2> texIn;
texture<float, 2> texOut;


__global__ void copy_const_kernel(float *input_grid, int w, int h) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y + blockDim.y;
    int offset = x + y * blockDim.x * w;
    if (x > w || y > h) return;

    float c = tex2D(texConstSrc, x, y);
    if (c != 0.0) input_grid[offset] = c;
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
    int y = threadIdx.y + blockIdx.y + blockDim.y;
    int offset = x + y * blockDim.x * w;
    if (x > w || y > h) return;

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

// Adapted from https://github.com/tpn/cuda-by-example/blob/master/common/book.h

__device__ unsigned char value( float n1, float n2, int hue ) {
    if (hue > 360)      hue -= 360;
    else if (hue < 0)   hue += 360;

    if (hue < 60)
        return (unsigned char)(255 * (n1 + (n2-n1)*hue/60));
    if (hue < 180)
        return (unsigned char)(255 * n2);
    if (hue < 240)
        return (unsigned char)(255 * (n1 + (n2-n1)*(240-hue)/60));
    return (unsigned char)(255 * n1);
}


__global__ void float_to_color(unsigned char *optr, const float *outSrc, int w_, int h_) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * w_;
    if (x > w_ || y > h_) return;

    float l = outSrc[offset];
    float s = 1;
    int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
    float m1, m2;

    if (l <= 0.5f)
        m2 = l * (1 + s);
    else
        m2 = l + s - l * s;
    m1 = 2 * l - m2;

    optr[offset*4 + 0] = value( m1, m2, h+120 );
    optr[offset*4 + 1] = value( m1, m2, h );
    optr[offset*4 + 2] = value( m1, m2, h -120 );
}

