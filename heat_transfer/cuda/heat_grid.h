#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>


class HeatGrid {
    public:
        int w;
        int h;
        int heat_img_size;
        unsigned char   *dev_heat_img;
        float           *dev_inSrc;
        float           *dev_outSrc;
        float           *dev_constSrc;
        unsigned char   *heat_img;

        cudaEvent_t     start;
        cudaEvent_t     stop;
        float           totalTime;
        int             frames;

        void init_grid(int w, int h);
        void init_heaters(float *heat_grid);
        void measure_elapsed_time();
        void destroy_grid();
};
