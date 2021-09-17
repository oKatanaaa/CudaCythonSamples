#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>


class HeatGrid {
    public:
        int w;
        int h;
        int float_grid_n_bytes;
        float           *dev_inSrc;
        float           *dev_outSrc;
        float           *dev_constSrc;
        float           *host_outSrc;

        cudaEvent_t     start;
        cudaEvent_t     stop;
        float           totalTime;
        int             frames;

        void init_grid(int w, int h);
        void init_heaters(float *heat_grid);
        void measure_elapsed_time();
        void destroy_grid();
};
