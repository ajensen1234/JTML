#pragma once

#include "gpu/gpu_frame.cuh"
#include "core/preprocessor-defs.h"\


namespace gpu_cost_function{
    class GPUDistanceMap : public GPUFrame {
        public:
            JTML_DLL GPUDistanceMap(int height, int width, int gpu_device, unsigned char* host_distance_map);
            JTML_DLL GPUDistanceMap();
            JTML_DLL ~GPUDistanceMap();
            JTML_DLL void write_out_distance_map();
    };
}
