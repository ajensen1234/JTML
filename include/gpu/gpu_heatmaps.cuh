#pragma once
#include "core/preprocessor-defs.h"

namespace gpu_cost_function{
    class GPUHeatmap{
        public:
            JTML_DLL GPUHeatmap(int width, int height,
                                int gpu_device, int num_keypoints,
                                unsigned char* host_heatmaps);
            JTML_DLL GPUHeatmap();
            JTML_DLL ~GPUHeatmap();

            JTML_DLL unsigned char* GetDeviceHeatmapPointer();

            JTML_DLL int GetFrameWidth();
            JTML_DLL int GetFrameHeight();
            JTML_DLL int GetNumKeypoints();

            JTML_DLL bool IsInitializedCorrectly();

        private:
            int height_;
            int width_;
            int num_keypoints_;
            bool initialized_correctly_;
            bool heatmap_on_gpu_;
            unsigned char* dev_heatmap_;
            int device_;
    };
}
