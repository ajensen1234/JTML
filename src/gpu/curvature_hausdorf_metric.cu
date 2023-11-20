#include "gpu/gpu_metrics.cuh"

#include <cuda_runtime.h>
#include <cuda.h>

/*Grayscale Colors*/
#include "gpu/pixel_grayscale_colors.h"

/*Launch Parameters*/
#include "cuda_launch_parameters.h"


__global__ void CurvatureHausdorfMetric_Kernel(){};
__global__ void Reset_CurvatureHausdorfScore_Kernel(int* dev_curv_haus_score){
    int i = threadIdx.x;
    dev_curv_haus_score[i] = 0;
}

namespace gpu_cost_function{
    double GPUMetrics::CurvatureHeatmapMetric(GPUImage* projected_image, GPUHeatmap* gpu_heatmap){
        double test = 0;
        return test;
    };
};
