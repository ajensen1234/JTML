#include "gpu/gpu_distance_map.cuh"

// We can just directly inherit some of the stuff from GPUFrame because the host image
// Exists as-is without needing to upload something separate.
// Just need to pass in the distance map instead of the intensity image.
namespace gpu_cost_function {
    GPUDistanceMap::GPUDistanceMap(int width, int height,
    int gpu_device,
    unsigned char* host_distance_map) : GPUFrame(width, height, gpu_device, host_distance_map){

    };

    GPUDistanceMap::~GPUDistanceMap(){};
    void GPUDistanceMap::write_out_distance_map(){
        GetGPUImage()->WriteImage("test_dm_image.png");
    }
}
