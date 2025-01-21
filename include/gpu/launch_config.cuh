#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <optional>

namespace gpu_cost_function {

struct LaunchConfiguration {
    dim3 block_size{1, 1, 1};
    dim3 grid_size{1, 1, 1};
    std::size_t shared_mem_size{0};
    float occupancy{0.0f};
};

class LaunchConfigBuilder {
public:
    static std::optional<LaunchConfiguration> buildTriangleConfig(
        int triangle_count, int gpu_device, bool verbose = false) {

        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, gpu_device);

        LaunchConfiguration config;

        // More conservative thread block size - aim for 256 threads per block
        const unsigned int TARGET_THREADS = 256;
        unsigned int block_dim = static_cast<unsigned int>(
            std::floor(std::sqrt(static_cast<double>(TARGET_THREADS))));
        
        // This will give us 16x16 = 256 threads per block
        config.block_size = dim3(block_dim, block_dim);

        // Calculate grid size based on triangle count and block size
        unsigned int grid_dim = static_cast<unsigned int>(std::ceil(
            std::sqrt(static_cast<double>(triangle_count) / 
                     static_cast<double>(block_dim * block_dim))));

        config.grid_size = dim3(grid_dim, grid_dim);

        // Calculate occupancy
        unsigned int blocks_per_sm = 
            (props.maxThreadsPerMultiProcessor + TARGET_THREADS - 1) / 
            TARGET_THREADS;
            
        config.occupancy = 
            static_cast<float>(TARGET_THREADS * blocks_per_sm) /
            static_cast<float>(props.maxThreadsPerMultiProcessor);

        if (verbose) {
            printf("Triangle Processing Configuration:\n");
            printf("  Block size: %d x %d\n", block_dim, block_dim);
            printf("  Grid size: %d x %d\n", grid_dim, grid_dim);
            printf("  Occupancy: %.2f%%\n", config.occupancy * 100.0f);
            printf("  Blocks per SM: %d\n", blocks_per_sm);
        }

        return config;
    }

    static std::optional<LaunchConfiguration> buildVertexConfig(
        int triangle_count, int gpu_device, bool verbose = false) {

        auto config =
            buildTriangleConfig(triangle_count * 3, gpu_device, false);

        if (verbose && config) {
            printf("Vertex Processing Configuration:\n");
            printf(
                "  Block size: %d x %d\n",
                config->block_size.x,
                config->block_size.y);
            printf(
                "  Grid size: %d x %d\n",
                config->grid_size.x,
                config->grid_size.y);
            printf("  Occupancy: %.2f%%\n", config->occupancy * 100.0f);
        }

        return config;
    }

    static std::optional<LaunchConfiguration> buildBoundingBoxConfig(
        int triangle_count, int gpu_device, bool verbose = false) {

        auto config =
            buildTriangleConfig(triangle_count * 4, gpu_device, false);

        if (verbose && config) {
            printf("Bounding Box Processing Configuration:\n");
            printf(
                "  Block size: %d x %d\n",
                config->block_size.x,
                config->block_size.y);
            printf(
                "  Grid size: %d x %d\n",
                config->grid_size.x,
                config->grid_size.y);
            printf("  Occupancy: %.2f%%\n", config->occupancy * 100.0f);
        }

        return config;
    }
};

} // namespace gpu_cost_function
