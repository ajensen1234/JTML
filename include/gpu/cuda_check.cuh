#pragma once

#include <cstdio>

#include "cuda_runtime.h"

#define CUDA_CHECK(call) \
    do { cudaError_t _e = (call); \
         if (_e != cudaSuccess) \
             fprintf(stderr, "CUDA error %s:%d: %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(_e)); \
    } while(0)

#define CUDA_CHECK_KERNEL(...) \
    do { __VA_ARGS__; \
         cudaError_t _e = cudaGetLastError(); \
         if (_e != cudaSuccess) \
             fprintf(stderr, "CUDA error %s:%d: %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(_e)); \
    } while(0)
