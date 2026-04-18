#ifndef CUDA_DELETERS_H
#define CUDA_DELETERS_H

#include <cuda_runtime.h> // For cudaFree and cudaFreeHost
#include <memory>         // For std::unique_ptr

// Custom deleter for cudaFree
struct CudaFreeDeleter {
    template <typename T>
    void operator()(T* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

// Custom deleter for cudaFreeHost
struct CudaFreeHostDeleter {
    template <typename T>
    void operator()(T* ptr) const {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
};

#endif // CUDA_DELETERS_H