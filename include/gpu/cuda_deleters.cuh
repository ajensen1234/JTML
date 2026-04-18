#ifndef CUDA_DELETERS_H
#define CUDA_DELETERS_H

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>

#include "gpu/cuda_check.cuh"

struct CudaFreeDeleter {
    template <typename T>
    void operator()(T* ptr) const noexcept {
        if (ptr) {
            CUDA_CHECK(cudaFree(ptr));
        }
    }
};

struct CudaFreeHostDeleter {
    template <typename T>
    void operator()(T* ptr) const noexcept {
        if (ptr) {
            CUDA_CHECK(cudaFreeHost(ptr));
        }
    }
};

template <typename T>
using unique_device_ptr = std::unique_ptr<T, CudaFreeDeleter>;

template <typename T>
using unique_host_ptr = std::unique_ptr<T, CudaFreeHostDeleter>;

template <typename T>
[[nodiscard]] unique_device_ptr<T> make_cuda_unique(std::size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    return unique_device_ptr<T>(ptr);
}

template <typename T>
[[nodiscard]] unique_host_ptr<T> make_cuda_host_unique(std::size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    return unique_host_ptr<T>(ptr);
}

#endif // CUDA_DELETERS_H
