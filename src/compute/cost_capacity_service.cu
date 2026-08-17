/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */
/* @file cost_capacity_service.cu
 *
 * Plan 010 U9/U12: device capacity queries plus the service-owned extra-bank
 * lifecycle. Bank 0 remains owned by the existing RenderEngine/GPUMetrics
 * compatibility path; this pool owns only banks 1..N-1.
 */
#include "compute/cost_capacity_service.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace gpu_cost_function {

bool DeviceAlloc(void** pointer, std::size_t bytes) {
    if (bytes == 0) {
        *pointer = nullptr;
        return true;
    }
    return cudaMalloc(pointer, bytes) == cudaSuccess;
}

bool HostAlloc(void** pointer, std::size_t bytes) {
    if (bytes == 0) {
        *pointer = nullptr;
        return true;
    }
    return cudaHostAlloc(pointer, bytes, cudaHostAllocDefault) == cudaSuccess;
}

void FreeDevice(void* pointer) {
    if (pointer != nullptr) cudaFree(pointer);
}

void FreeHost(void* pointer) {
    if (pointer != nullptr) cudaFreeHost(pointer);
}

struct BankAllocation {
    BankState view;

    ~BankAllocation() {
        if (view.stream != nullptr) {
            auto stream = reinterpret_cast<cudaStream_t>(view.stream);
            cudaStreamSynchronize(stream);
        }
        if (view.completion_event != nullptr) {
            cudaEventDestroy(reinterpret_cast<cudaEvent_t>(view.completion_event));
        }
        if (view.stream != nullptr) {
            cudaStreamDestroy(reinterpret_cast<cudaStream_t>(view.stream));
        }
        Release(view.primary);
        Release(view.secondary);
        Release(view.metrics);
    }

    static void Release(RenderBuffers& render) {
        FreeDevice(render.output);
        FreeHost(render.host_bounding_box);
        FreeDevice(render.dev_backface);
        FreeDevice(render.dev_transformed_vertex_zs);
        FreeDevice(render.dev_tangent_triangle);
        FreeDevice(render.dev_projected_triangles);
        FreeDevice(render.dev_projected_triangles_snapped);
        FreeDevice(render.dev_bounding_box_triangles);
        FreeDevice(render.dev_bounding_box_triangles_sizes);
        FreeDevice(render.dev_bounding_box_triangles_sizes_prefix);
        FreeDevice(render.dev_bounding_box);
        FreeDevice(render.dev_fragment_fill);
        FreeHost(render.host_fragment_fill);
        FreeDevice(render.dev_stride_prefixes);
        FreeDevice(render.dev_cub_storage);
    }

    static void Release(MetricBuffers& metrics) {
        FreeHost(metrics.host_pixel_score);
        FreeDevice(metrics.dev_pixel_score);
        FreeHost(metrics.host_intersection);
        FreeHost(metrics.host_union);
        FreeDevice(metrics.dev_intersection);
        FreeDevice(metrics.dev_union);
        FreeHost(metrics.host_white_count);
        FreeDevice(metrics.dev_white_count);
        FreeHost(metrics.host_distance_score);
        FreeDevice(metrics.dev_distance_score);
        FreeHost(metrics.host_edge_count);
        FreeDevice(metrics.dev_edge_count);
        FreeHost(metrics.host_curvature);
        FreeDevice(metrics.dev_curvature);
    }

    static bool AllocateRender(RenderBuffers& render,
                               const BankFootprintInput& in) {
        const std::size_t pixels = in.width * in.height;
        const std::size_t triangles = in.triangle_count;
        const std::size_t stride = in.maximum_stride_size;
        const std::size_t cub = in.cub_storage_bytes;
        return DeviceAlloc(&render.output, pixels * sizeof(std::uint8_t)) &&
               HostAlloc(&render.host_bounding_box, 4 * sizeof(std::int32_t)) &&
               DeviceAlloc(&render.dev_backface, triangles * sizeof(std::uint8_t)) &&
               DeviceAlloc(&render.dev_transformed_vertex_zs,
                           3 * triangles * sizeof(float)) &&
               DeviceAlloc(&render.dev_tangent_triangle,
                           3 * triangles * sizeof(std::uint8_t)) &&
               DeviceAlloc(&render.dev_projected_triangles,
                           6 * triangles * sizeof(float)) &&
               DeviceAlloc(&render.dev_projected_triangles_snapped,
                           6 * triangles * sizeof(std::int32_t)) &&
               DeviceAlloc(&render.dev_bounding_box_triangles,
                           4 * triangles * sizeof(std::int32_t)) &&
               DeviceAlloc(&render.dev_bounding_box_triangles_sizes,
                           triangles * sizeof(std::int32_t)) &&
               DeviceAlloc(&render.dev_bounding_box_triangles_sizes_prefix,
                           triangles * sizeof(std::int32_t)) &&
               DeviceAlloc(&render.dev_bounding_box, 4 * sizeof(std::int32_t)) &&
               DeviceAlloc(&render.dev_fragment_fill, sizeof(std::int32_t)) &&
               HostAlloc(&render.host_fragment_fill, sizeof(std::int32_t)) &&
               DeviceAlloc(&render.dev_stride_prefixes,
                           stride * sizeof(std::int32_t)) &&
               DeviceAlloc(&render.dev_cub_storage, cub);
    }

    static bool AllocateMetrics(MetricBuffers& metrics,
                                const BankFootprintInput& in) {
        const std::size_t curvature = in.curvature_capacity;
        return HostAlloc(&metrics.host_pixel_score, sizeof(std::int32_t)) &&
               DeviceAlloc(&metrics.dev_pixel_score, sizeof(std::int32_t)) &&
               HostAlloc(&metrics.host_intersection, sizeof(std::int32_t)) &&
               HostAlloc(&metrics.host_union, sizeof(std::int32_t)) &&
               DeviceAlloc(&metrics.dev_intersection, sizeof(std::int32_t)) &&
               DeviceAlloc(&metrics.dev_union, sizeof(std::int32_t)) &&
               HostAlloc(&metrics.host_white_count, sizeof(std::int32_t)) &&
               DeviceAlloc(&metrics.dev_white_count, sizeof(std::int32_t)) &&
               HostAlloc(&metrics.host_distance_score, sizeof(std::int32_t)) &&
               DeviceAlloc(&metrics.dev_distance_score, sizeof(std::int32_t)) &&
               HostAlloc(&metrics.host_edge_count, sizeof(std::int32_t)) &&
               DeviceAlloc(&metrics.dev_edge_count, sizeof(std::int32_t)) &&
               HostAlloc(&metrics.host_curvature,
                         curvature * sizeof(std::int32_t)) &&
               DeviceAlloc(&metrics.dev_curvature,
                           curvature * sizeof(std::int32_t));
    }

    static std::unique_ptr<BankAllocation> Create(
        std::size_t index, const BankFootprintInput& in) {
        auto bank = std::make_unique<BankAllocation>();
        bank->view.index = index;
        bank->view.in_flight = false;
        if (!AllocateRender(bank->view.primary, in) ||
            (in.biplane && !AllocateRender(bank->view.secondary, in)) ||
            !AllocateMetrics(bank->view.metrics, in)) {
            return nullptr;
        }
        cudaStream_t stream = nullptr;
        cudaEvent_t event = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
            if (event != nullptr) cudaEventDestroy(event);
            if (stream != nullptr) cudaStreamDestroy(stream);
            return nullptr;
        }
        bank->view.stream = reinterpret_cast<void*>(stream);
        bank->view.completion_event = reinterpret_cast<void*>(event);
        return bank;
    }
};

class BankStatePool {
public:
    bool Configure(std::size_t count, const BankFootprintInput& layout) {
        capacity_ = count;
        layout_ = layout;
        banks_.clear();
        banks_.resize(count > 0 ? count - 1 : 0);
        return count >= 1;
    }

    std::size_t Size() const { return capacity_; }

    int Checkout() {
        for (std::size_t offset = 0; offset < banks_.size(); ++offset) {
            if (banks_[offset] == nullptr) {
                banks_[offset] = BankAllocation::Create(offset + 1, layout_);
                if (banks_[offset] == nullptr) {
                    // Allocation failure fails closed: discard every extra bank
                    // and preserve the compatibility bank 0 only.
                    banks_.clear();
                    capacity_ = 1;
                    return -1;
                }
            }
            if (!banks_[offset]->view.in_flight) {
                banks_[offset]->view.in_flight = true;
                return static_cast<int>(banks_[offset]->view.index);
            }
        }
        return -1;
    }

    bool Recycle(std::size_t index, bool completion_ready) {
        BankState* state = Find(index);
        if (state == nullptr || !state->in_flight || !completion_ready) return false;
        state->in_flight = false;
        return true;
    }

    bool InFlight(std::size_t index) const {
        const BankState* state = Find(index);
        return state != nullptr && state->in_flight;
    }

    const BankState* State(std::size_t index) const { return Find(index); }

private:
    BankState* Find(std::size_t index) {
        if (index == 0 || index > banks_.size() || banks_[index - 1] == nullptr) return nullptr;
        return &banks_[index - 1]->view;
    }

    const BankState* Find(std::size_t index) const {
        if (index == 0 || index > banks_.size() || banks_[index - 1] == nullptr) return nullptr;
        return &banks_[index - 1]->view;
    }

    std::size_t capacity_ = 1;
    BankFootprintInput layout_;
    std::vector<std::unique_ptr<BankAllocation>> banks_;
};

CostCapacityService::CostCapacityService() = default;

CostCapacityService::~CostCapacityService() = default;

bool CostCapacityService::ConfigurePool(const BankFootprintInput& layout,
                                        std::size_t n_max) {
    const BankFootprint measured = bank_state_math::footprint(layout);
    const BankAdmission admission =
        bank_state_math::admit(static_cast<std::uint64_t>(snap_.free_device_bytes),
                               measured, n_max);
    snap_.per_bank_footprint_bytes =
        measured.valid ? static_cast<std::int64_t>(measured.total_bytes) : 0;
    snap_.n_max = n_max > static_cast<std::size_t>(INT_MAX)
                      ? INT_MAX
                      : static_cast<int>(n_max);
    pool_.reset();
    if (!admission.admitted) return false;

    auto candidate = std::make_unique<BankStatePool>();
    if (!candidate->Configure(admission.bank_count, layout)) return false;
    pool_ = std::move(candidate);
    return true;
}

std::size_t CostCapacityService::poolSize() const {
    return pool_ == nullptr ? 1 : pool_->Size();
}

int CostCapacityService::CheckoutBank() {
    return pool_ == nullptr ? -1 : pool_->Checkout();
}

bool CostCapacityService::RecycleBank(std::size_t index, bool completion_ready) {
    return pool_ != nullptr && pool_->Recycle(index, completion_ready);
}

bool CostCapacityService::bankInFlight(std::size_t index) const {
    return pool_ != nullptr && pool_->InFlight(index);
}

const BankState* CostCapacityService::bankState(std::size_t index) const {
    return pool_ == nullptr ? nullptr : pool_->State(index);
}

bool CostCapacityService::refreshDeviceSnapshot(int device) {
    cudaDeviceProp props{};
    cudaError_t err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess) {
        snap_ = DeviceCapacitySnapshot{};
        return false;
    }

    size_t free_bytes = 0, total_bytes = 0;
    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        snap_ = DeviceCapacitySnapshot{};
        return false;
    }

    snap_.sm_count = props.multiProcessorCount;
    snap_.max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    snap_.grid_dim_limit = props.maxGridSize[0];
    snap_.free_device_bytes = static_cast<std::int64_t>(free_bytes);
    snap_.safe_cap = static_cast<std::int64_t>(maximum_stride_size) *
                     (threads_per_block - 1);
    snap_.n_max = 0;
    return isCapacityAvailable(snap_);
}

int CostCapacityService::occupancyOptimalBlockSize(const void* kernel_func) {
    if (kernel_func == nullptr) return threads_per_block;
    int min_grid = 0;
    int block = 0;
    cudaError_t err =
        cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, kernel_func, 0, 0);
    return err == cudaSuccess && block > 0 ? block : threads_per_block;
}

}  // namespace gpu_cost_function
