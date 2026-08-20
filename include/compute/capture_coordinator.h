/* Plan 012 U3: CaptureCoordinator — process-wide exclusive capture lock + park registry.
 * CUDA-free (no cuda_runtime, no Qt) so headless tests can include it.
 * Header is auto-globbed; src/compute/capture_coordinator.cpp provides impl.
 */
#pragma once

#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gpu_cost_function {

class CaptureCoordinator {
public:
    using ParkFn = std::function<bool()>;
    using UnparkFn = std::function<void()>;

    CaptureCoordinator() = default;
    CaptureCoordinator(const CaptureCoordinator&) = delete;
    CaptureCoordinator& operator=(const CaptureCoordinator&) = delete;

    // Register a producer by name. Idempotent by name: if name exists, overwrite its park/unpark.
    void registerProducer(const std::string& name, ParkFn park, UnparkFn unpark);

    // Try to acquire exclusive capture ownership with bounded wait.
    // Reentrant: if owner == this_thread, return true without double-park.
    // On success, all registered producers are parked (parkedCount() == producer count).
    // On park failure, already-parked producers are unparked and false is returned.
    bool tryAcquireExclusive(std::chrono::milliseconds timeout);

    // Release exclusive ownership. No-op if not owner. Unparks only producers that were parked this acquire.
    void release();

    bool isOwner() const;
    std::size_t parkedCount() const;

private:
    mutable std::timed_mutex mutex_;
    std::thread::id owner_{};
    std::vector<std::string> names_;
    std::vector<ParkFn> parks_;
    std::vector<UnparkFn> unparks_;
    std::vector<bool> parkedFlags_;
    std::size_t parked_ = 0;
};

}  // namespace gpu_cost_function
