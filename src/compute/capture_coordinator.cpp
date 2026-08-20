/* Plan 012 U3: CaptureCoordinator implementation — CUDA-free. */

#include "compute/capture_coordinator.h"

namespace gpu_cost_function {

void CaptureCoordinator::registerProducer(const std::string& name, ParkFn park, UnparkFn unpark) {
    for (std::size_t i = 0; i < names_.size(); ++i) {
        if (names_[i] == name) {
            parks_[i] = std::move(park);
            unparks_[i] = std::move(unpark);
            return;
        }
    }
    names_.push_back(name);
    parks_.push_back(std::move(park));
    unparks_.push_back(std::move(unpark));
    parkedFlags_.push_back(false);
}

bool CaptureCoordinator::tryAcquireExclusive(std::chrono::milliseconds timeout) {
    if (owner_ == std::this_thread::get_id()) {
        return true;
    }
    if (!mutex_.try_lock_for(timeout)) {
        return false;
    }
    owner_ = std::this_thread::get_id();
    parked_ = 0;
    parkedFlags_.assign(names_.size(), false);
    for (std::size_t i = 0; i < names_.size(); ++i) {
        bool ok = parks_[i] ? parks_[i]() : true;
        if (!ok) {
            // Roll back already-parked producers
            for (std::size_t j = 0; j < i; ++j) {
                if (parkedFlags_[j] && unparks_[j]) {
                    unparks_[j]();
                }
            }
            parked_ = 0;
            parkedFlags_.assign(names_.size(), false);
            owner_ = std::thread::id{};
            mutex_.unlock();
            return false;
        }
        parkedFlags_[i] = true;
        ++parked_;
    }
    return true;
}

void CaptureCoordinator::release() {
    if (owner_ != std::this_thread::get_id()) {
        return;
    }
    for (std::size_t i = 0; i < names_.size(); ++i) {
        if (i < parkedFlags_.size() && parkedFlags_[i] && unparks_[i]) {
            unparks_[i]();
        }
    }
    parked_ = 0;
    parkedFlags_.assign(names_.size(), false);
    owner_ = std::thread::id{};
    mutex_.unlock();
}

bool CaptureCoordinator::isOwner() const {
    return owner_ == std::this_thread::get_id();
}

std::size_t CaptureCoordinator::parkedCount() const {
    return parked_;
}

}  // namespace gpu_cost_function
