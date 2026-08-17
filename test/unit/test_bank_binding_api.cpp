// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0
//
// Plan 010 U12 Stage 2: compile-time compatibility seam pins.  This test does
// not construct CUDA-owning objects and does not execute a new stream path; it
// ensures the non-owning bank/stream API remains additive and type-stable.
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

#include "compute/bank_state.cuh"
#include "compute/gpu_metrics.cuh"
#include "compute/render_engine.cuh"

using gpu_cost_function::BankState;
using gpu_cost_function::GPUMetrics;
using gpu_cost_function::RenderEngine;

TEST_CASE("U12 Stage 2 compatibility APIs are additive and non-owning", "[bank-binding]") {
    static_assert(std::is_same_v<decltype(&RenderEngine::SetActiveBank),
                                 void (RenderEngine::*)(BankState*)>);
    static_assert(std::is_same_v<decltype(&RenderEngine::SetExecutionStream),
                                 void (RenderEngine::*)(cudaStream_t)>);
    static_assert(std::is_same_v<decltype(&RenderEngine::GetActiveBank),
                                 BankState* (RenderEngine::*)() const>);
    static_assert(std::is_same_v<decltype(&RenderEngine::GetExecutionStream),
                                 cudaStream_t (RenderEngine::*)() const>);
    static_assert(std::is_same_v<decltype(&GPUMetrics::SetActiveBank),
                                 void (GPUMetrics::*)(BankState*)>);
    static_assert(std::is_same_v<decltype(&GPUMetrics::SetExecutionStream),
                                 void (GPUMetrics::*)(cudaStream_t)>);
    static_assert(std::is_same_v<decltype(&GPUMetrics::GetActiveBank),
                                 BankState* (GPUMetrics::*)() const>);
    static_assert(std::is_same_v<decltype(&GPUMetrics::GetExecutionStream),
                                 cudaStream_t (GPUMetrics::*)() const>);

    // A BankState is a view only. Stage 2 introduces no ownership or allocation.
    BankState view;
    REQUIRE(view.index == 0);
    REQUIRE_FALSE(view.in_flight);
}
