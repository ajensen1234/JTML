/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*
 * cost.cpp — implementation of the FFI-facing cost handle (cost.h).
 *
 * build_cpp_cost boards a std::function<double(const Point6D&)> into a
 * CppCost so the Rust DIRECT port can call it through the CXX opaque-type
 * bridge (evaluate(Point6D) -> double). This file carries no Qt/GPU — the
 * concrete std::function is assembled upstream by the GPU cost adapter
 * (BuildGpuCostAdapter, coordinator/compute side) and injected here.
 */
#include "domain/cost.h"

std::unique_ptr<CppCost> build_cpp_cost(CppCost::CostFunction fn) {
    return std::make_unique<CppCost>(std::move(fn));
}
