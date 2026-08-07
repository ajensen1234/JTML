// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#pragma once

#include <functional>
#include <vector>

#include "core/data_structures_6D.h"
#include "core/direct_data_storage.h"

// A generic DIRECT (DIviding RECTangles) global optimizer over a 6-D unit cube,
// with the cost function injected and zero Qt/VTK/CUDA/JTML dependencies.
//
// This is a faithful extraction of the DIRECT loop that ran inside
// OptimizerManager (see src/core/optimizer_manager.cpp): ConvexHull (Jarvis
// gift-wrapping), TrisectPotentiallyOptimal, DenormalizeRange /
// DenormalizeFromCenter, and the budget loop are preserved verbatim. The lone
// GPU touchpoint -- EvaluateCostFunction -- is replaced by the injected `cost`
// callback, which is invoked with the DENORMALIZED physical point.
//
// The search operates on a unit hypercube centered at (0.5, ..., 0.5). A unit
// point is mapped to a physical point via `starting_point` and `range`:
//   physical = starting_point + (unit - 0.5) * 2 * range
class DirectOptimizer {
public:
    using CostFunction = std::function<double(const Point6D&)>;

    DirectOptimizer(CostFunction cost, Point6D range, Point6D starting_point,
                    unsigned int budget);

    // Run the DIRECT loop until the budget is consumed, a stop is requested, or
    // an internal error occurs. Returns false on error (e.g. an all-zero range
    // or an empty storage matrix).
    bool Run();

    // Number of cost-function calls so far, including the seed evaluation.
    unsigned int GetCostFunctionCalls() const;

    // Best location (physical/denormalized) and cost found so far.
    Point6D GetOptimumLocation() const;
    double GetOptimumValue() const;

    // Cooperative early-stop request, checked between iterations (the DIRECT
    // loop cannot be interrupted mid-kernel by design).
    void Stop();

private:
    void ConvexHull();
    void TrisectPotentiallyOptimal();
    double EvaluateCostFunction(Point6D unit_point);
    Point6D DenormalizeRange(Point6D unit_point) const;
    Point6D DenormalizeFromCenter(Point6D unit_point) const;

    CostFunction cost_;

    Point6D range_;
    Point6D starting_point_;
    bool valid_range_ = false;

    unsigned int budget_ = 0;
    unsigned int cost_function_calls_ = 0;

    DirectDataStorage data_;
    std::vector<int> potentially_optimal_col_ids_;
    std::vector<HyperBox6D> potentially_optimal_hyperboxes_;

    bool error_occurrred_ = false;
    bool stop_requested_ = false;

    double current_optimum_value_ = 0.0;
    Point6D current_optimum_location_;
};
