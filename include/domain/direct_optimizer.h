// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#pragma once

#include <functional>
#include <optional>
#include <variant>
#include <vector>

#include "domain/cost.h"
#include "domain/data_structures_6D.h"
#include "domain/direct_data_storage.h"

// A generic DIRECT (DIviding RECTangles) global optimizer over a 6-D unit cube,
// with the cost function injected and zero Qt/VTK/CUDA/JTML dependencies.
//
// This is a faithful extraction of the DIRECT loop that ran inside
// OptimizerManager (see src/coordinator/optimizer_manager.cpp): ConvexHull
// (Jarvis gift-wrapping), TrisectPotentiallyOptimal, DenormalizeRange /
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

    // Plan 010 U11 (R12): the optional BATCH cost-query sibling. When set, the
    // per-iteration POH center batch replaces the per-point eval loop with ONE
    // call, results returned in INPUT ORDER; per-eval bookkeeping (calls++,
    // non-finite handling, optimum update, storage, improvement callback)
    // REPLAYS in that order, so cost_function_calls_, the optimum sequence,
    // storage order, and callback firing ORDER are identical whether the cost
    // layer batches or not. MAY batch any pairwise-independent eval set; the
    // single-point path is unchanged when unset. No CUDA types cross this
    // header -- how the cost layer executes the batch is its own detail.
    using BatchCostFunction =
        std::function<std::vector<double>(const std::vector<Point6D>&)>;

    // Per-stage optimizer-variant slot (plan 008 U8, origin R3). Plain data
    // whose defaults reproduce today's classic-DIRECT search BIT-IDENTICALLY:
    // each field maps line-by-line onto the current code -- Original
    // selection = the Jarvis gift-wrap hull with `slope >= highest_slope` and
    // no epsilon filter anywhere; epsilon = 0.0 disables any post-filter
    // entirely; delta_limit off; size_measure L2 = the sqrt-norm column size;
    // split_rule OneSide = one-side trisection on the largest DENORMALIZED
    // side; ties All = equal slopes included (no tie-breaking); center
    // sampling; hidden_constraints off (GLh surrogate); globally_biased off
    // (gb phase switch). Guarded-divergence contract (the run's R13 proof
    // strategy): every future divergence branch MUST guard on "different from
    // default" before diverging, so the defaults stay bit-identical.
    //
    // Scope boundary (plan 008, review-resolved): non-default fields are
    // FAIL-FAST STUBS in this unit -- selecting one makes the constructor
    // throw std::invalid_argument. No variant semantics ship here; the
    // divergence branches land with the algorithm plan (R9).
    struct Options {
        enum class SelectionMode { Original }; // today's Jarvis gift-wrap hull
        enum class SizeMeasure { L2 };         // sqrt-norm column size
        enum class SplitRule {
            OneSide
        }; // largest-denormalized one-side trisection
        enum class TieSelection {
            All
        }; // slope >= highest_slope keeps every tie

        // User-provided (empty) so `Options()` is valid as the ctor's default
        // argument below: a defaulted ctor would need this nested class's
        // default member initializers before the end of the enclosing class
        // (ill-formed). The per-field initializers still apply on every
        // construction path (verified: default-init, value-init, list-init).
        Options() {}

        SelectionMode selection = SelectionMode::Original;
        double epsilon = 0.0; // 0.0 disables the post-filter entirely
        bool delta_limit = false;
        SizeMeasure size_measure = SizeMeasure::L2;
        SplitRule split_rule = SplitRule::OneSide;
        TieSelection ties = TieSelection::All;
        bool hidden_constraints = false; // GLh surrogate, off
        bool globally_biased = false;    // gb phase switch, off
    };

    explicit DirectOptimizer(
        CostFunction cost,
        Point6D range,
        Point6D starting_point,
        unsigned int budget,
        Options options = Options());

    explicit DirectOptimizer(
        CppCost cost,
        Point6D range,
        Point6D starting_point,
        unsigned int budget,
        Options options = Options());

    // Run the DIRECT loop until the budget is consumed, a stop is requested, or
    // an internal error occurs. Returns false on error (e.g. an all-zero range
    // or an empty storage matrix).
    bool Run();

    // Number of cost-function calls so far, including the seed evaluation.
    unsigned int GetCostFunctionCalls() const;

    // Number of cost evaluations that returned a non-finite (NaN/Inf) result
    // (plan 008 U3). The finite-check lives at the single shared eval
    // chokepoint (EvaluateCostFunction), so every injected-cost variant is
    // covered regardless of the cost implementation. A non-finite eval is
    // INFEASIBLE: it is never stored, never updates the optimum, and is
    // surfaced here (plus one iteration-callback fire). Always 0 on the
    // all-finite path (DIRECT_DILATION is provably finite), so production
    // behavior is bit-identical to the pre-check code.
    unsigned int GetNonFiniteCount() const;

    // Best location (physical/denormalized) and cost found so far.
    Point6D GetOptimumLocation() const;
    double GetOptimumValue() const;

    // Cooperative early-stop request, checked between iterations (the DIRECT
    // loop cannot be interrupted mid-kernel by design).
    void Stop();

    // Optional callbacks for driving the production UI from the extracted
    // optimizer (R4/R6, plan U6). Both are backward compatible: unset by
    // default, and only ever *reported* back to the caller -- they do not
    // affect the search.

    using IterationCallback = std::function<void()>;
    using ImprovementCallback = std::function<void(const Point6D&, double)>;

    // Extend the cumulative budget by a fixed number of pre-consumed calls.
    // The production app runs one stage per DirectOptimizer instance while
    // keeping a single running counter across stages (the canonical
    // cumulative caps are trunk 20k -> 2x branch 25k/30k -> leaf 35k).
    // Setting the offset makes GetCostFunctionCalls() and the loop guard
    // reflect the stage's position in that cumulative count.
    void SetCallOffset(unsigned int offset);

    // Called after each ConvexHull + Trisect iteration (drives a ~30fps
    // progress display).
    void SetIterationCallback(IterationCallback cb);

    // Called whenever the search finds a new best point (drives a live
    // optimum display). Receives the physical/denormalized location and value.
    void SetImprovementCallback(ImprovementCallback cb);

    // Plan 010 U11 (R12): install the optional batch cost-query sibling.
    // Unset (default) => the exact pre-unit serial path. When set, the POH
    // center batch is evaluated in one call and results are replayed in input
    // order (the Tier-0 replay contract).
    void SetBatchCost(BatchCostFunction cb);

private:
    void ConvexHull();
    void TrisectPotentiallyOptimal();
    // Evaluates the injected cost at the given unit point. Returns the cost,
    // or std::nullopt when the cost returned a non-finite (NaN/Inf) result --
    // the eval is then infeasible and the caller must not store it. The GLh
    // surrogate hook (the point where DIRECT-GLh would substitute
    // phi = f_min + ||x - x_min|| and treat the eval as finite) is RESERVED
    // here -- plan 008 U3 does not wire it.
    std::optional<double> EvaluateCostFunction(Point6D unit_point);
    [[nodiscard]] Point6D DenormalizeRange(Point6D unit_point) const;
    [[nodiscard]] Point6D DenormalizeFromCenter(Point6D unit_point) const;

    // CostFunction cost_;
    std::variant<CostFunction, CppCost> cost_;
    // Plan 010 U11 (R12): optional batch cost-query sibling. Empty when unset
    // (default), in which case the per-point serial path is used unchanged.
    BatchCostFunction batch_cost_;

    Point6D range_;
    Point6D starting_point_;
    bool valid_range_ = false;

    unsigned int budget_ = 0;
    unsigned int cost_function_calls_ = 0;
    unsigned int call_offset_ = 0;
    // Plan 008 U8: the per-stage optimizer-variant slot. Default-constructed
    // to the bit-identical classic search (non-default fields are fail-fast
    // stubs in this unit).
    Options options_;
    // Zero-initialized at construction (guard-precondition lesson: a guard's
    // precondition must itself be initialized -- see
    // docs/solutions/logic-errors/
    // jtml-heatmap-guard-allocator-preconditions-2026-08-12.md).
    unsigned int non_finite_count_ = 0;

    IterationCallback iteration_callback_;
    ImprovementCallback improvement_callback_;

    DirectDataStorage data_;
    std::vector<int> potentially_optimal_col_ids_;
    std::vector<HyperBox6D> potentially_optimal_hyperboxes_;

    bool error_occurrred_ = false;
    bool stop_requested_ = false;

    double current_optimum_value_ = 0.0;
    Point6D current_optimum_location_;
};
