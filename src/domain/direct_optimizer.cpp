// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "domain/direct_optimizer.h"

#include <cfloat>
#include <climits>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

namespace {
// The unit-cube center used to seed the search (all DOFs at 0.5).
Point6D UnitCenter() {
    return Point6D(.5, .5, .5, .5, .5, .5);
}

// Plan 008 U8 (Cut C): non-default Options fields are FAIL-FAST STUBS in this
// unit (review-resolved scope boundary). Every field below maps line-by-line
// onto today's code, so the default path needs NO divergence branch -- the
// search code is byte-identical to the pre-Options code, which is the
// bit-identical-defaults proof (R13). The divergence branches (each guarded
// on "different from default") land with the algorithm plan (R9).
void ValidateOptions(const DirectOptimizer::Options& opts) {
    if (opts.selection != DirectOptimizer::Options::SelectionMode::Original) {
        throw std::invalid_argument(
            "DirectOptimizer::Options: selection != Original is a plan-008 "
            "fail-fast stub; variant semantics land with the algorithm plan");
    }
    if (opts.epsilon != 0.0) {
        throw std::invalid_argument(
            "DirectOptimizer::Options: epsilon != 0.0 is a plan-008 fail-fast "
            "stub; the epsilon post-filter lands with the algorithm plan");
    }
    if (opts.delta_limit) {
        throw std::invalid_argument(
            "DirectOptimizer::Options: delta_limit is a plan-008 fail-fast "
            "stub; the delta-limit refinement lands with the algorithm plan");
    }
    if (opts.size_measure != DirectOptimizer::Options::SizeMeasure::L2) {
        throw std::invalid_argument(
            "DirectOptimizer::Options: size_measure != L2 is a plan-008 "
            "fail-fast stub; alternative size measures land with the "
            "algorithm plan");
    }
    if (opts.split_rule != DirectOptimizer::Options::SplitRule::OneSide) {
        throw std::invalid_argument(
            "DirectOptimizer::Options: split_rule != OneSide is a plan-008 "
            "fail-fast stub; alternative partitioning lands with the "
            "algorithm plan");
    }
    if (opts.ties != DirectOptimizer::Options::TieSelection::All) {
        throw std::invalid_argument(
            "DirectOptimizer::Options: ties != All is a plan-008 fail-fast "
            "stub; tie-selection variants land with the algorithm plan");
    }
    if (opts.hidden_constraints) {
        throw std::invalid_argument(
            "DirectOptimizer::Options: hidden_constraints is a plan-008 "
            "fail-fast stub; the GLh surrogate lands with the algorithm plan");
    }
    if (opts.globally_biased) {
        throw std::invalid_argument(
            "DirectOptimizer::Options: globally_biased is a plan-008 "
            "fail-fast stub; the gb phase switch lands with the algorithm "
            "plan");
    }
}
} // namespace

DirectOptimizer::DirectOptimizer(
    CostFunction cost,
    Point6D range,
    Point6D starting_point,
    unsigned int budget,
    Options options) :
    cost_(std::move(cost)), range_(range), starting_point_(starting_point),
    budget_(budget), options_(std::move(options)) {
    // Fail fast at construction: a non-default Options field is a plan-008
    // stub, not a silent behavior change (guarded divergence -- the defaults
    // reproduce today's search bit-identically by construction).
    ValidateOptions(options_);

    // Mirror OptimizerManager::SetSearchRange: a zero (or negative-total) range
    // marks the search as invalid.
    if (range.x + range.y + range.z + range.xa + range.ya + range.za > 0) {
        valid_range_ = true;
    }
}

bool DirectOptimizer::Run() {
    // Mirror the per-stage body of OptimizerManager::Optimize() for a single
    // stage: seed with the unit center, then iterate ConvexHull + Trisect until
    // the (cumulative) budget is exhausted, a stop is requested, or an error.
    if (!valid_range_) {
        error_occurrred_ = true;
        return false;
    }

    std::optional<double> seed_result = EvaluateCostFunction(UnitCenter());
    // A non-finite seed eval leaves the optimum at a finite "no finite
    // optimum yet" sentinel (DBL_MAX): every later finite eval is an
    // improvement and wins, and the seed box in the storage stays finite by
    // construction (never store a non-finite result -- plan 008 U3).
    current_optimum_value_ = seed_result.has_value() ? *seed_result : DBL_MAX;
    current_optimum_location_ = starting_point_;
    data_ = DirectDataStorage(current_optimum_value_);

    // The loop guard uses the effective (cumulative) call count, so a stage
    // beginning part-way through the running counter is bounded by the same
    // cumulative budget as the original Optimize(): with call_offset_ == 0 the
    // guard is identically (calls < budget), preserving existing behavior.
    while ((cost_function_calls_ + call_offset_) < budget_ &&
           !stop_requested_) {
        ConvexHull();
        TrisectPotentiallyOptimal();

        /*Safety Break...Should Never Happen, but mirrors the original guard.*/
        if (potentially_optimal_col_ids_.size() == 0) {
            error_occurrred_ = true;
            break;
        }
        if (error_occurrred_) break;

        if (iteration_callback_) iteration_callback_();
    }

    return !error_occurrred_;
}

unsigned int DirectOptimizer::GetCostFunctionCalls() const {
    return cost_function_calls_ + call_offset_;
}

Point6D DirectOptimizer::GetOptimumLocation() const {
    return current_optimum_location_;
}

double DirectOptimizer::GetOptimumValue() const {
    return current_optimum_value_;
}

void DirectOptimizer::Stop() {
    stop_requested_ = true;
}

void DirectOptimizer::SetCallOffset(unsigned int offset) {
    call_offset_ = offset;
}

void DirectOptimizer::SetIterationCallback(IterationCallback cb) {
    iteration_callback_ = std::move(cb);
}

void DirectOptimizer::SetImprovementCallback(ImprovementCallback cb) {
    improvement_callback_ = std::move(cb);
}

void DirectOptimizer::SetBatchCost(BatchCostFunction cb) {
    batch_cost_ = std::move(cb);
}

void DirectOptimizer::ConvexHull() {
    /*Reset Potentially Optimal Vector*/
    potentially_optimal_col_ids_.clear();

    /*Jarvis's March (Gift Wrapping): if only one column add index 0. A small
     * epsilon is added to the smallest-sized column (as in the original).*/
    if (data_.GetNumberColumns() == 1) {
        potentially_optimal_col_ids_.push_back(0);
    } else if (data_.GetNumberColumns() > 1) {
        int right_index = data_.GetNumberColumns() - 1;
        int left_index = right_index - 1;
        unsigned potentially_optimal_index;
        double slope, highest_slope;
        double right_value, right_size;

        /*Convex Hull from the farthest right (largest side) first.*/
        potentially_optimal_col_ids_.push_back(right_index);

        /*Gift wrapping.*/
        while (right_index > 0) {
            highest_slope = -1 * DBL_MAX;
            right_value = data_.GetMinimumHyperboxValue(right_index);
            right_size = data_.GetSizeStoredInColumn(right_index);
            potentially_optimal_index = left_index;
            while (left_index >= 0) {
                slope =
                    (right_value - data_.GetMinimumHyperboxValue(left_index)) /
                    (right_size - data_.GetSizeStoredInColumn(left_index));
                if (slope >= highest_slope) {
                    highest_slope = slope;
                    potentially_optimal_index = left_index;
                }
                left_index--;
            }
            /*Never go back up after flattening out.*/
            if (highest_slope >= 0) {
                potentially_optimal_col_ids_.push_back(
                    potentially_optimal_index);
                right_index = potentially_optimal_index;
                left_index = right_index - 1;
            } else {
                break;
            }
        }
    } else {
        /*If this happens the storage matrix is empty... should never happen.*/
        error_occurrred_ = true;
    }
}

void DirectOptimizer::TrisectPotentiallyOptimal() {
    /*Populate potentially-optimal hyperboxes from column ids.*/
    potentially_optimal_hyperboxes_.clear();
    for (int i = 0; i < potentially_optimal_col_ids_.size(); i++) {
        potentially_optimal_hyperboxes_.push_back(
            data_.GetMinimumHyperbox(potentially_optimal_col_ids_[i]));
    }

    /*Delete old hyperboxes.*/
    data_.DeleteHyperBoxes(potentially_optimal_col_ids_);

    /*Trisect each potentially-optimal box: split along the largest
     * denormalized side, keep one box at the original center, and move the two
     * outer boxes to the +/- shifted centers and re-evaluate them.*/
    //
    // Plan 010 U11 (R12) batch seam: when batch_cost_ is set, the
    // changed-center evals of the WHOLE iteration are collected into ONE batch
    // (packing order: per POH box the +shift let this A center then the -shift
    // B center, boxes in POH-column order), sent to the cost layer, and the
    // results replayed in input order. The storage bookkeeping (oc box stored,
    // then A, then B per box) is deferred until AFTER the batch result is
    // size-validated, so the storage order, cost_function_calls_, optimum
    // sequence, and improvement- callback order are identical to the serial
    // path (R12 / AE2).
    if (batch_cost_) {
        // Pending boxes in serial storage order: per POH box {oc, A, B}.
        // changed_index holds the slot into `batch_centers` for A/B (-1 for
        // oc).
        std::vector<HyperBox6D*> pending;
        std::vector<int> pending_changed_index;
        std::vector<Point6D> batch_centers; // denormalized, in packing order

        for (int i = 0; i < potentially_optimal_hyperboxes_.size(); i++) {
            Point6D denormalized_sides =
                DenormalizeRange(potentially_optimal_hyperboxes_[i].GetSides());
            Direction largest_direction =
                denormalized_sides.GetLargestDirection();

            /*Unchanged-center hyperbox (no eval).*/
            auto oc = new HyperBox6D();
            *oc = potentially_optimal_hyperboxes_[i];
            oc->TrisectSide(largest_direction);
            pending.push_back(oc);
            pending_changed_index.push_back(-1);

            auto make_changed = [&](int sign) -> int {
                auto box = new HyperBox6D();
                *box = potentially_optimal_hyperboxes_[i];
                box->TrisectSide(largest_direction);
                Point6D updated = box->GetCenter();
                updated.UpdateDirection(
                    largest_direction,
                    updated.GetDirection(largest_direction) +
                        sign * box->GetSides().GetDirection(largest_direction));
                box->SetCenter(updated);
                const int idx = static_cast<int>(batch_centers.size());
                batch_centers.push_back(
                    DenormalizeFromCenter(box->GetCenter()));
                pending.push_back(box);
                pending_changed_index.push_back(idx);
                return idx;
            };

            make_changed(+1); // A: +shift
            make_changed(-1); // B: -shift
        }

        /*Single batch call over the whole iteration's changed centers.*/
        const std::vector<double> results = batch_cost_(batch_centers);

        /*Fail fast on a size-mismatched result (never partially consumed, never
         * silently re-fallen-back to per-point evaluation).*/
        if (results.size() != batch_centers.size()) {
            for (auto* b : pending)
                delete b;
            throw std::invalid_argument(
                "DirectOptimizer: batch cost returned the wrong result size "
                "(contract violation, plan 010 U11)");
        }

        /*Replay bookkeeping in the serial storage order. A/B results are read
         * from `results` by their packing slot; oc boxes store with no eval.*/
        for (std::size_t p = 0; p < pending.size(); ++p) {
            HyperBox6D* box = pending[p];
            const int cidx = pending_changed_index[p];
            if (cidx < 0) {
                data_.AddHyperBox(box);
                continue;
            }
            const double result = results[static_cast<std::size_t>(cidx)];
            const Point6D denormalized_point =
                batch_centers[static_cast<std::size_t>(cidx)];

            cost_function_calls_++;
            if (!std::isfinite(result)) {
                non_finite_count_++;
                if (iteration_callback_) iteration_callback_();
                delete box;
                continue;
            }
            box->value_ = result;
            data_.AddHyperBox(box);
            if (result < current_optimum_value_) {
                current_optimum_value_ = result;
                current_optimum_location_ = denormalized_point;
                if (improvement_callback_) {
                    improvement_callback_(
                        current_optimum_location_, current_optimum_value_);
                }
            }
        }
        return;
    }

    for (int i = 0; i < potentially_optimal_hyperboxes_.size(); i++) {
        Point6D denormalized_sides =
            DenormalizeRange(potentially_optimal_hyperboxes_[i].GetSides());
        Direction largest_direction = denormalized_sides.GetLargestDirection();

        /*Unchanged-center hyperbox.*/
        auto original_center_hyperbox_ = new HyperBox6D();
        *original_center_hyperbox_ = potentially_optimal_hyperboxes_[i];
        original_center_hyperbox_->TrisectSide(largest_direction);
        data_.AddHyperBox(original_center_hyperbox_);

        Point6D updated_center;

        /*Store-or-delete a freshly evaluated changed-center box (plan 008 U3):
         * a finite eval stores the box (value_ + AddHyperBox); a non-finite
         * eval deletes it — the box is a STANDALONE heap copy not yet linked
         * into storage, so deleting is safe (a stored NaN would accumulate as
         * dead weight and could poison the column minimum once all finite
         * boxes in the column are gone).*/
        auto store_or_delete =
            [this](HyperBox6D* box, const std::optional<double>& eval) {
                if (eval.has_value()) {
                    box->value_ = *eval;
                    data_.AddHyperBox(box);
                } else {
                    delete box;
                }
            };

        /*Changed-center hyperbox A.*/
        auto changed_hyperbox_a = new HyperBox6D();
        *changed_hyperbox_a = potentially_optimal_hyperboxes_[i];
        changed_hyperbox_a->TrisectSide(largest_direction);
        updated_center = changed_hyperbox_a->GetCenter();
        updated_center.UpdateDirection(
            largest_direction,
            updated_center.GetDirection(largest_direction) +
                changed_hyperbox_a->GetSides().GetDirection(largest_direction));
        changed_hyperbox_a->SetCenter(updated_center);
        store_or_delete(
            changed_hyperbox_a,
            EvaluateCostFunction(changed_hyperbox_a->GetCenter()));

        /*Changed-center hyperbox B.*/
        auto changed_hyperbox_b = new HyperBox6D();
        *changed_hyperbox_b = potentially_optimal_hyperboxes_[i];
        changed_hyperbox_b->TrisectSide(largest_direction);
        updated_center = changed_hyperbox_b->GetCenter();
        updated_center.UpdateDirection(
            largest_direction,
            updated_center.GetDirection(largest_direction) -
                changed_hyperbox_b->GetSides().GetDirection(largest_direction));
        changed_hyperbox_b->SetCenter(updated_center);
        store_or_delete(
            changed_hyperbox_b,
            EvaluateCostFunction(changed_hyperbox_b->GetCenter()));
    }
}

unsigned int DirectOptimizer::GetNonFiniteCount() const {
    return non_finite_count_;
}

std::optional<double>
DirectOptimizer::EvaluateCostFunction(Point6D unit_point) {
    Point6D denormalized_point = DenormalizeFromCenter(unit_point);
    double result = cost_(denormalized_point);

    cost_function_calls_++;

    // Finite-check at the one shared eval chokepoint (plan 008 U3; panoptes
    // angle 02 Round 2). A non-finite (NaN/Inf) cost is INFEASIBLE: it is
    // never stored, never wins the optimum (the finite-only optimum update
    // below), and is surfaced to callers via the counter and one
    // iteration-callback fire. Behavior-neutral for all finite evals
    // (DIRECT_DILATION is provably finite -- no oracle impact).
    //
    // GLh surrogate hook (RESERVED -- documented extension point, NOT wired in
    // this unit; the algorithm plan owns the semantics): when DIRECT-GLh
    // lands, substitute
    //   result = current_optimum_value_ + ||denormalized_point -
    //   current_optimum_location_||
    // here, making the eval FINITE so it flows through the store/optimum path
    // below. Until then a non-finite eval stays infeasible.
    if (!std::isfinite(result)) {
        non_finite_count_++;
        if (iteration_callback_) {
            iteration_callback_();
        }
        return std::nullopt;
    }

    /*Store optimum (mirrors the original, without the GUI signal).*/
    if (result < current_optimum_value_) {
        current_optimum_value_ = result;
        current_optimum_location_ = denormalized_point;
        if (improvement_callback_) {
            improvement_callback_(
                current_optimum_location_, current_optimum_value_);
        }
    }

    return result;
}

Point6D DirectOptimizer::DenormalizeRange(Point6D unit_point) const {
    return Point6D(
        unit_point.x * range_.x * 2.0,
        unit_point.y * range_.y * 2.0,
        unit_point.z * range_.z * 2.0,
        unit_point.xa * range_.xa * 2.0,
        unit_point.ya * range_.ya * 2.0,
        unit_point.za * range_.za * 2.0);
}

Point6D DirectOptimizer::DenormalizeFromCenter(Point6D unit_point) const {
    return Point6D(
        starting_point_.x + (unit_point.x - 0.5) * 2 * range_.x,
        starting_point_.y + (unit_point.y - 0.5) * 2 * range_.y,
        starting_point_.z + (unit_point.z - 0.5) * 2 * range_.z,
        starting_point_.xa + (unit_point.xa - 0.5) * 2 * range_.xa,
        starting_point_.ya + (unit_point.ya - 0.5) * 2 * range_.ya,
        starting_point_.za + (unit_point.za - 0.5) * 2 * range_.za);
}
