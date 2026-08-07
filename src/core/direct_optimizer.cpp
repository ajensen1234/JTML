// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "core/direct_optimizer.h"

#include <cfloat>
#include <climits>

namespace {
// The unit-cube center used to seed the search (all DOFs at 0.5).
Point6D UnitCenter() {
    return Point6D(.5, .5, .5, .5, .5, .5);
}
}  // namespace

DirectOptimizer::DirectOptimizer(CostFunction cost, Point6D range,
                                 Point6D starting_point, unsigned int budget)
    : cost_(std::move(cost)),
      range_(range),
      starting_point_(starting_point),
      budget_(budget) {
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

    current_optimum_value_ = EvaluateCostFunction(UnitCenter());
    current_optimum_location_ = starting_point_;
    data_ = DirectDataStorage(current_optimum_value_);

    while (cost_function_calls_ < budget_ && !stop_requested_) {
        ConvexHull();
        TrisectPotentiallyOptimal();

        /*Safety Break...Should Never Happen, but mirrors the original guard.*/
        if (potentially_optimal_col_ids_.size() == 0) {
            error_occurrred_ = true;
            break;
        }
        if (error_occurrred_) break;
    }

    return !error_occurrred_;
}

unsigned int DirectOptimizer::GetCostFunctionCalls() const {
    return cost_function_calls_;
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
                slope = (right_value -
                         data_.GetMinimumHyperboxValue(left_index)) /
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
    for (int i = 0; i < potentially_optimal_hyperboxes_.size(); i++) {
        Point6D denormalized_sides =
            DenormalizeRange(potentially_optimal_hyperboxes_[i].GetSides());
        Direction largest_direction =
            denormalized_sides.GetLargestDirection();

        /*Unchanged-center hyperbox.*/
        auto original_center_hyperbox_ = new HyperBox6D();
        *original_center_hyperbox_ = potentially_optimal_hyperboxes_[i];
        original_center_hyperbox_->TrisectSide(largest_direction);
        data_.AddHyperBox(original_center_hyperbox_);

        Point6D updated_center;

        /*Changed-center hyperbox A.*/
        auto changed_hyperbox_a = new HyperBox6D();
        *changed_hyperbox_a = potentially_optimal_hyperboxes_[i];
        changed_hyperbox_a->TrisectSide(largest_direction);
        updated_center = changed_hyperbox_a->GetCenter();
        updated_center.UpdateDirection(
            largest_direction,
            updated_center.GetDirection(largest_direction) +
                changed_hyperbox_a->GetSides().GetDirection(
                    largest_direction));
        changed_hyperbox_a->SetCenter(updated_center);
        changed_hyperbox_a->value_ =
            EvaluateCostFunction(changed_hyperbox_a->GetCenter());
        data_.AddHyperBox(changed_hyperbox_a);

        /*Changed-center hyperbox B.*/
        auto changed_hyperbox_b = new HyperBox6D();
        *changed_hyperbox_b = potentially_optimal_hyperboxes_[i];
        changed_hyperbox_b->TrisectSide(largest_direction);
        updated_center = changed_hyperbox_b->GetCenter();
        updated_center.UpdateDirection(
            largest_direction,
            updated_center.GetDirection(largest_direction) -
                changed_hyperbox_b->GetSides().GetDirection(
                    largest_direction));
        changed_hyperbox_b->SetCenter(updated_center);
        changed_hyperbox_b->value_ =
            EvaluateCostFunction(changed_hyperbox_b->GetCenter());
        data_.AddHyperBox(changed_hyperbox_b);
    }
}

double DirectOptimizer::EvaluateCostFunction(Point6D unit_point) {
    Point6D denormalized_point = DenormalizeFromCenter(unit_point);
    double result = cost_(denormalized_point);

    cost_function_calls_++;

    /*Store optimum (mirrors the original, without the GUI signal).*/
    if (result < current_optimum_value_) {
        current_optimum_value_ = result;
        current_optimum_location_ = denormalized_point;
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
