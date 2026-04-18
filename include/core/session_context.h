/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#ifndef SESSION_CONTEXT_H
#define SESSION_CONTEXT_H

#include "cost_functions/CostFunctionManager.h"
#include "core/calibration.h"
#include "core/frame.h"
#include "core/location_storage.h"
#include "core/model.h"
#include "core/optimizer_settings.h"

#include <memory>
#include <vector>

namespace jta_core {

/**
 * @brief Aggregate struct grouping all optimization and frame data members
 *        needed by service calls. Replaces scattered individual members.
 */
struct SessionContext {
    Calibration calibration_file_{};
    bool calibrated_for_monoplane_viewport_{false};
    bool calibrated_for_biplane_viewport_{false};
    std::vector<Frame> loaded_frames;
    std::vector<Frame> loaded_frames_B;
    std::vector<Model> loaded_models;
    LocationStorage model_locations_{};
    OptimizerSettings optimizer_settings_{};
    jta_cost_function::CostFunctionManager trunk_manager_{Stage::Trunk};
    jta_cost_function::CostFunctionManager branch_manager_{Stage::Branch};
    jta_cost_function::CostFunctionManager leaf_manager_{Stage::Leaf};
};

} // namespace jta_core

#endif /* SESSION_CONTEXT_H */
