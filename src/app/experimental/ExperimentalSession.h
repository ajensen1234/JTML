// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U4 (R3): the app-owned dataset — frames / models / LocationStorage /
// calibration plus the SessionState mirror (the widgets SyncSessionState()
// tail, reproduced headlessly). Owned by AppBridge (the hub, per the plan's
// bridge decomposition); StudyBridge mutates it through SessionController;
// the later adapters (OptimizerBridge U6, MlBridge U7, PoseBridge U8) read
// it. Plain data, no Qt event loop, no render binding — unit-testable
// headless.

#pragma once

#include <vector>

#include "compute/frame.h"
#include "domain/session_state.h"
#include "services/calibration.h"
#include "services/location_storage.h"
#include "services/model.h"

struct ExperimentalSession {
    std::vector<Frame> loaded_frames;
    std::vector<Model> loaded_models;
    LocationStorage model_locations;
    Calibration calibration_file;
    bool calibrated_for_monoplane_viewport = false;
    bool calibrated_for_biplane_viewport = false;
    jta::SessionState session_state;

    // Dataset replace (plan 005 U4 review fix): wipe the loaded data. The
    // calibration is deliberately KEPT — one-use per session (widgets
    // parity: the load-calibration button disables after a successful
    // load); a new calibration needs an app relaunch (dev tool).
    void ClearDataset() {
        loaded_frames.clear();
        loaded_models.clear();
        model_locations = LocationStorage();
        session_state = jta::SessionState();
    }
};
