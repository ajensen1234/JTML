/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*SaveLastPoseToStorage (plan 006 U3 / R10 part, R13, H4): the parameterized
 * save-last-pose core shared by BOTH front-ends. The four divergent call
 * sites — MainScreen::SaveLastPose (widgets), the camera-A and camera-B slot
 * inline copies (widgets, converge in U9), and
 * OptimizerBridge::saveScenePosesForCurrentSelection (QML) — all drive this
 * one function. Each of the four behaviors is correct for its source's
 * coordinate frame, so the call-site table is PINNED, not unified (the
 * conversion-divergence unification is a Deferred to Follow-Up Work cut):
 *
 *   1. widgets SaveLastPose:   previous selection, previous frame, viewer
 *      source, convert iff camera B checked (ConvertWhenCameraB).
 *   2. camera-A slot inline:   CURRENT selection, previous frame, actor-list
 *      source, ALWAYS convert B->A (ConvertBToA).
 *   3. camera-B slot inline:   current selection, previous frame, viewer
 *      source, NEVER convert (NeverConvert — the raw save is correct because
 *      the source is A-coords).
 *   4. QML mirror:             current selection, current frame, scene
 *      source, never convert (NeverConvert).
 *
 * The call-site table test (test/unit/save_last_pose_test.cpp) is the spec.
 * Relocated verbatim from MainScreen::SaveLastPose (mainscreen.cpp:4101)
 * with signature adaptation: the vw/actor/scene reads become the injected
 * pose_source functor, the camera radio becomes camera_is_a. Pure Qt/GPU-
 * free: headless-testable under Catch2.*/

#ifndef SAVE_LAST_POSE_H
#define SAVE_LAST_POSE_H

#include <functional>
#include <vector>

#include "domain/data_structures_6D.h"
#include "services/calibration.h"
#include "services/location_storage.h"

namespace jta {

/*Convert rule for the save-last-pose core (H4): each call site has a fixed,
 * correct rule for its source's coordinate frame — pin, don't unify.
 *  - NeverConvert: source poses are already in the storage frame (camera-B
 *    slot raw save; QML scene poses).
 *  - ConvertBToA: source poses are camera-B coordinates, storage is
 *    camera-A (camera-A slot inline; identity for monoplane calibrations).
 *  - ConvertWhenCameraB: convert iff camera B is the active camera —
 *    widgets SaveLastPose: camera A checked -> raw, else convert.*/
enum class SavePoseConvertRule {
    NeverConvert,
    ConvertBToA,
    ConvertWhenCameraB,
};

/*Persists the last-seen poses of `model_rows` at `frame` into `storage`.
 * `pose_source(row)` returns the live pose of model row `row` (viewer actor,
 * scene model, ...). `camera_is_a` is the camera-A radio state; consulted
 * only by ConvertWhenCameraB.
 *
 * No-op (returns 0) when model_rows is empty or frame < 0 — the widgets'
 * `previous_model_indices_.size() > 0 && previous_frame_index_ != -1`
 * guard. A row whose pose_source returns the ALL-ZERO pose is SKIPPED — the
 * "no valid pose" sentinel for out-of-range sources (the QML mirror's old
 * row-range guard, which is now expressed by the scene lambda returning
 * Point6D() for out-of-range rows); the other rows are still written.
 *
 * Returns the number of poses written.*/
int SaveLastPoseToStorage(
    int frame,
    const std::vector<int>& model_rows,
    const std::function<Point6D(int model_row)>& pose_source,
    bool camera_is_a,
    SavePoseConvertRule convert_rule,
    Calibration& calibration,
    LocationStorage& storage);

} // namespace jta

#endif /* SAVE_LAST_POSE_H */
