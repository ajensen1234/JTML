/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*SaveLastPoseToStorage (plan 006 U3): see include/services/save_last_pose.h
 * for the call-site table and the convert-rule contract. Relocated verbatim
 * from MainScreen::SaveLastPose (mainscreen.cpp:4101-4129) — R13 signature
 * adaptation: the vw reads become pose_source, the camera radio becomes
 * camera_is_a. The behavior is pinned by the call-site table test.*/

#include "services/save_last_pose.h"

namespace jta {

int SaveLastPoseToStorage(
    int frame,
    const std::vector<int>& model_rows,
    const std::function<Point6D(int model_row)>& pose_source,
    bool camera_is_a,
    SavePoseConvertRule convert_rule,
    Calibration& calibration,
    LocationStorage& storage) {
    /*No-op without a previous selection — the widgets guard
     * (previous_model_indices_.size() > 0 && previous_frame_index_ != -1).*/
    if (model_rows.empty() || frame < 0) {
        return 0;
    }

    int written = 0;
    for (const int row : model_rows) {
        Point6D pose = pose_source(row);

        /*All-zero = "no valid pose" sentinel (out-of-range source rows — the
         * QML mirror's old row-range guard): skip without writing a bogus
         * pose into storage and without corrupting the other rows.*/
        if (pose.x == 0.0 && pose.y == 0.0 && pose.z == 0.0 && pose.xa == 0.0 &&
            pose.ya == 0.0 && pose.za == 0.0) {
            continue;
        }

        /*Pinned convert rule (H4): each call site's coordinate-frame rule.
         * ConvertBToA always converts (identity for monoplane); the widgets
         * canonical row converts iff camera B is the active camera.*/
        const bool convert =
            convert_rule == SavePoseConvertRule::ConvertBToA ||
            (convert_rule == SavePoseConvertRule::ConvertWhenCameraB &&
             !camera_is_a);
        if (convert) {
            pose = calibration.convert_Pose_B_to_Pose_A(pose);
        }

        storage.SavePose(frame, row, pose);
        ++written;
    }
    return written;
}

} // namespace jta
