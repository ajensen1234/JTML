// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#pragma once

#include <istream>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "domain/data_structures_6D.h"

// Pure, Qt-free pose / kinematics file IO (plan U7, R8/R9: persistence is
// extracted as a service so it can be unit-tested against real fixture files
// without a widget or event loop).
//
// Format note (faithful to the GUI's MainScreen slots): columns are stored in
// the order X, Y, Z, Z_ROT, X_ROT, Y_ROT and mapped back to Point6D as
// (x, y, z, x_rot, y_rot, z_rot). Both JTA_EULER_POSE / JTA_EULER_KINEMATICS
// and the JointTrack JT_EULER_312 headers are accepted on read; columns may be
// separated by commas OR whitespace (a strict superset of the GUI's comma-only
// parsing, which lets the real JT_EULER_312 fixtures round-trip).
namespace jta {
namespace pose_file {

enum class FileKind {
    None,        // not a recognized pose/kinematics file
    Pose,        // single-pose file (JTA_EULER_POSE, or a raw single row)
    Kinematics,  // multi-frame file (JTA_EULER_KINEMATICS, JT_EULER_312)
};

struct LoadResult {
    bool ok = false;
    bool not_optimized = false;  // a row carried NOT_OPTIMIZED and was skipped
    FileKind kind = FileKind::None;
};

// Write a single pose in the JTA_EULER_POSE format. Returns false on stream error.
bool WritePose(std::ostream& out, const Point6D& pose);

// Read a single pose from an in-memory stream (JTA_EULER_POSE header or a raw
// single-row .jtp-style file). On success out holds the parsed pose.
LoadResult ReadPose(std::istream& in, Point6D& out);

// Write per-frame kinematics in JTA_EULER_KINEMATICS format (one row per pose).
bool WriteKinematics(std::ostream& out, const std::vector<Point6D>& poses);

// Read kinematics from a JTA_EULER_KINEMATICS or JT_EULER_312 stream.
// `out` is POSITION-PRESERVING: out[j] holds the pose for FRAME j (the (j+2)th
// data line), and NOT_OPTIMIZED / malformed rows yield std::nullopt for that
// frame so subsequent frames stay aligned (the original loader keyed frames by
// line index, not by a compacted count).
LoadResult ReadKinematics(std::istream& in,
                          std::vector<std::optional<Point6D>>& out);

// File-path convenience wrappers.
bool WritePoseFile(const std::string& path, const Point6D& pose);
LoadResult ReadPoseFile(const std::string& path, Point6D& out);
bool WriteKinematicsFile(const std::string& path,
                         const std::vector<Point6D>& poses);
LoadResult ReadKinematicsFile(const std::string& path,
                              std::vector<std::optional<Point6D>>& out);

}  // namespace pose_file
}  // namespace jta
