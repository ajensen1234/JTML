// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Pose / kinematics persistence round-trip tests (plan U7, R2/R8/R9).
//
// These exercise pose_file_io as a pure service: write->read must round-trip
// values, and the reader must parse the REAL fixture files (test/golden
// fem_golden.jts, JT_EULER_312) to the known-good poses in baseline.json. The
// format logic is NOT re-derived here -- the expected poses come from the
// committed golden baseline (independent ground truth).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <sstream>

#include "domain/pose_file_io.h"

using Catch::Approx;
using jta::pose_file::LoadResult;
using jta::pose_file::ReadKinematics;
using jta::pose_file::ReadPose;
using jta::pose_file::WriteKinematics;
using jta::pose_file::WritePose;

Point6D MakeSample() {
    return Point6D(18.52191, 19.69514, -1027.713, -7.419319, -0.2587041,
                   -26.69708);
}

void RequirePointNear(const Point6D& a, const Point6D& b) {
    REQUIRE(a.x == Approx(b.x).margin(1e-6));
    REQUIRE(a.y == Approx(b.y).margin(1e-6));
    REQUIRE(a.z == Approx(b.z).margin(1e-6));
    REQUIRE(a.xa == Approx(b.xa).margin(1e-6));
    REQUIRE(a.ya == Approx(b.ya).margin(1e-6));
    REQUIRE(a.za == Approx(b.za).margin(1e-6));
}

TEST_CASE("pose round-trips through JTA_EULER_POSE", "[pose_file]") {
    Point6D src = MakeSample();
    std::ostringstream os;
    REQUIRE(WritePose(os, src));
    std::istringstream is(os.str());
    Point6D parsed;
    LoadResult res = ReadPose(is, parsed);
    REQUIRE(res.ok);
    REQUIRE(res.kind == jta::pose_file::FileKind::Pose);
    RequirePointNear(parsed, src);
}

TEST_CASE("raw single-row (.jtp style) pose parses", "[pose_file]") {
    // .jtp files store a single data row with no header.
    std::istringstream is("18.52191,\t19.69514,\t-1027.713,\t-26.69708,\t-7.419319,\t-0.2587041");
    Point6D parsed;
    LoadResult res = ReadPose(is, parsed);
    REQUIRE(res.ok);
    RequirePointNear(parsed, MakeSample());
}

TEST_CASE("kinematics round-trip multiple frames", "[pose_file]") {
    Point6D a = MakeSample();
    Point6D b(16.4709, 16.4248, -1028.69, -7.678545, 0.3264978, -24.12223);
    std::vector<Point6D> src{a, b};
    std::ostringstream os;
    REQUIRE(WriteKinematics(os, src));
    std::istringstream is(os.str());
    std::vector<std::optional<Point6D>> parsed;
    LoadResult res = ReadKinematics(is, parsed);
    REQUIRE(res.ok);
    REQUIRE(res.kind == jta::pose_file::FileKind::Kinematics);
    REQUIRE(parsed.size() == 2);
    REQUIRE(parsed[0].has_value());
    REQUIRE(parsed[1].has_value());
    RequirePointNear(*parsed[0], a);
    RequirePointNear(*parsed[1], b);
}

TEST_CASE("kinematics NOT_OPTIMIZED rows keep frame alignment",
          "[pose_file]") {
    // A mid-file NOT_OPTIMIZED row must leave that frame unset WITHOUT shifting
    // the following frames (regression guard for the ReadKinematics re-index).
    std::istringstream is(
        "JTA_EULER_KINEMATICS\n"
        "X_TRAN\tY_TRAN\tZ_TRAN\tZ_ROT\tX_ROT\tY_ROT\n"
        "18.52191,\t19.69514,\t-1027.713,\t-26.69708,\t-7.419319,\t-0.2587041,\n"
        "NOT_OPTIMIZED,\t0,\t0,\t0,\t0,\t0,\n"
        "16.4709,\t16.4248,\t-1028.69,\t-24.12223,\t-7.678545,\t0.3264978,\n");
    std::vector<std::optional<Point6D>> parsed;
    LoadResult res = ReadKinematics(is, parsed);
    REQUIRE(res.ok);
    REQUIRE(res.not_optimized);
    REQUIRE(parsed.size() == 3);
    REQUIRE(parsed[0].has_value());   // frame 0 valid
    REQUIRE_FALSE(parsed[1].has_value());  // frame 1 NOT_OPTIMIZED -> unset
    REQUIRE(parsed[2].has_value());   // frame 2 STILL at index 2 (not shifted)
    RequirePointNear(*parsed[0], MakeSample());
    RequirePointNear(*parsed[2], Point6D(16.4709, 16.4248, -1028.69,
                                         -7.678545, 0.3264978, -24.12223));
}

TEST_CASE("kinematics all-NOT_OPTIMIZED yields a clean not-ok",
          "[pose_file]") {
    std::istringstream is(
        "JTA_EULER_KINEMATICS\nX_TRAN Y_TRAN Z_TRAN Z_ROT X_ROT Y_ROT\n"
        "NOT_OPTIMIZED 0 0 0 0 0\nNOT_OPTIMIZED 0 0 0 0 0\n");
    std::vector<std::optional<Point6D>> parsed;
    LoadResult res = ReadKinematics(is, parsed);
    REQUIRE(res.not_optimized);
    REQUIRE_FALSE(res.ok);  // no valid poses anywhere
}

TEST_CASE("kinematics malformed/few-column rows are skipped in place",
          "[pose_file]") {
    std::istringstream is(
        "JTA_EULER_KINEMATICS\nX_TRAN Y_TRAN Z_TRAN Z_ROT X_ROT Y_ROT\n"
        "18.5219 19.6951 -1027.713 -26.697 -7.4193 -0.2587\n"
        "garbage row that is not a pose\n"
        "16.4709 16.4248 -1028.69 -24.122 -7.6785 0.3264\n");
    std::vector<std::optional<Point6D>> parsed;
    LoadResult res = ReadKinematics(is, parsed);
    REQUIRE(res.ok);
    REQUIRE(parsed.size() == 3);
    REQUIRE(parsed[0].has_value());
    REQUIRE_FALSE(parsed[1].has_value());  // malformed row -> frame unset, no shift
    REQUIRE(parsed[2].has_value());
}

TEST_CASE("real golden fixture (JT_EULER_312) parses to baseline poses",
          "[pose_file][fixture]") {
    // test/golden/fem_golden.jts is the committed known-good 3-frame femur
    // kinematics file (JT_EULER_312, space-aligned columns). Expected values
    // come from test/golden/baseline.json "expected_pose_per_frame", NOT
    // derived from the code under test.
    std::ifstream f("test/golden/fem_golden.jts");
    REQUIRE(f.good());
    std::vector<std::optional<Point6D>> parsed;
    LoadResult res = ReadKinematics(f, parsed);
    REQUIRE(res.ok);
    REQUIRE(parsed.size() == 3);
    REQUIRE(parsed[0].has_value());
    REQUIRE(parsed[1].has_value());
    REQUIRE(parsed[2].has_value());

    // frame 0
    RequirePointNear(*parsed[0],
                     Point6D(18.52191, 19.69514, -1027.713, -7.419319,
                             -0.2587041, -26.69708));
    // frame 1
    RequirePointNear(*parsed[1],
                     Point6D(19.01747, 20.15555, -1026.732, -7.56846,
                             -0.2358893, -27.37827));
    // frame 2
    RequirePointNear(*parsed[2],
                     Point6D(16.4709, 16.4248, -1028.69, -7.678545, 0.3264978,
                             -24.12223));
}

TEST_CASE("malformed or empty pose file yields a clean failure", "[pose_file]") {
    {   // Empty input -> not ok, no crash.
        std::istringstream is("");
        Point6D p;
        REQUIRE_FALSE(ReadPose(is, p).ok);
    }
    {   // Garbage that never tokenizes to 6 columns -> not ok.
        std::istringstream is("hello world this is not a pose");
        Point6D p;
        REQUIRE_FALSE(ReadPose(is, p).ok);
    }
    {   // NOT_OPTIMIZED row is surfaced, not parsed as a pose.
        std::istringstream is(
            "JTA_EULER_POSE\nX_TRAN\t\tY_TRAN\t\tZ_TRAN\t\tZ_ROT\t\tX_ROT\t\tY_ROT\n"
            "NOT_OPTIMIZED,\t0,\t0,\t0,\t0,\t0,\n");
        Point6D p;
        LoadResult res = ReadPose(is, p);
        REQUIRE(res.not_optimized);
        REQUIRE_FALSE(res.ok);
    }
    {   // Unknown header (not pose/kinematics) -> not ok.
        std::istringstream is("SOME_OTHER_FORMAT\n1 2 3 4 5 6\n");
        Point6D p;
        REQUIRE_FALSE(ReadPose(is, p).ok);
    }
    {   // 6 columns but a non-numeric field (stod throws) -> not ok, no crash.
        std::istringstream is("1 2 3 4 five 6");
        Point6D p;
        REQUIRE_FALSE(ReadPose(is, p).ok);
    }
}

TEST_CASE("pose_file path wrappers fail cleanly on unopenable paths",
          "[pose_file]") {
    Point6D p;
    REQUIRE_FALSE(jta::pose_file::ReadPoseFile(
        "/nonexistent/does/not/exist.ptp", p).ok);
    std::vector<std::optional<Point6D>> kin;
    REQUIRE_FALSE(jta::pose_file::ReadKinematicsFile(
        "/nonexistent/does/not/exist.jts", kin).ok);
}
