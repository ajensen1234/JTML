// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel PBT for the EdgeProcessor (plan 004 U5, R13/R15). PBT complements the
// deterministic cases in edge_processor_test.cpp by locking the invariants the
// edge slots depend on:
//   - the dilation-constant decision table (clamp negatives to 0; the
//     DIRECT_MAHFOUZ override to 3 wins over everything),
//   - applying params then reset (the default-params application) restores the
//     original edge state,
//   - the same params applied to N frames built from the same source image
//     yield identical per-frame images (deterministic pipeline),
//   - apply-all equals per-frame single application,
//   - re-applying the same params is idempotent (determinism).
//
// Headless: compiles the processor + the pure-OpenCV Frame twin
// (test/unit/frame_headless.cpp) against the same header; the fixture is the
// synthetic 64x64 PNG written to the temp dir (shared with the deterministic
// twin).

#include <filesystem>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <hegel/hegel.h>

#include <catch2/catch_test_macros.hpp>

#include "compute/frame.h"
#include "domain/settings_constants.h"
#include "services/edge_processor.h"

namespace gs = hegel::generators;

namespace {

std::string FixturePath() {
    static const std::string path =
        (std::filesystem::temp_directory_path() /
         "jtml_edge_processor_fixture.png")
            .string();
    static const bool written = [&]() {
        cv::Mat img(64, 64, CV_8UC1, cv::Scalar(255));
        cv::rectangle(
            img, cv::Rect(16, 16, 32, 32), cv::Scalar(0), cv::FILLED);
        return cv::imwrite(path, img);
    }();
    REQUIRE(written);
    return path;
}

bool SameMat(const cv::Mat& a, const cv::Mat& b) {
    return a.size() == b.size() && a.type() == b.type() &&
           cv::countNonZero(a != b) == 0;
}

auto Aperture() { return gs::sampled_from<int>({3, 5, 7}); }
auto Threshold() {
    return gs::integers<int>({.min_value = 0, .max_value = 255});
}
auto Dilation() {
    return gs::integers<int>({.min_value = -3, .max_value = 10});
}
auto CostFunctionName() {
    return gs::sampled_from<std::string>(
        {"DIRECT_DILATION", "DIRECT_DILATION_T1", "DIRECT_DILATION_SAME_Z",
         "DIRECT_DILATION_POLE_CONSTRAINT", "DIRECT_MAHFOUZ"});
}

jta::EdgeProcessingParams DrawParams(hegel::TestCase& tc) {
    return jta::EdgeProcessingParams{
        tc.draw("aperture", Aperture()),
        tc.draw("low_threshold", Threshold()),
        tc.draw("high_threshold", Threshold()),
        tc.draw("dilation", Dilation()),
        tc.draw("cost_function_name", CostFunctionName())};
}

}  // namespace

TEST_CASE(
    "edge_processor[PBT]: dilation-constant decision table (clamp + Mahfouz "
    "override) holds for every input",
    "[edge_processor][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            const int raw = tc.draw("raw_dilation", Dilation());
            const std::string name = tc.draw("name", CostFunctionName());
            const int resolved =
                jta::EdgeProcessor::ResolveDilation(raw, name);

            if (name == "DIRECT_MAHFOUZ") {
                /*The override wins over everything (byte-identical to the
                 * UpdateDilationFrames Mahfouz branch).*/
                REQUIRE(resolved == 3);
            } else if (raw < 0) {
                REQUIRE(resolved == 0);
            } else {
                REQUIRE(resolved == raw);
            }
        },
        hegel::Settings{.test_cases = 400});
}

TEST_CASE(
    "edge_processor[PBT]: applying params then reset restores the original "
    "edge state",
    "[edge_processor][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            /*A frame built with the default params: its edge state IS the
             * post-reset state (the reset slot cascades into applications of
             * exactly the default params).*/
            Frame f(FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
            const cv::Mat original_edge = f.GetEdgeImage().clone();
            const cv::Mat original_dilated = f.GetDilationImage().clone();

            jta::EdgeProcessingParams params = DrawParams(tc);
            jta::EdgeProcessor::ApplyToFrame(params, f);

            /*Reset: the default-params application.*/
            jta::EdgeProcessor::ApplyToFrame(
                jta::EdgeProcessingParams{
                    APERTURE, LOW_THRESH, HIGH_THRESH, 0, "DIRECT_DILATION"},
                f);
            REQUIRE(SameMat(original_edge, f.GetEdgeImage()));
            REQUIRE(SameMat(original_dilated, f.GetDilationImage()));
            REQUIRE(f.GetAperture() == APERTURE);
            REQUIRE(f.GetLowThreshold() == LOW_THRESH);
            REQUIRE(f.GetHighThreshold() == HIGH_THRESH);
        },
        hegel::Settings{.test_cases = 300});
}

TEST_CASE(
    "edge_processor[PBT]: re-applying the same params is idempotent "
    "(deterministic pipeline)",
    "[edge_processor][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            Frame f(FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
            jta::EdgeProcessingParams params = DrawParams(tc);
            jta::EdgeProcessor::ApplyToFrame(params, f);
            const cv::Mat edge_once = f.GetEdgeImage().clone();
            const cv::Mat dilated_once = f.GetDilationImage().clone();

            jta::EdgeProcessor::ApplyToFrame(params, f);
            REQUIRE(SameMat(edge_once, f.GetEdgeImage()));
            REQUIRE(SameMat(dilated_once, f.GetDilationImage()));
        },
        hegel::Settings{.test_cases = 300});
}

TEST_CASE(
    "edge_processor[PBT]: the same params applied to N frames built from the "
    "same source image yield identical per-frame images",
    "[edge_processor][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            jta::EdgeProcessingParams params = DrawParams(tc);
            const int n = 3;
            std::vector<Frame> frames;
            for (int i = 0; i < n; ++i) {
                frames.emplace_back(
                    FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
            }
            for (Frame& f : frames) {
                jta::EdgeProcessor::ApplyToFrame(params, f);
            }
            for (int i = 1; i < n; ++i) {
                REQUIRE(
                    SameMat(frames[0].GetEdgeImage(), frames[i].GetEdgeImage()));
                REQUIRE(SameMat(
                    frames[0].GetDilationImage(),
                    frames[i].GetDilationImage()));
            }
        },
        hegel::Settings{.test_cases = 200});
}

TEST_CASE(
    "edge_processor[PBT]: apply-all equals per-frame single application",
    "[edge_processor][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            jta::EdgeProcessingParams params = DrawParams(tc);
            const int n = 3;
            std::vector<Frame> batched;
            std::vector<Frame> per_frame;
            for (int i = 0; i < n; ++i) {
                batched.emplace_back(
                    FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
                per_frame.emplace_back(
                    FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
            }

            jta::EdgeProcessor::ApplyToFrames(params, batched);
            for (Frame& f : per_frame) {
                jta::EdgeProcessor::ApplyToFrame(params, f);
            }

            for (int i = 0; i < n; ++i) {
                REQUIRE(SameMat(
                    batched[i].GetEdgeImage(), per_frame[i].GetEdgeImage()));
                REQUIRE(SameMat(
                    batched[i].GetDilationImage(),
                    per_frame[i].GetDilationImage()));
            }
        },
        hegel::Settings{.test_cases = 200});
}
