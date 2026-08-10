// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Deterministic tests for the EdgeProcessor (plan 004 U5, R7/R13/R14/R15).
//
// The processor is headless by construction: it takes (params, Frame&) and
// calls only the existing Frame methods. The production Frame is compute-layer
// (src/compute/frame.cu -- nvcc/torch/CUDA-linked), so these tests compile the
// pure-OpenCV Frame twin from test/unit/frame_headless.cpp against the same
// header; the fixture is a synthetic 64x64 PNG (white field, centered black
// 32x32 square) written to the temp dir.
//
// Pinned, not normalized: zero/negative/out-of-range params go straight to the
// same OpenCV calls the slots made before U5 (the processor never clamps
// aperture/thresholds; only the dilation-constant decision clamps, exactly as
// UpdateDilationFrames did).

#include <filesystem>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <catch2/catch_test_macros.hpp>

#include "compute/frame.h"
#include "domain/settings_constants.h"
#include "services/edge_processor.h"

namespace {

/*Fixture: 64x64 white field with a centered 32x32 black square (rows/cols
 * 16..47). Written once to the temp dir; the Frame constructor mirrors the
 * production flip+imread.*/
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

int NonZero(const cv::Mat& m) {
    return cv::countNonZero(m);
}

bool IsBinary(const cv::Mat& m) {
    cv::Mat neither_zero_nor_255;
    cv::bitwise_and((m != 0), (m != 255), neither_zero_nor_255);
    return cv::countNonZero(neither_zero_nor_255) == 0;
}

jta::EdgeProcessingParams DefaultParams() {
    return jta::EdgeProcessingParams{
        APERTURE, LOW_THRESH, HIGH_THRESH, 0, "DIRECT_DILATION"};
}

}  // namespace

TEST_CASE(
    "edge_processor: applying params produces the expected edge image on a "
    "synthetic fixture",
    "[edge_processor]") {
    Frame f(FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
    jta::EdgeProcessor::ApplyToFrame(
        jta::EdgeProcessingParams{3, 40, 120, 0, "DIRECT_DILATION"}, f);

    const cv::Mat edge = f.GetEdgeImage();
    REQUIRE(edge.size() == cv::Size(64, 64));
    REQUIRE(edge.type() == CV_8UC1);
    REQUIRE(IsBinary(edge));
    REQUIRE(NonZero(edge) > 0);

    /*The square's boundary ring (rows/cols 16..47) is the only structure: no
     * edges in the flat white exterior or the flat black interior.*/
    const int square_min = 16;
    const int square_max = 47;
    for (int r = 0; r < 64; ++r) {
        for (int c = 0; c < 64; ++c) {
            if (edge.at<uchar>(r, c) == 0) {
                continue;
            }
            /*2-px tolerance around the step edge for the Sobel + hysteresis
             * spread.*/
            REQUIRE(r >= square_min - 2);
            REQUIRE(r <= square_max + 2);
            REQUIRE(c >= square_min - 2);
            REQUIRE(c <= square_max + 2);
        }
    }
    /*Deep interior (4 px inside the square) and deep exterior are clean.*/
    cv::Rect interior(20, 20, 24, 24);
    cv::Rect exterior(0, 0, 10, 10);
    REQUIRE(NonZero(edge(interior)) == 0);
    REQUIRE(NonZero(edge(exterior)) == 0);

    /*Deterministic: re-applying the same params leaves the image unchanged.*/
    const cv::Mat before = edge.clone();
    jta::EdgeProcessor::ApplyToFrame(
        jta::EdgeProcessingParams{3, 40, 120, 0, "DIRECT_DILATION"}, f);
    REQUIRE(SameMat(before, f.GetEdgeImage()));
}

TEST_CASE(
    "edge_processor: stored constants track the applied params (the "
    "frame-sourced mechanism)",
    "[edge_processor]") {
    Frame f(FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
    jta::EdgeProcessor::ApplyToFrame(
        jta::EdgeProcessingParams{7, 33, 150, 2, "DIRECT_DILATION"}, f);
    REQUIRE(f.GetAperture() == 7);
    REQUIRE(f.GetLowThreshold() == 33);
    REQUIRE(f.GetHighThreshold() == 150);
}

TEST_CASE(
    "edge_processor: dilation grows the edge image by the resolved dilation",
    "[edge_processor]") {
    Frame f1(FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
    jta::EdgeProcessor::ApplyToFrame(
        jta::EdgeProcessingParams{3, 40, 120, 1, "DIRECT_DILATION"}, f1);
    const cv::Mat edge = f1.GetEdgeImage();
    const cv::Mat dilated1 = f1.GetDilationImage();

    /*Dilation is extensive: every edge pixel survives, and the ring grows.*/
    REQUIRE(NonZero(dilated1) >= NonZero(edge));
    REQUIRE(NonZero(dilated1) > NonZero(edge));

    Frame f2(FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
    jta::EdgeProcessor::ApplyToFrame(
        jta::EdgeProcessingParams{3, 40, 120, 2, "DIRECT_DILATION"}, f2);
    REQUIRE(NonZero(f2.GetDilationImage()) > NonZero(dilated1));

    /*Dilation 0 (the "no Dilation parameter" case) is pinned as today: the
     * same OpenCV dilate call the slots always made.*/
    Frame f0(FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
    jta::EdgeProcessor::ApplyToFrame(
        jta::EdgeProcessingParams{3, 40, 120, 0, "DIRECT_DILATION"}, f0);
    REQUIRE(NonZero(f0.GetDilationImage()) >= NonZero(edge));
}

TEST_CASE(
    "edge_processor: DIRECT_MAHFOUZ resolves the dilation to 3 (Mahfouz case)",
    "[edge_processor]") {
    REQUIRE(jta::EdgeProcessor::ResolveDilation(0, "DIRECT_MAHFOUZ") == 3);
    REQUIRE(jta::EdgeProcessor::ResolveDilation(9, "DIRECT_MAHFOUZ") == 3);
    REQUIRE(jta::EdgeProcessor::ResolveDilation(-4, "DIRECT_MAHFOUZ") == 3);

    /*End to end: a Mahfouz application equals an explicit dilation-3
     * application (the UpdateDilationFrames override, byte-identical).*/
    Frame f_mahfouz(FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
    jta::EdgeProcessor::ApplyToFrame(
        jta::EdgeProcessingParams{3, 40, 120, 0, "DIRECT_MAHFOUZ"}, f_mahfouz);
    Frame f_explicit(FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
    jta::EdgeProcessor::ApplyToFrame(
        jta::EdgeProcessingParams{3, 40, 120, 3, "DIRECT_DILATION"},
        f_explicit);
    REQUIRE(SameMat(f_mahfouz.GetEdgeImage(), f_explicit.GetEdgeImage()));
    REQUIRE(
        SameMat(f_mahfouz.GetDilationImage(), f_explicit.GetDilationImage()));
}

TEST_CASE(
    "edge_processor: zero/negative/out-of-range params behave as today "
    "(pinned, not normalized)",
    "[edge_processor]") {
    /*Dilation-constant decision: clamp negatives to 0, never touch
     * non-negatives (matches the slots' clamp before U5).*/
    REQUIRE(jta::EdgeProcessor::ResolveDilation(-5, "DIRECT_DILATION") == 0);
    REQUIRE(jta::EdgeProcessor::ResolveDilation(0, "DIRECT_DILATION") == 0);
    REQUIRE(jta::EdgeProcessor::ResolveDilation(6, "DIRECT_DILATION") == 6);

    /*Aperture/thresholds pass through unnormalized: low > high is legal and
     * deterministic (the same Canny call the slots always made).*/
    Frame f1(FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
    jta::EdgeProcessor::ApplyToFrame(
        jta::EdgeProcessingParams{3, 200, 50, 0, "DIRECT_DILATION"}, f1);
    const cv::Mat first = f1.GetEdgeImage().clone();
    jta::EdgeProcessor::ApplyToFrame(
        jta::EdgeProcessingParams{3, 200, 50, 0, "DIRECT_DILATION"}, f1);
    REQUIRE(SameMat(first, f1.GetEdgeImage()));

    /*Thresholds beyond any gradient magnitude: empty edge image.*/
    Frame f2(FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
    jta::EdgeProcessor::ApplyToFrame(
        jta::EdgeProcessingParams{3, 40, 5000, 0, "DIRECT_DILATION"}, f2);
    REQUIRE(NonZero(f2.GetEdgeImage()) == 0);

    /*Aperture 5 (a valid odd aperture) still detects the square.*/
    Frame f3(FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
    jta::EdgeProcessor::ApplyToFrame(
        jta::EdgeProcessingParams{5, 40, 120, 0, "DIRECT_DILATION"}, f3);
    REQUIRE(IsBinary(f3.GetEdgeImage()));
    REQUIRE(NonZero(f3.GetEdgeImage()) > 0);
}

TEST_CASE(
    "edge_processor: applying the default params restores the original edge "
    "state (reset cascade)",
    "[edge_processor]") {
    /*A frame built with the defaults; the reset slot's three setValue calls
     * cascade into applications of exactly the default params.*/
    Frame f(FixturePath(), APERTURE, LOW_THRESH, HIGH_THRESH, 0);
    const cv::Mat original_edge = f.GetEdgeImage().clone();
    const cv::Mat original_dilated = f.GetDilationImage().clone();

    /*Disturb the state first (proves the fixture actually changed).*/
    jta::EdgeProcessor::ApplyToFrame(
        jta::EdgeProcessingParams{5, 10, 200, 2, "DIRECT_DILATION"}, f);
    REQUIRE_FALSE(SameMat(original_edge, f.GetEdgeImage()));

    /*The reset application restores the original edge + dilated images and
     * the stored constants.*/
    jta::EdgeProcessor::ApplyToFrame(DefaultParams(), f);
    REQUIRE(SameMat(original_edge, f.GetEdgeImage()));
    REQUIRE(SameMat(original_dilated, f.GetDilationImage()));
    REQUIRE(f.GetAperture() == APERTURE);
    REQUIRE(f.GetLowThreshold() == LOW_THRESH);
    REQUIRE(f.GetHighThreshold() == HIGH_THRESH);

    /*ResetFromOriginal (the segmentation-reset path) is consistent too.*/
    f.ResetFromOriginal();
    REQUIRE(SameMat(original_edge, f.GetEdgeImage()));
    REQUIRE(SameMat(original_dilated, f.GetDilationImage()));
}

TEST_CASE(
    "edge_processor: apply-all equals per-frame single application",
    "[edge_processor]") {
    /*Canny accepts only odd apertures 3/5/7; the application overwrites the
     * constructor params.*/
    jta::EdgeProcessingParams params{5, 60, 180, 2, "DIRECT_DILATION_T1"};
    const int n = 3;

    std::vector<Frame> batched;
    std::vector<Frame> per_frame;
    for (int i = 0; i < n; ++i) {
        /*Distinct constructor params (odd apertures): the application must
         * overwrite them.*/
        batched.emplace_back(
            FixturePath(), APERTURE + i * 2, LOW_THRESH + i,
            HIGH_THRESH + i, i);
        per_frame.emplace_back(
            FixturePath(), APERTURE + i * 2, LOW_THRESH + i,
            HIGH_THRESH + i, i);
    }

    jta::EdgeProcessor::ApplyToFrames(params, batched);
    for (Frame& f : per_frame) {
        jta::EdgeProcessor::ApplyToFrame(params, f);
    }

    for (int i = 0; i < n; ++i) {
        REQUIRE(SameMat(batched[i].GetEdgeImage(), per_frame[i].GetEdgeImage()));
        REQUIRE(
            SameMat(batched[i].GetDilationImage(), per_frame[i].GetDilationImage()));
    }
}
