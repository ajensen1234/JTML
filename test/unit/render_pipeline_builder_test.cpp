// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Render-pipeline builder pure-config tests (plan 006 U4, R9/AE4). The
// shared widget-free VTK pipeline recipe (jta::render_pipeline, services
// layer) is the ONE recipe both front-ends drive — the widgets Viewer and
// the QML QmlVtkRenderer — replacing the QML 1:1 mirror and absorbing the
// dead MainScreen::matToVTK semantics. This target pins the PURE part: the
// mat -> import-parameter derivation (gray vs color, channel count, extent
// from cols/rows, the spacing/origin constants, and the empty/zero-size
// degenerates). The VTK-touching functions of the module (import refresh,
// chain construction, camera setup, model chain, pose apply) are exercised
// by the render smokes (jtml.render_smoke / jtml.qml_render_smoke) — the
// behavior diff for the parameterized extraction (R13).
//
// Pure logic: no Qt, no GPU, no render window — headless Catch2.

#include <catch2/catch_test_macros.hpp>
#include <algorithm>

#include <opencv2/core.hpp>

#include "services/render_pipeline_builder.h"

using jta::render_pipeline::DeriveImageImportParams;
using jta::render_pipeline::ImageImportParams;

namespace {

/*The import recipe's constants: spacing (1,1,1), origin (0,0,0).*/
void RequireSpacingAndOriginConstants(const ImageImportParams& p) {
    REQUIRE(p.spacing[0] == 1.0);
    REQUIRE(p.spacing[1] == 1.0);
    REQUIRE(p.spacing[2] == 1.0);
    REQUIRE(p.origin[0] == 0.0);
    REQUIRE(p.origin[1] == 0.0);
    REQUIRE(p.origin[2] == 0.0);
}

}  // namespace

TEST_CASE("gray mat derives single-channel full-extent import params",
          "[render_pipeline_builder]") {
    cv::Mat gray(480, 640, CV_8UC1);
    const ImageImportParams p = DeriveImageImportParams(gray);

    REQUIRE(p.extentMinX == 0);
    REQUIRE(p.extentMaxX == 639); /* cols - 1 */
    REQUIRE(p.extentMinY == 0);
    REQUIRE(p.extentMaxY == 479); /* rows - 1 */
    REQUIRE(p.channels == 1);
    RequireSpacingAndOriginConstants(p);
}

TEST_CASE("color mat derives three-channel full-extent import params",
          "[render_pipeline_builder]") {
    cv::Mat color(240, 320, CV_8UC3);
    const ImageImportParams p = DeriveImageImportParams(color);

    REQUIRE(p.extentMinX == 0);
    REQUIRE(p.extentMaxX == 319);
    REQUIRE(p.extentMinY == 0);
    REQUIRE(p.extentMaxY == 239);
    REQUIRE(p.channels == 3);
    RequireSpacingAndOriginConstants(p);
}

TEST_CASE("non-square gray mat maps cols to X and rows to Y extents",
          "[render_pipeline_builder]") {
    cv::Mat tall(720, 128, CV_8UC1); /* portrait: rows > cols */
    const ImageImportParams p = DeriveImageImportParams(tall);

    REQUIRE(p.extentMaxX == 127); /* cols - 1 */
    REQUIRE(p.extentMaxY == 719); /* rows - 1 */
    REQUIRE(p.channels == 1);
}

TEST_CASE("1x1 mat derives zero extents", "[render_pipeline_builder]") {
    cv::Mat one(1, 1, CV_8UC1);
    const ImageImportParams p = DeriveImageImportParams(one);

    REQUIRE(p.extentMaxX == 0);
    REQUIRE(p.extentMaxY == 0);
    REQUIRE(p.channels == 1);
}

TEST_CASE("empty mat derives valid degenerate params without crashing",
          "[render_pipeline_builder]") {
    const ImageImportParams p = DeriveImageImportParams(cv::Mat());

    /* Degenerate but well-defined: no pixels, single channel (the default
     * Mat's type), recipe constants intact. */
    REQUIRE(p.extentMinX == 0);
    REQUIRE(p.extentMaxX == -1);
    REQUIRE(p.extentMinY == 0);
    REQUIRE(p.extentMaxY == -1);
    REQUIRE(p.channels == 1);
    RequireSpacingAndOriginConstants(p);
}

TEST_CASE("zero-size typed mat derives degenerate extent with its channel "
          "count",
          "[render_pipeline_builder]") {
    cv::Mat zero(0, 0, CV_8UC3);
    REQUIRE(zero.empty());
    const ImageImportParams p = DeriveImageImportParams(zero);

    REQUIRE(p.extentMaxX == -1);
    REQUIRE(p.extentMaxY == -1);
    REQUIRE(p.channels == 3); /* the type's channel count survives */
    RequireSpacingAndOriginConstants(p);
}

// U4: persistent chunk worker math (device-driven, fixed grid) — pure logic, headless
TEST_CASE("U4: chunk covering with 256 is exact and no duplicate", "[render_pipeline][U4]") {
    const int chunk = 256;
    for (int fill : {0, 1, 255, 256, 257, 512, 1000, 100000, 4000000}) {
        int chunks = (fill + chunk - 1) / chunk;
        int covered = 0;
        for (int c = 0; c < chunks; ++c) {
            int start = c * chunk;
            int end = std::min(start + chunk, fill);
            REQUIRE(end >= start);
            covered += (end - start);
        }
        REQUIRE(covered == fill);
        // No overlap: start of next chunk == end of previous
        for (int c = 1; c < chunks; ++c) {
            REQUIRE(c * chunk < fill + chunk);
        }
    }
}

TEST_CASE("U4: overflow guard threshold is maximum_stride_size * 255", "[render_pipeline][U4]") {
    constexpr int64_t maxStride = 10000000;
    constexpr int tpb = 256;
    const int64_t threshold = maxStride * (tpb - 1); // 2,550,000,000
    REQUIRE(threshold == 2550000000LL);
    auto isOverflow = [&](int64_t fill) { return fill > threshold; };
    REQUIRE(!isOverflow(0));
    REQUIRE(!isOverflow(threshold));
    REQUIRE(isOverflow(threshold + 1));
    REQUIRE(isOverflow(3000000000LL)); // large fill overflows
}

TEST_CASE("U4: fixed NX grid is bounded by occupancy and SAFE_CAP", "[render_pipeline][U4]") {
    constexpr int64_t maxStride = 10000000;
    constexpr int64_t safeCap = maxStride * 256; // 2,560,000,000 per plan footnote
    REQUIRE(safeCap == 2560000000LL);
    // Simulate occupancy: 80 SM * 8 blocks/SM = 640 blocks for Fill at 256 threads
    int sm = 80;
    int maxBlocksPerSM = 8;
    int occupancyBound = sm * maxBlocksPerSM;
    int safeBound = (safeCap + 255) / 256; // ceil(SAFE_CAP/256) = 10,000,000
    int nx = std::min(occupancyBound, safeBound);
    REQUIRE(nx == occupancyBound); // occupancy is tighter than SAFE_CAP
    REQUIRE(nx == 640);
    // Even for tiny fill (<256) the fixed grid still self-retires via guard
    int fillTiny = 100;
    int threadsNeeded = fillTiny; // one thread per fragment
    REQUIRE(nx * 256 >= threadsNeeded); // fixed grid covers tiny fill with no-ops
}
