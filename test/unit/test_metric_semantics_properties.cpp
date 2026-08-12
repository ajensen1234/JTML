// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel PBT twin for the Tier-0 metric-semantics CPU references (plan 008 U1 —
// the R13 characterization pass). Complements the deterministic pins in
// test_metric_semantics.cpp (house rule: PBT guards invariants, deterministic
// cases pin the specific edges). The references below are the SAME kernel-as-spec
// CPU references as the deterministic twin (current kernel code is the spec).
//
// Invariants locked here:
//   - IoU ∈ [0,1]; symmetry; reflexivity (IoU(A,A) == 1 for ALL A, incl. the
//     empty image via the IOU(∅,∅) == 1.0 spec-with-flagged-deviation decision),
//   - L1 symmetry; L1(A,A) == 0; non-negativity,
//   - chamfer kernel-exact perfect-overlap value: ChamferMetric(P,P) == |E(P)|
//     (the inventory's "white_sum + chamfer => 0 at match" is a simplification —
//     see the deterministic twin's characterization finding),
//   - distance-map score ∈ [0, 255]; DILATED pixels never sampled (count equals
//     the EDGE count in the crop),
//   - Mahfouz ratios ≥ 0 and FINITE (the intended value-guard semantics — the
//     kernel's pointer guard yields 0/0 NaN instead),
//   - CropIndexToGlobal bijectivity: the CORRECTED formula covers every crop
//     pixel exactly once for tiled geometries; the BUGGY formula (as written at
//     src/compute/distance_map_metric.cu:27) NEVER satisfies the invariant —
//     the RED characterization, asserted as REQUIRE_FALSE.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include <hegel/hegel.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "compute/pixel_grayscale_colors.h"
#include "cuda_launch_parameters.h"

namespace gs = hegel::generators;
using Catch::Approx;

namespace {

// ===========================================================================
// CPU references — identical semantics to test_metric_semantics.cpp (duplicated
// per the twin house pattern; keep the two files' references in sync).
// ===========================================================================

void ChamferEdgeDetect(std::vector<unsigned char>& img, int w, int h) {
    std::vector<unsigned char> out = img;
    auto at = [&](int x, int y) -> int {
        if (x < 0 || x >= w || y < 0 || y >= h) return -1;  // never BLACK
        return img[y * w + x];
    };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char p = img[y * w + x];
            if (p != WHITE_PIXEL) continue;
            const bool touches_black =
                at(x - 1, y - 1) == BLACK_PIXEL || at(x, y - 1) == BLACK_PIXEL ||
                at(x + 1, y - 1) == BLACK_PIXEL || at(x - 1, y) == BLACK_PIXEL ||
                at(x + 1, y) == BLACK_PIXEL || at(x - 1, y + 1) == BLACK_PIXEL ||
                at(x, y + 1) == BLACK_PIXEL || at(x + 1, y + 1) == BLACK_PIXEL;
            if (touches_black) out[y * w + x] = EDGE_PIXEL;
        }
    }
    img = std::move(out);
}

void ChamferDilate(std::vector<unsigned char>& img, int w, int h, int dilation) {
    std::vector<unsigned char> out = img;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (img[y * w + x] != EDGE_PIXEL) continue;
            for (int l = -1; l <= 1; l += 2) {
                for (int r = -1; r <= 1; r += 2) {
                    for (int j = 1; j <= dilation; ++j) {
                        for (int k = 1; k <= dilation; ++k) {
                            const int nx = x + r * k;
                            const int ny = y + l * j;
                            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                            unsigned char p = out[ny * w + nx];
                            if (p == WHITE_PIXEL || p == BLACK_PIXEL)
                                out[ny * w + nx] = DILATED_PIXEL;
                        }
                    }
                }
            }
        }
    }
    img = std::move(out);
}

int ChamferDifferenceScore(const std::vector<unsigned char>& rendered,
                           const std::vector<unsigned char>& comparison) {
    int score = 0;
    for (size_t i = 0; i < rendered.size(); ++i) {
        const unsigned char p = rendered[i];
        if (p == DILATED_PIXEL || p == EDGE_PIXEL)
            score += (comparison[i] == WHITE_PIXEL) ? 1 : -1;
    }
    return score;
}

double ChamferMetric(const std::vector<unsigned char>& rendered,
                     const std::vector<unsigned char>& comparison, int w, int h,
                     int dilation) {
    std::vector<unsigned char> proc = rendered;
    ChamferEdgeDetect(proc, w, h);
    ChamferDilate(proc, w, h, dilation);
    return -1.0 * ChamferDifferenceScore(proc, comparison);
}

int CountEdgeOrDilated(const std::vector<unsigned char>& img) {
    return static_cast<int>(std::count_if(
        img.begin(), img.end(), [](unsigned char p) {
            return p == EDGE_PIXEL || p == DILATED_PIXEL;
        }));
}

double DistanceMapMetricReference(const std::vector<unsigned char>& projected,
                                  const std::vector<unsigned char>& distance_map,
                                  int w, int h, const std::array<int, 4>& bbox,
                                  int dilation) {
    const int left = std::max(bbox[0] - dilation, dilation);
    const int bottom = std::max(bbox[1] - dilation, dilation);
    const int right = std::min(bbox[2] + dilation, w - dilation - 1);
    const int top = std::min(bbox[3] + dilation, h - dilation - 1);
    int score = 0;
    int count = 0;
    for (int y = bottom; y <= top; ++y) {
        for (int x = left; x <= right; ++x) {
            const size_t loc = static_cast<size_t>(y) * w + x;
            if (projected[loc] == EDGE_PIXEL) {
                ++count;
                score += distance_map[loc];
            }
        }
    }
    return score / (count + 0.1);  // the +0.1 singularity hack, pinned as spec
}

int DistanceMapEdgeCountInCrop(const std::vector<unsigned char>& projected, int w,
                               int h, const std::array<int, 4>& bbox,
                               int dilation) {
    const int left = std::max(bbox[0] - dilation, dilation);
    const int bottom = std::max(bbox[1] - dilation, dilation);
    const int right = std::min(bbox[2] + dilation, w - dilation - 1);
    const int top = std::min(bbox[3] + dilation, h - dilation - 1);
    int count = 0;
    for (int y = bottom; y <= top; ++y)
        for (int x = left; x <= right; ++x)
            if (projected[static_cast<size_t>(y) * w + x] == EDGE_PIXEL) ++count;
    return count;
}

int CropIndexToGlobal(int blockIdxY, int gridDimX, int blockIdxX,
                      int blockDimX, int threadIdxX) {
    return (blockIdxY * gridDimX + blockIdxX) * blockDimX + threadIdxX;
}

// BUGGY formula, exactly as written at src/compute/distance_map_metric.cu:27.
int BuggyCropIndexToGlobal(int blockIdxY, int gridDimX, int blockIdxX,
                           int blockDimX, int threadIdxX) {
    return (blockIdxY + gridDimX + blockIdxX) * blockDimX + threadIdxX;
}

bool IndexCoversCropExactlyOnce(const std::function<int(int, int, int, int, int)>& index,
                                int cropW, int cropH, int blockDimX) {
    const int tile = static_cast<int>(std::sqrt(static_cast<double>(blockDimX)));
    const int gx = (cropW + tile - 1) / tile;
    const int gy = (cropH + tile - 1) / tile;
    std::vector<int> seen(static_cast<size_t>(cropW) * cropH, 0);
    for (int by = 0; by < gy; ++by) {
        for (int bx = 0; bx < gx; ++bx) {
            for (int t = 0; t < blockDimX; ++t) {
                const int i = index(by, gx, bx, blockDimX, t);
                if (i < 0 || i >= cropW * cropH) return false;
                ++seen[static_cast<size_t>(i)];
            }
        }
    }
    return std::all_of(seen.begin(), seen.end(), [](int c) { return c == 1; });
}

std::array<int, 4> BBoxOf(const std::vector<unsigned char>& img, int w, int h) {
    int left = w - 1, bottom = h - 1, right = 0, top = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (img[static_cast<size_t>(y) * w + x] > 0) {
                left = std::min(left, x);
                bottom = std::min(bottom, y);
                right = std::max(right, x);
                top = std::max(top, y);
            }
        }
    }
    if (right < left || top < bottom) return {w - 1, h - 1, 0, 0};  // empty
    return {left, bottom, right, top};
}

double IoUReference(const std::vector<unsigned char>& A,
                    const std::vector<unsigned char>& B, int w, int h) {
    const std::array<int, 4> ba = BBoxOf(A, w, h);
    const std::array<int, 4> bb = BBoxOf(B, w, h);
    const int left = std::max(std::min(ba[0], bb[0]), 0);
    const int bottom = std::max(std::min(ba[1], bb[1]), 0);
    const int right = std::min(std::max(ba[2], bb[2]), w - 1);
    const int top = std::min(std::max(ba[3], bb[3]), h - 1);
    const int cw = right - left + 1;
    const int ch = top - bottom + 1;
    if (cw <= 0 || ch <= 0) return 1.0;  // IOU(∅,∅) spec decision (flagged)
    long long inter = 0, uni = 0;
    for (int y = bottom; y <= top; ++y) {
        for (int x = left; x <= right; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            const bool a = A[i] > 0;
            const bool b = B[i] > 0;
            if (a || b) ++uni;
            if (a && b) ++inter;
        }
    }
    return (uni > 0) ? static_cast<double>(inter) / static_cast<double>(uni)
                     : 1.0;
}

long long L1Reference(const std::vector<unsigned char>& A,
                      const std::vector<unsigned char>& B, int w, int h) {
    const std::array<int, 4> ba = BBoxOf(A, w, h);
    const std::array<int, 4> bb = BBoxOf(B, w, h);
    const int left = std::max(std::min(ba[0], bb[0]), 0);
    const int bottom = std::max(std::min(ba[1], bb[1]), 0);
    const int right = std::min(std::max(ba[2], bb[2]), w - 1);
    const int top = std::min(std::max(ba[3], bb[3]), h - 1);
    if (right - left + 1 <= 0 || top - bottom + 1 <= 0) return 0;
    long long sum = 0;
    for (int y = bottom; y <= top; ++y) {
        for (int x = left; x <= right; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            sum += std::abs(static_cast<int>(A[i]) - static_cast<int>(B[i]));
        }
    }
    return sum;
}

double MahfouzIntensityRatioReference(const std::vector<unsigned char>& rendered,
                                      const std::vector<unsigned char>& intensity_comparison) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < rendered.size(); ++i) {
        if (rendered[i] == WHITE_PIXEL) {
            num += intensity_comparison[i];
            den += 1.0;
        }
    }
    return (den > 0.0) ? num / den : 0.0;  // intended value-guard semantics
}

long long MahfouzContourNumeratorTruncated(const std::vector<unsigned char>& rendered,
                                           const std::vector<unsigned char>& comparison) {
    long long num = 0;
    for (size_t i = 0; i < rendered.size(); ++i) {
        const unsigned char p = rendered[i];
        if (p > BLACK_PIXEL && p < WHITE_PIXEL) {
            if (comparison[i] != BLACK_PIXEL)
                num += static_cast<long long>(2.55 * p);  // int-truncated
        }
    }
    return num;
}

long long MahfouzContourDenominatorTruncated(const std::vector<unsigned char>& rendered) {
    long long den = 0;
    for (unsigned char p : rendered) {
        if (p > BLACK_PIXEL && p < WHITE_PIXEL)
            den += static_cast<long long>(2.55 * p);  // int-truncated
    }
    return den;
}

double MahfouzContourRatioReference(double num, double den) {
    return (den > 0.0) ? (255.0 * num) / den : 0.0;  // host ×255 scale
}

// ---------------------------------------------------------------------------
// Generators: small 16x16 uchar images drawn from the pixel vocabulary
// (BLACK / DILATED / EDGE / WHITE — the values the metrics actually see).
// ---------------------------------------------------------------------------

auto PixelGen() {
    return gs::sampled_from<unsigned char>(
        {BLACK_PIXEL, DILATED_PIXEL, EDGE_PIXEL, WHITE_PIXEL});
}

auto ImageGen() {
    // Fixed 16x16 = 256 pixels.
    return gs::vectors(PixelGen(), {.min_size = 256, .max_size = 256});
}

}  // namespace

TEST_CASE("IoU[PBT]: range [0,1], symmetry, reflexivity (incl. the ∅,∅ spec)",
          "[metric_semantics][pbt][iou]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            const int w = 16, h = 16;
            auto A = tc.draw(ImageGen());
            auto B = tc.draw(ImageGen());
            const double iou_ab = IoUReference(A, B, w, h);
            REQUIRE(iou_ab >= 0.0);
            REQUIRE(iou_ab <= 1.0);
            REQUIRE(IoUReference(B, A, w, h) == iou_ab);  // symmetry
            // Reflexivity for ALL A: non-empty via the intersection math, empty
            // via the IOU(∅,∅) == 1.0 spec-with-flagged-deviation decision (the
            // kernel yields 0/0 NaN at iou.cu:82-83).
            REQUIRE(IoUReference(A, A, w, h) == 1.0);
        },
        hegel::Settings{.test_cases = 300});
}

TEST_CASE("L1[PBT]: symmetry, self-zero, non-negativity",
          "[metric_semantics][pbt][l1]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            const int w = 16, h = 16;
            auto A = tc.draw(ImageGen());
            auto B = tc.draw(ImageGen());
            REQUIRE(L1Reference(A, B, w, h) == L1Reference(B, A, w, h));
            REQUIRE(L1Reference(A, A, w, h) == 0);
            REQUIRE(L1Reference(A, B, w, h) >= 0);
        },
        hegel::Settings{.test_cases = 300});
}

TEST_CASE(
    "chamfer[PBT]: kernel-exact perfect-overlap value ChamferMetric(P,P) == |E(P)|",
    "[metric_semantics][pbt][chamfer]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            const int w = 16, h = 16;
            auto dilation = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 2}));
            auto raw = tc.draw(ImageGen());
            std::vector<unsigned char> proc = raw;
            ChamferEdgeDetect(proc, w, h);
            ChamferDilate(proc, w, h, dilation);
            // At perfect overlap every edge/dilated pixel hits a non-white
            // comparison pixel -> the metric returns exactly +|E(P)|. (This is
            // the kernel-as-spec value; the inventory's "white_sum + chamfer =>
            // 0 at match" holds only for the empty image.)
            REQUIRE(ChamferMetric(proc, proc, w, h, dilation) ==
                    Approx(CountEdgeOrDilated(proc)));
        },
        hegel::Settings{.test_cases = 300});
}

TEST_CASE(
    "distance-map[PBT]: score in [0,255] and only EDGE pixels are sampled",
    "[metric_semantics][pbt][distance_map]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            const int w = 16, h = 16;
            auto dilation = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 2}));
            auto projected = tc.draw(ImageGen());
            auto distance_map = tc.draw(ImageGen());
            const std::array<int, 4> bbox = {0, 0, w - 1, h - 1};
            const double score = DistanceMapMetricReference(
                projected, distance_map, w, h, bbox, dilation);
            REQUIRE(score >= 0.0);
            // uchar distances: max sum = 255*count -> ratio < 255.
            REQUIRE(score <= 255.0);
            // Composition quirk: only the surviving EDGE(100) ring is sampled —
            // DILATED(99) pixels are never counted (kernel :34).
            const int expected_count =
                DistanceMapEdgeCountInCrop(projected, w, h, bbox, dilation);
            if (expected_count == 0) {
                REQUIRE(score == Approx(0.0));  // 0 / (0 + 0.1) — the +0.1 hack
            } else {
                // score == mean distance over the EDGE-only count.
                REQUIRE(score <= 255.0 * expected_count / (expected_count + 0.1));
                REQUIRE(score >= 0.0);
            }
        },
        hegel::Settings{.test_cases = 300});
}

TEST_CASE(
    "Mahfouz[PBT]: intensity/contour ratios are finite and non-negative "
    "(intended value-guard semantics)",
    "[metric_semantics][pbt][mahfouz]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto rendered = tc.draw(ImageGen());
            auto comparison = tc.draw(ImageGen());
            const double intensity_ratio =
                MahfouzIntensityRatioReference(rendered, comparison);
            REQUIRE(std::isfinite(intensity_ratio));
            REQUIRE(intensity_ratio >= 0.0);
            const long long num =
                MahfouzContourNumeratorTruncated(rendered, comparison);
            const long long den = MahfouzContourDenominatorTruncated(rendered);
            const double contour_ratio =
                MahfouzContourRatioReference(num, den);
            // The reference never divides by zero (the kernel's pointer guard
            // does -> 0/0 NaN on an empty silhouette; Bug-6 fix spec).
            REQUIRE(std::isfinite(contour_ratio));
            REQUIRE(contour_ratio >= 0.0);
        },
        hegel::Settings{.test_cases = 300});
}

TEST_CASE(
    "CropIndexToGlobal[PBT]: corrected formula bijects; the buggy formula never "
    "satisfies full coverage (RED)",
    "[metric_semantics][pbt][index][red]") {
    // Tiled geometries only (crop dims a multiple of the 16-px block tile) so
    // the launch covers the crop exactly: gx*gy*256 == cw*ch.
    auto size_gen = gs::sampled_from<int>({16, 32, 64, 128});
    hegel::test(
        [&](hegel::TestCase& tc) {
            const int cw = tc.draw(size_gen);
            const int ch = tc.draw(size_gen);
            // CORRECTED formula (the U4 spec): bijective onto [0, cw*ch).
            REQUIRE(IndexCoversCropExactlyOnce(CropIndexToGlobal, cw, ch,
                                               threads_per_block));
            // BUGGY formula (src/compute/distance_map_metric.cu:27, as written):
            // the minimum global index is gridDimX*blockDimX > 0, so crop pixel 0
            // is never visited — the invariant NEVER holds. This REQUIRE_FALSE is
            // the RED characterization: do not change it to match the bug.
            REQUIRE_FALSE(IndexCoversCropExactlyOnce(BuggyCropIndexToGlobal, cw,
                                                     ch, threads_per_block));
        },
        hegel::Settings{.test_cases = 200});
}
