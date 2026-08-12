// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Tier-0 metric-semantics pins + CPU references (plan 008 U1 — the run's R13
// characterization pass). Every CPU reference below encodes the CURRENT kernel
// code as spec (kernel-as-spec: "where the math is ambiguous the CURRENT KERNEL
// CODE is the spec", .panoptes/optimizer-deep-dive/angles/02-cost-path-foundation.org).
//
// RED pins (tagged [red]) document pre-fix bugs on today's source; they are the
// spec the behavior-neutral fixes (U2) and the live distance-map index fix (U4)
// must satisfy. Do NOT "fix" a test to match a bug — a RED pin flips green only
// when the production fix lands (sym_trap + stage-guard flipped in U2, when the
// transcriptions below were updated to the fixed calls at the same time).
//
// Pins in this file:
//   - chamfer stage functions (edge / quadrant-only dilation / difference) +
//     the kernel-exact perfect-overlap composition value,
//   - distance-map score (+0.1 singularity hack) + the EDGE-only sampling quirk,
//   - CropIndexToGlobal full-coverage pin (CORRECTED formula passes; the BUGGY
//     formula as written at src/compute/distance_map_metric.cu:27 fails — RED),
//   - Mahfouz truncated-vs-float ratio pair (2.55 / x255 / -2.67 / -1 scale chain),
//   - IoU/L1 CPU references incl. the IOU(empty,empty) == 1.0 spec decision
//     (flagged deviation from the kernel's 0/0 NaN at iou.cu:82-83),
//   - ComputeSumWhitePixels,
//   - dilation registry/constants pins (settings_constants.h IS a Flood tibia
//     transcription; engine runtime dilation 6/4/1; baseline.json's {6,3,1} is
//     the known-stale docs-claim),
//   - stage-guard semantics pin (Bug 3; fixed in U2 — the tautology
//     characterization became the observable getStage() pin via the minimal
//     accessor, the second documented wizard-region exception),
//   - DD PolePenalty extraction spec (init-0 + Y-axis, Bugs 1+2; production
//     fixes landed in U2),
//   - sym_trap tibia-transform pins (Bug 4; RED via [!mayfail] pre-U2, now
//     real assertions on the fixed call).
//
// Direct-compile pattern: the target compiles data_structures_6D.cpp (Point6D)
// and links the REAL CostFunctionManager from jtml_compute (constructor +
// getStage() only — CPU-only, no CUDA calls at runtime; the same pattern as
// jtml.experimental_settings / jtml.cost_function_registry). The sym_trap pin
// includes the REAL compute-layer transform header (compile-time CUDA/OpenCV
// headers only — no runtime GPU).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "compute/pixel_grayscale_colors.h"
#include "compute/Stage.h"
/*NOTE: must precede CostFunctionManager.h — that header includes the custom-
variables headers INSIDE the class body, so (with #pragma once) including it
first would textually nest the transform functions in class scope instead of
global scope.*/
#include "compute/sym_trap_functionCustomVariables.h"
#include "compute/CostFunctionManager.h"
#include "cuda_launch_parameters.h"
#include "domain/settings_constants.h"

using Catch::Approx;

namespace {

// ===========================================================================
// CPU references — CURRENT KERNEL CODE AS SPEC (kernel-as-spec).
// ===========================================================================

// ---------------------------------------------------------------------------
// Chamfer — FastImplantDilationMetric (src/compute/fast_implant_dilation_metric.cu).
// Stage semantics pinned: (a) edge = 8-neighbor boundary of WHITE(255) against
// BLACK(0) -> EDGE(100), interior/exterior untouched; (b) dilation writes
// DILATED(99) at quadrant offsets (±j,±k), j,k in [1,dilation], over WHITE/BLACK
// only — EDGE and already-DILATED are never overwritten and the axial cross
// (j==0 or k==0) is NEVER dilated (quadrant-only morphology, DilateKernel
// :83-118); (c) difference = +1 per dilated/edge pixel where comparison==WHITE,
// −1 otherwise (:120-145); (d) the metric returns −1·score; (e) the cost
// composition is comparison_white_sum + chamfer (+ distance-map, separate pin).
// Out-of-image neighbors never count as BLACK (the kernel's padded tiles stay
// in-image by construction); out-of-bounds dilation writes are skipped.
// ---------------------------------------------------------------------------

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
            // All four quadrant directions (±1, ±1) — the kernel's (l, r) pair
            // decomposition: l = 2*((i%4)/2)-1, r = 2*(i%2)-1.
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

// RAW difference score (before the metric's −1 factor): +1 per dilated/edge
// pixel where the comparison is WHITE, −1 otherwise.
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

// The full pipeline the GPU runs: edge-detect + dilate the rendered image in
// place, then the difference vs the comparison frame; returns −1·score.
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

int ComputeSumWhitePixelsReference(const std::vector<unsigned char>& img) {
    return static_cast<int>(std::count(img.begin(), img.end(), WHITE_PIXEL));
}

// ---------------------------------------------------------------------------
// Distance map — DistanceMapMetric (src/compute/distance_map_metric.cu).
// Semantics pinned: sum of distance_map values at pixels where the
// (already-mutated) projected image == EDGE(100) within the bbox±dilation crop,
// divided by (count + 0.1). The +0.1 avoids the 0/0 singularity (pinned as spec,
// flagged for review). Composition quirk: the input buffer arrives already
// mutated by FastImplantDilationMetric, so only the surviving EDGE(100) ring is
// sampled — DILATED(99) pixels are never counted (kernel :34,
// projected_image[orig_loc] == EDGE_PIXEL). The host crop formulas (:40-44) are
// reproduced verbatim.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Distance-map grid→pixel index mapping (Bug 5 — the ONE live kernel bug).
// DistanceMapMetric_Kernel (src/compute/distance_map_metric.cu:27) computes a
// global thread id i, then derives the crop pixel: bb_row = i/crop_w,
// bb_col = i%crop_w. The full-coverage invariant over the crop is: every crop
// pixel is visited exactly once — i.e. the multiset of i values over the launch
// geometry equals [0, crop_w*crop_h). CORRECTED formula (the fix U4 must land;
// matches the pattern already correct in iou.cu:37, l_1_1_matrix_diff_norm.cu:24,
// fast_implant_dilation_metric.cu:129):
// ---------------------------------------------------------------------------

int CropIndexToGlobal(int blockIdxY, int gridDimX, int blockIdxX,
                      int blockDimX, int threadIdxX) {
    return (blockIdxY * gridDimX + blockIdxX) * blockDimX + threadIdxX;
}

// BUGGY formula, exactly as written at src/compute/distance_map_metric.cu:27
// (pre-fix source truth — the RED characterization this file pins).
int BuggyCropIndexToGlobal(int blockIdxY, int gridDimX, int blockIdxX,
                           int blockDimX, int threadIdxX) {
    return (blockIdxY + gridDimX + blockIdxX) * blockDimX + threadIdxX;
}

// True iff the index multiset over the launch geometry (grid = ceil(crop/16) x
// ceil(crop/16) square 16x16 blocks, blockDimX threads each — the kernel's
// launch site, distance_map_metric.cu:56-68) covers every crop pixel exactly
// once (bijective onto [0, crop_w*crop_h)).
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

// ---------------------------------------------------------------------------
// Mahfouz ratios — ImplantMahfouzMetric (src/compute/implant_mahfouz_metric.cu).
// Scale chain (kernel-as-spec, pinned as-is):
//   - the contour kernels accumulate 2.55 * pixel into INT atomic counters
//     (atomicAdd(int*, double) truncates toward zero — Bug 6, :93/:122),
//   - the host scales the contour numerator by 255.0 (:~420) before the ratio,
//   - the final combination is contour_ratio * (-2.67) + intensity_ratio * (-1)
//     (:~465) — an unnormalized weighted sum, pinned as-is,
//   - the intensity branch is integer-valued (no scale; :129-155).
// The reference implements the INTENDED value-guard semantics (zero denominator
// -> ratio 0); the kernel's `if (pixel_score_ != 0)` pointer guard (:324/:446)
// always passes (pixel_score_ is never null), so the kernel yields 0/0 NaN for
// an empty silhouette — that guard is the Bug-6 fix spec (documented deviation).
// ---------------------------------------------------------------------------

// intensity ratio = Σ(comparison-white-silhouette intensity · rendered-white) /
//                   Σ(rendered-white)
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

// The kernel's ACTUAL int-truncated accumulation: (int)(2.55 * pixel).
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

// The INTENDED float accumulation (what a corrected kernel would sum).
double MahfouzContourNumeratorFloat(const std::vector<unsigned char>& rendered,
                                    const std::vector<unsigned char>& comparison) {
    double num = 0.0;
    for (size_t i = 0; i < rendered.size(); ++i) {
        const unsigned char p = rendered[i];
        if (p > BLACK_PIXEL && p < WHITE_PIXEL) {
            if (comparison[i] != BLACK_PIXEL) num += 2.55 * p;
        }
    }
    return num;
}

double MahfouzContourDenominatorFloat(const std::vector<unsigned char>& rendered) {
    double den = 0.0;
    for (unsigned char p : rendered) {
        if (p > BLACK_PIXEL && p < WHITE_PIXEL) den += 2.55 * p;
    }
    return den;
}

double MahfouzContourRatioReference(double num, double den) {
    return (den > 0.0) ? (255.0 * num) / den : 0.0;  // host ×255 scale
}

double MahfouzCostReference(double contour_ratio, double intensity_ratio) {
    return contour_ratio * (-2.67) + intensity_ratio * (-1.0);
}

// ---------------------------------------------------------------------------
// IoU — IOU (src/compute/iou.cu). Union of the two bounding boxes, pixel-nonzero
// criterion (A>0 || B>0), pixels outside the union bbox excluded
// (region-restricted). Empty-bbox convention = the renderer's inverted reset
// state (w-1, h-1, 0, 0).
// Reference decision (flagged deviation): IOU(∅,∅) == 1.0 — identical empty
// images are identical; the kernel divides 0/0 -> NaN (iou.cu:82-83, no guard).
// The CPU reference IS the spec U2/U4 must keep in sync.
// ---------------------------------------------------------------------------

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
    // Kernel host union-bbox formulas (iou.cu), verbatim:
    const int left = std::max(std::min(ba[0], bb[0]), 0);
    const int bottom = std::max(std::min(ba[1], bb[1]), 0);
    const int right = std::min(std::max(ba[2], bb[2]), w - 1);
    const int top = std::min(std::max(ba[3], bb[3]), h - 1);
    const int cw = right - left + 1;
    const int ch = top - bottom + 1;
    if (cw <= 0 || ch <= 0) {
        // Degenerate crop: both bboxes empty (the only way the kernel's crop
        // inverts). Spec decision: 1.0 — flagged deviation from the kernel's
        // 0/0 NaN at iou.cu:82-83.
        return 1.0;
    }
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

// ---------------------------------------------------------------------------
// L1 — L_1_1_MatrixDifferenceNorm (src/compute/l_1_1_matrix_diff_norm.cu):
// Σ|A−B| over the union-bbox crop (NOT the full frame). Both-empty -> 0 (the
// kernel's counters stay 0; no deviation to flag here).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Stage-guard semantics (Bug 3, CostFunctionManager.cpp:46-48 — FIXED in U2).
// The production guard was a tautology: `if (stage_ != Stage::Trunk ||
// stage_ != Stage::Branch || stage_ != Stage::Leaf) stage_ = Stage::Trunk;` —
// no Stage value can equal all three members, so EVERY constructed manager
// collapsed to Trunk (the Branch/Leaf constructors at settings_control.cpp:
// 1198-1199 were affected). U2 rewrote `||` -> `&&` and added the minimal
// getStage() accessor (the documented second wizard-region exception); the
// observable pin below asserts the fixed behavior on the REAL manager. The
// pure boolean below locks the exact expression the fix rewrites.
// ---------------------------------------------------------------------------

bool StageGuardCollapsesToTrunkCorrected(Stage s) {
    return s != Stage::Trunk && s != Stage::Branch && s != Stage::Leaf;
}

// ---------------------------------------------------------------------------
// DD_NEW_POLE_CONSTRAINT PolePenalty (Bugs 1+2 spec).
// Bug 1: `double min_dist;` (DD_NEW_POLE_CONSTRAINT.cpp:134) is read
// uninitialized when no axis flag is set (the registry defaults X/Y/Z_TRANS are
// all false, CostFunctionManager.cpp:365-369) — UB. Spec: initialize min_dist =
// 0.0 and sum only enabled axes; the production fix (U2) lands this helper.
// Bug 2: Y_dist reuses the X-axis vector (lines 117-120: y_del* = s_*·Δ), so
// Y_TRANS silently duplicates X_TRANS. With identity rotation r = R·{0,1,0} =
// (0,1,0), s = R·{1,0,0} = (1,0,0): a pure y-offset Δ=(0,5,0) gives correct
// Y_dist = |r·Δ| = 5 but the buggy Y_dist = |s·Δ| = 0.
// ---------------------------------------------------------------------------

double PolePenalty(double x_dist, double y_dist, double z_dist, bool x_tran,
                   bool y_tran, bool z_tran, double weight) {
    double min_dist = 0.0;
    if (x_tran) min_dist += x_dist * weight;
    if (y_tran) min_dist += y_dist * weight;
    if (z_tran) min_dist += z_dist * weight;
    return min_dist;
}

}  // namespace

// ===========================================================================
// Chamfer stage pins
// ===========================================================================

TEST_CASE("chamfer: edge detection marks the 8-neighbor white/black boundary",
          "[metric_semantics][chamfer]") {
    // 5x5, white 3x3 block at rows 1-3 / cols 1-3.
    std::vector<unsigned char> img(25, BLACK_PIXEL);
    for (int y = 1; y <= 3; ++y)
        for (int x = 1; x <= 3; ++x) img[y * 5 + x] = WHITE_PIXEL;

    ChamferEdgeDetect(img, 5, 5);

    // Border of the white block -> EDGE; interior center untouched (WHITE);
    // black exterior untouched.
    REQUIRE(img[1 * 5 + 1] == EDGE_PIXEL);
    REQUIRE(img[1 * 5 + 2] == EDGE_PIXEL);
    REQUIRE(img[2 * 5 + 1] == EDGE_PIXEL);
    REQUIRE(img[2 * 5 + 3] == EDGE_PIXEL);
    REQUIRE(img[3 * 5 + 3] == EDGE_PIXEL);
    REQUIRE(img[2 * 5 + 2] == WHITE_PIXEL);  // interior never touched
    REQUIRE(img[0 * 5 + 0] == BLACK_PIXEL);  // exterior never touched
    REQUIRE(img[4 * 5 + 4] == BLACK_PIXEL);
}

TEST_CASE("chamfer: dilation is quadrant-only (axial cross never dilated)",
          "[metric_semantics][chamfer]") {
    // 3x3, center EDGE. Dilation=1 must write only the four DIAGONAL corners;
    // the axial neighbors (j==0 or k==0) stay BLACK — the spec quirk.
    std::vector<unsigned char> img(9, BLACK_PIXEL);
    img[1 * 3 + 1] = EDGE_PIXEL;

    ChamferDilate(img, 3, 3, 1);

    REQUIRE(img[0 * 3 + 0] == DILATED_PIXEL);
    REQUIRE(img[0 * 3 + 2] == DILATED_PIXEL);
    REQUIRE(img[2 * 3 + 0] == DILATED_PIXEL);
    REQUIRE(img[2 * 3 + 2] == DILATED_PIXEL);
    REQUIRE(img[0 * 3 + 1] == BLACK_PIXEL);  // axial: never dilated
    REQUIRE(img[1 * 3 + 0] == BLACK_PIXEL);
    REQUIRE(img[1 * 3 + 2] == BLACK_PIXEL);
    REQUIRE(img[2 * 3 + 1] == BLACK_PIXEL);
    REQUIRE(img[1 * 3 + 1] == EDGE_PIXEL);  // the source edge survives
}

TEST_CASE("chamfer: EDGE/DILATED pixels are never overwritten by dilation",
          "[metric_semantics][chamfer]") {
    // (1,1) and (2,2) both EDGE: (1,1)'s quadrant offsets include (2,2), which
    // must NOT be overwritten.
    std::vector<unsigned char> img(16, BLACK_PIXEL);
    img[1 * 4 + 1] = EDGE_PIXEL;
    img[2 * 4 + 2] = EDGE_PIXEL;

    ChamferDilate(img, 4, 4, 1);

    REQUIRE(img[2 * 4 + 2] == EDGE_PIXEL);  // EDGE never overwritten
    REQUIRE(img[0 * 4 + 0] == DILATED_PIXEL);
    // Idempotence: a second pass changes nothing (DILATED/EDGE targets are not
    // WHITE/BLACK).
    std::vector<unsigned char> again = img;
    ChamferDilate(again, 4, 4, 1);
    REQUIRE(again == img);
}

TEST_CASE("chamfer: difference is +1 on comparison-white, -1 otherwise, metric negates",
          "[metric_semantics][chamfer]") {
    // rendered (already processed form): (0,0) EDGE, (1,1) DILATED, (3,3) DILATED.
    std::vector<unsigned char> rendered(16, BLACK_PIXEL);
    rendered[0] = EDGE_PIXEL;
    rendered[1 * 4 + 1] = DILATED_PIXEL;
    rendered[3 * 4 + 3] = DILATED_PIXEL;
    // comparison: (0,0) WHITE -> +1; (1,1) BLACK -> -1; (3,3) WHITE -> +1.
    std::vector<unsigned char> comparison(16, BLACK_PIXEL);
    comparison[0] = WHITE_PIXEL;
    comparison[3 * 4 + 3] = WHITE_PIXEL;

    REQUIRE(ChamferDifferenceScore(rendered, comparison) == 1);
    // The metric returns -1 * score (fast_implant_dilation_metric.cu:289-290).
    REQUIRE(ChamferMetric(rendered, comparison, 4, 4, 1) == Approx(-1.0));

    // A pixel that is neither EDGE nor DILATED (WHITE/BLACK) is never counted.
    std::vector<unsigned char> rendered2 = rendered;
    rendered2[2 * 4 + 2] = WHITE_PIXEL;
    REQUIRE(ChamferDifferenceScore(rendered2, comparison) == 1);
}

TEST_CASE("chamfer: kernel-exact perfect-overlap composition value",
          "[metric_semantics][chamfer]") {
    // The inventory's "white_sum + chamfer => 0 at perfect overlap" is a
    // simplification that holds only for the EMPTY image. Kernel-as-spec (the
    // difference rule + the −1 factor): at perfect overlap the chamfer term is
    // exactly +|E| (every edge/dilated pixel hits a non-white comparison
    // pixel), so the composition is white_sum + |E| >= 0. Pinned here as the
    // characterization finding — the fix units must NOT "correct" this to 0.
    std::vector<unsigned char> empty(64, BLACK_PIXEL);
    REQUIRE(ChamferMetric(empty, empty, 8, 8, 1) == Approx(0.0));
    REQUIRE(ComputeSumWhitePixelsReference(empty) == 0);
    // composition == 0 for the empty image (the only literal-zero case):
    REQUIRE(ComputeSumWhitePixelsReference(empty) +
                ChamferMetric(empty, empty, 8, 8, 1) ==
            Approx(0.0));

    // Non-empty: 5x5 white block in a 9x9 image, dilation 1.
    std::vector<unsigned char> a(81, BLACK_PIXEL);
    for (int y = 2; y <= 6; ++y)
        for (int x = 2; x <= 6; ++x) a[y * 9 + x] = WHITE_PIXEL;
    std::vector<unsigned char> proc = a;
    ChamferEdgeDetect(proc, 9, 9);
    ChamferDilate(proc, 9, 9, 1);

    const int white_sum = ComputeSumWhitePixelsReference(proc);
    const int edge_count = CountEdgeOrDilated(proc);
    REQUIRE(ChamferMetric(proc, proc, 9, 9, 1) == Approx(edge_count));
    REQUIRE(ComputeSumWhitePixelsReference(proc) + ChamferMetric(proc, proc, 9, 9, 1) ==
            Approx(white_sum + edge_count));
}

// ===========================================================================
// Distance-map pins
// ===========================================================================

TEST_CASE("distance map: score is sum-of-distances over EDGE pixels / (count+0.1)",
          "[metric_semantics][distance_map]") {
    // 8x8: EDGE pixels at (2,3) [dist 10] and (5,6) [dist 20]; a DILATED pixel
    // at (1,1) with a huge distance that must NOT be sampled; white/black noise.
    std::vector<unsigned char> projected(64, BLACK_PIXEL);
    projected[3 * 8 + 2] = EDGE_PIXEL;
    projected[6 * 8 + 5] = EDGE_PIXEL;
    projected[1 * 8 + 1] = DILATED_PIXEL;
    projected[4 * 8 + 4] = WHITE_PIXEL;
    std::vector<unsigned char> distance_map(64, 255);
    distance_map[3 * 8 + 2] = 10;
    distance_map[6 * 8 + 5] = 20;
    // The DILATED pixel's distance stays 255: the composition quirk (only the
    // surviving EDGE ring is sampled).
    const std::array<int, 4> bbox = {0, 0, 7, 7};
    REQUIRE(DistanceMapMetricReference(projected, distance_map, 8, 8, bbox, 1) ==
            Approx(30.0 / 2.1));
}

TEST_CASE("distance map: the +0.1 singularity hack (zero-edge crop -> 0.0)",
          "[metric_semantics][distance_map]") {
    std::vector<unsigned char> no_edges(64, BLACK_PIXEL);
    std::vector<unsigned char> distance_map(64, 200);
    const std::array<int, 4> bbox = {0, 0, 7, 7};
    // 0 / (0 + 0.1) == 0.0 — never NaN; the hack is pinned as spec (flagged).
    REQUIRE(DistanceMapMetricReference(no_edges, distance_map, 8, 8, bbox, 1) ==
            Approx(0.0));
}

// ===========================================================================
// CropIndexToGlobal pins (Bug 5 — the ONE live kernel bug)
// ===========================================================================

TEST_CASE(
    "distance map: CORRECTED CropIndexToGlobal covers the oracle's 64x64 crop "
    "exactly once",
    "[metric_semantics][distance_map][index]") {
    // The oracle's actual crop/block geometry: 64x64 crop with threads_per_block
    // = 256 (16x16 blocks) -> gridDim = (4, 4) — read from the launch site
    // (distance_map_metric.cu:56-68) and cuda_launch_parameters.h at compile
    // time, not hardcoded.
    REQUIRE(IndexCoversCropExactlyOnce(CropIndexToGlobal, 64, 64,
                                       threads_per_block));
}

TEST_CASE(
    "distance map: BUGGY index formula fails full coverage (RED pre-fix, "
    "src/compute/distance_map_metric.cu:27)",
    "[metric_semantics][distance_map][index][red]") {
    // The formula as written today fails the bijectivity invariant — this is the
    // RED characterization pin; U4's one-line fix (blockIdx.y + gridDim.x ->
    // blockIdx.y * gridDim.x) is what makes the corrected pin above the kernel's
    // behavior. Do NOT change this test to match the bug.
    REQUIRE_FALSE(IndexCoversCropExactlyOnce(BuggyCropIndexToGlobal, 64, 64,
                                             threads_per_block));
    // Concrete failure mode (the plan's arithmetic): block (bx,by) maps to
    // global-thread base (by + gridDim.x + bx) * blockDim.x instead of
    // (by * gridDim.x + bx) * blockDim.x.
    //   - block (0,0) starts at 1024 -> crop pixels 0..1023 (rows 0..15) are
    //     NEVER visited,
    //   - blocks (0,1) and (1,0) collide on the same 256-thread range.
    REQUIRE(BuggyCropIndexToGlobal(0, 4, 0, threads_per_block, 0) ==
            4 * threads_per_block);  // 1024, not 0
    REQUIRE(BuggyCropIndexToGlobal(1, 4, 0, threads_per_block, 0) ==
            BuggyCropIndexToGlobal(0, 4, 1, threads_per_block, 0));  // collision
    REQUIRE(CropIndexToGlobal(1, 4, 0, threads_per_block, 0) !=
            CropIndexToGlobal(0, 4, 1, threads_per_block, 0));  // corrected: distinct
}

// ===========================================================================
// Mahfouz ratio pins (kernel-as-spec on the 2.55 / x255 / -2.67 / -1 chain)
// ===========================================================================

TEST_CASE("mahfouz: truncated-vs-float accumulation differs (2.55 scale, Bug 6)",
          "[metric_semantics][mahfouz][red]") {
    // Three 99-valued dilated pixels, all overlapping a non-black comparison.
    // The kernel sums (int)(2.55*99) = 252 per pixel (atomicAdd(int*, double)
    // truncates); the intended float math keeps 252.45. The quantization is
    // pinned as current behavior — kernel-as-spec on the 2.55 scale.
    std::vector<unsigned char> rendered(3, 99);
    std::vector<unsigned char> comparison(3, 200);

    REQUIRE(MahfouzContourNumeratorTruncated(rendered, comparison) == 3 * 252);
    REQUIRE(MahfouzContourNumeratorFloat(rendered, comparison) ==
            Approx(3 * 252.45));
    REQUIRE(MahfouzContourNumeratorTruncated(rendered, comparison) !=
            static_cast<long long>(MahfouzContourNumeratorFloat(rendered, comparison)));
}

TEST_CASE(
    "mahfouz: truncated-vs-float contour ratio pair differs measurably "
    "(x255 host scale)",
    "[metric_semantics][mahfouz][red]") {
    // Rendered {9, 9, 7}: comparison non-black for the two 9s, black for the 7
    // (the numerator's extra comparison!=BLACK condition vs the denominator).
    //   truncated: num = 2*trunc(22.95) = 44, den = 44 + trunc(17.85) = 61
    //   float:     num = 45.9,            den = 45.9 + 17.85 = 63.75
    // contour ratio (host ×255) differs by ~0.33 — measurably.
    std::vector<unsigned char> rendered = {9, 9, 7};
    std::vector<unsigned char> comparison = {200, 200, 0};

    const long long num_t = MahfouzContourNumeratorTruncated(rendered, comparison);
    const long long den_t = MahfouzContourDenominatorTruncated(rendered);
    const double num_f = MahfouzContourNumeratorFloat(rendered, comparison);
    const double den_f = MahfouzContourDenominatorFloat(rendered);
    REQUIRE(num_t == 44);
    REQUIRE(den_t == 61);
    REQUIRE(num_f == Approx(45.9));
    REQUIRE(den_f == Approx(63.75));

    const double ratio_t = MahfouzContourRatioReference(num_t, den_t);
    const double ratio_f = MahfouzContourRatioReference(num_f, den_f);
    REQUIRE(ratio_t == Approx(255.0 * 44.0 / 61.0));
    REQUIRE(ratio_f == Approx(255.0 * 45.9 / 63.75));
    REQUIRE(std::fabs(ratio_t - ratio_f) > 0.1);  // differs measurably

    // Full cost chain (-2.67 contour weight, -1 intensity weight): differs too.
    const double cost_t = MahfouzCostReference(ratio_t, 0.0);
    const double cost_f = MahfouzCostReference(ratio_f, 0.0);
    REQUIRE(cost_t == Approx(ratio_t * (-2.67)));
    REQUIRE(cost_f == Approx(ratio_f * (-2.67)));
    REQUIRE(std::fabs(cost_t - cost_f) > 0.5);
}

TEST_CASE("mahfouz: intensity ratio + value-guard semantics",
          "[metric_semantics][mahfouz]") {
    // Two rendered-white pixels with intensities 100 and 200 -> ratio 150.
    std::vector<unsigned char> rendered = {WHITE_PIXEL, WHITE_PIXEL, BLACK_PIXEL};
    std::vector<unsigned char> intensity = {100, 200, 255};
    REQUIRE(MahfouzIntensityRatioReference(rendered, intensity) == Approx(150.0));

    // Zero rendered-white pixels: the INTENDED value-guard semantics yield ratio
    // 0 (finite). The kernel's pointer guard always passes (pixel_score_ is
    // never null) -> 0/0 NaN at implant_mahfouz_metric.cu:324/446 — the
    // documented deviation the Bug-6 guard fix must close.
    std::vector<unsigned char> empty(3, BLACK_PIXEL);
    REQUIRE(MahfouzIntensityRatioReference(empty, intensity) == Approx(0.0));
    REQUIRE(MahfouzContourRatioReference(0, 0) == Approx(0.0));
}

// ===========================================================================
// IoU / L1 / white-sum pins
// ===========================================================================

TEST_CASE("iou: identical, disjoint, partial-overlap, and empty hand-computed cases",
          "[metric_semantics][iou]") {
    const int w = 8, h = 8;
    // A: cols 0-3 white; B: cols 2-5 white -> union bbox (0,0,5,7), A∩B = cols
    // 2-3 (16 px), union = 6*8 = 48 px -> 16/48 = 1/3.
    std::vector<unsigned char> A(static_cast<size_t>(w) * h, BLACK_PIXEL);
    std::vector<unsigned char> B(static_cast<size_t>(w) * h, BLACK_PIXEL);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (x <= 3) A[static_cast<size_t>(y) * w + x] = WHITE_PIXEL;
            if (x >= 2 && x <= 5) B[static_cast<size_t>(y) * w + x] = WHITE_PIXEL;
        }
    }
    REQUIRE(IoUReference(A, B, w, h) == Approx(16.0 / 48.0));
    REQUIRE(IoUReference(B, A, w, h) == Approx(16.0 / 48.0));  // symmetry
    REQUIRE(IoUReference(A, A, w, h) == Approx(1.0));          // reflexivity

    // Disjoint: A cols 0-3, D cols 4-7 (both non-empty).
    std::vector<unsigned char> D(static_cast<size_t>(w) * h, BLACK_PIXEL);
    for (int y = 0; y < h; ++y)
        for (int x = 4; x < w; ++x) D[static_cast<size_t>(y) * w + x] = 200;
    REQUIRE(IoUReference(A, D, w, h) == Approx(0.0));

    // IOU(∅,∅) == 1.0 — the spec-with-flagged-deviation decision (identical
    // empties are identical). The kernel yields 0/0 NaN (iou.cu:82-83).
    std::vector<unsigned char> empty(static_cast<size_t>(w) * h, BLACK_PIXEL);
    REQUIRE(IoUReference(empty, empty, w, h) == Approx(1.0));
    // IOU(∅, non-empty) == 0.0 via the union-bbox math.
    REQUIRE(IoUReference(empty, A, w, h) == Approx(0.0));
}

TEST_CASE("l1: hand-computed values, symmetry, self-zero",
          "[metric_semantics][l1]") {
    const int w = 4, h = 4;
    std::vector<unsigned char> A(static_cast<size_t>(w) * h, BLACK_PIXEL);
    std::vector<unsigned char> B(static_cast<size_t>(w) * h, BLACK_PIXEL);
    // A: row 1 = {10, 20, 30, 40}; B: row 1 = {0, 30, 30, 40}, row 2 = {5,5,5,5}.
    for (int x = 0; x < w; ++x) A[1 * w + x] = static_cast<unsigned char>(10 * (x + 1));
    B[1 * w + 0] = 0;
    B[1 * w + 1] = 30;
    B[1 * w + 2] = 30;
    B[1 * w + 3] = 40;
    for (int x = 0; x < w; ++x) B[2 * w + x] = 5;
    // union bbox: A rows 1..1 -> (0,1,3,1); B rows 1..2 -> (0,1,3,2) => (0,1,3,2).
    // crop = rows 1-2, cols 0-3. |A-B| row1: 10+10+0+0=20; row2: 5*4=20 => 40.
    REQUIRE(L1Reference(A, B, w, h) == 40);
    REQUIRE(L1Reference(B, A, w, h) == 40);  // symmetry
    REQUIRE(L1Reference(A, A, w, h) == 0);   // L1(A,A)=0
    REQUIRE(L1Reference(A, B, w, h) >= 0);   // non-negativity
}

TEST_CASE("compute-sum-white-pixels: counts only WHITE(255)",
          "[metric_semantics][white_sum]") {
    std::vector<unsigned char> img = {WHITE_PIXEL, BLACK_PIXEL, EDGE_PIXEL,
                                      DILATED_PIXEL, 200, WHITE_PIXEL};
    REQUIRE(ComputeSumWhitePixelsReference(img) == 2);
}

// ===========================================================================
// Dilation registry/constants pins
// ===========================================================================

TEST_CASE(
    "settings_constants.h is the verbatim Flood tibia transcription (budgets, "
    "ranges, dilation lineage)",
    "[metric_semantics][dilation]") {
    // The values below ARE the tibia's trunk/branch/leaf shape; settings_constants.h
    // is their in-repo form (transcription pin). The budget shape matches
    // test/golden/baseline.json budget_cumulative_effective exactly.
    REQUIRE(TRUNK_BUDGET == 20000);
    REQUIRE(BRANCH_BUDGET == 5000);
    REQUIRE(NUMBER_BRANCHES == 2);
    REQUIRE(Z_SEARCH_BUDGET == 5000);

    const bool trunk_range_ok = TRUNK_RANGE.x == 35 && TRUNK_RANGE.y == 35 &&
                                 TRUNK_RANGE.z == 35 && TRUNK_RANGE.xa == 35 &&
                                 TRUNK_RANGE.ya == 35 && TRUNK_RANGE.za == 35;
    REQUIRE(trunk_range_ok);
    const bool branch_range_ok = BRANCH_RANGE.x == 15 && BRANCH_RANGE.y == 15 &&
                                 BRANCH_RANGE.z == 25 && BRANCH_RANGE.xa == 25 &&
                                 BRANCH_RANGE.ya == 25 && BRANCH_RANGE.za == 25;
    REQUIRE(branch_range_ok);
    const bool z_range_ok = Z_SEARCH_RANGE.x == 3 && Z_SEARCH_RANGE.y == 3 &&
                            Z_SEARCH_RANGE.z == 15 && Z_SEARCH_RANGE.xa == 3 &&
                            Z_SEARCH_RANGE.ya == 3 && Z_SEARCH_RANGE.za == 3;
    REQUIRE(z_range_ok);

    // Cumulative caps: 20000 / 25000 / 30000 / 35000 (baseline.json).
    const int caps[4] = {TRUNK_BUDGET,
                         TRUNK_BUDGET + BRANCH_BUDGET,
                         TRUNK_BUDGET + 2 * BRANCH_BUDGET,
                         TRUNK_BUDGET + 2 * BRANCH_BUDGET + Z_SEARCH_BUDGET};
    REQUIRE(caps[0] == 20000);
    REQUIRE(caps[1] == 25000);
    REQUIRE(caps[2] == 30000);
    REQUIRE(caps[3] == 35000);

    // Engine runtime dilation is 6/4/1 (the code path the oracle executes):
    // trunk = DIRECT_DILATION registry default 6 (CostFunctionManager.cpp:412),
    // branch = TRUNK_DILATION - BRANCH_DILATION_DECREASE = 4 (the explicit
    // branch/leaf override in settings_control.cpp:1201-1206 sets 4 and 1),
    // leaf = Z_SEARCH_DILATION = 1.
    REQUIRE(TRUNK_DILATION == 6);
    REQUIRE(BRANCH_DILATION_DECREASE == 2);
    REQUIRE(Z_SEARCH_DILATION == 1);
    REQUIRE(TRUNK_DILATION - BRANCH_DILATION_DECREASE == 4);
    // baseline.json's dilation_px {6,3,1} is the known-stale docs-claim; the
    // ENGINE runtime value is 6/4/1. Reconcile after U5's probe data (hygiene
    // pass) — this pin asserts the ENGINE value.
}

// ===========================================================================
// Stage-guard semantics pins (Bug 3)
// ===========================================================================

TEST_CASE(
    "stage guard: getStage() reports the stage the caller constructed (U2)",
    "[metric_semantics][stage_guard]") {
    // Bug 3 fix (CostFunctionManager.cpp:46-48): the guard was the tautology
    // `s != Trunk || s != Branch || s != Leaf`, so every manager collapsed to
    // Trunk (previously the accessor would always report Trunk). U2 rewrote it
    // as `&&` and added the minimal getStage() accessor (the documented second
    // wizard-region exception); this observable pin asserts the fixed behavior
    // on the REAL CostFunctionManager linked from jtml_compute.
    jta_cost_function::CostFunctionManager trunk(Stage::Trunk);
    jta_cost_function::CostFunctionManager branch(Stage::Branch);
    jta_cost_function::CostFunctionManager leaf(Stage::Leaf);
    REQUIRE(trunk.getStage() == Stage::Trunk);
    REQUIRE(branch.getStage() == Stage::Branch);
    REQUIRE(leaf.getStage() == Stage::Leaf);
}

TEST_CASE(
    "stage guard: the corrected AND-guard preserves every valid stage (U2 spec)",
    "[metric_semantics][stage_guard]") {
    // U2's one-line fix (`||` -> `&&` at CostFunctionManager.cpp:46) makes the
    // guard force Trunk only for invalid values — never for a valid stage. The
    // observable counterpart (construct CostFunctionManager(Stage::Branch);
    // REQUIRE(getStage() == Stage::Branch)) landed with the minimal getStage()
    // accessor in the same unit (see the pin above); this pure-semantics pin
    // locks the boolean the fix rewrites.
    REQUIRE_FALSE(StageGuardCollapsesToTrunkCorrected(Stage::Trunk));
    REQUIRE_FALSE(StageGuardCollapsesToTrunkCorrected(Stage::Branch));
    REQUIRE_FALSE(StageGuardCollapsesToTrunkCorrected(Stage::Leaf));
}

// ===========================================================================
// DD_NEW_POLE_CONSTRAINT PolePenalty pins (Bugs 1+2)
// ===========================================================================

TEST_CASE("DD PolePenalty: init-0 spec (Bug 1) + per-axis accumulation",
          "[metric_semantics][dd_pole]") {
    // Bug 1 spec: no flags -> exactly 0 contribution (the production `double
    // min_dist;` was read uninitialized (UB) until U2 initialized min_dist =
    // 0.0 — this helper IS the production semantics).
    REQUIRE(PolePenalty(1.0, 2.0, 3.0, false, false, false, 75.0) == 0.0);
    REQUIRE(PolePenalty(1.0, 2.0, 3.0, true, false, false, 75.0) == 75.0);
    REQUIRE(PolePenalty(1.0, 2.0, 3.0, false, true, false, 75.0) == 150.0);
    REQUIRE(PolePenalty(1.0, 2.0, 3.0, false, false, true, 75.0) == 225.0);
    REQUIRE(PolePenalty(1.0, 2.0, 3.0, true, true, true, 10.0) == 60.0);
}

TEST_CASE("DD PolePenalty: Y-axis spec (Bug 2) + identity-vector characterization",
          "[metric_semantics][dd_pole]") {
    // Bug 2 spec: a pure y-offset must penalize via the non-principal y axis.
    // U2's production fix (DD_NEW_POLE_CONSTRAINT.cpp:117-120) now projects
    // Y_dist onto r = R·{0,1,0} instead of reusing the x-axis vector s.
    REQUIRE(PolePenalty(0.0, 5.0, 0.0, false, true, false, 1.0) == 5.0);

    // Characterization on the vectors (identity rotation): r = R·{0,1,0} =
    // (0,1,0), s = R·{1,0,0} = (1,0,0); pure y-offset Δ = (0,5,0).
    //   correct Y_dist = |r·Δ| = 5
    //   buggy Y_dist   = |s·Δ| = 0  (pre-U2 lines 117-120 reused the s vector)
    const double correct_y_dist = std::fabs(0.0 * 0.0 + 1.0 * 5.0 + 0.0 * 0.0);
    const double buggy_y_dist = std::fabs(1.0 * 0.0 + 0.0 * 5.0 + 0.0 * 0.0);
    REQUIRE(correct_y_dist == 5.0);
    REQUIRE(buggy_y_dist == 0.0);
    // Pre-U2 Y_dist == X_dist by construction; the fix makes them differ.
    REQUIRE(correct_y_dist != buggy_y_dist);
}

// ===========================================================================
// sym_trap tibia-transform pins (Bug 4 — FIXED in U2)
// ===========================================================================

TEST_CASE(
    "sym_trap: tibia x-translation slot carries tibia x (Bug 4 fixed)",
    "[metric_semantics][sym_trap]") {
    // costFunctionsym_trap_function (sym_trap_function.cpp:104-112) builds the
    // tibia transform; the x slot must carry p.x_location_ (pre-U2 it received
    // p.z_location_ — the x slot duplicated z, collapsing the tibia's x offset;
    // the trap-analysis prerequisite). This pin replicates the POST-fix call
    // verbatim and asserts the spec. Was RED via [!mayfail] pre-U2:
    // x2tib[0][3] == 60 (pose.z), not 50 (pose.x).
    const Point6D pose(50, 20, 60, 15, -8, 30);
    float x2tib[4][4];
    // verbatim from costFunctionsym_trap_function (post-fix source truth):
    create_312_transform(x2tib, pose.x, pose.y, pose.z, pose.za, pose.xa, pose.ya);
    REQUIRE(x2tib[0][3] == Approx(pose.x));
    REQUIRE(x2tib[2][3] == Approx(pose.z));
}

TEST_CASE(
    "sym_trap: fem2tib relative translation column (Bug 4 fixed)",
    "[metric_semantics][sym_trap]") {
    // Femur at origin / tibia (10,0,0), identity angles. The relative
    // translation column of fem2tib = fem2x·x2tib is R_fem^T·t_tib. The
    // identity-angle 312 rotation part is the x↔z swap matrix
    // [[0,0,1],[0,1,0],[1,0,0]] (kernel-as-spec: create_312_transform at
    // zero angles yields transform[0][0] = cy*sx*sz - cz*sy = 0), so the
    // tibia x offset maps into the femur frame's Z slot:
    //   pre-U2:  t_tib = (0,0,0)  -> column (0,0,0)   (x offset dropped)
    //   post-U2: t_tib = (10,0,0) -> column (0,0,10) (x offset visible)
    // The U1 pin's draft expectation (10,0,0) assumed an identity rotation
    // part, which the kernel does not produce — corrected to the kernel-exact
    // composition per kernel-as-spec (approved U2 deviation, recorded in the
    // plan's commit history).
    const Point6D tibia(10, 0, 0, 0, 0, 0);
    const Point6D femur(0, 0, 0, 0, 0, 0);
    float x2tib[4][4], x2fem[4][4], fem2x[4][4], fem2tib[4][4];
    // verbatim from costFunctionsym_trap_function (post-fix source truth):
    create_312_transform(x2tib, tibia.x, tibia.y, tibia.z, tibia.za, tibia.xa,
                         tibia.ya);
    create_312_transform(x2fem, femur.x, femur.y, femur.z, femur.za, femur.xa,
                         femur.ya);
    invert_transformation(fem2x, x2fem);
    matmult(fem2tib, fem2x, x2tib);
    REQUIRE(fem2tib[0][3] == Approx(0.0));
    REQUIRE(fem2tib[1][3] == Approx(0.0));
    REQUIRE(fem2tib[2][3] == Approx(10.0));
}
