// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Tier-2 z-profile probe (plan 008 U5). NOT in the headless default suite —
// GPU-only, run explicitly on a GPU machine under the `oracle` label:
//   ctest --test-dir .build -L oracle -R z_profile
//   (or .build/bin/jtml_test_z_profile)
//
// Instrument: the cheapest thing that sees the z-weak axis. A 31-render Δz
// sweep (±15 mm, step 1) at fixed truth (fem.jts start pose per frame, other
// 5 axes pinned) over the SAME injected-cost lambda production uses — the
// oracle-twin pattern (oracle_test.cpp BuildFramePipeline + cost lambda),
// with a term-decomposition mode: at each sweep point, full / chamfer-only /
// dilated-overlap-only costs are read from ONE render (zero extra renders —
// the DIRECT_DILATION composition white_sum + FastImplantDilationMetric +
// DistanceMapMetric is exactly costFunctionDIRECT_DILATION's body, same call
// order: FIDM edge-marks the rendered image in place, then DMM counts those
// EDGE pixels).
//
// Metrics per curve (angle 03 R2-2 / angle 06 R2-1): valley_depth =
// (max − cost(0)) / (max − min) ∈ [0,1] (healthy V → ~1, flat → ~0);
// argmin_offset_mm; plateau_halfwidth_mm (operationalization below);
// noise_floor = spread of cost(0) over 5 repeats (expect 0 — the int-atomic
// path is value-deterministic; angle 05's verified finding).
//
// Assertions (run 1 = RECORD mode; pin-first: later runs enforce once the
// z_profiles block exists in test/golden/baseline.json):
//   p1 min-at-zero within noise (recorded band |argmin| ≤ 2 mm — 1 mm grid
//      + the oracle's known frame-0 z-offset of −0.89 mm)
//   p2 valley_depth < 0.05 ⇒ z-blind (recorded, informational; classifier
//      logic asserted on synthetic flat/V curves)
//   p3 ranking vd(DIRECT_DILATION) ≥ vd(DIRECT_MAHFOUZ) ≥ vd(DIRECT_DILATION_T1)
//      at (dilation 6, frame 0): run 1 RECORDS, later runs assert against the
//      recorded baseline block
//   p4 z_gap anti-correlates with valley_depth (recorded)
//
// Sweep set: DIRECT_DILATION at dilations {6,3,1,4,10,2} (the union of the
// plan's {6,3,1,4} + lineage sets S1 {6,4,1}, S2 {6,3,1}, S3 {10,6,1},
// S4 {6,2,1}) × frames {0,1,2}; DIRECT_MAHFOUZ + DIRECT_DILATION_T1 at
// (dilation 6, frame 0) for the p3 ranking (T1's trunk reachable path
// hardcodes rendered-dilation 1 — the always-Trunk guard, angle 02 bug 4 —
// flagged in the manifest); coupling mode (in-plane ±2 mm proxy for the
// perturbation suite's minimal-detectable-regression, recorded) at dilations
// {6,1} frame 0; yamazaki-polynomial mode (order-4 fit on the same 31
// renders: poly_argmin + fit residual).
//
// The probe writes test/golden/z_profiles.json (full machine record incl. the
// raw cost curves) and prints the condensed z_profiles block for
// test/golden/baseline.json. Run 1 records; later runs read the baseline
// block and enforce the ranking + valley bands.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cuda_runtime.h>

#include <opencv2/imgcodecs.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "domain/data_structures_6D.h"
#include "compute/frame.h"
#include "services/model.h"
#include "services/calibration.h"

#include "compute/CostFunctionManager.h"

#include "compute/camera_calibration.h"
#include "compute/gpu_metrics.cuh"
#include "compute/gpu_model.cuh"
#include "compute/pose_matrix.h"

using gpu_cost_function::Pose;
using gpu_cost_function::GPUEdgeFrame;
using gpu_cost_function::GPUDilatedFrame;
using gpu_cost_function::GPUIntensityFrame;
using gpu_cost_function::GPUFrame;
using gpu_cost_function::GPUHeatmap;
using gpu_cost_function::GPUImage;
using gpu_cost_function::GPUModel;
using gpu_cost_function::GPUMetrics;

namespace {

const std::string kStudyDir = "example_studies/Kneel_1/";
const std::vector<std::string> kBaseImages = {
    kStudyDir + "1024/2806.tif",
    kStudyDir + "1024/2807.tif",
    kStudyDir + "1024/2808.tif"};
const std::string kFemStl = kStudyDir + "KR_right_7_fem.stl";

const int kWidth = 1024;
const int kHeight = 1024;
const int kDevice = 0;

// fem.jts per-frame start poses (the fixed-truth sweep anchor — the golden
// pose, from baseline.json start_pose_per_frame / oracle_test.cpp StartPoses).
std::vector<Point6D> StartPoses() {
    return {
        Point6D(18.52191, 19.69514, -1027.713, -7.419319, -0.2587041,
                -26.69708),  // frame 0 (2806.tif)
        Point6D(19.01747, 20.15555, -1026.732, -7.56846, -0.2358893,
                -27.37827),  // frame 1 (2807.tif)
        Point6D(16.4709, 16.4248, -1028.69, -7.678545, 0.3264978,
                -24.12223),  // frame 2 (2808.tif)
    };
}

// Flatten a single-channel grayscale cv::Mat into a contiguous uchar buffer
// (used for GPU frame uploads from the Frame's processed images).
std::vector<unsigned char> MatToUchar(const cv::Mat& m) {
    std::vector<unsigned char> buf((size_t)m.rows * m.cols);
    for (int y = 0; y < m.rows; ++y) {
        const unsigned char* row = m.ptr<unsigned char>(y);
        std::copy(row, row + m.cols, buf.begin() + (size_t)y * m.cols);
    }
    return buf;
}

struct Pipeline {
    GPUModel* model = nullptr;
    GPUMetrics* metrics = nullptr;
    PoseMatrix* pose_storage = nullptr;
    std::vector<GPUEdgeFrame*> edge_a;
    std::vector<GPUDilatedFrame*> dilated_a;
    std::vector<GPUIntensityFrame*> intensity_a;
    std::vector<GPUFrame*> distance_maps;
    std::vector<GPUHeatmap*> heatmaps;
    std::vector<GPUModel*> non_principal;
    jta_cost_function::CostFunctionManager* trunk = nullptr;

    ~Pipeline() {
        delete trunk;
        for (auto* p : edge_a) delete p;
        for (auto* p : dilated_a) delete p;
        for (auto* p : intensity_a) delete p;
        for (auto* p : distance_maps) delete p;
        for (auto* p : heatmaps) delete p;
        delete pose_storage;
        delete metrics;
        delete model;
    }
};

Pose ToPose(const Point6D& p) {
    return Pose(p.x, p.y, p.z, p.xa, p.ya, p.za);
}

// Monoplane GPU pipeline for one base frame at the given dilation (mirrors
// OptimizerManager::Initialize + the oracle's BuildFramePipeline): uploads the
// processed Frame outputs (edge / dilation / intensity / distance-map /
// curvature heatmaps) and wires a trunk DIRECT_DILATION cost manager with the
// Dilation parameter set. The comparison dilated frame is dilated at the same
// value, exactly like the per-stage re-dilate in production. The caller owns
// the returned Pipeline.
Pipeline BuildFramePipeline(const std::string& base_image, int dilation) {
    Frame frame(base_image, 3, 0, 150, dilation);
    frame.setCurvatureHeatmaps();

    Pipeline p;
    p.metrics = new GPUMetrics();
    REQUIRE(p.metrics->IsInitializedCorrectly());
    p.pose_storage = new PoseMatrix();

    auto edge_upload = MatToUchar(frame.GetEdgeImage());
    auto edge = new GPUEdgeFrame(
        kWidth, kHeight, kDevice, edge_upload.data(),
        frame.GetHighThreshold(), frame.GetLowThreshold(), frame.GetAperture());
    REQUIRE(edge->IsInitializedCorrectly());
    p.edge_a.push_back(edge);

    auto dil_upload = MatToUchar(frame.GetDilationImage());
    auto dilated = new GPUDilatedFrame(kWidth, kHeight, kDevice,
                                       dil_upload.data(), dilation);
    REQUIRE(dilated->IsInitializedCorrectly());
    p.dilated_a.push_back(dilated);

    auto orig_upload = MatToUchar(frame.GetOriginalImage());
    auto inv_upload = MatToUchar(frame.GetInvertedImage());
    auto intensity = new GPUIntensityFrame(kWidth, kHeight, kDevice,
                                           orig_upload.data(), false,
                                           inv_upload.data());
    REQUIRE(intensity->IsInitializedCorrectly());
    p.intensity_a.push_back(intensity);

    auto dist_upload = MatToUchar(frame.GetDistanceMap());
    auto dm = new GPUFrame(kWidth, kHeight, kDevice, dist_upload.data());
    REQUIRE(dm->IsInitializedCorrectly());
    p.distance_maps.push_back(dm);

    auto hm = new GPUHeatmap(kWidth, kHeight, kDevice,
                             frame.GetNumCurvatureKeypoints(),
                             frame.getCurvatureHeatmaps().data());
    REQUIRE(hm->IsInitializedCorrectly());
    p.heatmaps.push_back(hm);

    Model femur(kFemStl, "femur", "femur");
    REQUIRE(femur.initialized_correctly_);
    int triangle_count =
        static_cast<int>(femur.triangle_vertices_.size() / 9);
    REQUIRE(triangle_count > 0);

    CameraCalibration cam(1198.0f, -1.0f * 0.0f, -1.0f * 0.0f, 0.373f);
    Calibration calib(cam);
    p.model = new GPUModel("femur", /*principal=*/true, kWidth, kHeight,
                           kDevice, /*use_backface_culling=*/false,
                           &femur.triangle_vertices_[0],
                           &femur.triangle_normals_[0], triangle_count,
                           calib.camera_A_principal_);
    REQUIRE(p.model->IsInitializedCorrectly());

    p.trunk = new jta_cost_function::CostFunctionManager(Stage::Trunk);
    p.trunk->setActiveCostFunction("DIRECT_DILATION");
    // Production path for the per-stage Dilation param (SettingsBridge.cpp:586,
    // mainscreen.cpp:4898-4900): setIntParameterValue on the ACTIVE class.
    // NOTE (probe finding, recorded in the run's z_profiles manifest): the
    // wizard-era CostFunctionManager::updateCostFunctionParameterValues int
    // overload is a silent NO-OP — getIntParameters() returns the vector BY
    // VALUE, so the write lands on a copy. The oracle fixture only worked
    // because the registered default is Dilation=6 and the oracle never
    // re-parameterizes; production never hits it because it uses
    // setIntParameterValue directly. The probe uses the production path so
    // the dilation axis actually takes effect.
    REQUIRE(p.trunk->getActiveCostFunctionClass()->setIntParameterValue(
        "Dilation", dilation));
    p.trunk->UploadData(&p.edge_a, &p.dilated_a, &p.intensity_a, &p.edge_a,
                        &p.dilated_a, &p.intensity_a, p.model,
                        &p.non_principal, p.metrics, p.pose_storage,
                        /*biplane=*/false);
    p.trunk->UploadDistanceMap(&p.distance_maps, &p.heatmaps);
    p.trunk->setCurrentFrameIndex(0);
    return p;
}

// ---------------------------------------------------------------------------
// Valley-curve metrics (angle 03 R2-2). Operationalizations:
//   valley_depth        = (max − cost(0)) / (max − min); 0.0 when the curve is
//                         flat (max−min below epsilon) or cost(0) ≥ max.
//   argmin_offset_mm    = dz at the sweep minimum (ties → closest to 0).
//   plateau_halfwidth   = half-width of the largest interval around the
//                         argmin where cost ≤ min + 0.9·(max−min) — the valley
//                         width at 90% depth; the full sweep (±15) when the
//                         curve never exits the band (z-blind signature).
// ---------------------------------------------------------------------------
struct ValleyMetrics {
    double valley_depth = 0.0;
    double argmin_offset_mm = 0.0;
    double plateau_halfwidth_mm = 0.0;
};

ValleyMetrics ComputeValley(const std::vector<double>& cost,
                            const std::vector<double>& dz) {
    REQUIRE(cost.size() == dz.size());
    REQUIRE(cost.size() == 31);
    ValleyMetrics m;
    auto min_it = std::min_element(cost.begin(), cost.end());
    auto max_it = std::max_element(cost.begin(), cost.end());
    double min_v = *min_it;
    double max_v = *max_it;
    double cost0 = cost[15];  // dz == 0 is the center of the 31-point sweep

    if (max_v - min_v < 1e-9) {
        m.valley_depth = 0.0;  // flat → z-blind
    } else {
        m.valley_depth = (max_v - cost0) / (max_v - min_v);
        m.valley_depth = std::max(0.0, std::min(1.0, m.valley_depth));
    }

    // Argmin: closest-to-zero among the minima.
    int best = 0;
    for (size_t i = 0; i < cost.size(); ++i) {
        if (cost[i] < cost[best] - 1e-12) {
            best = static_cast<int>(i);
        } else if (std::abs(cost[i] - cost[best]) <= 1e-12 &&
                   std::abs(dz[i]) < std::abs(dz[best])) {
            best = static_cast<int>(i);
        }
    }
    m.argmin_offset_mm = dz[best];

    // Plateau halfwidth at 90% depth: largest contiguous interval containing
    // the argmin where cost ≤ min + 0.9·(max−min).
    const double band = min_v + 0.9 * (max_v - min_v);
    int lo = best, hi = best;
    while (lo > 0 && cost[lo - 1] <= band + 1e-12) --lo;
    while (hi < (int)cost.size() - 1 && cost[hi + 1] <= band + 1e-12) ++hi;
    double left_mm = dz[lo];
    double right_mm = dz[hi];
    if (lo > 0) {  // interpolate the crossing between lo−1 and lo
        const double c0 = cost[lo - 1], c1 = cost[lo];
        const double d0 = dz[lo - 1], d1 = dz[lo];
        if (c1 - c0 > 1e-12)
            left_mm = d0 + (band - c0) * (d1 - d0) / (c1 - c0);
    }
    if (hi < (int)cost.size() - 1) {
        const double c0 = cost[hi], c1 = cost[hi + 1];
        const double d0 = dz[hi], d1 = dz[hi + 1];
        if (c1 - c0 > 1e-12)
            right_mm = d0 + (band - c0) * (d1 - d0) / (c1 - c0);
    }
    m.plateau_halfwidth_mm = 0.5 * (right_mm - left_mm);
    m.plateau_halfwidth_mm = std::max(0.0, m.plateau_halfwidth_mm);
    return m;
}

// p2 classifier: valley_depth < 0.05 ⇒ z-blind (recorded, informational).
bool ClassifyZBlind(double valley_depth) { return valley_depth < 0.05; }

// ---------------------------------------------------------------------------
// Yamazaki-polynomial mode (angle 06 R3-4): order-4 least-squares fit of
// c(z) ≈ a0 + a1 z + a2 z² + a3 z³ + a4 z⁴ on the same 31 renders; report the
// fit's argmin over [−15, 15] and the normalized RMS residual.
// ---------------------------------------------------------------------------
struct PolyFit4 {
    double a[5] = {0, 0, 0, 0, 0};
    double argmin_mm = 0.0;
    double normalized_rms = 0.0;
};

PolyFit4 FitPoly4(const std::vector<double>& z,
                  const std::vector<double>& cost) {
    REQUIRE(z.size() == cost.size());
    const int n = static_cast<int>(z.size());

    // Normal equations: A^T A c = A^T y, A row = [1, z, z², z³, z⁴].
    double ata[5][5] = {{0}};
    double aty[5] = {0};
    for (int i = 0; i < n; ++i) {
        double p[5] = {1.0, z[i], z[i] * z[i], z[i] * z[i] * z[i],
                       z[i] * z[i] * z[i] * z[i]};
        for (int r = 0; r < 5; ++r) {
            aty[r] += p[r] * cost[i];
            for (int c = 0; c < 5; ++c) ata[r][c] += p[r] * p[c];
        }
    }
    // Gaussian elimination with partial pivoting.
    double a[5][6];
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 5; ++c) a[r][c] = ata[r][c];
    for (int r = 0; r < 5; ++r) a[r][5] = aty[r];
    for (int col = 0; col < 5; ++col) {
        int piv = col;
        for (int r = col + 1; r < 5; ++r)
            if (std::abs(a[r][col]) > std::abs(a[piv][col])) piv = r;
        if (piv != col)
            for (int c = 0; c < 6; ++c) std::swap(a[col][c], a[piv][c]);
        if (std::abs(a[col][col]) < 1e-15) continue;  // degenerate (flat)
        for (int r = 0; r < 5; ++r) {
            if (r == col) continue;
            double f = a[r][col] / a[col][col];
            for (int c = col; c < 6; ++c) a[r][c] -= f * a[col][c];
        }
    }
    PolyFit4 fit;
    for (int r = 0; r < 5; ++r)
        fit.a[r] = (std::abs(a[r][r]) > 1e-15) ? a[r][5] / a[r][r] : 0.0;

    auto eval = [&fit](double zz) {
        return fit.a[0] + fit.a[1] * zz + fit.a[2] * zz * zz +
               fit.a[3] * zz * zz * zz + fit.a[4] * zz * zz * zz * zz;
    };

    // Argmin over [−15, 15] by fine scan (0.01 mm) + local refinement.
    double best_z = 0.0, best_v = eval(0.0);
    for (double zz = -15.0; zz <= 15.0001; zz += 0.01) {
        double v = eval(zz);
        if (v < best_v) {
            best_v = v;
            best_z = zz;
        }
    }
    fit.argmin_mm = best_z;

    // Normalized RMS residual: sqrt(mean((cost−fit)²)) / (max−min).
    double min_c = *std::min_element(cost.begin(), cost.end());
    double max_c = *std::max_element(cost.begin(), cost.end());
    double sse = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = cost[i] - eval(z[i]);
        sse += d * d;
    }
    double rms = std::sqrt(sse / n);
    fit.normalized_rms =
        (max_c - min_c > 1e-12) ? rms / (max_c - min_c) : 0.0;
    return fit;
}

// ---------------------------------------------------------------------------
// Model COG + focus→COG off-axis angle vs camera-z (recorded in the manifest).
// World COG = R(pose)·COG_model + t (RzRxRy per RenderEngine::SetPose); the
// focus (X-ray source) is at the world origin, so the ray direction is the
// world COG itself; camera-z is the z axis.
// ---------------------------------------------------------------------------
struct GeometryInfo {
    double off_axis_angle_deg = 0.0;
};

GeometryInfo ComputeGeometry(const Point6D& anchor) {
    Model femur(kFemStl, "femur", "femur");
    REQUIRE(femur.initialized_correctly_);
    const auto& v = femur.triangle_vertices_;
    REQUIRE(v.size() % 9 == 0);
    double cx = 0, cy = 0, cz = 0;
    size_t n = v.size() / 3;
    for (size_t i = 0; i + 2 < v.size(); i += 3) {
        cx += v[i];
        cy += v[i + 1];
        cz += v[i + 2];
    }
    cx /= n;
    cy /= n;
    cz /= n;

    // RzRxRy per RenderEngine::SetPose.
    double czz = std::cos(anchor.za * M_PI / 180.0);
    double sz = std::sin(anchor.za * M_PI / 180.0);
    double cxx = std::cos(anchor.xa * M_PI / 180.0);
    double sx = std::sin(anchor.xa * M_PI / 180.0);
    double cyy = std::cos(anchor.ya * M_PI / 180.0);
    double sy = std::sin(anchor.ya * M_PI / 180.0);
    double r00 = czz * cyy - sz * sx * sy;
    double r01 = -sz * cxx;
    double r02 = czz * sy + sz * cyy * sx;
    double r10 = sz * cyy + czz * sx * sy;
    double r11 = czz * cxx;
    double r12 = sz * sy - czz * cyy * sx;
    double r20 = -cxx * sy;
    double r21 = sx;
    double r22 = cxx * cyy;

    double wx = r00 * cx + r01 * cy + r02 * cz + anchor.x;
    double wy = r10 * cx + r11 * cy + r12 * cz + anchor.y;
    double wz = r20 * cx + r21 * cy + r22 * cz + anchor.z;

    GeometryInfo g;
    g.off_axis_angle_deg =
        std::atan2(std::hypot(wx, wy), std::abs(wz)) * 180.0 / M_PI;
    return g;
}

// ---------------------------------------------------------------------------
// Tiny JSON helpers (the probe's record is its own schema; no external JSON
// library in the repo).
// ---------------------------------------------------------------------------
std::string JsonNum(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

std::string JsonArr(const std::vector<double>& xs) {
    std::string s = "[";
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) s += ", ";
        s += JsonNum(xs[i]);
    }
    s += "]";
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// Pin-first enforcement (later runs): when baseline.json already carries the
// recorded z_profiles block (this probe's run-1 data event), assert the fresh
// ranking order and the per-variant valley depths at (d6, f0) stay within the
// recorded band. Run 1 (no block yet) only RECORDS — the plan's doctrine.
// The extractor is a tiny scanner for THIS probe's own block schema.
// ---------------------------------------------------------------------------
struct BaselineZProfiles {
    bool present = false;
    std::vector<std::string> ranking_d6_f0;
    std::vector<double> valley_depth_d6_f0;  // parallel to ranking
};

BaselineZProfiles ReadBaselineZProfiles() {
    BaselineZProfiles b;
    std::ifstream in("test/golden/baseline.json");
    if (!in.good()) return b;
    std::string text((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    size_t pos = text.find("\"z_profiles\"");
    if (pos == std::string::npos) return b;
    b.present = true;

    // ranking array: ["A", "B", ...]
    size_t rk = text.find("\"ranking_d6_f0\": [", pos);
    if (rk != std::string::npos) {
        size_t p = text.find('[', rk) + 1;
        while (true) {
            size_t q = text.find('"', p);
            if (q == std::string::npos || q >= text.size()) break;
            size_t end = text.find('"', q + 1);
            if (end == std::string::npos) break;
            b.ranking_d6_f0.push_back(text.substr(q + 1, end - q - 1));
            p = end + 1;
            if (text.find(']', p) < text.find('"', p)) break;
        }
    }
    // valley depths: "<variant>": <num> inside the valley_depth_d6_f0 object
    size_t vd = text.find("\"valley_depth_full_d6_f0\": {", pos);
    if (vd != std::string::npos) {
        for (const std::string& name : b.ranking_d6_f0) {
            std::string needle = "\"" + name + "\": ";
            size_t p = text.find(needle, vd);
            if (p == std::string::npos) {
                b.valley_depth_d6_f0.push_back(-1.0);
                continue;
            }
            p += needle.size();
            b.valley_depth_d6_f0.push_back(std::strtod(text.c_str() + p,
                                                       nullptr));
        }
    }
    return b;
}

// ---------------------------------------------------------------------------
// Pure-logic classifier + yamazaki fitter checks (no GPU): the z-flat injected
// variant edge case lives here — a flat cost curve must classify z-blind, a
// V-shaped curve must not, and the order-4 fitter recovers the argmin of a
// known quartic.
// ---------------------------------------------------------------------------
TEST_CASE("z-profile probe: classifier + poly-fit logic (U5)", "[oracle][gpu]") {
    // Injected z-flat variant: a constant cost curve.
    std::vector<double> dz31;
    for (int i = 0; i < 31; ++i) dz31.push_back(static_cast<double>(i - 15));
    std::vector<double> flat(31, 42.0);
    ValleyMetrics m_flat = ComputeValley(flat, dz31);
    REQUIRE(m_flat.valley_depth == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(ClassifyZBlind(m_flat.valley_depth));  // flat → z-blind

    // Synthetic V: min at the center (dz == 0), rising walls.
    std::vector<double> vcurve(31);
    for (int i = 0; i < 31; ++i) vcurve[i] = 100.0 * std::abs(dz31[i]);
    ValleyMetrics m_v = ComputeValley(vcurve, dz31);
    REQUIRE(m_v.valley_depth == Catch::Approx(1.0).margin(1e-9));
    REQUIRE(m_v.argmin_offset_mm == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(!ClassifyZBlind(m_v.valley_depth));

    // Order-4 fitter on a known quartic with a minimum at dz = +2.0 mm.
    std::vector<double> qz = dz31;
    std::vector<double> qc(31);
    for (int i = 0; i < 31; ++i) {
        double u = qz[i] - 2.0;
        qc[i] = 0.5 * u * u + 0.001 * u * u * u * u;
    }
    PolyFit4 fit = FitPoly4(qz, qc);
    REQUIRE(fit.argmin_mm == Catch::Approx(2.0).margin(0.05));
    REQUIRE(fit.normalized_rms < 1e-6);
}

// ---------------------------------------------------------------------------
// The probe itself.
// ---------------------------------------------------------------------------
TEST_CASE("Tier-2 z-profile probe: Δz sweep + term decomposition (U5)",
          "[oracle][gpu]") {
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<int> kDilations = {6, 3, 1, 4, 10, 2};
    const int kSweepHalf = 15;  // ±15 mm, step 1 → 31 points
    const int kNoiseRepeats = 5;
    const double kCouplingDx = 2.0;  // mm; documented proxy for the
                                     // minimal-detectable in-plane regression
                                     // (dilation-6 absorption radius ≈ 6 px ≈
                                     // 1.9 mm at the model plane), recorded
                                     // pending the perturbation suite's table.

    std::vector<double> dz31;
    for (int i = -kSweepHalf; i <= kSweepHalf; ++i)
        dz31.push_back(static_cast<double>(i));

    // Geometry (off-axis angle) from the frame-0 golden anchor.
    GeometryInfo geo = ComputeGeometry(StartPoses()[0]);
    std::cout << "[z_profile] focus→COG off-axis angle vs camera-z = "
              << geo.off_axis_angle_deg << " deg" << std::endl;

    // ------------------------------------------------------------------
    // Per (variant, dilation, frame) record.
    // ------------------------------------------------------------------
    struct SweepRecord {
        std::string variant;
        int dilation = 0;
        int frame = 0;
        std::vector<double> dz;
        std::vector<double> full;      // 31 (composed for DIRECT_DILATION)
        std::vector<double> chamfer;   // 31 (DIRECT_DILATION only)
        std::vector<double> dilated;   // 31 (DIRECT_DILATION only)
        double noise_floor = 0.0;
        double engine_max_abs_diff = 0.0;  // composed vs engine at {-15,0,15}
        double white_sum = 0.0;
        ValleyMetrics v_full;
        ValleyMetrics v_chamfer;
        ValleyMetrics v_dilated;
        bool z_blind = false;
        PolyFit4 poly_full;
        PolyFit4 poly_chamfer;
    };
    std::vector<SweepRecord> records;

    // ------------------------------------------------------------------
    // DIRECT_DILATION: 6 dilations × 3 frames, term-decomposed.
    // ------------------------------------------------------------------
    for (int frame = 0; frame < 3; ++frame) {
        for (int dilation : kDilations) {
            SweepRecord rec;
            rec.variant = "DIRECT_DILATION";
            rec.dilation = dilation;
            rec.frame = frame;
            rec.dz = dz31;

            Pipeline p = BuildFramePipeline(kBaseImages[frame], dilation);
            std::string err;
            REQUIRE(p.trunk->InitializeActiveCostFunction(err));

            // white_sum of the dilated comparison frame — the constant term in
            // the DIRECT_DILATION composition (initializeDIRECT_DILATION).
            cudaError_t ws_status = cudaSuccess;
            rec.white_sum = p.metrics->ComputeSumWhitePixels(
                p.dilated_a[0]->GetGPUImage(), &ws_status);
            REQUIRE(ws_status == cudaSuccess);

            Point6D anchor = StartPoses()[frame];
            auto composed_at = [&](double dz, double* full, double* chamfer,
                                   double* dilated) {
                Point6D phys = anchor;
                phys.z += dz;
                p.model->SetCurrentPrimaryCameraPose(ToPose(phys));
                REQUIRE(p.model->RenderPrimaryCamera(ToPose(phys)));
                cudaError_t e = cudaGetLastError();
                REQUIRE(e == cudaSuccess);
                GPUImage* rendered = p.model->GetPrimaryCameraRenderedImage();
                // Production order (costFunctionDIRECT_DILATION): FIDM first
                // (edge-marks the rendered image in place), then DMM counts
                // the EDGE pixels.
                *dilated = p.metrics->FastImplantDilationMetric(
                    rendered, p.dilated_a[0], dilation);
                *chamfer = p.metrics->DistanceMapMetric(
                    rendered, p.distance_maps[0], dilation);
                *full = rec.white_sum + *chamfer + *dilated;
            };

            for (double dz : dz31) {
                double full, chamfer, dilated;
                composed_at(dz, &full, &chamfer, &dilated);
                rec.full.push_back(full);
                rec.chamfer.push_back(chamfer);
                rec.dilated.push_back(dilated);
                REQUIRE(std::isfinite(full));
                REQUIRE(std::isfinite(chamfer));
                REQUIRE(std::isfinite(dilated));
            }

            // Noise floor: 5 engine repeats of cost(0) (the real production
            // path, callActiveCostFunction — re-renders internally).
            Point6D zero_pose = anchor;
            p.model->SetCurrentPrimaryCameraPose(ToPose(zero_pose));
            std::vector<double> repeats;
            for (int r = 0; r < kNoiseRepeats; ++r) {
                double c = p.trunk->callActiveCostFunction();
                REQUIRE(std::isfinite(c));
                REQUIRE(cudaGetLastError() == cudaSuccess);
                repeats.push_back(c);
            }
            auto mm = std::minmax_element(repeats.begin(), repeats.end());
            rec.noise_floor = *mm.second - *mm.first;

            // Composition check (DIAGNOSTIC-ONLY — probe finding, run 1):
            // composed full vs engine full at Δz ∈ {-15, 0, 15}. The FIRST
            // probe run exposed a dilation-3 divergence (diff=4650) traced to
            // a test-harness bug, not the production cost: the oracle
            // fixture's CostFunctionManager::updateCostFunctionParameterValues
            // int overload writes to a COPY (getIntParameters() returns the
            // vector by value), so the Dilation param never changed from the
            // registered default (6) — the oracle only worked because its
            // dilation 6 == default. Production sets the param via
            // setIntParameterValue on the ACTIVE class (SettingsBridge.cpp:
            // 586, mainscreen.cpp:4898-4900), which mutates the real object;
            // the probe uses that production path, and the check now shows
            // diff == 0 (or ~7e-12 float jitter) at every point. Recorded as
            // engine_max_abs_diff_vs_composed per sweep; NOT asserted on run 1
            // (a future regression would show non-zero diffs in the record).
            for (double dz : {-15.0, 0.0, 15.0}) {
                double full, chamfer, dilated;
                composed_at(dz, &full, &chamfer, &dilated);
                Point6D phys = anchor;
                phys.z += dz;
                p.model->SetCurrentPrimaryCameraPose(ToPose(phys));
                double engine_full = p.trunk->callActiveCostFunction();
                REQUIRE(std::isfinite(engine_full));
                double diff = std::abs(full - engine_full);
                rec.engine_max_abs_diff =
                    std::max(rec.engine_max_abs_diff, diff);
                std::cout << "[z_profile]   composition d" << dilation
                          << " f" << frame << " dz=" << dz
                          << ": composed=" << full
                          << " engine=" << engine_full << " diff=" << diff
                          << " (white=" << rec.white_sum
                          << " dilated=" << dilated << " chamfer=" << chamfer
                          << ")" << std::endl;
            }

            rec.v_full = ComputeValley(rec.full, dz31);
            rec.v_chamfer = ComputeValley(rec.chamfer, dz31);
            rec.v_dilated = ComputeValley(rec.dilated, dz31);
            rec.z_blind = ClassifyZBlind(rec.v_full.valley_depth);
            rec.poly_full = FitPoly4(dz31, rec.full);
            rec.poly_chamfer = FitPoly4(dz31, rec.chamfer);

            std::cout << "[z_profile] DIRECT_DILATION d" << dilation
                      << " frame " << frame << ": vd_full="
                      << rec.v_full.valley_depth << " argmin="
                      << rec.v_full.argmin_offset_mm << "mm plateau_hw="
                      << rec.v_full.plateau_halfwidth_mm << "mm noise="
                      << rec.noise_floor << " poly_argmin="
                      << rec.poly_full.argmin_mm << "mm z_blind="
                      << rec.z_blind << std::endl;

            records.push_back(rec);
        }
    }

    // ------------------------------------------------------------------
    // DIRECT_MAHFOUZ + DIRECT_DILATION_T1 at (dilation 6, frame 0) — the p3
    // ranking reference. Same 31-point sweep, full cost only (the ranking is
    // valley_depth of the FULL cost; the chamfer/dilated split is a
    // DIRECT_DILATION property). The always-Trunk guard (angle 02 bug 4)
    // means T1's reachable trunk path hardcodes the rendered-image dilation
    // to 1 — flagged in the manifest.
    // ------------------------------------------------------------------
    auto sweep_full_only = [&](const std::string& variant, int dilation,
                               int frame) {
        SweepRecord rec;
        rec.variant = variant;
        rec.dilation = dilation;
        rec.frame = frame;
        rec.dz = dz31;

        Pipeline p = BuildFramePipeline(kBaseImages[frame], dilation);
        std::string err;
        p.trunk->setActiveCostFunction(variant);
        REQUIRE(p.trunk->InitializeActiveCostFunction(err));

        Point6D anchor = StartPoses()[frame];
        for (double dz : dz31) {
            Point6D phys = anchor;
            phys.z += dz;
            p.model->SetCurrentPrimaryCameraPose(ToPose(phys));
            double c = p.trunk->callActiveCostFunction();
            REQUIRE(std::isfinite(c));
            REQUIRE(cudaGetLastError() == cudaSuccess);
            rec.full.push_back(c);
        }

        p.model->SetCurrentPrimaryCameraPose(ToPose(anchor));
        std::vector<double> repeats;
        for (int r = 0; r < kNoiseRepeats; ++r) {
            double c = p.trunk->callActiveCostFunction();
            REQUIRE(std::isfinite(c));
            repeats.push_back(c);
        }
        auto mm = std::minmax_element(repeats.begin(), repeats.end());
        rec.noise_floor = *mm.second - *mm.first;

        rec.v_full = ComputeValley(rec.full, dz31);
        rec.z_blind = ClassifyZBlind(rec.v_full.valley_depth);
        rec.poly_full = FitPoly4(dz31, rec.full);
        std::cout << "[z_profile] " << variant << " d" << dilation
                  << " frame " << frame << ": vd_full="
                  << rec.v_full.valley_depth << " argmin="
                  << rec.v_full.argmin_offset_mm << "mm noise="
                  << rec.noise_floor << " poly_argmin="
                  << rec.poly_full.argmin_mm << "mm z_blind="
                  << rec.z_blind << std::endl;
        records.push_back(rec);
    };
    sweep_full_only("DIRECT_MAHFOUZ", 6, 0);
    sweep_full_only("DIRECT_DILATION_T1", 6, 0);

    // ------------------------------------------------------------------
    // Coupling mode: repeat the sweep at in-plane x ± Δ to measure
    // dz*/d(in-plane) (how much the z-argmin shifts when the in-plane pose is
    // off by the minimal-detectable-regression proxy).
    // ------------------------------------------------------------------
    struct CouplingResult {
        int dilation = 0;
        double argmin_minus_mm = 0.0;
        double argmin_plus_mm = 0.0;
        double poly_argmin_minus_mm = 0.0;
        double poly_argmin_plus_mm = 0.0;
        double dz_star_d_inplane = 0.0;  // mm of z per mm of in-plane x
    };
    std::vector<CouplingResult> couplings;
    for (int dilation : {6, 1}) {
        CouplingResult cr;
        cr.dilation = dilation;
        Pipeline p = BuildFramePipeline(kBaseImages[0], dilation);
        std::string err;
        REQUIRE(p.trunk->InitializeActiveCostFunction(err));
        cudaError_t ws_status = cudaSuccess;
        double white_sum = p.metrics->ComputeSumWhitePixels(
            p.dilated_a[0]->GetGPUImage(), &ws_status);
        REQUIRE(ws_status == cudaSuccess);

        auto sweep_at_dx = [&](double dx, std::vector<double>* full_out,
                               std::vector<double>* dz_out) {
            Point6D anchor = StartPoses()[0];
            anchor.x += dx;
            for (double dz : dz31) {
                Point6D phys = anchor;
                phys.z += dz;
                p.model->SetCurrentPrimaryCameraPose(ToPose(phys));
                REQUIRE(p.model->RenderPrimaryCamera(ToPose(phys)));
                GPUImage* rendered = p.model->GetPrimaryCameraRenderedImage();
                double dilated = p.metrics->FastImplantDilationMetric(
                    rendered, p.dilated_a[0], dilation);
                double chamfer = p.metrics->DistanceMapMetric(
                    rendered, p.distance_maps[0], dilation);
                double full = white_sum + chamfer + dilated;
                REQUIRE(std::isfinite(full));
                full_out->push_back(full);
                dz_out->push_back(dz);
            }
        };
        std::vector<double> minus_full, minus_dz, plus_full, plus_dz;
        sweep_at_dx(-kCouplingDx, &minus_full, &minus_dz);
        sweep_at_dx(+kCouplingDx, &plus_full, &plus_dz);
        cr.argmin_minus_mm = ComputeValley(minus_full, minus_dz).argmin_offset_mm;
        cr.argmin_plus_mm = ComputeValley(plus_full, plus_dz).argmin_offset_mm;
        cr.poly_argmin_minus_mm = FitPoly4(minus_dz, minus_full).argmin_mm;
        cr.poly_argmin_plus_mm = FitPoly4(plus_dz, plus_full).argmin_mm;
        cr.dz_star_d_inplane =
            (cr.argmin_plus_mm - cr.argmin_minus_mm) / (2.0 * kCouplingDx);
        std::cout << "[z_profile] coupling d" << dilation
                  << ": argmin(x−2)=" << cr.argmin_minus_mm
                  << " argmin(x+2)=" << cr.argmin_plus_mm
                  << " dz*/d(in-plane)=" << cr.dz_star_d_inplane << " mm/mm"
                  << std::endl;
        couplings.push_back(cr);
    }

    // ------------------------------------------------------------------
    // p4 record: z_gap (recovered − fem.jts, baseline.json post-fix) vs the
    // per-frame DIRECT_DILATION d6 valley depth — the anti-correlation pair.
    // ------------------------------------------------------------------
    const std::vector<double> kZgapPerFrameMm = {-0.892, -1.193, -1.481};
    std::cout << "[z_profile] p4 (recorded): z_gap vs valley_depth (DD d6):\n";
    for (int f = 0; f < 3; ++f) {
        for (const auto& rec : records) {
            if (rec.variant == "DIRECT_DILATION" && rec.dilation == 6 &&
                rec.frame == f)
                std::cout << "[z_profile]   frame " << f
                          << ": z_gap=" << kZgapPerFrameMm[f]
                          << " mm, vd_full=" << rec.v_full.valley_depth
                          << std::endl;
        }
    }

    // ------------------------------------------------------------------
    // p3 ranking at (dilation 6, frame 0) — run 1 RECORDS (pin-first).
    // ------------------------------------------------------------------
    struct RankEntry {
        std::string variant;
        double vd = 0.0;
        double argmin = 0.0;
        bool z_blind_flag = false;
    };
    std::vector<RankEntry> ranking;
    for (const auto& rec : records) {
        if (rec.dilation == 6 && rec.frame == 0 &&
            (rec.variant == "DIRECT_DILATION" ||
             rec.variant == "DIRECT_MAHFOUZ" ||
             rec.variant == "DIRECT_DILATION_T1")) {
            ranking.push_back(
                {rec.variant, rec.v_full.valley_depth,
                 rec.v_full.argmin_offset_mm, rec.z_blind});
        }
    }
    REQUIRE(ranking.size() == 3);
    std::sort(ranking.begin(), ranking.end(),
              [](const RankEntry& a, const RankEntry& b) {
                  return a.vd > b.vd;
              });
    std::cout << "[z_profile] p3 ranking @ (d6, f0) by valley_depth:";
    for (const auto& r : ranking)
        std::cout << " " << r.variant << "(" << r.vd << ")";
    std::cout << std::endl;

    // ------------------------------------------------------------------
    // p1: min-at-zero within noise — the flagship happy path
    // (DIRECT_DILATION d6 f0 full curve) must have its min within the
    // recorded band (1 mm grid + the oracle's known −0.89 mm z-offset).
    // ------------------------------------------------------------------
    const SweepRecord* flagship = nullptr;
    for (const auto& rec : records) {
        if (rec.variant == "DIRECT_DILATION" && rec.dilation == 6 &&
            rec.frame == 0)
            flagship = &rec;
    }
    REQUIRE(flagship != nullptr);
    // p1: recorded band |argmin| ≤ 2 mm.
    REQUIRE(std::abs(flagship->v_full.argmin_offset_mm) <= 2.0);
    // p2: the flagship must NOT be z-blind (healthy V expected).
    REQUIRE(!flagship->z_blind);
    // noise_floor == 0 on the int-atomic path (5 repeats identical).
    REQUIRE(flagship->noise_floor == 0.0);

    // ------------------------------------------------------------------
    // Integration: probe + oracle agree on the recovered z at the golden
    // pose within the recorded band. baseline.json's frame-0 z-gap
    // (recovered − fem.jts) is −0.892 mm post-fix (rebaseline_2026-08-12);
    // the probe's argmin should land in [−2, +2] mm of it.
    // ------------------------------------------------------------------
    const double kOracleFrame0ZGapMm = -0.892;  // baseline.json post-fix
    REQUIRE(std::abs(flagship->v_full.argmin_offset_mm -
                     kOracleFrame0ZGapMm) <= 2.0);

    // ------------------------------------------------------------------
    // Write the machine record (test/golden/z_profiles.json) + print the
    // condensed block for baseline.json's z_profiles key.
    // ------------------------------------------------------------------
    std::ostringstream rec_json;
    rec_json << "{\n"
             << "  \"schema_version\": 1,\n"
             << "  \"instrument\": \"plan-008 U5 z-profile probe "
                "(test/oracle/z_profile_test.cpp)\",\n"
             << "  \"off_axis_angle_deg_vs_camera_z\": "
             << geo.off_axis_angle_deg << ",\n"
             << "  \"coupling_inplane_delta_mm\": " << kCouplingDx << ",\n"
             << "  \"coupling_delta_note\": \"proxy for the perturbation "
                "suite's minimal-detectable in-plane regression (dilation-6 "
                "absorption radius ~6 px ~1.9 mm at the model plane)\",\n"
             << "  \"z_profiles\": [\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const SweepRecord& r = records[i];
        rec_json << "    {\n"
                 << "      \"variant\": \"" << r.variant << "\",\n"
                 << "      \"dilation\": " << r.dilation << ",\n"
                 << "      \"frame\": " << r.frame << ",\n"
                 << "      \"valley_depth_full\": " << r.v_full.valley_depth
                 << ",\n"
                 << "      \"argmin_offset_full_mm\": "
                 << r.v_full.argmin_offset_mm << ",\n"
                 << "      \"plateau_halfwidth_full_mm\": "
                 << r.v_full.plateau_halfwidth_mm << ",\n"
                 << "      \"noise_floor\": " << r.noise_floor << ",\n"
                 << "      \"z_blind\": " << (r.z_blind ? "true" : "false")
                 << ",\n"
                 << "      \"white_sum\": " << r.white_sum << ",\n"
                 << "      \"engine_max_abs_diff_vs_composed\": "
                 << r.engine_max_abs_diff << ",\n";
        if (r.variant == "DIRECT_DILATION") {
            rec_json
                << "      \"valley_depth_chamfer\": "
                << r.v_chamfer.valley_depth << ",\n"
                << "      \"valley_depth_dilated\": "
                << r.v_dilated.valley_depth << ",\n"
                << "      \"argmin_offset_chamfer_mm\": "
                << r.v_chamfer.argmin_offset_mm << ",\n"
                << "      \"poly_argmin_full_mm\": "
                << r.poly_full.argmin_mm << ",\n"
                << "      \"poly_fit_residual_full_norm\": "
                << r.poly_full.normalized_rms << ",\n"
                << "      \"poly_argmin_chamfer_mm\": "
                << r.poly_chamfer.argmin_mm << ",\n"
                << "      \"poly_fit_residual_chamfer_norm\": "
                << r.poly_chamfer.normalized_rms << ",\n"
                << "      \"full_costs\": " << JsonArr(r.full) << ",\n"
                << "      \"chamfer_costs\": " << JsonArr(r.chamfer) << ",\n"
                << "      \"dilated_costs\": " << JsonArr(r.dilated) << "\n";
        } else {
            rec_json
                << "      \"poly_argmin_full_mm\": "
                << r.poly_full.argmin_mm << ",\n"
                << "      \"poly_fit_residual_full_norm\": "
                << r.poly_full.normalized_rms << ",\n"
                << "      \"full_costs\": " << JsonArr(r.full) << "\n";
        }
        rec_json << "    }" << (i + 1 < records.size() ? "," : "") << "\n";
    }
    rec_json << "  ],\n"
             << "  \"ranking_d6_f0\": [";
    for (size_t i = 0; i < ranking.size(); ++i) {
        if (i) rec_json << ", ";
        rec_json << "\"" << ranking[i].variant << "\"";
    }
    rec_json << "],\n"
             << "  \"valley_depth_full_d6_f0\": {";
    for (size_t i = 0; i < ranking.size(); ++i) {
        if (i) rec_json << ", ";
        rec_json << "\"" << ranking[i].variant << "\": "
                 << ranking[i].vd;
    }
    rec_json << "},\n"
             << "  \"z_gap_vs_valley_depth_d6_per_frame\": [";
    for (int f = 0; f < 3; ++f) {
        if (f) rec_json << ", ";
        double vd = -1.0;
        for (const auto& rec : records) {
            if (rec.variant == "DIRECT_DILATION" && rec.dilation == 6 &&
                rec.frame == f)
                vd = rec.v_full.valley_depth;
        }
        rec_json << "{\"frame\": " << f << ", \"z_gap_mm\": "
                 << kZgapPerFrameMm[f] << ", \"valley_depth\": " << vd
                 << "}";
    }
    rec_json << "],\n"
             << "  \"coupling\": [\n";
    for (size_t i = 0; i < couplings.size(); ++i) {
        const CouplingResult& c = couplings[i];
        rec_json << "    {\"dilation\": " << c.dilation
                 << ", \"argmin_minus_mm\": " << c.argmin_minus_mm
                 << ", \"argmin_plus_mm\": " << c.argmin_plus_mm
                 << ", \"dz_star_d_inplane_mm_per_mm\": "
                 << c.dz_star_d_inplane << "}"
                 << (i + 1 < couplings.size() ? "," : "") << "\n";
    }
    rec_json << "  ]\n"
             << "}\n";

    std::ofstream out("test/golden/z_profiles.json");
    REQUIRE(out.good());
    out << rec_json.str();
    out.close();

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_s =
        std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[z_profile] wrote test/golden/z_profiles.json; elapsed "
              << elapsed_s << " s" << std::endl;
    std::cout << "[z_profile] --- z_profiles record ---\n"
              << rec_json.str() << std::endl;

    // ------------------------------------------------------------------
    // Pin-first enforcement: only when a recorded block already exists
    // (run ≥ 2). Run 1 records; it must NOT assert the ranking.
    // ------------------------------------------------------------------
    BaselineZProfiles base = ReadBaselineZProfiles();
    if (base.present && base.ranking_d6_f0.size() == 3) {
        std::vector<std::string> fresh;
        for (const auto& r : ranking) fresh.push_back(r.variant);
        std::cout << "[z_profile] enforcement mode: recorded ranking [";
        for (size_t i = 0; i < base.ranking_d6_f0.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << base.ranking_d6_f0[i];
        }
        std::cout << "] vs fresh [";
        for (size_t i = 0; i < fresh.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << fresh[i];
        }
        std::cout << "]" << std::endl;
        REQUIRE(fresh == base.ranking_d6_f0);
        for (size_t i = 0; i < ranking.size() &&
                            i < base.valley_depth_d6_f0.size();
             ++i) {
            // Band per variant: the int-atomic variants (DIRECT_DILATION,
            // DIRECT_DILATION_T1) are value-deterministic → tight 0.05 band
            // (the z-blind threshold scale). DIRECT_MAHFOUZ is a float /
            // intensity path with a RECORDED non-zero noise floor (~0.3 in
            // cost units → ~0.2 in valley_depth jitter, run 1-3 measured), so
            // its valley depth is recorded-informational, not banded — the
            // RANKING ORDER is the p3 pin for every variant.
            const std::string& name = ranking[i].variant;
            if (name == "DIRECT_MAHFOUZ") {
                std::cout << "[z_profile]   band-check skipped for " << name
                          << " (float-path noise; vd fresh="
                          << ranking[i].vd << " recorded="
                          << base.valley_depth_d6_f0[i] << ")" << std::endl;
                continue;
            }
            // Recorded band: the z-blind classification threshold 0.05.
            REQUIRE(std::abs(ranking[i].vd - base.valley_depth_d6_f0[i]) <=
                    0.05);
        }
    } else {
        std::cout << "[z_profile] record mode (run 1): no prior z_profiles "
                     "block in baseline.json; ranking + valley metrics "
                     "recorded for later enforcement"
                  << std::endl;
    }
}
