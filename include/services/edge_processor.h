/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*EdgeProcessor (plan 004 U5 / R7): headless aperture/low-threshold/
 * high-threshold application, apply-all, and the dilation-constant decision
 * over Frame data, previously inline in the MainScreen edge slots.
 *
 * The processor owns the parameter-derived per-frame edge pipeline --
 * SetEdgeImage + SetDilatedImage -- including the dilation-constant decision
 * (the raw "Dilation" int-parameter value clamped to >= 0, then the
 * DIRECT_MAHFOUZ override to 3, byte-identical to
 * MainScreen::UpdateDilationFrames' Mahfouz branch). The view owns widget
 * reads, the SettingsService writes, the camera-dependent application targets
 * (which frames get processed depends on the camera radios), and the
 * re-render tail.
 *
 * The real Frame implementation is compute-layer (src/compute/frame.cu,
 * nvcc/torch/CUDA-linked); the headless tests compile a pure-OpenCV Frame
 * twin against the same header (test/unit/frame_headless.cpp) -- the
 * processor calls only the existing Frame methods.*/

#ifndef EDGE_PROCESSOR_H
#define EDGE_PROCESSOR_H

#include <string>
#include <vector>

#include "compute/frame.h"
#include "domain/settings_constants.h"

namespace jta {

/*Parameters for one edge-processing application. Values are exactly what the
 * view reads today: aperture/low_threshold/high_threshold come from the widget
 * (or, per-site, from the current frame via GetAperture/GetLowThreshold/
 * GetHighThreshold), dilation is the active cost function's "Dilation"
 * int-parameter value (0 when it has none), and cost_function_name is the
 * active cost function's name (the source of the DIRECT_MAHFOUZ override).*/
struct EdgeProcessingParams {
    int aperture = APERTURE;
    int low_threshold = LOW_THRESH;
    int high_threshold = HIGH_THRESH;
    int dilation = 0;
    std::string cost_function_name;
};

class EdgeProcessor {
public:
    /*Dilation-constant decision for the per-frame edge pipeline: the raw
     * "Dilation" int-parameter value clamped to >= 0, then the DIRECT_MAHFOUZ
     * override to 3 (the Mahfouz branch preserved byte-identical from
     * UpdateDilationFrames).*/
    static int ResolveDilation(
        int raw_dilation,
        const std::string& cost_function_name);

    /*Apply the parameter-derived edge pipeline to a single frame:
     * SetEdgeImage(aperture, low, high) + SetDilatedImage(resolved
     * dilation).*/
    static void ApplyToFrame(const EdgeProcessingParams& params, Frame& frame);

    /*Apply-all: the same pipeline to every frame in the range (identical to
     * calling ApplyToFrame per frame).*/
    static void ApplyToFrames(
        const EdgeProcessingParams& params,
        std::vector<Frame>& frames);

private:
    static void ApplyToFrameInternal(
        const EdgeProcessingParams& params,
        int dilation,
        Frame& frame);
};

}  // namespace jta

#endif  // EDGE_PROCESSOR_H
