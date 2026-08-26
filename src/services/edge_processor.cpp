// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*EdgeProcessor (plan 004 U5 / R7): the parameter-derived per-frame edge
 * pipeline relocated from the MainScreen edge slots. Arithmetic/control flow
 * preserved: SetEdgeImage(aperture, low, high) + SetDilatedImage(dilation)
 * with the dilation-constant decision (raw value clamped to >= 0, then the
 * DIRECT_MAHFOUZ override to 3 -- the Mahfouz branch kept byte-identical to
 * MainScreen::UpdateDilationFrames).*/

#include "services/edge_processor.h"

namespace jta {

int EdgeProcessor::ResolveDilation(
    int raw_dilation,
    const std::string& cost_function_name) {
    int dilation_val = raw_dilation;
    if (dilation_val < 0) {
        dilation_val = 0;
    }
    /*Mahfouz Case*/
    if (cost_function_name == "DIRECT_MAHFOUZ") {
        dilation_val = 3;
    }
    return dilation_val;
}

void EdgeProcessor::ApplyToFrameInternal(
    const EdgeProcessingParams& params,
    int dilation,
    Frame& frame) {
    frame.SetEdgeImage(
        params.aperture, params.low_threshold, params.high_threshold);
    frame.SetDilatedImage(dilation);
}

void EdgeProcessor::ApplyToFrame(
    const EdgeProcessingParams& params,
    Frame& frame) {
    ApplyToFrameInternal(
        params,
        ResolveDilation(params.dilation, params.cost_function_name),
        frame);
}

void EdgeProcessor::ApplyToFrames(
    const EdgeProcessingParams& params,
    std::vector<Frame>& frames) {
    const int dilation =
        ResolveDilation(params.dilation, params.cost_function_name);
    for (Frame& frame : frames) {
        ApplyToFrameInternal(params, dilation, frame);
    }
}

}  // namespace jta
