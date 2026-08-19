/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*
 * U3: Graph capture compatibility probe (preflight truth).
 * Header is CUDA-free (opaque void* handles) so headless unit tests can
 * include it without linking CUDA.  CUDA-owned logic lives in
 * src/compute/graph_preflight.cu.
 *
 * Probes are deterministic: a failed capture is not a crash, it returns
 * GraphPreflightResult{ capturable=false, reasonCode, failingNodeHint }.
 * The executor uses this to mark a recipe unavailable for the run (R8)
 * vs aborting a batch (R7).
 */

#pragma once

#include <string>

#include "compute/graph_recipe.h"

namespace gpu_cost_function {

// Reason codes for preflight failures — stable across runs.
// 0  = ok / capturable
// 1  = no eligible recipe
// 100 = cudaStreamSynchronize on captured stream (RenderPhase :1173)
// 101 = host AABB / fragment_fill dependency (packet barrier)
// 102 = zero-triangle degenerate (no work)
// 103 = maximum_stride_size overflow
// 200 = cudaGraph capture generic error (cudaErrorGraphExecUpdateFailure etc.)
// 201 = cudaGraphInstantiate failure
enum GraphPreflightReason : int {
    kPreflightOk = 0,
    kNoEligibleRecipe = 1,
    kSyncBlocker = 100,
    kHostAABBDependency = 101,
    kZeroTriangle = 102,
    kOverflow = 103,
    kGraphCaptureError = 200,
    kGraphInstantiateError = 201,
};

// Operation set callback for ProbeCapturableOpSet.
// Receives void* stream (cudaStream_t cast) and opaque context pointer.
// Should enqueue the full operation set on the stream (may include sync).
// Returns 0 on success, non-zero on CUDA error.
using CaptureOpFn = int (*)(void* stream, void* context);

// Core probe: wraps the operation set in cudaStreamBeginCapture(Global).
// Reports whether the operation set is graph-capturable + instantiable.
// Cleans up all resources and clears sticky errors on every path.
GraphPreflightResult ProbeCapturableOpSet(CaptureOpFn op_fn, void* context);

// Probe the current serial render path via a caller-supplied operation set.
// The callback is invoked inside ProbeCapturableOpSet's capture cycle so the
// probe discovers blockers from the ACTUAL hot-path kernels (e.g.
// RenderPhase's memset → WorldToPixel → CUB scan → D2H → synchronize),
// not from a self-contained representative duplicate.
//
// The callback must enqueue the full operation set on the provided stream
// (which is inside a cudaStreamBeginCapture/EndCapture cycle). Callers that
// temporarily substitute an owned stream must restore the original handle.
GraphPreflightResult ProbeCurrentSerialPath(CaptureOpFn op_fn, void* context);

// Synthetic capturable micro-graph: dummy kernel + cudaMemsetAsync with
// fixed grid. Proves the toolchain can capture at all.
// Takes optional stream (nullptr = create one).
GraphPreflightResult ProbeSyntheticMicroGraph(void* stream = nullptr);

// Edge cases — never crash, return capturable=false with specific
// reasonCode.
GraphPreflightResult ProbeZeroTriangle(void* stream = nullptr);
GraphPreflightResult ProbeOverflowCase(void* stream = nullptr);

// Helper to format a GraphPreflightResult for logging.
std::string FormatPreflightResult(const GraphPreflightResult& r);

} // namespace gpu_cost_function
