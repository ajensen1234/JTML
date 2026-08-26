/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*
 * U3: Graph capture compatibility probe — CUDA-owned implementation.
 * Only this TU includes cuda_runtime; header remains CUDA-free.
 *
 * ProbeCapturableOpSet is the core: wraps any operation set in
 * cudaStreamBeginCapture(Global) and reports capturability from the
 * observed CUDA API results.
 * ProbeCurrentSerialPath forwards a caller-supplied operation set to
 * ProbeCapturableOpSet. The probe discovers blockers from the ACTUAL
 * hot-path kernels (e.g. RenderPhase's memcpy D2H + synchronize), not
 * from a self-contained representative duplicate.
 */

#include <cuda_runtime.h>

#include <string>

#include "compute/graph_preflight.h"

namespace gpu_cost_function {

namespace {

__global__ void DummyPreflightKernel(int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && out != nullptr) {
        *out = 1;
    }
}

__global__ void DummyPreflightEmptyKernel() {}

std::string CudaErrorString(cudaError_t err) {
    const char* s = cudaGetErrorString(err);
    return s ? std::string(s) : std::string("unknown CUDA error");
}
// Helper that does a minimal synthetic graph capture with a dummy kernel
// and a memset node. Returns capturable=true on success.
GraphPreflightResult ProbeSyntheticInternal(
    void* stream_ptr,
    bool own_stream_if_null) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    bool ownStream = false;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    int* dev_out = nullptr;
    GraphPreflightResult result;
    result.capturable = false;
    result.reasonCode = kGraphCaptureError;

    cudaError_t err = cudaSuccess;

    // Create stream if needed.
    if (!stream && own_stream_if_null) {
        err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            result.reasonCode = static_cast<int>(err);
            result.failingNodeHint = "cudaStreamCreateWithFlags";
            result.reasonString = CudaErrorString(err);
            return result;
        }
        ownStream = true;
    }
    if (!stream) {
        // No stream provided and not allowed to create — treat as
        // non-capturable but not a crash. Return deterministic code.
        result.reasonCode = kGraphCaptureError;
        result.failingNodeHint = "null stream";
        result.reasonString = "null stream without auto-create";
        return result;
    }

    // Pre-allocate device memory before capture (allocations inside capture
    // are not allowed and would invalidate it).
    err = cudaMalloc(&dev_out, 256 * sizeof(int));
    if (err != cudaSuccess) {
        result.reasonCode = static_cast<int>(err);
        result.failingNodeHint = "cudaMalloc pre-alloc";
        result.reasonString = CudaErrorString(err);
        if (ownStream) {
            cudaStreamDestroy(stream);
        }
        return result;
    }

    // Begin capture — Global mode so even legacy-stream use invalidates.
    err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        result.reasonCode = kGraphCaptureError;
        result.failingNodeHint = "cudaStreamBeginCapture";
        result.reasonString = CudaErrorString(err);
        cudaFree(dev_out);
        if (ownStream) {
            cudaStreamDestroy(stream);
        }
        return result;
    }

    // Inside capture: dummy kernel + memset. Both are capturable.
    DummyPreflightKernel<<<1, 32, 0, stream>>>(dev_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        // End capture to get graph (which will be null on error) and clean up.
        cudaStreamEndCapture(stream, &graph);
        if (graph) {
            cudaGraphDestroy(graph);
        }
        if (exec) {
            cudaGraphExecDestroy(exec);
        }
        result.reasonCode = kGraphCaptureError;
        result.failingNodeHint = "DummyPreflightKernel launch";
        result.reasonString = CudaErrorString(err);
        cudaFree(dev_out);
        if (ownStream) {
            cudaStreamDestroy(stream);
        }
        return result;
    }

    err = cudaMemsetAsync(dev_out, 0, 256 * sizeof(int), stream);
    if (err != cudaSuccess) {
        cudaStreamEndCapture(stream, &graph);
        if (graph) {
            cudaGraphDestroy(graph);
        }
        result.reasonCode = kGraphCaptureError;
        result.failingNodeHint = "cudaMemsetAsync";
        result.reasonString = CudaErrorString(err);
        cudaFree(dev_out);
        if (ownStream) {
            cudaStreamDestroy(stream);
        }
        return result;
    }

    // Another dummy empty kernel to prove multiple nodes are capturable.
    DummyPreflightEmptyKernel<<<1, 1, 0, stream>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaStreamEndCapture(stream, &graph);
        if (graph) {
            cudaGraphDestroy(graph);
        }
        result.reasonCode = kGraphCaptureError;
        result.failingNodeHint = "DummyPreflightEmptyKernel launch";
        result.reasonString = CudaErrorString(err);
        cudaFree(dev_out);
        if (ownStream) {
            cudaStreamDestroy(stream);
        }
        return result;
    }

    err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess || graph == nullptr) {
        // Capture invalidation path — per cudaStreamIsCapturing/EndCapture
        // docs, graph is NULL on error. Must still clean up.
        if (graph) {
            cudaGraphDestroy(graph);
        }
        result.reasonCode = kGraphCaptureError;
        result.failingNodeHint = "cudaStreamEndCapture";
        result.reasonString = CudaErrorString(err);
        cudaFree(dev_out);
        if (ownStream) {
            cudaStreamDestroy(stream);
        }
        return result;
    }

    // Instantiate — this is where topology validation happens.
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        result.reasonCode = kGraphInstantiateError;
        result.failingNodeHint = "cudaGraphInstantiate";
        result.reasonString = CudaErrorString(err);
        cudaGraphDestroy(graph);
        cudaFree(dev_out);
        if (ownStream) {
            cudaStreamDestroy(stream);
        }
        return result;
    }

    // Success — clean up graph and exec (probe is not for execution).
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaFree(dev_out);
    if (ownStream) {
        cudaStreamDestroy(stream);
    }

    result.capturable = true;
    result.reasonCode = kPreflightOk;
    result.failingNodeHint.clear();
    result.reasonString = "capturable";
    return result;
}

}  // namespace

// ---------------------------------------------------------------------------
// Core capture probe: wraps any operation set in Global capture and reports
// capturability from observed CUDA API results.
// ---------------------------------------------------------------------------
GraphPreflightResult ProbeCapturableOpSet(CaptureOpFn op_fn, void* context) {
    GraphPreflightResult result;
    result.capturable = false;
    result.reasonCode = kGraphCaptureError;

    if (!op_fn) {
        result.reasonCode = kNoEligibleRecipe;
        result.failingNodeHint = "null op_fn";
        result.reasonString = "no operation set callback provided";
        return result;
    }

    // Create non-blocking stream.
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        result.reasonCode = static_cast<int>(err);
        result.failingNodeHint = "cudaStreamCreateWithFlags";
        result.reasonString = CudaErrorString(err);
        return result;
    }

    // Begin capture in Global mode — even legacy-stream use invalidates.
    err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        result.reasonCode = kGraphCaptureError;
        result.failingNodeHint = "cudaStreamBeginCapture";
        result.reasonString = CudaErrorString(err);
        cudaStreamDestroy(stream);
        return result;
    }

    // Execute the operation set on the captured stream.
    void* stream_ptr = static_cast<void*>(stream);
    int op_err = op_fn(stream_ptr, context);

    // End capture — may fail if the callback invalidated it (e.g. sync).
    cudaGraph_t graph = nullptr;
    err = cudaStreamEndCapture(stream, &graph);

    if (err != cudaSuccess || graph == nullptr) {
        // Capture was invalidated (e.g. cudaStreamSynchronize in Global mode).
        if (graph) {
            cudaGraphDestroy(graph);
        }

        // cudaErrorIllegalState (719) or cudaErrorStreamCaptureInvalidated
        // (919) both indicate the capture was invalidated by a blocking API
        // call.
        if (err == cudaErrorIllegalState ||
            err == cudaErrorStreamCaptureInvalidated) {
            result.reasonCode = kSyncBlocker;
            result.failingNodeHint =
                "cudaStreamSynchronize on captured stream (Global mode)";
            result.reasonString =
                "operation set includes blocking sync that invalidates capture";
        } else {
            result.reasonCode = kGraphCaptureError;
            result.failingNodeHint = "cudaStreamEndCapture";
            result.reasonString = CudaErrorString(err);
        }

        // Clear sticky errors from invalidated capture, drain and destroy.
        cudaGetLastError();
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        return result;
    }

    // Capture succeeded — check if the operation set itself reported an error.
    if (op_err != 0) {
        result.reasonCode = kGraphCaptureError;
        result.failingNodeHint = "operation set callback returned error";
        result.reasonString =
            "op_fn returned error code " + std::to_string(op_err);
        cudaGraphDestroy(graph);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        return result;
    }

    // Try to instantiate the graph — validates topology and resource usage.
    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        result.reasonCode = kGraphInstantiateError;
        result.failingNodeHint = "cudaGraphInstantiate";
        result.reasonString = CudaErrorString(err);
        cudaGraphDestroy(graph);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        return result;
    }

    // Complete success — clean up and report.
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    result.capturable = true;
    result.reasonCode = kPreflightOk;
    result.failingNodeHint.clear();
    result.reasonString = "capturable";
    return result;
}

// ---------------------------------------------------------------------------
// Probe the current serial render path via a caller-supplied operation set.
// Thin forwarder: the caller supplies the real hot-path callback, and the
// probe discovers blockers from the ACTUAL kernels (e.g. RenderPhase's
// memcpy D2H + cudaStreamSynchronize), not from a self-contained
// representative duplicate.
// ---------------------------------------------------------------------------
GraphPreflightResult ProbeCurrentSerialPath(CaptureOpFn op_fn, void* context) {
    return ProbeCapturableOpSet(op_fn, context);
}

GraphPreflightResult ProbeSyntheticMicroGraph(void* stream) {
    return ProbeSyntheticInternal(stream, true);
}

GraphPreflightResult ProbeZeroTriangle(void* stream) {
    // Zero-triangle degenerate — no work, must not crash, return
    // capturable=false with specific code. If a stream is provided we
    // still don't capture; the edge case is valid but not useful.
    (void)stream;
    GraphPreflightResult r;
    r.capturable = false;
    r.reasonCode = kZeroTriangle;
    r.failingNodeHint = "zero triangle_count";
    r.reasonString = "non-capturable: zero-triangle degenerate (no work)";
    return r;
}

GraphPreflightResult ProbeOverflowCase(void* stream) {
    (void)stream;
    GraphPreflightResult r;
    r.capturable = false;
    r.reasonCode = kOverflow;
    r.failingNodeHint =
        "fragment_fill > maximum_stride_size * (threads_per_block-1)";
    r.reasonString = "non-capturable: maximum_stride_size overflow guard";
    return r;
}

std::string FormatPreflightResult(const GraphPreflightResult& r) {
    std::string out = r.capturable ? "capturable=true" : "capturable=false";
    out += " reasonCode=" + std::to_string(r.reasonCode);
    if (!r.failingNodeHint.empty()) {
        out += " hint=" + r.failingNodeHint;
    }
    if (!r.reasonString.empty()) {
        out += " reason=" + r.reasonString;
    }
    return out;
}

}  // namespace gpu_cost_function
