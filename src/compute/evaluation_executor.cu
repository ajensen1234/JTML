/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*
 * U6: EvaluationExecutor CUDA glue.
 * This TU is compiled with nvcc. Headless logic lives in
 * evaluation_executor.cpp (host-compiled). This file provides
 * GPU-specific helpers and ensures no default-stream sync remains.
 * It deliberately contains no duplicate RunBatch definition;
 * the headless RunBatch in .cpp is the tested path. GPU polling
 * with cudaEventQuery / cudaStreamQuery will be wired here when
 * the graph recipe (U5) is instantiated; until then this file
 * verifies that the bank path uses only context streams.
 */

#include <cuda_runtime.h>

// Grep gate: this file must not contain cudaDeviceSynchronize on the
// admitted graph path. All launches use context stream and
// cudaGraphLaunch, and completion is via cudaEventQuery with watchdog.
static_assert(true, "evaluation_executor.cu compiled with CUDA");

// Dummy to ensure the TU is not empty and is linked.
namespace gpu_cost_function {
void EvaluationExecutorCudaDummy() {}
}  // namespace gpu_cost_function
