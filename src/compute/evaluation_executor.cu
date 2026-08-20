/* U6: EvaluationExecutor CUDA glue — hook installer for greedy feeder. */
#include "compute/evaluation_executor.h"
#include "compute/evaluation_context.h"
#include "compute/graph_recipe.h"

#include <cuda_runtime.h>
#include <chrono>
#include <thread>

#include <limits>

namespace gpu_cost_function {

void InstallCudaFeederHooks(EvaluationExecutor& exec) {
    exec.InstallEnqueueHook([&exec](std::size_t ctxIdx, std::size_t /*inputPos*/, const Point6D& pose) -> bool {
        EvaluationContext* ctx = exec.pool().context(ctxIdx);
        if (!ctx) return false;
        ctx->x_location = static_cast<float>(pose.x);
        ctx->y_location = static_cast<float>(pose.y);
        ctx->z_location = static_cast<float>(pose.z);
        ctx->x_angle = static_cast<float>(pose.xa);
        ctx->y_angle = static_cast<float>(pose.ya);
        ctx->z_angle = static_cast<float>(pose.za);
        const GraphRecipe* recipe = exec.registry().FindEligible("DIRECT_DILATION", false);
        if (!recipe) return false;
        void* wrapper = exec.graphExecAt(ctxIdx);
        if (!wrapper) return false;
        if (!recipe->updateParams(wrapper, *ctx)) return false;
        if (!recipe->launch(wrapper, ctx->stream)) return false;
        if (!ctx->completion_event || !ctx->stream) return false;
        cudaError_t err = cudaEventRecord(reinterpret_cast<cudaEvent_t>(ctx->completion_event),
                                          reinterpret_cast<cudaStream_t>(ctx->stream));
        return err == cudaSuccess;
    });

    exec.InstallPollHook([&exec](std::size_t ctxIdx) -> PollResult {
        EvaluationContext* ctx = exec.pool().context(ctxIdx);
        if (!ctx || !ctx->completion_event) return PollResult::Error;
        cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(ctx->completion_event);
        cudaError_t q = cudaEventQuery(ev);
        if (q == cudaSuccess) return PollResult::Done;
        if (q == cudaErrorNotReady) return PollResult::Pending;
        return PollResult::Error;
    });

    // Plan 013 U2: bounded pacing for the many-context sweep. The greedy loop
    // calls this between zero-completion sweeps. Fixed bounded sleep (10 us)
    // tuned to the ~97 us per-eval residency: enough to stop hot-spinning the
    // driver while keeping the host well ahead of device completion. No
    // cudaEventSynchronize anywhere on the admitted path (R13 / zero_sync).
    // Headless unit tests never install this hook; they use the no-op/yield
    // default, so the 1 ms-watchdog pins are unaffected.
    exec.InstallPacingHook([&exec]() {
        (void)exec;
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    });

    exec.InstallCompleteFromPinsHook([&exec](std::size_t ctxIdx) -> double {
        EvaluationContext* ctx = exec.pool().context(ctxIdx);
        if (!ctx) return std::numeric_limits<double>::quiet_NaN();
        const GraphRecipe* recipe = exec.registry().FindEligible("DIRECT_DILATION", false);
        if (!recipe) return std::numeric_limits<double>::quiet_NaN();
        return recipe->completeFromPins(*ctx);
    });

    exec.InstallTeardownHook([]() {
        // Hook-driven loop already handles ForceRelease on abort.
        // Teardown hook is a placeholder for future drain logic (U5 hang handling).
    });
}

// Keep TU non-empty guard
void EvaluationExecutorCudaDummy() {}

}  // namespace gpu_cost_function
