/* U2: Graph key assembler + CaptureGeneration + CostFunctionManager provider.
 * Plan 012 U2 — one complete GraphRecipeKey + CaptureGeneration, upload epoch.
 * Pure assembler + hash headless; CFM provider headless (CPU-only ctor).
 */
#include <catch2/catch_test_macros.hpp>

#include "compute/graph_recipe.h"
#include "compute/graph_key_assembler.h"
#include "compute/CostFunctionManager.h"

using gpu_cost_function::AssembleCaptureGeneration;
using gpu_cost_function::AssembleGraphRecipeKey;
using gpu_cost_function::CaptureGeneration;
using gpu_cost_function::CaptureGenerationAssemblerInputs;
using gpu_cost_function::GraphKeyAssemblerInputs;
using gpu_cost_function::GraphRecipeCaptureInputs;
using gpu_cost_function::GraphRecipeKey;
using gpu_cost_function::HashCameraCalibrationParams;
using gpu_cost_function::ValidateGraphKeyVsInputs;

// ---------------------------------------------------------------------------
// A. Pure assembler
// ---------------------------------------------------------------------------

TEST_CASE("AssembleGraphRecipeKey fills all fields", "[graph_key_assembler]") {
    GraphKeyAssemblerInputs in;
    in.recipeId = "direct_dilation_monoplane";
    in.width = 1024;
    in.height = 1024;
    in.triangle_count = 12412;
    in.dilation = 6;
    in.camera_calib_hash = 0x1198000000000175ULL;
    in.cub_storage_bytes = 4096;
    in.curvature_capacity = 2048;
    in.maximum_stride_size = 10000000;
    in.graph_overhead_bytes = 8192;
    in.biplane = false;
    in.version = "1";

    GraphRecipeKey k = AssembleGraphRecipeKey(in);
    REQUIRE(k.recipeId == "direct_dilation_monoplane");
    REQUIRE(k.width == 1024);
    REQUIRE(k.height == 1024);
    REQUIRE(k.triangle_count == 12412);
    REQUIRE(k.dilation == 6);
    REQUIRE(k.camera_calib_hash == 0x1198000000000175ULL);
    REQUIRE(k.cub_storage_bytes == 4096);
    REQUIRE(k.curvature_capacity == 2048);
    REQUIRE(k.maximum_stride_size == 10000000);
    REQUIRE(k.graph_overhead_bytes == 8192);
    REQUIRE(k.biplane == false);
    REQUIRE(k.version == "1");
}

TEST_CASE("AssembleGraphRecipeKey is deterministic and pose-independent",
          "[graph_key_assembler]") {
    GraphKeyAssemblerInputs in;
    in.recipeId = "direct_dilation_monoplane";
    in.width = 512;
    in.height = 512;
    in.triangle_count = 300000;
    in.dilation = 6;
    in.camera_calib_hash = 42;
    GraphRecipeKey a = AssembleGraphRecipeKey(in);
    GraphRecipeKey b = AssembleGraphRecipeKey(in);
    REQUIRE(a == b);
    // No pose input exists — assembly is stable for same inputs
    GraphKeyAssemblerInputs in2 = in;
    in2.dilation = 4;
    GraphRecipeKey c = AssembleGraphRecipeKey(in2);
    REQUIRE_FALSE(a == c);
}

TEST_CASE("HashCameraCalibrationParams is deterministic and distinct",
          "[graph_key_assembler]") {
    auto h1 = HashCameraCalibrationParams(100.0f, 10.0f, 20.0f, 0.5f, false);
    auto h1b = HashCameraCalibrationParams(100.0f, 10.0f, 20.0f, 0.5f, false);
    REQUIRE(h1 == h1b);

    // Change each float individually → distinct
    REQUIRE(HashCameraCalibrationParams(101.0f, 10.0f, 20.0f, 0.5f, false) != h1);
    REQUIRE(HashCameraCalibrationParams(100.0f, 11.0f, 20.0f, 0.5f, false) != h1);
    REQUIRE(HashCameraCalibrationParams(100.0f, 10.0f, 21.0f, 0.5f, false) != h1);
    REQUIRE(HashCameraCalibrationParams(100.0f, 10.0f, 20.0f, 0.6f, false) != h1);
    // biplane flag
    REQUIRE(HashCameraCalibrationParams(100.0f, 10.0f, 20.0f, 0.5f, true) != h1);
    // zero input is not zero hash (FNV offset basis)
    auto hz = HashCameraCalibrationParams(0.0f, 0.0f, 0.0f, 0.0f, false);
    REQUIRE(hz != 0);
}

TEST_CASE("CaptureGeneration equality reflects all identity fields",
          "[graph_key_assembler]") {
    // Use raw buffers as identities
    unsigned char img = 0, cmp = 0, dist = 0;
    CaptureGenerationAssemblerInputs base;
    base.frame_index = 2;
    base.stage_id = 0;
    base.dilation = 6;
    base.upload_epoch = 5;
    base.rendered_image = &img;
    base.comparison_frame = &cmp;
    base.distance_map = &dist;

    CaptureGeneration g0 = AssembleCaptureGeneration(base);
    REQUIRE(g0 == AssembleCaptureGeneration(base));

    // upload_epoch change → unequal (C7 core)
    {
        auto b = base;
        b.upload_epoch = 6;
        REQUIRE_FALSE(g0 == AssembleCaptureGeneration(b));
    }
    {
        auto b = base;
        b.frame_index = 3;
        REQUIRE_FALSE(g0 == AssembleCaptureGeneration(b));
    }
    {
        auto b = base;
        b.stage_id = 1;
        REQUIRE_FALSE(g0 == AssembleCaptureGeneration(b));
    }
    {
        auto b = base;
        b.dilation = 4;
        REQUIRE_FALSE(g0 == AssembleCaptureGeneration(b));
    }
    {
        unsigned char other = 0;
        auto b = base;
        b.rendered_image = &other;
        REQUIRE_FALSE(g0 == AssembleCaptureGeneration(b));
    }
    {
        unsigned char other = 0;
        auto b = base;
        b.comparison_frame = &other;
        REQUIRE_FALSE(g0 == AssembleCaptureGeneration(b));
    }
    {
        unsigned char other = 0;
        auto b = base;
        b.distance_map = &other;
        REQUIRE_FALSE(g0 == AssembleCaptureGeneration(b));
    }
}

TEST_CASE("ValidateGraphKeyVsInputs gates on dilation match and rendered image",
          "[graph_key_assembler]") {
    GraphRecipeKey key;
    key.dilation = 6;

    unsigned char img = 1;
    unsigned char cmp = 2;
    unsigned char dist = 3;

    GraphRecipeCaptureInputs inputs;
    inputs.rendered_image = reinterpret_cast<gpu_cost_function::GPUImage*>(&img);
    inputs.comparison_frame = reinterpret_cast<gpu_cost_function::GPUDilatedFrame*>(&cmp);
    inputs.distance_map = reinterpret_cast<gpu_cost_function::GPUFrame*>(&dist);
    inputs.dilation = 6;

    REQUIRE(ValidateGraphKeyVsInputs(key, inputs));

    // dilation mismatch → false
    {
        auto k2 = key;
        k2.dilation = 4;
        REQUIRE_FALSE(ValidateGraphKeyVsInputs(k2, inputs));
    }
    // null rendered_image → false
    {
        GraphRecipeCaptureInputs bad = inputs;
        bad.rendered_image = nullptr;
        REQUIRE_FALSE(ValidateGraphKeyVsInputs(key, bad));
    }
    // null comparison_frame with everything else OK → true (only dilation+rendered gated)
    {
        GraphRecipeCaptureInputs ok = inputs;
        ok.comparison_frame = nullptr;
        REQUIRE(ValidateGraphKeyVsInputs(key, ok));
    }
    // null distance_map → true
    {
        GraphRecipeCaptureInputs ok = inputs;
        ok.distance_map = nullptr;
        REQUIRE(ValidateGraphKeyVsInputs(key, ok));
    }
}

// ---------------------------------------------------------------------------
// D. CFM-level (real CostFunctionManager, CPU-only ctor, headless)
// ---------------------------------------------------------------------------

TEST_CASE("CostFunctionManager upload epoch bumps monotonically",
          "[graph_key_assembler]") {
    jta_cost_function::CostFunctionManager cfm;
    REQUIRE(cfm.getUploadEpoch() == 0);
    cfm.BumpUploadEpoch();
    REQUIRE(cfm.getUploadEpoch() == 1);
    cfm.BumpUploadEpoch();
    REQUIRE(cfm.getUploadEpoch() == 2);
}

TEST_CASE("GetGraphRecipeCaptureInputs returns false when GPU objects are null",
          "[graph_key_assembler]") {
    jta_cost_function::CostFunctionManager cfm;
    gpu_cost_function::GraphRecipeCaptureInputs out;
    bool ok = cfm.GetGraphRecipeCaptureInputs(out);
    REQUIRE_FALSE(ok);
}

TEST_CASE("GetGraphRecipeCaptureInputs reads active dilation live",
          "[graph_key_assembler]") {
    jta_cost_function::CostFunctionManager cfm;
    // set Dilation on DIRECT_DILATION to a distinct value
    auto* cls = cfm.getCostFunctionClass("DIRECT_DILATION");
    REQUIRE(cls != nullptr);
    cls->setIntParameterValue("Dilation", 4);

    gpu_cost_function::GraphRecipeCaptureInputs out;
    bool ok = cfm.GetGraphRecipeCaptureInputs(out);
    // GPU objects are null → still false (headless), but the provider must
    // still forward the live dilation it read before the null gate.
    REQUIRE_FALSE(ok);
    REQUIRE(out.dilation == 4);
}

TEST_CASE("CostFunctionManager getCurrentFrameIndex mirrors set", "[graph_key_assembler]") {
    jta_cost_function::CostFunctionManager cfm;
    cfm.setCurrentFrameIndex(3);
    REQUIRE(cfm.getCurrentFrameIndex() == 3);
    cfm.setCurrentFrameIndex(0);
    REQUIRE(cfm.getCurrentFrameIndex() == 0);
}

// ---------------------------------------------------------------------------
// E. EvaluationExecutor Prepare (fake hooks, headless)
// ---------------------------------------------------------------------------
#include "compute/evaluation_executor.h"
#include "compute/batch_outcome.h"

TEST_CASE("Prepare with fake success hook stores wrappers, contexts idle", "[graph_key_assembler][prepare]") {
    gpu_cost_function::EvaluationExecutor exec;
    exec.pool().InitForTest(4);
    // Hook returns a dummy wrapper pointer (non-null = success); Prepare stores it
    // into graphExecs_[idx]. This pins wrapper OWNERSHIP by the executor.
    exec.InstallPrepareHook([](std::size_t, const gpu_cost_function::GraphRecipeKey&) -> void* {
        return reinterpret_cast<void*>(0x1ULL);
    });
    exec.InstallDestroyHook([](std::size_t) {});
    gpu_cost_function::GraphRecipeKey key;
    key.recipeId = "direct_dilation_monoplane";
    auto outcome = exec.Prepare(key, 2);
    REQUIRE(outcome.isOrderedScores());
    REQUIRE(exec.graphExecsSize() >= 2);
    REQUIRE(exec.preparedContextCount() >= 2);
    // C4: after successful prepare, contexts are idle-but-graph-ready (not in flight)
    REQUIRE_FALSE(exec.pool().IsInFlight(0));
    REQUIRE_FALSE(exec.pool().IsInFlight(1));
    // C2: executor owns the wrapper; ctx.graph_exec stays null
    REQUIRE(exec.pool().context(0)->graph_exec == nullptr);
    REQUIRE_FALSE(exec.firstSubmission());
}

TEST_CASE("Prepare failure returns NotSubmitted and cleans up all created wrappers", "[graph_key_assembler][prepare]") {
    gpu_cost_function::EvaluationExecutor exec;
    exec.pool().InitForTest(4);
    int callCount = 0;
    int destroyCount = 0;
    // Fails on the 2nd context; returns a live wrapper pointer on the 1st.
    exec.InstallPrepareHook([&](std::size_t, const gpu_cost_function::GraphRecipeKey&) -> void* {
        ++callCount;
        if (callCount == 2) return nullptr;
        return reinterpret_cast<void*>(0x1ULL);
    });
    exec.InstallDestroyHook([&](std::size_t) { ++destroyCount; });
    gpu_cost_function::GraphRecipeKey key;
    key.recipeId = "direct_dilation_monoplane";
    auto outcome = exec.Prepare(key, 3);
    REQUIRE(outcome.kind == gpu_cost_function::BatchOutcome::Kind::NotSubmitted);
    // C4: the successfully-created context-0 wrapper must ALSO be destroyed;
    // destroyCount == 1 (only one was created) not just >=1 (catches wrapper leak).
    REQUIRE(destroyCount == 1);
    // wrappers cleared back to 0; contexts drained (not in flight)
    REQUIRE(exec.preparedContextCount() == 0);
    REQUIRE_FALSE(exec.pool().IsInFlight(0));
    REQUIRE_FALSE(exec.pool().IsInFlight(1));
    REQUIRE_FALSE(exec.firstSubmission());
}

TEST_CASE("Prepare does not set firstSubmission", "[graph_key_assembler][prepare]") {
    gpu_cost_function::EvaluationExecutor exec;
    exec.pool().InitForTest(2);
    exec.InstallPrepareHook([](std::size_t, const gpu_cost_function::GraphRecipeKey&) { return reinterpret_cast<void*>(0x1ULL); });
    exec.InstallDestroyHook([](std::size_t) {});
    REQUIRE_FALSE(exec.firstSubmission());
    gpu_cost_function::GraphRecipeKey key;
    auto o = exec.Prepare(key, 2);
    REQUIRE(o.isOrderedScores());
    REQUIRE_FALSE(exec.firstSubmission());
}
