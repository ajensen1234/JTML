/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*ImplantEstimator (plan 004 U8 / R12): the per-frame implant pose estimate
 * math, relocated verbatim from the MainScreen estimate slots. The estimator
 * is stateless: the view builds one ImplantEstimateContext per estimate run
 * (GPU model, torch pose model, scratch buffers, calibration) and calls
 * EstimateImplantPose once per frame, keeping the slots' progress/render
 * interleave byte-identical (per-frame controller API -- the view owns the
 * loop).
 *
 * Placement note: the estimator lands in jtml_services (not jtml_compute)
 * because the math's only non-compute dependency is the header-only
 * calibration math (services/calibration.h: Matrix_3_3 +
 * Calibration::multiplication_mat_mat), and the 003 layered split keeps
 * jtml_compute standalone (no compute -> services include). jtml_services is
 * already GPU-linked in U8 (torch + jtml_compute on its link line, see
 * src/services/CMakeLists.txt), and the compute headers the .cpp needs
 * (gpu_model.cuh etc.) are CXX-compilable (the oracle test compiles the same
 * header chain under plain CXX).*/

#ifndef IMPLANT_ESTIMATOR_H
#define IMPLANT_ESTIMATOR_H

/*Torch (Tensor + jit::Module)*/
#include <torch/torch.h>

/*OpenCV*/
#include <opencv2/core/mat.hpp>

/*Calibration math (header-only: Matrix_3_3, Calibration::
 * multiplication_mat_mat, camera_A_principal_)*/
#include "services/calibration.h"

/*Pose type*/
#include "domain/data_structures_6D.h"

/*GPUModel is only referenced by pointer here; the full definition
 * (compute/gpu_model.cuh) stays in the .cpp.*/
namespace gpu_cost_function {
class GPUModel;
}

namespace jta {

/*Per-frame estimate context: everything the estimate math for one frame
 * needs. Built once by the view before its per-frame loop; the estimator
 * never owns any of it (the view mallocs/frees host_image and orientation
 * around the loop, exactly as the slots did, and owns the torch pose model +
 * GPU model).*/
struct ImplantEstimateContext {
    gpu_cost_function::GPUModel* gpu_mod = nullptr;
    torch::jit::Module* model = nullptr;
    unsigned char* host_image = nullptr;
    float* orientation = nullptr;
    torch::Tensor gpu_byte_placeholder;
    Calibration calibration;
    unsigned int input_width = 1024;
    unsigned int input_height = 1024;
    unsigned int orig_width = 0;
    unsigned int orig_height = 0;
};

/*Implant pose estimate math for one frame (relocated verbatim from the two
 * estimate slots; the only slot-to-slot difference is the femoral z clamp,
 * reproduced via clamp_z_to_principal -- the femoral slot passes true, the
 * tibial false). Returns the pose the view saves via SavePose.*/
class ImplantEstimator {
public:
    static Point6D EstimateImplantPose(
        ImplantEstimateContext& ctx,
        const cv::Mat& orig_inverted,
        bool clamp_z_to_principal);
};

}  // namespace jta

#endif /* IMPLANT_ESTIMATOR_H */
