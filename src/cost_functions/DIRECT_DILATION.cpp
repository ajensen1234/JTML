/*DIRECT_DILATION Source*/
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "CostFunctionManager.h"

namespace jta_cost_function {
bool CostFunctionManager::initializeDIRECT_DILATION(
    std::string &error_message) {
    /*Any cost function stage initialization proceedings go here.
    This is called when the optimizer begins a new stage.
    Must return whether or not the initialization was successful.
    To display an error message, simply store the message in
    the "error_message" variable and return false.*/

    /*CUDA Error status container*/
    cudaError cudaStatus;

    /*Compute the sum of the white pixels in the comparison dilated frame*/
    DIRECT_DILATION_current_white_pix_sum_dilated_comparison_image_A_ =
        gpu_metrics_->ComputeSumWhitePixels(
            (*gpu_dilated_frames_A_)[current_frame_index_]->GetGPUImage(),
            &cudaStatus);
    /*If Biplane Mode*/
    if (biplane_mode_) {
        DIRECT_DILATION_current_white_pix_sum_dilated_comparison_image_B_ =
            gpu_metrics_->ComputeSumWhitePixels(
                (*gpu_dilated_frames_B_)[current_frame_index_]->GetGPUImage(),
                &cudaStatus);
    }
    error_message = cudaGetErrorString(cudaStatus);

    /*Store Current Dilation Value*/
    this->getActiveCostFunctionClass()->getIntParameterValue(
        "Dilation", DIRECT_DILATION_current_dilation_parameter);

    /*Return if success or not*/
    return (cudaStatus == cudaSuccess);
}
bool CostFunctionManager::destructDIRECT_DILATION(std::string &error_message) {
    /*Any cost function stage initialization proceedings that involve
    creating new variables should be destructed here.
    This is called when the optimizer ends a stage.
    Must return whether or not the initialization was successful.
    To display an error message, simply store the message in
    the "error_message" variable and return false.*/

    return true;
}
double CostFunctionManager::costFunctionDIRECT_DILATION() {
    /*Cost function implementation goes here.
    This procedure is called every time the optimizer wants to
    query the cost function at a given point.
    One must return this value as a double.*/

    /*Render*/
    gpu_principal_model_->RenderPrimaryCamera(
        gpu_principal_model_->GetCurrentPrimaryCameraPose());

    /*Dilate rendered image to 1 dilation if in trunk mode*/
    double metric_score;

    /*(DIFFERENT FROM JTA PAPER) Dilate rendered image to same dilation as
     * comparison image*/
    metric_score =
        (DIRECT_DILATION_current_white_pix_sum_dilated_comparison_image_A_ +
         gpu_metrics_->FastImplantDilationMetric(
             gpu_principal_model_->GetPrimaryCameraRenderedImage(),
             gpu_dilated_frames_A_->at(current_frame_index_),
             DIRECT_DILATION_current_dilation_parameter));

    metric_score += gpu_metrics_->DistanceMapMetric(
        gpu_principal_model_->GetPrimaryCameraRenderedImage(),
        gpu_distance_maps_->at(current_frame_index_),
        DIRECT_DILATION_current_dilation_parameter);

    /*Biplane Mode Only*/
    if (biplane_mode_) {
        /*Render*/
        gpu_principal_model_->RenderSecondaryCamera(
            gpu_principal_model_->GetCurrentSecondaryCameraPose());

        /*(DIFFERENT FROM JTA PAPER) Dilate rendered image to same dilation as
         * comparison image*/
        double dist_score =
            (DIRECT_DILATION_current_white_pix_sum_dilated_comparison_image_B_ +
             gpu_metrics_->FastImplantDilationMetric(
                 gpu_principal_model_->GetSecondaryCameraRenderedImage(),
                 gpu_dilated_frames_B_->at(current_frame_index_),
                 DIRECT_DILATION_current_dilation_parameter));
        metric_score += (dist_score * dist_score);
    }
    // gpu_metrics_->DistanceMapMetric(
    //     gpu_principal_model_->GetPrimaryCameraRenderedImage(),
    //     gpu_distance_maps_->at(current_frame_index_),
    //     DIRECT_DILATION_current_dilation_parameter);

    return metric_score;
}
}  // namespace jta_cost_function
