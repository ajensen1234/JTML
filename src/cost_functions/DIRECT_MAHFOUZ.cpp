/*DIRECT_MAHFOUZ Source*/
#include "CostFunctionManager.h"

namespace jta_cost_function {
bool CostFunctionManager::initializeDIRECT_MAHFOUZ(std::string& error_message) {
    /*Any cost function stage initialization proceedings go here.
    This is called when the optimizer begins a new stage.
    Must return whether or not the initialization was successful.
    To display an error message, simply store the message in
    the "error_message" variable and return false.*/

    return true;
}
bool CostFunctionManager::destructDIRECT_MAHFOUZ(std::string& error_message) {
    /*Any cost function stage initialization proceedings that involve
    creating new variables should be destructed here.
    This is called when the optimizer ends a stage.
    Must return whether or not the initialization was successful.
    To display an error message, simply store the message in
    the "error_message" variable and return false.*/

    return true;
}
double CostFunctionManager::costFunctionDIRECT_MAHFOUZ() {
    /*Cost function implementation goes here.
    This procedure is called every time the optimizer wants to
    query the cost function at a given point.
    One must return this value as a double.*/

    /*Render*/
    gpu_principal_model_->RenderPrimaryCamera(
        gpu_principal_model_->GetCurrentPrimaryCameraPose());

    /*Mahfouz Fast Cost Function*/
    double metric_score = gpu_metrics_->ImplantMahfouzMetric(
        gpu_principal_model_->GetPrimaryCameraRenderedImage(),
        gpu_dilated_frames_A_->at(current_frame_index_),
        gpu_intensity_frames_A_->at(current_frame_index_));

    /*Biplane Mode Only*/
    if (biplane_mode_) {
        /*Render*/
        gpu_principal_model_->RenderSecondaryCamera(
            gpu_principal_model_->GetCurrentSecondaryCameraPose());

        /*Add Mahfouz Metric Score for Secondary Camera Image*/
        metric_score += gpu_metrics_->ImplantMahfouzMetric(
            gpu_principal_model_->GetSecondaryCameraRenderedImage(),
            gpu_dilated_frames_B_->at(current_frame_index_),
            gpu_intensity_frames_B_->at(current_frame_index_));
    }

    return metric_score;
}
}  // namespace jta_cost_function