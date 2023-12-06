// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*DIRECT_DILATION_POLE_CONSTRAINT Source*/
#include "CostFunctionManager.h"

namespace jta_cost_function {
	bool CostFunctionManager::initializeDIRECT_DILATION_POLE_CONSTRAINT(std::string& error_message) {
		/*Any cost function stage initialization proceedings go here.
		This is called once when the optimizer begins a new stage.
		Must return whether or not the initialization was successful.
		To display an error message, simply store the message in
		the "error_message" variable and return false.*/
		if (gpu_non_principal_models_->size() == 1) {
			gpu_cost_function::Pose temp_pose;
			pose_storage_->GetModelPose("tibia", 0, &temp_pose);
			x_loc_non = temp_pose.x_location_;
			/*Initialize DIRECT DILATION*/
			/*CUDA Error status container*/
			cudaError cudaStatus;

			/*Compute the sum of the white pixels in the comparison dilated frame*/
			DIRECT_DILATION_current_white_pix_sum_dilated_comparison_image_A_ =
				gpu_metrics_->ComputeSumWhitePixels((*gpu_dilated_frames_A_)[current_frame_index_]->GetGPUImage(), &cudaStatus);
			/*If Biplane Mode*/
			if (biplane_mode_) {
				DIRECT_DILATION_current_white_pix_sum_dilated_comparison_image_B_ =
					gpu_metrics_->ComputeSumWhitePixels((*gpu_dilated_frames_B_)[current_frame_index_]->GetGPUImage(), &cudaStatus);
			}
			error_message = cudaGetErrorString(cudaStatus);

			/*Store Current Dilation Value*/
			this->getActiveCostFunctionClass()->getIntParameterValue("Dilation", DIRECT_DILATION_current_dilation_parameter);

			/*Return if success or not*/
			return (cudaStatus == cudaSuccess);
		}
		else {
			error_message = "Need exactly 2 models.";
			return false;
		}
	}

	bool CostFunctionManager::destructDIRECT_DILATION_POLE_CONSTRAINT(std::string& error_message) {
		/*Any cost function stage initialization proceedings that involve
		creating new variables should be destructed here.
		Must return whether or not the destruction was successful.
		To display an error message, simply store the message in
		the "error_message" variable and return false.*/

		return true;
	}
	double CostFunctionManager::costFunctionDIRECT_DILATION_POLE_CONSTRAINT() {
		/*Cost function implementation goes here.
		This procedure is called every time the optimizer wants to
		query the cost function at a given point.
		One must return this value as a double.*/
		gpu_cost_function::Pose p = gpu_principal_model_->GetCurrentPrimaryCameraPose();
		gpu_cost_function::Pose np = (*gpu_non_principal_models_)[0]->GetCurrentPrimaryCameraPose();

		/*Create shorthand variables for trig vals*/
		float cz = cos(np.z_angle_*3.14159265358979323846f / 180.0f);
		float sz = sin(np.z_angle_*3.14159265358979323846f / 180.0f);
		float cx = cos(np.x_angle_*3.14159265358979323846f / 180.0f);
		float sx = sin(np.x_angle_*3.14159265358979323846f / 180.0f);
		float cy = cos(np.y_angle_*3.14159265358979323846f / 180.0f);
		float sy = sin(np.y_angle_*3.14159265358979323846f / 180.0f);

		/*Result of multiplying R matrix by {0,1,0} unit vector. */
		float r_x = -1.0 * sz * cx;
		float r_y = cz * cx;
		float r_z = sx;
		/*Result of multiplying R matrix by {1,0,0} unit vector*/
		float s_x = cy*cz - sx*sy*sz;
		float s_y = cz*sx*sy + cy*sz;
		float s_z = -cx*sy;

		/*2-Norm (minimized) solution for alpha and beta*/
		double alpha = (r_x*p.x_location_ + r_y*p.y_location_ + r_z*p.z_location_ - (r_x*np.x_location_ + r_y*np.y_location_ + r_z*np.z_location_)) / (r_x*r_x + r_y*r_y + r_z*r_z);
		double beta = (s_x*p.x_location_ + s_y*p.y_location_ + s_z*p.z_location_ - (s_x*np.x_location_ + s_y*np.y_location_ + s_z*np.z_location_)) / (s_x*s_x + s_y*s_y + s_z*s_z);

		/*Find distance components (x,y,z) from this plane */
		double v_x = alpha*r_x + beta*s_x + np.x_location_ - p.x_location_;
		double v_y = alpha*r_y + beta*s_y + np.y_location_ - p.y_location_;
		double v_z = alpha*r_z + beta*s_z + np.z_location_ - p.z_location_;
		/*Shift tibia using magnitude of minimized distances*/
		double shortest_distance = sqrt(v_x*v_x + v_y*v_y + v_z*v_z);

		/*Parameter*/
		double pole_weight;
		this->getActiveCostFunctionClass()->getDoubleParameterValue("PoleWeight", pole_weight);

		/*Direct Dilation begin */
		/*Render*/
		gpu_principal_model_->RenderPrimaryCamera(p);

		/*(DIFFERENT FROM JTA PAPER) Dilate rendered image to same dilation as comparison image*/
		double metric_score = (DIRECT_DILATION_current_white_pix_sum_dilated_comparison_image_A_ +
			gpu_metrics_->FastImplantDilationMetric(gpu_principal_model_->GetPrimaryCameraRenderedImage(),
				gpu_dilated_frames_A_->at(current_frame_index_), DIRECT_DILATION_current_dilation_parameter));

		return metric_score + pole_weight*shortest_distance;
	}
}