#pragma once

/*Render Engine Header for Pose Class*/
#include "gpu/render_engine.cuh"

/*Standard Library*/
#include <vector>
#include <string>

/*Class for Storing and Retrieving Pose linked to a unique Frame/Model Pair*/
class PoseMatrix {
public:
	/*Blank Constructor/Destructor*/
	__declspec(dllexport) PoseMatrix();
	__declspec(dllexport) ~PoseMatrix();

	/*Add New Model to Pose Matrix*/
	__declspec(dllexport) void AddModel(std::vector<gpu_cost_function::Pose > , std::string model_name, bool is_principal_model);

	/*Get Model Pose (True if Successful, Else False) - Pose is Returned by Passing via reference*/
	__declspec(dllexport) bool GetModelPose(std::string model_name, int frame_index, gpu_cost_function::Pose* pose_container);
	/*Get Principal Model Pose*/
	__declspec(dllexport) bool GetModelPose(int frame_index, gpu_cost_function::Pose* pose_container);
	/*Update Stored Pose for Principal Model at given frame*/
	__declspec(dllexport) bool UpdatePrincipalModelPose(int frame_index, gpu_cost_function::Pose pose_container);

private:
	/*Principal Model Name*/
	std::string principal_model_name_;

	/*Principal Model Lookup Index*/
	int principal_model_index_;

	/*Vector of All Models (Order Implies Index)*/
	std::vector<std::string> model_names_;

	/*Vector of All Model's Vector of Frame Poses
	Size of pose_matrix_ = # of models by # of frames*/
	std::vector<std::vector<gpu_cost_function::Pose>> pose_matrix_;

};