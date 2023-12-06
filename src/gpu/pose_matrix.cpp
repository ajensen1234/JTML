// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*Pose Matrix Header*/
#include "pose_matrix.h"

/*Constructor/Destructor*/
PoseMatrix::PoseMatrix() {
	/*Nonsensical Index For Error Querying*/
	principal_model_index_ = -1;
};

PoseMatrix::~PoseMatrix() {
};

/*Add New Model to Pose Matrix*/
void PoseMatrix::AddModel(std::vector<gpu_cost_function::Pose> model_poses, std::string model_name,
                          bool is_principal_model) {
	/*If Principal Model*/
	if (is_principal_model) {
		principal_model_index_ = model_names_.size();
		principal_model_name_ = model_name;
	}

	/*Add to Storage*/
	model_names_.push_back(model_name);
	pose_matrix_.push_back(model_poses);
};

/*Get Model Pose (True if Successful, Else False) - Pose is Returned by Passing via reference*/
bool PoseMatrix::GetModelPose(std::string model_name, int frame_index, gpu_cost_function::Pose* pose_container) {
	/*Check if Pose Mat Empty or Frame Index Out of Bounds */
	if (pose_matrix_.size() == 0)
		return false;
	if (pose_matrix_[0].size() <= frame_index)
		return false;

	/*Get Model Index*/
	int model_index = -1;
	for (int i = 0; i < model_names_.size(); i++) {
		if (model_names_[i] == model_name)
			model_index = i;
	}

	if (model_index == -1) {
		return false;
	}
	*pose_container = pose_matrix_.at(model_index).at(frame_index);
	return true;

};

/*Gets Principal Model Pose*/
bool PoseMatrix::GetModelPose(int frame_index, gpu_cost_function::Pose* pose_container) {
	/*Check if Pose Mat Empty or Frame Index Out of Bounds */
	if (pose_matrix_.size() == 0)
		return false;
	if (pose_matrix_[0].size() <= frame_index)
		return false;

	/*Get Pose*/
	if (principal_model_index_ == -1) {
		return false;
	}
	*pose_container = pose_matrix_.at(principal_model_index_).at(frame_index);
	return true;
};
/*Update Stored Pose for Principal Model at given frame*/
bool PoseMatrix::UpdatePrincipalModelPose(int frame_index, gpu_cost_function::Pose pose_container) {
	pose_matrix_.at(principal_model_index_).at(frame_index) = pose_container;
	return true;
};
