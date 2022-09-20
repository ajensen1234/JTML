/*Optimizer Manaer*/
#include "core/optimizer_manager.h"

/*Pose Matrix Class*/
#include "pose_matrix.h"
#include<stdlib.h>
#include<chrono>
#include<thread>

OptimizerManager::OptimizerManager(QObject* parent) :
	QObject(parent)
{
	//this->sym_trap_obj = nullptr;
}

/*Initialize*/
bool OptimizerManager::Initialize(
	QThread& optimizer_thread,
	Calibration calibration_file,
	std::vector<Frame> camera_A_frame_list, std::vector<Frame> camera_B_frame_list, unsigned int current_frame_index,
	std::vector<Model> model_list, QModelIndexList selected_models, unsigned int primary_model_index,
	LocationStorage pose_matrix,
	OptimizerSettings opt_settings,
	jta_cost_function::CostFunctionManager trunk_manager, jta_cost_function::CostFunctionManager branch_manager, jta_cost_function::CostFunctionManager leaf_manager,
	QString opt_directive,
	QString& error_message,
	int iter_count) {

/*	if (this->sym_trap_obj == nullptr && sym_trap_obj_ != nullptr) {
		this->sym_trap_obj = sym_trap_obj_;
	} */

	/*Success?*/
	succesfull_initialization_ = true;

	/*Error Check for Optimizer*/
	error_occurrred_ = false;

	/*Set up Thread Connections*/
	/*Connect Start of Thread to Optimisation Loop and Emergency Stop*/
	QObject::connect(&optimizer_thread, SIGNAL(started()), this, SLOT(Optimize()));

	/*Destructor Connections*/
	QObject::connect(this, SIGNAL(finished()), &optimizer_thread, SLOT(quit()));
	QObject::connect(this, SIGNAL(finished()), this, SLOT(deleteLater()));
	QObject::connect(&optimizer_thread, SIGNAL(finished()), &optimizer_thread, SLOT(deleteLater()));

	/*Store Calibration File Locally*/
	calibration_ = calibration_file;
	optimization_directive_ = opt_directive;

	/*Just In Case Have to Delete*/
	gpu_principal_model_ = 0;
	gpu_metrics_ = 0;
	
	/*Store Camera Frame Lists Locally and Check That, if Biplane is Enabled -> both lists are the same size.
	Also Check that the current frame index is within the range of the frame list sizes.*/
	frames_A_ = camera_A_frame_list;
	frames_B_ = camera_B_frame_list;
	if (calibration_.biplane_calibration && frames_A_.size() != frames_B_.size()) {
		error_message = "Biplane mode enabled, but each camera has a different number of frames!";
		succesfull_initialization_ = false;
		return succesfull_initialization_;
	}
	if (current_frame_index >= frames_A_.size() || (current_frame_index >= frames_B_.size() && calibration_.biplane_calibration)) {
		error_message = "Current frame index is out of scope!";
		succesfull_initialization_ = false;
		return succesfull_initialization_;
	}
	
	/*Store Model List, Non-Primary Selected Models (Blue) and Primary Model
	Also store indices of all models and the index of the primary model*/
	all_models_ = model_list;
	if (selected_models.size() == 0 || selected_models[0].row() != primary_model_index) {
		error_message = "Can't find primary model!";
		succesfull_initialization_ = false;
		return succesfull_initialization_;
	}
	for (int i = 1; i < selected_models.size(); i++) {
		selected_non_primary_models_.push_back(all_models_[selected_models[i].row()]);
	}
	primary_model_ = all_models_[primary_model_index];
	selected_model_list_ = selected_models;
	primary_model_index_ = primary_model_index;

	/*Store Optimizer Settings Locally*/
	optimizer_settings_ = opt_settings;

	/*Store Cost Function Managers Locally*/
	trunk_manager_ = trunk_manager;
	branch_manager_ = branch_manager;
	leaf_manager_ = leaf_manager;

	/*Store Post Matrix on Cost Functions*/
	for (int i = 0; i < selected_models.size(); i++) {
		/*Construct Vector of Poses for Each Frame for Model*/
		int index_for_model = selected_models[i].row();
		std::vector<Pose> poses_each_frame_for_given_model;
		for (int j = 0; j < pose_matrix.GetFrameCount(); j++) {
			Point6D temp_p6d = pose_matrix.GetPose(j, index_for_model);
			Pose temp_pose = Pose(temp_p6d.x, temp_p6d.y, temp_p6d.z,
				temp_p6d.xa, temp_p6d.ya, temp_p6d.za);
			poses_each_frame_for_given_model.push_back(temp_pose);
		}
		/*If i ==0, principal model*/
		if (i == 0) {
			pose_storage_.AddModel(poses_each_frame_for_given_model, all_models_[index_for_model].model_name_, true);
		}
		else {
			pose_storage_.AddModel(poses_each_frame_for_given_model, all_models_[index_for_model].model_name_, false);
		}
	}

	sym_trap_call = false;

	/*Use Optimization Directive To Resolve the Following local variables*/
	if (opt_directive == "Single") {
		/*Should we progess to next frame?*/
		progress_next_frame_ = false;
		/*Should we initialize with previous frame's best guess?*/
		init_prev_frame_ = false;
		/*Index For Starting Frame in Optimization*/
		start_frame_index_ = current_frame_index;
		end_frame_index_ = current_frame_index;
		
	}
	else if (opt_directive == "All") {
		/*Should we progess to next frame?*/
		progress_next_frame_ = true;
		/*Should we initialize with previous frame's best guess?*/
		init_prev_frame_ = true;
		/*Index For Starting Frame in Optimization*/
		start_frame_index_ = 0;
		end_frame_index_ = frames_A_.size() - 1;
	} 
	else if (opt_directive == "Each") {
		/*Should we progess to next frame?*/
		progress_next_frame_ = true;
		/*Should we initialize with previous frame's best guess?*/
		init_prev_frame_ = false;
		/*Index For Starting Frame in Optimization*/
		start_frame_index_ = 0;
		end_frame_index_ = frames_A_.size() -1;
	}
	else if (opt_directive == "From") {
		/*Should we progess to next frame?*/
		progress_next_frame_ = true;
		/*Should we initialize with previous frame's best guess?*/
		init_prev_frame_ = true;
		/*Index For Starting Frame in Optimization*/
		start_frame_index_ = current_frame_index;
		end_frame_index_ = frames_A_.size() - 1;
	}
	else if (opt_directive == "Sym_Trap") {
		/*Should we progess to next frame?*/
		progress_next_frame_ = false;
		/*Should we initialize with previous frame's best guess?*/
		init_prev_frame_ = false;
		/*Index For Starting Frame in Optimization*/
		start_frame_index_ = current_frame_index;

		sym_trap_call = true;
	}
	else if (opt_directive == "Backward") {
		progress_next_frame_ = true;
		init_prev_frame_ = true;
		start_frame_index_ = current_frame_index;
		end_frame_index_ = 0;
	}
	else {
		error_message = "Unrecognized optimization directive: " + opt_directive;
		succesfull_initialization_ = false;
		return succesfull_initialization_;
	}
	/*Setting Up image indices based on the directive launched*/
	create_image_indices(img_indices_, start_frame_index_, end_frame_index_);
	/*Set Up Settings*/
	SetSearchRange(optimizer_settings_.trunk_range);
	SetStartingPoint(pose_matrix.GetPose(start_frame_index_, primary_model_index_));
	budget_ = optimizer_settings_.trunk_budget;
	cost_function_calls_ = 0;
	current_optimum_value_ = DBL_MAX;
	current_optimum_location_ = starting_point_;

	/*Initialize GPU CUDA Cost Function Library Tools*/
	/*Get Width and Height (Safe since did error check before launching this function)*/
	int width = frames_A_[0].GetEdgeImage().cols;
	int height = frames_A_[0].GetEdgeImage().rows;

	/*Check CUDA Compatibility*/
	int cuda_device_id = 0, gpu_device_count = 0, device_count;
	struct cudaDeviceProp properties;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&device_count);
	if (cudaResultCode != cudaSuccess)
		device_count = 0;
	/* Machines with no GPUs can still report one emulation device */
	for (int device = 0; device < device_count; ++device) {
		cudaGetDeviceProperties(&properties, device);
		if (properties.major != 9999 && properties.major >= 5) /* 9999 means emulation only */
			++gpu_device_count;
	}
	/*If no Cuda Compatitble Devices with Compute Capability Greater Than 5, Exit*/
	if (gpu_device_count == 0) {
		error_message = "No Cuda Compatitble Devices with Compute Capability Greater Than 5!";
		succesfull_initialization_ = false;
		return succesfull_initialization_;
	}

	/*Get Dilation Values for Trunk, Branch, and Leaf*/
	/*Trunk*/
	trunk_dilation_val_ = 0;
	std::vector<jta_cost_function::Parameter<int>> active_int_params = trunk_manager_.getActiveCostFunctionClass()->getIntParameters();
	for (int i = 0; i < active_int_params.size(); i++) {
		if (active_int_params[i].getParameterName() == "Dilation" || active_int_params[i].getParameterName() == "DILATION" || active_int_params[i].getParameterName() == "dilation") {
			trunk_dilation_val_ = trunk_manager_.getActiveCostFunctionClass()->getIntParameters().at(i).getParameterValue();
		}
	}
	if (trunk_dilation_val_ <= 0) trunk_dilation_val_ = 0;
	/*Check Special Mahfouz Case*/
	if (trunk_manager_.getActiveCostFunction() == "DIRECT_MAHFOUZ") trunk_dilation_val_ = 3;
	/*Branch*/
	branch_dilation_val_ = 0;
	active_int_params = branch_manager_.getActiveCostFunctionClass()->getIntParameters();
	for (int i = 0; i < active_int_params.size(); i++) {
		if (active_int_params[i].getParameterName() == "Dilation" || active_int_params[i].getParameterName() == "DILATION" || active_int_params[i].getParameterName() == "dilation") {
			branch_dilation_val_ = branch_manager_.getActiveCostFunctionClass()->getIntParameters().at(i).getParameterValue();
		}
	}
	if (branch_dilation_val_ <= 0) branch_dilation_val_ = 0;
	/*Check Special Mahfouz Case*/
	if (branch_manager_.getActiveCostFunction() == "DIRECT_MAHFOUZ") branch_dilation_val_ = 3;
	/*Leaf*/
	leaf_dilation_val_ = 0;
	active_int_params = leaf_manager_.getActiveCostFunctionClass()->getIntParameters();
	for (int i = 0; i < active_int_params.size(); i++) {
		if (active_int_params[i].getParameterName() == "Dilation" || active_int_params[i].getParameterName() == "DILATION" || active_int_params[i].getParameterName() == "dilation") {
			leaf_dilation_val_ = leaf_manager_.getActiveCostFunctionClass()->getIntParameters().at(i).getParameterValue();
		}
	}
	if (leaf_dilation_val_ <= 0) leaf_dilation_val_ = 0;
	/*Check Special Mahfouz Case*/
	if (leaf_manager_.getActiveCostFunction() == "DIRECT_MAHFOUZ") leaf_dilation_val_ = 3;

	/*Get Black Silhouette? Values for Trunk, Branch, and Leaf*/
	/*Black Silhouette Values Based on Bool Parameter Names (Black_Silhouette or Dark_Silhouette or BLACK_SILHOUETTE or DARK_SILHOUETTE or black_silhouette or dark_silhouette)*/
	/*Trunk*/
	trunk_dark_silhouette_val_ = false;
	std::vector<jta_cost_function::Parameter<bool>> active_bool_params = trunk_manager_.getActiveCostFunctionClass()->getBoolParameters();
	for (int i = 0; i < active_bool_params.size(); i++) {
		if (active_bool_params[i].getParameterName() == "Black_Silhouette" || active_bool_params[i].getParameterName() == "Dark_Silhouette" ||
			active_bool_params[i].getParameterName() == "BLACK_SILHOUETTE" || active_bool_params[i].getParameterName() == "DARK_SILHOUETTE" ||
			active_bool_params[i].getParameterName() == "black_silhouette" || active_bool_params[i].getParameterName() == "dark_silhouette") {
			trunk_dark_silhouette_val_ = trunk_manager_.getActiveCostFunctionClass()->getBoolParameters().at(i).getParameterValue();
		}
	}
	/*Branch*/
	branch_dark_silhouette_val_ = false;
	active_bool_params = branch_manager_.getActiveCostFunctionClass()->getBoolParameters();
	for (int i = 0; i < active_bool_params.size(); i++) {
		if (active_bool_params[i].getParameterName() == "Black_Silhouette" || active_bool_params[i].getParameterName() == "Dark_Silhouette" ||
			active_bool_params[i].getParameterName() == "BLACK_SILHOUETTE" || active_bool_params[i].getParameterName() == "DARK_SILHOUETTE" ||
			active_bool_params[i].getParameterName() == "black_silhouette" || active_bool_params[i].getParameterName() == "dark_silhouette") {
			branch_dark_silhouette_val_ = branch_manager_.getActiveCostFunctionClass()->getBoolParameters().at(i).getParameterValue();
		}
	}
	/*Leaf*/
	leaf_dark_silhouette_val_ = false;
	active_bool_params = leaf_manager_.getActiveCostFunctionClass()->getBoolParameters();
	for (int i = 0; i < active_bool_params.size(); i++) {
		if (active_bool_params[i].getParameterName() == "Black_Silhouette" || active_bool_params[i].getParameterName() == "Dark_Silhouette" ||
			active_bool_params[i].getParameterName() == "BLACK_SILHOUETTE" || active_bool_params[i].getParameterName() == "DARK_SILHOUETTE" ||
			active_bool_params[i].getParameterName() == "black_silhouette" || active_bool_params[i].getParameterName() == "dark_silhouette") {
			leaf_dark_silhouette_val_ = leaf_manager_.getActiveCostFunctionClass()->getBoolParameters().at(i).getParameterValue();
		}
	}

	/*Upload GPU Frames*/
	/*Intensity Frames
	/*Trunk*/
	/*Camera A*/
	for (int i = 0; i < frames_A_.size(); i++) {
		GPUIntensityFrame* intensity_frame = new GPUIntensityFrame(width, height, cuda_device_id,
			frames_A_[i].GetOriginalImage().data, trunk_dark_silhouette_val_, frames_A_[i].GetInvertedImage().data);
		if (intensity_frame->IsInitializedCorrectly()) {
			gpu_intensity_frames_trunk_A_.push_back(intensity_frame);
		}
		else {
			delete intensity_frame;
			error_message = "Error uploading intensity frame to GPU!";
			succesfull_initialization_ = false;
			return succesfull_initialization_;
		}
	}
	/*Camera B*/
	if (calibration_.biplane_calibration) {
		for (int i = 0; i < frames_B_.size(); i++) {
			GPUIntensityFrame* intensity_frame = new GPUIntensityFrame(width, height, cuda_device_id,
				frames_B_[i].GetOriginalImage().data, trunk_dark_silhouette_val_, frames_B_[i].GetInvertedImage().data);
			if (intensity_frame->IsInitializedCorrectly()) {
				gpu_intensity_frames_trunk_B_.push_back(intensity_frame);
			}
			else {
				delete intensity_frame;
				error_message = "Error uploading intensity frame to GPU!";
				succesfull_initialization_ = false;
				return succesfull_initialization_;
			}
		}
	}
	/*Branch*/
	/*Camera A*/
	for (int i = 0; i < frames_A_.size(); i++) {
		GPUIntensityFrame* intensity_frame = new GPUIntensityFrame(width, height, cuda_device_id,
			frames_A_[i].GetOriginalImage().data, branch_dark_silhouette_val_, frames_A_[i].GetInvertedImage().data);
		if (intensity_frame->IsInitializedCorrectly()) {
			gpu_intensity_frames_branch_A_.push_back(intensity_frame);
		}
		else {
			delete intensity_frame;
			error_message = "Error uploading intensity frame to GPU!";
			succesfull_initialization_ = false;
			return succesfull_initialization_;
		}
	}
	/*Camera B*/
	if (calibration_.biplane_calibration) {
		for (int i = 0; i < frames_B_.size(); i++) {
			GPUIntensityFrame* intensity_frame = new GPUIntensityFrame(width, height, cuda_device_id,
				frames_B_[i].GetOriginalImage().data, branch_dark_silhouette_val_, frames_B_[i].GetInvertedImage().data);
			if (intensity_frame->IsInitializedCorrectly()) {
				gpu_intensity_frames_branch_B_.push_back(intensity_frame);
			}
			else {
				delete intensity_frame;
				error_message = "Error uploading intensity frame to GPU!";
				succesfull_initialization_ = false;
				return succesfull_initialization_;
			}
		}
	}
	/*Leaf*/
	/*Camera A*/
	for (int i = 0; i < frames_A_.size(); i++) {
		GPUIntensityFrame* intensity_frame = new GPUIntensityFrame(width, height, cuda_device_id,
			frames_A_[i].GetOriginalImage().data, leaf_dark_silhouette_val_, frames_A_[i].GetInvertedImage().data);
		if (intensity_frame->IsInitializedCorrectly()) {
			gpu_intensity_frames_leaf_A_.push_back(intensity_frame);
		}
		else {
			delete intensity_frame;
			error_message = "Error uploading intensity frame to GPU!";
			succesfull_initialization_ = false;
			return succesfull_initialization_;
		}
	}
	/*Camera B*/
	if (calibration_.biplane_calibration) {
		for (int i = 0; i < frames_B_.size(); i++) {
			GPUIntensityFrame* intensity_frame = new GPUIntensityFrame(width, height, cuda_device_id,
				frames_B_[i].GetOriginalImage().data, leaf_dark_silhouette_val_, frames_B_[i].GetInvertedImage().data);
			if (intensity_frame->IsInitializedCorrectly()) {
				gpu_intensity_frames_leaf_B_.push_back(intensity_frame);
			}
			else {
				delete intensity_frame;
				error_message = "Error uploading intensity frame to GPU!";
				succesfull_initialization_ = false;
				return succesfull_initialization_;
			}
		}
	}

	/*Dilation Frames*/
	/*Reverse Order So That The Dilation Images Show Trunk Values*/
	/*Leaf*/
	/*Camera A*/
	/*Update Dilation Images to Leaf Mode*/
	for (int i = 0; i < frames_A_.size(); i++) {
		cv::dilate(frames_A_[i].GetEdgeImage(), frames_A_[i].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), leaf_dilation_val_); /*Reset Dilation In That Image*/
	}
	for (int i = 0; i < frames_A_.size(); i++) {
		GPUDilatedFrame* dilated_frame = new GPUDilatedFrame(width, height, cuda_device_id,
			frames_A_[i].GetDilationImage().data, leaf_dilation_val_);
		if (dilated_frame->IsInitializedCorrectly()) {
			gpu_dilated_frames_leaf_A_.push_back(dilated_frame);
		}
		else {
			delete dilated_frame;
			error_message = "Error uploading dilated frame to GPU!";
			succesfull_initialization_ = false;
			return succesfull_initialization_;
		}
	}
	/*Camera B*/
	if (calibration_.biplane_calibration) {
		for (int i = 0; i < frames_B_.size(); i++) {
			cv::dilate(frames_B_[i].GetEdgeImage(), frames_B_[i].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), leaf_dilation_val_); /*Reset Dilation In That Image*/
		}
	}
	if (calibration_.biplane_calibration) {
		for (int i = 0; i < frames_B_.size(); i++) {
			GPUDilatedFrame* dilated_frame = new GPUDilatedFrame(width, height, cuda_device_id,
				frames_B_[i].GetDilationImage().data, leaf_dilation_val_);
			if (dilated_frame->IsInitializedCorrectly()) {
				gpu_dilated_frames_leaf_B_.push_back(dilated_frame);
			}
			else {
				delete dilated_frame;
				error_message = "Error uploading dilated frame to GPU!";
				succesfull_initialization_ = false;
				return succesfull_initialization_;
			}
		}
	}
	/*Branch*/
	/*Camera A*/
	for (int i = 0; i < frames_A_.size(); i++) {
		cv::dilate(frames_A_[i].GetEdgeImage(), frames_A_[i].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), branch_dilation_val_); /*Reset Dilation In That Image*/
	}
	for (int i = 0; i < frames_A_.size(); i++) {
		GPUDilatedFrame* dilated_frame = new GPUDilatedFrame(width, height, cuda_device_id,
			frames_A_[i].GetDilationImage().data, branch_dilation_val_);
		if (dilated_frame->IsInitializedCorrectly()) {
			gpu_dilated_frames_branch_A_.push_back(dilated_frame);
		}
		else {
			delete dilated_frame;
			error_message = "Error uploading dilated frame to GPU!";
			succesfull_initialization_ = false;
			return succesfull_initialization_;
		}
	}
	/*Camera B*/
	if (calibration_.biplane_calibration) {
		for (int i = 0; i < frames_B_.size(); i++) {
			cv::dilate(frames_B_[i].GetEdgeImage(), frames_B_[i].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), branch_dilation_val_); /*Reset Dilation In That Image*/
		}
	}
	if (calibration_.biplane_calibration) {
		for (int i = 0; i < frames_B_.size(); i++) {
			GPUDilatedFrame* dilated_frame = new GPUDilatedFrame(width, height, cuda_device_id,
				frames_B_[i].GetDilationImage().data, branch_dilation_val_);
			if (dilated_frame->IsInitializedCorrectly()) {
				gpu_dilated_frames_branch_B_.push_back(dilated_frame);
			}
			else {
				delete dilated_frame;
				error_message = "Error uploading dilated frame to GPU!";
				succesfull_initialization_ = false;
				return succesfull_initialization_;
			}
		}
	}
	/*Trunk*/
	/*Camera A*/
	for (int i = 0; i < frames_A_.size(); i++) {
		cv::dilate(frames_A_[i].GetEdgeImage(), frames_A_[i].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), trunk_dilation_val_); /*Reset Dilation In That Image*/
	}
	for (int i = 0; i < frames_A_.size(); i++) {
		GPUDilatedFrame* dilated_frame = new GPUDilatedFrame(width, height, cuda_device_id,
			frames_A_[i].GetDilationImage().data, trunk_dilation_val_);
		if (dilated_frame->IsInitializedCorrectly()) {
			gpu_dilated_frames_trunk_A_.push_back(dilated_frame);
		}
		else {
			delete dilated_frame;
			error_message = "Error uploading dilated frame to GPU!";
			succesfull_initialization_ = false;
			return succesfull_initialization_;
		}
	}
	/*Camera B*/
	if (calibration_.biplane_calibration) {
		for (int i = 0; i < frames_B_.size(); i++) {
			cv::dilate(frames_B_[i].GetEdgeImage(), frames_B_[i].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), trunk_dilation_val_); /*Reset Dilation In That Image*/
		}
	}
	if (calibration_.biplane_calibration) {
		for (int i = 0; i < frames_B_.size(); i++) {
			GPUDilatedFrame* dilated_frame = new GPUDilatedFrame(width, height, cuda_device_id,
				frames_B_[i].GetDilationImage().data, trunk_dilation_val_);
			if (dilated_frame->IsInitializedCorrectly()) {
				gpu_dilated_frames_trunk_B_.push_back(dilated_frame);
			}
			else {
				delete dilated_frame;
				error_message = "Error uploading dilated frame to GPU!";
				succesfull_initialization_ = false;
				return succesfull_initialization_;
			}
		}
	}

	/*Edge Frames*/
	/*Camera A*/
	for (int i = 0; i < frames_A_.size(); i++) {
		GPUEdgeFrame* edge_frame = new GPUEdgeFrame(width, height, cuda_device_id, frames_A_[i].GetEdgeImage().data,
			frames_A_[i].GetHighThreshold(), frames_A_[i].GetLowThreshold(), frames_A_[i].GetAperture());
		if (edge_frame->IsInitializedCorrectly()) {
			gpu_edge_frames_A_.push_back(edge_frame);
		}
		else {
			delete edge_frame;
			error_message = "Error uploading edge frame to GPU!";
			succesfull_initialization_ = false;
			return succesfull_initialization_;
		}
	}
	/*Camera B*/
	if (calibration_.biplane_calibration) {
		for (int i = 0; i < frames_B_.size(); i++) {
			GPUEdgeFrame* edge_frame = new GPUEdgeFrame(width, height, cuda_device_id, frames_B_[i].GetEdgeImage().data,
				frames_B_[i].GetHighThreshold(), frames_B_[i].GetLowThreshold(), frames_B_[i].GetAperture());
			if (edge_frame->IsInitializedCorrectly()) {
				gpu_edge_frames_B_.push_back(edge_frame);
			}
			else {
				delete edge_frame;
				error_message = "Error uploading edge frame to GPU!";
				succesfull_initialization_ = false;
				return succesfull_initialization_;
			}
		}
	}

	/*Upload GPU Models*/
	/*Monoplane Calibration*/
	if (!calibration_.biplane_calibration) {
		/*Principal Model*/
		gpu_principal_model_ = new GPUModel(primary_model_.model_name_, true, width, height, cuda_device_id, true,
			&primary_model_.triangle_vertices_[0], &primary_model_.triangle_normals_[0], primary_model_.triangle_vertices_.size() / 9, calibration_.camera_A_principal_);
		if (!gpu_principal_model_->IsInitializedCorrectly()) {
			delete gpu_principal_model_;
			gpu_principal_model_ = 0;
			error_message = "Error uploading principal model to GPU!";
			succesfull_initialization_ = false;
			return succesfull_initialization_;
		}
		/*Non-principal models*/
		for (int i = 1; i < selected_model_list_.size(); i++) {
			GPUModel* gpu_non_principal_model = new GPUModel(all_models_[selected_model_list_[i].row()].model_name_, true, width, height, cuda_device_id, true,
				&all_models_[selected_model_list_[i].row()].triangle_vertices_[0], &all_models_[selected_model_list_[i].row()].triangle_normals_[0], all_models_[selected_model_list_[i].row()].triangle_vertices_.size() / 9, calibration_.camera_A_principal_);
			if (gpu_non_principal_model->IsInitializedCorrectly()) {
				gpu_non_principal_models_.push_back(gpu_non_principal_model);
			}
			else {
				delete gpu_non_principal_model;
				error_message = "Error uploading non-principal model to GPU!";
				succesfull_initialization_ = false;
				return succesfull_initialization_;
			}
		}
	}
	/*Biplane Calibration*/
	else {
		/*Principal Model*/
		gpu_principal_model_ = new GPUModel(primary_model_.model_name_, true, width, height, cuda_device_id, cuda_device_id, true, true,
			&primary_model_.triangle_vertices_[0], &primary_model_.triangle_normals_[0], primary_model_.triangle_vertices_.size() / 9, calibration_.camera_A_principal_, calibration_.camera_B_principal_);
		if (!gpu_principal_model_->IsInitializedCorrectly()) {
			delete gpu_principal_model_;
			gpu_principal_model_ = 0;
			error_message = "Error uploading principal model to GPU!";
			succesfull_initialization_ = false;
			return succesfull_initialization_;
		}
		/*Non-principal models*/
		for (int i = 1; i < selected_model_list_.size(); i++) {
			GPUModel* gpu_non_principal_model = new GPUModel(all_models_[selected_model_list_[i].row()].model_name_, true, width, height, cuda_device_id, cuda_device_id, true, true,
				&all_models_[selected_model_list_[i].row()].triangle_vertices_[0], &all_models_[selected_model_list_[i].row()].triangle_normals_[0], all_models_[selected_model_list_[i].row()].triangle_vertices_.size() / 9, calibration_.camera_A_principal_, calibration_.camera_B_principal_);
			if (gpu_non_principal_model->IsInitializedCorrectly()) {
				gpu_non_principal_models_.push_back(gpu_non_principal_model);
			}
			else {
				delete gpu_non_principal_model;
				error_message = "Error uploading non-principal model to GPU!";
				succesfull_initialization_ = false;
				return succesfull_initialization_;
			}
		}
	}

	/*Initialize GPU Metrics*/
	gpu_metrics_ = new GPUMetrics();
	if (!gpu_metrics_->IsInitializedCorrectly()) {
		error_message = "GPU metrics class not initialized correctly!";
		succesfull_initialization_ = false;
		return succesfull_initialization_;
	}

	/*Upload Data To CostFunction Managers*/
	trunk_manager_.UploadData(&gpu_edge_frames_A_,
		&gpu_dilated_frames_trunk_A_,
		&gpu_intensity_frames_trunk_A_,
		&gpu_edge_frames_B_,
		&gpu_dilated_frames_trunk_B_,
		&gpu_intensity_frames_trunk_B_,
		gpu_principal_model_,
		&gpu_non_principal_models_,
		gpu_metrics_,
		&pose_storage_,
		calibration_.biplane_calibration);
	branch_manager_.UploadData(&gpu_edge_frames_A_,
		&gpu_dilated_frames_branch_A_,
		&gpu_intensity_frames_branch_A_,
		&gpu_edge_frames_B_,
		&gpu_dilated_frames_branch_B_,
		&gpu_intensity_frames_branch_B_,
		gpu_principal_model_,
		&gpu_non_principal_models_,
		gpu_metrics_,
		&pose_storage_,
		calibration_.biplane_calibration);
	leaf_manager_.UploadData(&gpu_edge_frames_A_,
		&gpu_dilated_frames_leaf_A_,
		&gpu_intensity_frames_leaf_A_,
		&gpu_edge_frames_B_,
		&gpu_dilated_frames_leaf_B_,
		&gpu_intensity_frames_leaf_B_,
		gpu_principal_model_,
		&gpu_non_principal_models_,
		gpu_metrics_,
		&pose_storage_,
		calibration_.biplane_calibration);

	return succesfull_initialization_;
};

void OptimizerManager::SetSearchRange(Point6D range) {
	/*Check Search Range is Not Zero*/
	if (range.x + range.y + range.z + range.xa + range.ya + range.za > 0) {
		range_ = range;
		valid_range_ = true;
	}
	else
		valid_range_ = false;
}

void OptimizerManager::SetStartingPoint(Point6D starting_point) {
	starting_point_ = starting_point;
}

void OptimizerManager::Optimize() {

	/*Check That Succesfull Initialization*/
	if (!succesfull_initialization_) {

		/*Restore Dilation OpenCV Images*/
		for (int i = 0; i < frames_A_.size(); i++) {
			cv::dilate(frames_A_[i].GetEdgeImage(), frames_A_[i].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), trunk_dilation_val_); /*Reset Dilation In That Image*/
		}
		/*Camera B*/
		if (calibration_.biplane_calibration) {
			for (int i = 0; i < frames_B_.size(); i++) {
				cv::dilate(frames_B_[i].GetEdgeImage(), frames_B_[i].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), trunk_dilation_val_); /*Reset Dilation In That Image*/
			}
		}

		/*Finish And Return Dont Have to Delete the Renderer and Metric as This has Been Done*/
		emit finished();
		return;
	}

	/*Container for String Message*/
	std::string error_message;

	/*Loop Over Each Frame Loaded*/
	for (int frame_index : img_indices_) {
		if (!sym_trap_call) {

			/*Set Up Search Range and Starting Point*/
			SetSearchRange(optimizer_settings_.trunk_range);
			if (!init_prev_frame_ || frame_index == 0) {
				Pose starting_pose;
				pose_storage_.GetModelPose(frame_index, &starting_pose);
				SetStartingPoint(Point6D(starting_pose.x_location_, starting_pose.y_location_, starting_pose.z_location_,
					starting_pose.x_angle_, starting_pose.y_angle_, starting_pose.z_angle_));
			}
			else SetStartingPoint(current_optimum_location_);

			/*Set Current Primary (and if biplane, secondary) Poses for Non Principal Models*/
			for (int non_prin_model_ind = 0; non_prin_model_ind < gpu_non_principal_models_.size(); non_prin_model_ind++) {
				Pose temp_primary_pose;
				if (pose_storage_.GetModelPose(gpu_non_principal_models_[non_prin_model_ind]->GetModelName(), frame_index, &temp_primary_pose)) {
					gpu_non_principal_models_[non_prin_model_ind]->SetCurrentPrimaryCameraPose(temp_primary_pose);
				}
				else {
					emit OptimizerError(QString::fromStdString("Could not retrieve pose for non-principal model \"" + gpu_non_principal_models_[non_prin_model_ind]->GetModelName() + "\" at frame " + QString::number(frame_index).toStdString() + "!"));
					error_occurrred_ = true;
					break;
				}
				if (calibration_.biplane_calibration) {
					Point6D temp_primary_point = Point6D(temp_primary_pose.x_location_, temp_primary_pose.y_location_, temp_primary_pose.z_location_,
						temp_primary_pose.x_angle_, temp_primary_pose.y_angle_, temp_primary_pose.z_angle_);
					Point6D temp_secondary_point = calibration_.convert_Pose_A_to_Pose_B(temp_primary_point);
					gpu_non_principal_models_[non_prin_model_ind]->SetCurrentSecondaryCameraPose(Pose(temp_secondary_point.x, temp_secondary_point.y, temp_secondary_point.z,
						temp_secondary_point.xa, temp_secondary_point.ya, temp_secondary_point.za));
				}
			}

			/*Set Current Frame Index for CFMs*/
			trunk_manager_.setCurrentFrameIndex(frame_index);
			branch_manager_.setCurrentFrameIndex(frame_index);
			leaf_manager_.setCurrentFrameIndex(frame_index);

			/*Reset Budget and Cost Function Calls*/
			budget_ = optimizer_settings_.trunk_budget;
			cost_function_calls_ = 0;

			/*Initialize Search Stage Flag as Trunk*/
			search_stage_flag_ = SearchStageFlag::Trunk;

			/*Start Clock*/
			start_clock_ = clock();
			update_screen_clock_ = clock();

			/*****************TRUNK SECTION BEGIN **********************/
			/*Call Trunk Initializer*/
			if (!trunk_manager_.InitializeActiveCostFunction(error_message)) {
				emit OptimizerError(QString::fromStdString(error_message));
				error_occurrred_ = true;
			}

			/*Initialize with Unit Sized HyperBox at Center*/
			if (!error_occurrred_) {
				current_optimum_value_ = EvaluateCostFunction(Point6D(.5, .5, .5, .5, .5, .5));
				current_optimum_location_ = starting_point_;
				data_ = DirectDataStorage(current_optimum_value_);
			}

			/*Make Sure Dilation Image is Showing Trunk Value (Should be Unnecessary)*/
			cv::dilate(frames_A_[frame_index].GetEdgeImage(), frames_A_[frame_index].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), trunk_dilation_val_); /*Reset Dilation In That Image*/
			if (calibration_.biplane_calibration) {
				cv::dilate(frames_B_[frame_index].GetEdgeImage(), frames_B_[frame_index].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), trunk_dilation_val_); /*Reset Dilation In That Image*/
			}
			emit UpdateDilationBackground();

			/*Main Loop*/
			if (!error_occurrred_) {
				while (cost_function_calls_ < budget_) {
					/*Scroll Through Convex Hull to Get List of Potentially
					Optimal Hyperboxes to Evaluate. This is stored as list of
					column IDs in DATA that need to have least Fvalued
					hyperbox (last in list) returned for trisection and evaluation.*/
					ConvexHull();

					/*Trisect Potentially Optimal Rectangles*/
					TrisectPotentiallyOptimal();

					/*Safety Break...Should Never Happen*/
					if (potentially_optimal_col_ids_.size() == 0) {
						emit OptimizerError("Error, no potentially optimal hyper rectangles found!");
						error_occurrred_ = true;
						break;
					}

					/*If Error*/
					if (error_occurrred_) break;

					/*Update Screen at Rate of 30 FPS*/
					if ((clock() - update_screen_clock_) > 33) {
						emit UpdateDisplay((double)(clock() - start_clock_) / (double)cost_function_calls_, (int)cost_function_calls_, current_optimum_value_, primary_model_index_);
						update_screen_clock_ = clock();
					}
				}
			}

			/*Destruct Trunk Manager Initialization*/
			if (!trunk_manager_.DestructActiveCostFunction(error_message)) {
				emit OptimizerError(QString::fromStdString(error_message));
				error_occurrred_ = true;
			}
			/*****************TRUNK SECTION END **********************/


			/*****************BRANCH SECTION BEGIN **********************/
			/*Construct Branch Manager Initialization*/
			if (optimizer_settings_.enable_branch_ && optimizer_settings_.number_branches > 0 && !error_occurrred_) {
				if (!branch_manager_.InitializeActiveCostFunction(error_message)) {
					emit OptimizerError(QString::fromStdString(error_message));
					error_occurrred_ = true;
				}

				/*Make Sure Dilation Image is Showing Branch Value */
				cv::dilate(frames_A_[frame_index].GetEdgeImage(), frames_A_[frame_index].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), branch_dilation_val_); /*Reset Dilation In That Image*/
				if (calibration_.biplane_calibration) {
					cv::dilate(frames_B_[frame_index].GetEdgeImage(), frames_B_[frame_index].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), branch_dilation_val_); /*Reset Dilation In That Image*/
				}
				emit UpdateDilationBackground();
			}


			/*Move to Branch If Necessary*/
			for (int branch_index = 0; branch_index < optimizer_settings_.enable_branch_ * optimizer_settings_.number_branches; branch_index++) {
				/*If Error*/
				if (error_occurrred_) break;

				/*Update Search Stage Flag as Branch*/
				search_stage_flag_ = SearchStageFlag::Branch;

				/*Reset Storage, Starting Point, Range, new budget, comparison image*/
				/*Reset Starting Point*/
				SetStartingPoint(current_optimum_location_);
				/*Reset Range*/
				SetSearchRange(optimizer_settings_.branch_range);
				/*Reset Budget and Cost Function Calls*/
				budget_ += optimizer_settings_.branch_budget;
				/*Reset Storage*/
				data_.DeleteAllStoredHyperboxes();
				/*Initialize with Unit Sized HyperBox at Center*/
				current_optimum_value_ = EvaluateCostFunction(Point6D(.5, .5, .5, .5, .5, .5));
				current_optimum_location_ = starting_point_;
				data_ = DirectDataStorage(current_optimum_value_);

				/*Main Loop*/
				while (cost_function_calls_ < budget_) {
					/*Scroll Through Convex Hull to Get List of Potentially
					Optimal Hyperboxes to Evaluate. This is stored as list of
					column IDs in DATA that need to have least Fvalued
					hyperbox (last in list) returned for trisection and evaluation.*/
					ConvexHull();

					/*Trisect Potentially Optimal Rectangles*/
					TrisectPotentiallyOptimal();

					/*Safety Break...Should Never Happen*/
					if (potentially_optimal_col_ids_.size() == 0) {
						emit OptimizerError("Error, no potentialy optimal hyper rectangles found!");
						error_occurrred_ = true;
						break;
					}

					/*If Error*/
					if (error_occurrred_) break;

					/*Update Screen at Rate of 30 FPS*/
					if ((clock() - update_screen_clock_) > 33) {
						emit UpdateDisplay((double)(clock() - start_clock_) / (double)cost_function_calls_, (int)cost_function_calls_, current_optimum_value_, primary_model_index_);
						update_screen_clock_ = clock();
					}
				}
			}
		}
		
		/*****************BRANCH SECTION END **********************/

		/*****************LEAF SECTION BEGIN **********************/
		/*Construct Leaf Initialization*/
		
		if (optimizer_settings_.enable_leaf_ && !error_occurrred_) {
			if (!leaf_manager_.InitializeActiveCostFunction(error_message)) {
				emit OptimizerError(QString::fromStdString(error_message));
				error_occurrred_ = true;
			}
			/*Make Sure Dilation Image is Showing Leaf Value */
			cv::dilate(frames_A_[frame_index].GetEdgeImage(), frames_A_[frame_index].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), leaf_dilation_val_); /*Reset Dilation In That Image*/
			if (calibration_.biplane_calibration) {
				cv::dilate(frames_B_[frame_index].GetEdgeImage(), frames_B_[frame_index].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), leaf_dilation_val_); /*Reset Dilation In That Image*/
			}
			emit UpdateDilationBackground();
		}

		if (sym_trap_call) { 
			CalculateSymTrap();
		}

		/*Move to Leaf Search If Necessary*/
		if (optimizer_settings_.enable_leaf_ && !error_occurrred_ && !sym_trap_call) {

			/*Update Search Stage Flag as Leaf*/
			search_stage_flag_ = SearchStageFlag::Leaf;

			/*Reset Storage, Starting Point, Range, new budget, comparison image*/
			/*Reset Starting Point*/
			SetStartingPoint(current_optimum_location_);
			/*Reset Range*/
			SetSearchRange(optimizer_settings_.leaf_range);
			/*Reset Budget and Cost Function Calls*/
			budget_ += optimizer_settings_.leaf_budget;
			/*Reset Storage*/
			data_.DeleteAllStoredHyperboxes();
			/*Initialize with Unit Sized HyperBox at Center*/
			current_optimum_value_ = EvaluateCostFunction(Point6D(.5, .5, .5, .5, .5, .5));
			current_optimum_location_ = starting_point_;
			data_ = DirectDataStorage(current_optimum_value_);

			/*Main Loop*/
			while (cost_function_calls_ < budget_) {
				/*Scroll Through Convex Hull to Get List of Potentially
				Optimal Hyperboxes to Evaluate. This is stored as list of
				column IDs in DATA that need to have least Fvalued
				hyperbox (last in list) returned for trisection and evaluation.*/
				ConvexHull();

				/*Trisect Potentially Optimal Rectangles*/
				TrisectPotentiallyOptimal();

				/*Safety Break...Should Never Happen*/
				if (potentially_optimal_col_ids_.size() == 0) {
					emit OptimizerError("Error, no potentialy optimal hyper rectangles found!");
					error_occurrred_ = true;
					break;
				}

				/*If Error*/
				if (error_occurrred_) break;

				/*Update Screen at Rate of 30 FPS*/
				if ((clock() - update_screen_clock_) > 33) {
					emit UpdateDisplay((double)(clock() - start_clock_) / (double)cost_function_calls_, (int)cost_function_calls_, current_optimum_value_, primary_model_index_);
					update_screen_clock_ = clock();
				}
			}
		}

		/*Destruct Leaf Initialization CFM*/
		if (optimizer_settings_.enable_leaf_ && !error_occurrred_) {
			if (!leaf_manager_.DestructActiveCostFunction(error_message)) {
				emit OptimizerError(QString::fromStdString(error_message));
				error_occurrred_ = true;
			}
		}

		/*****************LEAF SECTION END **********************/

		/*Clean Up and Return true*/
		data_.DeleteAllStoredHyperboxes();

		/*Update Comparison Image in Dilation Metric and Dilation Metric Dilation Level to Original*/
		cv::dilate(frames_A_[frame_index].GetEdgeImage(), frames_A_[frame_index].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), trunk_dilation_val_); /*Reset Dilation In That Image*/
		if (calibration_.biplane_calibration) {
			cv::dilate(frames_B_[frame_index].GetEdgeImage(), frames_B_[frame_index].GetDilationImage(), cv::Mat(), cv::Point(-1, -1), trunk_dilation_val_); /*Reset Dilation In That Image*/
		}
		emit UpdateDilationBackground();

		/*Move on and Wrap Up*/
		if (error_occurrred_ || frame_index == end_frame_index_) progress_next_frame_ = false;
		emit OptimizedFrame(current_optimum_location_.x, current_optimum_location_.y, current_optimum_location_.z,
			current_optimum_location_.xa, current_optimum_location_.ya, current_optimum_location_.za, progress_next_frame_, primary_model_index_, error_occurrred_, optimization_directive_);
		
		if (sym_trap_call) {
			emit finished();
			return;
		}
		
		emit UpdateDisplay((double)(clock() - start_clock_) / (double)cost_function_calls_, (int)cost_function_calls_, current_optimum_value_, primary_model_index_);
		update_screen_clock_ = clock();

		/*Update Pose Storage*/
		Pose current_opt_pose = Pose(current_optimum_location_.x, current_optimum_location_.y, current_optimum_location_.z,
			current_optimum_location_.xa, current_optimum_location_.ya, current_optimum_location_.za);
		pose_storage_.UpdatePrincipalModelPose(frame_index, current_opt_pose);

		/*If Error Occurred or Not Progressing Breank (Which Ends)*/
		if (!progress_next_frame_) {
			break;
		}
	}
	
	/*Finish And Return*/
	emit finished();
	return;
}

void OptimizerManager::CalculateSymTrap() {
	if (current_optimum_location_.xa == 0 && current_optimum_location_.ya == 0 && current_optimum_location_.za == 0) {
		cout << "ERROR: INVALID STARTING POSE FOR SYMMETRY TRAP" << endl;
		return;
	}
	//Store cost values to input to csv
	std::vector <double> Costs;

	// Get number of iterations from sym_trap spin box
	//int iter_val = sym_trap_obj->getIterCount() * 3;
	int iter_val = 60;// iter_count * 3;
	std::cout << "Sym Trap Iteration size: " << iter_val << std::endl;

	// Get pose list from sym trap
	std::vector<Point6D> pose_list(0);
	Point6D pose_6D(current_optimum_location_);
	create_vector_of_poses(pose_list, pose_6D, 20);

	int progress_val = 0;
	// Calculate cost function at each pose
	for (int i = 0; i < iter_val; i++) {
		emit onUpdateOrientationSymTrap(pose_list.at(i).x, pose_list.at(i).y, pose_list.at(i).z, pose_list.at(i).xa, pose_list.at(i).ya, pose_list.at(i).za);
		std::this_thread::sleep_for(std::chrono::milliseconds(5000/iter_val));
		double myCost = EvaluateCostFunctionAtPoint(pose_list.at(i), 2); // Use leaf
		Costs.push_back(myCost);
		std::cout << i + 1 << ": " << myCost << " @ rotation (" << pose_list.at(i).xa << " " << pose_list.at(i).ya << " " << pose_list.at(i).za << ")" << std::endl;

		// Update progress bar according to number of iterations
		progress_val = (i + 1) * 100 / iter_val;		
		emit onProgressBarUpdate(progress_val);
	}

	//set model back to intial pose
	emit onUpdateOrientationSymTrap(pose_6D.x, pose_6D.y, pose_6D.z, pose_6D.xa, pose_6D.ya, pose_6D.za);

	//Csv of position and cost value (xangle,yangle,zangle,cost value \n)
	std::ofstream myfile;
	myfile.open("Results.csv");
	for (int i = 0; i < iter_val; i++) {
		myfile << pose_list.at(i).xa << "," << pose_list.at(i).ya << "," << pose_list.at(i).za << "," << Costs.at(i) << "\n";

	}
	myfile.close();

	// Used for Sym Trap VTK plot
	std::ofstream myfile2;
	myfile2.open("Results.xyz");
	for (int i = 0; i < iter_val; i++) {
		myfile2 << pose_list.at(i).xa << " " << pose_list.at(i).ya << " " << Costs.at(i) << "\n";

	}
	myfile2.close();

	std::ofstream myfile3;
	myfile3.open("Results2D.xy");
	for (int i = 0; i < iter_val; i++) {
		myfile3 << i - iter_val / 3 << " " << Costs.at(i) << "\n";

	}
	myfile3.close();

	emit onProgressBarUpdate(100);
}

double OptimizerManager::EvaluateCostFunctionAtPoint(Point6D point, int stage) {
	enum Dilation { Trunk, Branch, Leaf };

	/*Send normal pose not denormalized pose*/
	//Point6D denormalized_point = DenormalizeFromCenter(point);
	gpu_cost_function::Pose pose(point.x, point.y, point.z,
		point.xa, point.ya, point.za);
	gpu_principal_model_->SetCurrentPrimaryCameraPose(pose);

	double result = 0;
	switch (stage) {
	case Trunk: result = trunk_manager_.callActiveCostFunction();
		break;
	case Branch: result = branch_manager_.callActiveCostFunction();
		break;
	case Leaf: result = leaf_manager_.callActiveCostFunction();
		break;
	}
	// cost_function_calls_++;
	emit CostFuncAtPoint(result);
	

	return result;
}

double OptimizerManager::EvaluateCostFunction(Point6D point) {
	/*Get Actual Pose from Normalized Version and Send to Cost Function Manager*/
	Point6D denormalized_point = DenormalizeFromCenter(point);
	gpu_cost_function::Pose pose(denormalized_point.x, denormalized_point.y, denormalized_point.z,
		denormalized_point.xa, denormalized_point.ya, denormalized_point.za);
	gpu_principal_model_->SetCurrentPrimaryCameraPose(pose);
	if (calibration_.biplane_calibration) {
		Point6D denormalized_point_B = calibration_.convert_Pose_A_to_Pose_B(denormalized_point);
		gpu_cost_function::Pose pose_B(denormalized_point_B.x, denormalized_point_B.y, denormalized_point_B.z,
			denormalized_point_B.xa, denormalized_point_B.ya, denormalized_point_B.za);
		gpu_principal_model_->SetCurrentSecondaryCameraPose(pose_B);
	}

	/*Compute Cost Function Value*/
	double result = 0;
	switch (search_stage_flag_) {
	case SearchStageFlag::Trunk: result = trunk_manager_.callActiveCostFunction();
		break;
	case SearchStageFlag::Branch: result = branch_manager_.callActiveCostFunction();
		break;
	case SearchStageFlag::Leaf: result = leaf_manager_.callActiveCostFunction();
		break;
	}
	cost_function_calls_++;

	/*Store Optimum*/
	if (result < current_optimum_value_) { //<= should be better like this
		current_optimum_value_ = result;
		current_optimum_location_ = denormalized_point;
		emit UpdateOptimum(current_optimum_location_.x, current_optimum_location_.y, current_optimum_location_.z,
			current_optimum_location_.xa, current_optimum_location_.ya, current_optimum_location_.za, primary_model_index_);
	}

	return result;
}

void OptimizerManager::ConvexHull() {
	/*Reset Potentially Optimal Vector*/
	potentially_optimal_col_ids_.clear();

	/*Jarvis's Match (Gift Wrapping)*/
	/*If only one column add index 0*/
	if (data_.GetNumberColumns() == 1) {
		potentially_optimal_col_ids_.push_back(0);
	}
	else if (data_.GetNumberColumns() > 1) {
		/*Perform Gift Wrapping Algorithm. However, add a small epsilon to the smallest sized
		column. The size of this epsilon is to be determined by experiment. Different from
		previous JTA.*/

		/*Initialize Indexing/Intermediate Variables*/
		int right_index = data_.GetNumberColumns() - 1;
		int left_index = right_index - 1;
		unsigned potentially_optimal_index;
		double slope, highest_slope;
		double right_value, right_size;

		/*Convex Hull from Fatherst Right (Largest Point) so always add that first*/
		potentially_optimal_col_ids_.push_back(right_index);

		/*Gift Wrapping O(data_.GetNumberColumns()*# P.O. Hyperboxes)*/
		while (right_index > 0) {
			highest_slope = -1 * DBL_MAX;
			right_value = data_.GetMinimumHyperboxValue(right_index);
			right_size = data_.GetSizeStoredInColumn(right_index);
			potentially_optimal_index = left_index;
			while (left_index >= 0) {
				slope = (right_value - data_.GetMinimumHyperboxValue(left_index)) /
					(right_size - data_.GetSizeStoredInColumn(left_index));
				if (slope >= highest_slope) {
					highest_slope = slope;
					potentially_optimal_index = left_index;
				}
				left_index--;
			}
			/*Make Sure Convex Hull Never Goes Back Up After Flattening out*/
			if (highest_slope >= 0) {
				potentially_optimal_col_ids_.push_back(potentially_optimal_index);
				right_index = potentially_optimal_index;
				left_index = right_index - 1;
			}
			else break;
		}
	}
	else {
		/*If This Happens the Storage Matrix is Empty...This should NEVER Happen*/
		error_occurrred_ = true;
		emit OptimizerError("ERROR: Storage Matrix Empty!");
	}
}

void OptimizerManager::TrisectPotentiallyOptimal() {
	/*Populate Potentially Optimal Hyperboxes from Column IDs*/
	potentially_optimal_hyperboxes_.clear();
	for (int i = 0; i < potentially_optimal_col_ids_.size(); i++) {
		potentially_optimal_hyperboxes_.push_back(data_.GetMinimumHyperbox(potentially_optimal_col_ids_[i]));
	}

	/*Delete Old HyperBoxes*/
	data_.DeleteHyperBoxes(potentially_optimal_col_ids_);

	/*Trisect Each Potentially Optimal*/
	for (int i = 0; i < potentially_optimal_hyperboxes_.size(); i++) {
		/*Must add three hyperrectangles to data_. This is done by splitting along
		the largest denormalized side and resizing the side lenghts for all three.
		The center/fvalue is only changed in two of the hyperrectangles (obviously). We call
		the unchanged hyperrectangle original_center_hyperbox_ and the ones with
		different centers changed_hyperbox_a and changed_hyperbox_b*/

		/*Chose largest denormalized side to trisect*/
		Point6D denormalized_sides = DenormalizeRange(potentially_optimal_hyperboxes_[i].GetSides());
		Direction largest_direction = denormalized_sides.GetLargestDirection();

		/*Unchanged Center HyperBox*/
		HyperBox6D* original_center_hyperbox_ = new HyperBox6D();
		*original_center_hyperbox_ = potentially_optimal_hyperboxes_[i];
		original_center_hyperbox_->TrisectSide(largest_direction);
		data_.AddHyperBox(original_center_hyperbox_);

		/*Holder for new updated centers*/
		Point6D updated_center;

		/*Changed Center HyperBox A*/
		HyperBox6D* changed_hyperbox_a = new HyperBox6D();
		*changed_hyperbox_a = potentially_optimal_hyperboxes_[i];
		/*Trisect Side*/
		changed_hyperbox_a->TrisectSide(largest_direction);
		/*Update Center*/
		updated_center = changed_hyperbox_a->GetCenter();
		updated_center.UpdateDirection(largest_direction,
			updated_center.GetDirection(largest_direction) +
			changed_hyperbox_a->GetSides().GetDirection(largest_direction));
		changed_hyperbox_a->SetCenter(updated_center);
		/*Update Value*/
		changed_hyperbox_a->value_ = EvaluateCostFunction(changed_hyperbox_a->GetCenter());
		data_.AddHyperBox(changed_hyperbox_a);

		/*Changed Center HyperBox B*/
		HyperBox6D* changed_hyperbox_b = new HyperBox6D();
		*changed_hyperbox_b = potentially_optimal_hyperboxes_[i];
		/*Trisect Side*/
		changed_hyperbox_b->TrisectSide(largest_direction);
		/*Update Center*/
		updated_center = changed_hyperbox_b->GetCenter();
		updated_center.UpdateDirection(largest_direction,
			updated_center.GetDirection(largest_direction) -
			changed_hyperbox_b->GetSides().GetDirection(largest_direction));
		changed_hyperbox_b->SetCenter(updated_center);
		/*Update Value*/
		changed_hyperbox_b->value_ = EvaluateCostFunction(changed_hyperbox_b->GetCenter());
		data_.AddHyperBox(changed_hyperbox_b);
	}

}

Point6D OptimizerManager::DenormalizeRange(Point6D unit_point) {
	return Point6D(unit_point.x * range_.x * 2.0, unit_point.y * range_.y * 2.0, unit_point.z * range_.z * 2.0,
		unit_point.xa * range_.xa * 2.0, unit_point.ya * range_.ya * 2.0, unit_point.za * range_.za * 2.0);
}

Point6D OptimizerManager::DenormalizeFromCenter(Point6D unit_point) {
	return Point6D(starting_point_.x + (unit_point.x - 0.5) * 2 * range_.x,
		starting_point_.y + (unit_point.y - 0.5) * 2 * range_.y,
		starting_point_.z + (unit_point.z - 0.5) * 2 * range_.z,
		starting_point_.xa + (unit_point.xa - 0.5) * 2 * range_.xa,
		starting_point_.ya + (unit_point.ya - 0.5) * 2 * range_.ya,
		starting_point_.za + (unit_point.za - 0.5) * 2 * range_.za);
}

void OptimizerManager::onStopOptimizer() {
	error_occurrred_ = true;
}

void OptimizerManager::create_image_indices(std::vector<int> &img_indices, int start, int end) {
	if (start < end) {
		for (int i = start; i <= end; i++) {
			img_indices.push_back(i);
		}
	}
	else if (start > end) {
		for (int i = start; i >= end; i--) {
			img_indices.push_back(i);
		}
	}
	else if (start == end) {
		int i = start;
		img_indices.push_back(i);
	}
}

/*Destructor*/
OptimizerManager::~OptimizerManager() {
	/*GPU Metrics Class*/
	delete gpu_metrics_;

	/* DESTRUCT CUDA Cost Function Objects (Vector of GPU Models and vector of GPU Frames - note Dilated and Intensity must have own vector
	for each stage because their values could change with the stage from a black silhouette bool or a dilation int)*/
	/*Camera A (Monoplane or Biplane)*/
	for (int i = 0; i < gpu_intensity_frames_trunk_A_.size(); i++)
		delete gpu_intensity_frames_trunk_A_[i];
	for (int i = 0; i < gpu_intensity_frames_branch_A_.size(); i++)
		delete gpu_intensity_frames_branch_A_[i];
	for (int i = 0; i < gpu_intensity_frames_leaf_A_.size(); i++)
		delete gpu_intensity_frames_leaf_A_[i];
	for (int i = 0; i < gpu_edge_frames_A_.size(); i++)
		delete gpu_edge_frames_A_[i];
	for (int i = 0; i < gpu_dilated_frames_trunk_A_.size(); i++)
		delete gpu_dilated_frames_trunk_A_[i];
	for (int i = 0; i < gpu_dilated_frames_branch_A_.size(); i++)
		delete gpu_dilated_frames_branch_A_[i];
	for (int i = 0; i < gpu_dilated_frames_leaf_A_.size(); i++)
		delete gpu_dilated_frames_leaf_A_[i];
	/*Camera B (Biplane only)*/
	for (int i = 0; i < gpu_intensity_frames_trunk_B_.size(); i++)
		delete gpu_intensity_frames_trunk_B_[i];
	for (int i = 0; i < gpu_intensity_frames_branch_B_.size(); i++)
		delete gpu_intensity_frames_branch_B_[i];
	for (int i = 0; i < gpu_intensity_frames_leaf_B_.size(); i++)
		delete gpu_intensity_frames_leaf_B_[i];
	for (int i = 0; i < gpu_edge_frames_B_.size(); i++)
		delete gpu_edge_frames_B_[i];
	for (int i = 0; i < gpu_dilated_frames_trunk_B_.size(); i++)
		delete gpu_dilated_frames_trunk_B_[i];
	for (int i = 0; i < gpu_dilated_frames_branch_B_.size(); i++)
		delete gpu_dilated_frames_branch_B_[i];
	for (int i = 0; i < gpu_dilated_frames_leaf_B_.size(); i++)
		delete gpu_dilated_frames_leaf_B_[i];

	/*Models*/
	delete gpu_principal_model_;
	for (int i = 0; i < gpu_non_principal_models_.size(); i++)
		delete gpu_non_principal_models_[i];
};