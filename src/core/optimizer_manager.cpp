// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*Optimizer Manaer*/
#include "core/optimizer_manager.h"

/*Pose Matrix Class*/
#include <stdlib.h>

#include <chrono>
#include <thread>

#include "gpu_heatmaps.cuh"
#include "gpu_model.cuh"
#include "pose_matrix.h"

OptimizerManager::OptimizerManager(QObject* parent) : QObject(parent) {
    /*Start Update Timer*/
    optimum_update_timer_.start();
}

/*Initialize*/
bool OptimizerManager::Initialize(
    QThread& optimizer_thread,
    Calibration calibration_file,
    std::vector<Frame> camera_A_frame_list,
    std::vector<Frame> camera_B_frame_list,
    unsigned int current_frame_index,
    std::vector<Model> model_list,
    QModelIndexList selected_models,
    unsigned int primary_model_index,
    LocationStorage pose_matrix,
    OptimizerSettings opt_settings,
    jta_cost_function::CostFunctionManager trunk_manager,
    jta_cost_function::CostFunctionManager branch_manager,
    jta_cost_function::CostFunctionManager leaf_manager,
    QString opt_directive,
    QString& error_message,
    int iter_count) {
    /*Success?*/
    succesfull_initialization_ = true;

    /*Error Check for Optimizer*/
    error_occurrred_ = false;

    /*Mirror Vectors Reset*/
    raw_gpu_intensity_frames_trunk_A_.clear();
    raw_gpu_intensity_frames_branch_A_.clear();
    raw_gpu_intensity_frames_leaf_A_.clear();
    raw_gpu_edge_frames_A_.clear();
    raw_gpu_dilated_frames_trunk_A_.clear();
    raw_gpu_dilated_frames_branch_A_.clear();
    raw_gpu_dilated_frames_leaf_A_.clear();
    raw_gpu_distance_maps_.clear();
    raw_gpu_heatmaps_.clear();
    raw_gpu_intensity_frames_trunk_B_.clear();
    raw_gpu_intensity_frames_branch_B_.clear();
    raw_gpu_intensity_frames_leaf_B_.clear();
    raw_gpu_edge_frames_B_.clear();
    raw_gpu_dilated_frames_trunk_B_.clear();
    raw_gpu_dilated_frames_branch_B_.clear();
    raw_gpu_dilated_frames_leaf_B_.clear();
    raw_gpu_non_principal_models_.clear();

    /*Set up Thread Connections*/
    /*Connect Start of Thread to Optimisation Loop and Emergency Stop*/
    connect(&optimizer_thread, SIGNAL(started()), this, SLOT(Optimize()));

    /*Destructor Connections*/
    connect(this, SIGNAL(finished()), &optimizer_thread, SLOT(quit()));
    connect(this, SIGNAL(finished()), this, SLOT(deleteLater()));
    connect(
        &optimizer_thread,
        SIGNAL(finished()),
        &optimizer_thread,
        SLOT(deleteLater()));

    /*Store Calibration File Locally*/
    calibration_ = calibration_file;
    optimization_directive_ = opt_directive;

    /*Just In Case Have to Reset*/
    gpu_principal_model_ = nullptr;
    gpu_metrics_ = nullptr;

    /*Store Camera Frame Lists Locally and Check That, if Biplane is Enabled ->
    both lists are the same size. Also Check that the current frame index is
    within the range of the frame list sizes.*/
    frames_A_ = camera_A_frame_list;
    frames_B_ = camera_B_frame_list;
    if (calibration_.biplane_calibration &&
        frames_A_.size() != frames_B_.size()) {
        error_message = "Biplane mode enabled, but each camera has a different "
                        "number of frames!";
        succesfull_initialization_ = false;
        return succesfull_initialization_;
    }
    if (current_frame_index >= frames_A_.size() ||
        (current_frame_index >= frames_B_.size() &&
         calibration_.biplane_calibration)) {
        error_message = "Current frame index is out of scope!";
        succesfull_initialization_ = false;
        return succesfull_initialization_;
    }

    /*Store Model List, Non-Primary Selected Models (Blue) and Primary Model
    Also store indices of all models and the index of the primary model*/
    all_models_ = model_list;
    if (selected_models.size() == 0 ||
        selected_models[0].row() != primary_model_index) {
        error_message = "Can't find primary model!";
        succesfull_initialization_ = false;
        return succesfull_initialization_;
    }
    for (int i = 1; i < selected_models.size(); i++) {
        selected_non_primary_models_.push_back(
            all_models_[selected_models[i].row()]);
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
            auto temp_pose = Pose(
                temp_p6d.x,
                temp_p6d.y,
                temp_p6d.z,
                temp_p6d.xa,
                temp_p6d.ya,
                temp_p6d.za);
            poses_each_frame_for_given_model.push_back(temp_pose);
        }
        /*If i ==0, principal model*/
        if (i == 0) {
            pose_storage_.AddModel(
                poses_each_frame_for_given_model,
                all_models_[index_for_model].model_name_,
                true);
        } else {
            pose_storage_.AddModel(
                poses_each_frame_for_given_model,
                all_models_[index_for_model].model_name_,
                false);
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

    } else if (opt_directive == "All") {
        /*Should we progess to next frame?*/
        progress_next_frame_ = true;
        /*Should we initialize with previous frame's best guess?*/
        init_prev_frame_ = true;
        /*Index For Starting Frame in Optimization*/
        start_frame_index_ = 0;
        end_frame_index_ = frames_A_.size() - 1;
    } else if (opt_directive == "Each") {
        /*Should we progess to next frame?*/
        progress_next_frame_ = true;
        /*Should we initialize with previous frame's best guess?*/
        init_prev_frame_ = false;
        /*Index For Starting Frame in Optimization*/
        start_frame_index_ = 0;
        end_frame_index_ = frames_A_.size() - 1;
    } else if (opt_directive == "From") {
        /*Should we progess to next frame?*/
        progress_next_frame_ = true;
        /*Should we initialize with previous frame's best guess?*/
        init_prev_frame_ = true;
        /*Index For Starting Frame in Optimization*/
        start_frame_index_ = current_frame_index;
        end_frame_index_ = frames_A_.size() - 1;
    } else if (opt_directive == "Sym_Trap") {
        /*Should we progess to next frame?*/
        progress_next_frame_ = false;
        /*Should we initialize with previous frame's best guess?*/
        init_prev_frame_ = false;
        /*Index For Starting Frame in Optimization*/
        start_frame_index_ = current_frame_index;

        sym_trap_call = true;
    } else if (opt_directive == "Backward") {
        progress_next_frame_ = true;
        init_prev_frame_ = true;
        start_frame_index_ = current_frame_index;
        end_frame_index_ = 0;
    } else {
        error_message = "Unrecognized optimization directive: " + opt_directive;
        succesfull_initialization_ = false;
        return succesfull_initialization_;
    }
    /*Setting Up image indices based on the directive launched*/
    create_image_indices(img_indices_, start_frame_index_, end_frame_index_);
    /*Set Up Settings*/
    SetSearchRange(optimizer_settings_.trunk_range);
    SetStartingPoint(
        pose_matrix.GetPose(start_frame_index_, primary_model_index_));
    budget_ = optimizer_settings_.trunk_budget;
    cost_function_calls_ = 0;
    current_optimum_value_ = DBL_MAX;
    current_optimum_location_ = starting_point_;

    /*Initialize GPU CUDA Cost Function Library Tools*/
    /*Get Width and Height (Safe since did error check before launching this
     * function)*/
    int width = frames_A_[0].GetEdgeImage().cols;
    int height = frames_A_[0].GetEdgeImage().rows;

    /*Check CUDA Compatibility*/
    int cuda_device_id = 0, gpu_device_count = 0, device_count;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&device_count);
    if (cudaResultCode != cudaSuccess) device_count = 0;
    /* Machines with no GPUs can still report one emulation device */
    for (int device = 0; device < device_count; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999 &&
            properties.major >= 5) /* 9999 means emulation only */
            ++gpu_device_count;
    }
    /*If no Cuda Compatitble Devices with Compute Capability Greater Than 5,
     * Exit*/
    if (gpu_device_count == 0) {
        error_message =
            "No Cuda Compatitble Devices with Compute Capability Greater Than "
            "5!";
        succesfull_initialization_ = false;
        return succesfull_initialization_;
    }

    /*Get Dilation Values for Trunk, Branch, and Leaf*/
    /*Trunk*/
    trunk_dilation_val_ = 0;
    std::vector<jta_cost_function::Parameter<int>> active_int_params =
        trunk_manager_.getActiveCostFunctionClass()->getIntParameters();
    for (int i = 0; i < active_int_params.size(); i++) {
        if (active_int_params[i].getParameterName() == "Dilation" ||
            active_int_params[i].getParameterName() == "DILATION" ||
            active_int_params[i].getParameterName() == "dilation") {
            trunk_dilation_val_ = trunk_manager_.getActiveCostFunctionClass()
                                      ->getIntParameters()
                                      .at(i)
                                      .getParameterValue();
        }
    }
    if (trunk_dilation_val_ <= 0) trunk_dilation_val_ = 0;
    /*Check Special Mahfouz Case*/
    if (trunk_manager_.getActiveCostFunction() == "DIRECT_MAHFOUZ")
        trunk_dilation_val_ = 3;
    /*Branch*/
    branch_dilation_val_ = 0;
    active_int_params =
        branch_manager_.getActiveCostFunctionClass()->getIntParameters();
    for (int i = 0; i < active_int_params.size(); i++) {
        if (active_int_params[i].getParameterName() == "Dilation" ||
            active_int_params[i].getParameterName() == "DILATION" ||
            active_int_params[i].getParameterName() == "dilation") {
            branch_dilation_val_ = branch_manager_.getActiveCostFunctionClass()
                                       ->getIntParameters()
                                       .at(i)
                                       .getParameterValue();
        }
    }
    if (branch_dilation_val_ <= 0) branch_dilation_val_ = 0;
    /*Check Special Mahfouz Case*/
    if (branch_manager_.getActiveCostFunction() == "DIRECT_MAHFOUZ")
        branch_dilation_val_ = 3;
    /*Leaf*/
    leaf_dilation_val_ = 0;
    active_int_params =
        leaf_manager_.getActiveCostFunctionClass()->getIntParameters();
    for (int i = 0; i < active_int_params.size(); i++) {
        if (active_int_params[i].getParameterName() == "Dilation" ||
            active_int_params[i].getParameterName() == "DILATION" ||
            active_int_params[i].getParameterName() == "dilation") {
            leaf_dilation_val_ = leaf_manager_.getActiveCostFunctionClass()
                                     ->getIntParameters()
                                     .at(i)
                                     .getParameterValue();
        }
    }
    if (leaf_dilation_val_ <= 0) leaf_dilation_val_ = 0;
    /*Check Special Mahfouz Case*/
    if (leaf_manager_.getActiveCostFunction() == "DIRECT_MAHFOUZ")
        leaf_dilation_val_ = 3;

    /*Get Black Silhouette? Values for Trunk, Branch, and Leaf*/
    /*Black Silhouette Values Based on Bool Parameter Names (Black_Silhouette or
     * Dark_Silhouette or BLACK_SILHOUETTE or DARK_SILHOUETTE or
     * black_silhouette or dark_silhouette)*/
    /*Trunk*/
    trunk_dark_silhouette_val_ = false;
    std::vector<jta_cost_function::Parameter<bool>> active_bool_params =
        trunk_manager_.getActiveCostFunctionClass()->getBoolParameters();
    for (int i = 0; i < active_bool_params.size(); i++) {
        if (active_bool_params[i].getParameterName() == "Black_Silhouette" ||
            active_bool_params[i].getParameterName() == "Dark_Silhouette" ||
            active_bool_params[i].getParameterName() == "BLACK_SILHOUETTE" ||
            active_bool_params[i].getParameterName() == "DARK_SILHOUETTE" ||
            active_bool_params[i].getParameterName() == "black_silhouette" ||
            active_bool_params[i].getParameterName() == "dark_silhouette") {
            trunk_dark_silhouette_val_ =
                trunk_manager_.getActiveCostFunctionClass()
                    ->getBoolParameters()
                    .at(i)
                    .getParameterValue();
        }
    }
    /*Branch*/
    branch_dark_silhouette_val_ = false;
    active_bool_params =
        branch_manager_.getActiveCostFunctionClass()->getBoolParameters();
    for (int i = 0; i < active_bool_params.size(); i++) {
        if (active_bool_params[i].getParameterName() == "Black_Silhouette" ||
            active_bool_params[i].getParameterName() == "Dark_Silhouette" ||
            active_bool_params[i].getParameterName() == "BLACK_SILHOUETTE" ||
            active_bool_params[i].getParameterName() == "DARK_SILHOUETTE" ||
            active_bool_params[i].getParameterName() == "black_silhouette" ||
            active_bool_params[i].getParameterName() == "dark_silhouette") {
            branch_dark_silhouette_val_ =
                branch_manager_.getActiveCostFunctionClass()
                    ->getBoolParameters()
                    .at(i)
                    .getParameterValue();
        }
    }
    /*Leaf*/
    leaf_dark_silhouette_val_ = false;
    active_bool_params =
        leaf_manager_.getActiveCostFunctionClass()->getBoolParameters();
    for (int i = 0; i < active_bool_params.size(); i++) {
        if (active_bool_params[i].getParameterName() == "Black_Silhouette" ||
            active_bool_params[i].getParameterName() == "Dark_Silhouette" ||
            active_bool_params[i].getParameterName() == "BLACK_SILHOUETTE" ||
            active_bool_params[i].getParameterName() == "DARK_SILHOUETTE" ||
            active_bool_params[i].getParameterName() == "black_silhouette" ||
            active_bool_params[i].getParameterName() == "dark_silhouette") {
            leaf_dark_silhouette_val_ =
                leaf_manager_.getActiveCostFunctionClass()
                    ->getBoolParameters()
                    .at(i)
                    .getParameterValue();
        }
    }

    /*Upload GPU Frames*/
    /*Intensity Frames
    /*Trunk*/
    /*Camera A*/
    for (int i = 0; i < frames_A_.size(); i++) {
        auto intensity_frame = new GPUIntensityFrame(
            width,
            height,
            cuda_device_id,
            frames_A_[i].GetOriginalImage().data,
            trunk_dark_silhouette_val_,
            frames_A_[i].GetInvertedImage().data);
        if (intensity_frame->IsInitializedCorrectly()) {
            raw_gpu_intensity_frames_trunk_A_.push_back(intensity_frame);
            gpu_intensity_frames_trunk_A_.push_back(
                std::unique_ptr<GPUIntensityFrame>(intensity_frame));
        } else {
            delete intensity_frame;
            error_message = "Error uploading Intensity frame to GPU!";
            succesfull_initialization_ = false;
            return succesfull_initialization_;
        }

    }
    /*Camera B*/
    if (calibration_.biplane_calibration) {
        for (int i = 0; i < frames_B_.size(); i++) {
            auto intensity_frame = new GPUIntensityFrame(
                width,
                height,
                cuda_device_id,
                frames_B_[i].GetOriginalImage().data,
                trunk_dark_silhouette_val_,
                frames_B_[i].GetInvertedImage().data);
            if (intensity_frame->IsInitializedCorrectly()) {
                raw_gpu_intensity_frames_trunk_B_.push_back(intensity_frame);
                gpu_intensity_frames_trunk_B_.push_back(
                    std::unique_ptr<GPUIntensityFrame>(intensity_frame));
            } else {
                delete intensity_frame;
                error_message = "Error uploading Intensity frame to GPU!";
                succesfull_initialization_ = false;
                return succesfull_initialization_;
            }

        }
    }
    /*Branch*/
    /*Camera A*/
    for (int i = 0; i < frames_A_.size(); i++) {
        auto intensity_frame = new GPUIntensityFrame(
            width,
            height,
            cuda_device_id,
            frames_A_[i].GetOriginalImage().data,
            branch_dark_silhouette_val_,
            frames_A_[i].GetInvertedImage().data);
        if (intensity_frame->IsInitializedCorrectly()) {
            raw_gpu_intensity_frames_branch_A_.push_back(intensity_frame);
            gpu_intensity_frames_branch_A_.push_back(
                std::unique_ptr<GPUIntensityFrame>(intensity_frame));
        } else {
            delete intensity_frame;
            error_message = "Error uploading Intensity frame to GPU!";
            succesfull_initialization_ = false;
            return succesfull_initialization_;
        }

    }
    /*Camera B*/
    if (calibration_.biplane_calibration) {
        for (int i = 0; i < frames_B_.size(); i++) {
            auto intensity_frame = new GPUIntensityFrame(
                width,
                height,
                cuda_device_id,
                frames_B_[i].GetOriginalImage().data,
                branch_dark_silhouette_val_,
                frames_B_[i].GetInvertedImage().data);
            if (intensity_frame->IsInitializedCorrectly()) {
                raw_gpu_intensity_frames_branch_B_.push_back(intensity_frame);
                gpu_intensity_frames_branch_B_.push_back(
                    std::unique_ptr<GPUIntensityFrame>(intensity_frame));
            } else {
                delete intensity_frame;
                error_message = "Error uploading Intensity frame to GPU!";
                succesfull_initialization_ = false;
                return succesfull_initialization_;
            }

        }
    }
    /*Leaf*/
    /*Camera A*/
    for (int i = 0; i < frames_A_.size(); i++) {
        auto intensity_frame = new GPUIntensityFrame(
            width,
            height,
            cuda_device_id,
            frames_A_[i].GetOriginalImage().data,
            leaf_dark_silhouette_val_,
            frames_A_[i].GetInvertedImage().data);
        if (intensity_frame->IsInitializedCorrectly()) {
            raw_gpu_intensity_frames_leaf_A_.push_back(intensity_frame);
            gpu_intensity_frames_leaf_A_.push_back(
                std::unique_ptr<GPUIntensityFrame>(intensity_frame));
        } else {
            delete intensity_frame;
            error_message = "Error uploading Intensity frame to GPU!";
            succesfull_initialization_ = false;
            return succesfull_initialization_;
        }

    }
    /*Camera B*/
    if (calibration_.biplane_calibration) {
        for (int i = 0; i < frames_B_.size(); i++) {
            auto intensity_frame = new GPUIntensityFrame(
                width,
                height,
                cuda_device_id,
                frames_B_[i].GetOriginalImage().data,
                leaf_dark_silhouette_val_,
                frames_B_[i].GetInvertedImage().data);
            if (intensity_frame->IsInitializedCorrectly()) {
                raw_gpu_intensity_frames_leaf_B_.push_back(intensity_frame);
                gpu_intensity_frames_leaf_B_.push_back(
                    std::unique_ptr<GPUIntensityFrame>(intensity_frame));
            } else {
                delete intensity_frame;
                error_message = "Error uploading Intensity frame to GPU!";
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
        dilate(
            frames_A_[i].GetEdgeImage(),
            frames_A_[i].GetDilationImage(),
            cv::Mat(),
            cv::Point(-1, -1),
            leaf_dilation_val_); /*Reset Dilation In That Image*/
    }
    for (int i = 0; i < frames_A_.size(); i++) {
        auto dilated_frame = new GPUDilatedFrame(
            width,
            height,
            cuda_device_id,
            frames_A_[i].GetDilationImage().data,
            leaf_dilation_val_);
        if (dilated_frame->IsInitializedCorrectly()) {
            raw_gpu_dilated_frames_leaf_A_.push_back(dilated_frame);
            gpu_dilated_frames_leaf_A_.push_back(
                std::unique_ptr<GPUDilatedFrame>(dilated_frame));
        } else {
            delete dilated_frame;
            error_message = "Error uploading dilated frame to GPU!";
            succesfull_initialization_ = false;
            return succesfull_initialization_;
        }
    }
    /*Camera B*/
    if (calibration_.biplane_calibration) {
        for (int i = 0; i < frames_B_.size(); i++) {
            dilate(
                frames_B_[i].GetEdgeImage(),
                frames_B_[i].GetDilationImage(),
                cv::Mat(),
                cv::Point(-1, -1),
                leaf_dilation_val_); /*Reset Dilation In That Image*/
        }
    }
    if (calibration_.biplane_calibration) {
        for (int i = 0; i < frames_B_.size(); i++) {
            auto dilated_frame = new GPUDilatedFrame(
                width,
                height,
                cuda_device_id,
                frames_B_[i].GetDilationImage().data,
                leaf_dilation_val_);
            if (dilated_frame->IsInitializedCorrectly()) {
                raw_gpu_dilated_frames_leaf_B_.push_back(dilated_frame);
                gpu_dilated_frames_leaf_B_.push_back(
                    std::unique_ptr<GPUDilatedFrame>(dilated_frame));
            } else {
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
        dilate(
            frames_A_[i].GetEdgeImage(),
            frames_A_[i].GetDilationImage(),
            cv::Mat(),
            cv::Point(-1, -1),
            branch_dilation_val_); /*Reset Dilation In That Image*/
    }
    for (int i = 0; i < frames_A_.size(); i++) {
        auto dilated_frame = new GPUDilatedFrame(
            width,
            height,
            cuda_device_id,
            frames_A_[i].GetDilationImage().data,
            branch_dilation_val_);
        if (dilated_frame->IsInitializedCorrectly()) {
            raw_gpu_dilated_frames_branch_A_.push_back(dilated_frame);
            gpu_dilated_frames_branch_A_.push_back(
                std::unique_ptr<GPUDilatedFrame>(dilated_frame));
        } else {
            delete dilated_frame;
            error_message = "Error uploading dilated frame to GPU!";
            succesfull_initialization_ = false;
            return succesfull_initialization_;
        }
    }
    /*Camera B*/
    if (calibration_.biplane_calibration) {
        for (int i = 0; i < frames_B_.size(); i++) {
            dilate(
                frames_B_[i].GetEdgeImage(),
                frames_B_[i].GetDilationImage(),
                cv::Mat(),
                cv::Point(-1, -1),
                branch_dilation_val_); /*Reset Dilation In That Image*/
        }
    }
    if (calibration_.biplane_calibration) {
        for (int i = 0; i < frames_B_.size(); i++) {
            auto dilated_frame = new GPUDilatedFrame(
                width,
                height,
                cuda_device_id,
                frames_B_[i].GetDilationImage().data,
                branch_dilation_val_);
            if (dilated_frame->IsInitializedCorrectly()) {
                raw_gpu_dilated_frames_branch_B_.push_back(dilated_frame);
                gpu_dilated_frames_branch_B_.push_back(
                    std::unique_ptr<GPUDilatedFrame>(dilated_frame));
            } else {
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
        dilate(
            frames_A_[i].GetEdgeImage(),
            frames_A_[i].GetDilationImage(),
            cv::Mat(),
            cv::Point(-1, -1),
            trunk_dilation_val_); /*Reset Dilation In That Image*/
    }
    for (int i = 0; i < frames_A_.size(); i++) {
        auto dilated_frame = new GPUDilatedFrame(
            width,
            height,
            cuda_device_id,
            frames_A_[i].GetDilationImage().data,
            trunk_dilation_val_);
        if (dilated_frame->IsInitializedCorrectly()) {
            raw_gpu_dilated_frames_trunk_A_.push_back(dilated_frame);
            gpu_dilated_frames_trunk_A_.push_back(
                std::unique_ptr<GPUDilatedFrame>(dilated_frame));
        } else {
            delete dilated_frame;
            error_message = "Error uploading dilated frame to GPU!";
            succesfull_initialization_ = false;
            return succesfull_initialization_;
        }
    }
    /*Camera B*/
    if (calibration_.biplane_calibration) {
        for (int i = 0; i < frames_B_.size(); i++) {
            dilate(
                frames_B_[i].GetEdgeImage(),
                frames_B_[i].GetDilationImage(),
                cv::Mat(),
                cv::Point(-1, -1),
                trunk_dilation_val_); /*Reset Dilation In That Image*/
        }
    }
    if (calibration_.biplane_calibration) {
        for (int i = 0; i < frames_B_.size(); i++) {
            auto dilated_frame = new GPUDilatedFrame(
                width,
                height,
                cuda_device_id,
                frames_B_[i].GetDilationImage().data,
                trunk_dilation_val_);
            if (dilated_frame->IsInitializedCorrectly()) {
                raw_gpu_dilated_frames_trunk_B_.push_back(dilated_frame);
                gpu_dilated_frames_trunk_B_.push_back(
                    std::unique_ptr<GPUDilatedFrame>(dilated_frame));
            } else {
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
        auto edge_frame = new GPUEdgeFrame(
            width,
            height,
            cuda_device_id,
            frames_A_[i].GetEdgeImage().data,
            frames_A_[i].GetHighThreshold(),
            frames_A_[i].GetLowThreshold(),
            frames_A_[i].GetAperture());
        if (edge_frame->IsInitializedCorrectly()) {
            raw_gpu_edge_frames_A_.push_back(edge_frame);
            gpu_edge_frames_A_.push_back(std::unique_ptr<GPUEdgeFrame>(edge_frame));
        } else {
            delete edge_frame;
            error_message = "Error uploading Edge frame to GPU!";
            succesfull_initialization_ = false;
            return succesfull_initialization_;
        }

    }
    /*Camera B*/
    if (calibration_.biplane_calibration) {
        for (int i = 0; i < frames_B_.size(); i++) {
            auto edge_frame = new GPUEdgeFrame(
                width,
                height,
                cuda_device_id,
                frames_B_[i].GetEdgeImage().data,
                frames_B_[i].GetHighThreshold(),
                frames_B_[i].GetLowThreshold(),
                frames_B_[i].GetAperture());
            if (edge_frame->IsInitializedCorrectly()) {
                raw_gpu_edge_frames_B_.push_back(edge_frame);
                gpu_edge_frames_B_.push_back(std::unique_ptr<GPUEdgeFrame>(edge_frame));
            } else {
                delete edge_frame;
                error_message = "Error uploading Edge frame to GPU!";
                succesfull_initialization_ = false;
                return succesfull_initialization_;
            }

        }
    }

    /*Upload distance maps*/
    for (int i = 0; i < frames_A_.size(); i++) {
        auto distance_map = new GPUFrame(
            width, height, cuda_device_id, frames_A_[i].GetDistanceMap().data);
        if (distance_map->IsInitializedCorrectly()) {
            raw_gpu_distance_maps_.push_back(distance_map);
            gpu_distance_maps_.push_back(std::unique_ptr<GPUFrame>(distance_map));
        } else {
            delete distance_map;
            error_message = "Error uploading Distance Map to GPU!";
            succesfull_initialization_ = false;
            return succesfull_initialization_;
        }
    }
    for (int i = 0; i < frames_A_.size(); i++) {
        auto heatmap = new GPUHeatmap(
            width,
            height,
            cuda_device_id,
            frames_A_[i].GetNumCurvatureKeypoints(),
            frames_A_[i].getCurvatureHeatmaps().data());
        if (heatmap->IsInitializedCorrectly()) {
            raw_gpu_heatmaps_.push_back(heatmap);
            gpu_heatmaps_.push_back(std::unique_ptr<GPUHeatmap>(heatmap));
        } else {
            delete heatmap;
            error_message = "Error uploading Heatmap to GPU!";
            succesfull_initialization_ = false;
            return succesfull_initialization_;
        }
    }

    /*Upload GPU Models*/
    /*Monoplane Calibration*/
    if (!calibration_.biplane_calibration) {
        /*Principal model*/
        gpu_principal_model_.reset(new GPUModel(
            primary_model_.model_name_,
            true,
            width,
            height,
            cuda_device_id,
            true,
            &primary_model_.triangle_vertices_[0],
            &primary_model_.triangle_normals_[0],
            primary_model_.triangle_vertices_.size() / 9,
            calibration_.camera_A_principal_));

        if (!gpu_principal_model_->IsInitializedCorrectly()) {
            gpu_principal_model_ = nullptr;
            error_message = "Error uploading principal model to GPU!";
            succesfull_initialization_ = false;
            return succesfull_initialization_;
        }
        /*Non-principal models*/
        for (int i = 1; i < selected_model_list_.size(); i++) {
            auto gpu_non_principal_model = new GPUModel(
                all_models_[selected_model_list_[i].row()].model_name_,
                true,
                width,
                height,
                cuda_device_id,
                true,
                &all_models_[selected_model_list_[i].row()]
                     .triangle_vertices_[0],
                &all_models_[selected_model_list_[i].row()]
                     .triangle_normals_[0],
                all_models_[selected_model_list_[i].row()]
                        .triangle_vertices_.size() /
                    9,
                calibration_.camera_A_principal_);
            if (gpu_non_principal_model->IsInitializedCorrectly()) {
                raw_gpu_non_principal_models_.push_back(gpu_non_principal_model);
                gpu_non_principal_models_.push_back(
                    std::unique_ptr<GPUModel>(gpu_non_principal_model));
            } else {
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
        gpu_principal_model_.reset(new GPUModel(
            primary_model_.model_name_,
            true,
            width,
            height,
            cuda_device_id,
            cuda_device_id,
            true,
            true,
            &primary_model_.triangle_vertices_[0],
            &primary_model_.triangle_normals_[0],
            primary_model_.triangle_vertices_.size() / 9,
            calibration_.camera_A_principal_,
            calibration_.camera_B_principal_));
        if (!gpu_principal_model_->IsInitializedCorrectly()) {
            gpu_principal_model_ = nullptr;
            error_message = "Error uploading principal model to GPU!";
            succesfull_initialization_ = false;
            return succesfull_initialization_;
        }
        /*Non-principal models*/
        for (int i = 1; i < selected_model_list_.size(); i++) {
            auto gpu_non_principal_model = new GPUModel(
                all_models_[selected_model_list_[i].row()].model_name_,
                true,
                width,
                height,
                cuda_device_id,
                cuda_device_id,
                true,
                true,
                &all_models_[selected_model_list_[i].row()]
                     .triangle_vertices_[0],
                &all_models_[selected_model_list_[i].row()]
                     .triangle_normals_[0],
                all_models_[selected_model_list_[i].row()]
                        .triangle_vertices_.size() /
                    9,
                calibration_.camera_A_principal_,
                calibration_.camera_B_principal_);
            if (gpu_non_principal_model->IsInitializedCorrectly()) {
                raw_gpu_non_principal_models_.push_back(gpu_non_principal_model);
                gpu_non_principal_models_.push_back(
                    std::unique_ptr<GPUModel>(gpu_non_principal_model));
            } else {
                delete gpu_non_principal_model;
                error_message = "Error uploading non-principal model to GPU!";
                succesfull_initialization_ = false;
                return succesfull_initialization_;
            }
        }
    }

    /*Initialize GPU Metrics*/
    gpu_metrics_ = std::make_unique<GPUMetrics>();
    if (!gpu_metrics_->IsInitializedCorrectly()) {
        error_message = "GPU metrics class not initialized correctly!";
        succesfull_initialization_ = false;
        return succesfull_initialization_;
    }

    /*Upload Data To CostFunction Managers*/
    trunk_manager_.UploadData(
        &raw_gpu_edge_frames_A_,
        &raw_gpu_dilated_frames_trunk_A_,
        &raw_gpu_intensity_frames_trunk_A_,
        &raw_gpu_edge_frames_B_,
        &raw_gpu_dilated_frames_trunk_B_,
        &raw_gpu_intensity_frames_trunk_B_,
        gpu_principal_model_.get(),
        &raw_gpu_non_principal_models_,
        gpu_metrics_.get(),
        &pose_storage_,
        calibration_.biplane_calibration);
    branch_manager_.UploadData(
        &raw_gpu_edge_frames_A_,
        &raw_gpu_dilated_frames_branch_A_,
        &raw_gpu_intensity_frames_branch_A_,
        &raw_gpu_edge_frames_B_,
        &raw_gpu_dilated_frames_branch_B_,
        &raw_gpu_intensity_frames_branch_B_,
        gpu_principal_model_.get(),
        &raw_gpu_non_principal_models_,
        gpu_metrics_.get(),
        &pose_storage_,
        calibration_.biplane_calibration);
    leaf_manager_.UploadData(
        &raw_gpu_edge_frames_A_,
        &raw_gpu_dilated_frames_leaf_A_,
        &raw_gpu_intensity_frames_leaf_A_,
        &raw_gpu_edge_frames_B_,
        &raw_gpu_dilated_frames_leaf_B_,
        &raw_gpu_intensity_frames_leaf_B_,
        gpu_principal_model_.get(),
        &raw_gpu_non_principal_models_,
        gpu_metrics_.get(),
        &pose_storage_,
        calibration_.biplane_calibration);
    trunk_manager_.UploadDistanceMap(&raw_gpu_distance_maps_, &raw_gpu_heatmaps_);
    branch_manager_.UploadDistanceMap(&raw_gpu_distance_maps_, &raw_gpu_heatmaps_);
    leaf_manager_.UploadDistanceMap(&raw_gpu_distance_maps_, &raw_gpu_heatmaps_);

    return succesfull_initialization_;
};

void OptimizerManager::SetSearchRange(Point6D range) {
    /*Check Search Range is Not Zero*/
    if (range.x + range.y + range.z + range.xa + range.ya + range.za > 0) {
        range_ = range;
        valid_range_ = true;
    } else {
        valid_range_ = false;
    }
}

void OptimizerManager::SetStartingPoint(Point6D starting_point) {
    starting_point_ = starting_point;
}

void OptimizerManager::Optimize() {
    /*Check That Succesfull Initialization*/
    if (!succesfull_initialization_) {
        /*Restore Dilation OpenCV Images*/
        for (int i = 0; i < frames_A_.size(); i++) {
            dilate(
                frames_A_[i].GetEdgeImage(),
                frames_A_[i].GetDilationImage(),
                cv::Mat(),
                cv::Point(-1, -1),
                trunk_dilation_val_); /*Reset Dilation In That Image*/
        }
        /*Camera B*/
        if (calibration_.biplane_calibration) {
            for (int i = 0; i < frames_B_.size(); i++) {
                dilate(
                    frames_B_[i].GetEdgeImage(),
                    frames_B_[i].GetDilationImage(),
                    cv::Mat(),
                    cv::Point(-1, -1),
                    trunk_dilation_val_); /*Reset Dilation In That Image*/
            }
        }

        /*Finish And Return Dont Have to Delete the Renderer and Metric as This
         * has Been Done*/
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
                SetStartingPoint(Point6D(
                    starting_pose.x_location_,
                    starting_pose.y_location_,
                    starting_pose.z_location_,
                    starting_pose.x_angle_,
                    starting_pose.y_angle_,
                    starting_pose.z_angle_));
            } else {
                SetStartingPoint(current_optimum_location_);
            }

            /*Set Current Primary (and if biplane, secondary) Poses for Non
             * Principal Models*/
            for (int non_prin_model_ind = 0;
                 non_prin_model_ind < gpu_non_principal_models_.size();
                 non_prin_model_ind++) {
                Pose temp_primary_pose;
                if (pose_storage_.GetModelPose(
                        gpu_non_principal_models_[non_prin_model_ind]
                            ->GetModelName(),
                        frame_index,
                        &temp_primary_pose)) {
                    gpu_non_principal_models_[non_prin_model_ind]
                        ->SetCurrentPrimaryCameraPose(temp_primary_pose);
                } else {
                    emit OptimizerError(QString::fromStdString(
                        "Could not retrieve pose for non-principal model \"" +
                        gpu_non_principal_models_[non_prin_model_ind]
                            ->GetModelName() +
                        "\" at frame " +
                        QString::number(frame_index).toStdString() + "!"));
                    error_occurrred_ = true;
                    break;
                }
                if (calibration_.biplane_calibration) {
                    auto temp_primary_point = Point6D(
                        temp_primary_pose.x_location_,
                        temp_primary_pose.y_location_,
                        temp_primary_pose.z_location_,
                        temp_primary_pose.x_angle_,
                        temp_primary_pose.y_angle_,
                        temp_primary_pose.z_angle_);
                    Point6D temp_secondary_point =
                        calibration_.convert_Pose_A_to_Pose_B(
                            temp_primary_point);
                    gpu_non_principal_models_[non_prin_model_ind]
                        ->SetCurrentSecondaryCameraPose(Pose(
                            temp_secondary_point.x,
                            temp_secondary_point.y,
                            temp_secondary_point.z,
                            temp_secondary_point.xa,
                            temp_secondary_point.ya,
                            temp_secondary_point.za));
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
            search_stage_flag_ = Trunk;

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
                current_optimum_value_ =
                    EvaluateCostFunction(Point6D(.5, .5, .5, .5, .5, .5));
                current_optimum_location_ = starting_point_;
                data_ = DirectDataStorage(current_optimum_value_);
            }

            /*Make Sure Dilation Image is Showing Trunk Value (Should be
             * Unnecessary)*/
            dilate(
                frames_A_[frame_index].GetEdgeImage(),
                frames_A_[frame_index].GetDilationImage(),
                cv::Mat(),
                cv::Point(-1, -1),
                trunk_dilation_val_); /*Reset Dilation In That Image*/
            if (calibration_.biplane_calibration) {
                dilate(
                    frames_B_[frame_index].GetEdgeImage(),
                    frames_B_[frame_index].GetDilationImage(),
                    cv::Mat(),
                    cv::Point(-1, -1),
                    trunk_dilation_val_); /*Reset Dilation In That Image*/
            }
            emit UpdateDilationBackground();

            /*Main Loop*/
            if (!error_occurrred_) {
                while (cost_function_calls_ < budget_) {
                    /*Scroll Through Convex Hull to Get List of Potentially
                    Optimal Hyperboxes to Evaluate. This is stored as list of
                    column IDs in DATA that need to have least Fvalued
                    hyperbox (last in list) returned for trisection and
                    evaluation.*/
                    ConvexHull();

                    /*Trisect Potentially Optimal Rectangles*/
                    TrisectPotentiallyOptimal();

                    /*Safety Break...Should Never Happen*/
                    if (potentially_optimal_col_ids_.size() == 0) {
                        emit OptimizerError(
                            "Error, no potentially optimal hyper rectangles "
                            "found!");
                        error_occurrred_ = true;
                        break;
                    }

                    /*If Error*/
                    if (error_occurrred_) break;

                    /*Update Screen at Rate of 30 FPS*/
                    if ((clock() - update_screen_clock_) > 33) {
                        emit UpdateDisplay(
                            static_cast<double>(clock() - start_clock_) /
                                static_cast<double>(cost_function_calls_),
                            static_cast<int>(cost_function_calls_),
                            current_optimum_value_,
                            primary_model_index_);
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
            if (optimizer_settings_.enable_branch_ &&
                optimizer_settings_.number_branches > 0 && !error_occurrred_) {
                if (!branch_manager_.InitializeActiveCostFunction(
                        error_message)) {
                    emit OptimizerError(QString::fromStdString(error_message));
                    error_occurrred_ = true;
                }

                /*Make Sure Dilation Image is Showing Branch Value */
                dilate(
                    frames_A_[frame_index].GetEdgeImage(),
                    frames_A_[frame_index].GetDilationImage(),
                    cv::Mat(),
                    cv::Point(-1, -1),
                    branch_dilation_val_); /*Reset Dilation In That Image*/
                if (calibration_.biplane_calibration) {
                    dilate(
                        frames_B_[frame_index].GetEdgeImage(),
                        frames_B_[frame_index].GetDilationImage(),
                        cv::Mat(),
                        cv::Point(-1, -1),
                        branch_dilation_val_); /*Reset Dilation In That Image*/
                }
                emit UpdateDilationBackground();
            }

            /*Move to Branch If Necessary*/
            for (int branch_index = 0;
                 branch_index < optimizer_settings_.enable_branch_ *
                                    optimizer_settings_.number_branches;
                 branch_index++) {
                /*If Error*/
                if (error_occurrred_) break;

                /*Update Search Stage Flag as Branch*/
                search_stage_flag_ = Branch;

                /*Reset Storage, Starting Point, Range, new budget, comparison
                 * image*/
                /*Reset Starting Point*/
                SetStartingPoint(current_optimum_location_);
                /*Reset Range*/
                SetSearchRange(optimizer_settings_.branch_range);
                /*Reset Budget and Cost Function Calls*/
                budget_ += optimizer_settings_.branch_budget;
                /*Reset Storage*/
                data_.DeleteAllStoredHyperboxes();
                /*Initialize with Unit Sized HyperBox at Center*/
                current_optimum_value_ =
                    EvaluateCostFunction(Point6D(.5, .5, .5, .5, .5, .5));
                current_optimum_location_ = starting_point_;
                data_ = DirectDataStorage(current_optimum_value_);

                /*Main Loop*/
                while (cost_function_calls_ < budget_) {
                    /*Scroll Through Convex Hull to Get List of Potentially
                    Optimal Hyperboxes to Evaluate. This is stored as list of
                    column IDs in DATA that need to have least Fvalued
                    hyperbox (last in list) returned for trisection and
                    evaluation.*/
                    ConvexHull();

                    /*Trisect Potentially Optimal Rectangles*/
                    TrisectPotentiallyOptimal();

                    /*Safety Break...Should Never Happen*/
                    if (potentially_optimal_col_ids_.size() == 0) {
                        emit OptimizerError(
                            "Error, no potentialy optimal hyper rectangles "
                            "found!");
                        error_occurrred_ = true;
                        break;
                    }

                    /*If Error*/
                    if (error_occurrred_) break;

                    /*Update Screen at Rate of 30 FPS*/
                    if ((clock() - update_screen_clock_) > 33) {
                        emit UpdateDisplay(
                            static_cast<double>(clock() - start_clock_) /
                                static_cast<double>(cost_function_calls_),
                            static_cast<int>(cost_function_calls_),
                            current_optimum_value_,
                            primary_model_index_);
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
            dilate(
                frames_A_[frame_index].GetEdgeImage(),
                frames_A_[frame_index].GetDilationImage(),
                cv::Mat(),
                cv::Point(-1, -1),
                leaf_dilation_val_); /*Reset Dilation In That Image*/
            if (calibration_.biplane_calibration) {
                dilate(
                    frames_B_[frame_index].GetEdgeImage(),
                    frames_B_[frame_index].GetDilationImage(),
                    cv::Mat(),
                    cv::Point(-1, -1),
                    leaf_dilation_val_); /*Reset Dilation In That Image*/
            }
            emit UpdateDilationBackground();
        }

        if (sym_trap_call) {
            CalculateSymTrap();
        }

        /*Move to Leaf Search If Necessary*/
        if (optimizer_settings_.enable_leaf_ && !error_occurrred_ &&
            !sym_trap_call) {
            /*Update Search Stage Flag as Leaf*/
            search_stage_flag_ = Leaf;

            /*Reset Storage, Starting Point, Range, new budget, comparison
             * image*/
            /*Reset Starting Point*/
            SetStartingPoint(current_optimum_location_);
            /*Reset Range*/
            SetSearchRange(optimizer_settings_.leaf_range);
            /*Reset Budget and Cost Function Calls*/
            budget_ += optimizer_settings_.leaf_budget;
            /*Reset Storage*/
            data_.DeleteAllStoredHyperboxes();
            /*Initialize with Unit Sized HyperBox at Center*/
            current_optimum_value_ =
                EvaluateCostFunction(Point6D(.5, .5, .5, .5, .5, .5));
            current_optimum_location_ = starting_point_;
            data_ = DirectDataStorage(current_optimum_value_);

            /*Main Loop*/
            while (cost_function_calls_ < budget_) {
                /*Scroll Through Convex Hull to Get List of Potentially
                Optimal Hyperboxes to Evaluate. This is stored as list of
                column IDs in DATA that need to have least Fvalued
                hyperbox (last in list) returned for trisection and
                evaluation.*/
                ConvexHull();

                /*Trisect Potentially Optimal Rectangles*/
                TrisectPotentiallyOptimal();

                /*Safety Break...Should Never Happen*/
                if (potentially_optimal_col_ids_.size() == 0) {
                    emit OptimizerError(
                        "Error, no potentialy optimal hyper rectangles found!");
                    error_occurrred_ = true;
                    break;
                }

                /*If Error*/
                if (error_occurrred_) break;

                /*Update Screen at Rate of 30 FPS*/
                if ((clock() - update_screen_clock_) > 33) {
                    emit UpdateDisplay(
                        static_cast<double>(clock() - start_clock_) /
                            static_cast<double>(cost_function_calls_),
                        static_cast<int>(cost_function_calls_),
                        current_optimum_value_,
                        primary_model_index_);
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

        /*Update Comparison Image in Dilation Metric and Dilation Metric
         * Dilation Level to Original*/
        dilate(
            frames_A_[frame_index].GetEdgeImage(),
            frames_A_[frame_index].GetDilationImage(),
            cv::Mat(),
            cv::Point(-1, -1),
            trunk_dilation_val_); /*Reset Dilation In That Image*/
        if (calibration_.biplane_calibration) {
            dilate(
                frames_B_[frame_index].GetEdgeImage(),
                frames_B_[frame_index].GetDilationImage(),
                cv::Mat(),
                cv::Point(-1, -1),
                trunk_dilation_val_); /*Reset Dilation In That Image*/
        }
        emit UpdateDilationBackground();

        /*Move on and Wrap Up*/
        if (error_occurrred_ || frame_index == end_frame_index_) {
            emit finished();
            break;
        } else {
            /*Update Pose*/
            /*Signal Main Window That Frame Is Optimized and Update Blue Model*/
            QString result_info = QString::fromStdString(
                OptimizerManager::print_location_and_orientation_of_point(
                    current_optimum_location_));
            emit OptimizedFrame(
                current_optimum_location_.x,
                current_optimum_location_.y,
                current_optimum_location_.z,
                current_optimum_location_.xa,
                current_optimum_location_.ya,
                current_optimum_location_.za,
                progress_next_frame_,
                primary_model_index_,
                init_prev_frame_,
                result_info);
        }
    }
}

void OptimizerManager::CalculateSymTrap() {
    /* TODO: Restore symmetry trap functionality.
       Currently calling missing sym_trap_functions::CalculateSymTrap.
       Commenting out to allow project build. */
    /*
    if (current_optimum_location_.xa == 0 &&
        current_optimum_location_.ya == 0 &&
        current_optimum_location_.za == 0) {
...
    // za - angle
    res = sym_trap_functions::CalculateSymTrap(
        this, pose_6D, current_frame, 5, 2.0);
    results.push_back(res);
    */
}

double OptimizerManager::EvaluateCostFunctionAtPoint(Point6D point, int stage) {
    auto pose = Pose(
        point.x, point.y, point.z, point.xa, point.ya, point.za);
    gpu_principal_model_->SetCurrentPrimaryCameraPose(pose);

    double result = 0;

    if (stage == 0) {
        result = trunk_manager_.callActiveCostFunction();
    } else if (stage == 1) {
        result = branch_manager_.callActiveCostFunction();
    } else if (stage == 2) {
        result = leaf_manager_.callActiveCostFunction();
    }

    return result;
}

double OptimizerManager::EvaluateCostFunction(Point6D unit_point) {
    /*Initialize denormalized point*/
    Point6D denormalized_point = DenormalizeRange(unit_point);

    /*Set Current Model Pose For Cost Functions to Read In*/
    Pose pose(
        denormalized_point.x,
        denormalized_point.y,
        denormalized_point.z,
        denormalized_point.xa,
        denormalized_point.ya,
        denormalized_point.za);
    gpu_principal_model_->SetCurrentPrimaryCameraPose(pose);
    if (calibration_.biplane_calibration) {
        Point6D secondary_pose =
            calibration_.convert_Pose_A_to_Pose_B(denormalized_point);
        gpu_principal_model_->SetCurrentSecondaryCameraPose(Pose(
            secondary_pose.x,
            secondary_pose.y,
            secondary_pose.z,
            secondary_pose.xa,
            secondary_pose.ya,
            secondary_pose.za));
    }

    /*Compute Cost Function Value*/
    double result = 0;
    switch (search_stage_flag_) {
    case Trunk:
        result = trunk_manager_.callActiveCostFunction();
        break;
    case Branch:
        result = branch_manager_.callActiveCostFunction();
        break;
    case Leaf:
        result = leaf_manager_.callActiveCostFunction();
        break;
    }
    cost_function_calls_++;

    /*Store Optimum*/
    if (result < current_optimum_value_) {
        //<= should be better like this
        current_optimum_value_ = result;
        current_optimum_location_ = denormalized_point;

        /*Throttle UI updates to ~30 FPS (every 33ms)*/
        if (optimum_update_timer_.elapsed() > 33) {
            emit UpdateOptimum(
                current_optimum_location_.x,
                current_optimum_location_.y,
                current_optimum_location_.z,
                current_optimum_location_.xa,
                current_optimum_location_.ya,
                current_optimum_location_.za,
                primary_model_index_);
            optimum_update_timer_.restart();
        }
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
    } else if (data_.GetNumberColumns() > 1) {
        /*Perform Gift Wrapping Algorithm. However, add a small epsilon to the
        smallest sized column. The size of this epsilon is to be determined by
        experiment. Different from previous JTA.*/

        /*Initialize Indexing/Intermediate Variables*/
        int right_index = data_.GetNumberColumns() - 1;
        int bottom_index = data_.GetLowestFValColId();

        /*Add Bottom Point (always optimal)*/
        potentially_optimal_col_ids_.push_back(bottom_index);

        /*While haven't reached the right-most point*/
        int current_index = bottom_index;
        while (current_index != right_index) {
            /*Find next point on convex hull (using Gift Wrapping)*/
            double min_slope = DBL_MAX;
            int next_index = current_index;
            for (int i = current_index + 1; i < data_.GetNumberColumns(); i++) {
                double slope =
                    (data_.GetFValAtColId(i) - data_.GetFValAtColId(current_index)) /
                    (data_.GetSizeAtColId(i) - data_.GetSizeAtColId(current_index));
                if (slope <= min_slope) {
                    min_slope = slope;
                    next_index = i;
                }
            }
            potentially_optimal_col_ids_.push_back(next_index);
            current_index = next_index;
        }
    }
}

void OptimizerManager::TrisectPotentiallyOptimal() {
    /*Loop Over Potentially Optimal Columns*/
    for (int i = 0; i < potentially_optimal_col_ids_.size(); i++) {
        /*Get Potentially Optimal Hyperbox*/
        HyperBox6D pot_opt_hb =
            data_.GetLowestFValHyperBoxAtColId(potentially_optimal_col_ids_[i]);

        /*Remove Hyperbox From Data Storage*/
        data_.RemoveHyperBoxAtColId(potentially_optimal_col_ids_[i], pot_opt_hb);

        /*Identify Largest Direction*/
        Direction largest_direction = pot_opt_hb.sides_.GetLargestDirection();

        /*Initialize 2 new Centers*/
        Point6D center_plus = pot_opt_hb.center_;
        Point6D center_minus = pot_opt_hb.center_;

        /*Update Centers*/
        double step_size = pot_opt_hb.sides_.GetDirection(largest_direction) / 3.0;
        center_plus.UpdateDirection(
            largest_direction,
            pot_opt_hb.center_.GetDirection(largest_direction) + step_size);
        center_minus.UpdateDirection(
            largest_direction,
            pot_opt_hb.center_.GetDirection(largest_direction) - step_size);

        /*Trisect current side*/
        pot_opt_hb.TrisectSide(largest_direction);

        /*Evaluate cost functions at new centers*/
        double value_plus = EvaluateCostFunction(center_plus);
        double value_minus = EvaluateCostFunction(center_minus);

        /*Add New Hyperboxes*/
        data_.AddHyperBox(HyperBox6D(value_plus, center_plus, pot_opt_hb.sides_));
        data_.AddHyperBox(
            HyperBox6D(value_minus, center_minus, pot_opt_hb.sides_));
        data_.AddHyperBox(pot_opt_hb); // Add self (trisected)
    }
}

Point6D OptimizerManager::DenormalizeRange(Point6D unit_point) {
    /*Denormalize Unit Point to Search Range*/
    Point6D denormalized_point;
    denormalized_point.x = starting_point_.x + (unit_point.x - 0.5) * range_.x;
    denormalized_point.y = starting_point_.y + (unit_point.y - 0.5) * range_.y;
    denormalized_point.z = starting_point_.z + (unit_point.z - 0.5) * range_.z;
    denormalized_point.xa = starting_point_.xa + (unit_point.xa - 0.5) * range_.xa;
    denormalized_point.ya = starting_point_.ya + (unit_point.ya - 0.5) * range_.ya;
    denormalized_point.za = starting_point_.za + (unit_point.za - 0.5) * range_.za;

    return denormalized_point;
}

Point6D OptimizerManager::DenormalizeFromCenter(Point6D unit_point) {
    /*Unused In Current Version*/
    return Point6D();
}

void OptimizerManager::onStopOptimizer() {
    error_occurrred_ = true;
}

void OptimizerManager::create_image_indices(
    std::vector<int>& img_indices, int start, int end) {
    /*Image Indices List*/
    img_indices.clear();
    if (start < end) {
        for (int i = start; i <= end; i++) {
            img_indices.push_back(i);
        }
    } else if (start > end) {
        for (int i = start; i >= end; i--) {
            img_indices.push_back(i);
        }
    } else if (start == end) {
        int i = start;
        img_indices.push_back(i);
    }
}

/*Destructor*/
OptimizerManager::~OptimizerManager() {
    /* All members handled by unique_ptr */
}

std::string OptimizerManager::print_location_and_orientation_of_point(Point6D p) {
    std::string output = "X: " + std::to_string(p.x) + " Y: " + std::to_string(p.y) +
                         " Z: " + std::to_string(p.z) + " XA: " + std::to_string(p.xa) +
                         " YA: " + std::to_string(p.ya) + " ZA: " + std::to_string(p.za);
    return output;
}
