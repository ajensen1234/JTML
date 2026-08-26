// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*Optimizer Manaer*/
#include "coordinator/optimizer_manager.h"

/*Pose Matrix Class*/
#include <stdlib.h>

#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>

#include "compute/batch_outcome.h"
#include "compute/cuda_launch_parameters.h"
#include "compute/evaluation_executor.h"
#include "compute/gpu_heatmaps.cuh"
#include "compute/gpu_model.cuh"
#include "compute/graph_admission_policy.h"
#include "compute/graph_key_assembler.h"
#include "compute/pose_matrix.h"
#include "direct-rs_bridge/lib.h"

OptimizerManager::OptimizerManager(QObject* parent) : QObject(parent) {
    // this->sym_trap_obj = nullptr;
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
    int iter_count,
    DirectOptimizer::Options direct_options) {
    /*Success?*/
    succesfull_initialization_ = true;

    /*Error Check for Optimizer*/
    error_occurrred_ = false;

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

    /*Just In Case Have to Delete*/
    gpu_principal_model_ = 0;
    capacity_service_ = nullptr;
    gpu_metrics_ = 0;

    /*Store Camera Frame Lists Locally and Check That, if Biplane is Enabled ->
    both lists are the same size. Also Check that the current frame index is
    within the range of the frame list sizes.*/
    frames_A_ = camera_A_frame_list;
    frames_B_ = camera_B_frame_list;
    if (calibration_.biplane_calibration &&
        frames_A_.size() != frames_B_.size()) {
        error_message =
            "Biplane mode enabled, but each camera has a different "
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
    /*Store the per-stage optimizer-variant slot (plan 008 U8) -- consumed by
     * RunDirectStage's DirectOptimizer ctor; the defaults reproduce the
     * pre-Options search bit-identically.*/
    direct_options_ = direct_options;

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
        /*U6: pin the single-frame scope (mirrors the Single branch). Without
         * this, end_frame_index_ stays UNINITIALIZED and create_image_indices
         * reads garbage (descending negative range -> UB, or a many-thousand-
         * iteration sym-trap hang). The tibia-after-femur oracle (plan 008
         * U6, jtml.oracle_multistage) is the first executor of this path. */
        end_frame_index_ = current_frame_index;

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

    /*Plan 008 U9 (Cut B): build the run's StageScript ONCE from the settings
     * + directive — the container's stage policy is DATA (U7's pure builder),
     * and Optimize()'s per-frame loop iterates it. The manager's own
     * directive validation above runs FIRST (its "Unrecognized optimization
     * directive" error path is unchanged), so the builder's unknown-directive
     * error is unreachable here; a NEGATIVE-budget settings corruption fails
     * fast through the manager's existing error path (error_message + failed
     * Initialize) instead of the engine's silent acceptance — the pure
     * builder's documented strictness (U7), never hit by the tested shapes.*/
    try {
        stage_script_ = jta::BuildStageScript(
            optimizer_settings_, optimization_directive_.toStdString());
    } catch (const std::invalid_argument& e) {
        error_message = QString::fromStdString(e.what());
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
    if (cudaResultCode != cudaSuccess) {
        device_count = 0;
    }
    /* Machines with no GPUs can still report one emulation device */
    for (int device = 0; device < device_count; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999 &&
            properties.major >= 5) { /* 9999 means emulation only */
            ++gpu_device_count;
        }
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

    /*Get the Dilation + Dark Silhouette values for Trunk, Branch, and Leaf —
     * one DeriveStageParams call per stage manager (the U7 pure relocation of
     * the inline scan; the six-name variant list lives in ONE place, the pure
     * TU). Values are identical to the pre-Cut-C inline scans (last-match-
     * wins, ≤0 clamp, DIRECT_MAHFOUZ → 3, the six bool-name variants) — the
     * U6 oracle re-verifies bit-identity.*/
    const jta::StageCostParams trunk_params = DeriveStageParams(trunk_manager_);
    trunk_dilation_val_ = trunk_params.dilation;
    trunk_dark_silhouette_val_ = trunk_params.dark_silhouette;

    const jta::StageCostParams branch_params =
        DeriveStageParams(branch_manager_);
    branch_dilation_val_ = branch_params.dilation;
    branch_dark_silhouette_val_ = branch_params.dark_silhouette;

    const jta::StageCostParams leaf_params = DeriveStageParams(leaf_manager_);
    leaf_dilation_val_ = leaf_params.dilation;
    leaf_dark_silhouette_val_ = leaf_params.dark_silhouette;

    /*Upload GPU Frames*/
    /*Intensity Frames*/
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
            gpu_intensity_frames_trunk_A_.push_back(intensity_frame);
        } else {
            delete intensity_frame;
            error_message = "Error uploading intensity frame to GPU!";
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
                gpu_intensity_frames_trunk_B_.push_back(intensity_frame);
            } else {
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
        auto intensity_frame = new GPUIntensityFrame(
            width,
            height,
            cuda_device_id,
            frames_A_[i].GetOriginalImage().data,
            branch_dark_silhouette_val_,
            frames_A_[i].GetInvertedImage().data);
        if (intensity_frame->IsInitializedCorrectly()) {
            gpu_intensity_frames_branch_A_.push_back(intensity_frame);
        } else {
            delete intensity_frame;
            error_message = "Error uploading intensity frame to GPU!";
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
                gpu_intensity_frames_branch_B_.push_back(intensity_frame);
            } else {
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
        auto intensity_frame = new GPUIntensityFrame(
            width,
            height,
            cuda_device_id,
            frames_A_[i].GetOriginalImage().data,
            leaf_dark_silhouette_val_,
            frames_A_[i].GetInvertedImage().data);
        if (intensity_frame->IsInitializedCorrectly()) {
            gpu_intensity_frames_leaf_A_.push_back(intensity_frame);
        } else {
            delete intensity_frame;
            error_message = "Error uploading intensity frame to GPU!";
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
                gpu_intensity_frames_leaf_B_.push_back(intensity_frame);
            } else {
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
            gpu_dilated_frames_leaf_A_.push_back(dilated_frame);
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
                gpu_dilated_frames_leaf_B_.push_back(dilated_frame);
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
            gpu_dilated_frames_branch_A_.push_back(dilated_frame);
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
                gpu_dilated_frames_branch_B_.push_back(dilated_frame);
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
            gpu_dilated_frames_trunk_A_.push_back(dilated_frame);
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
                gpu_dilated_frames_trunk_B_.push_back(dilated_frame);
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
            gpu_edge_frames_A_.push_back(edge_frame);
        } else {
            delete edge_frame;
            error_message = "Error uploading edge frame to GPU!";
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
                gpu_edge_frames_B_.push_back(edge_frame);
            } else {
                delete edge_frame;
                error_message = "Error uploading edge frame to GPU!";
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
            gpu_distance_maps_.push_back(distance_map);

        } else {
            delete distance_map;
            error_message = "Error uploading distance map to GPU!";
            succesfull_initialization_ = false;
            return succesfull_initialization_;
        }
    }
    for (int i = 0; i < frames_A_.size(); i++) {
        /*Owner fix (2026-08-12): curvature heatmaps only exist after an ML
         * segmentation (Frame::setCurvatureHeatmaps); a study loaded
         * without segmentation has none, and the upload must not abort the
         * run. The GPUHeatmap ctor treats num_keypoints <= 0 as a
         * legitimate no-upload state (initialized, 0 keypoints), keeping
         * the vector frame-aligned so the cost functions' per-frame at(i)
         * access stays valid (GetNumKeypoints() == 0 -> the curvature
         * costs no-op; the distance-map costs are unaffected).*/
        auto frame_heatmaps = frames_A_[i].getCurvatureHeatmaps();
        auto heatmap = new GPUHeatmap(
            width,
            height,
            cuda_device_id,
            frames_A_[i].GetNumCurvatureKeypoints(),
            frame_heatmaps.empty() ? nullptr : frame_heatmaps.data());
        if (heatmap->IsInitializedCorrectly()) {
            gpu_heatmaps_.push_back(heatmap);
        } else {
            delete heatmap;
            error_message = "Error uploading heatmap to GPU!";
            succesfull_initialization_ = false;
            return succesfull_initialization_;
        }
    }

    /*Upload GPU Models*/
    /*Monoplane Calibration*/
    if (!calibration_.biplane_calibration) {
        /*Principal Model*/
        gpu_principal_model_ = new GPUModel(
            primary_model_.model_name_,
            true,
            width,
            height,
            cuda_device_id,
            true,
            &primary_model_.triangle_vertices_[0],
            &primary_model_.triangle_normals_[0],
            primary_model_.triangle_vertices_.size() / 9,
            calibration_.camera_A_principal_);

        if (!gpu_principal_model_->IsInitializedCorrectly()) {
            delete gpu_principal_model_;
            gpu_principal_model_ = 0;
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
                gpu_non_principal_models_.push_back(gpu_non_principal_model);
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
        gpu_principal_model_ = new GPUModel(
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
            calibration_.camera_B_principal_);
        if (!gpu_principal_model_->IsInitializedCorrectly()) {
            delete gpu_principal_model_;
            gpu_principal_model_ = 0;
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
                gpu_non_principal_models_.push_back(gpu_non_principal_model);
            } else {
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

    /* Plan 010 U12: configure the service-owned extra-bank pool only for the
     * supported monoplane principal DIRECT_DILATION path. Bank 0 remains owned
     * by the model/metrics compatibility objects; unsupported/biplane paths
     * retain poolSize()==1 and the exact serial adapter. */
    capacity_service_ = new CostCapacityService();
    if (!calibration_.biplane_calibration &&
        capacity_service_->refreshDeviceSnapshot(cuda_device_id)) {
        gpu_cost_function::BankFootprintInput bank_layout;
        bank_layout.width = static_cast<std::uint64_t>(width);
        bank_layout.height = static_cast<std::uint64_t>(height);
        bank_layout.triangle_count = static_cast<std::uint64_t>(
            primary_model_.triangle_vertices_.size() / 9);
        bank_layout.maximum_stride_size = maximum_stride_size;
        bank_layout.cub_storage_bytes =
            gpu_principal_model_->GetPrimaryCubStorageBytes();
        bank_layout.biplane = false;
        const bool admitted = capacity_service_->ConfigurePool(bank_layout, 3);
        if (admitted) {
            const int probe_bank = capacity_service_->CheckoutBank();
            if (probe_bank >= 0) {
                capacity_service_->RecycleBank(
                    static_cast<std::size_t>(probe_bank), true);
            }
        }
        gpu_principal_model_->SetCapacityService(capacity_service_);
    }

    /* Plan 012 U1 (C10): graph pool is lazy — default-deny means no preparation
     * attempt and no 8 GiB dummy allocation at manager setup.
     * evaluation_executor_ stays constructed but uninitialized (poolSize()==0).
     */
    evaluation_executor_ = new gpu_cost_function::EvaluationExecutor();

    /*Upload Data To CostFunction Managers*/
    trunk_manager_.UploadData(
        &gpu_edge_frames_A_,
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
    branch_manager_.UploadData(
        &gpu_edge_frames_A_,
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
    leaf_manager_.UploadData(
        &gpu_edge_frames_A_,
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
    trunk_manager_.UploadDistanceMap(&gpu_distance_maps_, &gpu_heatmaps_);
    branch_manager_.UploadDistanceMap(&gpu_distance_maps_, &gpu_heatmaps_);
    leaf_manager_.UploadDistanceMap(&gpu_distance_maps_, &gpu_heatmaps_);

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
                    emit OptimizerError(
                        QString::fromStdString(
                            "Could not retrieve pose for non-principal model "
                            "\"" +
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
            search_stage_flag_ = Stage::Trunk;

            /*Start Clock*/
            start_clock_ = clock();
            update_screen_clock_ = clock();
        }

        /*****************SCRIPT-DRIVEN STAGE LOOP (plan 008 U9) ******/
        /*The run's stage policy is DATA — stage_script_, built once in
         * Initialize from the settings + directive (U7's BuildStageScript; the
         * Sym_Trap directive yields the leaf-only [{Leaf, repeat=0}]
         * script, the normal directives the trunk/branch/leaf shape). The
         * loop sits OUTSIDE the !sym_trap_call guard, mirroring the pre-Cut-B
         * leaf section: under Sym_Trap the script holds only the leaf spec, so
         * only leaf-init + dilate + emit + CalculateSymTrap run (no search;
         * costCalls stays 0 — the U6 sym-trap pins). Each
         * spec names its CostFunctionManager by cfm_index (0/1/2 -> the
         * trunk/branch/leaf managers) and its cost parameters are derived
         * from that manager's parameter registry via DeriveStageCostParams
         * — the U7 pure relocation of the scan Initialize used to perform
         * inline (same values: 6/4/1 dilation on the production shape).
         * The emit order, the error gating, and the cumulative budget
         * accounting are transcribed VERBATIM from the pre-Cut-B blocks:
         *  - trunk: init + dilate + emit UNCONDITIONAL; search gated on
         *    !error_occurrred_; destruct UNCONDITIONAL (the trunk side of
         *    the leaf-destruct error-gating asymmetry, preserved verbatim
         *    — flagged to the hygiene pass, NOT fixed);
         *  - branch: init + dilate + emit ONCE PER GROUP (gated on
         *    enable_branch_ && number_branches > 0 && !error_occurrred_ —
         *    the group-once dilation pin); per-repeat re-seed from the
         *    CURRENT optimum + budget_ += spec.budget (cumulative);
         *  - leaf: init + dilate + emit gated on enable_leaf_ &&
         *    !error_occurrred_; CalculateSymTrap under the Sym_Trap
         *    directive (the repeat=0 no-search leaf); search gated on
         *    enable_leaf_ && !error_occurrred_ && !sym_trap_call &&
         *    repeat > 0; destruct gated on enable_leaf_ && !error_occurrred_
         *    (the leaf side of the asymmetry, preserved verbatim);
         *  - a cfm_index outside 0..2 fails fast through the manager's
         *    existing error path (OptimizerError + error_occurrred_, no
         *    silent stage skip).
         * budget_ is NOT re-touched for the trunk spec (the pre-trunk
         * block above already reset it to trunk_budget); branch/leaf
         * accumulate so the caps gate stays on the cumulative
         * 20/25/30/35k shape.*/
        for (const jta::StageSpec& spec : stage_script_) {
            /*cfm_index -> the three managers (fail fast on a bad index).*/
            jta_cost_function::CostFunctionManager* stage_manager = nullptr;
            switch (spec.cfm_index) {
            case 0u:
                stage_manager = &trunk_manager_;
                break;
            case 1u:
                stage_manager = &branch_manager_;
                break;
            case 2u:
                stage_manager = &leaf_manager_;
                break;
            default:
                emit OptimizerError(
                    QString::fromStdString(
                        "OptimizerManager: stage cfm_index " +
                        std::to_string(spec.cfm_index) +
                        " out of range (valid 0..2); run aborted"));
                error_occurrred_ = true;
                break;
            }
            if (stage_manager == nullptr) {
                break; /*bad cfm_index: the error was emitted above*/
            }

            switch (spec.kind) {
            /**************TRUNK SPEC (cfm 0)**********************/
            case jta::StageKind::Trunk: {
                /*Call Trunk Initializer (unconditional — verbatim)*/
                if (!stage_manager->InitializeActiveCostFunction(
                        error_message)) {
                    emit OptimizerError(QString::fromStdString(error_message));
                    error_occurrred_ = true;
                }

                const jta::StageCostParams trunk_params =
                    DeriveStageParams(*stage_manager);

                /*Make Sure Dilation Image is Showing Trunk Value (Should be
                 * Unnecessary)*/
                ResetStageDilation(frame_index, trunk_params.dilation);

                /*Run the trunk stage of DIRECT bound to the real GPU cost
                 * (budget_ was just reset to trunk_budget and
                 * cost_function_calls_ to 0 above; RunDirectStage uses the
                 * cumulative call-offset and drives the live UpdateDisplay /
                 * UpdateOptimum signals).*/
                if (!error_occurrred_) {
                    RunDirectStage(spec.range, *stage_manager);
                }

                /*Destruct Trunk Manager Initialization (unconditional —
                 * verbatim)*/
                if (!stage_manager->DestructActiveCostFunction(error_message)) {
                    emit OptimizerError(QString::fromStdString(error_message));
                    error_occurrred_ = true;
                }
                break;
            }
            /**************BRANCH SPEC (cfm 1)*********************/
            case jta::StageKind::Branch: {
                /*Construct Branch Manager Initialization — the GROUP init +
                 * dilate + emit fires exactly once per frame (the group-once
                 * dilation pin).*/
                if (optimizer_settings_.enable_branch_ &&
                    optimizer_settings_.number_branches > 0 &&
                    !error_occurrred_) {
                    if (!stage_manager->InitializeActiveCostFunction(
                            error_message)) {
                        emit OptimizerError(
                            QString::fromStdString(error_message));
                        error_occurrred_ = true;
                    }

                    const jta::StageCostParams branch_params =
                        DeriveStageParams(*stage_manager);

                    /*Make Sure Dilation Image is Showing Branch Value */
                    ResetStageDilation(frame_index, branch_params.dilation);
                }

                /*Move to Branch If Necessary: one search per repeat, each
                 * re-seeded from the CURRENT optimum (the per-repeat re-seed
                 * lineage invariant — reading current_optimum_location_,
                 * never a captured one).*/
                for (unsigned int branch_index = 0; branch_index < spec.repeat;
                     branch_index++) {
                    /*If Error*/
                    if (error_occurrred_) {
                        break;
                    }

                    /*Update Search Stage Flag as Branch*/
                    search_stage_flag_ = Stage::Branch;

                    /*Reset Storage, Starting Point, Range, new budget,
                     * comparison image*/
                    /*Reset Starting Point*/
                    SetStartingPoint(current_optimum_location_);
                    /*Reset Range*/
                    SetSearchRange(spec.range);
                    /*Reset Budget and Cost Function Calls*/
                    budget_ += spec.budget;
                    /*Run this branch stage of DIRECT bound to the real GPU
                     * cost. budget_ is cumulative (trunk + branch);
                     * RunDirectStage uses it as the stage cap against the
                     * running cost_function_calls_ offset.*/
                    RunDirectStage(spec.range, *stage_manager);
                }
                break;
            }
            /**************LEAF SPEC (cfm 2)***********************/
            case jta::StageKind::Leaf: {
                /*Construct Leaf Initialization*/
                if (optimizer_settings_.enable_leaf_ && !error_occurrred_) {
                    if (!stage_manager->InitializeActiveCostFunction(
                            error_message)) {
                        emit OptimizerError(
                            QString::fromStdString(error_message));
                        error_occurrred_ = true;
                    }

                    const jta::StageCostParams leaf_params =
                        DeriveStageParams(*stage_manager);

                    /*Make Sure Dilation Image is Showing Leaf Value */
                    ResetStageDilation(frame_index, leaf_params.dilation);
                }

                /*Sym_Trap: the repeat=0 no-search leaf — init + dilate +
                 * emit + CalculateSymTrap, NO search (gated ONLY on
                 * sym_trap_call, verbatim — the engine runs CalculateSymTrap
                 * even after a leaf-init error, a latent hazard preserved
                 * here).*/
                if (sym_trap_call) {
                    CalculateSymTrap();
                }

                /*Move to Leaf Search If Necessary*/
                if (optimizer_settings_.enable_leaf_ && !error_occurrred_ &&
                    !sym_trap_call && spec.repeat > 0) {
                    /*Update Search Stage Flag as Leaf*/
                    search_stage_flag_ = Stage::Leaf;

                    /*Reset Storage, Starting Point, Range, new budget,
                     * comparison image*/
                    /*Reset Starting Point*/
                    SetStartingPoint(current_optimum_location_);
                    /*Reset Range*/
                    SetSearchRange(spec.range);
                    /*Reset Budget and Cost Function Calls*/
                    budget_ += spec.budget;
                    /*Run the leaf stage of DIRECT bound to the real GPU cost.
                     * budget_ is cumulative (trunk + branch + leaf);
                     * RunDirectStage uses it as the stage cap against the
                     * running cost_function_calls_ offset.*/
                    RunDirectStage(spec.range, *stage_manager);
                }

                /*Destruct Leaf Initialization CFM — gated on
                 * !error_occurrred_ (the leaf side of the leaf-destruct
                 * error-gating asymmetry; preserved verbatim, flagged to the
                 * hygiene pass).*/
                if (optimizer_settings_.enable_leaf_ && !error_occurrred_) {
                    if (!stage_manager->DestructActiveCostFunction(
                            error_message)) {
                        emit OptimizerError(
                            QString::fromStdString(error_message));
                        error_occurrred_ = true;
                    }
                }
                break;
            }
            }
        }

        /*****************STAGE LOOP END *****************************/

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
            progress_next_frame_ = false;
        }
        emit OptimizedFrame(
            current_optimum_location_.x,
            current_optimum_location_.y,
            current_optimum_location_.z,
            current_optimum_location_.xa,
            current_optimum_location_.ya,
            current_optimum_location_.za,
            progress_next_frame_,
            primary_model_index_,
            error_occurrred_,
            optimization_directive_);

        if (sym_trap_call) {
            emit finished();
            return;
        }

        emit UpdateDisplay(
            static_cast<double>(clock() - start_clock_) /
                static_cast<double>(cost_function_calls_),
            static_cast<int>(cost_function_calls_),
            current_optimum_value_,
            primary_model_index_);
        update_screen_clock_ = clock();

        /*Update Pose Storage*/
        auto current_opt_pose = Pose(
            current_optimum_location_.x,
            current_optimum_location_.y,
            current_optimum_location_.z,
            current_optimum_location_.xa,
            current_optimum_location_.ya,
            current_optimum_location_.za);
        pose_storage_.UpdatePrincipalModelPose(frame_index, current_opt_pose);

        /*If Error Occurred or Not Progressing Breank (Which Ends)*/
        if (!progress_next_frame_) {
            break;
        }
    }

    /*Finish And Return*/
    emit finished();
}

jta::StageCostParams OptimizerManager::DeriveStageParams(
    jta_cost_function::CostFunctionManager& manager) {
    /*The one-line shim over jta::DeriveStageCostParams (U7 pure TU — the
     * six-name variant list lives there, in ONE place). Every call site stays
     * exactly where the pre-shim DeriveStageCostParams calls were: in the
     * stage loop the call runs AFTER the stage's InitializeActiveCostFunction
     * (the init-gating order is load-bearing) — never hoisted above the kind
     * switch.*/
    return jta::DeriveStageCostParams(
        manager.getActiveCostFunction(),
        manager.getActiveCostFunctionClass()->getIntParameters(),
        manager.getActiveCostFunctionClass()->getBoolParameters());
}

void OptimizerManager::ResetStageDilation(size_t frame_index, int dilation) {
    /*Make Sure the Dilation Image is Showing the Given Dilation Value (Reset
     * Dilation In That Image) — the dilate-A / dilate-B (if biplane) / emit
     * UpdateDilationBackground block, one place for the trunk/branch/leaf
     * stage specs (the dilation value is the only per-stage difference).*/
    dilate(
        frames_A_[frame_index].GetEdgeImage(),
        frames_A_[frame_index].GetDilationImage(),
        cv::Mat(),
        cv::Point(-1, -1),
        dilation); /*Reset Dilation In That Image*/
    if (calibration_.biplane_calibration) {
        dilate(
            frames_B_[frame_index].GetEdgeImage(),
            frames_B_[frame_index].GetDilationImage(),
            cv::Mat(),
            cv::Point(-1, -1),
            dilation); /*Reset Dilation In That Image*/
    }
    emit UpdateDilationBackground();
    // Plan 012 U2: bump upload epoch for generation identity (C7) — CPU dilate
    // rewrites shared frames_A_/B_
    trunk_manager_.BumpUploadEpoch();
    branch_manager_.BumpUploadEpoch();
    leaf_manager_.BumpUploadEpoch();
}

void OptimizerManager::RunDirectStage(
    Point6D range,
    jta_cost_function::CostFunctionManager& stage_manager) {
    /*Cross the extracted pure optimizer boundary with the real GPU cost. The
     * DirectOptimizer hands the injected lambda the *denormalized physical*
     * point, so set the GPU model poses from it directly (primary, + biplane
     * secondary via the calibration), then score the stage's cost function.
     * The injected cost IS jta::BuildGpuCostAdapter (plan 008 U9) — the one
     * shared lambda body the production runner, the Tier-2 oracle twin, and
     * the z-profile probe converge on (set pose -> score the stage's active
     * cost function; Calibration by value, monoplane default).
     * No here-optimum tracking: DirectOptimizer owns that internally and we
     * read it back after Run().*/

    auto serial_cost = jta::BuildGpuCostAdapter(
        gpu_principal_model_, calibration_, stage_manager);

#if USE_RUST_DIRECT
    CppCost cost = CppCost(serial_cost);
    std::cout << budget_ << "\n";

    rust::Box<direct_rs::DirectOptimizer> rust_opt = direct_rs::new_rust_opt(
        range.to_array(),
        starting_point_.to_array(),
        (budget_ - cost_function_calls_));

    RunOutcome out = rust_opt->run_rust_opt(cost);

    cost_function_calls_ += out.num_iter;
    current_optimum_location_ = Point6D(out.optimal_location);
    current_optimum_value_ = out.optimal_value;

#else
    DirectOptimizer opt(
        serial_cost, range, starting_point_, budget_, direct_options_);

    /* U12 production batch bridge: only the admitted monoplane
     * DIRECT_DILATION path receives the bank scheduler. All unsupported,
     * biplane, unavailable, or pool-size-one cases retain the exact serial
     * adapter above. */
    if (capacity_service_ != nullptr && capacity_service_->poolSize() > 1 &&

        !calibration_.biplane_calibration &&
        stage_manager.getActiveCostFunction() == "DIRECT_DILATION") {
        std::cout << "Using capacity service\n";
        opt.SetBatchCost([this, &stage_manager, serial_cost](
                             const std::vector<Point6D>& poses) {
            const auto enqueue = [this, &stage_manager](
                                     const Point6D& physical,
                                     gpu_cost_function::BankState& bank) {
                gpu_principal_model_->SetCurrentPrimaryCameraPose(Pose(
                    physical.x,
                    physical.y,
                    physical.z,
                    physical.xa,
                    physical.ya,
                    physical.za));
                const auto error =
                    stage_manager.EnqueueDirectDilationOnBank(bank);
                if (error != cudaSuccess) {
                    return static_cast<int>(error);
                }
                return static_cast<int>(error);
            };
            const auto complete =
                [&stage_manager](gpu_cost_function::BankState& bank) {
                    return stage_manager.CompleteDirectDilationOnBank(bank);
                };
            return capacity_service_->RunCostBatchGreedy(
                poses, serial_cost, enqueue, complete);
        });
    }

    /* Plan 012 U1+U2: graph executor admission is default-deny (C10/R14) and
     * uses the U2 full key+generation assembler. The U12/serial adapter above
     * stays installed unless a complete admission transaction (policy +
     * recipe + preflight) succeeds —
     * which no production path can reach yet. */
    {
        bool monoplaneEligible = !calibration_.biplane_calibration &&
            stage_manager.getActiveCostFunction() == "DIRECT_DILATION";
        const auto* recipe = (evaluation_executor_ != nullptr)
            ? evaluation_executor_->registry().FindEligible(
                  "DIRECT_DILATION", false)
            : nullptr;
        bool recipeFound = recipe != nullptr;
        bool preflightCapturable = false;
        gpu_cost_function::GraphRecipeKey preparedKey;
        bool haveKey = false;
        if (recipeFound) {
            int liveDilation = 6;
            // Canonical dilation read (graph_recipe.h:9-11) — NOT a by-value
            // getAvailableCostFunctions() copy, which could diverge from the
            // provider. The active cost function is DIRECT_DILATION here (gated
            // above) and always present in available_cost_functions_, so
            // getActiveCostFunctionClass() hits and does not leak.
            if (auto* cls = stage_manager.getActiveCostFunctionClass()) {
                int v = 6;
                cls->getIntParameterValue("Dilation", v);
                liveDilation = v;
            }
            (void)liveDilation;  // consumed below in kin.dilation
            gpu_cost_function::GraphKeyAssemblerInputs kin;
            kin.recipeId = recipe->recipeId();
            kin.width = gpu_principal_model_
                ? gpu_principal_model_->GetPrimaryWidth()
                : 0;
            kin.height = gpu_principal_model_
                ? gpu_principal_model_->GetPrimaryHeight()
                : 0;
            kin.triangle_count = gpu_principal_model_
                ? static_cast<std::uint64_t>(
                      gpu_principal_model_->GetPrimaryTriangleCount())
                : 0;
            kin.dilation = liveDilation;
            kin.camera_calib_hash =
                gpu_cost_function::HashCameraCalibrationParams(
                    calibration_.camera_A_principal_.principal_distance_,
                    calibration_.camera_A_principal_.principal_x_,
                    calibration_.camera_A_principal_.principal_y_,
                    calibration_.camera_A_principal_.pixel_pitch_,
                    calibration_.biplane_calibration);
            kin.cub_storage_bytes = gpu_principal_model_
                ? gpu_principal_model_->GetPrimaryCubStorageBytes()
                : 0;
            kin.curvature_capacity = 0;
            kin.maximum_stride_size =
                static_cast<std::uint64_t>(maximum_stride_size);
            kin.graph_overhead_bytes = 0;
            kin.biplane = false;
            kin.version = "1";
            auto key = gpu_cost_function::AssembleGraphRecipeKey(kin);
            preparedKey = key;
            haveKey = true;
            gpu_cost_function::GraphRecipeCaptureInputs capInputs;
            bool inputsOk =
                stage_manager.GetGraphRecipeCaptureInputs(capInputs);
            int stage_id = 0;
            if (&stage_manager == &branch_manager_) {
                stage_id = 1;
            } else if (&stage_manager == &leaf_manager_) {
                stage_id = 2;
            }
            gpu_cost_function::CaptureGenerationAssemblerInputs gin;
            gin.frame_index =
                static_cast<int>(stage_manager.getCurrentFrameIndex());
            gin.stage_id = stage_id;
            gin.dilation = key.dilation;
            gin.upload_epoch = stage_manager.getUploadEpoch();
            gin.rendered_image = capInputs.rendered_image;
            gin.comparison_frame = capInputs.comparison_frame;
            gin.distance_map = capInputs.distance_map;
            auto gen = gpu_cost_function::AssembleCaptureGeneration(gin);
            (void)gen;
            if (!inputsOk ||
                !gpu_cost_function::ValidateGraphKeyVsInputs(key, capInputs)) {
                preflightCapturable = false;
            } else {
                auto pre = recipe->preflight(key);
                preflightCapturable = pre.capturable;
            }
        }
        bool executorReady = evaluation_executor_ != nullptr &&
            evaluation_executor_->poolSize() > 1;
        gpu_cost_function::GraphAdmissionInputs inputs;
        inputs.executorReady = executorReady;
        inputs.monoplaneEligible = monoplaneEligible;
        inputs.recipeFound = recipeFound;
        inputs.preflightCapturable = preflightCapturable;
        inputs.evidence = gpu_cost_function::GraphAdmissionEvidence{};
        gpu_cost_function::GraphAdmissionPolicy defaultPolicy;
        auto decision =
            gpu_cost_function::DecideGraphAdmission(inputs, defaultPolicy);
        if (decision.install) {
            auto* exec = evaluation_executor_;
            // Plan 012 U4 (C5): install CUDA feeder hooks only after admission.
            gpu_cost_function::InstallCudaFeederHooks(*exec);
            bool prepareOk = false;
            if (haveKey && exec->poolSize() > 1) {
                auto prep = exec->Prepare(preparedKey, exec->poolSize());
                prepareOk = prep.isOrderedScores();
            } else if (haveKey) {
                // Pool not yet sized (lazy) — treat as prepare not needed for
                // U4 seam compile
                prepareOk = true;
            }
            if (prepareOk) {
                opt.SetBatchCost(
                    [exec, serial_cost](const std::vector<Point6D>& poses)
                        -> std::vector<double> {
                        return gpu_cost_function::MaterializeOrderedScores(
                            exec->RunBatch(poses, serial_cost));
                    });
            }
        }
    }

    /*Cumulative budget semantics: this stage continues from the running call
     * count, so the extracted optimizer's loop guard uses call_offset_ + its
     * own count against the (already-accumulated) budget_ member.*/
    opt.SetCallOffset(cost_function_calls_);

    /*Live optimum display when the search improves (mirrors the original
     * UpdateOptimum emit inside EvaluateCostFunction).*/
    opt.SetImprovementCallback([this](const Point6D& loc, double) {
        emit UpdateOptimum(
            loc.x, loc.y, loc.z, loc.xa, loc.ya, loc.za, primary_model_index_);
    });

    /*Progress at ~30fps + cooperative stop, fired after each ConvexHull+Trisect
     * iteration (the stage's cooperative break boundary). onStopOptimizer sets
     * error_occurrred_, which is NOT polled inside DirectOptimizer::Run(), so
     * forward it to opt.Stop() here -- mirroring the original per-stage
     * `if (error_occurrred_) break;`.*/
    opt.SetIterationCallback([this, &opt]() {
        if (error_occurrred_) {
            opt.Stop();
            return;
        }
        if ((clock() - update_screen_clock_) > 33) {
            emit UpdateDisplay(
                static_cast<double>(clock() - start_clock_) /
                    static_cast<double>(opt.GetCostFunctionCalls()),
                static_cast<int>(opt.GetCostFunctionCalls()),
                opt.GetOptimumValue(),
                primary_model_index_);
            update_screen_clock_ = clock();
        }
    });

    {
        QString stageError;
        if (!jta::RunDirectStageGuarded(opt, &stageError)) {
            emit OptimizerError(
                stageError.isEmpty()
                    ? QStringLiteral("Error optimizing current frame!")
                    : stageError);
            error_occurrred_ = true;
            return;
        }
    }

    /*Write the stage result back into the running members.*/
    cost_function_calls_ = opt.GetCostFunctionCalls();
    current_optimum_location_ = opt.GetOptimumLocation();
    current_optimum_value_ = opt.GetOptimumValue();
#endif
}

void OptimizerManager::CalculateSymTrap() {
    if (current_optimum_location_.xa == 0 &&
        current_optimum_location_.ya == 0 &&
        current_optimum_location_.za == 0) {
        std::cout << "ERROR: INVALID STARTING POSE FOR SYMMETRY TRAP" << endl;
        return;
    }
    // Store cost values to input to csv
    std::vector<double> Costs;

    // Get number of iterations from sym_trap spin box
    // int iter_val = sym_trap_obj->getIterCount() * 3;
    int iter_val = 60;  // iter_count * 3;
    std::cout << "Sym Trap Iteration size: " << iter_val << std::endl;

    // Get pose list from sym trap
    std::vector<Point6D> pose_list(0);
    Point6D pose_6D(current_optimum_location_);
    create_vector_of_poses(pose_list, pose_6D, 20);

    int progress_val = 0;
    // Calculate cost function at each pose
    for (int i = 0; i < iter_val; i++) {
        emit onUpdateOrientationSymTrap(
            pose_list.at(i).x,
            pose_list.at(i).y,
            pose_list.at(i).z,
            pose_list.at(i).xa,
            pose_list.at(i).ya,
            pose_list.at(i).za);
        std::this_thread::sleep_for(std::chrono::milliseconds(5000 / iter_val));
        double myCost =
            EvaluateCostFunctionAtPoint(pose_list.at(i), 2);  // Use leaf
        Costs.push_back(myCost);
        std::cout << i + 1 << ": " << myCost << " @ rotation ("
                  << pose_list.at(i).xa << " " << pose_list.at(i).ya << " "
                  << pose_list.at(i).za << ")" << std::endl;

        // Update progress bar according to number of iterations
        progress_val = (i + 1) * 100 / iter_val;
        emit onProgressBarUpdate(progress_val);
    }

    // set model back to intial pose
    emit onUpdateOrientationSymTrap(
        pose_6D.x, pose_6D.y, pose_6D.z, pose_6D.xa, pose_6D.ya, pose_6D.za);

    // Csv of position and cost value (xangle,yangle,zangle,cost value \n)
    std::ofstream myfile;
    myfile.open("Results.csv");
    for (int i = 0; i < iter_val; i++) {
        myfile << pose_list.at(i).xa << "," << pose_list.at(i).ya << ","
               << pose_list.at(i).za << "," << Costs.at(i) << "\n";
    }
    myfile.close();

    // Used for Sym Trap VTK plot
    std::ofstream myfile2;
    myfile2.open("Results.xyz");
    for (int i = 0; i < iter_val; i++) {
        myfile2 << pose_list.at(i).xa << " " << pose_list.at(i).ya << " "
                << Costs.at(i) << "\n";
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

    /*Send the already-physical pose directly (no denormalize step).*/
    Pose pose(point.x, point.y, point.z, point.xa, point.ya, point.za);
    gpu_principal_model_->SetCurrentPrimaryCameraPose(pose);

    double result = 0;
    switch (stage) {
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
    emit CostFuncAtPoint(result);

    return result;
}

void OptimizerManager::onStopOptimizer() {
    error_occurrred_ = true;
}

void OptimizerManager::create_image_indices(
    std::vector<int>& img_indices,
    int start,
    int end) {
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
    /*GPU Metrics Class*/
    delete gpu_metrics_;
    /* U12 service-owned extra banks must be destroyed before model/metric
     * owners. */
    delete capacity_service_;
    capacity_service_ = nullptr;
    // U6: EvaluationExecutor must be destroyed before model/metric owners
    // (waits for streams/events)
    delete evaluation_executor_;
    evaluation_executor_ = nullptr;

    /* DESTRUCT CUDA Cost Function Objects (Vector of GPU Models and vector of
    GPU Frames - note Dilated and Intensity must have own vector for each stage
    because their values could change with the stage from a black silhouette
    bool or a dilation int)*/
    /*Camera A (Monoplane or Biplane)*/
    for (int i = 0; i < gpu_intensity_frames_trunk_A_.size(); i++) {
        delete gpu_intensity_frames_trunk_A_[i];
    }
    for (int i = 0; i < gpu_intensity_frames_branch_A_.size(); i++) {
        delete gpu_intensity_frames_branch_A_[i];
    }
    for (int i = 0; i < gpu_intensity_frames_leaf_A_.size(); i++) {
        delete gpu_intensity_frames_leaf_A_[i];
    }
    for (int i = 0; i < gpu_edge_frames_A_.size(); i++) {
        delete gpu_edge_frames_A_[i];
    }
    for (int i = 0; i < gpu_dilated_frames_trunk_A_.size(); i++) {
        delete gpu_dilated_frames_trunk_A_[i];
    }
    for (int i = 0; i < gpu_dilated_frames_branch_A_.size(); i++) {
        delete gpu_dilated_frames_branch_A_[i];
    }
    for (int i = 0; i < gpu_dilated_frames_leaf_A_.size(); i++) {
        delete gpu_dilated_frames_leaf_A_[i];
    }
    /*Camera B (Biplane only)*/
    for (int i = 0; i < gpu_intensity_frames_trunk_B_.size(); i++) {
        delete gpu_intensity_frames_trunk_B_[i];
    }
    for (int i = 0; i < gpu_intensity_frames_branch_B_.size(); i++) {
        delete gpu_intensity_frames_branch_B_[i];
    }
    for (int i = 0; i < gpu_intensity_frames_leaf_B_.size(); i++) {
        delete gpu_intensity_frames_leaf_B_[i];
    }
    for (int i = 0; i < gpu_edge_frames_B_.size(); i++) {
        delete gpu_edge_frames_B_[i];
    }
    for (int i = 0; i < gpu_dilated_frames_trunk_B_.size(); i++) {
        delete gpu_dilated_frames_trunk_B_[i];
    }
    for (int i = 0; i < gpu_dilated_frames_branch_B_.size(); i++) {
        delete gpu_dilated_frames_branch_B_[i];
    }
    for (int i = 0; i < gpu_dilated_frames_leaf_B_.size(); i++) {
        delete gpu_dilated_frames_leaf_B_[i];
    }

    /*Models*/
    delete gpu_principal_model_;
    for (int i = 0; i < gpu_non_principal_models_.size(); i++) {
        delete gpu_non_principal_models_[i];
    }
};

namespace jta {
bool RunDirectStageGuarded(::DirectOptimizer& opt, QString* errorOut) {
    try {
        bool ok = opt.Run();
        if (!ok) {
            if (errorOut) {
                *errorOut = QStringLiteral("Error optimizing current frame!");
            }
            return false;
        }
        if (errorOut) {
            errorOut->clear();
        }
        return true;
    } catch (const gpu_cost_function::CoordinatorBatchAbort& e) {
        if (errorOut) {
            *errorOut = QString::fromStdString(std::string(e.what()));
        }
        return false;
    } catch (const std::invalid_argument& e) {
        if (errorOut) {
            *errorOut = QString::fromStdString(
                std::string("DirectOptimizer contract violation: ") + e.what());
        }
        return false;
    }
}
}  // namespace jta

std::function<double(const Point6D&)> jta::BuildGpuCostAdapter(
    gpu_cost_function::GPUModel* principal_model,
    Calibration calibration,
    jta_cost_function::CostFunctionManager& stage_manager) {
    /*Plan 008 U9 (Cut B): the shared GPU cost adapter — the pre-Cut-B
     * RunDirectStage injected-cost lambda body (and the oracle twin's body,
     * its monoplane specialization), transcribed verbatim: set the
     * already-physical pose on the principal model (biplane: camera-A-to-B
     * conversion), then score the stage's ACTIVE cost function. Calibration
     * is carried BY VALUE (monoplane default — the future biplane consumer
     * needs no signature change). Three consumers converge on this function:
     * the production runner (OptimizerManager::RunDirectStage), the Tier-2
     * oracle (test/oracle/oracle_test.cpp), and the z-profile probe's cost
     * path. The caller owns `principal_model` and `stage_manager`; both must
     * outlive the returned std::function (the DirectOptimizer runs
     * synchronously inside RunDirectStage, so the reference capture is
     * safe — identical to the pre-Cut-B lambda's capture).*/
    return [principal_model, calibration, &stage_manager](
               const Point6D& physical) mutable -> double {
        Pose pose(
            physical.x,
            physical.y,
            physical.z,
            physical.xa,
            physical.ya,
            physical.za);
        principal_model->SetCurrentPrimaryCameraPose(pose);
        if (calibration.biplane_calibration) {
            /*convert_Pose_A_to_Pose_B is a NON-const Calibration member (it
             * builds local matrices only); `mutable` keeps the by-value
             * capture writable without changing behavior (calibration is
             * never modified).*/
            Point6D physical_B = calibration.convert_Pose_A_to_Pose_B(physical);
            principal_model->SetCurrentSecondaryCameraPose(Pose(
                physical_B.x,
                physical_B.y,
                physical_B.z,
                physical_B.xa,
                physical_B.ya,
                physical_B.za));
        }
        return stage_manager.callActiveCostFunction();
    };
}
