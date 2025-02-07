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
    gpu_metrics_ = 0;

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