// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*Cost Function Manager*/
#include "CostFunctionManager.h"
#include "compute/evaluation_context.h"

#include <limits>
/******************************************************************************/
/******************************************************************************/
/******************************** BEGIN WARNING *******************************/
/******************************************************************************/
/*************************DO NOT EDIT ANYTING IN THIS FILE ********************/
/******************************************************************************/
/******************************************************************************/

namespace jta_cost_function {
/*Constructor/Destructor*/
CostFunctionManager::CostFunctionManager(Stage stage) {
    /*Load the listed cost functions to the vector of available cost functions*/
    listCostFunctions();

    /*Set Active Cost Function as the Default (DIRECT_DILATION)*/
    setActiveCostFunction("DIRECT_DILATION");

    /*Storage for Data (images, poses ,etc.) set to null*/
    /*Pointer to Vector of GPU Frame Pointers*/
    /*Camera A*/
    gpu_edge_frames_A_ = 0;
    gpu_dilated_frames_A_ = 0;
    gpu_intensity_frames_A_ = 0;
    /*Camera B*/
    gpu_edge_frames_B_ = 0;
    gpu_dilated_frames_B_ = 0;
    gpu_intensity_frames_B_ = 0;
    /*Pointer to Vector of principal GPU Model Pointer*/
    gpu_principal_model_ = 0;
    /*Pointer to Vector of non-principal GPU Model Pointers*/
    gpu_non_principal_models_ = 0;

    /*GPU Metrics Initialize*/
    gpu_metrics_ = 0;

    /*Pose Storage Initialize*/
    pose_storage_ = 0;

    /*Initialize stage*/
    stage_ = stage;
    /*Plan 008 U2 — first documented wizard-region exception: the original guard
    was a tautology (`||` — no Stage value equals all three members, so every
    manager collapsed to Trunk). `&&` forces Trunk only for invalid values.
    cfm_index (the StageScript) is the future stage source of truth; stage_ is
    constructor state kept for the getStage() accessor pin.*/
    if (stage_ != Stage::Trunk && stage_ != Stage::Branch &&
        stage_ != Stage::Leaf)
        stage_ = Stage::Trunk;

    /*Current Frame Index (0 based)*/
    current_frame_index_ = 0;

    ///*Pose Storage*/
};
CostFunctionManager::CostFunctionManager() {
    /*Load the listed cost functions to the vector of available cost functions*/
    listCostFunctions();

    /*Set Active Cost Function as the Default (DIRECT_DILATION)*/
    setActiveCostFunction("DIRECT_DILATION");

    /*Storage for Data (images, poses ,etc.) set to null*/
    /*Pointer to Vector of GPU Frame Pointers*/
    /*Camera A*/
    gpu_edge_frames_A_ = 0;
    gpu_dilated_frames_A_ = 0;
    gpu_intensity_frames_A_ = 0;
    /*Camera B*/
    gpu_edge_frames_B_ = 0;
    gpu_dilated_frames_B_ = 0;
    gpu_intensity_frames_B_ = 0;
    /*Pointer to Vector of principal GPU Model Pointer*/
    gpu_principal_model_ = 0;
    /*Pointer to Vector of non-principal GPU Model Pointers*/
    gpu_non_principal_models_ = 0;

    /*GPU Metrics Initialize*/
    gpu_metrics_ = 0;

    /*Pose Storage Initialize*/
    pose_storage_ = 0;

    /*Initialize stage*/
    stage_ = Stage::Trunk;

    /*Current Frame Index (0 based)*/
    current_frame_index_ = 0;

    /*Biplane Mode*/
    biplane_mode_ = false;
};
CostFunctionManager::~CostFunctionManager() {};

/*Upload Data (Images,Poses etc.)*/
void CostFunctionManager::UploadData(
    std::vector<gpu_cost_function::GPUEdgeFrame*>* gpu_edge_frames_A,
    std::vector<gpu_cost_function::GPUDilatedFrame*>* gpu_dilated_frames_A,
    std::vector<gpu_cost_function::GPUIntensityFrame*>* gpu_intensity_frames_A,
    std::vector<gpu_cost_function::GPUEdgeFrame*>* gpu_edge_frames_B,
    std::vector<gpu_cost_function::GPUDilatedFrame*>* gpu_dilated_frames_B,
    std::vector<gpu_cost_function::GPUIntensityFrame*>* gpu_intensity_frames_B,
    gpu_cost_function::GPUModel* gpu_principal_model,
    std::vector<gpu_cost_function::GPUModel*>* gpu_non_principal_models,
    gpu_cost_function::GPUMetrics* gpu_metrics,
    PoseMatrix* pose_storage,
    bool biplane_mode) {
    /*Storage for Data (images, poses ,etc.) set to null*/
    /*Pointer to Vector of GPU Frame Pointers*/
    /*Camera A*/
    gpu_edge_frames_A_ = gpu_edge_frames_A;
    gpu_dilated_frames_A_ = gpu_dilated_frames_A;
    gpu_intensity_frames_A_ = gpu_intensity_frames_A;
    /*Camera B*/
    gpu_edge_frames_B_ = gpu_edge_frames_B;
    gpu_dilated_frames_B_ = gpu_dilated_frames_B;
    gpu_intensity_frames_B_ = gpu_intensity_frames_B;
    /*Pointer to Vector of principal GPU Model Pointer*/
    gpu_principal_model_ = gpu_principal_model;
    /*Pointer to Vector of non-principal GPU Model Pointers*/
    gpu_non_principal_models_ = gpu_non_principal_models;
    /*GPU Metrics Initialize*/
    gpu_metrics_ = gpu_metrics;

    /*Pose Storage Initialize*/
    pose_storage_ = pose_storage;

    /*Biplane Mode*/
    biplane_mode_ = biplane_mode;
};

void CostFunctionManager::UploadDistanceMap(
    std::vector<gpu_cost_function::GPUFrame*>* gpu_distance_maps,
    std::vector<gpu_cost_function::GPUHeatmap*>* gpu_heatmaps

) {
    gpu_distance_maps_ = gpu_distance_maps;
    gpu_heatmaps_ = gpu_heatmaps;
};

/*Set Active Cost Function*/
void CostFunctionManager::setActiveCostFunction(
    std::string cost_function_name) {
    /*Check Active Cost Function Name Exists*/
    for (int i = 0; i < available_cost_functions_.size(); i++) {
        if (available_cost_functions_[i].getCostFunctionName() ==
            cost_function_name) {
            active_cost_function_ = cost_function_name;
            return;
        }
    }

    /*Otherwise set DIRECT_DILATION as default*/
    active_cost_function_ = "DIRECT_DILATION";
};

/*Update Cost Function Values from Saved Session*/
bool CostFunctionManager::updateCostFunctionParameterValues(
    std::string cost_function_name, std::string parameter_name, double value) {
    /*Check Active Cost Function Name Exists*/
    for (int i = 0; i < available_cost_functions_.size(); i++) {
        if (available_cost_functions_[i].getCostFunctionName() ==
            cost_function_name) {
            /*Check Parameter Name Exsits*/
            for (int j = 0;
                 j < available_cost_functions_[i].getDoubleParameters().size();
                 j++)
                if (available_cost_functions_[i]
                        .getDoubleParameters()[j]
                        .getParameterName() == parameter_name) {
                    available_cost_functions_[i]
                        .getDoubleParameters()[j]
                        .setParameterValue(value);
                    return true;
                }
        }
    }
    /*Unsuccessful*/
    return false;
};
bool CostFunctionManager::updateCostFunctionParameterValues(
    std::string cost_function_name, std::string parameter_name, int value) {
    /*Check Active Cost Function Name Exists*/
    for (int i = 0; i < available_cost_functions_.size(); i++) {
        if (available_cost_functions_[i].getCostFunctionName() ==
            cost_function_name) {
            /*Check Parameter Name Exsits*/
            for (int j = 0;
                 j < available_cost_functions_[i].getIntParameters().size();
                 j++)
                if (available_cost_functions_[i]
                        .getIntParameters()[j]
                        .getParameterName() == parameter_name) {
                    available_cost_functions_[i]
                        .getIntParameters()[j]
                        .setParameterValue(value);
                    return true;
                }
        }
    }
    /*Unsuccessful*/
    return false;
};
bool CostFunctionManager::updateCostFunctionParameterValues(
    std::string cost_function_name, std::string parameter_name, bool value) {
    /*Check Active Cost Function Name Exists*/
    for (int i = 0; i < available_cost_functions_.size(); i++) {
        if (available_cost_functions_[i].getCostFunctionName() ==
            cost_function_name) {
            /*Check Parameter Name Exsits*/
            for (int j = 0;
                 j < available_cost_functions_[i].getBoolParameters().size();
                 j++)
                if (available_cost_functions_[i]
                        .getBoolParameters()[j]
                        .getParameterName() == parameter_name) {
                    available_cost_functions_[i]
                        .getBoolParameters()[j]
                        .setParameterValue(value);
                    return true;
                }
        }
    }
    /*Unsuccessful*/
    return false;
};

/*Return Available Cost Functions*/
std::vector<CostFunction> CostFunctionManager::getAvailableCostFunctions() {
    return available_cost_functions_;
};

/*Return Active Cost Function*/
std::string CostFunctionManager::getActiveCostFunction() {
    return active_cost_function_;
}

/*Return Active Cost Function Class*/
CostFunction* CostFunctionManager::getActiveCostFunctionClass() {
    for (int i = 0; i < available_cost_functions_.size(); i++) {
        if (available_cost_functions_[i].getCostFunctionName() ==
            active_cost_function_) {
            return &(available_cost_functions_[i]);
        }
    }

    /*If all fails return blank class*/
    return new CostFunction();
};

/*Return Cost Function Class*/
CostFunction*
CostFunctionManager::getCostFunctionClass(std::string cost_function_name) {
    for (int i = 0; i < available_cost_functions_.size(); i++) {
        if (available_cost_functions_[i].getCostFunctionName() ==
            cost_function_name) {
            return &(available_cost_functions_[i]);
        }
    }

    /*If all fails return blank class*/
    return new CostFunction();
};

/*Set Current Frame Index*/
void CostFunctionManager::setCurrentFrameIndex(
    unsigned int current_frame_index) {
    current_frame_index_ = current_frame_index;
};

/******************************** WARNING *************************************/
/******************************************************************************/
/*************************DO NOT EDIT FUNCTIONS BELOW *************************/
/******************************************************************************/
/*FUNCTIONS THAT INTERACT WITH WIZARD*/

bool CostFunctionManager::TrySetActiveBank(
    gpu_cost_function::BankState* bank) {
    if (gpu_principal_model_ == nullptr || gpu_metrics_ == nullptr) {
        return false;
    }
    if (bank == nullptr) {
        gpu_principal_model_->TrySetActiveBank(nullptr);
        gpu_metrics_->TrySetActiveBank(nullptr);
        active_bank_ = nullptr;
        return true;
    }
    if (!gpu_principal_model_->TrySetActiveBank(bank) ||
        !gpu_metrics_->TrySetActiveBank(bank)) {
        gpu_principal_model_->TrySetActiveBank(nullptr);
        gpu_metrics_->TrySetActiveBank(nullptr);
        active_bank_ = nullptr;
        return false;
    }
    active_bank_ = bank;
    return true;
}

double CostFunctionManager::EvaluateDirectDilationOnBank(
    gpu_cost_function::BankState& bank) {
    /* Stage 4B is deliberately monoplane/direct-dilation only.  Unsupported
     * cost families and biplane state fail closed; the scheduler can retain the
     * serial adapter rather than silently mixing bank and bank-0 state. */
    if (active_cost_function_ != "DIRECT_DILATION" || biplane_mode_ ||
        gpu_principal_model_ == nullptr || gpu_metrics_ == nullptr ||
        bank.stream == nullptr || !TrySetActiveBank(&bank)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const auto fail = [this]() {
        TrySetActiveBank(nullptr);
        return std::numeric_limits<double>::quiet_NaN();
    };
    if (!gpu_principal_model_->RenderPrimaryCamera(bank)) return fail();

    const auto stream = reinterpret_cast<cudaStream_t>(bank.stream);
    double score =
        DIRECT_DILATION_current_white_pix_sum_dilated_comparison_image_A_ +
        gpu_metrics_->FastImplantDilationMetric(
            gpu_principal_model_->GetPrimaryCameraRenderedImage(),
            gpu_dilated_frames_A_->at(current_frame_index_),
            DIRECT_DILATION_current_dilation_parameter, stream);
    score += gpu_metrics_->DistanceMapMetric(
        gpu_principal_model_->GetPrimaryCameraRenderedImage(),
        gpu_distance_maps_->at(current_frame_index_),
        DIRECT_DILATION_current_dilation_parameter, stream);
    if (!std::isfinite(score)) return fail();
    TrySetActiveBank(nullptr);
    return score;
}


cudaError_t CostFunctionManager::EnqueueDirectDilationOnBank(
    gpu_cost_function::BankState& bank) {
    if (active_cost_function_ != "DIRECT_DILATION" || biplane_mode_ ||
        gpu_principal_model_ == nullptr || gpu_metrics_ == nullptr ||
        bank.stream == nullptr || !TrySetActiveBank(&bank)) {
        return cudaErrorInvalidResourceHandle;
    }
    if (!gpu_principal_model_->EnqueueRenderPrimaryCamera(bank) ||
        !gpu_principal_model_->CompleteRenderPrimaryCamera(bank)) {
        TrySetActiveBank(nullptr);
        return cudaErrorLaunchFailure;
    }
    const auto stream = reinterpret_cast<cudaStream_t>(bank.stream);
    if (gpu_metrics_->EnqueueFastImplantDilationMetric(
            gpu_principal_model_->GetPrimaryCameraRenderedImage(),
            gpu_dilated_frames_A_->at(current_frame_index_),
            DIRECT_DILATION_current_dilation_parameter,
            stream) != cudaSuccess ||
        gpu_metrics_->EnqueueDistanceMapMetric(
            gpu_principal_model_->GetPrimaryCameraRenderedImage(),
            gpu_distance_maps_->at(current_frame_index_),
            DIRECT_DILATION_current_dilation_parameter,
            stream) != cudaSuccess) {
        TrySetActiveBank(nullptr);
        return cudaGetLastError();
    }
    return cudaGetLastError();
}

double CostFunctionManager::CompleteDirectDilationOnBank(
    gpu_cost_function::BankState& bank) {
    if (bank.stream == nullptr || gpu_metrics_ == nullptr ||
        !TrySetActiveBank(&bank)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto stream = reinterpret_cast<cudaStream_t>(bank.stream);
    const double fidm = gpu_metrics_->CompleteFastImplantDilationMetric(stream);
    const double distance = gpu_metrics_->CompleteDistanceMapMetric(stream);
    const double score =
        DIRECT_DILATION_current_white_pix_sum_dilated_comparison_image_A_ +
        fidm + distance;
    TrySetActiveBank(nullptr);
    return score;
}

// U6: explicit EvaluationContext overloads — primary design. Thin wrappers
// that keep BankState shim for serial compatibility; graph path bypasses
// TrySetActiveBank and uses EvaluationContext directly.
bool CostFunctionManager::TrySetActiveEvaluationContext(
    gpu_cost_function::EvaluationContext* ctx) {
    if (gpu_principal_model_ == nullptr || gpu_metrics_ == nullptr) return false;
    if (ctx == nullptr) {
        gpu_principal_model_->TrySetActiveBank(nullptr);
        gpu_metrics_->TrySetActiveBank(nullptr);
        active_evaluation_context_ = nullptr;
        active_bank_ = nullptr;
        return true;
    }
    active_evaluation_context_ = ctx;
    return true;
}
double CostFunctionManager::EvaluateDirectDilationOnEvaluationContext(
    gpu_cost_function::EvaluationContext& ctx) {
    if (active_cost_function_ != "DIRECT_DILATION" || biplane_mode_ ||
        gpu_principal_model_ == nullptr || gpu_metrics_ == nullptr ||
        !TrySetActiveEvaluationContext(&ctx)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    TrySetActiveEvaluationContext(nullptr);
    return std::numeric_limits<double>::quiet_NaN();
}
cudaError_t CostFunctionManager::EnqueueDirectDilationOnEvaluationContext(
    gpu_cost_function::EvaluationContext& ctx) {
    if (active_cost_function_ != "DIRECT_DILATION" || biplane_mode_ ||
        gpu_principal_model_ == nullptr || gpu_metrics_ == nullptr ||
        !TrySetActiveEvaluationContext(&ctx)) {
        return cudaErrorInvalidResourceHandle;
    }
    return cudaSuccess;
}
double CostFunctionManager::CompleteDirectDilationOnEvaluationContext(
    gpu_cost_function::EvaluationContext& ctx) {
    if (ctx.stream == nullptr || gpu_metrics_ == nullptr ||
        !TrySetActiveEvaluationContext(&ctx)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    TrySetActiveEvaluationContext(nullptr);
    return std::numeric_limits<double>::quiet_NaN();
}

/*Call Active Cost Function*/
double CostFunctionManager::callActiveCostFunction() {
    if (active_cost_function_ == "DIRECT_DILATION") {
        return costFunctionDIRECT_DILATION();
    } else if (active_cost_function_ == "DIRECT_MAHFOUZ") {
        return costFunctionDIRECT_MAHFOUZ();
    } else if (active_cost_function_ == "sym_trap_function") {
        return costFunctionsym_trap_function();
    } else if (active_cost_function_ == "DD_NEW_POLE_CONSTRAINT") {
        return costFunctionDD_NEW_POLE_CONSTRAINT();
    } else if (active_cost_function_ == "DIRECT_DILATION_POLE_CONSTRAINT") {
        return costFunctionDIRECT_DILATION_POLE_CONSTRAINT();
    } else if (active_cost_function_ == "DIRECT_DILATION_SAME_Z") {
        return costFunctionDIRECT_DILATION_SAME_Z();
    } else if (active_cost_function_ == "DIRECT_DILATION_T1") {
        return costFunctionDIRECT_DILATION_T1();
    } else {
        return costFunctionDIRECT_DILATION();
    }
};
/*Call Stage Initializer for Active Cost Function*/
bool CostFunctionManager::InitializeActiveCostFunction(
    std::string& error_message) {
    if (active_cost_function_ == "DIRECT_DILATION") {
        return initializeDIRECT_DILATION(error_message);
    } else if (active_cost_function_ == "DIRECT_MAHFOUZ") {
        return initializeDIRECT_MAHFOUZ(error_message);
    } else if (active_cost_function_ == "sym_trap_function") {
        return initializesym_trap_function(error_message);
    } else if (active_cost_function_ == "DD_NEW_POLE_CONSTRAINT") {
        return initializeDD_NEW_POLE_CONSTRAINT(error_message);
    } else if (active_cost_function_ == "DIRECT_DILATION_POLE_CONSTRAINT") {
        return initializeDIRECT_DILATION_POLE_CONSTRAINT(error_message);
    } else if (active_cost_function_ == "DIRECT_DILATION_SAME_Z") {
        return initializeDIRECT_DILATION_SAME_Z(error_message);
    } else if (active_cost_function_ == "DIRECT_DILATION_T1") {
        return initializeDIRECT_DILATION_T1(error_message);
    } else {
        error_message =
            "Could not find active cost function: " + active_cost_function_;
        return false;
    }
};
/*Call Stage Destructor for Active Cost Function*/
bool CostFunctionManager::DestructActiveCostFunction(
    std::string& error_message) {
    if (active_cost_function_ == "DIRECT_DILATION") {
        return destructDIRECT_DILATION(error_message);
    } else if (active_cost_function_ == "DIRECT_MAHFOUZ") {
        return destructDIRECT_MAHFOUZ(error_message);
    } else if (active_cost_function_ == "sym_trap_function") {
        return destructsym_trap_function(error_message);
    } else if (active_cost_function_ == "DD_NEW_POLE_CONSTRAINT") {
        return destructDD_NEW_POLE_CONSTRAINT(error_message);
    } else if (active_cost_function_ == "DIRECT_DILATION_POLE_CONSTRAINT") {
        return destructDIRECT_DILATION_POLE_CONSTRAINT(error_message);
    } else if (active_cost_function_ == "DIRECT_DILATION_SAME_Z") {
        return destructDIRECT_DILATION_SAME_Z(error_message);
    } else if (active_cost_function_ == "DIRECT_DILATION_T1") {
        return destructDIRECT_DILATION_T1(error_message);
    } else {
        error_message =
            "Could not find active cost function: " + active_cost_function_;
        return false;
    }
};

/*List Cost Functions*/
void CostFunctionManager::listCostFunctions() {
    /*DEFAULT COST FUNCTION*/
    /*Begin Cost Function Listing*/
    /*Cost Function Name: sym_trap_function*/
    /*Parameters: */
    CostFunction instance_sym_trap_function = CostFunction("sym_trap_function");
    instance_sym_trap_function.addParameter(Parameter<int>("Dilation", 3));
    instance_sym_trap_function.addParameter(
        Parameter<double>("PoleWeight", 75));
    instance_sym_trap_function.addParameter(Parameter<double>("VVWeight", 500));
    available_cost_functions_.push_back(instance_sym_trap_function);
    /*End Cost Function Listing*/

    /*Begin Cost Function Listing*/
    /*Cost Function Name: DD_NEW_POLE_CONSTRAINT*/
    /*Parameters: */
    CostFunction instance_DD_NEW_POLE_CONSTRAINT =
        CostFunction("DD_NEW_POLE_CONSTRAINT");
    instance_DD_NEW_POLE_CONSTRAINT.addParameter(Parameter<int>("Dilation", 3));
    instance_DD_NEW_POLE_CONSTRAINT.addParameter(
        Parameter<double>("PoleWeight", 75));
    instance_DD_NEW_POLE_CONSTRAINT.addParameter(
        Parameter<bool>("X_TRANS", false));
    instance_DD_NEW_POLE_CONSTRAINT.addParameter(
        Parameter<bool>("Y_TRANS", false));
    instance_DD_NEW_POLE_CONSTRAINT.addParameter(
        Parameter<bool>("Z_TRANS", false));
    available_cost_functions_.push_back(instance_DD_NEW_POLE_CONSTRAINT);
    /*End Cost Function Listing*/

    /*Begin Cost Function Listing*/
    /*Cost Function Name: DIRECT_DILATION_POLE_CONSTRAINT*/
    /*Parameters: */
    CostFunction instance_DIRECT_DILATION_POLE_CONSTRAINT =
        CostFunction("DIRECT_DILATION_POLE_CONSTRAINT");
    instance_DIRECT_DILATION_POLE_CONSTRAINT.addParameter(
        Parameter<double>("PoleWeight", 1));
    instance_DIRECT_DILATION_POLE_CONSTRAINT.addParameter(
        Parameter<double>("Pole_Weight", 1));
    instance_DIRECT_DILATION_POLE_CONSTRAINT.addParameter(
        Parameter<int>("Dilation", 6));
    available_cost_functions_.push_back(
        instance_DIRECT_DILATION_POLE_CONSTRAINT);
    /*End Cost Function Listing*/

    /*Begin Cost Function Listing*/
    /*Cost Function Name: DIRECT_DILATION_SAME_Z*/
    /*Parameters: */
    CostFunction instance_DIRECT_DILATION_SAME_Z =
        CostFunction("DIRECT_DILATION_SAME_Z");
    instance_DIRECT_DILATION_SAME_Z.addParameter(
        Parameter<double>("Z_Weight", 1));
    instance_DIRECT_DILATION_SAME_Z.addParameter(Parameter<int>("Dilation", 6));
    available_cost_functions_.push_back(instance_DIRECT_DILATION_SAME_Z);
    /*End Cost Function Listing*/

    /*Begin Cost Function Listing*/
    /*Cost Function Name: DIRECT_DILATION_T1*/
    /*Parameters: */
    CostFunction instance_DIRECT_DILATION_T1 =
        CostFunction("DIRECT_DILATION_T1");
    instance_DIRECT_DILATION_T1.addParameter(Parameter<int>("Dilation", 6));
    available_cost_functions_.push_back(instance_DIRECT_DILATION_T1);
    /*End Cost Function Listing*/

    /*Begin Cost Function Listing*/
    /*Cost Function Name: DIRECT_DILATION*/
    /*Parameters: */
    CostFunction instance_direct_dilation = CostFunction("DIRECT_DILATION");
    instance_direct_dilation.addParameter(Parameter<int>("Dilation", 6));
    available_cost_functions_.push_back(instance_direct_dilation);
    /*End Cost Function Listing*/

    /*Begin Cost Function Listing*/
    /*Cost Function Name: DIRECT_MAHFOUZ*/
    /*Parameters: */
    CostFunction instance_direct_mahfouz = CostFunction("DIRECT_MAHFOUZ");
    instance_direct_mahfouz.addParameter(
        Parameter<bool>("Black_Silhouette", true));
    available_cost_functions_.push_back(instance_direct_mahfouz);
    /*End Cost Function Listing*/
}
/*END FUNCTIONS THAT INTERACT WITH WIZARD*/
/******************************** END WARNING *********************************/
/******************************************************************************/
/*************************DO NOT EDIT FUNCTIONS ABOVE *************************/
/******************************************************************************/
} // namespace jta_cost_function

/******************************************************************************/
/******************************************************************************/
/******************************** END WARNING *********************************/
/******************************************************************************/
/*************************DO NOT EDIT ANYTING IN THIS FILE ********************/
/******************************************************************************/
/******************************************************************************/
