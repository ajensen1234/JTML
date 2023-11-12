/*Cost Function Manager*/
#include "CostFunctionManager.h"
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
    if (stage_ != Stage::Trunk || stage_ != Stage::Branch ||
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
CostFunctionManager::~CostFunctionManager(){};

/*Upload Data (Images,Poses etc.)*/
void CostFunctionManager::UploadData(
    std::vector<gpu_cost_function::GPUEdgeFrame *> *gpu_edge_frames_A,
    std::vector<gpu_cost_function::GPUDilatedFrame *> *gpu_dilated_frames_A,
    std::vector<gpu_cost_function::GPUIntensityFrame *> *gpu_intensity_frames_A,
    std::vector<gpu_cost_function::GPUEdgeFrame *> *gpu_edge_frames_B,
    std::vector<gpu_cost_function::GPUDilatedFrame *> *gpu_dilated_frames_B,
    std::vector<gpu_cost_function::GPUIntensityFrame *> *gpu_intensity_frames_B,
    gpu_cost_function::GPUModel *gpu_principal_model,
    std::vector<gpu_cost_function::GPUModel *> *gpu_non_principal_models,
    gpu_cost_function::GPUMetrics *gpu_metrics, PoseMatrix *pose_storage,
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
    std::vector<gpu_cost_function::GPUFrame *> *gpu_distance_maps) {
    gpu_distance_maps_ = gpu_distance_maps;
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
CostFunction *CostFunctionManager::getActiveCostFunctionClass() {
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
CostFunction *CostFunctionManager::getCostFunctionClass(
    std::string cost_function_name) {
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
    std::string &error_message) {
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
    std::string &error_message) {
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
}  // namespace jta_cost_function

/******************************************************************************/
/******************************************************************************/
/******************************** END WARNING *********************************/
/******************************************************************************/
/*************************DO NOT EDIT ANYTING IN THIS FILE ********************/
/******************************************************************************/
/******************************************************************************/
