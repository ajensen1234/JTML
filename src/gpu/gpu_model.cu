/*Gpu model header*/
#include "gpu/gpu_model.cuh"

namespace gpu_cost_function {

/*Monoplane constructor*/
GPUModel::GPUModel(std::string model_name, bool principal_model, int width,
                   int height, int device_primary_cam,
                   bool use_backface_cullling_primary_cam, float* triangles,
                   float* normals, int triangle_count,
                   CameraCalibration camera_calibration_primary_cam) {
    /*Initialize Model Names, Type, and Primary*/
    model_name_ = model_name;
    principal_model_ = principal_model;

    /*Monoplane Mode*/
    biplane_mode_ = false;

    /*Initialize Primary Cam Render Engine*/
    primary_cam_render_engine_ = new RenderEngine(
        width, height, device_primary_cam, use_backface_cullling_primary_cam,
        triangles, normals, triangle_count, camera_calibration_primary_cam);
    secondary_cam_render_engine_ = 0;

    /*Check to see if Render Engine Initialized Correctly*/
    if (primary_cam_render_engine_->IsInitializedCorrectly()) {
        initialized_correctly_ = true;
    } else {
        initialized_correctly_ = false;
    }
};

/*Biplane constructor*/
GPUModel::GPUModel(std::string model_name, bool principal_model, int width,
                   int height, int device_primary_cam, int device_secondary_cam,
                   bool use_backface_cullling_primary_cam,
                   bool use_backface_cullling_secondary_cam, float* triangles,
                   float* normals, int triangle_count,
                   CameraCalibration camera_calibration_primary_cam,
                   CameraCalibration camera_calibration_secondary_cam) {
    /*Initialize Model Names, Type, and Primary*/
    model_name_ = model_name;
    principal_model_ = principal_model;

    /*Biplane Mode*/
    biplane_mode_ = true;

    /*Initialize Primary Cam Render Engine*/
    primary_cam_render_engine_ = new RenderEngine(
        width, height, device_primary_cam, use_backface_cullling_primary_cam,
        triangles, normals, triangle_count, camera_calibration_primary_cam);

    /*Initialize Secondary Cam Render Engine*/
    secondary_cam_render_engine_ = new RenderEngine(
        width, height, device_secondary_cam,
        use_backface_cullling_secondary_cam, triangles, normals, triangle_count,
        camera_calibration_secondary_cam);

    /*Check to see if Render Engine Initialized Correctly*/
    if (primary_cam_render_engine_->IsInitializedCorrectly() &&
        secondary_cam_render_engine_->IsInitializedCorrectly()) {
        initialized_correctly_ = true;
    } else {
        initialized_correctly_ = false;
    }
};

/*Default Constructor and Destructor*/
GPUModel::GPUModel() {
    /*The Default constructor should really never be called*/
    primary_cam_render_engine_ = 0;
    secondary_cam_render_engine_ = 0;
    initialized_correctly_ = false;
};

GPUModel::~GPUModel() {
    /*Render engines' destructors should safely run even if they did not
     * initialize correctly*/
    delete primary_cam_render_engine_;
    delete secondary_cam_render_engine_;
};

/*Render to cache function (returns true if worked correctly)
Primary is used in monoplane and biplane, Secondary only used in biplane*/
bool GPUModel::RenderPrimaryCamera(Pose model_pose) {
    if (initialized_correctly_) {
        primary_cam_render_engine_->SetPose(model_pose);
        if (cudaSuccess == primary_cam_render_engine_->Render()) {
            return true;
        }
        return false;
    }
    return false;
};
bool GPUModel::RenderPrimaryCamera_RotationMatrix(
    RotationMatrix model_pose_matrix) {
    if (initialized_correctly_) {
        primary_cam_render_engine_->SetRotationMatrix(model_pose_matrix);
        if (cudaSuccess == primary_cam_render_engine_->Render()) {
            return true;
        }
        return false;
    }
    return false;
}
void GPUModel::RenderPrimaryCameraAndWriteImage(Pose model_pose,
                                                std::string img_name) {
    RenderPrimaryCamera(model_pose);
    primary_cam_render_engine_->WriteImage(img_name);
};

bool GPUModel::RenderSecondaryCamera(Pose model_pose) {
    if (initialized_correctly_ && biplane_mode_) {
        secondary_cam_render_engine_->SetPose(model_pose);
        if (cudaSuccess == secondary_cam_render_engine_->Render()) {
            return true;
        }
        return false;
    }
    return false;
};

/*Render DRR to cache function (returns true if worked correctly)
Primary is used in monoplane and biplane, Secondary only used in biplane*/
bool GPUModel::RenderDRRPrimaryCamera(Pose model_pose, float lower_bound,
                                      float upper_bound) {
    if (initialized_correctly_) {
        primary_cam_render_engine_->SetPose(model_pose);
        if (cudaSuccess ==
            primary_cam_render_engine_->RenderDRR(lower_bound, upper_bound)) {
            return true;
        }
        return false;
    }
    return false;
};

bool GPUModel::RenderDRRSecondaryCamera(Pose model_pose, float lower_bound,
                                        float upper_bound) {
    if (initialized_correctly_ && biplane_mode_) {
        secondary_cam_render_engine_->SetPose(model_pose);
        if (cudaSuccess ==
            secondary_cam_render_engine_->RenderDRR(lower_bound, upper_bound)) {
            return true;
        }
        return false;
    }
    return false;
};

/*Get pointer to rendered image on GPU
Primary is used in monoplane and biplane, Secondary only used in biplane*/
unsigned char* GPUModel::GetPrimaryCameraRenderedImagePointer() {
    return primary_cam_render_engine_->GetRenderOutput()
        ->GetDeviceImagePointer();
};

unsigned char* GPUModel::GetSecondaryCameraRenderedImagePointer() {
    return secondary_cam_render_engine_->GetRenderOutput()
        ->GetDeviceImagePointer();
};

/*Get pointer to rendered image on GPU (GPUImage)
Primary is used in monoplane and biplane, Secondary only used in biplane*/
GPUImage* GPUModel::GetPrimaryCameraRenderedImage() {
    return primary_cam_render_engine_->GetRenderOutput();
};

GPUImage* GPUModel::GetSecondaryCameraRenderedImage() {
    return secondary_cam_render_engine_->GetRenderOutput();
};

cv::Mat GPUModel::GetOpenCVPrimaryRenderedImage() {
    return primary_cam_render_engine_->GetcvMatImage();
}

/*Write Image to File for Primary or Secondary Cameras (bool indicates success).
Include the image extension (e.g. "femur_image.png").*/
bool GPUModel::WritePrimaryCameraRenderedImage(std::string file_name) {
    if (initialized_correctly_) {
        return primary_cam_render_engine_->WriteImage(file_name);
    }
    return false;
};

bool GPUModel::WriteSecondaryCameraRenderedImage(std::string file_name) {
    if (biplane_mode_ && initialized_correctly_) {
        return secondary_cam_render_engine_->WriteImage(file_name);
    }
    return false;
};

/*Set/Get Model Name*/
void GPUModel::SetModelName(std::string model_name) {
    model_name_ = model_name;
};

std::string GPUModel::GetModelName() { return model_name_; };

/*Check if model is the principal one*/
bool GPUModel::IsPrincipalModel() { return principal_model_; };

/*Check if GPU Model was initialized correctly*/
bool GPUModel::IsInitializedCorrectly() { return initialized_correctly_; };

/*Check if running in biplane mode or monoplane mode*/
bool GPUModel::IsBiplaneMode() { return biplane_mode_; };

/*Get/Set Current Pose*/
Pose GPUModel::GetCurrentPrimaryCameraPose() { return current_pose_A_; };

void GPUModel::SetCurrentPrimaryCameraPose(Pose current_pose) {
    current_pose_A_ = current_pose;
};

Pose GPUModel::GetCurrentSecondaryCameraPose() { return current_pose_B_; };

void GPUModel::SetCurrentSecondaryCameraPose(Pose current_pose) {
    current_pose_B_ = current_pose;
};
}  // namespace gpu_cost_function
