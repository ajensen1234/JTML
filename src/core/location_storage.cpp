/*Location Storage*/
#include "core/location_storage.h"

/*Add New Model to JTA-GPU So Initialize ALl Loaded Frames with
Default Pose (0,0,-.25*principal_distance / pixel_pitch,0,0,0)*/
void LocationStorage::LoadNewModel(double principal_distance,
                                   double pixel_pitch) {
    for (int i = 0; i < location_storage_matrix_.size(); i++) {
        location_storage_matrix_[i].push_back(
            Point6D(0, 0, -.25 * principal_distance / pixel_pitch, 0, 0, 0));
    }
    no_image_location_storage_vector_.push_back(
        Point6D(0, 0, -.25 * principal_distance / pixel_pitch, 0, 0, 0));
}
void LocationStorage::LoadNewModel(Calibration calibration) {
    double z_pos;
    if (calibration.type_ == "Denver") {
        z_pos = 0.25 * calibration.camera_A_principal_.fx();
    } else if (calibration.type_ == "UF") {
        z_pos = -0.25 * calibration.camera_A_principal_.principal_distance_ /
                calibration.camera_A_principal_.pixel_pitch_;
    }
    for (int i = 0; i < location_storage_matrix_.size(); i++) {
        location_storage_matrix_[i].push_back(Point6D(0, 0, z_pos, 0, 0, 0));
    }
    no_image_location_storage_vector_.push_back(Point6D(0, 0, z_pos, 0, 0, 0));
}

/*Add New Frame to JTA-GPU So Initialize ALl Loaded Models with
Default Pose (0,0,-.25*principal_distance / pixel_pitch,0,0,0) for that Frame*/
void LocationStorage::LoadNewFrame() {
    location_storage_matrix_.push_back(no_image_location_storage_vector_);
}

/*Acces A Pose from Matrix*/
Point6D LocationStorage::GetPose(int frame_index, int model_index) {
    /*check in range*/
    if (frame_index >= 0 && frame_index < location_storage_matrix_.size()) {
        if (model_index >= 0 &&
            model_index < location_storage_matrix_[frame_index].size()) {
            return location_storage_matrix_[frame_index][model_index];
        }
        return Point6D(0, 0, 0, 0, 0, 0);
    }
    if (model_index >= 0 &&
        model_index < no_image_location_storage_vector_.size()) {
        return no_image_location_storage_vector_[model_index];
    }
    return Point6D(0, 0, 0, 0, 0, 0); /*SHOULD NEVER HAPPEN*/
}

/*Store a Pose to Matrix*/
void LocationStorage::SavePose(int frame_index, int model_index,
                               Point6D model_pose) {
    /*check in range*/
    if (frame_index >= 0 && frame_index < location_storage_matrix_.size()) {
        if (model_index >= 0 &&
            model_index < location_storage_matrix_[frame_index].size()) {
            location_storage_matrix_[frame_index][model_index] = model_pose;
        }
    }
}

/*Get Frame Storage Size*/
int LocationStorage::GetFrameCount() {
    return location_storage_matrix_.size();
};

/*Get Model Storage Size (-1 if Inconsistent Sizes)*/
int LocationStorage::GetModelCount() {
    int size = 0;
    if (location_storage_matrix_.size() > 0)
        size = location_storage_matrix_[0].size();
    for (int i = 0; i < location_storage_matrix_.size(); i++) {
        if (location_storage_matrix_[i].size() != size) return -1;
    }
    return size;
};
