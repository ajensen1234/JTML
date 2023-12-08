/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*Location Storage Class is a Matrix for Different Model Poses for Each of the
 * Different Loaded Images (Or Image Pairs if Optimized*/
#ifndef LOCATION_STORAGE_H
#define LOCATION_STORAGE_H

/*Standard*/
#include <vector>

/*Direct Library*/
#include "core/calibration.h"
#include "data_structures_6D.h"

class LocationStorage {
   public:
    LocationStorage(){};
    ~LocationStorage(){};

    /*Add New Model to JTA-GPU So Initialize ALl Loaded Frames with
    Default Pose (0,0,-.25*principal_distance / pixel_pitch,0,0,0)*/
    void LoadNewModel(double principal_distance, double pixel_pitch);
    void LoadNewModel(Calibration calibration);
    /*Add New Frame to JTA-GPU So Initialize ALl Loaded Models with
    Default Pose (0,0,-.25*principal_distance / pixel_pitch,0,0,0) for that
    Frame*/
    void LoadNewFrame();

    /*Acces A Pose from Matrix*/
    Point6D GetPose(int frame_index, int model_index);

    /*Store a Pose to Matrix*/
    void SavePose(int frame_index, int model_index, Point6D model_pose);

    /*Get Frame Storage Size*/
    int GetFrameCount();

    /*Get Model Storage Size (-1 if Inconsistent Sizes)*/
    int GetModelCount();

   private:
    /*List of Model Location Points for Each Frame
    (So The Outside Vector is the length of the loaded frames
    and the inside vector is the length of the loaded models)*/
    std::vector<std::vector<Point6D>> location_storage_matrix_;

    /*No Image Model Location Points
    (This stores location points for models when there are no images
    loaded, this is the size of the models loaded)*/
    std::vector<Point6D> no_image_location_storage_vector_;
};

#endif /* LOCATION_STORAGE_H */
