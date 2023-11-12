#pragma once

/*Study*/
#include "Study.h"

/*Standard Library Files*/
#include <string>
#include <vector>

/*Cost Function Tools Library*/
#include "gpu_dilated_frame.cuh"
#include "gpu_edge_frame.cuh"
#include "gpu_frame.cuh"
#include "gpu_image.cuh"
#include "gpu_image_functions.cuh"
#include "gpu_intensity_frame.cuh"
#include "gpu_metrics.cuh"
#include "gpu_model.cuh"
#include "render_engine.cuh"

/*Basic Linear Algebra Structures*/
#include "BasicLinearAlgebraStructures.h"

/*An Image Info Struct which Contains :
- Corresponding Study class
- File Paths for the Image and It's Silhouette Label Images for all of it's
models
- Poses for all models
- Functions to append relevant information to text file given a text file path
(one for each model)*/

struct ImageInfo {
    /*Constructor*/
    ImageInfo(Study study, std::string image_path,
              vector<std::string> model_types,
              vector<std::string> label_img_paths,
              vector<gpu_cost_function::Pose> pose_img_models,
              vector<vector<basic_la::XYPoint>> norm_KP_points_list);

    /*Append Information to Text File*/
    void AppendInformation(
        std::string text_file_path,
        int model_type_index);  // int model_type_index is index of model type
                                // in study

    /*Study*/
    Study study_;

    /*Paths*/
    std::string image_path_;
    vector<std::string>
        label_img_paths_;  // For each model in the image (stored same order as
                           // models in study)

    /*Model Types Present*/
    vector<std::string> model_types_;

    /*Poses for each model in the image (stored same order as models in study)*/
    vector<gpu_cost_function::Pose> pose_img_models_;

    /*Vector of Vector of Normalized KPs (empty vector stored if no .kp file for
     * model or not generating KP data) for each model (stored same order as
     * models in study)*/
    vector<vector<basic_la::XYPoint>> norm_KP_points_list_;
};
