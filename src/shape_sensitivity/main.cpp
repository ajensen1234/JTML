#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "core/calibration.h"
#include "core/model.h"
#include "descriptors.h"
#include "gpu/gpu_model.cuh"
#include "matrix_vector_utils.h"
int main() {
    auto mod_name = std::string("sca");
    std::ofstream iartd_file("iartd-" + mod_name + ".csv");
    std::ofstream hu_file("hu_moments-" + mod_name + ".csv");
    if (!iartd_file.is_open() || !hu_file.is_open()) {
        std::cerr << "Error Opening results files" << std::endl;
        return 0;
    }
    // Creating a pointer do our CUDA class for image descriptors
    bool speed_test = false;
    int speed_test_iter = 500;
    auto img_desc_gpu = new img_desc(1024, 1024, 0);
    // filepath to stl model
    auto mod_fp = std::string(
        "/media/ajensen123@ad.ufl.edu/Andrew's External "
        "SSD/Data/Datasets_TSA/Nagoya_Organized/Patient_M01/Session_1/"
        "Movement_1/sca.stl");
    std::ifstream file(mod_fp);
    if (!file) {
        return 1;
    } else {
        std::cout << "File exists!~" << std::endl;
    }

    // CPU version of our model
    auto primary_model_ = Model(mod_fp, mod_name, mod_name);
    int width = 1024;
    int height = 1024;
    int cuda_device_id = 0;
    //(weird old calibration params)
    auto cam_cal = CameraCalibration(1000, 0, 0, 0.32);
    auto calibration_ = Calibration(cam_cal);

    // GPU version of the model
    auto gpu_principal_model_ = new gpu_cost_function::GPUModel(
        primary_model_.model_name_, true, width, height, cuda_device_id, true,
        &primary_model_.triangle_vertices_[0],
        &primary_model_.triangle_normals_[0],
        primary_model_.triangle_vertices_.size() / 9,
        calibration_.camera_A_principal_);
    if (!gpu_principal_model_->IsInitializedCorrectly()) {
        delete gpu_principal_model_;
        gpu_principal_model_ = 0;
    } else {
        std::cout << "GPU Initialized correctly!" << std::endl;
    }
    // Setting the rotation ranges for x and y
    float x_rot_range = 30;
    float y_rot_range = 30;
    float z_rot_range = 30;
    float step = 5;
    float z_step = 5;
    // Total number of instances for outputting progress bar
    int tot = (((2 * x_rot_range) / step) + 1) *
              (((2 * y_rot_range) / step) + 1) *
              (((2 * z_rot_range) / z_step) + 1);

    // vectors for holding outputs from algorithm
    std::vector<float> iartd, hu;
    int iter = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int zr = -1 * z_rot_range; zr <= z_rot_range; zr += z_step) {
        for (int xr = -1 * x_rot_range; xr <= x_rot_range; xr += step) {
            for (int yr = -1 * y_rot_range; yr <= y_rot_range; yr += step) {
                // Initializing starting pose
                gpu_cost_function::Pose curr_pose(0, 0, -850, xr, yr, zr);

                // Rendering the camera at the current pose
                gpu_principal_model_->RenderPrimaryCamera(
                    gpu_cost_function::Pose(0, 0, -850, xr, yr, zr));
                // Calculating angular radial transform moments
                iartd = calculateIARTD(
                    img_desc_gpu,
                    gpu_principal_model_->GetPrimaryCameraRenderedImage());
                hu = img_desc_gpu->hu_moments(
                    gpu_principal_model_->GetPrimaryCameraRenderedImage());

                // Doing +/- x-axis rotation shifts
                RotationMatrix x_pos_nudge = rotation_nudge(curr_pose, 1, "x");
                RotationMatrix x_neg_nudge = rotation_nudge(curr_pose, -1, "x");
                RotationMatrix y_pos_nudge = rotation_nudge(curr_pose, 1, "y");
                RotationMatrix y_neg_nudge = rotation_nudge(curr_pose, -1, "y");
                RotationMatrix z_pos_nudge = rotation_nudge(curr_pose, 1, "z");
                RotationMatrix z_neg_nudge = rotation_nudge(curr_pose, -1, "z");

                // Starting with positive x nudge and render
                gpu_principal_model_->RenderPrimaryCamera_RotationMatrix(
                    x_pos_nudge);
                std::vector<float> iartd_pos = calculateIARTD(
                    img_desc_gpu,
                    gpu_principal_model_->GetPrimaryCameraRenderedImage());
                std::vector<float> hu_pos = img_desc_gpu->hu_moments(
                    gpu_principal_model_->GetPrimaryCameraRenderedImage());
                // Negative x nudge
                gpu_principal_model_->RenderPrimaryCamera_RotationMatrix(
                    x_neg_nudge);

                std::vector<float> iartd_neg = calculateIARTD(
                    img_desc_gpu,
                    gpu_principal_model_->GetPrimaryCameraRenderedImage());
                std::vector<float> hu_neg = img_desc_gpu->hu_moments(
                    gpu_principal_model_->GetPrimaryCameraRenderedImage());

                std::vector<float> iartd_dx =
                    vector_differece(iartd_pos, iartd_neg);
                std::vector<float> hu_dx = vector_differece(hu_pos, hu_neg);

                gpu_principal_model_->RenderPrimaryCamera_RotationMatrix(
                    y_pos_nudge);

                iartd_pos = calculateIARTD(
                    img_desc_gpu,
                    gpu_principal_model_->GetPrimaryCameraRenderedImage());
                hu_pos = img_desc_gpu->hu_moments(
                    gpu_principal_model_->GetPrimaryCameraRenderedImage());

                gpu_principal_model_->RenderPrimaryCamera_RotationMatrix(
                    y_neg_nudge);

                iartd_neg = calculateIARTD(
                    img_desc_gpu,
                    gpu_principal_model_->GetPrimaryCameraRenderedImage());
                hu_neg = img_desc_gpu->hu_moments(
                    gpu_principal_model_->GetPrimaryCameraRenderedImage());

                std::vector<float> iartd_dy =
                    vector_differece(iartd_pos, iartd_neg);
                std::vector<float> hu_dy = vector_differece(hu_pos, hu_neg);

                gpu_principal_model_->RenderPrimaryCamera_RotationMatrix(
                    z_pos_nudge);

                iartd_pos = calculateIARTD(
                    img_desc_gpu,
                    gpu_principal_model_->GetPrimaryCameraRenderedImage());
                hu_pos = img_desc_gpu->hu_moments(
                    gpu_principal_model_->GetPrimaryCameraRenderedImage());

                gpu_principal_model_->RenderPrimaryCamera_RotationMatrix(
                    z_neg_nudge);

                iartd_neg = calculateIARTD(
                    img_desc_gpu,
                    gpu_principal_model_->GetPrimaryCameraRenderedImage());
                hu_neg = img_desc_gpu->hu_moments(
                    gpu_principal_model_->GetPrimaryCameraRenderedImage());

                std::vector<float> iartd_dz =
                    vector_differece(iartd_pos, iartd_neg);
                std::vector<float> hu_dz = vector_differece(hu_pos, hu_neg);

                {
                    iartd_file << xr << "," << yr << "," << zr << ",dx,";
                    for (auto e : iartd_dx) {
                        iartd_file << e << ",";
                    }
                    iartd_file << std::endl;
                    iartd_file << xr << "," << yr << "," << zr << ",dy,";
                    for (auto e : iartd_dy) {
                        iartd_file << e << ",";
                    }
                    iartd_file << std::endl;
                    iartd_file << xr << "," << yr << "," << zr << ",dz,";
                    for (auto e : iartd_dz) {
                        iartd_file << e << ",";
                    }
                    iartd_file << std::endl;
                }
                {
                    hu_file << xr << "," << yr << "," << zr << ",dx,";
                    for (auto e : hu_dx) {
                        hu_file << e << ",";
                    }
                    hu_file << std::endl;
                    hu_file << xr << "," << yr << "," << zr << ",dy,";
                    for (auto e : hu_dy) {
                        hu_file << e << ",";
                    }
                    hu_file << std::endl;
                    hu_file << xr << "," << yr << "," << zr << ",dz,";
                    for (auto e : hu_dz) {
                        hu_file << e << ",";
                    }
                    hu_file << std::endl;
                }
                std::cout << iter << "/" << tot << std::endl;
                iter++;
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto dur =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    iartd_file.close();
    hu_file.close();

    std::cout << dur.count() << "ms elapsed for " << iter << " iterations"
              << std::endl;

    if (speed_test) {
        std::vector<float> iartd_test, hu_test;

        auto iartd_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < speed_test_iter; i++) {
            iartd_test = calculateIARTD(
                img_desc_gpu,
                gpu_principal_model_->GetPrimaryCameraRenderedImage());
        }
        auto iartd_end = std::chrono::high_resolution_clock::now();

        auto hu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < speed_test_iter; i++) {
            hu_test = img_desc_gpu->hu_moments(
                gpu_principal_model_->GetPrimaryCameraRenderedImage());
        }
        auto hu_end = std::chrono::high_resolution_clock::now();

        auto iartd_dur = std::chrono::duration_cast<std::chrono::microseconds>(
            iartd_end - iartd_start);
        auto hu_dur = std::chrono::duration_cast<std::chrono::microseconds>(
            hu_end - hu_start);

        std::cout << "IARTD Average Time: "
                  << iartd_dur.count() / speed_test_iter << "microseconds"
                  << std::endl;
        std::cout << "Hu Average Time: " << hu_dur.count() / speed_test_iter
                  << "microseconds" << std::endl;
    }
    delete (img_desc_gpu);
    delete (gpu_principal_model_);
    return 0;
}
