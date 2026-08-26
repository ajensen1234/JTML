/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#include "services/implant_estimator.h"

/*CUDA memcpy*/
#include <cuda_runtime.h>

/*OpenCV image ops (flip / resize / phaseCorrelate / sum)*/
#include <opencv2/imgproc.hpp>

/*GPUModel definition (declaration-only header chain, CXX-safe -- the oracle
 * test compiles the same chain under plain CXX)*/
#include "compute/gpu_model.cuh"

using namespace std;

namespace jta {

namespace {
/*Same literal as MainScreen's pi member (relocated with the math).*/
const double pi = 3.14159265358979323846;
}  // namespace

Point6D ImplantEstimator::EstimateImplantPose(
    ImplantEstimateContext& ctx,
    const cv::Mat& orig_inverted,
    bool clamp_z_to_principal) {
    /*Send Each Segmented Image to GPU Tensor, Predict Orientation, Then Z
    (From Area), then X,Y. After this, convert to non (0,0) centered
    orientation. Finally, update */
    cv::Mat padded;
    if (orig_inverted.cols > orig_inverted.rows) {
        padded.create(
            orig_inverted.cols, orig_inverted.cols, orig_inverted.type());
    } else {
        padded.create(
            orig_inverted.rows, orig_inverted.rows, orig_inverted.type());
    }
    unsigned int padded_width = padded.cols;
    unsigned int padded_height = padded.rows;
    padded.setTo(cv::Scalar::all(0));
    orig_inverted.copyTo(
        padded(cv::Rect(0, 0, orig_inverted.cols, orig_inverted.rows)));
    cv::resize(padded, padded, cv::Size(ctx.input_width, ctx.input_height));

    cudaMemcpy(
        ctx.gpu_byte_placeholder.data_ptr(),
        padded.data,
        ctx.input_width * ctx.input_height * sizeof(unsigned char),
        cudaMemcpyHostToDevice);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(ctx.gpu_byte_placeholder.to(dtype(torch::kFloat))
                         .flip({2}));  // Must flip first
    cudaMemcpy(
        ctx.orientation,
        ctx.model->forward(inputs)
            .toTensor()
            .to(dtype(torch::kFloat))
            .data_ptr(),
        3 * sizeof(float),
        cudaMemcpyDeviceToHost);
    /*Flip Segment*/
    auto output_mat_seg =
        cv::Mat(orig_inverted.rows, orig_inverted.cols, CV_8UC1);
    flip(orig_inverted, output_mat_seg, 0);

    /*Render*/
    ctx.gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(
        0,
        0,
        -ctx.calibration.camera_A_principal_.principal_distance_,
        ctx.orientation[1],
        ctx.orientation[2],
        ctx.orientation[0]));

    /*Copy To Mat*/
    cudaMemcpy(
        ctx.host_image,
        ctx.gpu_mod->GetPrimaryCameraRenderedImagePointer(),
        ctx.orig_width * ctx.orig_height * sizeof(unsigned char),
        cudaMemcpyDeviceToHost);

    /*OpenCV Image Container/Write Function*/
    auto projection_mat = cv::Mat(
        ctx.orig_height,
        ctx.orig_width,
        CV_8UC1,
        ctx.host_image); /*Reverse before flip*/
    auto output_mat = cv::Mat(ctx.orig_width, ctx.orig_height, CV_8UC1);
    flip(projection_mat, output_mat, 0);

    /*Get Scale*/
    double sum_seg = sum(sum(output_mat_seg))[0] / 255.0;
    double sum_proj = sum(sum(output_mat))[0] / 255.0;
    double z;
    if (clamp_z_to_principal) {
        /* Creating A check to ensure that the z translation is not greater
         * than the principal distance */
        if (sum_proj / sum_seg > 1) {
            z = -ctx.calibration.camera_A_principal_.principal_distance_;
        } else {
            z = -ctx.calibration.camera_A_principal_.principal_distance_ *
                sqrt(sum_proj / sum_seg);
        }
    } else {
        z = -ctx.calibration.camera_A_principal_.principal_distance_ *
            sqrt(sum_proj / sum_seg);
    }

    /*Reproject*/
    /*Render*/
    ctx.gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(
        0, 0, z, ctx.orientation[1], ctx.orientation[2], ctx.orientation[0]));
    cudaMemcpy(
        ctx.host_image,
        ctx.gpu_mod->GetPrimaryCameraRenderedImagePointer(),
        ctx.orig_width * ctx.orig_height * sizeof(unsigned char),
        cudaMemcpyDeviceToHost);
    projection_mat =
        cv::Mat(ctx.orig_height, ctx.orig_width, CV_8UC1, ctx.host_image);
    output_mat = cv::Mat(ctx.orig_width, ctx.orig_height, CV_8UC1);
    flip(projection_mat, output_mat, 0);

    /*cv::imwrite("C:/Users/pflood/Desktop/output_mat.png", output_mat);
    cv::imwrite("C:/Users/pflood/Desktop/output_mat_seg.png",
    output_mat_seg);*/

    /*Get X and Y*/
    cv::Mat proj64;
    output_mat.convertTo(proj64, CV_64FC1);
    cv::Mat seg64;
    output_mat_seg.convertTo(seg64, CV_64FC1);
    cv::Point2d x_y_point = phaseCorrelate(proj64, seg64) *
        (ctx.calibration.camera_A_principal_.pixel_pitch_ * z * -1) /
        ctx.calibration.camera_A_principal_.principal_distance_;
    double x = x_y_point.x;
    double y = -1 * x_y_point.y;

    // QMessageBox::critical(this, "Error!", QString::number(x) + ", " +
    //	QString::number(y) + ", " +
    //	QString::number(z) + ", " +
    //	QString::number(orientation[1]) + ", " +
    //	QString::number(orientation[2]) + ", " +
    //	QString::number(orientation[0]), QMessageBox::Ok);

    /*Convert from (0,0) Centered*/
    float za_rad = ctx.orientation[0] * pi / 180.0;
    float xa_rad = ctx.orientation[1] * pi / 180.0;
    float ya_rad = ctx.orientation[2] * pi / 180.0;
    float cz = cos(za_rad);
    float sz = sin(za_rad);
    float cx = cos(xa_rad);
    float sx = sin(xa_rad);
    float cy = cos(ya_rad);
    float sy = sin(ya_rad);
    Matrix_3_3 R_g(
        cz * cy - sz * sx * sy,
        -1.0 * sz * cx,
        cz * sy + sz * cy * sx,
        sz * cy + cz * sx * sy,
        cz * cx,
        sz * sy - cz * cy * sx,
        -1.0 * cx * sy,
        sx,
        cx * cy);
    float theta_x = std::atan(-1.0 * y / z);
    float theta_y = std::asin(-1.0 * x / std::sqrt(x * x + y * y + z * z));
    Matrix_3_3 R_x(
        1, 0, 0, 0, cos(theta_x), -sin(theta_x), 0, sin(theta_x), cos(theta_x));
    Matrix_3_3 R_y(
        cos(theta_y), 0, sin(theta_y), 0, 1, 0, -sin(theta_y), 0, cos(theta_y));
    Matrix_3_3 R_orig = ctx.calibration.multiplication_mat_mat(
        R_y, ctx.calibration.multiplication_mat_mat(R_x, R_g));
    /*Rot Mat To Eul ZXY*/
    /*Algorithm To Recover Z - X - Y Euler Angles*/
    float xa, ya, za;
    if (R_orig.A_32_ < 1) {
        if (R_orig.A_32_ > -1) {
            xa = asin(R_orig.A_32_);
            za = atan2(-1 * R_orig.A_12_, R_orig.A_22_);
            ya = atan2(-1 * R_orig.A_31_, R_orig.A_33_);

        } else {
            xa = -pi / 2.0;
            za = -1 * atan2(R_orig.A_13_, R_orig.A_11_);
            ya = 0;
        }
    } else {
        xa = pi / 2.0;
        za = atan2(R_orig.A_13_, R_orig.A_11_);
        ya = 0;
    }

    xa = xa * 180.0 / pi;
    ya = ya * 180.0 / pi;
    za = za * 180.0 / pi;
    /*
                    QMessageBox::critical(this, "Error!",
       QString::number(x)
       +
       ", " + QString::number(y) + ", " + QString::number(z) + ", " +
                            QString::number(xa) + ", " +
                            QString::number(ya) + ", " +
                            QString::number(za), QMessageBox::Ok);*/
    return Point6D(x, y, z, xa, ya, za);
}

}  // namespace jta
