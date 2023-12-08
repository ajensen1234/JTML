// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "machine_learning_tools.h"

cv::Mat segment_image(const cv::Mat& orig_image, bool black_sil_used,
                      torch::jit::Module* model, unsigned int input_width,
                      unsigned int input_height) {
    /*Create a GPU byte placeholder for memory purposes*/
    torch::Tensor gpu_byte_placeholder(
        torch::zeros({1, 1, input_height, input_width},
                     torch::device(torch::kCUDA).dtype(torch::kByte)));
    /*Get the correct inversion for the image*/
    cv::Mat correct_inversion =
        (255 * black_sil_used) + ((1 - 2 * black_sil_used) * orig_image);
    cv::Mat padded;

    /*Pad the image to a square based on the larger dimension*/
    if (correct_inversion.cols > correct_inversion.rows) {
        padded.create(correct_inversion.cols, correct_inversion.cols,
                      correct_inversion.type());
    } else {
        padded.create(correct_inversion.rows, correct_inversion.rows,
                      correct_inversion.type());
    }

    const unsigned int padded_width = padded.cols;
    const unsigned int padded_height = padded.rows;

    padded.setTo(cv::Scalar::all(0));

    /*Copy things over to the GPU for the forward pass*/
    correct_inversion.copyTo(
        padded(cv::Rect(0, 0, correct_inversion.cols, correct_inversion.rows)));
    cv::resize(padded, padded, cv::Size(input_width, input_height));
    cudaMemcpy(gpu_byte_placeholder.data_ptr(), padded.data,
               input_height * input_width * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    /*Define the machine learning inputs*/
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(
        gpu_byte_placeholder.to(torch::dtype(torch::kFloat)).flip({2}));

    /*Forward Pass and bring it back to host*/
    cudaMemcpy(padded.data,
               (255 * (model->forward(inputs).toTensor() > 0))
                   .to(torch::dtype(torch::kByte))
                   .flip({2})
                   .data_ptr(),
               input_height * input_width * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);

    cv::resize(padded, padded, cv::Size(padded_width, padded_height));
    cv::Mat unpadded =
        padded(cv::Rect(0, 0, correct_inversion.cols, correct_inversion.rows));

    return unpadded;
}