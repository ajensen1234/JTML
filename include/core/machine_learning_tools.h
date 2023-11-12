#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat segment_image(const cv::Mat& orig_image, bool black_sil_used,
                      torch::jit::Module* model, unsigned int input_width,
                      unsigned int input_height);
