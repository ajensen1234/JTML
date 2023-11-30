#ifndef DESCRIPTORS_H_
#define DESCRIPTORS_H_
#include <complex.h>

#include <opencv2/opencv.hpp>

#include "art.cuh"
#include "gpu/gpu_image.cuh"

std::vector<float> calculateIARTD(img_desc* img_desc_gpu,
                                  gpu_cost_function::GPUImage* dev_image);

#endif  // DESCRIPTORS_H_
