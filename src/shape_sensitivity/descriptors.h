#ifndef DESCRIPTORS_H_
#define DESCRIPTORS_H_
#include <complex.h>

#include <opencv2/opencv.hpp>

#include "art.cuh"

std::vector<double> calculateIARTD(img_desc* img_desc_gpu);

#endif  // DESCRIPTORS_H_
