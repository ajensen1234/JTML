#ifndef DESCRIPTORS_H_
#define DESCRIPTORS_H_
#include <complex.h>

#include <opencv2/opencv.hpp>

#include "art.cuh"

std::vector<double> calculateIARTD(cv::Mat binary_image);

#endif  // DESCRIPTORS_H_
