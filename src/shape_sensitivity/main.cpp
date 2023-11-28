#include <iostream>
#include <opencv2/opencv.hpp>

#include "descriptors.h"
// #include "gpu/gpu_frame.cuh"
// #include "gpu/gpu_metrics.cuh"
// #include "gpu/gpu_model.cuh"
int main() {
    cv::Mat binary_image = cv::imread(
        "/media/ajensen123@ad.ufl.edu/Andrew's External "
        "SSD/Data/JTML_ALL_DATA/Actiyas/"
        "fem_label_grid_Actiyas_000000000001.tif",
        cv::IMREAD_GRAYSCALE);

    cv::Mat binary_image2 = cv::imread(
        "/media/ajensen123@ad.ufl.edu/Andrew's External "
        "SSD/Data/JTML_ALL_DATA/Ghent/"
        "fem_label_grid_Ghent_000000000000.tif",
        cv::IMREAD_GRAYSCALE);
    std::vector<double> iartd = calculateIARTD(binary_image);
    // std::vector<double> iartd2 = calculateIARTD(binary_image2);
    for (int i = 0; i < iartd.size(); i++) {
        std::cout << iartd[i] << std::endl;
    }
    return 0;
}
