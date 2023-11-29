#include <chrono>
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

    auto img_desc_gpu = new img_desc(binary_image.rows, binary_image.cols, 0,
                                     binary_image.data);

    int num_runs = 250;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        std::vector<double> iartd = calculateIARTD(img_desc_gpu);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Average time taken by Angular Radial Transform (" << num_runs
              << " runs): " << duration.count() / num_runs << " milliseconds"
              << std::endl;  // std::vector<double> iartd2 =
                             // calculateIARTD(binary_image2);
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        std::vector<double> hu = img_desc_gpu->hu_moments();
    }
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);
    std::cout << "Average time taken by Hu Moments (" << num_runs
              << " runs): " << duration2.count() / num_runs << " milliseconds"
              << std::endl;  // std::vector<double> iartd2 =

    delete (img_desc_gpu);
    return 0;
}
