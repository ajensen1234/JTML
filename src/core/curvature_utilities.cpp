#include <core/curvature_utilities.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void extract_contour_points(cv::Mat input_edge_image,
                            std::vector<std::vector<cv::Point>> *contour) {
    cv::findContours(input_edge_image, *contour, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_NONE);
};
void calculate_curvature_along_contour(std::vector<cv::Point_<int>> contour,
                                       float *curvature) {
    std::cout << "Before creating pointers to three poitns " << std::endl;
    cv::Point_<int> *p1 = new cv::Point_<int>;
    cv::Point_<int> *p2 = new cv::Point_<int>;
    std::cout << "Null ptr: " << *p1 << std::endl;
    int dist = 18;
    std::cout << "before curvature for loop" << std::endl;
    for (int idx = 0; idx < contour.size(); idx++) {
        std::cout << "Before defining value of the reference points"
                  << std::endl;
        std::cout << "Reference point values: " << contour[idx].x << ", "
                  << contour[idx].y << std::endl;
        cv::Point ref_pt = contour[idx];
        // Pick the three points along the contour that we will use to create
        // the triangle and calculate curvature
        std::cout << "before pick 3 points" << std::endl;
        pick_three_points(contour, idx, dist, p1, ref_pt, p2);
        std::cout << "Before setting curvature value based on menger curvature"
                  << std::endl;
        curvature[idx] = (menger_curvature(*p1, ref_pt, *p2));
    }
    delete (p1);
    delete (p2);
    // TODO: Throw an error here if the length of the curvature is not the same
    // length as the contourkj
    return;
};
void determine_regions_of_high_curvature() { return; };

float menger_curvature(cv::Point_<int> p1, cv::Point_<int> ref_pt,
                       cv::Point_<int> p2) {
    cv::Vec<int, 2> vec1 = cv::Vec<int, 2>(p1.x - ref_pt.x, p1.y - ref_pt.y);
    cv::Vec<int, 2> vec2 = cv::Vec<int, 2>(p2.x - ref_pt.x, p2.y - ref_pt.y);
    cv::Vec<int, 2> vec3 = cv::Vec<int, 2>(p2.x - p1.x, p2.y - p1.y);
    float norm1 = cv::norm(vec1);
    float norm2 = cv::norm(vec2);
    float norm3 = cv::norm(vec3);
    std::cout << "Norms: " << norm1 << ", " << norm2 << ", " << norm3
              << std::endl;
    float inv_radius = 4 * cv::abs(vec1[0] * vec2[1] - vec1[1] * vec2[0]) /
                       (norm1 * norm2 * norm3);
    return inv_radius;
};

void pick_three_points(std::vector<cv::Point_<int>> contour_points, int idx,
                       int dist, cv::Point_<int> *p1, cv::Point_<int> ref_pt,
                       cv::Point_<int> *p2) {
    std::cout << "Before determining contour length" << std::endl;
    int contour_length = contour_points.size();
    // First, we make sure that the starting point doesn't get shoved behind the
    // vector
    std::cout << "Before setting p1/2 values" << std::endl;
    std::cout << "Pointer address: " << &p1 << std ::endl;
    std::cout << "Pointer Value(p1): " << p1 << std::endl;
    std::cout << "Pointer Value(*p1): " << *p1 << std::endl;
    if (idx < dist) {
        std::cout << "If idx<dist" << std::endl;
        std::cout << contour_points[contour_length - (dist - idx)] << std::endl;
        *p1 = contour_points[contour_length - (dist - idx)];
        *p2 = contour_points[idx + dist];
    } else if ((contour_length - idx) < dist) {
        *p1 = contour_points[idx - dist];
        *p2 = contour_points[dist - (contour_length - idx)];
    } else {
        *p1 = contour_points[idx - dist];
        *p2 = contour_points[idx + dist];
    }
    return;
};

void generate_curvature_heatmaps(cv::Mat input_image) {
    // std::vector<cv::Mat> &heatmaps) {
    /*
    ** This is the primary function that will call all of the other
    *sub-functions.
    ** It takes as input an input edge_image from frame, then:
    ** 1) Extracts external contour points
    ** 2) Calculates curvature along the contour using Menger's Algorithm
    ** 3) Determines the regions of high curvature, and extracts "centroids" of
    *those regions
    ** 4) Applies a distance transform to a pixel located at each region of
    *curvature interest
    ** 5) Saves these curvature heatmaps for use in the cost function
    */

    // Contour placeholder
    std::vector<std::vector<cv::Point_<int>>> *contour =
        new std::vector<std::vector<cv::Point_<int>>>;
    extract_contour_points(input_image, contour);
    draw_contours(contour);
    std::cout << "Contour Hierarchy Size: " << contour->size() << std::endl;
    std::cout << "Largest Contour at :" << contour->back().size() << std::endl;
    for (auto ctr : *contour) {
        std::cout << "Size: " << ctr.size() << std::endl;
    }
    std::cout << "Before creatuing curvature pointer" << std::endl;
    float *curvature = new float[contour->back().size()];
    std::cout << "Before calculating curvature values" << std::endl;
    calculate_curvature_along_contour(contour->back(), curvature);
    for (int idx = 0; idx < contour->back().size(); idx++) {
        std::cout << curvature[idx] << std::endl;
    }
    std::cout << "Before calculating curvature mean" << std::endl;

    float curv_mean = calculate_mean(curvature, contour->back().size());
    std::cout << "Mean: " << curv_mean << std::endl;
    std::cout << "Before caluculating stdev" << std::endl;
    float curv_std = calculate_std(curvature, contour->back().size());
    std::cout << "Curv STDev" << curv_std << std::endl;

    float alpha = 1.5;  // How many standard deviations we care about
    float curv_threshold = curv_mean + alpha * curv_std;
    // bool *curv_thresh_array = new bool[contour[0].size()];
    delete (contour);
    delete (curvature);
    // delete (curv_thresh_array);
    return;
};

float calculate_mean(float *vals, int len) {
    float sum = 0.0;
    for (int i = 0; i < len; i++) {
        sum += vals[i];
    }
    return sum / len;
}
float calculate_std(float *vals, int len) {
    float mean = calculate_mean(vals, len);
    float stdev = 0.0;
    for (int i = 0; i < len; i++) {
        stdev += pow(vals[i] - mean, 2);
    }
    return sqrt(stdev / len);
}

void draw_contours(std::vector<std::vector<cv::Point_<int>>> *contour) {
    // create a source image to hold the contour
    cv::Mat dst = cv::Mat(1024, 1024, CV_8UC1);
    cv::drawContours(dst, *contour, 0, 255);
    cv::imwrite("contour.png", dst);
    return;
}
