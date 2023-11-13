#include <core/curvature_utilities.h>
#include <opencv2/core/hal/interface.h>

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
    cv::Point_<int> *p1 = new cv::Point_<int>;
    cv::Point_<int> *p2 = new cv::Point_<int>;
    int dist = 18;
    for (int idx = 0; idx < contour.size(); idx++) {
        cv::Point ref_pt = contour[idx];
        // Pick the three points along the contour that we will use to create
        // the triangle and calculate curvature
        pick_three_points(contour, idx, dist, p1, ref_pt, p2);
        curvature[idx] = (menger_curvature(*p1, ref_pt, *p2));
    }
    delete (p1);
    delete (p2);
    // TODO: Throw an error here if the length of the curvature is not the same
    // length as the contourkj
    return;
};

float menger_curvature(cv::Point_<int> p1, cv::Point_<int> ref_pt,
                       cv::Point_<int> p2) {
    cv::Vec<int, 2> vec1 = cv::Vec<int, 2>(p1.x - ref_pt.x, p1.y - ref_pt.y);
    cv::Vec<int, 2> vec2 = cv::Vec<int, 2>(p2.x - ref_pt.x, p2.y - ref_pt.y);
    cv::Vec<int, 2> vec3 = cv::Vec<int, 2>(p2.x - p1.x, p2.y - p1.y);
    float norm1 = cv::norm(vec1);
    float norm2 = cv::norm(vec2);
    float norm3 = cv::norm(vec3);
    float inv_radius = 4 * cv::abs(vec1[0] * vec2[1] - vec1[1] * vec2[0]) /
                       (norm1 * norm2 * norm3);
    return inv_radius;
};

void pick_three_points(std::vector<cv::Point_<int>> contour_points, int idx,
                       int dist, cv::Point_<int> *p1, cv::Point_<int> ref_pt,
                       cv::Point_<int> *p2) {
    int contour_length = contour_points.size();
    // First, we make sure that the starting point doesn't get shoved behind the
    // vector
    if (idx < dist) {
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

std::vector<cv::Mat> generate_curvature_heatmaps(cv::Mat input_image) {
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
    int N = contour->back().size();
    float *curvature = new float[N];
    calculate_curvature_along_contour(contour->back(), curvature);

    float curv_mean = calculate_mean(curvature, contour->back().size());
    float curv_std = calculate_std(curvature, contour->back().size());
    float alpha = 1.5;  // How many standard deviations we care about
    float curv_threshold = curv_mean + alpha * curv_std;
    // bool *curv_thresh_array = new bool[contour[0].size()];
    float *smoothed_curvature = new float[N];
    int kern_size = 5;
    double sigma = 2;

    gaussian_convolution(curvature, N, sigma, smoothed_curvature);

    float *curvature_derivative = new float[N];

    calculate_derivative(smoothed_curvature, curvature_derivative, 1, N);

    std::vector<int> key_curvature_points = positive_inflection_points(
        smoothed_curvature, curvature_derivative, N, curv_threshold);
    // Grab size of the input image
    int W = input_image.cols, H = input_image.rows;
    // Creating storage for the vector of heatmaps
    std::vector<cv::Mat> heatmaps;
    int heatmap_idx = 0;
    for (auto pt_idx : key_curvature_points) {
        cv::Point hm_point = contour->back()[pt_idx];
        cv::Mat flipped_single_hm =
            heatmap_at_point(hm_point.x, hm_point.y, H, W);
        heatmaps.push_back(heatmap_at_point(hm_point.x, hm_point.y, H, W));
        cv::Mat single_hm = cv::Mat(H, W, CV_8UC1);
        cv::flip(flipped_single_hm, single_hm, 0);
        std::string fname = "hm" + std::to_string(heatmap_idx) + ".png";
        cv::imwrite(fname, single_hm);
        heatmap_idx++;
    }
    delete (contour);
    delete (curvature_derivative);
    delete (curvature);
    delete (smoothed_curvature);
    // delete (curv_thresh_array);
    return heatmaps;
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

float array_at_idx(float *arr, int idx, int N) {
    if (idx < 0) {
        return arr[N + idx];
    } else if (idx >= N) {
        return arr[idx - N];
    } else {
        return arr[idx];
    }
}

float dot(float arr1[], float arr2[], int N) {
    float sum = 0;

    for (int i = 0; i < N; i++) {
        sum += arr1[i] * arr2[i];
    }
    return sum;
}

void gaussian_convolution(float *arr, int N, float sigma, float *result) {
    auto gaussianKernel = [sigma](double x) {
        float coefficient = 1.0 / (std::sqrt(2.0 * M_PI) * sigma);
        float exponent = -0.5 * (x * x) / (sigma * sigma);
        return coefficient * std::exp(exponent);
    };
    // creating the kernel
    float kernel[9] = {
        gaussianKernel(-4), gaussianKernel(-3), gaussianKernel(-2),
        gaussianKernel(-1), gaussianKernel(0),  gaussianKernel(1),
        gaussianKernel(2),  gaussianKernel(3),  gaussianKernel(4)};

    for (int i = 0; i < N; i++) {
        float arr_subset[9] = {
            array_at_idx(arr, i - 4, N), array_at_idx(arr, i - 3, N),
            array_at_idx(arr, i - 2, N), array_at_idx(arr, i - 1, N),
            array_at_idx(arr, i, N),     array_at_idx(arr, i + 1, N),
            array_at_idx(arr, i + 2, N), array_at_idx(arr, i + 3, N),
            array_at_idx(arr, i + 4, N),
        };
        result[i] = dot(arr_subset, kernel, 9) / arr_sum(kernel, 9);
    }
}

float arr_sum(float arr[], int N) {
    float res = 0;
    for (int i = 0; i < N; i++) {
        res += arr[i];
    }
    return res;
}
void calculate_derivative(float *arr, float *der, int del_x, int N) {
    // This is basically the first thing that you learn in calc 1.
    // For the input point, we are looking del_x in front and behind it
    // Then determining the discrete derivative.

    for (int i = 0; i < N; i++) {
        der[i] =
            array_at_idx(arr, i + del_x, N) - array_at_idx(arr, i - del_x, N);
    }
}
std::vector<int> positive_inflection_points(float *arr, float *der, int N,
                                            float threshold) {
    std::vector<int> infl_pts;
    for (int i = 0; i < N; i++) {
        bool infl =
            (array_at_idx(der, i - 1, N) > 0) && (array_at_idx(der, i, N) < 0);
        if ((infl) && (array_at_idx(arr, i, N) > threshold)) {
            infl_pts.push_back(i);
        }
    }
    return infl_pts;
}

cv::Mat heatmap_at_point(int x, int y, int height, int width) {
    cv::Mat single_dot = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));
    single_dot.at<uchar>(y, x) = 0;
    // now create the distance transform to that single point
    cv::Mat heatmap = cv::Mat(height, width, CV_8UC1);
    cv::distanceTransform(single_dot, heatmap, cv::DIST_L1, 5, CV_8UC1);
    return heatmap;
}
