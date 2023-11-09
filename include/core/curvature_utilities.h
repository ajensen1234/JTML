#pragma once
// file: curvature_utilities.h
//  Here, we are creating all the utilities needed to calculate the curvature of
//  shapes. Ultimately, we will return the heatmaps based on the contour
//  function
#include <math.h>

#include <array>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

void extract_contour_points(cv::Mat input_edge_image,
                            std::vector<std::vector<cv::Point>>* contour);
void calculate_curvature_along_contour(std::vector<cv::Point_<int>> contour,
                                       float* curvature);
void determine_regions_of_high_curvature();
void generate_curvature_heatmaps(cv::Mat input_image);

float menger_curvature(cv::Point_<int> p1, cv::Point_<int> ref_pt,
                       cv::Point_<int> p2);

void pick_three_points(std::vector<cv::Point_<int>> contour_points, int idx,
                       int dist, cv::Point_<int>* p1, cv::Point_<int> ref_pt,
                       cv::Point_<int>* p2);

float calculate_mean(float* vals, int len);
float calculate_std(float* vals, int len);
void draw_contours(std::vector<std::vector<cv::Point_<int>>>* contour);
