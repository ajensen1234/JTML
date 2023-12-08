/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*Header for Frame Class:
The Frame class stores:
1. Width and Height information for an x-ray image
2. Original x-ray image
3. Inverted grayscale of the x-ray image
3. Edge Detection constants (aperture, low/high threshold)
4. Edge Detected version of the image
5. maximum dilation constant
6. dilated version of the edge image*/

#ifndef FRAME_H
#define FRAME_H

/*Standard*/
#include <string>

/*OpenCV 3.1 Library*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "core/curvature_utilities.h"

class Frame {
   public:
    Frame(std::string file_location, int aperture, int low_threshold,
          int high_threshold, int dilation);
    ~Frame(){};

    /*Get Set Methods for Main Variables*/
    /*Return Original Image*/
    cv::Mat GetOriginalImage();
    /*Return Edge Detected Image*/
    cv::Mat GetEdgeImage();
    /*Return Dilated Edge Detected Image*/
    cv::Mat GetDilationImage();
    /*Return Inverted Intensity Image*/
    cv::Mat GetInvertedImage();

    cv::Mat GetDistanceMap();

    /*Store Public String Location*/
    std::string file_location_;

    /*Reset From Original (Resets Inverted/Segmented, Edge, Dilation from
    Original, Useful if Trying to Reset from Segmentation)*/
    void ResetFromOriginal();

    /*Recalculate Edge Detected Image*/
    void SetEdgeImage(int aperture, int low_threshold, int high_threshold,
                      bool use_reverse = false);
    /*Recalculate Dilated Image*/
    void SetDilatedImage(int dilation);
    void SetDistanceMap();

    /*Get Canny Parameters*/
    int GetAperture();
    int GetHighThreshold();
    int GetLowThreshold();

    /*Curvature Heatmap Getter and Setter*/
    void setCurvatureHeatmaps();
    std::vector<uchar> getCurvatureHeatmaps();

    std::vector<uchar> flattenVector(
        const std::vector<std::vector<uchar>>& vecOfVecs);
    int GetNumCurvatureKeypoints();

   private:
    /*Original Matrix*/
    cv::Mat original_image_;
    /*Edge Detected Matrix*/
    cv::Mat edge_image_;
    /*Dilation Matrix*/
    cv::Mat dilation_image_;
    /*Inverted Matrix  (Store in Inverted)*/
    cv::Mat inverted_image_;
    cv::Mat distance_map_;
    std::vector<cv::Mat> curvature_heatmaps_;
    std::vector<uchar> curvature_heatmap_chars_;
    int num_curvature_keypoints_;
    /*Constants*/
    int aperture_;
    int low_threshold_;
    int high_threshold_;
    int dilation_;
    int width_;
    int height_;
};

#endif /* FRAME_H */
