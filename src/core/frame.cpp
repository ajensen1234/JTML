/*Frame Header*/
#include "core/frame.h"
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

/*Constructor*/
Frame::Frame(std::string file_location, int aperture, int low_threshold,
             int high_threshold, int dilation) {

  /*Save File Location*/
  file_location_ = file_location;

  /*Read in to the original*/
  flip(cv::imread(file_location, CV_8UC1), original_image_, 0);
  height_ = original_image_.rows;
  width_ = original_image_.cols;

  /*Initialize Other Matrices*/
  edge_image_ = cv::Mat(height_, width_, CV_8UC1);
  dilation_image_ = cv::Mat(height_, width_, CV_8UC1);
  inverted_image_ = cv::Mat(height_, width_, CV_8UC1);
  distance_map_ = cv::Mat(height_, width_, CV_8UC1);

  /*Create Alternative Images*/
  SetEdgeImage(aperture, low_threshold, high_threshold);
  SetDilatedImage(dilation);
  inverted_image_ = (255 - original_image_);
  SetDistanceMap();

  /*Store Constants*/
  aperture_ = aperture;
  low_threshold_ = low_threshold;
  high_threshold_ = high_threshold;
  dilation_ = dilation;
};
/*Recalculate Edge Detected Image*/
void Frame::SetEdgeImage(int aperture, int low_threshold, int high_threshold,
                         bool use_reverse) {
  aperture_ = aperture;
  low_threshold_ = low_threshold;
  high_threshold_ = high_threshold;
  if (!use_reverse)
    Canny(original_image_, edge_image_, low_threshold, high_threshold,
          aperture);
  else
    Canny(inverted_image_, edge_image_, low_threshold, high_threshold,
          aperture);
}
/*Recalculate Dilated Image*/
void Frame::SetDilatedImage(int dilation) {
  dilation_ = dilation;
  dilate(edge_image_, dilation_image_, cv::Mat(), cv::Point(-1, -1), dilation_);
}

/*Return Original Image*/
cv::Mat Frame::GetOriginalImage() { return original_image_; }
/*Return Edge Detected Image*/
cv::Mat Frame::GetEdgeImage() { return edge_image_; }
/*Return Dilated Edge Detected Image*/
cv::Mat Frame::GetDilationImage() { return dilation_image_; }
/*Return Inverted Intensity Image*/
cv::Mat Frame::GetInvertedImage() { return inverted_image_; }

cv::Mat Frame::GetDistanceMap() { return distance_map_; }

/*Reset From Original (Resets Inverted/Segmented, Edge, Dilation from Original,
Useful if Trying to Reset from Segmentation)*/
void Frame::ResetFromOriginal() {

  /*Reset Inverted Image*/
  inverted_image_ = (255 - original_image_);

  /*Reset Edge*/
  SetEdgeImage(aperture_, low_threshold_, high_threshold_);

  /*Reset Dilation*/
  SetDilatedImage(dilation_);
}

void Frame::SetDistanceMap() {
  // We are going to do a few operations inline.
  // We grab the edge image, invert it, then run distance
  // transform on that inverted edge image.
  // We do this because distance transform finds closest black pixel

  cv::Mat inverse_edge = cv::Mat(height_, width_, CV_8UC1);
  // cv::bitwise_not(GetEdgeImage(), inverse_edge);
  inverse_edge = (255 - edge_image_);
  cv::distanceTransform(inverse_edge, distance_map_, cv::DIST_L1, 5, CV_8UC1);
}

/*Get Canny Parameters*/
int Frame::GetAperture() { return aperture_; };
int Frame::GetHighThreshold() { return high_threshold_; };
int Frame::GetLowThreshold() { return low_threshold_; };
