#include <nfd/nfd_instance.h>

using namespace cv;
using namespace alglib;
nfd_instance::nfd_instance(GPUModel &gpu_mod, float xt, float yt, float zt,
                           float xr, float yr, float zr) {
    instance_pose_ = Pose(xt, yt, zt, xr, yr, zr);
    rot_x_ = xr;
    rot_y_ = yr;

    gpu_mod.RenderPrimaryCamera(instance_pose_);
    cv::Mat img = gpu_mod.GetOpenCVPrimaryRenderedImage();

    get_contour_points(img);
}

nfd_instance::~nfd_instance() {}

void nfd_instance::print_contour_points() {
    // std::cout << x_points_resampled_.size() << ", "<< sz_ << std::endl;
    for (int i = 0; i < x_points_resampled_.size(); i++) {
        std::cout << x_points_resampled_[i] << ", " << y_points_resampled_[i]
                  << std::endl;
    }
}

void nfd_instance::get_contour_points(cv::Mat img) {
    Mat canny_output;
    Canny(img, canny_output, 0, 150);
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(canny_output, contours, hierarchy, RETR_TREE,
                 CHAIN_APPROX_NONE);  // TODO: figure out what to do with the
                                      // hierarchy. Make sure that you are only
                                      // grabbing the largest of the contours.
    for (std::vector<Point> ctr : contours) {
        if (ctr.size() > 200) {
            sz_ = ctr.size();
            contour_points_raw_.setlength(sz_, 2);
            for (int i = 0; i < ctr.size(); i++) {
                // Point pt = ctr[i].x;
                contour_points_raw_[i][0] = ctr[i].x;
                contour_points_raw_[i][1] = ctr[i].y;
            }
            break;
        }
    }

    pspline2buildperiodic(contour_points_raw_, sz_, 2, 0, contour_spline_);

    for (float i = 0; i < 128; i++) {
        double x_tmp;
        double y_tmp;
        pspline2calc(contour_spline_, i / 128, x_tmp, y_tmp);
        x_points_resampled_.push_back(x_tmp);
        y_points_resampled_.push_back(y_tmp);
    }
}

void nfd_instance::print_raw_points() {
    // std::cout << contour_points_raw_.rows() << ", " << sz_ << std::endl;

    for (int i = 0; i < sz_; i++) {
        std::cout << contour_points_raw_[i][0] << ", "
                  << contour_points_raw_[i][1] << std::endl;
    }
}
