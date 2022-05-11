#include "nfd_instance.h"

using namespace cv;
using Vec = std::vector<double>;
nfd_instance::nfd_instance(GPUModel &gpu_mod,float xt, float yt, float zt, float xr, float yr, float zr) {
	instance_pose_ = Pose(xt, yt, zt, xr, yr, zr);
	rot_x_ = xr;
	rot_y_ = yr;

	gpu_mod.RenderPrimaryCamera(instance_pose_);
	cv::Mat img = gpu_mod.GetOpenCVPrimaryRenderedImage();

	get_contour_points(img);

	Spline<vec2d, float> _spline_curve();




}

nfd_instance::~nfd_instance()
{
	
}


void nfd_instance::get_contour_points(cv::Mat img) {
	Mat canny_output;
	Canny(img, canny_output, 0, 150);
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;
	findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
	for (std::vector<Point> ctr : contours) {
		for (Point pt : ctr) {
			//std::cout << pt.x << pt.y << std::endl;
			vec2d tmp;
			tmp.set_x(pt.x);
			tmp.set_y(pt.y);
			std::vector<int> vect{ pt.x, pt.y };
			contour_pts_.push_back(tmp);
			ctr_pts_.push_back({ pt.x,pt.y });
		}
	}

	
}


