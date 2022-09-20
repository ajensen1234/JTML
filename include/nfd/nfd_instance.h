#pragma once
#include<complex>
#include<vector>
#include<array>
#include "render_engine.cuh"
#include "gpu_model.cuh"
#include <opencv2/core/mat.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
<<<<<<< HEAD:nfd/nfd_instance.h
#include "../alglib/interpolation.h"
=======
#include "alglib/interpolation.h"
>>>>>>> update-directory-structure-main:include/nfd/nfd_instance.h

using namespace gpu_cost_function;
using namespace alglib;
/*
This class is beeing used to store a single instance of the NFD libarary data.

A single instance is hereby going to refer to a single projection geometry (meaning only a single x/y rotation),
along with all the other data that might be associated with that. Multiple normalization coefficients are going 
to be defined for any one instance (as you see in the Banks paper).

The nfd_library class is going to be used to store a library of all the instances that we create for a single projection,
with some extra metadata surrounding it.

The main nfd function is going to manage all the initialization of the different models, as well as populating each of 
the respective classes with the values that come from the different image projections.

*/
struct vec2d {
	float x;
	float y;
	void set_x(float xval) {
		x = xval;
	}
	void set_y(float yval) {
		y = yval;
	}
};

class nfd_instance
{
public:

	/*Constructor takes in the pose and creates the instance - will eventually also do the projections*/
	nfd_instance(GPUModel &gpu_mod,float xt, float yt, float zt, float xr, float yr, float zr);
	~nfd_instance();

	void print_contour_points();
	void print_raw_points();

	/*Set the pose of the instance - 312 rotation order when projected*/
	

private:

	std::complex<float> centroid_;
	std::vector<std::complex<float>> fourier_coefficients_[128];
	float magnitude_;
	std::vector<float> angle_;
	float rot_x_;
	float rot_y_;
	Pose instance_pose_;
	std::vector<std::vector<int>> contour_pts_;
	//std::vector<int[2]> cntr_pts_;
	real_2d_array contour_points_raw_;
	std::vector<double> y_points_resampled_;
	std::vector<double> x_points_resampled_;
	pspline2interpolant contour_spline_;
	int sz_;
	void get_contour_points(cv::Mat img);

};