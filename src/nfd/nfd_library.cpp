// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include <nfd/nfd_library.h>

nfd_library::nfd_library(Calibration cal_file,GPUModel &gpu_model,int x_range,int y_range,float x_inc,float y_inc) {
	x_range_ = x_range;
	y_range_ = y_range;
	x_inc_ = x_inc;
	y_inc_ = y_inc;
	calibration_ = cal_file;
	gpu_mod = &gpu_model;

	create_rot_indices(x_range_, y_range_, x_inc_, y_inc_);


	/*My Testing Code - delete when you are ready*/
	nfd_instance my_inst = testing_projection();

	my_inst.print_contour_points();
	/*End testing code*/
}



void nfd_library::create_nfd_library()
{
	for (std::array<float, 2> rots : rot_indices_) {
		nfd_instance temp_instance(*gpu_mod,0,0, -0.9*this->calibration_.camera_A_principal_.principal_distance_, 
			rots[0], rots[1], 0);

	}
}

void nfd_library::create_rot_indices(int x_range, int y_range, float x_inc, float y_inc)
{
	for (float i = -x_range; i < x_range + x_inc; i = i + x_inc) {
		for (float j = -y_range; j < y_range + y_inc; j = j + y_inc) {
			std::array<float, 2> val = { i,j };
			rot_indices_.push_back(val);
		}
	}
}


nfd_instance nfd_library::testing_projection() {
	nfd_instance inst(*gpu_mod, 40, -20, -1000, 0, 0, 0);

	return inst;
}

nfd_library::~nfd_library() {
	
}