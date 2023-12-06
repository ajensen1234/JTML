/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#pragma once
#include <vector>
#include "nfd_instance.h"
#include "nfd.h"
#include <array>
#include "gpu/gpu_model.cuh"

/*
This NFD library is going to store a vector of each of the NFD instances at a given pose, as well as some metadata
about the overall scope of the library (rotation parameters and ranges, etc).
*/
class nfd_library
{
public:
	nfd_library(
		Calibration cal_file,
		GPUModel &gpu_model,
		int x_range,
		int y_range,
		float x_inc,
		float y_inc
	);
	~nfd_library();
	std::vector<nfd_instance> get_library();

	void create_nfd_library();

private:
	int x_range_;
	int y_range_;
	float x_inc_;
	float y_inc_;
	GPUModel* gpu_mod;
	std::vector<std::array<float, 2>> rot_indices_;
	void create_rot_indices(int x_range, int y_range, float x_inc, float y_inc);

	std::vector<nfd_instance> library_;
	Calibration calibration_;

	nfd_instance testing_projection();
	
};

