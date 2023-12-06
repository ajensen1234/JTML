/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#pragma once
/*Standard Library*/
#include <string>
#include <math.h>

using namespace std;

namespace basic_la {
	struct XYPoint {
		/*Functions*/
		XYPoint(double X, double Y);
		XYPoint() {};

		/*Data*/
		double X_;
		double Y_;
	};

	struct XYZPoint {
		/*Functions*/
		XYZPoint(double X, double Y, double Z);
		XYZPoint() {};

		/*Data*/
		double X_;
		double Y_;
		double Z_;
	};

	struct RotationMatrixZXY {
		/*Functions*/
		RotationMatrixZXY(double Z_angle, double X_angle, double Y_angle);
		RotationMatrixZXY() {};
		XYZPoint RotatePoint(XYZPoint point);

		/*Data*/
		double rotation_00_; double rotation_01_; double rotation_02_;
		double rotation_10_; double rotation_11_; double rotation_12_;
		double rotation_20_; double rotation_21_; double rotation_22_;
	};
}