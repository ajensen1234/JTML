#ifndef CALIBRATION_H
#define CALIBRATION_H

/*Includes*/
#include "data_structures_6D.h"
#include "camera_calibration.h" //*Camera Calibration For Renderer (principal distance, principal x/y, pix pitch)

/*Vec 3*/
struct Vect_3 {
	Vect_3(float v_1, float v_2, float v_3) {
		v_1_ = v_1;
		v_2_ = v_2;
		v_3_ = v_3;
	}
	Vect_3() {
		v_1_ = 0;
		v_2_ = 0;
		v_3_ = 0;
	}

	/*Storage*/
	float v_1_; float v_2_; float v_3_;
};

/*3 by 3 Matrix*/
struct Matrix_3_3 {
	Matrix_3_3(float A_11, float A_12, float A_13,
		float A_21, float A_22, float A_23,
		float A_31, float A_32, float A_33) {
		A_11_ = A_11; A_12_ = A_12; A_13_ = A_13;
		A_21_ = A_21; A_22_ = A_22; A_23_ = A_23;
		A_31_ = A_31; A_32_ = A_32; A_33_ = A_33;
	};
	Matrix_3_3() {
		A_11_ = 0; A_12_ = 0; A_13_ = 0;
		A_21_ = 0; A_22_ = 0; A_23_ = 0;
		A_31_ = 0; A_32_ = 0; A_33_ = 0;
	}
	/*Storage*/
	float A_11_; float A_12_; float A_13_;
	float A_21_; float A_22_; float A_23_;
	float A_31_; float A_32_; float A_33_;

	/*Perform Transpose*/
	Matrix_3_3 tranpose() {
		return Matrix_3_3(
			A_11_, A_21_, A_31_,
			A_12_, A_22_, A_32_,
			A_13_, A_23_, A_33_);
	};
};

struct Calibration {

	/* Constructors for Monoplane and Biplane*/
	Calibration(CameraCalibration monoplane_principal) {
		biplane_calibration = false;
		camera_A_principal_ = monoplane_principal;
	};
	Calibration(CameraCalibration biplane_A_principal, CameraCalibration biplane_B_principal,
		Vect_3 origin_B, Matrix_3_3 axes_B) {
		biplane_calibration = true;
		camera_A_principal_ = biplane_A_principal;
		camera_B_principal_ = biplane_B_principal;
		origin_B_ = origin_B;
		axes_B_ = axes_B;
	};
	Calibration() {
		biplane_calibration = false;
	};

	/*Calibrated For Biplane?*/
	bool biplane_calibration;

	/*Storage*/
	CameraCalibration camera_A_principal_; /*used for both monoplane and biplane*/
	CameraCalibration camera_B_principal_; /*only used for biplane*/
	Vect_3 origin_B_; /*Origin of Camera B with respect o A which is assumed to be at (0,0,0) */
	Matrix_3_3 axes_B_; /*Orthogonal Coordinate System of B where A is assumed to have standard unit system*/

	/*Perform Multiplication*/
	Matrix_3_3 multiplication_mat_mat(Matrix_3_3 X, Matrix_3_3 Y) {
		return Matrix_3_3(
			X.A_11_*Y.A_11_ + X.A_12_*Y.A_21_ + X.A_13_*Y.A_31_,
			X.A_11_*Y.A_12_ + X.A_12_*Y.A_22_ + X.A_13_*Y.A_32_,
			X.A_11_*Y.A_13_ + X.A_12_*Y.A_23_ + X.A_13_*Y.A_33_,

			X.A_21_*Y.A_11_ + X.A_22_*Y.A_21_ + X.A_23_*Y.A_31_,
			X.A_21_*Y.A_12_ + X.A_22_*Y.A_22_ + X.A_23_*Y.A_32_,
			X.A_21_*Y.A_13_ + X.A_22_*Y.A_23_ + X.A_23_*Y.A_33_,

			X.A_31_*Y.A_11_ + X.A_32_*Y.A_21_ + X.A_33_*Y.A_31_,
			X.A_31_*Y.A_12_ + X.A_32_*Y.A_22_ + X.A_33_*Y.A_32_,
			X.A_31_*Y.A_13_ + X.A_32_*Y.A_23_ + X.A_33_*Y.A_33_
			);
	};
	Vect_3 multiplication_mat_vec(Matrix_3_3 X, Vect_3 u) {
		return Vect_3(
			X.A_11_*u.v_1_ + X.A_12_*u.v_2_ + X.A_13_*u.v_3_,

			X.A_21_*u.v_1_ + X.A_22_*u.v_2_ + X.A_23_*u.v_3_,

			X.A_31_*u.v_1_ + X.A_32_*u.v_2_ + X.A_33_*u.v_3_
			);
	};

	/*Camera A Pose to Camera B Pose*/
	Point6D convert_Pose_A_to_Pose_B(Point6D poseA){
		if (biplane_calibration) {
			/*Deal with Location*/
			Vect_3 location_B = multiplication_mat_vec(axes_B_.tranpose(),
				Vect_3(poseA.x - origin_B_.v_1_,
				poseA.y - origin_B_.v_2_,
				poseA.z - origin_B_.v_3_));

			/*Deal with Orientation*/
			/*Construct ROtation Matrices for A: Rz, Rx, Ry
			Then Find R = Rz*Rx*Ry
			Then Tranform as R_B = Q'*R where Q is the axes_B_ matrix
			Then recover theta_x,y, and z for camera B (may not be unique)*/
			/*Convert To Rads*/
			float PI = 3.141592653589793238462643383279502884;
			float theta_x_A = poseA.xa*(PI / 180.0);
			float theta_y_A = poseA.ya*(PI / 180.0);
			float theta_z_A = poseA.za*(PI / 180.0);
			Matrix_3_3 R_x(
				1, 0, 0,
				0, cos(theta_x_A), -1 * sin(theta_x_A),
				0, sin(theta_x_A), cos(theta_x_A));
			Matrix_3_3 R_y(
				cos(theta_y_A), 0, sin(theta_y_A),
				0, 1, 0,
				-1 * sin(theta_y_A), 0, cos(theta_y_A));
			Matrix_3_3 R_z(
				cos(theta_z_A), -1 * sin(theta_z_A), 0,
				sin(theta_z_A), cos(theta_z_A), 0,
				0, 0, 1);
			Matrix_3_3 R = multiplication_mat_mat(R_z, multiplication_mat_mat(R_x, R_y));
			Matrix_3_3 R_B = multiplication_mat_mat(axes_B_.tranpose(), R);

			/*Algorithm To Recover Z - X - Y Euler Angles*/
			float theta_x_B, theta_y_B, theta_z_B;
			if (R_B.A_32_ < 1) {
				if (R_B.A_32_ > -1) {
					theta_x_B = asin(R_B.A_32_);
					theta_z_B = atan2(-1 * R_B.A_12_, R_B.A_22_);
					theta_y_B = atan2(-1 * R_B.A_31_, R_B.A_33_);

				}
				else {
					theta_x_B = -PI / 2.0;
					theta_z_B = -1 * atan2(R_B.A_13_, R_B.A_11_);
					theta_y_B = 0;
				}
			}
			else {
				theta_x_B = PI / 2.0;
				theta_z_B = atan2(R_B.A_13_, R_B.A_11_);
				theta_y_B = 0;
			}

			/*Return New Pose*/
			return Point6D(location_B.v_1_, location_B.v_2_, location_B.v_3_,
				theta_x_B * (180.0 / PI), theta_y_B * (180.0 / PI), theta_z_B * (180.0 / PI));
		}
		else return poseA; //Just return the same.
	};

	/*Camera B Pose to Camera A Pose*/
	Point6D convert_Pose_B_to_Pose_A(Point6D poseA){
		if (biplane_calibration) {
			/*Deal with Location*/
			Vect_3 location_B = multiplication_mat_vec(axes_B_,
				Vect_3(poseA.x, poseA.y, poseA.z));
			location_B = Vect_3(location_B.v_1_ + origin_B_.v_1_,
				location_B.v_2_ + origin_B_.v_2_,
				location_B.v_3_ + origin_B_.v_3_);

			/*Deal with Orientation*/
			/*Construct ROtation Matrices for B: Rz, Rx, Ry
			Then Find R = Rz*Rx*Ry
			Then Tranform as R_B = Q'*R*Q where Q is the axes_B_ matrix
			Then recover theta_x,y, and z for camera B (may not be unique)*/
			/*Convert To Rads*/
			float PI = 3.141592653589793238462643383279502884;
			float theta_x_A = poseA.xa*(PI / 180.0);
			float theta_y_A = poseA.ya*(PI / 180.0);
			float theta_z_A = poseA.za*(PI / 180.0);
			Matrix_3_3 R_x(
				1, 0, 0,
				0, cos(theta_x_A), -1 * sin(theta_x_A),
				0, sin(theta_x_A), cos(theta_x_A));
			Matrix_3_3 R_y(
				cos(theta_y_A), 0, sin(theta_y_A),
				0, 1, 0,
				-1 * sin(theta_y_A), 0, cos(theta_y_A));
			Matrix_3_3 R_z(
				cos(theta_z_A), -1 * sin(theta_z_A), 0,
				sin(theta_z_A), cos(theta_z_A), 0,
				0, 0, 1);
			Matrix_3_3 R = multiplication_mat_mat(R_z, multiplication_mat_mat(R_x, R_y));
			Matrix_3_3 R_B = multiplication_mat_mat(axes_B_, R);

			/*Algorithm To Recover Z - X - Y Euler Angles*/
			float theta_x_B, theta_y_B, theta_z_B;
			if (R_B.A_32_ < 1) {
				if (R_B.A_32_ > -1) {
					theta_x_B = asin(R_B.A_32_);
					theta_z_B = atan2(-1 * R_B.A_12_, R_B.A_22_);
					theta_y_B = atan2(-1 * R_B.A_31_, R_B.A_33_);

				}
				else {
					theta_x_B = -PI / 2.0;
					theta_z_B = -1 * atan2(R_B.A_13_, R_B.A_11_);
					theta_y_B = 0;
				}
			}
			else {
				theta_x_B = PI / 2.0;
				theta_z_B = atan2(R_B.A_13_, R_B.A_11_);
				theta_y_B = 0;
			}

			/*Return New Pose*/
			return Point6D(location_B.v_1_, location_B.v_2_, location_B.v_3_,
				theta_x_B * (180.0 / PI), theta_y_B * (180.0 / PI), theta_z_B * (180.0 / PI));
		}
		else return poseA; //Just return the same.
	};

};
#endif /* CALIBRATION_H */