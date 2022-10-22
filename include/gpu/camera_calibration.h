#ifndef CAMERA_CALIBRATION_H
#define CAMERA_CALIBRATION_H

#include <iostream>

struct CameraCalibration {
	CameraCalibration(float principal_distance, float principal_x, float principal_y, float pixel_pitch) {
		principal_distance_ = principal_distance;
		principal_x_ = principal_x;
		principal_y_ = principal_y;
		pixel_pitch_ = pixel_pitch;
		fx_ = principal_distance_ / pixel_pitch_;
		fy_ = principal_distance_ / pixel_pitch_;
		cx_ = principal_x_ / pixel_pitch_;
		cy_ = principal_y_ / pixel_pitch_;

		camera_matrix_[0] = fx_;
		camera_matrix_[1] = 0;
		camera_matrix_[2] = cx_;
		camera_matrix_[3] = 0;
		camera_matrix_[4] = fy_;
		camera_matrix_[5] = cy_;
		camera_matrix_[6] = 0;
		camera_matrix_[7] = 0;
		camera_matrix_[8] = 1;


	};
	/**
	 * @brief Constructor that accepts the parameters for the standard camera matrix.
	 * @param fx X focal length [px].
	 * @param sc Scale factor[unitless].
	 * @param cx X principal point[px].
	 * @param fy Y focal length[px].
	 * @param cy Y principal point[px].
	 */
	CameraCalibration(float fx, float sc, float cx, float fy, float cy){
		fx_ = fx;
		fy_ = fy;
		cx_ = cx;
		cy_ = cy;
		pixel_pitch_ = 0.375; // TODO: placeholder pixel pitch for denver
		principal_distance_ = fx_ * pixel_pitch_;
		principal_x_ = (cx_ - 512) * pixel_pitch_;
		principal_y_ = (512 - cy_)* pixel_pitch_;
		camera_matrix_[0] = fx_;
		camera_matrix_[1] = sc;
		camera_matrix_[2] = cx_;
		camera_matrix_[3] = 0;
		camera_matrix_[4] = fy_;
		camera_matrix_[5] = cy_;
		camera_matrix_[6] = 0;
		camera_matrix_[7] = 0;
		camera_matrix_[8] = 1;
	}
	CameraCalibration() {
		principal_distance_ = 0;
		principal_x_ = 0;
		principal_y_ = 0;
		pixel_pitch_ = 0;
	}
	/*Camera Location & Calibration Locations ~ Right Hand Axis System (Positive Z Towards Oneself)*/
	float principal_distance_; /* (mm) */
	float principal_x_; /* (mm) */
	float principal_y_; /* (mm) */
	float pixel_pitch_; /* pixel size in mm (mm/pixel) */

	float camera_matrix_[9];

	float fx() {
		return fx_;
	}
	float fy() {
		return fy_;
	}
	float cx() {
		return cx_;
	}
	float cy() {
		return cy_;
	}

	void print_values() {
		for (auto i : camera_matrix_) {
			std::cout << i << std::endl;
		}
		std::cout << fx() << std::endl;
		std::cout << fy() << std::endl;
		std::cout << cx() << std::endl;
		std::cout << cy() << std::endl;
	}

private:
	float fx_, fy_, cx_, cy_;

};

#endif /* CAMERA_CALIBRATION_H */