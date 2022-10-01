#ifndef CAMERA_CALIBRATION_H
#define CAMERA_CALIBRATION_H

struct CameraCalibration {
	CameraCalibration(float principal_distance, float principal_x, float principal_y, float pixel_pitch) {
		principal_distance_ = principal_distance;
		principal_x_ = principal_x;
		principal_y_ = principal_y;
		pixel_pitch_ = pixel_pitch;

		camera_matrix_[0] = principal_distance_ / pixel_pitch_;
		camera_matrix_[1] = 0;
		camera_matrix_[2] = principal_x_ / pixel_pitch_;
		camera_matrix_[3] = 0;
		camera_matrix_[4] = principal_distance_/pixel_pitch_;
		camera_matrix_[5] = principal_y_/pixel_pitch_;
		camera_matrix_[6] = 0;
		camera_matrix_[7] = 0;
		camera_matrix_[8] = 1;
		
	};
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
};

#endif /* CAMERA_CALIBRATION_H */