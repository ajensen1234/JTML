/*Image Info Structure for Model Training*/
#include "ImageInfo.h"

/*Standard Lib*/
#include <fstream>


ImageInfo::ImageInfo(Study study, std::string image_path, vector<std::string> model_types, vector<std::string> label_img_paths,
	vector<gpu_cost_function::Pose> pose_img_models, vector<vector<basic_la::XYPoint>> norm_KP_points_list) {
	/*Study*/
	study_ = study;

	/*Model Types*/
	model_types_ = model_types;

	/*Paths*/
	image_path_ = image_path;
	label_img_paths_ = label_img_paths; // For each model in the image (stored same order as models in study)

	/*Poses for each model in the image (stored same order as models in study)*/
	pose_img_models_ = pose_img_models;
	/*Store Vector of Vector of Normalized KPs (empty vector stored if no .kp file for model or not generating KP data) for each model (stored same order as models in study)*/
	norm_KP_points_list_ = norm_KP_points_list;
}

void ImageInfo::AppendInformation(std::string text_file_path, int model_type_index) {
	std::ofstream outfile;
	outfile.open(text_file_path, std::ios_base::app);
	outfile << "Image Path: " << image_path_ << endl;
	outfile << "Image Label Path: " << label_img_paths_[model_type_index] << endl;
	outfile << "Overall Study Name: " << study_.study_name_ << endl;
	outfile << "Patient Name: " << study_.patient_name_ << endl;
	outfile << "Session Number: " << study_.sess_num_ << endl;
	outfile << "Movement Name: " << study_.mov_name_ << endl;
	outfile << "Movement Number: " << study_.mov_num_ << endl;
	outfile << "STL Name: " << study_.stl_basenames_[model_type_index] << endl;
	outfile << "STL Type: " << study_.stl_types_[model_type_index] << endl;
	outfile << "Original Image Width: " << study_.width_ << endl;
	outfile << "Original Image Height: " << study_.height_ << endl;
	outfile << "Pose (x,y,z,za,xa,ya): " << pose_img_models_[model_type_index].x_location_ 
		<< "," << pose_img_models_[model_type_index].y_location_ << "," << pose_img_models_[model_type_index].z_location_ << ","
		<< pose_img_models_[model_type_index].z_angle_ << "," << pose_img_models_[model_type_index].x_angle_ << ","
		<< pose_img_models_[model_type_index].y_angle_ << endl << endl;

}