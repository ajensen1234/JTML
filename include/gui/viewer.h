#ifndef VIEWER_H
#define VIEWER_H

#pragma once
#include <vector>

#include <vtkRenderWindow.h>
#include <vtkTransform.h>
#include <vtkMatrix4x4.h>
#include <vtkImageActor.h>
#include <vtkProperty.h>
#include <vtkCamera.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkVersion.h>
#include <vtkSTLReader.h>
#include <vtkImageImport.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballActor.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkAutoInit.h> // Added post migration to Banks' lab computer
#include <vtkInteractorStyleTrackballCamera.h> /*Alternate Camera*/
#include <iostream>
#include <qabstractitemmodel.h>
#include <opencv2/core.hpp>
#include <QList>
#include <vtkInteractorStyleTrackballActor.h>

#include "core/frame.h"
#include "core/calibration.h"
#include "core/model.h"


class Viewer {
public:
	Viewer();
	~Viewer();

	void initialize_vtk_pointers();
	void initialize_vtk_mappers();
	void initialize_vtk_renderers();
	vtkSmartPointer<vtkRenderer> get_renderer();
	vtkSmartPointer<vtkActor> get_actor_image();
	vtkSmartPointer<vtkImageData> get_current_background();
	vtkSmartPointer<vtkSTLReader> get_stl_reader();
	vtkSmartPointer<vtkDataSetMapper> get_image_mapper();
	vtkSmartPointer<vtkTextActor> get_actor_text();
	vtkSmartPointer<vtkImageImport> get_importer();

	void update_display_background(cv::Mat desiredBackground);
	void set_importer_output_to_background();
	void set_loaded_frames(std::vector<Frame>& frames);
	void set_loaded_frames_b(std::vector<Frame>& frames);
	void setup_camera_calibration(Calibration calibration);
	void place_image_actors_according_to_calibration(Calibration calibration, int img_w, int img_h);
	void load_3d_models_into_actor_and_mapper_list();
	void load_model_actors_and_mappers_with_3d_data();
	std::vector<vtkSmartPointer<vtkPolyDataMapper>> get_model_mapper_list();
	std::vector<vtkSmartPointer<vtkActor>> get_model_actor_list();
	void set_3d_model_color(int index, double RGB[3]);
	void load_models(QStringList cad_files, QStringList cad_models);
	bool are_models_loaded_correctly(int index);
	bool are_models_loaded_incorrectly(int index);
	void change_model_opacity_to_original(int index);
	void change_model_opacity_to_wire_frame(int index);
	void change_model_opacity_to_solid(int index);
	void change_model_opacity_to_transparent(int index);
	void set_model_position_at_index(int index, double x, double y, double z);
	double* get_model_position_at_index(int index);
	double* get_model_orientation_at_index(int index);
	void set_model_orientation_at_index(int index, double xrot, double yrot, double zrot);
	std::string print_location_and_orientation_of_model_at_index(int index);
	void set_actor_text(std::string desired_text);
	void set_actor_text_color_to_model_color_at_index(int index);
	void render_scene();
	void display_actors_in_renderer();
	int get_number_of_model_actors();

	void set_render_window_and_display();
	void make_image_invisible();
	void make_model_invisible_and_nonpickable_at_index(int index);
	void make_model_visible_and_pickable_at_index(int index);
	void make_all_models_invisible();

	// Changing Background Functions
	void update_display_background_to_edge_image(int frame_number, bool CameraASelected);
	void update_display_background_to_dilation_image(int frame_number, bool CameraASelected);
	void update_display_background_to_original_image(int frame_number, bool CameraASelected);
	void update_display_background_to_inverted_image(int frame_number, bool CameraASelected);

	// Render Window Setup
	void load_renderers_into_render_window();
	void load_render_window(vtkSmartPointer<vtkRenderWindow> in);

	std::shared_ptr<std::vector<Model>> get_loaded_models();
	void print_render_window();
	void make_actor_text_invisible();
	void make_actor_text_visible();

	void load_in_interactor_style(vtkSmartPointer<vtkInteractorStyleTrackballActor> in);
	int model_actor_list_size();

	void print_interactor_information();
	vtkActor* get_model_actor_at_index(int index);
	vtkSmartPointer<vtkRenderWindowInteractor> get_interactor();

	void calculate_and_set_window_center_from_calibration(const int w, const int h, const float cx, const float cy);
	void calculate_and_set_viewing_angle_from_calibration(const int h, const int fy);
	void calculate_and_set_camera_aspect_from_calibration(const int fx, const int fy);
	void set_vtk_camera_from_calibration_and_image_size_if_jta(Calibration cal, int w, int h);
	void set_vtk_camera_from_calibration_and_image_if_camera_matrix(Calibration cal, int w, int h);


private:
	vtkSmartPointer<vtkRenderWindowInteractor> render_window_interactor_ = nullptr;
	std::vector<vtkSmartPointer<vtkActor>> model_actor_list_;
	std::vector<vtkSmartPointer<vtkPolyDataMapper>> model_mapper_list_;
	vtkSmartPointer<vtkRenderer> background_renderer_;
	vtkSmartPointer<vtkRenderer> scene_renderer_;
	vtkSmartPointer<vtkImageData> current_background_;
	vtkSmartPointer<vtkSTLReader> stl_reader_;
	vtkSmartPointer<vtkDataSetMapper> image_mapper_;
	vtkSmartPointer<vtkActor> actor_image_;
	vtkSmartPointer<vtkTextActor> actor_text_;
	vtkSmartPointer<vtkImageImport> importer_;
	vtkSmartPointer<vtkRenderWindow> qvtk_render_window_;
	vtkSmartPointer<vtkCamera> scene_camera_;
	vtkSmartPointer<vtkCamera> background_camera_;
	vtkSmartPointer<vtkRenderWindow> render_window_;


	bool initialized_pointers_;

	std::vector<Frame> loaded_frames_;
	std::vector<Frame> loaded_frames_B_;
	std::vector<Model> loaded_models_b;

	std::shared_ptr<std::vector<Model>> loaded_models_ = std::make_shared<std::vector<Model>>();





};

#endif
