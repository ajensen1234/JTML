#include "gui/viewer.h"

Viewer::Viewer() {
	initialize_vtk_pointers();
	initialize_vtk_mappers();
	initialize_vtk_renderers();

}

Viewer::~Viewer() {
}

void Viewer::initialize_vtk_pointers() {

	//TODO: Throw a catch statement here to make sure there are no errors
	background_renderer_ = vtkSmartPointer<vtkRenderer>::New();
	actor_image_ = vtkSmartPointer<vtkActor>::New();
	current_background_ = vtkSmartPointer<vtkImageData>::New();
	stl_reader_ = vtkSmartPointer<vtkSTLReader>::New();
	image_mapper_ = vtkSmartPointer<vtkDataSetMapper>::New();
	actor_text_ = vtkSmartPointer<vtkTextActor>::New();
	importer_ = vtkSmartPointer<vtkImageImport>::New();
	actor_text_->GetTextProperty()->SetFontSize(16);
	actor_text_->GetTextProperty()->SetFontFamilyToCourier();
	actor_text_->SetPosition2(0, 0);
	actor_text_->GetTextProperty()->SetColor(214.0 / 255.0, 108.0 / 255.0, 35.0 / 255.0); //Earth Reda
	render_window_ = vtkSmartPointer<vtkRenderWindow>::New();
}

void Viewer::initialize_vtk_mappers() {
	image_mapper_->SetInputData(current_background_);
	actor_image_->SetPickable(0);
	actor_text_->SetPickable(0);
	actor_image_->SetMapper(image_mapper_);
}

void Viewer::initialize_vtk_renderers() {
	background_renderer_->AddActor(actor_image_);
	background_renderer_->AddActor2D(actor_text_);

}

void Viewer::load_render_window(vtkSmartPointer<vtkRenderWindow> in) {
	qvtk_render_window_ = in;
}

vtkSmartPointer<vtkRenderer> Viewer::get_renderer() {
	return background_renderer_;
}

int Viewer::get_number_of_model_actors() {
	return this->model_actor_list_.size();
}


vtkSmartPointer<vtkActor> Viewer::get_actor_image() {
	return actor_image_;
}

vtkSmartPointer<vtkImageData> Viewer::get_current_background() {
	return current_background_;
}

vtkSmartPointer<vtkSTLReader> Viewer::get_stl_reader() {
	return stl_reader_;
}

vtkSmartPointer<vtkDataSetMapper> Viewer::get_image_mapper() {
	return image_mapper_;
}

vtkSmartPointer<vtkTextActor> Viewer::get_actor_text() {
	return actor_text_;
}

vtkSmartPointer<vtkImageImport> Viewer::get_importer() {
	return importer_;
}

void Viewer::update_display_background(cv::Mat desiredBackground) {
	if (current_background_) {
		this->set_importer_output_to_background();
	}
	importer_->SetDataSpacing(1, 1, 1);
	importer_->SetDataOrigin(0, 0, 0);
	importer_->SetWholeExtent(0, desiredBackground.size().width - 1, 0, desiredBackground.size().height - 1, 0, 0);
	importer_->SetDataExtentToWholeExtent();
	importer_->SetDataScalarTypeToUnsignedChar();
	importer_->SetNumberOfScalarComponents(desiredBackground.channels());
	importer_->SetImportVoidPointer(desiredBackground.data);
	importer_->Modified();
	importer_->Update();
}

void Viewer::set_importer_output_to_background() {
	importer_->SetOutput(current_background_);
}

void Viewer::make_image_invisible() {
	actor_image_->SetVisibility(false);
}


void Viewer::set_loaded_frames(std::vector<Frame>& frames) {
	loaded_frames_ = frames;
}


void Viewer::set_loaded_frames_b(std::vector<Frame>& frames) {
	loaded_frames_B_ = frames;
}

void Viewer::update_display_background_to_edge_image(int frame_number, bool CameraASelected) {
	if (CameraASelected) {
		update_display_background(loaded_frames_[frame_number].GetEdgeImage());
	}
	else if (!CameraASelected) {
		update_display_background(loaded_frames_B_[frame_number].GetEdgeImage());
	}
}

void Viewer::update_display_background_to_original_image(int frame_number, bool CameraASelected) {
	if (CameraASelected) {
		update_display_background(loaded_frames_[frame_number].GetOriginalImage());
	}
	else if (!CameraASelected) {
		update_display_background(loaded_frames_B_[frame_number].GetOriginalImage());
	}
}


void Viewer::update_display_background_to_dilation_image(int frame_number, bool CameraASelected) {
	if (CameraASelected) {
		update_display_background(loaded_frames_[frame_number].GetDilationImage());
	}
	else if (!CameraASelected) {
		update_display_background(loaded_frames_B_[frame_number].GetDilationImage());
	}
}

void Viewer::update_display_background_to_inverted_image(int frame_number, bool CameraASelected) {
	if (CameraASelected) {
		update_display_background(loaded_frames_[frame_number].GetInvertedImage());
	}
	else if (!CameraASelected) {
		update_display_background(loaded_frames_B_[frame_number].GetInvertedImage());
	}
}

void Viewer::setup_camera_calibration(Calibration calibration) {
	background_renderer_->GetActiveCamera()->SetFocalPoint(0, 0, -1);
	background_renderer_->GetActiveCamera()->SetPosition(0, 0, 0);
	background_renderer_->GetActiveCamera()->SetClippingRange(0.1, 2.0 *
	                                                          calibration.camera_A_principal_.principal_distance_ /
	                                                          calibration.camera_A_principal_.pixel_pitch_);
}

void Viewer::place_image_actors_according_to_calibration(Calibration cal, int img_w, int img_h) {
	actor_image_->SetPosition(
		-0.5 * img_w + cal.camera_A_principal_.principal_x_ / cal.camera_A_principal_.pixel_pitch_,
		-0.5 * img_h + cal.camera_A_principal_.principal_y_ / cal.camera_A_principal_.pixel_pitch_,
		-1 * cal.camera_A_principal_.principal_distance_ / cal.camera_A_principal_.pixel_pitch_);
}


void Viewer::load_3d_models_into_actor_and_mapper_list() {
	for (int i = 0; i < loaded_models_->size(); i++) {
		std::cout << loaded_models_->at(i).model_name_ << std::endl;
		vtkSmartPointer<vtkActor> new_actor = vtkSmartPointer<vtkActor>::New();
		vtkSmartPointer<vtkPolyDataMapper> new_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
		new_mapper->SetInputConnection(loaded_models_->at(i).cad_reader_->GetOutputPort());
		new_actor->SetMapper(new_mapper);
		model_actor_list_.push_back(new_actor);
		model_mapper_list_.push_back(new_mapper);
		background_renderer_->AddActor(new_actor);
	}
}

void Viewer::load_model_actors_and_mappers_with_3d_data() {
	for (int i = 0; i < model_actor_list_.size(); i++) {
		model_actor_list_[i]->PickableOff();
		model_actor_list_[i]->VisibilityOff();
	}
}

std::vector<vtkSmartPointer<vtkPolyDataMapper>> Viewer::get_model_mapper_list() {
	return model_mapper_list_;
}

std::vector<vtkSmartPointer<vtkActor>> Viewer::get_model_actor_list() {
	return model_actor_list_;
}

void Viewer::set_3d_model_color(int index, double RGB[3]) {
	model_actor_list_[index]->GetProperty()->SetColor(RGB[0] / 255.0, RGB[1] / 255.0, RGB[2] / 255.0);
	std::cout << "Set 3D Idx: " << std::to_string(index) << std::endl;

}

void Viewer::load_models(QStringList cad_files, QStringList cad_models) {
	for (int i = 0; i < cad_files.size(); i++) {
		loaded_models_->push_back(Model(cad_files[i].toStdString(), cad_models[i].toStdString(), "BLANK"));
	}
}

bool Viewer::are_models_loaded_correctly(int index) {
	return loaded_models_->at(index).initialized_correctly_;
}

bool Viewer::are_models_loaded_incorrectly(int index) {
	return !loaded_models_->at(index).initialized_correctly_;
}

void Viewer::change_model_opacity_to_original(int index) {
	model_actor_list_[index]->PickableOn();
	model_actor_list_[index]->VisibilityOn();
	model_actor_list_[index]->GetProperty()->SetRepresentationToSurface();
	model_actor_list_[index]->GetProperty()->SetAmbient(0);
	model_actor_list_[index]->GetProperty()->SetDiffuse(1);
	model_actor_list_[index]->GetProperty()->SetOpacity(1);
}

void Viewer::change_model_opacity_to_wire_frame(int index) {
	model_actor_list_[index]->PickableOn();
	model_actor_list_[index]->VisibilityOn();
	model_actor_list_[index]->GetProperty()->SetAmbient(0);
	model_actor_list_[index]->GetProperty()->SetDiffuse(1);
	model_actor_list_[index]->GetProperty()->SetRepresentationToWireframe();
	model_actor_list_[index]->GetProperty()->SetOpacity(1);
}

void Viewer::change_model_opacity_to_transparent(int index) {
	model_actor_list_[index]->PickableOn();
	model_actor_list_[index]->VisibilityOn();
	model_actor_list_[index]->GetProperty()->SetRepresentationToSurface();
	model_actor_list_[index]->GetProperty()->SetAmbient(1);
	model_actor_list_[index]->GetProperty()->SetDiffuse(0);
	model_actor_list_[index]->GetProperty()->SetOpacity(0.2);
}

void Viewer::change_model_opacity_to_solid(int index) {
	model_actor_list_[index]->PickableOn();
	model_actor_list_[index]->VisibilityOn();
	model_actor_list_[index]->GetProperty()->SetRepresentationToSurface();
	model_actor_list_[index]->GetProperty()->SetAmbient(1);
	model_actor_list_[index]->GetProperty()->SetDiffuse(0);
	model_actor_list_[index]->GetProperty()->SetOpacity(1);
}

void Viewer::set_model_position_at_index(int index, double x, double y, double z) {
	model_actor_list_[index]->SetPosition(x, y, z);
}

void Viewer::set_model_orientation_at_index(int index, double xrot, double yrot, double zrot) {
	model_actor_list_[index]->SetOrientation(xrot, yrot, zrot);
}

std::string Viewer::print_location_and_orientation_of_model_at_index(int index) {
	std::string infoText = "Location: <";
	infoText += std::to_string(static_cast<long double>(model_actor_list_[index]->GetPosition()[0])) + ","
		+ std::to_string(static_cast<long double>(model_actor_list_[index]->GetPosition()[1])) + ","
		+ std::to_string(static_cast<long double>(model_actor_list_[index]->GetPosition()[2])) + ">\nOrientation: <"
		+ std::to_string(static_cast<long double>(model_actor_list_[index]->GetOrientation()[0])) + ","
		+ std::to_string(static_cast<long double>(model_actor_list_[index]->GetOrientation()[1])) + ","
		+ std::to_string(static_cast<long double>(model_actor_list_[index]->GetOrientation()[2])) + ">";

	return infoText;
}

void Viewer::set_actor_text(std::string desired_text) {
	actor_text_->SetInput(desired_text.c_str());
}

void Viewer::set_actor_text_color_to_model_color_at_index(int index) {
	actor_text_->GetTextProperty()->SetColor(model_actor_list_[index]->GetProperty()->GetColor());
}


void Viewer::render_scene() {
	background_renderer_->Render();

}

void Viewer::display_actors_in_renderer() {
	background_renderer_->GetActors()->Print(std::cout);
}

void Viewer::set_render_window_and_display() {
	render_window_->AddRenderer(background_renderer_);
	render_window_->SetWindowName("My Window");
	render_window_->Render();
	vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	interactor->SetRenderWindow(render_window_);
	render_window_->Render();
	interactor->Start();
}

double* Viewer::get_model_orientation_at_index(int index) {
	return model_actor_list_[index]->GetOrientation();

}

double* Viewer::get_model_position_at_index(int index) {
	return model_actor_list_[index]->GetPosition();
}


void Viewer::make_model_invisible_and_nonpickable_at_index(int index) {
	model_actor_list_[index]->PickableOff();
	model_actor_list_[index]->VisibilityOff();
}

void Viewer::make_model_visible_and_pickable_at_index(int index) {
	model_actor_list_[index]->PickableOn();
	model_actor_list_[index]->VisibilityOn();
}
void Viewer::make_all_models_invisible() {
	for (auto model : model_actor_list_) {
		model->PickableOff();
		model->VisibilityOff();
	}
}


std::shared_ptr<std::vector<Model>> Viewer::get_loaded_models() {
	return loaded_models_;
}
