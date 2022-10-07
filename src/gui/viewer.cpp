#include "gui/viewer.h"

viewer::viewer() {
	initialize_vtk_pointers();
	initialize_vtk_mappers();
	initialize_vtk_renderers();

}

viewer::~viewer() {
}

void viewer::initialize_vtk_pointers() {

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

void viewer::initialize_vtk_mappers() {
	image_mapper_->SetInputData(current_background_);
	actor_image_->SetPickable(0);
	actor_text_->SetPickable(0);
	actor_image_->SetMapper(image_mapper_);
}

void viewer::initialize_vtk_renderers() {
	background_renderer_->AddActor(actor_image_);
	background_renderer_->AddActor2D(actor_text_);

}

void viewer::load_render_window(vtkSmartPointer<vtkRenderWindow> in) {
	qvtk_render_window_ = in;
}

vtkSmartPointer<vtkRenderer> viewer::get_renderer() {
	return background_renderer_;
}

vtkSmartPointer<vtkActor> viewer::get_actor_image() {
	return actor_image_;
}

vtkSmartPointer<vtkImageData> viewer::get_current_background() {
	return current_background_;
}

vtkSmartPointer<vtkSTLReader> viewer::get_stl_reader() {
	return stl_reader_;
}

vtkSmartPointer<vtkDataSetMapper> viewer::get_image_mapper() {
	return image_mapper_;
}

vtkSmartPointer<vtkTextActor> viewer::get_actor_text() {
	return actor_text_;
}

vtkSmartPointer<vtkImageImport> viewer::get_importer() {
	return importer_;
}

void viewer::updateDisplayBackground(cv::Mat desiredBackground) {
	if (current_background_) {
		this->setImporterOutputToBackground();
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

void viewer::setImporterOutputToBackground() {
	importer_->SetOutput(current_background_);
}

void viewer::setLoadedFrames(std::vector<Frame>& frames) {
	loaded_frames_ = frames;
}


void viewer::setLoadedFrames_B(std::vector<Frame>& frames) {
	loaded_frames_B_ = frames;
}

void viewer::updateDisplayBackgroundtoEdgeImage(int frame_number, bool CameraASelected) {
	if (CameraASelected) {
		updateDisplayBackground(loaded_frames_[frame_number].GetEdgeImage());
	}
	else if (!CameraASelected) {
		updateDisplayBackground(loaded_frames_B_[frame_number].GetEdgeImage());
	}
}

void viewer::updateDisplayBackgroundtoOriginalImage(int frame_number, bool CameraASelected) {
	if (CameraASelected) {
		updateDisplayBackground(loaded_frames_[frame_number].GetOriginalImage());
	}
	else if (!CameraASelected) {
		updateDisplayBackground(loaded_frames_B_[frame_number].GetOriginalImage());
	}
}


void viewer::updateDisplayBackgroundtoDilationImage(int frame_number, bool CameraASelected) {
	if (CameraASelected) {
		updateDisplayBackground(loaded_frames_[frame_number].GetDilationImage());
	}
	else if (!CameraASelected) {
		updateDisplayBackground(loaded_frames_B_[frame_number].GetDilationImage());
	}
}

void viewer::updateDisplayBackgroundtoInvertedImage(int frame_number, bool CameraASelected) {
	if (CameraASelected) {
		updateDisplayBackground(loaded_frames_[frame_number].GetInvertedImage());
	}
	else if (!CameraASelected) {
		updateDisplayBackground(loaded_frames_B_[frame_number].GetInvertedImage());
	}
}

void viewer::setupCameraCalibration(Calibration calibration) {
	background_renderer_->GetActiveCamera()->SetFocalPoint(0, 0, -1);
	background_renderer_->GetActiveCamera()->SetPosition(0, 0, 0);
	background_renderer_->GetActiveCamera()->SetClippingRange(0.1, 2.0 *
	                                                          calibration.camera_A_principal_.principal_distance_ /
	                                                          calibration.camera_A_principal_.pixel_pitch_);
}

void viewer::placeImageActorsAccordingToCalibration(Calibration cal, int img_w, int img_h) {
	actor_image_->SetPosition(
		-0.5 * img_w + cal.camera_A_principal_.principal_x_ / cal.camera_A_principal_.pixel_pitch_,
		-0.5 * img_h + cal.camera_A_principal_.principal_y_ / cal.camera_A_principal_.pixel_pitch_,
		-1 * cal.camera_A_principal_.principal_distance_ / cal.camera_A_principal_.pixel_pitch_);
}


void viewer::load3DModelsIntoActorAndMapperList() {
	for (int i = 0; i < loaded_models_->size(); i++) {
		vtkSmartPointer<vtkActor> new_actor = vtkSmartPointer<vtkActor>::New();
		vtkSmartPointer<vtkPolyDataMapper> new_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
		new_actor->GetProperty()->SetColor(33.0 / 255.0, 88.0 / 255.0, 170.0 / 255.0); // Go gators
		new_mapper->SetInputConnection(loaded_models_->at(i).cad_reader_->GetOutputPort());
		new_actor->SetMapper(new_mapper);
		model_actor_list_.push_back(new_actor);
		model_mapper_list_.push_back(new_mapper);
		background_renderer_->AddActor(new_actor);
	}
}

void viewer::loadModelActorsAndMappersWith3DData() {
	for (int i = 0; i < model_actor_list_.size(); i++) {
		model_actor_list_[i]->PickableOff();
		model_actor_list_[i]->VisibilityOff();
		model_actor_list_[i]->GetProperty()->SetColor(33.0 / 255.0, 88.0 / 255.0, 170.0 / 255.0);
		model_mapper_list_[i]->SetInputConnection(loaded_models_->at(i).cad_reader_->GetOutputPort());
	}
}

std::vector<vtkSmartPointer<vtkPolyDataMapper>> viewer::getModelMapperList() {
	return model_mapper_list_;
}

std::vector<vtkSmartPointer<vtkActor>> viewer::getModelActorList() {
	return model_actor_list_;
}

void viewer::set3DModelColor(int index, double RGB[3]) {
	model_actor_list_[index]->GetProperty()->SetColor(RGB[0] / 255.0, RGB[1] / 255.0, RGB[2] / 255.0);

}

void viewer::loadModels(QStringList cad_files, QStringList cad_models) {
	for (int i = 0; i < cad_files.size(); i++) {
		loaded_models_->push_back(Model(cad_files[i].toStdString(), cad_models[i].toStdString(), "BLANK"));
	}
}

bool viewer::areModelsLoadedCorrectly(int index) {
	return loaded_models_->at(index).initialized_correctly_;
}

bool viewer::areModelsLoadedIncorrectly(int index) {
	return !loaded_models_->at(index).initialized_correctly_;
}

void viewer::changeModelOpacityToOriginal(int index) {
	model_actor_list_[index]->PickableOn();
	model_actor_list_[index]->VisibilityOn();
	model_actor_list_[index]->GetProperty()->SetRepresentationToSurface();
	model_actor_list_[index]->GetProperty()->SetAmbient(0);
	model_actor_list_[index]->GetProperty()->SetDiffuse(1);
	model_actor_list_[index]->GetProperty()->SetOpacity(1);
}

void viewer::changeModelOpacityToWireFrame(int index) {
	model_actor_list_[index]->PickableOn();
	model_actor_list_[index]->VisibilityOn();
	model_actor_list_[index]->GetProperty()->SetAmbient(0);
	model_actor_list_[index]->GetProperty()->SetDiffuse(1);
	model_actor_list_[index]->GetProperty()->SetRepresentationToWireframe();
	model_actor_list_[index]->GetProperty()->SetOpacity(1);
}

void viewer::changeModelOpacityToTransparent(int index) {
	model_actor_list_[index]->PickableOn();
	model_actor_list_[index]->VisibilityOn();
	model_actor_list_[index]->GetProperty()->SetRepresentationToSurface();
	model_actor_list_[index]->GetProperty()->SetAmbient(1);
	model_actor_list_[index]->GetProperty()->SetDiffuse(0);
	model_actor_list_[index]->GetProperty()->SetOpacity(1);
}

void viewer::changeModelOpacityToSolid(int index) {
	model_actor_list_[index]->PickableOn();
	model_actor_list_[index]->VisibilityOn();
	model_actor_list_[index]->GetProperty()->SetRepresentationToSurface();
	model_actor_list_[index]->GetProperty()->SetAmbient(0);
	model_actor_list_[index]->GetProperty()->SetDiffuse(1);
	model_actor_list_[index]->GetProperty()->SetOpacity(1);
}

void viewer::setModelPositionAtIndex(int index, double x, double y, double z) {
	model_actor_list_[index]->SetPosition(x, y, z);
}

void viewer::setModelOrientationAtIndex(int index, double xrot, double yrot, double zrot) {
	model_actor_list_[index]->SetOrientation(xrot, yrot, zrot);
}

std::string viewer::printLocationAndOrientationOfModelAtIndex(int index) {
	std::string infoText = "Location: <";
	infoText += std::to_string(static_cast<long double>(model_actor_list_[index]->GetPosition()[0])) + ","
		+ std::to_string(static_cast<long double>(model_actor_list_[index]->GetPosition()[1])) + ","
		+ std::to_string(static_cast<long double>(model_actor_list_[index]->GetPosition()[2])) + ">\nOrientation: <"
		+ std::to_string(static_cast<long double>(model_actor_list_[index]->GetOrientation()[0])) + ","
		+ std::to_string(static_cast<long double>(model_actor_list_[index]->GetOrientation()[1])) + ","
		+ std::to_string(static_cast<long double>(model_actor_list_[index]->GetOrientation()[2])) + ">";

	return infoText;
}

void viewer::setActorText(std::string desired_text) {
	actor_text_->SetInput(desired_text.c_str());
}

void viewer::setActorTextColorToModelColorAtIndex(int index) {
	actor_text_->GetTextProperty()->SetColor(model_actor_list_[index]->GetProperty()->GetColor());
}


void viewer::renderScene() {
	background_renderer_->Render();

}

void viewer::displayActorsInRenderer() {
	background_renderer_->GetActors()->Print(std::cout);
}

void viewer::setRenderWindowAndDisplay() {
	render_window_->AddRenderer(background_renderer_);
	render_window_->SetWindowName("My Window");
	render_window_->Render();
	vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	interactor->SetRenderWindow(render_window_);
	render_window_->Render();
	interactor->Start();
}
