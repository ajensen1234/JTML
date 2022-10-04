#include "gui/viewer.h"

viewer::viewer()
{
    initialize_vtk_pointers();
    initialize_vtk_mappers();
    initialize_vtk_renderers();
    
}

viewer::~viewer()
{

}

void viewer::initialize_vtk_pointers(){

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
	actor_text_->GetTextProperty()->SetColor(214.0 / 255.0, 108.0 / 255.0, 35.0 / 255.0); //Earth Red
}

void viewer::initialize_vtk_mappers(){
    image_mapper_->SetInputData(this->current_background_);
    actor_image_->SetPickable(0);
    actor_text_->SetPickable(0);
    actor_image_->SetMapper(this->image_mapper_);
}

void viewer::initialize_vtk_renderers(){
    this->background_renderer_->AddActor(this->actor_image_);
    this->background_renderer_->AddActor2D(this->actor_text_);
}

void viewer::load_render_window(vtkSmartPointer<vtkRenderWindow> in){
    this->qvtk_render_window_ = in;
}

vtkSmartPointer<vtkRenderer> viewer::get_renderer(){
    return this->background_renderer_;
}

vtkSmartPointer<vtkActor> viewer::get_actor_image(){
    return this->actor_image_;
}

vtkSmartPointer<vtkImageData> viewer::get_current_background(){
    return this->current_background_;
}
vtkSmartPointer<vtkSTLReader> viewer::get_stl_reader(){
    return this->stl_reader_;
}
vtkSmartPointer<vtkDataSetMapper> viewer::get_image_mapper(){
    return this->image_mapper_;
}
vtkSmartPointer<vtkTextActor> viewer::get_actor_text(){
    return this->actor_text_;
}
vtkSmartPointer<vtkImageImport> viewer::get_importer(){
    return this->importer_;
}

void viewer::updateDisplayBackground(cv::Mat desiredBackground){
    if(current_background_){
        this->setImporterOutputToBackground();
    }
    importer_->SetDataSpacing(1,1,1);
    importer_->SetDataOrigin(0,0,0);
    importer_->SetWholeExtent(0,desiredBackground.size().width - 1, 0, desiredBackground.size().height -1, 0,0);
    importer_->SetDataExtentToWholeExtent();
    importer_->SetDataScalarTypeToUnsignedChar();
    importer_->SetNumberOfScalarComponents(desiredBackground.channels());
    importer_->SetImportVoidPointer(desiredBackground.data);
    importer_->Modified();
    importer_->Update();
}

void viewer::setImporterOutputToBackground(){
    importer_->SetOutput(current_background_);
}

void viewer::setLoadedFrames(std::vector<Frame>& frames){
    loaded_frames_ = frames;
}


void viewer::setLoadedFrames_B(std::vector<Frame>& frames){
    loaded_frames_B_ = frames;
}

void viewer::updateDisplayBackgroundtoEdgeImage(int frame_number,bool CameraASelected){
    if (CameraASelected){
        updateDisplayBackground(loaded_frames_[frame_number].GetEdgeImage());
    } 
    else if (!CameraASelected){
        updateDisplayBackground(loaded_frames_B_[frame_number].GetEdgeImage());
    }
}

void viewer::updateDisplayBackgroundtoOriginalImage(int frame_number,bool CameraASelected){
    if (CameraASelected){
        updateDisplayBackground(loaded_frames_[frame_number].GetOriginalImage());
    } 
    else if (!CameraASelected){
        updateDisplayBackground(loaded_frames_B_[frame_number].GetOriginalImage());
    }
}


void viewer::updateDisplayBackgroundtoDilationImage(int frame_number,bool CameraASelected){
    if (CameraASelected){
        updateDisplayBackground(loaded_frames_[frame_number].GetDilationImage());
    } 
    else if (!CameraASelected){
        updateDisplayBackground(loaded_frames_B_[frame_number].GetDilationImage());
    }
}

void viewer::updateDisplayBackgroundtoInvertedImage(int frame_number,bool CameraASelected){
    if (CameraASelected){
        updateDisplayBackground(loaded_frames_[frame_number].GetInvertedImage());
    } 
    else if (!CameraASelected){
        updateDisplayBackground(loaded_frames_B_[frame_number].GetInvertedImage());
    }
}