#include "gui/viewer.h"

viewer::viewer()
{
    std::cout << "I have initialized the viewer" <<std::endl;
}

viewer::~viewer()
{

}

void viewer::initialize_vtk_pointers(){

    //TODO: Throw a catch statement here to make sure there are no errors
    std::cout << "This far"<<std::endl;
    background_renderer_ = vtkSmartPointer<vtkRenderer>::New();
    std::cout << "that far (2)" <<std::endl;
    this->actor_image_ = vtkSmartPointer<vtkActor>::New();
    //this->current_background_ = vtkSmartPointer<vtkImageData>::New();
    //this->stl_reader_ = vtkSmartPointer<vtkSTLReader>::New();
    //this->image_mapper_ = vtkSmartPointer<vtkDataSetMapper>::New();
    //this->actor_text_ = vtkSmartPointer<vtkTextActor>::New();
    //this->importer_ = vtkSmartPointer<vtkImageImport>::New();
	//this->actor_text_->GetTextProperty()->SetFontSize(16);
	//this->actor_text_->GetTextProperty()->SetFontFamilyToCourier();
	//this->actor_text_->SetPosition2(0, 0);
	//this->actor_text_->GetTextProperty()->SetColor(214.0 / 255.0, 108.0 / 255.0, 35.0 / 255.0); //Earth Red
}

void viewer::initialize_vtk_mappers(){
    this->image_mapper_->SetInputData(this->current_background_);
    this->actor_image_->SetPickable(0);
    this->actor_text_->SetPickable(0);
    this->actor_image_->SetMapper(this->image_mapper_);
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