#ifndef VIEWER_H
#define VIEWER_H

#pragma once
#include <vector>

#include <vtkRenderWindow.h>
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

class viewer
{
public:
    viewer();
    ~viewer();
    int my_viewer_number;

    void initialize_vtk_pointers();
    void initialize_vtk_mappers();
    void load_render_window(vtkSmartPointer<vtkRenderWindow> in);
    void initialize_vtk_renderers();
    vtkSmartPointer<vtkRenderer> get_renderer();
    vtkSmartPointer<vtkActor> get_actor_image();
    

private:

	std::vector<vtkSmartPointer<vtkActor>> model_actor_list_;
	std::vector<vtkSmartPointer<vtkPolyDataMapper>> model_mapper_list_;
	vtkSmartPointer<vtkRenderer> background_renderer_;
	vtkSmartPointer<vtkImageData> current_background_;
	vtkSmartPointer<vtkSTLReader> stl_reader_;
	vtkSmartPointer<vtkDataSetMapper> image_mapper_;
	vtkSmartPointer<vtkActor> actor_image_;
	vtkSmartPointer<vtkTextActor> actor_text_;
	vtkSmartPointer<vtkImageImport> importer_;
    vtkSmartPointer<vtkRenderWindow> qvtk_render_window_;
    
    bool initialized_pointers_;

};

#endif