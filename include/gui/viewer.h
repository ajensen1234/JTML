#ifndef VIEWER_H
#define VIEWER_H

#pragma once
#include <vector>

#include <vtkRenderWindow.h>
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
#include <opencv2/core.hpp>

#include "core/frame.h"
#include "core/calibration.h"
#include "core/model.h"

class viewer
{
public:
    viewer();
    ~viewer();

    void initialize_vtk_pointers();
    void initialize_vtk_mappers();
    void load_render_window(vtkSmartPointer<vtkRenderWindow> in);
    void initialize_vtk_renderers();
    vtkSmartPointer<vtkRenderer> get_renderer();
    vtkSmartPointer<vtkActor> get_actor_image();
    vtkSmartPointer<vtkImageData> get_current_background();
    vtkSmartPointer<vtkSTLReader> get_stl_reader();
    vtkSmartPointer<vtkDataSetMapper> get_image_mapper();
    vtkSmartPointer<vtkTextActor> get_actor_text();
    vtkSmartPointer<vtkImageImport> get_importer();

    void updateDisplayBackground(cv::Mat desiredBackground);
    void setImporterOutputToBackground();
    void updateDisplayBackgroundtoEdgeImage(int frame_number, bool CameraASelected);
    void updateDisplayBackgroundtoDilationImage(int frame_number, bool CameraASelected);
    void updateDisplayBackgroundtoOriginalImage(int frame_number, bool CameraASelected);
    void updateDisplayBackgroundtoInvertedImage(int frame_number, bool CameraASelected);
    void setLoadedFrames(std::vector<Frame>& frames);
    void setLoadedFrames_B(std::vector<Frame>& frames);
    void setLoadedModels(std::vector<Model> models);
    void setLoadedModels_B(std::vector<Model> models);
    void setupCameraCalibration(Calibration calibration);
    void placeImageActorsAccordingToCalibration(Calibration calibration, int img_w, int img_h);
    void load3DModelsIntoActorAndMapperList();
    void loadModelActorsAndMappersWith3DData();
    std::vector<vtkSmartPointer<vtkPolyDataMapper>> getModelMapperList();
    std::vector<vtkSmartPointer<vtkActor>> getModelActorList();
    void set3DModelColor(int index, double RGB[3]);
    void loadModels(QStringList cad_files, QStringList cad_models);
    bool areModelsLoadedCorrectly(int index);
    bool areModelsLoadedIncorrectly(int index);
    void changeModelOpacityToOriginal(int index);
    void changeModelOpacityToWireFrame(int index);
    void changeModelOpacityToSolid(int index);
    void changeModelOpacityToTransparent(int index);
    void setModelPositionAtIndex(int index, double x, double y, double z);
    void setModelOrientationAtIndex(int index, double xrot, double yrot, double zrot);
    std::string printLocationAndOrientationOfModelAtIndex(int index);
    void setActorText(std::string desired_text);
    void setActorTextColorToModelColorAtIndex(int index);
    void renderScene();
    void displayActorsInRenderer();

    void setRenderWindowAndDisplay();
    

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
    vtkSmartPointer<vtkCamera> my_image_camera_;
    vtkSmartPointer<vtkCamera> my_model_camera_;
    vtkSmartPointer<vtkRenderWindow> render_window_;
    
    bool initialized_pointers_;

    std::vector<Frame> loaded_frames_;
    std::vector<Frame> loaded_frames_B_;
   // std::vector<Model> loaded_models_;
    std::vector<Model> loaded_models_b;

    std::shared_ptr<std::vector<Model>> loaded_models_ = std::make_shared<std::vector<Model>>();
    

};

#endif