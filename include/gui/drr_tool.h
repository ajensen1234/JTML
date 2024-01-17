/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#ifndef DRR_TOOL_H
#define DRR_TOOL_H

#include <qdialog.h>

#include "ui_drr_tool.h"

/*Standard Library*/
#include <vector>

/*VTK*/
#include <vtkActor.h>
#include <vtkAlgorithm.h>
#include <vtkCamera.h>
#include <vtkInteractorStyleTrackballActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>

/*Models*/
#include "core/model.h"

/*GPU Models*/
#include "gpu/gpu_model.cuh"

// About JTA Popup Header
class DRRTool : public QDialog {
    Q_OBJECT

   public:
    DRRTool(Model model, CameraCalibration calibration, double model_z_plane,
            QWidget* parent = 0, Qt::WindowFlags flags = Qt::WindowFlags());
    ~DRRTool() override;

    /*Draw DRR*/
    void DrawDRR();

   public slots:
    /*Threshold Changes*/
    void on_minLowerSpinBox_valueChanged();
    void on_maxLowerSpinBox_valueChanged();
    void on_minUpperSpinBox_valueChanged();
    void on_maxUpperSpinBox_valueChanged();
    void on_minSlider_valueChanged();
    void on_maxSlider_valueChanged();

   private:
    Ui::drrTool ui;

    /*VTK*/
    vtkSmartPointer<vtkRenderer> renderer_;

    /*Actor and Mapper and Model (CPU and GPU*)*/
    vtkSmartPointer<vtkActor> actor_;
    vtkSmartPointer<vtkPolyDataMapper> mapper_;
    Model model_;
    gpu_cost_function::GPUModel* gpu_model_;

    /*Camera Stuff*/
    float principal_distance_;
    float pixel_pitch_;
    float principal_y_;
    CameraCalibration calibration_;

    /*Array for Storing Device Image on Host*/
    unsigned char* host_image_;

    /*QImage for Converting Host to Viewable Image*/
    QImage qt_host_image_;
};

#endif  // DRR_TOOL_H
