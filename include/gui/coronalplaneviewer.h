#ifndef CORONALPLANEVIEWER_H
#define CORONALPLANEVIEWER_H

#include <qdialog.h>
#include "ui_coronalplaneviewer.h"

/*Standard Library*/
#include <vector>

/*VTK*/
#include <vtkInteractorStyleTrackballActor.h>
#include <vtkRenderer.h>
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkCamera.h>
#include <vtkProperty.h>
#include <vtkAlgorithm.h>

/*Models*/
#include "core/model.h"

/*GPU Models*/
#include "gpu/gpu_model.cuh"

namespace Ui {
class CoronalPlaneViewer;
}

class CoronalPlaneViewer : public QDialog
{
    Q_OBJECT

public:
    CoronalPlaneViewer(Model model, CameraCalibration calibration, double model_z_plane, QWidget* parent = 0,
		Qt::WindowFlags flags = 0);
    ~CoronalPlaneViewer();

	/*Draw cpv*/
	void DrawCPV();

private:
    Ui::CoronalPlaneViewer ui;

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

#endif CORONALPLANEVIEWER_H
