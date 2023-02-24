#include "gui/coronalplaneviewer.h"
/*Message Box*/
#include <qmessagebox.h>

/*OpenCV 3.1 Library*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

CoronalPlaneViewer::CoronalPlaneViewer(Model model, CameraCalibration calibration, double model_z_plane, 
        QWidget* parent = 0, Qt::WindowFlags flags = 0) :
    QDialog(parent) {
    ui.setupUi(this);

	/*Set up VTK*/
	vtkObject::GlobalWarningDisplayOff(); /*Turn off error display*/
	renderer_ = vtkSmartPointer<vtkRenderer>::New();
	ui.qvtkWidget->renderWindow()->AddRenderer(renderer_);
	actor_ = vtkSmartPointer<vtkActor>::New();
	mapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
	actor_->GetProperty()->SetColor(214.0 / 255.0, 108.0 / 255.0, 35.0 / 255.0);
	actor_->SetMapper(mapper_);
	renderer_->AddActor(actor_);
	actor_->PickableOn();
	actor_->VisibilityOn();
	model_ = model;
	mapper_->SetInputConnection(model_.cad_reader_->GetOutputPort());
	actor_->SetPosition(0, 0, model_z_plane);
	actor_->SetOrientation(0, 0, 0);
}

CoronalPlaneViewer::~CoronalPlaneViewer()
{
    delete ui;
}
