/*About Window Header*/
#include "gui/drr_tool.h"

/*Interactor*/
#include "core/drr_interactor.h"

/*Message Box*/
#include <qmessagebox.h>

/*OpenCV 3.1 Library*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*Global Interactor Variable*/
vtkSmartPointer<DRRInteractorStyle> drr_interactor;

//About JTA Popup CPP
DRRTool::DRRTool(Model model, CameraCalibration calibration, double model_z_plane, QWidget *parent, Qt::WindowFlags flags)
	: QDialog(parent, flags)
{
	ui.setupUi(this);

	/*Set up VTK*/
	vtkObject::GlobalWarningDisplayOff(); /*Turn off error display*/
	renderer_ = vtkSmartPointer<vtkRenderer>::New();
	drr_interactor = vtkSmartPointer<DRRInteractorStyle>::New();
	drr_interactor->initialize_DRRTool(this);
	ui.qvtkWidget->GetRenderWindow()->AddRenderer(renderer_);
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

	/*Store Camera Values*/
	principal_distance_ = calibration.principal_distance_;
	pixel_pitch_ = calibration.pixel_pitch_;
	principal_y_ = calibration.principal_y_;
	calibration_ = calibration;

	/*Update Camera*/
	renderer_->GetActiveCamera()->SetFocalPoint(0, 0,
		-1 * principal_distance_ / pixel_pitch_);
	renderer_->GetActiveCamera()->SetPosition(0, 0, 0);
	renderer_->GetActiveCamera()->SetClippingRange(.1, 2.0 * principal_distance_ / pixel_pitch_);
	double y = ui.qvtkWidget->height()*pixel_pitch_ / 2.0 + abs(principal_y_);
	renderer_->GetActiveCamera()->SetViewAngle(180.0 / 3.1415926535897932384626433832795028841971693993751 * 2.0 *
		atan2(y, principal_distance_));

	/*GPU*/
	gpu_model_ = new gpu_cost_function::GPUModel(model_.model_name_, true,
		ui.qvtkWidget->width(), ui.qvtkWidget->height(), 0, true,
		&model_.triangle_vertices_[0],
		&model_.triangle_normals_[0],
		model_.triangle_vertices_.size() / 9,
		calibration_);
	if (!gpu_model_->IsInitializedCorrectly()) {

		QMessageBox::critical(this, "Error!", "Error initializing DRR GPU Model.", QMessageBox::Ok);
		this->close();
		
	}

	/*Interactor*/
	drr_interactor->AutoAdjustCameraClippingRangeOff();
	ui.qvtkWidget->GetRenderWindow()->GetInteractor()->SetInteractorStyle(drr_interactor);

	/*Initialize Local Image Memory*/
	host_image_ = (unsigned char*)malloc(ui.drr_image_label->height()*ui.drr_image_label->width() * sizeof(unsigned char));

	/*Initialize QT Host Image*/
	qt_host_image_ = QImage::QImage(host_image_,
		ui.drr_image_label->width(), ui.drr_image_label->height(), static_cast<int>(ui.drr_image_label->width()),
		QImage::Format_Grayscale8);

	/*Draw DRR*/
	ui.minValue->setText(QString::number(((ui.minUpperSpinBox->value() - ui.minLowerSpinBox->value())*((double)ui.minSlider->value() / 1000.0) + ui.minLowerSpinBox->value())));
	ui.maxValue->setText(QString::number(((ui.maxUpperSpinBox->value() - ui.maxLowerSpinBox->value())*((double)ui.maxSlider->value() / 1000.0) + ui.maxLowerSpinBox->value())));
	DrawDRR();

	/*Update*/
	ui.qvtkWidget->update();

}

DRRTool::~DRRTool()
{
	/*Delete GPU Models*/
	delete gpu_model_;

	/*Delete Host Image*/
	free(host_image_);
}

void DRRTool::DrawDRR() {
	/*DRR Render*/	
	gpu_model_->RenderDRRPrimaryCamera(gpu_cost_function::Pose(
		actor_->GetPosition()[0], actor_->GetPosition()[1], actor_->GetPosition()[2],
		actor_->GetOrientation()[0], actor_->GetOrientation()[1], actor_->GetOrientation()[2]),
		(ui.minUpperSpinBox->value() - ui.minLowerSpinBox->value())*((double)ui.minSlider->value() / 1000.0) + ui.minLowerSpinBox->value(),
		(ui.maxUpperSpinBox->value() - ui.maxLowerSpinBox->value())*((double)ui.maxSlider->value() / 1000.0) + ui.maxLowerSpinBox->value());

	/*Download DRR Render to Host*/
	cudaMemcpy(host_image_, gpu_model_->GetPrimaryCameraRenderedImagePointer(), ui.drr_image_label->height() * ui.drr_image_label->width() * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	/*Connect Host Image Pixmap to Label*/
	ui.drr_image_label->setPixmap(QPixmap::fromImage(qt_host_image_.mirrored(false, true)));

	/*Update QVTK*/
	ui.qvtkWidget->update();
}

/*Threshold Changes*/
void DRRTool::on_minLowerSpinBox_valueChanged() {
	if (ui.minLowerSpinBox->value() >= ui.minUpperSpinBox->value())
		ui.minUpperSpinBox->setValue(ui.minLowerSpinBox->value());
	ui.minValue->setText(QString::number(((ui.minUpperSpinBox->value() - ui.minLowerSpinBox->value())*((double)ui.minSlider->value() / 1000.0) + ui.minLowerSpinBox->value())));
	ui.maxValue->setText(QString::number(((ui.maxUpperSpinBox->value() - ui.maxLowerSpinBox->value())*((double)ui.maxSlider->value() / 1000.0) + ui.maxLowerSpinBox->value())));
	DrawDRR();
};
void DRRTool::on_maxLowerSpinBox_valueChanged() {
	if (ui.maxLowerSpinBox->value() >= ui.maxUpperSpinBox->value())
		ui.maxUpperSpinBox->setValue(ui.maxLowerSpinBox->value());
	if (ui.minUpperSpinBox->value() >= ui.maxLowerSpinBox->value())
		ui.minUpperSpinBox->setValue(ui.maxLowerSpinBox->value());
	ui.minValue->setText(QString::number(((ui.minUpperSpinBox->value() - ui.minLowerSpinBox->value())*((double)ui.minSlider->value() / 1000.0) + ui.minLowerSpinBox->value())));
	ui.maxValue->setText(QString::number(((ui.maxUpperSpinBox->value() - ui.maxLowerSpinBox->value())*((double)ui.maxSlider->value() / 1000.0) + ui.maxLowerSpinBox->value())));
	DrawDRR();
};
void DRRTool::on_minUpperSpinBox_valueChanged() {
	if (ui.minLowerSpinBox->value() >= ui.minUpperSpinBox->value())
		ui.minLowerSpinBox->setValue(ui.minUpperSpinBox->value());
	if (ui.minUpperSpinBox->value() >= ui.maxLowerSpinBox->value())
		ui.maxLowerSpinBox->setValue(ui.minUpperSpinBox->value());

	ui.minValue->setText(QString::number(((ui.minUpperSpinBox->value() - ui.minLowerSpinBox->value())*((double)ui.minSlider->value() / 1000.0) + ui.minLowerSpinBox->value())));
	ui.maxValue->setText(QString::number(((ui.maxUpperSpinBox->value() - ui.maxLowerSpinBox->value())*((double)ui.maxSlider->value() / 1000.0) + ui.maxLowerSpinBox->value())));
	DrawDRR();
};
void DRRTool::on_maxUpperSpinBox_valueChanged() {
	if (ui.maxLowerSpinBox->value() >= ui.maxUpperSpinBox->value())
		ui.maxLowerSpinBox->setValue(ui.maxUpperSpinBox->value());
	ui.minValue->setText(QString::number(((ui.minUpperSpinBox->value() - ui.minLowerSpinBox->value())*((double)ui.minSlider->value() / 1000.0) + ui.minLowerSpinBox->value())));
	ui.maxValue->setText(QString::number(((ui.maxUpperSpinBox->value() - ui.maxLowerSpinBox->value())*((double)ui.maxSlider->value() / 1000.0) + ui.maxLowerSpinBox->value())));
	DrawDRR();
};
void DRRTool::on_minSlider_valueChanged() {
	ui.minValue->setText(QString::number(((ui.minUpperSpinBox->value() - ui.minLowerSpinBox->value())*((double)ui.minSlider->value() / 1000.0) + ui.minLowerSpinBox->value())));
	ui.maxValue->setText(QString::number(((ui.maxUpperSpinBox->value() - ui.maxLowerSpinBox->value())*((double)ui.maxSlider->value() / 1000.0) + ui.maxLowerSpinBox->value())));
	DrawDRR();
};
void DRRTool::on_maxSlider_valueChanged() {
	ui.minValue->setText(QString::number(((ui.minUpperSpinBox->value() - ui.minLowerSpinBox->value())*((double)ui.minSlider->value() / 1000.0) + ui.minLowerSpinBox->value())));
	ui.maxValue->setText(QString::number(((ui.maxUpperSpinBox->value() - ui.maxLowerSpinBox->value())*((double)ui.maxSlider->value() / 1000.0) + ui.maxLowerSpinBox->value())));
	DrawDRR();
};