#include "gui/mainscreen.h"
#include "gui/calibration_manager.h"
#include <QMessageBox>
#include <QFileDialog>
#include <QTextStream>

MainScreen::MainScreen(QWidget* parent)
    : QMainWindow(parent) 
{
    ui.setupUi(this);
    
    // Initialize calibration manager
    calibration_manager_ = std::make_unique<CalibrationManager>();
    connect(calibration_manager_.get(), &CalibrationManager::calibrationLoaded,
            this, &MainScreen::onCalibrationLoaded);
    connect(calibration_manager_.get(), &CalibrationManager::calibrationError,
            this, &MainScreen::onCalibrationError);

    this->start_time = -1;
    sym_trap_running = false;
}

MainScreen::~MainScreen() {
    // Smart pointers will handle cleanup automatically
}

void MainScreen::onCalibrationLoaded() {
    const auto& cal = calibration_manager_->getCurrentCalibration();
    
    if (calibration_manager_->isMonoplaneCalibrated()) {
        setupMonoplaneView();
    } else if (calibration_manager_->isBiplaneCalibrated()) {
        setupBiplaneView();
    }
    
    calibration_file_ = cal; // Keep the old member updated for now
    calibrated_for_monoplane_viewport_ = calibration_manager_->isMonoplaneCalibrated();
    calibrated_for_biplane_viewport_ = calibration_manager_->isBiplaneCalibrated();
}

void MainScreen::onCalibrationError(const QString& message) {
    QMessageBox::critical(this, "Error!", message, QMessageBox::Ok);
}

void MainScreen::on_load_calibration_button_clicked() {
    QMessageBox msgBox;
    msgBox.setText("Select calibration type:");
    msgBox.addButton("UF (Single File)", QMessageBox::AcceptRole);
    msgBox.addButton("Denver (Two Files)", QMessageBox::AcceptRole);
    msgBox.addButton("Cancel", QMessageBox::RejectRole);

    int ret = msgBox.exec();
    if (ret == 2) return; // Cancel

    if (ret == 0) { // UF calibration
        QString path = QFileDialog::getOpenFileName(
            this, tr("Load UF Calibration"), ".", tr("Calibration File (*.txt)"));
        if (!path.isEmpty()) {
            calibration_manager_->loadCalibration(path);
        }
    } else { // Denver calibration
        QString cal1 = QFileDialog::getOpenFileName(
            this, tr("Load Camera 1 Calibration"), ".", tr("Calibration File (*.txt)"));
        if (cal1.isEmpty()) return;
        
        QString cal2 = QFileDialog::getOpenFileName(
            this, tr("Load Camera 2 Calibration"), ".", tr("Calibration File (*.txt)"));
        if (!cal2.isEmpty()) {
            calibration_manager_->loadDenverCalibration(cal1, cal2);
        }
    }
}

void MainScreen::setupMonoplaneView() {
    const auto& cal = calibration_manager_->getCurrentCalibration();
    
    vw->setup_camera_calibration(cal);
    coronal_vw->setup_camera_calibration(cal);
    coronal_vw->setup_camera_coronal_plane();
    
    ui.camera_A_radio_button->setChecked(true);
    ui.camera_A_radio_button->setDisabled(true);
    ui.camera_B_radio_button->setDisabled(true);
    
    if (!loaded_frames.empty()) {
        auto currentFrame = ui.image_list_widget->currentIndex().row();
        auto& frame = loaded_frames[currentFrame];
        
        vw->place_image_actors_according_to_calibration(
            cal,
            frame.GetOriginalImage().cols,
            frame.GetOriginalImage().rows
        );
    }
    
    updateCalibrationButtons();
}

void MainScreen::setupBiplaneView() {
    const auto& cal = calibration_manager_->getCurrentCalibration();
    
    vw->setup_camera_calibration(cal);
    coronal_vw->setup_camera_calibration(cal);
    
    ui.camera_A_radio_button->setChecked(true);
    ui.camera_A_radio_button->setEnabled(true);
    ui.camera_B_radio_button->setEnabled(true);
    
    if (!loaded_frames.empty()) {
        auto currentFrame = ui.image_list_widget->currentIndex().row();
        auto& frame = loaded_frames[currentFrame];
        
        vw->place_image_actors_according_to_calibration(
            cal.camera_A_principal_,
            frame.GetOriginalImage().rows,
            frame.GetOriginalImage().cols
        );
    }
    
    updateCalibrationButtons();
}

void MainScreen::updateCameraView(bool isViewA) {
    if (loaded_frames.empty()) return;
    
    const auto& cal = calibration_manager_->getCurrentCalibration();
    auto currentFrame = ui.image_list_widget->currentIndex().row();
    auto& frame = isViewA ? loaded_frames[currentFrame] : loaded_frames_B[currentFrame];
        
    vw->place_image_actors_according_to_calibration(
        isViewA ? cal.camera_A_principal_ : cal.camera_B_principal_,
        frame.GetOriginalImage().rows,
        frame.GetOriginalImage().cols
    );
    
    updateImageDisplay();
}

void MainScreen::updateCalibrationButtons() {
    ui.load_calibration_button->setDisabled(true);
    ui.actionReset_View->setDisabled(false);
    ui.actionReset_Normal_Up->setDisabled(false);
    ui.actionModel_Interaction_Mode->setDisabled(false);
    ui.actionCamera_Interaction_Mode->setDisabled(false);
}

void MainScreen::updateImageDisplay() {
    if (!hasLoadedImages()) return;
    
    if (ui.original_image_radio_button->isChecked()) {
        on_original_image_radio_button_clicked();
    } else if (ui.inverted_image_radio_button->isChecked()) {
        on_inverted_image_radio_button_clicked();
    } else if (ui.edges_image_radio_button->isChecked()) {
        on_edges_image_radio_button_clicked();
    } else if (ui.dilation_image_radio_button->isChecked()) {
        on_dilation_image_radio_button_clicked();
    }
}