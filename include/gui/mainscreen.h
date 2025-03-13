#ifndef MAINSCREEN_H
#define MAINSCREEN_H

#include <QtWidgets>
#include <QMainWindow>
#include <QFileDialog>
#include <memory>

#include "ui_mainscreen.h"
#include "gui/viewer.h"
#include "gui/calibration_manager.h"
#include "core/frame.h"
#include "core/location_storage.h"
#include "core/model.h"

class MainScreen : public QMainWindow {
    Q_OBJECT

public:
    MainScreen(QWidget* parent = nullptr);
    ~MainScreen();

public Q_SLOTS:
    // Call Optimizer Launch
    void optimizer_launch_slot();

    // Calibration slots
    void onCalibrationLoaded();
    void onCalibrationError(const QString& message);
    void on_load_calibration_button_clicked();

    // Image slots
    void on_load_image_button_clicked();
    void on_image_list_widget_currentRowChanged(int);
    void on_original_image_radio_button_clicked();
    void on_inverted_image_radio_button_clicked();
    void on_edges_image_radio_button_clicked();
    void on_dilation_image_radio_button_clicked();

private:
    bool hasLoadedImages() const { return !loaded_frames.empty(); }
    void setupMonoplaneView();
    void setupBiplaneView();
    void updateCameraView(bool isViewA);
    void updateCalibrationButtons();
    void updateImageDisplay();

    double pi = 3.14159265358979323846;
    Ui::MainScreenClass ui;
    std::unique_ptr<Viewer> vw;
    std::unique_ptr<Viewer> coronal_vw;
    std::unique_ptr<CalibrationManager> calibration_manager_;
    int start_time;
    bool sym_trap_running;

    // State flags
    bool calibrated_for_monoplane_viewport_ = false;
    bool calibrated_for_biplane_viewport_ = false;
    Calibration calibration_file_;  // Legacy, will be removed

    // Data storage
    std::vector<Frame> loaded_frames;
    std::vector<Frame> loaded_frames_B;  // For biplane
    std::vector<Model> loaded_models;
    LocationStorage model_locations_;
};

#endif // MAINSCREEN_H