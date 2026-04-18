// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*Main Screen Header*/
#include "gui/mainscreen.h"

#include "gui/calibration_service.h"
#include "gui/scene_controller.h"
#include "gui/settings_service.h"

/*Font Manipulation*/
#include <qfontmetrics.h>

#include <QScreen>
#include <QSignalBlocker>
#include <opencv2/highgui.hpp>

/*Settings Constants*/
#include "core/curvature_utilities.h"
#include "core/settings_constants.h"

/*Size Constants*/
#include "core/mainscreen_size_constants.h"

/*Process Events*/
#include <qapplication.h>

/*Settings*/
#include <qdesktopwidget.h>
#include <qguiapplication.h>

/*File Processing*/
#include <qfiledialog.h>
#include <qtextstream.h>

#include "gui/interactor.h"

/*Messages*/
#include <qmessagebox.h>

/*OpenCV 3.1 Library*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*Custom Interactor*/
#include "gui/interactor.h"

/*About Window*/
#include "gui/about.h"

/*Control Window*/
#include "gui/controls.h"

/*STL Reader*/
#include "core/STLReader.h"

/* PyTorch 1.0 CPP Torch Script*/
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAMacros.h>
#include <torch/cuda.h> // For torch::cuda::empty_cache()
#include <torch/script.h>
#include <torch/torch.h>

#include "core/ambiguous_pose_processing.h"
#include "core/machine_learning_tools.h"
#include <iostream> // For std::cerr

using namespace std;

namespace {

std::vector<int> ToRowVector(const QModelIndexList& model_indices) {
    std::vector<int> rows;
    rows.reserve(model_indices.size());
    for (const QModelIndex& model_index : model_indices) {
        rows.push_back(model_index.row());
    }
    return rows;
}

jta_gui::ModelOpacityMode ResolveModelOpacityMode(const Ui::MainScreenClass& ui) {
    if (ui.solid_model_radio_button->isChecked()) {
        return jta_gui::ModelOpacityMode::Solid;
    }
    if (ui.transparent_model_radio_button->isChecked()) {
        return jta_gui::ModelOpacityMode::Transparent;
    }
    if (ui.wireframe_model_radio_button->isChecked()) {
        return jta_gui::ModelOpacityMode::Wireframe;
    }
    return jta_gui::ModelOpacityMode::Original;
}

} // namespace

/*Temporary Functions to Ease VTK Interaction and STL Loading*/
/*Mat to VTK Function*/
void MainScreen::matToVTK(cv::Mat Input, vtkSmartPointer<vtkImageData> Output) {
    // assert(Input.data != NULL);
    // vtkImageImport *importer = vtkImageImport::New();
    if (Output) {
        importer->SetOutput(Output);
    }
    importer->SetDataSpacing(1, 1, 1);
    importer->SetDataOrigin(0, 0, 0);
    importer->SetWholeExtent(
        0, Input.size().width - 1, 0, Input.size().height - 1, 0, 0);
    importer->SetDataExtentToWholeExtent();
    importer->SetDataScalarTypeToUnsignedChar();
    importer->SetNumberOfScalarComponents(Input.channels());
    importer->SetImportVoidPointer(Input.data);
    importer->Modified();
    importer->Update();
}

int MainScreen::curr_frame() {
    return ui.image_list_widget->currentIndex().row();
}

/*Global Interactor Variable*/
vtkSmartPointer<KeyPressInteractorStyle> key_press_vtk;

/*New Function*/
double MainScreen::CalculateViewingAngle(int width, int height, bool CameraA) {
    // Used to Set Height/2 = To The Bigger of the Width/2 + X Offset vs
    // Height/2
    // + Y Offset,
    //  now just set to height/2 + y_offset
    if (CameraA) {
        double y =
            height * session_.calibration_file_.camera_A_principal_.pixel_pitch_ / 2.0 +
            abs(session_.calibration_file_.camera_A_principal_.principal_y_);
        return 180.0 / pi * 2.0 *
               atan2(
                   y,
                   session_.calibration_file_.camera_A_principal_.principal_distance_);
    }
    double y =
        height * session_.calibration_file_.camera_B_principal_.pixel_pitch_ / 2.0 +
        abs(session_.calibration_file_.camera_B_principal_.principal_y_);
    return 180.0 / pi * 2.0 *
           atan2(y, session_.calibration_file_.camera_B_principal_.principal_distance_);
}

/*Constructor*/
MainScreen::MainScreen(QWidget* parent) : QMainWindow(parent) {
    ui.setupUi(this);

    this->start_time = -1;
    sym_trap_running = false;

    /*Set Minimum and Maximum for Sliders*/
    ui.low_threshold_slider->setMinimum(0);
    ui.high_threshold_slider->setMinimum(0);
    ui.low_threshold_slider->setMaximum(800);
    ui.high_threshold_slider->setMaximum(800);

    settings_service_ = std::make_unique<jta_gui::SettingsService>();

    /*Load Settings (THIS MUST BE DONE FIRST)*/
    LoadSettingsBetweenSessions();

    /*Set Label to Threshold Values*/
    ui.low_threshold_value->setText(
        QString::number(ui.low_threshold_slider->value()));
    ui.high_threshold_value->setText(
        QString::number(ui.high_threshold_slider->value()));

    /*Set Font, Font Size*/
    QFont application_font("Segoe UI", FONT_SIZE);
    QApplication::setFont(application_font);

    /*Set Up Settings Control Window*/
    settings_control = new SettingsControl(this);
    image_loading_service_ = new jta_gui::ImageLoadingService(this);
    worker_orchestrator_ = new jta_gui::WorkerOrchestrator(this);
    connect(
        settings_control,
        SIGNAL(SaveSettings(
            OptimizerSettings,
            jta_cost_function::CostFunctionManager,
            jta_cost_function::CostFunctionManager,
            jta_cost_function::CostFunctionManager)),
        this,
        SLOT(onSaveSettings(
            OptimizerSettings,
            jta_cost_function::CostFunctionManager,
            jta_cost_function::CostFunctionManager,
            jta_cost_function::CostFunctionManager)),
        Qt::DirectConnection);
    connect(
        image_loading_service_,
        &jta_gui::ImageLoadingService::error,
        this,
        [this](const QString& message) {
            QMessageBox::critical(this, "Error!", message, QMessageBox::Ok);
        });
    connect(
        worker_orchestrator_,
        &jta_gui::WorkerOrchestrator::imageSegmented,
        this,
        &MainScreen::onImageSegmented);
    connect(
        worker_orchestrator_,
        &jta_gui::WorkerOrchestrator::progressUpdated,
        this,
        &MainScreen::onSegmentationProgress);
    connect(
        worker_orchestrator_,
        &jta_gui::WorkerOrchestrator::segmentationFinished,
        this,
        &MainScreen::onSegmentationFinished);
    connect(
        worker_orchestrator_,
        &jta_gui::WorkerOrchestrator::poseEstimated,
        this,
        &MainScreen::onPoseEstimated);
    connect(
        worker_orchestrator_,
        &jta_gui::WorkerOrchestrator::estimationFinished,
        this,
        &MainScreen::onEstimationFinished);

    /* SYM TRAP */
    // Setup Sym Trap Window Obj
    // this->sym_trap_control = new sym_trap();
    // Connect signals for launching sym trap optimizer and updating progress
    // bar connect(sym_trap_control->ui.optimize, SIGNAL(clicked()), this,
    // SLOT(optimizer_launch_slot())); connect(this,
    // SIGNAL(UpdateTimeRemaining(int)), sym_trap_control->ui.progressBar,
    // SLOT(setValue(int)));

    /*Disable Stop Optimizer*/
    ui.actionStop_Optimizer->setDisabled(true);

    /*Set Up Icons For Files*/
    /*File*/
    ui.actionLoad_Pose->setIcon(QPixmap(":Menu_Icons/Resources/load_icon.png"));
    ui.actionSave_Pose->setIcon(QPixmap(":Menu_Icons/Resources/save_icon.png"));
    ui.actionQuit->setIcon(QPixmap(":Menu_Icons/Resources/quit_icon.ico"));

    /*View*/
    ui.actionReset_View->setIcon(
        QPixmap(":Menu_Icons/Resources/camerared.png"));
    ui.actionReset_Normal_Up->setIcon(
        QPixmap(":Menu_Icons/Resources/normaluppink.png"));

    /*Options*/
    ui.actionOptimizer_Settings->setIcon(
        QPixmap(":Menu_Icons/Resources/optimizer_settings_icon.png"));
    ui.actionRegion_Selection->setIcon(
        QPixmap(":Menu_Icons/Resources/selection_icon.ico"));
    ui.actionCenter_Placement->setIcon(
        QPixmap(":Menu_Icons/Resources/center_placement_icon.png"));
    ui.actionStop_Optimizer->setIcon(
        QPixmap(":Menu_Icons/Resources/stop_icon.png"));

    /*Help*/
    ui.actionAbout_JointTrack_Auto->setIcon(
        QPixmap(":Menu_Icons/Resources/help_icon.png"));
    ui.actionControls->setIcon(
        QPixmap(":Menu_Icons/Resources/controls_icon.png"));

    /*Set up RadioButton like functionality for View Menu Interaction Modes*/
    alignmentGroup = new QActionGroup(this);
    alignmentGroup->addAction(ui.actionModel_Interaction_Mode);
    alignmentGroup->addAction(ui.actionCamera_Interaction_Mode);
    ui.actionModel_Interaction_Mode->setChecked(true);

    /*Set up RadioButton like functionality for Segment Menu*/
    alignmentGroupSegment = new QActionGroup(this);
    alignmentGroupSegment->addAction(
        ui.actionBlack_Implant_Silhouettes_in_Original_Image_s);
    alignmentGroupSegment->addAction(
        ui.actionWhite_Implant_Silhouettes_in_Original_Image_s);
    ui.actionBlack_Implant_Silhouettes_in_Original_Image_s->setChecked(true);

    /*Disable View Menu Until Calbration Loaded*/
    ui.actionReset_View->setDisabled(true);
    ui.actionReset_Normal_Up->setDisabled(true);
    ui.actionModel_Interaction_Mode->setDisabled(true);
    ui.actionCamera_Interaction_Mode->setDisabled(true);

    /*Set Up Minimum Sizes*/
    this->setMinimumSize(QSize(MINIMUM_WIDTH, MINIMUM_HEIGHT));
    this->resize(QSize(MINIMUM_WIDTH, MINIMUM_HEIGHT));

    /*Arrange Main Screen Layout*/
    ArrangeMainScreenLayout(application_font);

    /*Maximize*/
    // QRect rec = QApplication::desktop()->availableGeometry();
    // QRect rec = QApplication::desktop()->screenGeometry();
    QRect rec;

    if (!QGuiApplication::screens().isEmpty()) {
        QScreen* primaryScreen = QGuiApplication::primaryScreen();
        rec = primaryScreen->availableGeometry();
    }
    if (MINIMUM_WIDTH <= rec.width() && MINIMUM_HEIGHT <= rec.height()) {
        showMaximized();
    }

    /*INitialize Location Storage*/
    session_.model_locations_ = LocationStorage();
    vw->initialize_vtk_pointers();
    vw->initialize_vtk_mappers();
    vw->initialize_vtk_renderers();
    coronal_vw->initialize_vtk_pointers();
    coronal_vw->initialize_vtk_mappers();
    coronal_vw->initialize_vtk_renderers();
    /*Selection Model for Models*/
    ui.single_model_radio_button->setChecked(true);
    ui.model_list_widget->setSelectionMode(QAbstractItemView::SingleSelection);

    /*Have NOT Loaded Calibration Files Yet*/
    session_.calibrated_for_monoplane_viewport_ = false;
    session_.calibrated_for_biplane_viewport_ = false;

    /*Index of Previously Selected Frame/Models*/
    previous_frame_index_ = -1;
    ///*Set up VTK*/
    vtkObject::GlobalWarningDisplayOff(); /*Turn off error display*/
    renderer = vw->get_renderer();
    coronal_renderer = renderer;
    actor_image = vw->get_actor_image();
    current_background = vw->get_current_background();
    stl_reader = vw->get_stl_reader();
    model_mapper_list = vw->get_model_mapper_list();
    model_actor_list = vw->get_model_actor_list();
    image_mapper = vw->get_image_mapper();
    actor_text = vw->get_actor_text();
    importer = vw->get_importer();
    key_press_vtk =
        vtkSmartPointer<KeyPressInteractorStyle>::New(); /*Custom Interactor
                                                            from JTA*/
    key_press_vtk->initialize_MainScreen(this);
    key_press_vtk->initialize_viewer(vw);
    camera_style_interactor = vtkSmartPointer<CameraInteractorStyle>::New();
    // vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    // /*Alternate Angled Interactor*/
    /*Text Actor Property*/
    actor_text->GetTextProperty()->SetFontSize(16);
    actor_text->GetTextProperty()->SetFontFamilyToCourier();
    actor_text->SetPosition2(0, 0);
    actor_text->GetTextProperty()->SetColor(
        214.0 / 255.0,
        108.0 / 255.0,
        35.0 / 255.0); // Earth Red

    /*Set Up Connections*/
    image_mapper->SetInputData(current_background);
    actor_image->SetPickable(0);
    actor_text->SetPickable(0);
    actor_image->SetMapper(image_mapper);
    vw->load_render_window(ui.qvtk_widget->renderWindow());
    coronal_vw->load_render_window(ui.qvtk_cpv->renderWindow());

    scene_controller_ = std::make_unique<jta_gui::SceneController>(
        *vw, *coronal_vw, session_, this);
    connect(
        scene_controller_.get(),
        &jta_gui::SceneController::requestThresholdControlSync,
        this,
        [this](int aperture, int low_threshold, int high_threshold) {
            ui.aperture_spin_box->setValue(aperture);
            ui.low_threshold_slider->setValue(low_threshold);
            ui.high_threshold_slider->setValue(high_threshold);
        });
    connect(
        scene_controller_.get(),
        &jta_gui::SceneController::requestCameraButtonSync,
        this,
        [this](bool disable_camera_a, bool disable_camera_b) {
            ui.camera_A_radio_button->setDisabled(disable_camera_a);
            ui.camera_B_radio_button->setDisabled(disable_camera_b);
        });
    connect(
        scene_controller_.get(),
        &jta_gui::SceneController::requestInteractorCameraMode,
        this,
        [](bool camera_b_mode) { interactor_camera_B = camera_b_mode; });
    connect(
        scene_controller_.get(),
        &jta_gui::SceneController::requestPrincipalSelectionOrder,
        this,
        [this](const QVector<int>& reordered_indices) {
            if (reordered_indices.isEmpty() ||
                ui.model_list_widget->selectionModel() == nullptr) {
                return;
            }

            QSignalBlocker blocker(ui.model_list_widget->selectionModel());
            for (int row = 0; row < ui.model_list_widget->count(); ++row) {
                ui.model_list_widget->item(row)->setSelected(false);
            }

            for (const int row : reordered_indices) {
                if (row < 0 || row >= ui.model_list_widget->count()) {
                    continue;
                }
                ui.model_list_widget->item(row)->setSelected(true);
            }
        });
    connect(
        scene_controller_.get(),
        &jta_gui::SceneController::requestUiRefresh,
        this,
        [this]() {
            ui.qvtk_widget->update();
            ui.qvtk_widget->renderWindow()->Render();
            ui.qvtk_cpv->update();
            ui.qvtk_cpv->renderWindow()->Render();
        });

    calibration_service_ =
        new jta_gui::CalibrationService(vw.get(), coronal_vw.get(), renderer, ui.image_list_widget, this);
    connect(
        calibration_service_,
        &jta_gui::CalibrationService::calibrationLoaded,
        this,
        &MainScreen::onCalibrationLoaded);
    connect(
        calibration_service_,
        &jta_gui::CalibrationService::error,
        this,
        [this](const QString& message) {
            QMessageBox::critical(this, "Error!", message, QMessageBox::Ok);
        });
    // vw->load_renderers_into_render_window();
    ui.qvtk_widget->renderWindow()->Render();

    /*Interactor*/
    key_press_vtk->AutoAdjustCameraClippingRangeOff();
    vw->load_in_interactor_style(key_press_vtk);
    // ui.qvtk_widget->GetRenderWindow()->GetInteractor()->SetInteractorStyle(key_press_vtk);

    /*Pose Estimate Progress and Label Not Visible*/
    ui.pose_progress->setValue(0);
    ui.pose_progress->setVisible(false);
    ui.pose_label->setVisible(false);

    /*Update*/
    ui.qvtk_widget->update();
    ui.qvtk_widget->renderWindow()->Render();

    /*Not Currently Optimizing*/
    currently_optimizing_ = false;

    QSizePolicy p = ui.qvtk_widget->sizePolicy();
    p.setHeightForWidth(true);
    ui.qvtk_widget->setSizePolicy(p);
}

/*Destructor*/
MainScreen::~MainScreen() {
    /*Delete QAction Group for View Menu*/
    delete alignmentGroup;
}

/*Arrange Main Screen*/
void MainScreen::ArrangeMainScreenLayout(QFont application_font) {
    /*Resize Buttons based on Font Size In Order to be Compatible with High
     * DPI Monitors*/
    QFontMetrics font_metrics(application_font);

    /*Adjust for Title Height*/
    this->setStyleSheet(
        this->styleSheet() += "QGroupBox { margin-top: " +
                              QString::number(font_metrics.height() / 2) +
                              "px; }");
    int group_box_to_top_button_y = font_metrics.height() / 2;

    /*Preprocessor Width*/
    /*Find Width of Buttons*/
    int preprocessor_button_width =
        font_metrics.horizontalAdvance(ui.preprocessor_box->title());
    if (preprocessor_button_width <
        font_metrics.horizontalAdvance(ui.load_calibration_button->text())) {
        preprocessor_button_width =
            font_metrics.horizontalAdvance(ui.load_calibration_button->text());
    }
    if (preprocessor_button_width <
        font_metrics.horizontalAdvance(ui.load_image_button->text())) {
        preprocessor_button_width =
            font_metrics.horizontalAdvance(ui.load_image_button->text());
    }
    if (preprocessor_button_width <
        font_metrics.horizontalAdvance(ui.load_model_button->text())) {
        preprocessor_button_width =
            font_metrics.horizontalAdvance(ui.load_model_button->text());
    }
    /*Augment with Padding to find width of group_box*/
    preprocessor_button_width += INSIDE_BUTTON_PADDING_X;
    int preprocessor_group_box_width =
        preprocessor_button_width + 2 * GROUP_BOX_TO_BUTTON_PADDING_X;

    /*Optimization Directives Width*/
    int optimizer_button_width =
        font_metrics.horizontalAdvance(ui.optimize_button->text());
    if (optimizer_button_width <
        font_metrics.horizontalAdvance(ui.optimize_all_button->text())) {
        optimizer_button_width =
            font_metrics.horizontalAdvance(ui.optimize_all_button->text());
    }
    if (optimizer_button_width <
        font_metrics.horizontalAdvance(ui.optimize_each_button->text())) {
        optimizer_button_width =
            font_metrics.horizontalAdvance(ui.optimize_each_button->text());
    }
    if (optimizer_button_width <
        font_metrics.horizontalAdvance(ui.optimize_from_button->text())) {
        optimizer_button_width =
            font_metrics.horizontalAdvance(ui.optimize_from_button->text());
    }
    /*Augment with padding to find width of group box*/
    optimizer_button_width += INSIDE_BUTTON_PADDING_X;
    int optimization_group_box_width =
        font_metrics.horizontalAdvance(ui.optimization_box->title());
    if (optimization_group_box_width < 2 * optimizer_button_width +
                                           BUTTON_TO_BUTTON_PADDING_X +
                                           2 * GROUP_BOX_TO_BUTTON_PADDING_X) {
        optimization_group_box_width = 2 * optimizer_button_width +
                                       BUTTON_TO_BUTTON_PADDING_X +
                                       2 * GROUP_BOX_TO_BUTTON_PADDING_X;
    }

    /*image View Width*/
    int image_view_button_width =
        font_metrics.horizontalAdvance(ui.original_image_radio_button->text());
    if (image_view_button_width < font_metrics.horizontalAdvance(
                                      ui.inverted_image_radio_button->text())) {
        image_view_button_width = font_metrics.horizontalAdvance(
            ui.inverted_image_radio_button->text());
    }
    if (image_view_button_width <
        font_metrics.horizontalAdvance(ui.edges_image_radio_button->text())) {
        image_view_button_width =
            font_metrics.horizontalAdvance(ui.edges_image_radio_button->text());
    }
    if (image_view_button_width < font_metrics.horizontalAdvance(
                                      ui.dilation_image_radio_button->text())) {
        image_view_button_width = font_metrics.horizontalAdvance(
            ui.dilation_image_radio_button->text());
    }
    /*Augment with Padding to find width of group_box*/
    image_view_button_width += INSIDE_RADIO_BUTTON_PADDING_X;
    int image_view_group_box_width =
        font_metrics.horizontalAdvance(ui.image_view_box->title());
    if (image_view_group_box_width < 2 * image_view_button_width +
                                         BUTTON_TO_BUTTON_PADDING_X +
                                         2 * GROUP_BOX_TO_BUTTON_PADDING_X) {
        image_view_group_box_width = 2 * image_view_button_width +
                                     BUTTON_TO_BUTTON_PADDING_X +
                                     2 * GROUP_BOX_TO_BUTTON_PADDING_X;
    }

    /*image Selection Width*/
    int image_selection_button_width =
        font_metrics.horizontalAdvance(ui.camera_A_radio_button->text());
    if (image_selection_button_width <
        font_metrics.horizontalAdvance(ui.camera_B_radio_button->text())) {
        image_selection_button_width =
            font_metrics.horizontalAdvance(ui.camera_B_radio_button->text());
    }

    /*Augment with Padding to find width of group_box*/
    image_selection_button_width += INSIDE_RADIO_BUTTON_PADDING_X;
    int image_selection_group_box_width =
        font_metrics.horizontalAdvance(ui.image_selection_box->title());
    if (image_selection_group_box_width <
        2 * image_selection_button_width + BUTTON_TO_BUTTON_PADDING_X +
            2 * GROUP_BOX_TO_BUTTON_PADDING_X) {
        image_selection_group_box_width = 2 * image_selection_button_width +
                                          BUTTON_TO_BUTTON_PADDING_X +
                                          2 * GROUP_BOX_TO_BUTTON_PADDING_X;
    }

    /*Get Largest Width Across Left Side Column*/
    int left_column_width = std::max(
        std::max(
            std::max(
                preprocessor_group_box_width, optimization_group_box_width),
            image_view_group_box_width),
        image_selection_group_box_width);

    /*Set Group Box Widths, Heights, and Member Objects Accordingly*/
    int text_height = font_metrics.height();
    /*Preprocessor*/
    ui.preprocessor_box->setGeometry(QRect(
        APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X,
        APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y,
        left_column_width,
        2 * GROUP_BOX_TO_BUTTON_PADDING_Y + 2 * BUTTON_TO_BUTTON_PADDING_Y +
            3 * (text_height + INSIDE_BUTTON_PADDING_Y) +
            group_box_to_top_button_y));
    ui.load_calibration_button->setGeometry(QRect(
        (left_column_width - preprocessor_button_width) / 2,
        GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
        preprocessor_button_width,
        text_height + INSIDE_BUTTON_PADDING_Y));
    ui.load_image_button->setGeometry(QRect(
        (left_column_width - preprocessor_button_width) / 2,
        GROUP_BOX_TO_BUTTON_PADDING_Y +
            (text_height + INSIDE_BUTTON_PADDING_Y +
             BUTTON_TO_BUTTON_PADDING_Y) +
            group_box_to_top_button_y,
        preprocessor_button_width,
        text_height + INSIDE_BUTTON_PADDING_Y));
    ui.load_model_button->setGeometry(QRect(
        (left_column_width - preprocessor_button_width) / 2,
        GROUP_BOX_TO_BUTTON_PADDING_Y +
            2 * (text_height + INSIDE_BUTTON_PADDING_Y +
                 BUTTON_TO_BUTTON_PADDING_Y) +
            group_box_to_top_button_y,
        preprocessor_button_width,
        text_height + INSIDE_BUTTON_PADDING_Y));
    /*Optimization Directive*/
    ui.optimization_box->setGeometry(QRect(
        APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X,
        GROUP_BOX_TO_GROUP_BOX_Y +
            ui.preprocessor_box->geometry().bottomLeft().y(),
        left_column_width,
        2 * GROUP_BOX_TO_BUTTON_PADDING_Y + 1 * BUTTON_TO_BUTTON_PADDING_Y +
            2 * (text_height + INSIDE_BUTTON_PADDING_Y) +
            group_box_to_top_button_y));
    ui.optimize_button->setGeometry(QRect(
        (left_column_width -
         (2 * optimizer_button_width + BUTTON_TO_BUTTON_PADDING_X)) /
            2,
        GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
        optimizer_button_width,
        text_height + INSIDE_BUTTON_PADDING_Y));
    ui.optimize_all_button->setGeometry(QRect(
        (left_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
        GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
        optimizer_button_width,
        text_height + INSIDE_BUTTON_PADDING_Y));
    ui.optimize_each_button->setGeometry(QRect(
        (left_column_width -
         (2 * optimizer_button_width + BUTTON_TO_BUTTON_PADDING_X)) /
            2,
        GROUP_BOX_TO_BUTTON_PADDING_Y +
            (text_height + INSIDE_BUTTON_PADDING_Y +
             BUTTON_TO_BUTTON_PADDING_Y) +
            group_box_to_top_button_y,
        optimizer_button_width,
        text_height + INSIDE_BUTTON_PADDING_Y));
    ui.optimize_from_button->setGeometry(QRect(
        (left_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
        GROUP_BOX_TO_BUTTON_PADDING_Y +
            (text_height + INSIDE_BUTTON_PADDING_Y +
             BUTTON_TO_BUTTON_PADDING_Y) +
            group_box_to_top_button_y,
        optimizer_button_width,
        text_height + INSIDE_BUTTON_PADDING_Y));
    /*image View*/
    ui.image_view_box->setGeometry(QRect(
        APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X,
        GROUP_BOX_TO_GROUP_BOX_Y +
            ui.optimization_box->geometry().bottomLeft().y(),
        left_column_width,
        2 * GROUP_BOX_TO_BUTTON_PADDING_Y + 1 * BUTTON_TO_BUTTON_PADDING_Y +
            2 * (text_height + INSIDE_RADIO_BUTTON_PADDING_Y) +
            group_box_to_top_button_y));
    ui.original_image_radio_button->setGeometry(QRect(
        (left_column_width -
         (2 * image_view_button_width + BUTTON_TO_BUTTON_PADDING_X)) /
            2,
        GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
        image_view_button_width,
        text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
    ui.inverted_image_radio_button->setGeometry(QRect(
        (left_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
        GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
        image_view_button_width,
        text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
    ui.edges_image_radio_button->setGeometry(QRect(
        (left_column_width -
         (2 * image_view_button_width + BUTTON_TO_BUTTON_PADDING_X)) /
            2,
        GROUP_BOX_TO_BUTTON_PADDING_Y +
            (text_height + INSIDE_RADIO_BUTTON_PADDING_Y +
             BUTTON_TO_BUTTON_PADDING_Y) +
            group_box_to_top_button_y,
        image_view_button_width,
        text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
    ui.dilation_image_radio_button->setGeometry(QRect(
        (left_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
        GROUP_BOX_TO_BUTTON_PADDING_Y +
            (text_height + INSIDE_RADIO_BUTTON_PADDING_Y +
             BUTTON_TO_BUTTON_PADDING_Y) +
            group_box_to_top_button_y,
        image_view_button_width,
        text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
    /*image Selection*/
    /*Check Size of Application, If not big enough for listwidget, resize
     * application*/
    int image_selection_box_height =
        2 * GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y +
        text_height + INSIDE_RADIO_BUTTON_PADDING_Y +
        RADIO_BUTTON_TO_LIST_WIDGET_PADDING_Y + MINIMUM_LIST_WIDGET_SIZE;

    int total_box_height = GROUP_BOX_TO_GROUP_BOX_Y +
                           ui.image_view_box->geometry().bottomLeft().y() +
                           APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y +
                           group_box_to_top_button_y + ui.menuBar->height() +
                           image_selection_box_height;

    /*(total_box_height < this->height()) ? image_selection_box_height =
       this->height() - (GROUP_BOX_TO_GROUP_BOX_Y +
       ui.image_view_box->geometry(). bottomLeft().y() +
       APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y + group_box_to_top_button_y
       + ui.menuBar-> height()) :
       this->setMinimumHeight(total_box_height);*/

    ui.image_selection_box->setGeometry(QRect(
        APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X,
        GROUP_BOX_TO_GROUP_BOX_Y +
            ui.image_view_box->geometry().bottomLeft().y(),
        left_column_width,
        image_selection_box_height));
    ui.camera_A_radio_button->setGeometry(QRect(
        (left_column_width -
         (2 * image_selection_button_width + BUTTON_TO_BUTTON_PADDING_X)) /
            2,
        GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
        image_selection_button_width,
        text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
    ui.camera_B_radio_button->setGeometry(QRect(
        (left_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
        GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
        image_selection_button_width,
        text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
    ui.image_list_widget->setGeometry(QRect(
        GROUP_BOX_TO_BUTTON_PADDING_X,
        ui.camera_A_radio_button->geometry().bottomLeft().y() +
            RADIO_BUTTON_TO_LIST_WIDGET_PADDING_Y,
        left_column_width - 2 * GROUP_BOX_TO_BUTTON_PADDING_X,
        ui.image_selection_box->height() -
            (ui.camera_A_radio_button->geometry().bottomLeft().y() +
             RADIO_BUTTON_TO_LIST_WIDGET_PADDING_Y +
             GROUP_BOX_TO_BUTTON_PADDING_Y)));

    /*Dilation Box Width*/
    int right_button_bigger =
        font_metrics.horizontalAdvance(ui.apply_all_edge_button->text());

    /*Edge Detection Box Width*/
    int edge_detection_box =
        font_metrics.horizontalAdvance(ui.edge_detection_box->title());
    int x_padding = LABEL_TO_SPIN_BOX_PADDING_X +
                    INSIDE_RADIO_BUTTON_PADDING_X +
                    font_metrics.horizontalAdvance("888");
    if (edge_detection_box <
        font_metrics.horizontalAdvance(ui.aperture_label->text()) + x_padding) {
        edge_detection_box =
            font_metrics.horizontalAdvance(ui.aperture_label->text()) +
            x_padding + SPIN_BOX_TO_GROUP_BOX_PADDING_X;
    }
    if (edge_detection_box <
        font_metrics.horizontalAdvance(ui.low_threshold_label->text()) +
            x_padding) {
        edge_detection_box =
            font_metrics.horizontalAdvance(ui.low_threshold_label->text()) +
            x_padding + SPIN_BOX_TO_GROUP_BOX_PADDING_X;
    }
    if (edge_detection_box <
        font_metrics.horizontalAdvance(ui.high_threshold_label->text()) +
            x_padding) {
        edge_detection_box =
            font_metrics.horizontalAdvance(ui.high_threshold_label->text()) +
            x_padding + SPIN_BOX_TO_GROUP_BOX_PADDING_X;
    }
    if (edge_detection_box < 2 * right_button_bigger +
                                 BUTTON_TO_BUTTON_PADDING_X +
                                 2 * INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X) {
        edge_detection_box = 2 * right_button_bigger +
                             BUTTON_TO_BUTTON_PADDING_X +
                             2 * INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X +
                             2 * GROUP_BOX_TO_BUTTON_PADDING_X;
    }

    /*model View Width*/
    int model_view_button_width =
        font_metrics.horizontalAdvance(ui.original_model_radio_button->text());
    if (model_view_button_width <
        font_metrics.horizontalAdvance(ui.solid_model_radio_button->text())) {
        model_view_button_width =
            font_metrics.horizontalAdvance(ui.solid_model_radio_button->text());
    }
    if (model_view_button_width <
        font_metrics.horizontalAdvance(
            ui.transparent_model_radio_button->text())) {
        model_view_button_width = font_metrics.horizontalAdvance(
            ui.transparent_model_radio_button->text());
    }
    if (model_view_button_width <
        font_metrics.horizontalAdvance(
            ui.wireframe_model_radio_button->text())) {
        model_view_button_width = font_metrics.horizontalAdvance(
            ui.wireframe_model_radio_button->text());
    }
    /*Augment with Padding to find width of group_box*/
    model_view_button_width += INSIDE_RADIO_BUTTON_PADDING_X;
    int model_view_group_box_width =
        font_metrics.horizontalAdvance(ui.model_view_box->title());
    if (model_view_group_box_width < 2 * model_view_button_width +
                                         BUTTON_TO_BUTTON_PADDING_X +
                                         2 * GROUP_BOX_TO_BUTTON_PADDING_X) {
        model_view_group_box_width = 2 * model_view_button_width +
                                     BUTTON_TO_BUTTON_PADDING_X +
                                     2 * GROUP_BOX_TO_BUTTON_PADDING_X;
    }

    /*model Selection Width*/
    int model_selection_button_width =
        font_metrics.horizontalAdvance(ui.single_model_radio_button->text());
    if (model_selection_button_width <
        font_metrics.horizontalAdvance(
            ui.multiple_model_radio_button->text())) {
        model_selection_button_width = font_metrics.horizontalAdvance(
            ui.multiple_model_radio_button->text());
    }
    /*Augment with Padding to find width of group_box*/
    model_selection_button_width += INSIDE_RADIO_BUTTON_PADDING_X;
    int model_selection_group_box_width =
        font_metrics.horizontalAdvance(ui.model_selection_box->title());
    int model_box_pad = 2 * model_selection_button_width +
                        BUTTON_TO_BUTTON_PADDING_X +
                        2 * GROUP_BOX_TO_BUTTON_PADDING_X;

    if (model_selection_group_box_width < model_box_pad) {
        model_selection_group_box_width = model_box_pad;
    }

    /*Get Largest Width Across Right Side Column*/
    int right_column_width = std::max(
        std::max(edge_detection_box, model_view_group_box_width),
        model_selection_group_box_width);

    /*Set Group Box Widths, Heights, and Member Objects Accordingly*/
    /*Edge Detection Box*/
    ui.edge_detection_box->setGeometry(QRect(
        this->width() - APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X -
            right_column_width,
        APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y,
        // GROUP_BOX_TO_GROUP_BOX_Y +
        // ui.dilation_box->geometry().bottomLeft().y(),
        right_column_width,
        2 * GROUP_BOX_TO_BUTTON_PADDING_Y + 7 * text_height +
            2 * INSIDE_BUTTON_PADDING_Y + 3 * INSIDE_SPIN_BOX_PADDING_Y +
            group_box_to_top_button_y + 3 * SPIN_BOX_TO_SPIN_BOX_PADDING_Y +
            BUTTON_TO_BUTTON_PADDING_Y));
    ui.aperture_spin_box->setGeometry(QRect(
        right_column_width -
            (SPIN_BOX_TO_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X +
             font_metrics.horizontalAdvance("888")),
        GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
        INSIDE_SPIN_BOX_PADDING_X + font_metrics.horizontalAdvance("888"),
        text_height + INSIDE_SPIN_BOX_PADDING_Y));
    ui.aperture_label->setGeometry(QRect(
        right_column_width -
            (font_metrics.horizontalAdvance(ui.aperture_label->text()) +
             LABEL_TO_SPIN_BOX_PADDING_X + SPIN_BOX_TO_GROUP_BOX_PADDING_X +
             INSIDE_SPIN_BOX_PADDING_X + font_metrics.horizontalAdvance("888")),
        GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
        font_metrics.horizontalAdvance(ui.aperture_label->text()),
        text_height + INSIDE_SPIN_BOX_PADDING_Y));

    ui.low_threshold_label->setGeometry(QRect(
        right_column_width -
            (font_metrics.horizontalAdvance(ui.low_threshold_label->text()) +
             LABEL_TO_SPIN_BOX_PADDING_X + SPIN_BOX_TO_GROUP_BOX_PADDING_X +
             INSIDE_SPIN_BOX_PADDING_X + font_metrics.horizontalAdvance("888")),
        ui.aperture_label->geometry().bottom() + SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
        font_metrics.horizontalAdvance(ui.low_threshold_label->text()),
        text_height + INSIDE_SPIN_BOX_PADDING_Y));

    ui.low_threshold_value->setGeometry(QRect(
        3 + right_column_width -
            (SPIN_BOX_TO_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X +
             font_metrics.horizontalAdvance("888")),
        ui.aperture_label->geometry().bottom() + SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
        font_metrics.horizontalAdvance(ui.low_threshold_label->text()),
        text_height + INSIDE_SPIN_BOX_PADDING_Y));

    ui.low_threshold_slider->setGeometry(QRect(
        (right_column_width - BUTTON_TO_BUTTON_PADDING_X) / 2 -
            INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X - right_button_bigger,
        ui.low_threshold_label->geometry().bottom() +
            .5 * SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
        2 * right_button_bigger + BUTTON_TO_BUTTON_PADDING_X +
            2 * INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X,
        1.5 * text_height));

    ui.high_threshold_label->setGeometry(QRect(
        3 + right_column_width -
            (font_metrics.horizontalAdvance(ui.high_threshold_label->text()) +
             LABEL_TO_SPIN_BOX_PADDING_X + SPIN_BOX_TO_GROUP_BOX_PADDING_X +
             INSIDE_SPIN_BOX_PADDING_X + font_metrics.horizontalAdvance("888")),
        ui.low_threshold_slider->geometry().bottom() +
            SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
        font_metrics.horizontalAdvance(ui.high_threshold_label->text()),
        text_height + INSIDE_SPIN_BOX_PADDING_Y));

    ui.high_threshold_value->setGeometry(QRect(
        right_column_width -
            (SPIN_BOX_TO_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X +
             font_metrics.horizontalAdvance("888")),
        ui.low_threshold_slider->geometry().bottom() +
            SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
        font_metrics.horizontalAdvance(ui.high_threshold_label->text()),
        text_height + INSIDE_SPIN_BOX_PADDING_Y));

    ui.high_threshold_slider->setGeometry(QRect(
        (right_column_width - BUTTON_TO_BUTTON_PADDING_X) / 2 -
            INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X - right_button_bigger,
        ui.high_threshold_label->geometry().bottom() +
            .5 * SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
        2 * right_button_bigger + BUTTON_TO_BUTTON_PADDING_X +
            2 * INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X,
        1.5 * text_height));

    ui.apply_all_edge_button->setGeometry(QRect(
        (right_column_width - BUTTON_TO_BUTTON_PADDING_X) / 2 -
            INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X - right_button_bigger,
        ui.high_threshold_slider->geometry().bottom() +
            GROUP_BOX_TO_BUTTON_PADDING_Y,
        2 * right_button_bigger + BUTTON_TO_BUTTON_PADDING_X +
            2 * INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X,
        text_height + INSIDE_BUTTON_PADDING_Y));
    ui.reset_edge_button->setGeometry(QRect(
        (right_column_width - BUTTON_TO_BUTTON_PADDING_X) / 2 -
            INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X - right_button_bigger,
        ui.apply_all_edge_button->geometry().bottom() +
            BUTTON_TO_BUTTON_PADDING_Y,
        2 * right_button_bigger + BUTTON_TO_BUTTON_PADDING_X +
            2 * INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X,
        text_height + INSIDE_BUTTON_PADDING_Y));
    ui.edge_detection_box->setGeometry(QRect(
        this->width() - APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X -
            right_column_width,
        APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y,
        // GROUP_BOX_TO_GROUP_BOX_Y +
        // ui.dilation_box->geometry().bottomLeft().y(),
        right_column_width,
        ui.reset_edge_button->geometry().bottom() +
            GROUP_BOX_TO_BUTTON_PADDING_Y));

    /*model View*/
    ui.model_view_box->setGeometry(QRect(
        this->width() - APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X -
            right_column_width,
        GROUP_BOX_TO_GROUP_BOX_Y +
            ui.edge_detection_box->geometry().bottomLeft().y(),
        right_column_width,
        2 * GROUP_BOX_TO_BUTTON_PADDING_Y + 1 * BUTTON_TO_BUTTON_PADDING_Y +
            2 * (text_height + INSIDE_RADIO_BUTTON_PADDING_Y) +
            group_box_to_top_button_y));
    ui.original_model_radio_button->setGeometry(QRect(
        (right_column_width -
         (2 * model_view_button_width + BUTTON_TO_BUTTON_PADDING_X)) /
            2,
        GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
        model_view_button_width,
        text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
    ui.solid_model_radio_button->setGeometry(QRect(
        (right_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
        GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
        model_view_button_width,
        text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
    ui.transparent_model_radio_button->setGeometry(QRect(
        (right_column_width -
         (2 * model_view_button_width + BUTTON_TO_BUTTON_PADDING_X)) /
            2,
        GROUP_BOX_TO_BUTTON_PADDING_Y +
            (text_height + INSIDE_RADIO_BUTTON_PADDING_Y +
             BUTTON_TO_BUTTON_PADDING_Y) +
            group_box_to_top_button_y,
        model_view_button_width,
        text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
    ui.wireframe_model_radio_button->setGeometry(QRect(
        (right_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
        GROUP_BOX_TO_BUTTON_PADDING_Y +
            (text_height + INSIDE_RADIO_BUTTON_PADDING_Y +
             BUTTON_TO_BUTTON_PADDING_Y) +
            group_box_to_top_button_y,
        model_view_button_width,
        text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
    /*model Selection*/
    /*Check Size of Application, If not big enough for listwidget, resize
     * application*/
    int model_selection_box_height =
        2 * GROUP_BOX_TO_BUTTON_PADDING_Y + INSIDE_RADIO_BUTTON_PADDING_Y +
        RADIO_BUTTON_TO_LIST_WIDGET_PADDING_Y + MINIMUM_LIST_WIDGET_SIZE +
        group_box_to_top_button_y + text_height;

    total_box_height = GROUP_BOX_TO_GROUP_BOX_Y +
                       ui.model_view_box->geometry().bottomLeft().y() +
                       APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y +
                       ui.menuBar->height() + model_selection_box_height +
                       group_box_to_top_button_y;

    if (total_box_height < this->height()) {
        model_selection_box_height =
            this->height() - (total_box_height - model_selection_box_height);
    } else {
        int old_height = this->height();
        this->setMinimumHeight(total_box_height);
        QRect updated_image_selection_group_box_geometry =
            ui.image_selection_box->geometry();
        updated_image_selection_group_box_geometry.setHeight(
            updated_image_selection_group_box_geometry.height() +
            this->height() - old_height);
        QRect updated_image_list_widget_geometry =
            ui.image_list_widget->geometry();
        updated_image_list_widget_geometry.setHeight(
            updated_image_list_widget_geometry.height() + this->height() -
            old_height);
        ui.image_selection_box->setGeometry(
            updated_image_selection_group_box_geometry);
        ui.image_list_widget->setGeometry(updated_image_list_widget_geometry);
    }
    ui.model_selection_box->setGeometry(QRect(
        this->width() - APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X -
            right_column_width,
        GROUP_BOX_TO_GROUP_BOX_Y +
            ui.model_view_box->geometry().bottomLeft().y(),
        right_column_width,
        model_selection_box_height));
    ui.single_model_radio_button->setGeometry(QRect(
        (right_column_width -
         (2 * model_selection_button_width + BUTTON_TO_BUTTON_PADDING_X)) /
            2,
        GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
        model_selection_button_width,
        text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
    ui.multiple_model_radio_button->setGeometry(QRect(
        (right_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
        GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
        model_selection_button_width,
        text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
    ui.model_list_widget->setGeometry(QRect(
        GROUP_BOX_TO_BUTTON_PADDING_X,
        ui.single_model_radio_button->geometry().bottomLeft().y() +
            RADIO_BUTTON_TO_LIST_WIDGET_PADDING_Y,
        right_column_width - 2 * GROUP_BOX_TO_BUTTON_PADDING_X,
        ui.model_selection_box->height() -
            (ui.single_model_radio_button->geometry().bottomLeft().y() +
             RADIO_BUTTON_TO_LIST_WIDGET_PADDING_Y +
             GROUP_BOX_TO_BUTTON_PADDING_Y)));
    int qvtk_side_length;

    /*Arrange QVTK Widget*/
    /*Check if there is enough room*/
    int widget_detection_width = ui.edge_detection_box->geometry().left() -
                                 ui.preprocessor_box->geometry().right() -
                                 2 * GROUP_BOX_TO_QVTK_PADDING_X;
    int widget_detection_height =
        ui.model_selection_box->geometry().bottom() -
        (ui.preprocessor_box->geometry().top() + font_metrics.height() / 2);

    if (widget_detection_width > MINIMUM_QVTK_WIDGET_WIDTH) {
        qvtk_side_length =
            ((widget_detection_width) > (widget_detection_height + 1))
                ? widget_detection_height + 1 // original height
                : widget_detection_width;

        ui.qvtk_widget->setGeometry(QRect(
            ui.preprocessor_box->geometry().right() +
                GROUP_BOX_TO_QVTK_PADDING_X,
            ui.preprocessor_box->geometry().top() + font_metrics.height() / 2,
            qvtk_side_length,
            qvtk_side_length));
    } else {
        qvtk_side_length =
            (MINIMUM_QVTK_WIDGET_WIDTH > widget_detection_height + 1)
                ? widget_detection_height + 1
                : MINIMUM_QVTK_WIDGET_WIDTH;
    }
    ui.qvtk_widget->setGeometry(QRect(
        ui.preprocessor_box->geometry().right() + GROUP_BOX_TO_QVTK_PADDING_X,
        ui.preprocessor_box->geometry().top() + font_metrics.height() / 2,
        qvtk_side_length,
        qvtk_side_length));
    ui.qvtk_widget->resize(qvtk_side_length, qvtk_side_length);
    /*New Minimum*/
    this->setMinimumWidth(
        ui.edge_detection_box->geometry().right() +
        APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X);

    /*Initialize the Starting heights and widths*/
    /*Original Sizes After Construction for Main Screen List Widgets, their
     * Group Boxes and QVTK Widget*/
    image_list_widget_starting_height_ =
        ui.image_list_widget->geometry().height();
    image_selection_box_starting_height_ =
        ui.image_selection_box->geometry().height();
    model_list_widget_starting_height_ =
        ui.model_list_widget->geometry().height();
    model_selection_box_starting_height_ =
        ui.model_selection_box->geometry().height();
    qvtk_widget_starting_height_ = ui.qvtk_widget->geometry().height();
    qvtk_widget_starting_width_ = ui.qvtk_widget->geometry().height();
}

/* Commented out as this causes flickering on resize */
/*Handle Resize Event*/

void MainScreen::resizeEvent(QResizeEvent* event) {
    /*Process Events*/
    qApp->processEvents();

    /*Resize Event*/
    QMainWindow::resizeEvent(event);
    if (vw->windowCenterSet()) {
        vw->update_window_center_on_resize();
    }

    /*Expansion Constants*/
    // int horizontal_expansion = this->width() - this->minimumWidth();
    // int vertical_expansion = this->height() - this->minimumHeight();
    // int total_expansion;

    // if (horizontal_expansion > vertical_expansion) {
    // 	total_expansion = vertical_expansion;
    // }
    // else {
    // 	total_expansion = horizontal_expansion;
    // }

    // /* Maintain square aspect ratio and correct positioning for the main
    // viewer window and the coronal plane viewer */ QPoint p_main =
    // QPoint(ui.gridLayout_3->geometry().left() +
    // (ui.gridLayout_3->geometry().width() / 2) -
    // ((qvtk_widget_starting_height_
    // + total_expansion) / 2), ui.pose_progress->geometry().bottom());
    // QRect r_main = QRect(p_main, QSize(qvtk_widget_starting_height_ +
    // total_expansion, qvtk_widget_starting_height_ + total_expansion));

    // QPoint p_cpv = QPoint(ui.Right->geometry().left(),
    // ui.Right->geometry().top()); QRect r_cpv = QRect(p_cpv,
    // QSize(ui.qvtk_cpv->geometry().width(),
    // ui.qvtk_cpv->geometry().width()));

    // /*Expand QVTK Widgets*/
    // ui.qvtk_widget->setGeometry(r_main);
    // ui.qvtk_cpv->setGeometry(r_cpv);
}

/*MENU BAR BUTTONS*/
/*File Menu*/
/*Save Pose*/
void MainScreen::on_actionSave_Pose_triggered() {
    // Save Single Pose
    // Selection Check
    /*Load Models Selected Indices*/
    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    if (ui.image_list_widget->currentRow() < 0 || selected.size() == 0) {
        QMessageBox::critical(
            this, "Error!", "Select Frame and Model First!", QMessageBox::Ok);
        return;
    }

    // Must be in Single Selection Mode to Load Pose
    if (ui.multiple_model_radio_button->isChecked()) {
        QMessageBox::critical(
            this,
            "Error!",
            "Must Be in Single Model Selection Mode to Save Pose!",
            QMessageBox::Ok);
        return;
    }

    /*Save Pose*/
    SaveLastPose();

    /*Get Pose to Save*/
    Point6D saved_pose = session_.model_locations_.GetPose(
        ui.image_list_widget->currentRow(), selected[0].row());

    // Open Save File Dialogue
    QString SavePoseExtension = QFileDialog::getSaveFileName(
        this,
        tr("Save Pose"),
        ".",
        tr("JTA Pose File (*.jtap);; Pose File (*.txt)"));
    QFile file(SavePoseExtension);
    if (file.open(QIODevice::ReadWrite | QIODevice::Text)) {
        QTextStream stream(&file);
        stream << "JTA_EULER_POSE\nX_TRAN\t\tY_TRAN\t\tZ_TRAN\t\tZ_ROT\t\tX_"
                  "ROT\t\tY_ROT\n";
        if (QString::number(saved_pose.x).length() < 7) {
            stream << saved_pose.x << ",\t\t";
        } else {
            stream << saved_pose.x << ",\t";
        }
        if (QString::number(saved_pose.y).length() < 7) {
            stream << saved_pose.y << ",\t\t";
        } else {
            stream << saved_pose.y << ",\t";
        }
        if (QString::number(saved_pose.z).length() < 7) {
            stream << saved_pose.z << ",\t\t";
        } else {
            stream << saved_pose.z << ",\t";
        }
        if (QString::number(saved_pose.za).length() < 7) {
            stream << saved_pose.za << ",\t\t";
        } else {
            stream << saved_pose.za << ",\t";
        }
        if (QString::number(saved_pose.xa).length() < 7) {
            stream << saved_pose.xa << ",\t\t";
        } else {
            stream << saved_pose.xa << ",\t";
        }
        if (QString::number(saved_pose.ya).length() < 7) {
            stream << saved_pose.ya << ",\t\t";
        } else {
            stream << saved_pose.ya << ",\n";
        }
    }
}

/*Save Kinematics*/
void MainScreen::on_actionSave_Kinematics_triggered() {
    // Save Single Pose
    /*Load Models Selected Indices*/
    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    if (ui.image_list_widget->currentRow() < 0 || selected.size() == 0) {
        QMessageBox::critical(
            this,
            "Error!",
            "Select Model and Load Frames First!",
            QMessageBox::Ok);
        return;
    }

    // Must be in Single Selection Mode to Load Pose
    if (ui.multiple_model_radio_button->isChecked()) {
        QMessageBox::critical(
            this,
            "Error!",
            "Must Be in Single Model Selection Mode to Save Kinematics!",
            QMessageBox::Ok);
        return;
    }

    /*Save Pose*/
    SaveLastPose();

    // Open Save File Dialogue
    QString SavePoseExtension = QFileDialog::getSaveFileName(
        this,
        tr("Save Kinematics"),
        ".",
        tr("JTA Kinematics File (*.jtak);; Kinematics File (*.txt)"));
    QFile file(SavePoseExtension);
    if (file.open(QIODevice::ReadWrite | QIODevice::Text)) {
        QTextStream stream(&file);
        stream << "JTA_EULER_KINEMATICS\nX_TRAN\t\tY_TRAN\t\tZ_TRAN\t\tZ_"
                  "ROT\t\tX_"
                  "ROT\t\tY_ROT\n";
        /*Get Pose to Save*/
        for (int i = 0; i < ui.image_list_widget->count(); i++) {
            Point6D saved_pose = session_.model_locations_.GetPose(i, selected[0].row());

            if (QString::number(saved_pose.x).length() < 7) {
                stream << saved_pose.x << ",\t\t";
            } else {
                stream << saved_pose.x << ",\t";
            }
            if (QString::number(saved_pose.y).length() < 7) {
                stream << saved_pose.y << ",\t\t";
            } else {
                stream << saved_pose.y << ",\t";
            }
            if (QString::number(saved_pose.z).length() < 7) {
                stream << saved_pose.z << ",\t\t";
            } else {
                stream << saved_pose.z << ",\t";
            }
            if (QString::number(saved_pose.za).length() < 7) {
                stream << saved_pose.za << ",\t\t";
            } else {
                stream << saved_pose.za << ",\t";
            }
            if (QString::number(saved_pose.xa).length() < 7) {
                stream << saved_pose.xa << ",\t\t";
            } else {
                stream << saved_pose.xa << ",\t";
            }
            stream << saved_pose.ya << ",\n";
        }
    }
}

/*Load Pose*/
void MainScreen::on_actionLoad_Pose_triggered() {
    // Load Pose
    // Selection Check
    /*Load Models Selected Indices*/
    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    if (ui.image_list_widget->currentRow() < 0 || selected.size() == 0) {
        QMessageBox::critical(
            this, "Error!", "Select Frame and Model First!", QMessageBox::Ok);
        return;
    }

    // Must be in Single Selection Mode to Load Pose
    if (ui.multiple_model_radio_button->isChecked()) {
        QMessageBox::critical(
            this,
            "Error!",
            "Must Be in Single Model Selection Mode to Load Pose!",
            QMessageBox::Ok);
        return;
    }

    // Load File Dialog
    QString LoadPoseExtension = QFileDialog::getOpenFileName(
        this,
        tr("Load Pose"),
        ".",
        tr("JTA Pose File (*.jtap);; JointTrack Pose File (*.jtp);; Pose "
           "File "
           "(*.txt);;"));
    QFile inputFile(LoadPoseExtension);
    QFileInfo inputFileInfo(inputFile);
    if (inputFile.open(QIODevice::ReadOnly)) {
        QTextStream in(&inputFile);
        QStringList InputList =
            in.readAll().split(QRegExp("[\r\n]"), Qt::SkipEmptyParts);
        if (InputList.size() == 0) {
            QMessageBox::critical(
                this, "Error!", "Invalid Pose File!", QMessageBox::Ok);
            inputFile.close();
            return;
        }
        if (inputFileInfo.suffix() == "jtp") {
            QStringList LineList =
                InputList[0].split(QRegExp("[,]"), Qt::SkipEmptyParts);
            if (LineList.size() >= 6) {
                LineList[0].replace(" ", "");
                if (LineList[0] == "NOT_OPTIMIZED") {
                    QMessageBox::critical(
                        this, "Error!", "No Pose Exists!", QMessageBox::Ok);
                    inputFile.close();
                    return;
                }
                auto loaded_pose = Point6D(
                    LineList[0].toDouble(),
                    LineList[1].toDouble(),
                    LineList[2].toDouble(),
                    LineList[4].toDouble(),
                    LineList[5].toDouble(),
                    LineList[3].toDouble());
                session_.model_locations_.SavePose(
                    ui.image_list_widget->currentRow(),
                    selected[0].row(),
                    loaded_pose);
                vw->set_model_position_at_index(
                    selected[0].row(),
                    loaded_pose.x,
                    loaded_pose.y,
                    loaded_pose.z);
                vw->set_model_orientation_at_index(
                    selected[0].row(),
                    loaded_pose.xa,
                    loaded_pose.ya,
                    loaded_pose.za);
                coronal_vw->set_model_position_at_index(
                    selected[0].row(),
                    loaded_pose.x,
                    loaded_pose.y,
                    loaded_pose.z);
                coronal_vw->set_model_orientation_at_index(
                    selected[0].row(),
                    loaded_pose.xa,
                    loaded_pose.ya,
                    loaded_pose.za);
                ui.qvtk_widget->update();
                ui.qvtk_widget->renderWindow()->Render();
                ui.qvtk_cpv->update();
                ui.qvtk_cpv->renderWindow()->Render();
            } else {
                QMessageBox::critical(
                    this, "Error!", "Invalid Pose!", QMessageBox::Ok);
                inputFile.close();
                return;
            }
        } else {
            if (InputList[0] == "JTA_EULER_POSE") {
                QStringList LineList =
                    InputList[2].split(QRegExp("[,]"), Qt::SkipEmptyParts);
                if (LineList.size() >= 6) {
                    LineList[0].replace(" ", "");
                    if (LineList[0] == "NOT_OPTIMIZED") {
                        QMessageBox::critical(
                            this, "Error!", "No Pose Exists!", QMessageBox::Ok);
                        inputFile.close();
                        return;
                    }
                    auto loaded_pose = Point6D(
                        LineList[0].toDouble(),
                        LineList[1].toDouble(),
                        LineList[2].toDouble(),
                        LineList[4].toDouble(),
                        LineList[5].toDouble(),
                        LineList[3].toDouble());
                    session_.model_locations_.SavePose(
                        ui.image_list_widget->currentRow(),
                        selected[0].row(),
                        loaded_pose);
                    vw->set_model_position_at_index(
                        selected[0].row(),
                        loaded_pose.x,
                        loaded_pose.y,
                        loaded_pose.z);
                    vw->set_model_orientation_at_index(
                        selected[0].row(),
                        loaded_pose.xa,
                        loaded_pose.ya,
                        loaded_pose.za);
                    ui.qvtk_widget->update();
                    ui.qvtk_widget->renderWindow()->Render();
                    ui.qvtk_cpv->update();
                    ui.qvtk_cpv->renderWindow()->Render();
                } else {
                    QMessageBox::critical(
                        this, "Error!", "Invalid Pose!", QMessageBox::Ok);
                    inputFile.close();
                    return;
                }
            } else {
                QMessageBox::critical(
                    this, "Error!", "Invalid Pose File!", QMessageBox::Ok);
                inputFile.close();
                return;
            }
        }

        inputFile.close();
    }
}

/*Copy Previous Pose*/
void MainScreen::on_actionCopy_Previous_Pose_triggered() {
    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    if (ui.image_list_widget->currentRow() < 0 || selected.size() == 0) {
        QMessageBox::critical(
            this,
            "Error!",
            "Select Model and Load Frames First!",
            QMessageBox::Ok);
        return;
    }

    if (ui.multiple_model_radio_button->isChecked()) {
        QMessageBox::critical(
            this,
            "Error!",
            "Must Be in Single Model Selection Mode to Load Kinematics!",
            QMessageBox::Ok);
        return;
    }
    Point6D prev_pose = session_.model_locations_.GetPose(
        ui.image_list_widget->currentRow() - 1, selected[0].row());
    session_.model_locations_.SavePose(
        ui.image_list_widget->currentRow(),
        ui.model_list_widget->currentRow(),
        prev_pose);
    vw->set_model_position_at_index(
        selected[0].row(), prev_pose.x, prev_pose.y, prev_pose.z);
    vw->set_model_orientation_at_index(
        selected[0].row(), prev_pose.xa, prev_pose.ya, prev_pose.za);
    coronal_vw->set_model_position_at_index(
        selected[0].row(), prev_pose.x, prev_pose.y, prev_pose.z);
    coronal_vw->set_model_orientation_at_index(
        selected[0].row(), prev_pose.xa, prev_pose.ya, prev_pose.za);
    ui.qvtk_widget->update();
    ui.qvtk_widget->renderWindow()->Render();
    ui.qvtk_cpv->update();
    ui.qvtk_cpv->renderWindow()->Render();
}

// For passing current pose into sym_trap window
Point6D MainScreen::copy_current_pose() {
    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    if (ui.image_list_widget->currentRow() < 0 || selected.size() == 0) {
        QMessageBox::critical(
            this,
            "Error!",
            "Select Model and Load Frames First!",
            QMessageBox::Ok);
        return Point6D();
    }

    if (ui.multiple_model_radio_button->isChecked()) {
        QMessageBox::critical(
            this,
            "Error!",
            "Must Be in Single Model Selection Mode to Load Kinematics!",
            QMessageBox::Ok);
        return Point6D();
    }
    Point6D pose = session_.model_locations_.GetPose(
        ui.image_list_widget->currentRow(), selected[0].row());
    return pose;
}

/*Copy Next Pose*/

void MainScreen::on_actionCopy_Next_Pose_triggered() {
    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    if (ui.image_list_widget->currentRow() < 0 || selected.size() == 0) {
        QMessageBox::critical(
            this,
            "Error!",
            "Select Model and Load Frames First!",
            QMessageBox::Ok);
        return;
    }

    if (ui.multiple_model_radio_button->isChecked()) {
        QMessageBox::critical(
            this,
            "Error!",
            "Must Be in Single Model Selection Mode to Load Kinematics!",
            QMessageBox::Ok);
        return;
    }
    Point6D next_pose = session_.model_locations_.GetPose(
        ui.image_list_widget->currentRow() + 1, selected[0].row());
    session_.model_locations_.SavePose(
        ui.image_list_widget->currentRow(),
        ui.model_list_widget->currentRow(),
        next_pose);

    vw->set_model_position_at_index(
        selected[0].row(), next_pose.x, next_pose.y, next_pose.z);
    vw->set_model_orientation_at_index(
        selected[0].row(), next_pose.xa, next_pose.ya, next_pose.za);
    coronal_vw->set_model_position_at_index(
        selected[0].row(), next_pose.x, next_pose.y, next_pose.z);
    coronal_vw->set_model_orientation_at_index(
        selected[0].row(), next_pose.xa, next_pose.ya, next_pose.za);
    ui.qvtk_widget->update();
    ui.qvtk_widget->renderWindow()->Render();
    ui.qvtk_cpv->update();
    ui.qvtk_cpv->renderWindow()->Render();
}

/*Load Kinematics*/
void MainScreen::on_actionLoad_Kinematics_triggered() {
    // Load Kinematics to Frames
    // Selection Check
    /*Load Models Selected Indices*/
    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    if (ui.image_list_widget->currentRow() < 0 || selected.size() == 0) {
        QMessageBox::critical(
            this,
            "Error!",
            "Select Model and Load Frames First!",
            QMessageBox::Ok);
        return;
    }

    // Must be in Single Selection Mode to Load Pose
    if (ui.multiple_model_radio_button->isChecked()) {
        QMessageBox::critical(
            this,
            "Error!",
            "Must Be in Single Model Selection Mode to Load Kinematics!",
            QMessageBox::Ok);
        return;
    }

    // Load Frame Dialog
    QString LoadPoseExtension = QFileDialog::getOpenFileName(
        this,
        tr("Load Kinematics"),
        ".",
        tr("JTA Kinematics File (*.jtak);; "
           "JointTrack Kinematics File (*.jts);; "
           "Kinematics File (*.txt)"));
    QFile inputFile(LoadPoseExtension);
    if (inputFile.open(QIODevice::ReadOnly)) {
        QTextStream in(&inputFile);
        QStringList InputList =
            in.readAll().split(QRegExp("[\r\n]"), Qt::SkipEmptyParts);
        if (InputList.size() == 0) {
            QMessageBox::critical(
                this, "Error!", "Invalid Kinematics File!", QMessageBox::Ok);
            inputFile.close();
            return;
        }
        if (InputList[0] == "JTA_EULER_KINEMATICS" ||
            InputList[0] == "JT_EULER_312") {
            for (int i = 2; i < InputList.length() &&
                            (i - 2) < ui.image_list_widget->count();
                 i++) {
                QStringList LineList =
                    InputList[i].split(QRegExp("[,]"), Qt::SkipEmptyParts);
                if (LineList.size() >= 6) {
                    LineList[0].replace(" ", "");
                    if (LineList[0] != "NOT_OPTIMIZED") {
                        auto loaded_pose = Point6D(
                            LineList[0].toDouble(),
                            LineList[1].toDouble(),
                            LineList[2].toDouble(),
                            LineList[4].toDouble(),
                            LineList[5].toDouble(),
                            LineList[3].toDouble());
                        session_.model_locations_.SavePose(
                            i - 2,
                            ui.model_list_widget->currentRow(),
                            loaded_pose);
                    }
                }
            }
            if (ui.image_list_widget->currentRow() >= 0) {
                Point6D loaded_pose = session_.model_locations_.GetPose(
                    ui.image_list_widget->currentRow(), selected[0].row());
                vw->set_model_position_at_index(
                    selected[0].row(),
                    loaded_pose.x,
                    loaded_pose.y,
                    loaded_pose.z);
                vw->set_model_orientation_at_index(
                    selected[0].row(),
                    loaded_pose.xa,
                    loaded_pose.ya,
                    loaded_pose.za);
                coronal_vw->set_model_position_at_index(
                    selected[0].row(),
                    loaded_pose.x,
                    loaded_pose.y,
                    loaded_pose.z);
                coronal_vw->set_model_orientation_at_index(
                    selected[0].row(),
                    loaded_pose.xa,
                    loaded_pose.ya,
                    loaded_pose.za);
                ui.qvtk_widget->update();
                ui.qvtk_widget->renderWindow()->Render();
                ui.qvtk_cpv->update();
                ui.qvtk_cpv->renderWindow()->Render();
            }
        } else {
            QMessageBox::critical(
                this, "Error!", "Invalid Kinematics File!", QMessageBox::Ok);
            inputFile.close();
            return;
        }
        inputFile.close();
    }
}

// Start Symtrap Optimizer
void MainScreen::optimizer_launch_slot() {
    if (!sym_trap_running) {
        LaunchOptimizer("Sym_Trap");
    }
}

/*Stop Optimizer*/
void MainScreen::on_actionStop_Optimizer_triggered() {
    if (ui.actionStop_Optimizer->isEnabled()) {
        emit StopOptimizer();
        QMessageBox::warning(
            this, "Warning!", "Optimizer stopped!", QMessageBox::Ok);
    }
}

/*View Menu*/
void MainScreen::on_actionReset_View_triggered() {
    /*Reset to Model Interaction Mode*/
    if (ui.actionModel_Interaction_Mode->isChecked()) {
        renderer->GetActiveCamera()->SetViewUp(0, 1, 0);
        renderer->GetActiveCamera()->SetPosition(0, 0, 0);
        renderer->GetActiveCamera()->SetFocalPoint(
            0,
            0,
            -1 * session_.calibration_file_.camera_A_principal_.principal_distance_ /
                session_.calibration_file_.camera_A_principal_.pixel_pitch_);
        renderer->GetActiveCamera()->SetClippingRange(
            .1,
            2.0 * session_.calibration_file_.camera_A_principal_.principal_distance_ /
                session_.calibration_file_.camera_A_principal_.pixel_pitch_);
        if (session_.loaded_frames.size() > 0) {
            renderer->GetActiveCamera()->SetViewAngle(CalculateViewingAngle(
                session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                    .GetOriginalImage()
                    .cols,
                session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                    .GetOriginalImage()
                    .rows,
                true));
        }
        ui.qvtk_widget->renderWindow()->GetInteractor()->SetInteractorStyle(
            key_press_vtk);
        ui.qvtk_widget->update();
        ui.qvtk_widget->renderWindow()->Render();
        ui.qvtk_cpv->update();
        ui.qvtk_cpv->renderWindow()->Render();
    } else {
        /*Rest to Camera Interaction Mode*/
        renderer->GetActiveCamera()->SetViewUp(0, 1, 0);
        renderer->GetActiveCamera()->SetPosition(0, 0, 0);
        renderer->GetActiveCamera()->SetFocalPoint(
            0,
            0,
            -1 * session_.calibration_file_.camera_A_principal_.principal_distance_ /
                session_.calibration_file_.camera_A_principal_.pixel_pitch_);
        renderer->GetActiveCamera()->SetClippingRange(
            .1,
            2.0 * session_.calibration_file_.camera_A_principal_.principal_distance_ /
                session_.calibration_file_.camera_A_principal_.pixel_pitch_);
        if (session_.loaded_frames.size() > 0) {
            renderer->GetActiveCamera()->SetViewAngle(CalculateViewingAngle(
                session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                    .GetOriginalImage()
                    .cols,
                session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                    .GetOriginalImage()
                    .rows,
                true));
        }
        QModelIndexList selected =
            ui.model_list_widget->selectionModel()->selectedRows();
        if (selected.size() > 0) {
            renderer->GetActiveCamera()->SetFocalPoint(
                0, 0, model_actor_list[selected[0].row()]->GetPosition()[2]);
        } else {
            renderer->GetActiveCamera()->SetFocalPoint(
                0,
                0,
                -1 * session_.calibration_file_.camera_A_principal_.principal_distance_ /
                    session_.calibration_file_.camera_A_principal_.pixel_pitch_);
        }
        ui.qvtk_widget->renderWindow()->GetInteractor()->SetInteractorStyle(
            camera_style_interactor);
        ui.qvtk_widget->update();
        ui.qvtk_widget->renderWindow()->Render();
        ui.qvtk_cpv->update();
        ui.qvtk_cpv->renderWindow()->Render();
    }
};

void MainScreen::on_actionReset_Normal_Up_triggered() {
    renderer->GetActiveCamera()->SetViewUp(0, 1, 0);
    ui.qvtk_widget->update();
    ui.qvtk_widget->renderWindow()->Render();
    ui.qvtk_cpv->update();
    ui.qvtk_cpv->renderWindow()->Render();
}

void MainScreen::on_actionModel_Interaction_Mode_triggered() {
    if (session_.loaded_models.size() == 0 || session_.loaded_frames.size() == 0) {
        QMessageBox::critical(
            this,
            "Error!",
            "Please load at least one model and one image before "
            "changing the interaction mode!",
            QMessageBox::Ok);
        ui.actionModel_Interaction_Mode->setChecked(true);
        return;
    }
    ui.qvtk_widget->renderWindow()->GetInteractor()->SetInteractorStyle(
        key_press_vtk);
    ui.qvtk_widget->update();
    ui.qvtk_widget->renderWindow()->Render();
    ui.qvtk_cpv->update();
    ui.qvtk_cpv->renderWindow()->Render();
};

void MainScreen::on_actionCamera_Interaction_Mode_triggered() {
    if (session_.loaded_models.size() == 0 || session_.loaded_frames.size() == 0) {
        QMessageBox::critical(
            this,
            "Error!",
            "Please load at least one model and one image before "
            "changing the interaction mode!",
            QMessageBox::Ok);
        ui.actionModel_Interaction_Mode->setChecked(true);
        return;
    }
    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    if (selected.size() > 0) {
        renderer->GetActiveCamera()->SetFocalPoint(
            0, 0, model_actor_list[selected[0].row()]->GetPosition()[2]);
    } else {
        renderer->GetActiveCamera()->SetFocalPoint(
            0,
            0,
            -1 * session_.calibration_file_.camera_A_principal_.principal_distance_ /
                session_.calibration_file_.camera_A_principal_.pixel_pitch_);
    }
    ui.qvtk_widget->renderWindow()->GetInteractor()->SetInteractorStyle(
        camera_style_interactor);
    ui.qvtk_widget->update();
    ui.qvtk_widget->renderWindow()->Render();
    ui.qvtk_cpv->update();
    ui.qvtk_cpv->renderWindow()->Render();
};

/*About Menu*/
void MainScreen::on_actionAbout_JointTrack_Auto_triggered() {
    /*Open About Window*/
    About abt;
    abt.setVersion(VER_FIRST_NUM, VER_MIDDLE_NUM, VER_LAST_NUM);
    abt.exec();
}

void MainScreen::on_actionSegment_FemHR_triggered() {
    /*Deserialize the ScriptModule from a file using torch::jit::load().
    NOTE: Because this is a traced model, it can only be used with a batch
    size of 1. To work around this, one must convert to Torch Script via
    Annotation.*/

    QString pt_model_location = QFileDialog::getOpenFileName(
        this,
        tr("Load Trained Femoral Segmentation Architecture"),
        ".",
        tr("Torch File (*.pt)"));
    if (pt_model_location.toStdString() != "") {
        segmentHelperFunction(pt_model_location.toStdString(), 1024, 1024);
    }
}

void MainScreen::on_actionSegment_TibHR_triggered() {
    /*Deserialize the ScriptModule from a file using torch::jit::load().
    NOTE: Because this is a traced model, it can only be used with a batch
    size of 1. To work around this, one must convert to Torch Script via
    Annotation.*/

    QString pt_model_location = QFileDialog::getOpenFileName(
        this,
        tr("Load Trained Tibial Segmentation Architecture"),
        ".",
        tr("Torch File (*.pt)"));
    if (pt_model_location.toStdString() != "") {
        segmentHelperFunction(pt_model_location.toStdString(), 1024, 1024);
    }
}

void MainScreen::update_image_list_widget() {
    /*If Viewing Inverted Images Update*/
    if (ui.inverted_image_radio_button->isChecked()) {
        vw->update_display_background_to_inverted_image(
            this->curr_frame(), ui.camera_A_radio_button->isChecked());
    }
    /*If Viewing Edge Images Update*/
    else if (ui.edges_image_radio_button->isChecked()) {
        vw->update_display_background_to_edge_image(
            this->curr_frame(), ui.camera_A_radio_button->isChecked());
    }
    /*If Viewing Dilated Images Update*/
    else if (ui.dilation_image_radio_button->isChecked()) {
        vw->update_display_background_to_dilation_image(
            this->curr_frame(), ui.camera_A_radio_button->isChecked());
    } else if (ui.original_image_radio_button->isChecked()) {
        vw->update_display_background_to_original_image(
            this->curr_frame(), ui.original_image_radio_button->isChecked());
    }
    ui.qvtk_widget->update();
    ui.qvtk_widget->renderWindow()->Render();
    ui.qvtk_cpv->update();
    ui.qvtk_cpv->renderWindow()->Render();
}

void MainScreen::segmentHelperFunction(
    std::string pt_model_location,
    unsigned int input_width,
    unsigned int input_height) {
    int dilation_val = 0;
    session_.trunk_manager_.getActiveCostFunctionClass()->getIntParameterValue(
        "Dilation", dilation_val);

    worker_orchestrator_->ConfigureSegmentation(
        {.pt_model_location = std::move(pt_model_location),
         .input_width = input_width,
         .input_height = input_height,
         .black_sil_used = ui.actionBlack_Implant_Silhouettes_in_Original_Image_s->isChecked(),
         .dilation_val = dilation_val,
         .aperture = ui.aperture_spin_box->value(),
         .low_threshold = ui.low_threshold_slider->value(),
         .high_threshold = ui.high_threshold_slider->value()});

    ui.pose_progress->setValue(0);
    ui.pose_progress->setVisible(true);
    ui.pose_label->setText("Initializing PyTorch...");
    ui.pose_label->setVisible(true);

    worker_orchestrator_->StartSegmentation(session_);
}

void MainScreen::onImageSegmented(int index, cv::Mat segmented, bool isBiplane) {
    int dilation_val = 0;
    session_.trunk_manager_.getActiveCostFunctionClass()->getIntParameterValue(
        "Dilation", dilation_val);

    if (!isBiplane) {
        segmented.copyTo(session_.loaded_frames[index].GetInvertedImage());
        session_.loaded_frames[index].SetEdgeImage(
            ui.aperture_spin_box->value(),
            ui.low_threshold_slider->value(),
            ui.high_threshold_slider->value(),
            true);
        session_.loaded_frames[index].SetDilatedImage(dilation_val);
        session_.loaded_frames[index].SetDistanceMap();
        session_.loaded_frames[index].setCurvatureHeatmaps();
    } else {
        segmented.copyTo(session_.loaded_frames_B[index].GetInvertedImage());
        session_.loaded_frames_B[index].SetEdgeImage(
            ui.aperture_spin_box->value(),
            ui.low_threshold_slider->value(),
            ui.high_threshold_slider->value(),
            true);
        session_.loaded_frames_B[index].SetDilatedImage(dilation_val);
    }

    vw->render_scene();
    coronal_vw->render_scene();
}

void MainScreen::onSegmentationProgress(int value, QString status) {
    ui.pose_progress->setValue(value);
    ui.pose_label->setText(status);
}

void MainScreen::onSegmentationFinished(bool success, QString errorMessage) {
    if (!success) {
        QMessageBox::critical(this, "Segmentation Error", errorMessage);
    }

    if (ui.image_list_widget->currentIndex().row() >= 0) {
        update_image_list_widget();
    }

    ui.pose_progress->setVisible(false);
    ui.pose_label->setVisible(false);
}

void MainScreen::on_actionReset_Remove_All_Segmentation_triggered() {
    for (int i = 0; i < ui.image_list_widget->count(); i++) {
        session_.loaded_frames[i].ResetFromOriginal();
    }

    if (ui.image_list_widget->currentIndex().row() >= 0) {
        update_image_list_widget();
    }
}

void MainScreen::on_actionEstimate_Femoral_Implant_s_triggered() {
    // Must be in Single Selection Mode to Load Pose
    if (ui.multiple_model_radio_button->isChecked()) {
        QMessageBox::critical(
            this,
            "Error!",
            "Must Be in Single Model Selection Mode to "
            "Estimate Kinematics!",
            QMessageBox::Ok);
        return;
    }

    // Must load a model
    if (session_.loaded_models.size() < 1) {
        QMessageBox::critical(
            this, "Error!", "Must load a model!", QMessageBox::Ok);
        return;
    }

    // Must have loaded image
    if (session_.loaded_frames.size() < 1) {
        QMessageBox::critical(
            this, "Error!", "Must load images!", QMessageBox::Ok);
        return;
    }

    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    std::string stl_path = session_.loaded_models[selected[0].row()].file_location_;

    QString pt_model_location = QFileDialog::getOpenFileName(
        this,
        tr("Load Trained Femoral Pose Regression Architecture"),
        ".",
        tr("Torch File (*.pt)"));

    if (pt_model_location.isEmpty()) return;

    worker_orchestrator_->ConfigureEstimation(
        {.stl_path = std::move(stl_path),
         .pt_model_location = pt_model_location.toStdString(),
         .black_sil_used = ui.actionBlack_Implant_Silhouettes_in_Original_Image_s->isChecked()});

    ui.pose_progress->setValue(0);
    ui.pose_progress->setVisible(true);
    ui.pose_label->setText("Initializing estimation...");
    ui.pose_label->setVisible(true);

    worker_orchestrator_->StartEstimation(session_);
}

void MainScreen::onPoseEstimated(int index, Point6D pose) {
    session_.model_locations_.SavePose(
        index, ui.model_list_widget->currentRow(), pose);

    if (index == ui.image_list_widget->currentIndex().row()) {
        vw->render_scene();
        coronal_vw->render_scene();
    }
}

void MainScreen::onEstimationFinished(bool success, QString errorMessage) {
    if (!success) {
        QMessageBox::critical(this, "Estimation Error", errorMessage);
    }

    ui.pose_progress->setVisible(false);
    ui.pose_label->setVisible(false);

    if (ui.image_list_widget->currentIndex().row() >= 0) {
        update_image_list_widget();
    }
}
/*Viewing Controls*/
void MainScreen::on_actionControls_triggered() {

    // Open Viewing Window Controls Window
    Controls cntrls;
    cntrls.exec();
}

/*Optimizer Window*/
void MainScreen::on_actionOptimizer_Settings_triggered() {
    /*Load the Optimizer Settings to the Window*/
    settings_control->LoadSettings(
        session_.trunk_manager_, session_.branch_manager_, session_.leaf_manager_, session_.optimizer_settings_);

    // Open Optimizer Settings Window
    settings_control->show();
}

/*Symmetry Trap Window*/

/*DRR Settings Window*/
void MainScreen::on_actionDRR_Settings_triggered() {
    /*CHeck if Loaded Models Yet*/
    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    if (selected.size() > 0) {
        /*Open DRR Window*/
        DRRTool drt(
            session_.loaded_models[selected[0].row()],
            session_.calibration_file_.camera_A_principal_,
            model_actor_list[selected[0].row()]->GetPosition()[2]);
        drt.exec();
    }
}

/*PREPROCESSOR BUTTONS*/
/*Load Calibration Button*/
void MainScreen::on_load_calibration_button_clicked() {
    calibration_service_->LoadCalibration(this, session_);
}

void MainScreen::onCalibrationLoaded(Calibration calibration) {
    Q_UNUSED(calibration);

    if (session_.calibrated_for_monoplane_viewport_) { // I loaded a single-plane
                                               // calibration
        /*Set Checked To Monoplane but disable from further clicking*/
        ui.camera_A_radio_button->setChecked(true);
        ui.camera_A_radio_button->setDisabled(true);

        /*Disable Biplane*/
        ui.camera_B_radio_button->setDisabled(true);

        /*If Already loaded images CANT HAPPEN ANYMORE AS CALIBRATION IS ONE
         * USE BUTTON*/
        /*Disable Reloading Calibration File*/
        ui.load_calibration_button->setDisabled(true);
        /*Enable View Menu Until Calbration Loaded*/
        ui.actionReset_View->setDisabled(false);
        ui.actionReset_Normal_Up->setDisabled(false);
        ui.actionModel_Interaction_Mode->setDisabled(false);
        ui.actionCamera_Interaction_Mode->setDisabled(false);
    }
    /*Biplane Calibration*/
    /*BROKEN BIPLANE HISTORIC IMPLEMENTATION*/
    else if (session_.calibrated_for_biplane_viewport_) {
        /*Set Checked To Biplane A (aka Monoplane) and Change Text Boxes*/
        ui.camera_A_radio_button->setChecked(true);
        ui.camera_A_radio_button->setEnabled(true);
        ui.camera_B_radio_button->setEnabled(true);

        /*Disable Reloading Calibration File*/
        ui.load_calibration_button->setDisabled(true);

        /*Enable View Menu Until Calbration Loaded*/
        ui.actionReset_View->setDisabled(false);
        ui.actionReset_Normal_Up->setDisabled(false);
        ui.actionModel_Interaction_Mode->setDisabled(false);
        ui.actionCamera_Interaction_Mode->setDisabled(false);
    }
}

/*Load Image Button*/
void MainScreen::on_load_image_button_clicked() {
    /*If TRUNK is Has Integer Parameter called Dilation, Update Dilation
     * Values for Viewing Purposes*/
    int dilation_val = 0;
    std::vector<jta_cost_function::Parameter<int>> active_int_params =
        session_.trunk_manager_.getActiveCostFunctionClass()->getIntParameters();
    for (int i = 0; i < active_int_params.size(); i++) {
        if (active_int_params[i].getParameterName() == "Dilation") {
            dilation_val = (session_.trunk_manager_.getActiveCostFunctionClass())
                               ->getIntParameters()
                               .at(i)
                               .getParameterValue();
        }
    }
    if (dilation_val < 0) {
        dilation_val = 0;
    }

    /*Check to See if Calibration Loaded*/
    if (session_.calibrated_for_monoplane_viewport_ == false &&
        session_.calibrated_for_biplane_viewport_ == false) {
        QMessageBox::critical(
            this, "Error!", "Load Calibration First!", QMessageBox::Ok);
        return;
    }

    /*If MONOPLANE Only*/
    if (session_.calibrated_for_monoplane_viewport_) {
        const QStringList tiff_file_extensions = QFileDialog::getOpenFileNames(
            this,
            tr("Load Image(s)"),
            ".",
            tr("Image File(s) (*.tif *.tiff *.png)"));
        const auto result = image_loading_service_->LoadImages(
            session_.loaded_frames,
            session_.loaded_frames_B,
            tiff_file_extensions,
            {},
            ui.aperture_spin_box->value(),
            ui.low_threshold_slider->value(),
            ui.high_threshold_slider->value(),
            dilation_val);

        for (int i = 0; i < result.loaded_count; ++i) {
            session_.loaded_frames.push_back(result.frames_a[i]);
            ui.image_list_widget->addItem(result.display_names[i]);
            session_.model_locations_.LoadNewFrame();
        }

        vw->set_loaded_frames(session_.loaded_frames);

        // If No Loaded Frames, Default Select First
        if (ui.image_list_widget->currentRow() < 0 &&
            session_.loaded_frames.size() > 0) {
            ui.image_list_widget->setCurrentRow(0);
        }

        // this->vw->set_loaded_frames(session_.loaded_frames);

    } else if (session_.calibrated_for_biplane_viewport_) {
        // Load TIFF images for Camera A and Camera B - Must Be Same Amount
        // or Error and None Will Load!
        const QStringList tiff_file_extensions_camera_a = QFileDialog::getOpenFileNames(
            this,
            tr("Load Image(s) for Camera A"),
            ".",
            tr("Image File(s) (*.tif *.tiff)"));
        const QStringList tiff_file_extensions_camera_b = QFileDialog::getOpenFileNames(
            this,
            tr("Load Image(s) for Camera B"),
            ".",
            tr("Image File(s) (*.tif *.tiff)"));

        const auto result = image_loading_service_->LoadImages(
            session_.loaded_frames,
            session_.loaded_frames_B,
            tiff_file_extensions_camera_a,
            tiff_file_extensions_camera_b,
            ui.aperture_spin_box->value(),
            ui.low_threshold_slider->value(),
            ui.high_threshold_slider->value(),
            dilation_val);

        for (int i = 0; i < result.loaded_count; ++i) {
            session_.loaded_frames.push_back(result.frames_a[i]);
            session_.loaded_frames_B.push_back(result.frames_b[i]);
            ui.image_list_widget->addItem(result.display_names[i]);
            session_.model_locations_.LoadNewFrame();
        }

        // If No Loaded Frames, Default Select First
        if (ui.image_list_widget->currentRow() < 0 &&
            session_.loaded_frames.size() > 0) {
            ui.image_list_widget->setCurrentRow(0);
        }
        vw->set_loaded_frames(session_.loaded_frames);
        vw->set_loaded_frames_b(session_.loaded_frames_B);
    }
}

/*Load Model Button*/
void MainScreen::on_load_model_button_clicked() {
    /*Check to See if Calibration Loaded*/
    if (session_.calibrated_for_monoplane_viewport_ == false &&
        session_.calibrated_for_biplane_viewport_ == false) {
        QMessageBox::critical(
            this, "Error!", "Load Calibration First!", QMessageBox::Ok);
        return;
    }

    // Load CAD Model
    const QStringList cad_file_extensions = QFileDialog::getOpenFileNames(
        this, tr("Load Implant Model(s)"), ".", tr("CAD File(s) (*.stl)"));

    const auto result =
        image_loading_service_->LoadModels(session_.loaded_models, cad_file_extensions);

    vw->load_models(result.models);
    coronal_vw->load_models(result.models);
    for (int i = 0; i < result.models.size(); ++i) {
        session_.loaded_models.push_back(result.models[i]);
        session_.model_locations_.LoadNewModel(session_.calibration_file_);
    }
    for (int i = 0; i < result.models.size(); ++i) {
        if (!result.models[i].initialized_correctly_) {
            QMessageBox::warning(
                this,
                "Warning!",
                "It is Possible that " + result.model_names[i] + " (" +
                    cad_file_extensions[i] + ") " +
                    " is an invalid or corrupted STL file format. "
                    "Proceed with caution!",
                QMessageBox::Ok);
        }
    }

    // Populate Model List Widget
    for (int i = 0; i < result.model_names.size(); ++i) {
        ui.model_list_widget->addItem(result.model_names[i]);
    }
    vw->load_3d_models_into_actor_and_mapper_list();
    coronal_vw->load_3d_models_into_actor_and_mapper_list();
    vw->load_model_actors_and_mappers_with_3d_data();
    // If No Loaded Models, Default Select First
    if (ui.model_list_widget->selectionModel()->selectedRows().size() == 0) {
        ui.model_list_widget->setCurrentRow(0);
    }
    if (!session_.loaded_frames.empty() && session_.calibration_file_.type_ == "UF") {
        vw->set_vtk_camera_from_calibration_and_image_size_if_jta(
            session_.calibration_file_,
            session_.loaded_frames[0].GetOriginalImage().cols,
            session_.loaded_frames[0].GetOriginalImage().rows);
        // coronal_vw->set_vtk_camera_from_calibration_and_image_size_if_jta(
        //     session_.calibration_file_,
        //     session_.loaded_frames[0].GetOriginalImage().cols,
        //     session_.loaded_frames[0].GetOriginalImage().rows);
    } else if (
        !session_.loaded_frames.empty() &&
        session_.calibration_file_.type_ == "Denver") {
        vw->set_vtk_camera_from_calibration_and_image_if_camera_matrix(
            session_.calibration_file_,
            session_.loaded_frames[0].GetOriginalImage().cols,
            session_.loaded_frames[0].GetOriginalImage().rows);
        // coronal_vw->set_vtk_camera_from_calibration_and_image_if_camera_matrix(
        //     session_.calibration_file_,
        //     session_.loaded_frames[0].GetOriginalImage().cols,
        //     session_.loaded_frames[0].GetOriginalImage().rows);
    }
}

/*Biplane View Button (Camera A,Camera B*/
/*Biplane View A OR Monoplane*/
void MainScreen::on_camera_A_radio_button_clicked() {
    const int frame_index = ui.image_list_widget->currentIndex().row();
    if (frame_index < 0) {
        return;
    }

    jta_gui::SelectionSyncState state;
    state.frame_index = frame_index;
    state.previous_frame_index = previous_frame_index_;
    state.camera_a_selected = true;
    state.currently_optimizing = currently_optimizing_;
    state.calibrated_for_biplane_viewport =
        session_.calibrated_for_biplane_viewport_;
    state.actor_text_visible = actor_text->GetTextProperty()->GetOpacity() > 0.5;
    state.opacity_mode = ResolveModelOpacityMode(ui);
    state.selected_model_indices =
        ToRowVector(ui.model_list_widget->selectionModel()->selectedRows());

    if (!scene_controller_->OnCameraASelected(state)) {
        return;
    }

    update_image_list_widget();
}

/*Biplane View B*/
void MainScreen::on_camera_B_radio_button_clicked() {
    const int frame_index = ui.image_list_widget->currentIndex().row();
    if (frame_index < 0) {
        return;
    }

    jta_gui::SelectionSyncState state;
    state.frame_index = frame_index;
    state.previous_frame_index = previous_frame_index_;
    state.camera_a_selected = false;
    state.currently_optimizing = currently_optimizing_;
    state.calibrated_for_biplane_viewport =
        session_.calibrated_for_biplane_viewport_;
    state.actor_text_visible = actor_text->GetTextProperty()->GetOpacity() > 0.5;
    state.opacity_mode = ResolveModelOpacityMode(ui);
    state.selected_model_indices =
        ToRowVector(ui.model_list_widget->selectionModel()->selectedRows());

    if (!scene_controller_->OnCameraBSelected(state)) {
        return;
    }

    update_image_list_widget();
}

/*List Widgets (Model and Frame)*/
/*Frame Widget*/
/* this is where we are setting the current background */
void MainScreen::on_image_list_widget_itemSelectionChanged() {
    /*Make Sure A View is Selected*/
    if (!ui.original_image_radio_button->isChecked() &&
        !ui.inverted_image_radio_button->isChecked() &&
        !ui.edges_image_radio_button->isChecked() &&
        !ui.dilation_image_radio_button->isChecked()) {
        ui.original_image_radio_button->setChecked(true);
    }

    /*Save Last Pair Pose if not currently optimizing*/
    if (!currently_optimizing_) {
        SaveLastPose();
    }

    /*Update Last Viewed Index as This One*/
    previous_frame_index_ = ui.image_list_widget->currentIndex().row();

    const int frame_index = ui.image_list_widget->currentIndex().row();
    if (frame_index < 0) {
        return;
    }

    jta_gui::SelectionSyncState state;
    state.frame_index = frame_index;
    state.camera_a_selected = ui.camera_A_radio_button->isChecked();
    state.currently_optimizing = currently_optimizing_;
    state.calibrated_for_biplane_viewport =
        session_.calibrated_for_biplane_viewport_;
    state.actor_text_visible = actor_text->GetTextProperty()->GetOpacity() > 0.5;
    state.opacity_mode = ResolveModelOpacityMode(ui);
    state.selected_model_indices =
        ToRowVector(ui.model_list_widget->selectionModel()->selectedRows());

    if (!scene_controller_->OnImageSelectionChanged(state)) {
        return;
    }

    update_image_list_widget();
}

QModelIndexList MainScreen::selected_model_indices() {
    return ui.model_list_widget->selectionModel()->selectedRows();
}
void MainScreen::remove_background_highlights_from_model_list_widget() {
    for (int i = 0; i < session_.loaded_models.size(); i++) {
        ui.model_list_widget->item(i)->setBackground(Qt::transparent);
    }
}
void MainScreen::print_selected_item() {
    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    for (int i = 0; i < selected.size(); i++) {
    }
}

/*Model Widget*/
void MainScreen::on_model_list_widget_itemSelectionChanged() {
    /*Save Last Pair Pose if not currently optimizing*/
    if (!currently_optimizing_) {
        SaveLastPose(); // Needs Work
    }
    /*Update Last Viewed Index as This One*/
    previous_model_indices_ =
        ui.model_list_widget->selectionModel()->selectedRows();

    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    /*Keep at least one selected row once model list has entries*/
    if (selected.size() == 0) {
        if (ui.model_list_widget->currentIndex().row() >= 0) {
            ui.model_list_widget
                ->item(ui.model_list_widget->currentIndex().row())
                ->setSelected(true);
            return;
        }
    }

    // Set a style sheet for selected items in the list widget, controls
    // background colors for .stl model names
    ui.model_list_widget->setStyleSheet(
        "QListView::item{background-color: rgb("
        ");}" // Leaving it blank sets unselected items to share the
              // background color of JTML
        "QListView::item:selected{background-color: rgb(250, 70, "
        "22);}");

    jta_gui::SelectionSyncState state;
    state.frame_index = ui.image_list_widget->currentIndex().row();
    state.camera_a_selected = ui.camera_A_radio_button->isChecked();
    state.currently_optimizing = currently_optimizing_;
    state.calibrated_for_biplane_viewport =
        session_.calibrated_for_biplane_viewport_;
    state.actor_text_visible = actor_text->GetTextProperty()->GetOpacity() > 0.5;
    state.opacity_mode = ResolveModelOpacityMode(ui);
    state.selected_model_indices = ToRowVector(selected);

    scene_controller_->OnModelSelectionChanged(state);
}

/*Make Selected Actor Principal from VTK*/
void MainScreen::VTKMakePrincipalSignal(vtkActor* new_principal_actor) {
    jta_gui::SelectionSyncState state;
    state.frame_index = ui.image_list_widget->currentIndex().row();
    state.camera_a_selected = ui.camera_A_radio_button->isChecked();
    state.actor_text_visible = actor_text->GetTextProperty()->GetOpacity() > 0.5;
    state.opacity_mode = ResolveModelOpacityMode(ui);
    state.selected_model_indices =
        ToRowVector(ui.model_list_widget->selectionModel()->selectedRows());

    QString error_message;
    if (scene_controller_->OnMakePrincipalActor(new_principal_actor, state, error_message)) {
        return;
    }

    if (!error_message.isEmpty()) {
        QMessageBox::critical(this, "Error!", error_message, QMessageBox::Ok);
    }
}

/*Multiple Selection For Models Radio buttons*/
void MainScreen::on_single_model_radio_button_clicked() {
    std::cout << "You've pressed single model radio button" << std::endl;
    /*Change Selection Mode*/
    ui.model_list_widget->setSelectionMode(QAbstractItemView::SingleSelection);

    /*If Multiple Selections Choose First One Selected*/
    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    if (selected.size() > 0) ui.model_list_widget->setCurrentIndex(selected[0]);
};

void MainScreen::on_multiple_model_radio_button_clicked() {
    ui.model_list_widget->setSelectionMode(QAbstractItemView::MultiSelection);
}

/*Display Original Image*/
void MainScreen::on_original_image_radio_button_clicked() {
    if (ui.camera_A_radio_button->isChecked()) {
        vw->update_display_background_to_original_image(
            this->curr_frame(), true);
    } else {
        vw->update_display_background_to_original_image(
            this->curr_frame(), false);
    }
    ui.qvtk_widget->update();
    ui.qvtk_widget->renderWindow()->Render();
}

/*Display Inverted Image*/
void MainScreen::on_inverted_image_radio_button_clicked() {
    if (ui.camera_A_radio_button->isChecked()) {
        vw->update_display_background_to_inverted_image(
            this->curr_frame(), true);
    } else {
        vw->update_display_background_to_inverted_image(
            this->curr_frame(), false);
    }
    ui.qvtk_widget->update();
    ui.qvtk_widget->renderWindow()->Render();
}

/*Display Edge Detected Image*/
void MainScreen::on_edges_image_radio_button_clicked() {
    if (ui.camera_A_radio_button->isChecked()) {
        vw->update_display_background_to_edge_image(this->curr_frame(), true);
    } else {
        vw->update_display_background_to_edge_image(this->curr_frame(), false);
    }
    ui.qvtk_widget->update();
    ui.qvtk_widget->renderWindow()->Render();
}

/*Display Dilated Image*/
void MainScreen::on_dilation_image_radio_button_clicked() {
    if (ui.camera_A_radio_button->isChecked()) {
        vw->update_display_background_to_dilation_image(
            this->curr_frame(), true);
    } else {
        vw->update_display_background_to_dilation_image(
            this->curr_frame(), false);
    }
    ui.qvtk_widget->update();
    ui.qvtk_widget->renderWindow()->Render();
}

void MainScreen::on_original_model_radio_button_clicked() {
    if (!ui.original_model_radio_button->isChecked()) {
        return;
    }

    scene_controller_->SyncSelectedModelOpacity(
        ToRowVector(ui.model_list_widget->selectionModel()->selectedRows()),
        jta_gui::ModelOpacityMode::Original);
}

void MainScreen::on_solid_model_radio_button_clicked() {
    if (!ui.solid_model_radio_button->isChecked()) {
        return;
    }

    scene_controller_->SyncSelectedModelOpacity(
        ToRowVector(ui.model_list_widget->selectionModel()->selectedRows()),
        jta_gui::ModelOpacityMode::Solid);
}

void MainScreen::on_transparent_model_radio_button_clicked() {
    if (!ui.transparent_model_radio_button->isChecked()) {
        return;
    }

    scene_controller_->SyncSelectedModelOpacity(
        ToRowVector(ui.model_list_widget->selectionModel()->selectedRows()),
        jta_gui::ModelOpacityMode::Transparent);
}

void MainScreen::on_wireframe_model_radio_button_clicked() {
    if (!ui.wireframe_model_radio_button->isChecked()) {
        return;
    }

    scene_controller_->SyncSelectedModelOpacity(
        ToRowVector(ui.model_list_widget->selectionModel()->selectedRows()),
        jta_gui::ModelOpacityMode::Wireframe);
}

/*KeyPress Event*/
void MainScreen::keyPressEvent(QKeyEvent* event) {
    /*Stop Optimizer*/
    if (event->key() == Qt::Key_Escape) {
        if (ui.actionStop_Optimizer->isEnabled()) {
            emit StopOptimizer();
            QMessageBox::warning(
                this, "Warning!", "Optimizer stopped!", QMessageBox::Ok);
        }
    }
    /*Toggle Information View*/
    if (event->key() == Qt::Key_I) {
        std::cout << "I!" << std::endl;
        if (actor_text->GetTextProperty()->GetOpacity() > 0.5) {
            actor_text->GetTextProperty()->SetOpacity(0.0);
        } else {
            actor_text->GetTextProperty()->SetOpacity(1.0);
        }
        ui.qvtk_widget->update();
        ui.qvtk_widget->renderWindow()->Render();
    }
    /* Update cpv on left click ISNT WORKING :( */
    if (event->key() == Qt::Key_Space) {
        QModelIndexList selected =
            ui.model_list_widget->selectionModel()->selectedRows();
        std::cout << "Left Click!" << std::endl;
        /*If Camera A View*/
        for (int i = 0; i < selected.size(); i++) {
            if (ui.camera_A_radio_button->isChecked()) {
                /*Set Model Pose*/
                Point6D loaded_pose = session_.model_locations_.GetPose(
                    ui.image_list_widget->currentIndex().row(),
                    selected[i].row());
                coronal_vw->set_model_position_at_index(
                    selected[i].row(),
                    loaded_pose.x,
                    loaded_pose.y,
                    loaded_pose.z);
                coronal_vw->set_model_orientation_at_index(
                    selected[i].row(),
                    loaded_pose.xa,
                    loaded_pose.ya,
                    loaded_pose.za);
            } else {
                /*Else, Camera B View*/
                /*Convert To relative Camera B Pose as storage is done in
                 * camera A coordinates and rotations*/
                Point6D loaded_pose = session_.model_locations_.GetPose(
                    ui.image_list_widget->currentIndex().row(),
                    selected[i].row());
                Point6D relative_B_pose =
                    session_.calibration_file_.convert_Pose_A_to_Pose_B(loaded_pose);
                coronal_vw->set_model_position_at_index(
                    selected[i].row(),
                    relative_B_pose.x,
                    relative_B_pose.y,
                    relative_B_pose.z);
                coronal_vw->set_model_orientation_at_index(
                    selected[i].row(),
                    relative_B_pose.xa,
                    relative_B_pose.ya,
                    relative_B_pose.za);
            }
        }
        ui.qvtk_cpv->update();
        ui.qvtk_cpv->renderWindow()->Render();
    }
}

void MainScreen::VTKEscapeSignal() {
    if (currently_optimizing_ && ui.actionStop_Optimizer->isEnabled()) {
        emit StopOptimizer();
        QMessageBox::warning(
            this, "Warning!", "Optimizer stopped!", QMessageBox::Ok);
    }
}

/*Edge Detection Buttons*/
void MainScreen::on_aperture_spin_box_valueChanged() {
    /*Make Sure Images Loaded First*/
    if (session_.loaded_frames.size() > 0) {
        /*Get High Value from Frame*/
        int low_val = LOW_THRESH;
        int high_val = HIGH_THRESH;
        if (ui.camera_A_radio_button->isChecked()) {
            low_val = session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                          .GetLowThreshold();
            high_val = session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                           .GetHighThreshold();
        } else if (
            ui.camera_B_radio_button->isChecked() &&
            session_.calibrated_for_biplane_viewport_) {
            low_val =
                session_.loaded_frames_B[ui.image_list_widget->currentIndex().row()]
                    .GetLowThreshold();
            high_val =
                session_.loaded_frames_B[ui.image_list_widget->currentIndex().row()]
                    .GetHighThreshold();
        }

        /*If TRUNK is Has Integer Parameter called Dilation, Update Dilation
         * Values for Viewing Purposes*/
        int dilation_val = 0;
        std::vector<jta_cost_function::Parameter<int>> active_int_params =
            session_.trunk_manager_.getActiveCostFunctionClass()->getIntParameters();
        for (int i = 0; i < active_int_params.size(); i++) {
            if (active_int_params[i].getParameterName() == "Dilation") {
                dilation_val = session_.trunk_manager_.getActiveCostFunctionClass()
                                   ->getIntParameters()
                                   .at(i)
                                   .getParameterValue();
            }
        }
        if (dilation_val < 0) {
            dilation_val = 0;
        }

        if (ui.image_list_widget->currentIndex().row() >= 0 &&
            ui.image_list_widget->currentIndex().row() < session_.loaded_frames.size()) {
            if (ui.camera_A_radio_button->isChecked()) {
                session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                    .SetEdgeImage(
                        ui.aperture_spin_box->value(), low_val, high_val);
                session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                    .SetDilatedImage(dilation_val);
            } else if (
                ui.camera_B_radio_button->isChecked() &&
                session_.calibrated_for_biplane_viewport_) {
                session_.loaded_frames_B[ui.image_list_widget->currentIndex().row()]
                    .SetEdgeImage(
                        ui.aperture_spin_box->value(), low_val, high_val);
                session_.loaded_frames_B[ui.image_list_widget->currentIndex().row()]
                    .SetDilatedImage(dilation_val);
            }
        }

        /*   Update image based on selected radio button   */
        if (ui.image_list_widget->currentIndex().row() >= 0) {
            /*If Original View Selected*/
            if (ui.original_image_radio_button->isChecked()) {
                on_original_image_radio_button_clicked();
            }
            /*If Inverted View Selected*/
            if (ui.inverted_image_radio_button->isChecked()) {
                on_inverted_image_radio_button_clicked();
            }
            /*If Edge View Selected*/
            if (ui.edges_image_radio_button->isChecked()) {
                on_edges_image_radio_button_clicked();
            }
            /*If Dilation View Selected*/
            if (ui.dilation_image_radio_button->isChecked()) {
                on_dilation_image_radio_button_clicked();
            }
        }
        settings_service_->SaveEdgeDetectionSettings(
            ui.aperture_spin_box->value(), low_val, high_val);
    }
};

void MainScreen::on_low_threshold_slider_valueChanged() {
    /*Set Label Text*/
    ui.low_threshold_value->setText(
        QString::number(ui.low_threshold_slider->value()));

    /*Make Sure Images Loaded First*/
    if (session_.loaded_frames.size() > 0) {
        /*Get High Value from Frame*/
        int aperture = APERTURE;
        int high_val = HIGH_THRESH;
        if (ui.camera_A_radio_button->isChecked()) {
            aperture = session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                           .GetAperture();
            high_val = session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                           .GetHighThreshold();
        } else if (
            ui.camera_B_radio_button->isChecked() &&
            session_.calibrated_for_biplane_viewport_) {
            aperture =
                session_.loaded_frames_B[ui.image_list_widget->currentIndex().row()]
                    .GetAperture();
            high_val =
                session_.loaded_frames_B[ui.image_list_widget->currentIndex().row()]
                    .GetHighThreshold();
        }

        /*If TRUNK is Has Integer Parameter called Dilation, Update Dilation
         * Values for Viewing Purposes*/
        int dilation_val = 0;
        std::vector<jta_cost_function::Parameter<int>> active_int_params =
            session_.trunk_manager_.getActiveCostFunctionClass()->getIntParameters();
        for (int i = 0; i < active_int_params.size(); i++) {
            if (active_int_params[i].getParameterName() == "Dilation") {
                dilation_val = session_.trunk_manager_.getActiveCostFunctionClass()
                                   ->getIntParameters()
                                   .at(i)
                                   .getParameterValue();
            }
        }
        if (dilation_val < 0) {
            dilation_val = 0;
        }

        if (ui.image_list_widget->currentIndex().row() >= 0 &&
            ui.image_list_widget->currentIndex().row() < session_.loaded_frames.size()) {
            if (ui.camera_A_radio_button->isChecked()) {
                session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                    .SetEdgeImage(
                        aperture, ui.low_threshold_slider->value(), high_val);
                session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                    .SetDilatedImage(dilation_val);
            } else if (
                ui.camera_B_radio_button->isChecked() &&
                session_.calibrated_for_biplane_viewport_) {
                session_.loaded_frames_B[ui.image_list_widget->currentIndex().row()]
                    .SetEdgeImage(
                        aperture, ui.low_threshold_slider->value(), high_val);
                session_.loaded_frames_B[ui.image_list_widget->currentIndex().row()]
                    .SetDilatedImage(dilation_val);
            }
        }
        /*   Update image based on selected radio button   */
        if (ui.image_list_widget->currentIndex().row() >= 0) {
            /*If Original View Selected*/
            if (ui.original_image_radio_button->isChecked()) {
                on_original_image_radio_button_clicked();
            }
            /*If Inverted View Selected*/
            if (ui.inverted_image_radio_button->isChecked()) {
                on_inverted_image_radio_button_clicked();
            }
            /*If Edge View Selected*/
            if (ui.edges_image_radio_button->isChecked()) {
                on_edges_image_radio_button_clicked();
            }
            /*If Dilation View Selected*/
            if (ui.dilation_image_radio_button->isChecked()) {
                on_dilation_image_radio_button_clicked();
            }
        }
        settings_service_->SaveEdgeDetectionSettings(
            aperture, ui.low_threshold_slider->value(), high_val);
    }
};

void MainScreen::on_high_threshold_slider_valueChanged() {
    /*Set Label Text*/
    ui.high_threshold_value->setText(
        QString::number(ui.high_threshold_slider->value()));

    /*Make Sure Images Loaded First*/
    if (session_.loaded_frames.size() > 0) {
        /*Get Low Value from Frame*/
        int aperture = APERTURE;
        int low_val = LOW_THRESH;
        if (ui.camera_A_radio_button->isChecked()) {
            aperture = session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                           .GetAperture();
            low_val = session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                          .GetLowThreshold();
        } else if (
            ui.camera_B_radio_button->isChecked() &&
            session_.calibrated_for_biplane_viewport_) {
            aperture =
                session_.loaded_frames_B[ui.image_list_widget->currentIndex().row()]
                    .GetAperture();
            low_val =
                session_.loaded_frames_B[ui.image_list_widget->currentIndex().row()]
                    .GetLowThreshold();
        }

        /*If TRUNK is Has Integer Parameter called Dilation, Update Dilation
         * Values for Viewing Purposes*/
        int dilation_val = 0;
        std::vector<jta_cost_function::Parameter<int>> active_int_params =
            session_.trunk_manager_.getActiveCostFunctionClass()->getIntParameters();
        for (int i = 0; i < active_int_params.size(); i++) {
            if (active_int_params[i].getParameterName() == "Dilation") {
                dilation_val = session_.trunk_manager_.getActiveCostFunctionClass()
                                   ->getIntParameters()
                                   .at(i)
                                   .getParameterValue();
            }
        }
        if (dilation_val < 0) {
            dilation_val = 0;
        }

        if (ui.image_list_widget->currentIndex().row() >= 0 &&
            ui.image_list_widget->currentIndex().row() < session_.loaded_frames.size()) {
            if (ui.camera_A_radio_button->isChecked()) {
                session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                    .SetEdgeImage(
                        aperture, low_val, ui.high_threshold_slider->value());
                session_.loaded_frames[ui.image_list_widget->currentIndex().row()]
                    .SetDilatedImage(dilation_val);
            } else if (
                ui.camera_B_radio_button->isChecked() &&
                session_.calibrated_for_biplane_viewport_) {
                session_.loaded_frames_B[ui.image_list_widget->currentIndex().row()]
                    .SetEdgeImage(
                        aperture, low_val, ui.high_threshold_slider->value());
                session_.loaded_frames_B[ui.image_list_widget->currentIndex().row()]
                    .SetDilatedImage(dilation_val);
            }
        }
        /*   Update image based on selected radio button   */
        if (ui.image_list_widget->currentIndex().row() >= 0) {
            /*If Original View Selected*/
            if (ui.original_image_radio_button->isChecked()) {
                on_original_image_radio_button_clicked();
            }
            /*If Inverted View Selected*/
            if (ui.inverted_image_radio_button->isChecked()) {
                on_inverted_image_radio_button_clicked();
            }
            /*If Edge View Selected*/
            if (ui.edges_image_radio_button->isChecked()) {
                on_edges_image_radio_button_clicked();
            }
            /*If Dilation View Selected*/
            if (ui.dilation_image_radio_button->isChecked()) {
                on_dilation_image_radio_button_clicked();
            }
        }
        settings_service_->SaveEdgeDetectionSettings(
            aperture, low_val, ui.high_threshold_slider->value());
    }
};
/*Apply All Edges*/
void MainScreen::on_apply_all_edge_button_clicked() {
    /*If TRUNK is Has Integer Parameter called Dilation, Update Dilation
     * Values for Viewing Purposes*/
    int dilation_val = 0;
    std::vector<jta_cost_function::Parameter<int>> active_int_params =
        session_.trunk_manager_.getActiveCostFunctionClass()->getIntParameters();
    for (int i = 0; i < active_int_params.size(); i++) {
        if (active_int_params[i].getParameterName() == "Dilation") {
            dilation_val = session_.trunk_manager_.getActiveCostFunctionClass()
                               ->getIntParameters()
                               .at(i)
                               .getParameterValue();
        }
    }
    if (dilation_val < 0) {
        dilation_val = 0;
    }

    /*Apply Edge Detect to All Images*/
    for (int i = 0; i < session_.loaded_frames.size(); i++) {
        session_.loaded_frames[i].SetEdgeImage(
            ui.aperture_spin_box->value(),
            ui.low_threshold_slider->value(),
            ui.high_threshold_slider->value());
        session_.loaded_frames[i].SetDilatedImage(dilation_val);
    }
    if (session_.calibrated_for_biplane_viewport_) {
        for (int i = 0; i < session_.loaded_frames_B.size(); i++) {
            session_.loaded_frames_B[i].SetEdgeImage(
                ui.aperture_spin_box->value(),
                ui.low_threshold_slider->value(),
                ui.high_threshold_slider->value());
            session_.loaded_frames_B[i].SetDilatedImage(dilation_val);
        }
    }
    /*   Update image based on selected radio button   */
    if (ui.image_list_widget->currentIndex().row() >= 0) {
        /*If Original View Selected*/
        if (ui.original_image_radio_button->isChecked()) {
            on_original_image_radio_button_clicked();
        }
        /*If Inverted View Selected*/
        if (ui.inverted_image_radio_button->isChecked()) {
            on_inverted_image_radio_button_clicked();
        }
        /*If Edge View Selected*/
        if (ui.edges_image_radio_button->isChecked()) {
            on_edges_image_radio_button_clicked();
        }
        /*If Dilation View Selected*/
        if (ui.dilation_image_radio_button->isChecked()) {
            on_dilation_image_radio_button_clicked();
        }
    }
    settings_service_->SaveEdgeDetectionSettings(
        ui.aperture_spin_box->value(),
        ui.low_threshold_slider->value(),
        ui.high_threshold_slider->value());
}

/*Reset Edge Detection Values*/
void MainScreen::on_reset_edge_button_clicked() {
    ui.aperture_spin_box->setValue(APERTURE);
    ui.low_threshold_slider->setValue(LOW_THRESH);
    ui.high_threshold_slider->setValue(HIGH_THRESH);
}

/*Optimize Buttons*/
/*Optimize Buttons*/
void MainScreen::on_optimize_button_clicked() {
    LaunchOptimizer("Single");
}

/*Optimize All Button*/
void MainScreen::on_optimize_all_button_clicked() {
    LaunchOptimizer("All");
}

/*Optimize Each Button*/
void MainScreen::on_optimize_each_button_clicked() {
    LaunchOptimizer("Each");
}

/*Optimize From Button*/
void MainScreen::on_optimize_from_button_clicked() {
    LaunchOptimizer("From");
}

void MainScreen::on_actionOptimize_Backward_triggered() {
    LaunchOptimizer("Backward");
}

/*Disable/Enable During Optimization*/
void MainScreen::DisableAll() {
    ui.load_calibration_button->setDisabled(true);
    ui.load_image_button->setDisabled(true);
    ui.load_model_button->setDisabled(true);
    ui.optimize_button->setDisabled(true);
    ui.optimize_all_button->setDisabled(true);
    ui.optimize_each_button->setDisabled(true);
    ui.optimize_from_button->setDisabled(true);
    ui.apply_all_edge_button->setDisabled(true);
    ui.reset_edge_button->setDisabled(true);
    ui.single_model_radio_button->setDisabled(true);
    ui.multiple_model_radio_button->setDisabled(true);
    ui.image_list_widget->setDisabled(true);
    ui.model_list_widget->setDisabled(true);
    ui.aperture_spin_box->setDisabled(true);
    ui.low_threshold_slider->setDisabled(true);
    ui.high_threshold_slider->setDisabled(true);
    /*Reverse for Stop Optimizer*/
    ui.actionStop_Optimizer->setEnabled(true);
}

void MainScreen::EnableAll() {
    /*Only Re-enable load calibration if for some reason neither are
     * clibrated (don't know how this would ever happen)...*/
    if (session_.calibrated_for_monoplane_viewport_ == false &&
        session_.calibrated_for_biplane_viewport_ == false) {
        ui.load_calibration_button->setEnabled(true);
    }
    ui.load_image_button->setEnabled(true);
    ui.load_model_button->setEnabled(true);
    ui.optimize_button->setEnabled(true);
    ui.optimize_all_button->setEnabled(true);
    ui.optimize_each_button->setEnabled(true);
    ui.optimize_from_button->setEnabled(true);
    ui.apply_all_edge_button->setEnabled(true);
    ui.reset_edge_button->setEnabled(true);
    ui.single_model_radio_button->setEnabled(true);
    ui.multiple_model_radio_button->setEnabled(true);
    ui.image_list_widget->setEnabled(true);
    ui.model_list_widget->setEnabled(true);
    ui.aperture_spin_box->setEnabled(true);
    ui.low_threshold_slider->setEnabled(true);
    ui.high_threshold_slider->setEnabled(true);
    /*Reverse for Stop Optimizer*/
    ui.actionStop_Optimizer->setDisabled(true);
    currently_optimizing_ = false;
}

/*NON GUI FUNCTIONS*/
/*Save Last Pose (Do this when optimizing or when chaninging the list
 * widgets*/
void MainScreen::SaveLastPose() {
    /*Save Last Pair Pose*/
    if (previous_model_indices_.size() > 0 && previous_frame_index_ != -1) {
        for (int i = 0; i < previous_model_indices_.size(); i++) {
            double* position_curr = vw->get_model_position_at_index(
                previous_model_indices_[i].row());
            double* orientation_curr = vw->get_model_orientation_at_index(
                previous_model_indices_[i].row());
            Point6D last_pose(
                position_curr[0],
                position_curr[1],
                position_curr[2],
                orientation_curr[0],
                orientation_curr[1],
                orientation_curr[2]);
            /*If Camera B View, Save in Camera A coordinates*/
            if (ui.camera_A_radio_button->isChecked()) {
                session_.model_locations_.SavePose(
                    previous_frame_index_,
                    previous_model_indices_[i].row(),
                    last_pose);
            } else {
                session_.model_locations_.SavePose(
                    previous_frame_index_,
                    previous_model_indices_[i].row(),
                    session_.calibration_file_.convert_Pose_B_to_Pose_A(last_pose));
            }
        }
    }
}

/*Optimization Function: Packages Off The Optimization process in
a new thread*/
/*Launch Optimizer*/
void MainScreen::LaunchOptimizer(QString directive) {
    if (currently_optimizing_) {
        return;
    }
    /*Save Last Pair Pose*/
    SaveLastPose();
    int iter_count;

    if (directive == "Sym_Trap") {
        sym_trap_running = true;
        // iter_count = sym_trap_control->getIterCount();
    } else {
        iter_count = 0;
    }
    /*Can Only Optimize If Chosen Frame and Model*/
    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    if (selected.size() == 0 || previous_frame_index_ < 0 ||
        ui.image_list_widget->currentIndex().row() != previous_frame_index_ ||
        ui.image_list_widget->currentIndex().row() >= session_.loaded_frames.size() ||
        ui.model_list_widget->currentIndex().row() >= session_.loaded_models.size()) {
        QMessageBox::critical(
            this, "Error!", "Select Frame and Model First!", QMessageBox::Ok);
        return;
    }

    /*Check Frame List by Model List and Guess Matrix Dimensions are the
     * Same Size*/
    if (session_.model_locations_.GetFrameCount() != session_.loaded_frames.size() ||
        session_.model_locations_.GetModelCount() != session_.loaded_models.size()) {
        QMessageBox::critical(
            this,
            "Critical Error!",
            "Pose Dimension Matrix Differs in Size from Frame "
            "and Models Loaded! Please Contact Support!",
            QMessageBox::Ok);
        return;
    }

    /*Initialize Thread*/
    /*Set Up Connections*/
    // Master Thread to Carry Optimizer
    optimizer_manager = new OptimizerManager();        // Create Master
    optimizer_thread = new QThread();                  // Create QThread
    optimizer_manager->moveToThread(optimizer_thread); // Move Master to QThread

    /*Send the Following Information to the Optimizer Thread:
    0). Calibration Class
    1). Frame List(s)
    2). Selected Models List and Primary Model
    3). Current Pose Matrix
    4). Optimizer Settings Class
    5). The Three Cost Function Manager Classes
    6). Optimization Directives (All, From, Each, Single)
    7). Error Message Reference*/
    /*Initialze the Optimizer by Sending All the Previous Information and
     * Setting Up/Checking the CUDA Connecitons*/
    QString error_mess;
    bool initialized_correctly = optimizer_manager->Initialize(
        *optimizer_thread,
        session_.calibration_file_,
        session_.loaded_frames,
        session_.loaded_frames_B,
        ui.image_list_widget->currentIndex().row(),
        session_.loaded_models,
        selected,
        selected[0].row(),
        session_.model_locations_,
        session_.optimizer_settings_,
        session_.trunk_manager_,
        session_.branch_manager_,
        session_.leaf_manager_,
        directive,
        error_mess,
        iter_count);

    /*If Didnt't Initialize Correctly DESTROY*/
    if (!initialized_correctly) {
        optimizer_thread->start();
        QMessageBox::critical(this, "Error!", error_mess, QMessageBox::Ok);
        return;
    }

    /*Connect Optimizer Threads*/
    connect(
        optimizer_manager,
        SIGNAL(UpdateDisplay(double, int, double, unsigned int)),
        this,
        SLOT(onUpdateDisplay(
            double,
            int,
            double,
            unsigned int))); // Update Display
    connect(
        optimizer_manager,
        SIGNAL(OptimizerError(QString)),
        this,
        SLOT(onOptimizerError(QString)));
    // Optimizer Error Check
    connect(
        optimizer_manager,
        SIGNAL(UpdateOptimum(
            double, double, double, double, double, double, unsigned int)),
        this,
        SLOT(onUpdateOptimum(
            double, double, double, double, double, double, unsigned int)));
    // Update Guess Connection
    connect(
        optimizer_manager,
        SIGNAL(OptimizedFrame(
            double,
            double,
            double,
            double,
            double,
            double,
            bool,
            unsigned int,
            bool,
            QString)),
        this,
        SLOT(onOptimizedFrame(
            double,
            double,
            double,
            double,
            double,
            double,
            bool,
            unsigned int,
            bool,
            QString)));
    // Update Optimized Frame/View
    connect(
        this,
        SIGNAL(StopOptimizer()),
        optimizer_manager,
        SLOT(onStopOptimizer()),
        Qt::DirectConnection);
    /*Stops Optimizer*/
    connect(
        optimizer_manager,
        SIGNAL(UpdateDilationBackground()),
        this,
        SLOT(onUpdateDilationBackground()));
    /*UPDATE DILATION BACKGROUND	*/
    connect(
        optimizer_manager,
        SIGNAL(onUpdateOrientationSymTrap(
            double, double, double, double, double, double)),
        this,
        SLOT(updateOrientationSymTrap_MS(
            double, double, double, double, double, double)));

    // Connect sym trap progress bar to thread
    // connect(optimizer_manager, SIGNAL(onProgressBarUpdate(int)),
    // sym_trap_control->ui.progressBar, SLOT(setValue(int)));

    /*Start*/
    if (directive == "Each" || directive == "All") {
        ui.image_list_widget->setCurrentRow(0);
    }
    actor_text->GetTextProperty()->SetColor(
        214.0 / 255.0,
        108.0 / 255.0,
        35.0 / 255.0); // Set Orange;
    currently_optimizing_ = true;
    DisableAll();
    display_optimizer_settings_ = session_.optimizer_settings_;
    optimizer_thread->start();
}

void MainScreen::updateOrientationSymTrap_MS(
    double x, double y, double z, double xa, double ya, double za) {
    // put the update logic here
    //  look at loading kinematics for help
    //  or copy pose
    //  need to update the current model
    Point6D new_orientation(x, y, z, xa, ya, za);
    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();
    session_.model_locations_.SavePose(
        ui.image_list_widget->currentRow(),
        ui.model_list_widget->currentRow(),
        new_orientation);
    vw->set_model_position_at_index(
        selected[0].row(),
        new_orientation.x,
        new_orientation.y,
        new_orientation.z);
    vw->set_model_orientation_at_index(
        selected[0].row(),
        new_orientation.xa,
        new_orientation.ya,
        new_orientation.za);
    ui.qvtk_widget->update();
    ui.qvtk_widget->renderWindow()->Render();
    ui.qvtk_cpv->update();
    ui.qvtk_cpv->renderWindow()->Render();
}

/*OPTIMIZATION
 */
/*Update Blue Current Optimum*/
void MainScreen::onUpdateOptimum(
    double x,
    double y,
    double z,
    double xa,
    double ya,
    double za,
    unsigned int primary_model_index) {
    /*Update Blue's Location*/
    auto CurrentPose = Point6D(x, y, z, xa, ya, za);
    if (ui.camera_B_radio_button->isChecked()) {
        CurrentPose = session_.calibration_file_.convert_Pose_A_to_Pose_B(CurrentPose);
    }
    if (primary_model_index <
        session_.loaded_models.size()) { // TODO: Find a better way to represent this
        vw->set_model_position_at_index(
            primary_model_index, CurrentPose.x, CurrentPose.y, CurrentPose.z);
        vw->set_model_orientation_at_index(
            primary_model_index,
            CurrentPose.xa,
            CurrentPose.ya,
            CurrentPose.za);
        ui.qvtk_widget->update();
        ui.qvtk_widget->renderWindow()->Render();
        coronal_vw->set_model_position_at_index(
            primary_model_index, CurrentPose.x, CurrentPose.y, CurrentPose.z);
        coronal_vw->set_model_orientation_at_index(
            primary_model_index,
            CurrentPose.xa,
            CurrentPose.ya,
            CurrentPose.za);
        ui.qvtk_cpv->update();
        ui.qvtk_cpv->renderWindow()->Render();
    }
}

/*Finished Optimizing Frame, Send Optimum to MainScreen*/
void MainScreen::onOptimizedFrame(
    double x,
    double y,
    double z,
    double xa,
    double ya,
    double za,
    bool move_next_frame,
    unsigned int primary_model_index,
    bool error_occurred,
    QString optimizer_directive) {
    /*Update Actor*/
    auto CurrentPose = Point6D(x, y, z, xa, ya, za);
    if (ui.camera_B_radio_button->isChecked()) {
        CurrentPose = session_.calibration_file_.convert_Pose_A_to_Pose_B(CurrentPose);
    }
    if (primary_model_index <
        session_.loaded_models.size()) { // todo: Find a better way to get size of
                                // model_actor_list
        vw->set_model_position_at_index(
            primary_model_index, CurrentPose.x, CurrentPose.y, CurrentPose.z);
        vw->set_model_orientation_at_index(
            primary_model_index,
            CurrentPose.xa,
            CurrentPose.ya,
            CurrentPose.za);
        ui.qvtk_widget->update();
        ui.qvtk_widget->renderWindow()->Render();
        coronal_vw->set_model_position_at_index(
            primary_model_index, CurrentPose.x, CurrentPose.y, CurrentPose.z);
        coronal_vw->set_model_orientation_at_index(
            primary_model_index,
            CurrentPose.xa,
            CurrentPose.ya,
            CurrentPose.za);
        ui.qvtk_cpv->update();
        ui.qvtk_cpv->renderWindow()->Render();
    } else {
        /*Display Finished*/
        QMessageBox::critical(
            this, "Error!", "Model index out of bounds!", QMessageBox::Ok);
        /*Program Will Crash after this message but that is fine, this
         * should never ever occurr...*/
    }

    /*Save Indices*/
    int current_frame_index = ui.image_list_widget->currentIndex().row();

    if (optimizer_directive == "Backward") {
        if (move_next_frame && current_frame_index > 0) {
            ui.image_list_widget->setCurrentRow(current_frame_index - 1);
            session_.model_locations_.SavePose(
                current_frame_index,
                primary_model_index,
                Point6D(x, y, z, xa, ya, za));
        } else {
            /*Save Pose To Storage*/
            session_.model_locations_.SavePose(
                current_frame_index,
                primary_model_index,
                Point6D(x, y, z, xa, ya, za));
            /*Not Currently Optimzing*/
            currently_optimizing_ = false;
            EnableAll();
            /*Display Finished*/
            if (!error_occurred) {
                QMessageBox::information(
                    this,
                    "Finished!",
                    "All frames optimized!",
                    QMessageBox::Ok);
            }
        }

    } else {
        /*If Commanded to Move To Next Frame Do So*/
        if (move_next_frame &&
            current_frame_index + 1 < ui.image_list_widget->count()) {
            /*Bring Up Next Frame*/
            ui.image_list_widget->setCurrentRow(current_frame_index + 1);
            /*Save Pose To Storage*/
            session_.model_locations_.SavePose(
                current_frame_index,
                primary_model_index,
                Point6D(x, y, z, xa, ya, za));
        } else {
            /*Save Pose To Storage*/
            session_.model_locations_.SavePose(
                current_frame_index,
                primary_model_index,
                Point6D(x, y, z, xa, ya, za));
            /*Not Currently Optimzing*/
            currently_optimizing_ = false;
            EnableAll();
            /*Display Finished*/
            if (!error_occurred && !sym_trap_running) {
                QMessageBox::information(
                    this,
                    "Finished!",
                    "All frames optimized!",
                    QMessageBox::Ok);
            }
            sym_trap_running = false;
        }
    }
}

/*Uh oh There was an Error. The int is the code.
1: Could not update comparison image
2: no potentialy optimal hyper rectangles found
3: Storage Matrix Empty!
4: Renderering failure!
5: Error: Negative Metric!
*/
void MainScreen::onOptimizerError(QString error_message) {
    QMessageBox::critical(this, "Error!", error_message, QMessageBox::Ok);
}

/*Update Display with Speed, Cost Function Calls, Current Minimum*/
void MainScreen::onUpdateDisplay(
    double iteration_speed,
    int current_iteration,
    double current_minimum,
    unsigned int primary_model_index) {
    div_t divresult;
    divresult =
        div(static_cast<int>(
                static_cast<double>(
                    display_optimizer_settings_.trunk_budget +
                    display_optimizer_settings_.enable_branch_ *
                        display_optimizer_settings_.number_branches *
                        display_optimizer_settings_.branch_budget +
                    display_optimizer_settings_.enable_leaf_ *
                        display_optimizer_settings_.leaf_budget -
                    current_iteration) /
                (1000.0 / iteration_speed)),
            60);
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(2) << divresult.rem;
    std::string infoText = "Optimum Location: <";
    std::stringstream level;
    if (current_iteration < display_optimizer_settings_.trunk_budget) {
        level << "Trunk";
    } else if (
        current_iteration <
        display_optimizer_settings_.trunk_budget +
            display_optimizer_settings_.enable_branch_ *
                display_optimizer_settings_.number_branches *
                display_optimizer_settings_.branch_budget) {
        div_t divresultLevel;
        divresultLevel =
            div(current_iteration - display_optimizer_settings_.trunk_budget,
                display_optimizer_settings_.branch_budget);
        level << "Branch " << divresultLevel.quot + 1;
    } else if (
        current_iteration <
        display_optimizer_settings_.trunk_budget +
            display_optimizer_settings_.enable_branch_ *
                display_optimizer_settings_.number_branches *
                display_optimizer_settings_.branch_budget +
            display_optimizer_settings_.enable_leaf_ *
                display_optimizer_settings_.leaf_budget) {
        level << "Extra Z-Translation";
    } else {
        level << "Finished";
    }
    auto current_position =
        vw->get_model_position_at_index(primary_model_index);
    auto current_orientation =
        vw->get_model_orientation_at_index(primary_model_index);
    auto CurrentPose = Point6D(
        current_position[0],
        current_position[1],
        current_position[2],
        current_orientation[0],
        current_orientation[1],
        current_orientation[2]);
    if (ui.camera_B_radio_button->isChecked()) {
        CurrentPose = session_.calibration_file_.convert_Pose_B_to_Pose_A(CurrentPose);
    }

    infoText +=
        std::to_string(static_cast<long double>(CurrentPose.x)) + "," +
        std::to_string(static_cast<long double>(CurrentPose.y)) + "," +
        std::to_string(static_cast<long double>(CurrentPose.z)) +
        ">\nOptimum Orientation: <" +
        std::to_string(static_cast<long double>(CurrentPose.xa)) + "," +
        std::to_string(static_cast<long double>(CurrentPose.ya)) + "," +
        std::to_string(static_cast<long double>(CurrentPose.za)) +
        ">\nMinimum Function Value: " +
        std::to_string(static_cast<long double>(current_minimum)) +
        "\nIterations Per Second: " +
        std::to_string(static_cast<long double>(1000.0) / iteration_speed) +
        "\nIteration Count: " +
        std::to_string(static_cast<long long>(current_iteration)) +
        "\nSearch Level: " + level.str() +
        "\nEstimated Time Remaining for Frame: <" +
        std::to_string(static_cast<long long>(divresult.quot)) + ":" +
        ss.str() + ">";
    actor_text->SetInput(infoText.c_str());
    actor_text->GetTextProperty()->SetColor(
        214.0 / 255.0, 108.0 / 255.0, 35.0 / 255.0);

    /*update qvtk*/
    ui.qvtk_widget->update();
    ui.qvtk_widget->renderWindow()->Render();
    ui.qvtk_cpv->update();
    ui.qvtk_cpv->renderWindow()->Render();
}

/*Update Background if Dilation Selected and Moving From Trunk to Branch OR
 * Branch to Z Search*/
void MainScreen::onUpdateDilationBackground() {
    if (ui.dilation_image_radio_button->isChecked()) {
        on_dilation_image_radio_button_clicked();
        ui.qvtk_cpv->update();
        ui.qvtk_cpv->renderWindow()->Render();
    }
}

/*Function to load settings from registry and also check if First Time
 * Loading*/
void MainScreen::LoadSettingsBetweenSessions() {
    settings_service_->LoadSettings(session_);

    const auto edge_settings = settings_service_->GetEdgeDetectionSettings();
    ui.aperture_spin_box->setValue(edge_settings.aperture);
    ui.low_threshold_slider->setValue(edge_settings.low_threshold);
    ui.high_threshold_slider->setValue(edge_settings.high_threshold);

    if (!settings_service_->WasFirstTimeLoading()) {
        return;
    }

    /*Check CUDA Compatibility*/
    int gpu_device_count = 0, device_count;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&device_count);
    if (cudaResultCode != cudaSuccess) {
        device_count = 0;
    }
    /* Machines with no GPUs can still report one emulation device */
    for (int device = 0; device < device_count; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999 &&
            properties.major >= 5) /* 9999 means emulation only */
        {
            ++gpu_device_count;
        }
    }
    /*If no Cuda Compatitble Devices with Compute Capability Greater
     * Than 5, Exit*/
    if (gpu_device_count == 0) {
        if (device_count == 0) {
            QMessageBox::critical(
                this,
                "Error!",
                "No CUDA capable GPU detected! Optimizer will not run!",
                QMessageBox::Ok);
        } else if (properties.major == 9999) {
            QMessageBox::critical(
                this,
                "Error!",
                "GPU is emulation only! Optimizer will not run!",
                QMessageBox::Ok);
        } else {
            QMessageBox::critical(
                this,
                "Error!",
                "GPU does not have high enough compute "
                "capability! Optimizer will "
                "not run!\nPlease upgrade to device with "
                "compute capability 5.0 or "
                "higher!",
                QMessageBox::Ok);
        }
    }
}

/*Function to Save Settings from Optimizer Control Window to both Registry
 * and Optimizer Settings Class*/
/*On Optimizer Control Windows Save Setting*/
void MainScreen::onSaveSettings(
    OptimizerSettings opt_settings,
    jta_cost_function::CostFunctionManager trunk_manager,
    jta_cost_function::CostFunctionManager branch_manager,
    jta_cost_function::CostFunctionManager leaf_manager) {
    /*Save to Optimizer Settings*/
    session_.optimizer_settings_ = opt_settings;

    /*Save 3 Cost Function Managers*/
    session_.trunk_manager_ = trunk_manager;
    session_.branch_manager_ = branch_manager;
    session_.leaf_manager_ = leaf_manager;

    settings_service_->SaveEdgeDetectionSettings(
        ui.aperture_spin_box->value(),
        ui.low_threshold_slider->value(),
        ui.high_threshold_slider->value());
    settings_service_->SaveSettings(session_);

    /*Update Dilation Frames*/
    UpdateDilationFrames();
}

/*Function That Saves Dilation as 0 if No Trunk Manager has a Dilation Int
Parameter, else saves all the Dilation Images for Each Frame as the Dilation
Constant*/
void MainScreen::UpdateDilationFrames() {
    /*If TRUNK is Has Integer Parameter called Dilation, Update Dilation
     * Values for Viewing Purposes*/
    int dilation_val = 0;
    std::vector<jta_cost_function::Parameter<int>> active_int_params =
        session_.trunk_manager_.getActiveCostFunctionClass()->getIntParameters();
    for (int i = 0; i < active_int_params.size(); i++) {
        if (active_int_params[i].getParameterName() == "Dilation") {
            dilation_val = session_.trunk_manager_.getActiveCostFunctionClass()
                               ->getIntParameters()
                               .at(i)
                               .getParameterValue();
        }
    }
    if (dilation_val < 0) {
        dilation_val = 0;
    }
    /*Mahfouz Case*/
    if (session_.trunk_manager_.getActiveCostFunction() == "DIRECT_MAHFOUZ") {
        dilation_val = 3;
    }

    /*Apply Dilation to All Images*/
    for (int i = 0; i < session_.loaded_frames.size(); i++) {
        session_.loaded_frames[i].SetDilatedImage(dilation_val);
        if (session_.calibrated_for_biplane_viewport_) {
            session_.loaded_frames_B[i].SetDilatedImage(dilation_val);
        }
    }
    /*If Dilation View Selected*/
    if (ui.image_list_widget->currentIndex().row() >= 0 &&
        ui.dilation_image_radio_button->isChecked()) {
        on_dilation_image_radio_button_clicked();
        ui.qvtk_cpv->update();
        ui.qvtk_cpv->renderWindow()->Render();
    }
}

void MainScreen::on_actionEstimate_Tibial_Implant_s_triggered() {
    if (session_.optimizer_settings_.enable_leaf_) {
        worker_orchestrator_->StartEstimation(session_);
    }
}

void MainScreen::on_actionAmbiguous_Pose_Processing_triggered() {
    if (ui.model_list_widget->selectionModel()->selectedRows().size() != 2) {
        QMessageBox::critical(
            this,
            "Error!",
            "Must Be in Multiple Model Selection Mode to Run "
            "Ambiguous Pose Analysis!",
            QMessageBox::Ok);
        return;
    }

    // Loop through each of the frames
    QModelIndexList selected =
        ui.model_list_widget->selectionModel()->selectedRows();

    // save the current location of the image

    for (int i = 0; i < ui.image_list_widget->count(); i++) {
        Point6D fem_pose = session_.model_locations_.GetPose(i, selected[1].row());
        Point6D tib_pose_orig = session_.model_locations_.GetPose(i, selected[0].row());

        Point6D tib_pose_final = tibial_pose_selector(fem_pose, tib_pose_orig);
        session_.model_locations_.SavePose(i, selected[0].row(), tib_pose_final);
    }
    // Need to update the location of the frame that is currently on screen
    int selected_img_idx =
        ui.image_list_widget->selectionModel()->selectedRows()[0].row();
    Point6D current_img_pos =
        session_.model_locations_.GetPose(selected_img_idx, selected[0].row());
    model_actor_list[selected[0].row()]->SetPosition(
        current_img_pos.x, current_img_pos.y, current_img_pos.z);
    model_actor_list[selected[0].row()]->SetOrientation(
        current_img_pos.xa, current_img_pos.ya, current_img_pos.za);

    ui.qvtk_widget->update();
    ui.qvtk_widget->renderWindow()->Render();
    ui.qvtk_cpv->update();
    ui.qvtk_cpv->renderWindow()->Render();
}
