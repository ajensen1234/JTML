/**
 * @file mainscreen.h
 * @author Andrew Jensen (andrewjensen321@gmail.com)
 * @brief This is the MainScreen object class that controls the GUI.
 * @version 0.1
 * @date 2022-10-01
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef MAINSCREEN_H
#define MAINSCREEN_H

/*Relevant QT Includes*/
#include <memory.h>
#include <qactiongroup.h>

#include <QtWidgets/QMainWindow>

#include "ui_mainscreen.h"
/*Font*/
#include <qfont.h>

/*Key Event*/
#include <QKeyEvent>

/*Direct Data Structures*/
#include "compute/curvature_utilities.h"
#include "domain/data_structures_6D.h"
/*Custom Calibration Struct (Used in CUDA GPU METRICS)*/
#include "services/calibration.h"

/*VTK*/
#include <vtkActor.h>
#include <vtkAutoInit.h>  // Added post migration to Banks' lab computer
#include <vtkCamera.h>
#include <vtkDataSetMapper.h>
#include <vtkImageData.h>
#include <vtkInteractorStyleTrackballActor.h>
#include <vtkInteractorStyleTrackballCamera.h> /*Alternate Camera*/
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkVersion.h>

/*Frame and Model and Location Storage*/
#include "compute/frame.h"
#include "services/location_storage.h"
#include "services/model.h"

/*Optimizer Settings*/
#include "services/optimizer_settings.h"

/*Settings persistence service (plan 004 U3 / R9): owns the QSettings
 * round-trip for cost-function/optimizer/edge settings; widget-free.*/
#include "services/settings_service.h"

/*Session controller (plan 004 U6 / R6+R10): owns the load path (calibration/
 * image/model parsing + dataset population) and the camera A/B switching
 * state; widget-free and headless-testable (it never touches interactor.h's
 * file-scope globals -- this TU is the only interactor.h includer).*/
#include "services/session_controller.h"

/*Study-load controller (plan 006 U7 / R11): the ONE shared load path both
 * front-ends call -- calibration one-use-per-session + dataset-replace
 * policy, parse -> populate -> dedup -> counts relocated verbatim from the
 * load slots (R13); the scene/background/VTK updates stay view-side. It
 * wraps session_controller_ (declared before it) -- the shared active-
 * camera / count mirrors stay on the one instance the camera slots use --
 * and consults the injected run-in-flight probe (L17) at each load, wired
 * from the session-state controller's M7 probe below.*/
#include "services/study_load_controller.h"

/*Optimizer Run Controller (plan 006 U5): the shared run controller — gate,
 * drive sequence, run-state machine, progress, stop, seed, epoch/thread
 * lifecycle, destructor contract. MainScreen's LaunchOptimizer + locking
 * thin onto it; the 16-control DisableAll/EnableAll stays a view-side
 * mapper with EnableAll driven by the controller's terminal-frame relay
 * (pinned unlock-after-error).*/
#include "coordinator/optimizer_run_controller.h"

/*Session-state controller (plan 006 U6): the QObject notification shell
 * over jta::SessionState — MainScreen's SyncSessionState + the
 * previous-frame/model bookkeeping relocate here (R5/R6/R10, AE2). The
 * controller wraps session_state_ by pointer (declared before it) and
 * diffs + emits; the selection handlers advance the mirrors through
 * CommitSelection AFTER their save-last-pose, exactly like the old
 * previous_frame_index_ / previous_model_indices_ writes (H2/M9).*/
#include "coordinator/session_state_controller.h"

/*Optimizer Settings Control Window*/
#include "view/settings_control.h"

/*App-State Service (plan U7, R8/E11)*/
#include "domain/session_state.h"

/*DRR Settings Control Window*/
#include "drr_tool.h"

/* Symmetry Trap Analysis Window*/

/*Cost Function Library*/
#include "compute/CostFunctionManager.h"

/*CostFunctionTools*/
#include "compute/camera_calibration.h"

/*machine_learning_tools*/
#include "compute/machine_learning_tools.h"

/*Segmentation controller (plan 004 U8 / R12): owns the per-frame segment +
 * implant-estimate ops (GPU/torch). The view owns the per-frame loops, the
 * progress, processEvents, and the render interleave; the controller exposes
 * per-frame operations only. jtml_services is GPU-linked as of U8.
 *
 * INCLUDE ORDER NOTE: this header pulls torch, and PyTorch's ivalue_inl.h
 * does `#undef slots` (the Qt keyword macro). It must stay AFTER the
 * coordinator/view headers that use the raw `public slots:` keyword
 * (optimizer_manager.h / settings_control.h / drr_tool.h) -- same constraint
 * as the torch-bearing machine_learning_tools.h include above.*/
#include "services/ml_orchestrator.h"
#include "services/segmentation_controller.h"
#include "view/viewer.h"

/*List view-models (plan 004 U2, R4/R5): the image/model QListViews render
 * these passively; selection lives in the views' QItemSelectionModel (model +
 * selectionModel together are the headless-testable unit).*/
#include "view/frame_list_model.h"
#include "view/model_list_model.h"

/**
 * @brief The MainScreen object that inherits the QMainWindow object type. This
 * object serves as the class hosting all the items on the main window.
 *
 * Role (plan 004, R2): View + composition root. Widget wiring, VTK render
 * binding, layout/resize, and the irreducible view-only slots (display-mode
 * radios, interaction modes, reset view, key handling) stay here; everything
 * else lives in the extracted seams: view-models (FrameListModel /
 * ModelListModel), services (SessionController, SettingsService, EdgeProcessor,
 * SegmentationController, ImplantEstimator), domain (pose_copy, pose_file_io,
 * SessionState, OptimizeIntentController, ModelListBuilder,
 * ambiguous_pose_processing), coordinator (OptimizeCoordinator).
 */
class MainScreen : public QMainWindow {
    Q_OBJECT

public:
    MainScreen(QWidget* parent = 0);

    ~MainScreen() override;

    /*Escape Signal from VTK to stop optimizer*/
    void VTKEscapeSignal();

    /*Make Selected Actor Principal from VTK*/
    void VTKMakePrincipalSignal(vtkActor* new_principal_actor);

    /*Bool to see if currently optimizing*/

    bool currently_optimizing_;

Q_SIGNALS:
    /*Update Whether To Write TO Text Display*/
    void UpdateDisplayText(bool);

private:
    double pi = 3.14159265358979323846;

    Ui::MainScreenClass ui;

    double UF_BLUE[3] = {0, 72, 204};
    double UF_ORANGE[3] = {255, 77, 0};

    int curr_frame();

    float start_time;

    /*GUI FUNCTIONS*/
    /*Arrange Layout (Do this in code so scales across different DPI monitors
     * and handles weird fonts)*/
    void ArrangeMainScreenLayout(QFont application_font);

    /*Private Variables*/
    /*Original Sizes After Construction for Main Screen List Widgets, their
     * Group Boxes and QVTK Widget*/
    int image_list_widget_starting_height_;
    int image_selection_box_starting_height_;
    int model_list_widget_starting_height_;
    int model_selection_box_starting_height_;
    int qvtk_widget_starting_height_;
    int qvtk_widget_starting_width_;

    /*Monoplane and Biplane Calibration Viewport Files*/
    Calibration calibration_file_; /*Used in monoplane and biplane*/

    /*Variables Indicating Calibration Status for Mono and Biplane*/
    bool calibrated_for_monoplane_viewport_;
    bool calibrated_for_biplane_viewport_;

    /*VTK Variables for main Viewer*/
    std::vector<vtkSmartPointer<vtkActor>> model_actor_list;
    std::vector<vtkSmartPointer<vtkPolyDataMapper>> model_mapper_list;
    vtkSmartPointer<vtkRenderer> renderer;
    vtkSmartPointer<vtkImageData> current_background;
    vtkSmartPointer<vtkSTLReader> stl_reader;
    vtkSmartPointer<vtkDataSetMapper> image_mapper;
    vtkSmartPointer<vtkActor> actor_image;
    vtkSmartPointer<vtkTextActor> actor_text;
    vtkSmartPointer<vtkInteractorStyleTrackballCamera> camera_style_interactor;

    /* VTK Variables for Coronal Plane Viewer*/
    vtkSmartPointer<vtkRenderer> coronal_renderer;

    // Main viewer
    std::shared_ptr<Viewer> vw = std::make_shared<Viewer>();

    // Coronal Plane Viewer
    std::shared_ptr<Viewer> coronal_vw = std::make_shared<Viewer>();

    /*View Menu Radio Button Container*/
    QActionGroup *alignmentGroup, *alignmentGroupSegment;

    /*Frame/Model Containers*/
    std::vector<Frame> loaded_frames;
    std::vector<Frame> loaded_frames_B; /*If Biplane mode, need second group of
                                           loaded frames for camera B*/
    std::vector<Model> loaded_models;
    /*Location Storage Class*/
    LocationStorage model_locations_;

    QModelIndexList selected_model_indices();

    /*List view-models (plan 004 U2): write-once display-name models behind
     * the two passive QListViews (ui.image_list_widget / model_list_widget).
     * MainScreen's list bookkeeping (addItem/count/currentRow) is gone; the
     * views read the models, and selection state lives in the views'
     * QItemSelectionModel, which SyncSessionState reads.*/
    FrameListModel frame_list_model_;
    ModelListModel model_list_model_;

    /*App-State Service: owns the pure, widget-free session facts (model
     * list, selection, primary model, current frame). MainScreen keeps it
     * current from widget events; the rest of MainScreen reads it instead of
     * reaching into the UI directly (plan U7, R8/E11). Holds no widgets or
     * render binding, so it is headless-testable. NOT an observable
     * ViewModel (R12: no binding framework).*/
    jta::SessionState session_state_;

    /*Shared session-state controller (plan 006 U6): the diff + notification
     * shell wrapping session_state_ (declared BEFORE it — the controller
     * holds &session_state_). SyncSessionState writes through
     * UpdateSession; the selection handlers call CommitSelection after
     * their save-last-pose (the old previous_frame_index_ /
     * previous_model_indices_ writes are gone — the mirrors live in the
     * session state, H2). The injected run-in-flight probe (M7) reads the
     * run controller; the seed-clear (H5/M10b) drops its pending seed on a
     * dataset clear (the widgets app has no clear path today — the wiring
     * keeps ResetForDatasetClear complete).*/
    SessionStateController session_state_controller_;

    /*Session controller (plan 004 U6 / R6+R10): owns the load path
     * (calibration/image/model parsing + dataset population) and the camera
     * A/B switching state. Operates on the view's dataset
     * (loaded_frames/loaded_models/model_locations_) by reference; the view
     * keeps ownership + the dialogs, view-model insertion, interactor.h
     * global writes, and VTK wiring.*/
    jta::SessionController session_controller_;

    /*Shared study-load controller (plan 006 U7 / R11): the load slots thin
     * onto it (calibration one-use + dataset-replace policy, parse ->
     * populate -> dedup -> counts); it wraps session_controller_ (declared
     * before it) and probes session_state_controller_.runInFlight() (M7 -
     * L17) at each load. The camera slots keep using session_controller_
     * directly (U9 thins them later).*/
    jta::StudyLoadController study_load_controller_;

    /*Segmentation controller (plan 004 U8 / R12): per-frame segment +
     * implant-estimate operations (SegmentFrame / EstimateImplantPose). The
     * view keeps the loops, the torch model loading, the progress/render
     * interleave, and the Frame post-processing; the controller wraps the
     * GPU/torch calls verbatim (per-frame API -- no controller-owned loop).*/
    jta::SegmentationController segmentation_controller_;

    /*Shared ML orchestrator (plan 006 U8 / R12 part): the per-frame
     * segment -> estimate -> SavePose -> seed chain. The slots inject the
     * torch/CUDA ops (wrapping segmentation_controller_ above) and keep
     * the .pt loads, the per-frame loops, the dilation/edge parameter
     * sourcing and the progress/render interleave. The estimate's
     * SavePose into model_locations_ IS the widgets seed (LaunchOptimizer
     * copies the storage by value — no explicit run-controller seed).*/
    jta::MlOrchestrator ml_orchestrator_;

    /*Pull the current widget state into session_state_. Called wherever the
     * model/frame lists or their selection/current rows change.*/
    void SyncSessionState();

    /*Save the Pose From The Last Selected Frame*/
    void SaveLastPose();

    /*Settings persistence (plan 004 U3): the service owns the QSettings
     * round-trip (registry parity: org JointTrackAutoGPU / app Version340 /
     * groups CostFunctionSettings / OptimizerSettings / EdgeDetectionSettings /
     * FirstTime). The view maps the GPU-linked CostFunctionManagers <-> raw
     * registry entries and owns the CUDA probe + dialogs.*/
    jta::SettingsService settings_service_;

    /*Optimizer Settings That Must Be Set in Constructor and Changed on
     * OSettings Update */
    OptimizerSettings optimizer_settings_;

    /*Copy of the Above Only Used While Optimizing to Display Output*/
    OptimizerSettings display_optimizer_settings_;

    /*Cost Function Managers (from JTA Cost Function Library) for each stage of
     * DIRECT-JTA Optimizer*/
    jta_cost_function::CostFunctionManager trunk_manager_;
    jta_cost_function::CostFunctionManager branch_manager_;
    jta_cost_function::CostFunctionManager
        leaf_manager_;  // For extra Z-translation usually (esp. when monoplane)

    /*Function That Saves Dilation as 0 if No Trunk Manager has a Dilation Int
    Parameter, else saves all the Dilation Images for Each Frame as the Dilation
    Constant*/
    void UpdateDilationFrames();

    /*Optimization Function: Packages Off The Optimization process in
    a new thread*/

    /*Launch Optimizer*/

    void LaunchOptimizer(
        OptimizerRunController::Directive
            directive);  // Directive Says whether it is Optimize Single,
                         // From, All, or Each (or Sym_Trap)

    /*The shared optimizer-run controller (plan 006 U5): owns the drive
     * sequence + thread lifecycle; the view maps the relays onto the
     * widgets (actors, selection advance, DisableAll/EnableAll mapper).*/
    OptimizerRunController optimizer_run_controller_;

    /*Disable and Enable MainScreen During and After Optimization*/
    void DisableAll();

    void EnableAll();

    /*Function That Loads Settings from Registry or (If First Time Loading
    Saves Default Settings*/
    void LoadSettingsBetweenSessions();

    /*Builds the raw CostFunctionSettings registry entries (key format
     * STAGE@ACTIVE_CF / STAGE@CFname@ParamName@TYPE) from the three cost
     * function managers -- relocated verbatim from the first-run save and
     * onSaveSettings (plan 004 U3). The service round-trips the entries;
     * the view maps managers <-> entries because the GPU-linked
     * CostFunctionManager must not enter the QtCore-only service.*/
    std::vector<jta::RegistryEntry> BuildCostFunctionRegistryEntries(
        jta_cost_function::CostFunctionManager& trunk_manager,
        jta_cost_function::CostFunctionManager& branch_manager,
        jta_cost_function::CostFunctionManager& leaf_manager) const;

    /*Optimizer Window Control*/
    SettingsControl* settings_control;

    /*Sym Trap Window*/

    /*Calculate Viewing Angle (Accounts for Offsets)*/
    double CalculateViewingAngle(int width, int height, bool CameraA);

    /*Helper Function To Segment And Update Frames According to Model File*/
    void segmentHelperFunction(
        std::string pt_model_location,
        unsigned int input_width,
        unsigned int input_height);

    // Helper function for sym_trap to get information about the current pose
    Point6D copy_current_pose();

    bool sym_trap_running;

    void update_image_list_widget(); /*Updates ui.image_list_widget*/

public Q_SLOTS:
    // Call Optimizer Launch
    void optimizer_launch_slot();

    /*Load Buttons*/
    void on_load_calibration_button_clicked(); /*Load Calibration Clicked*/
    void on_load_image_button_clicked();       /*Load Images*/
    void on_load_model_button_clicked();       /*Load Models*/

    /*Biplane View Button (Monoplane is Biplane A, Biplans is Biplane B*/
    /*Plan 006 U9: the VM slice of both camera slots is thinned onto the
     * shared seams — the active-camera mirror write (SetActiveCamera on the
     * shared SessionController; the radio stays the source of truth) and the
     * inline save-last-pose copies (SaveLastPoseToStorage, the pinned
     * camera-A/B table rows). Radio decisions (DecideCameraRadios) and the
     * display-only blocks stay view-side (L16).*/
    void on_camera_A_radio_button_clicked();

    void on_single_model_radio_button_clicked();
    void on_multiple_model_radio_button_clicked();

    void on_camera_B_radio_button_clicked();

    /*List Widgets*/
    void
    on_image_list_widget_itemSelectionChanged(); /*Image List Widget Changed*/
    void
    on_model_list_widget_itemSelectionChanged(); /*Model List Widget Changed*/

    void on_original_image_radio_button_clicked();
    void on_inverted_image_radio_button_clicked();
    void on_edges_image_radio_button_clicked();
    void on_dilation_image_radio_button_clicked();

    /*Model Radio Buttons*/
    void on_original_model_radio_button_clicked();

    void on_solid_model_radio_button_clicked();

    void on_transparent_model_radio_button_clicked();

    void on_wireframe_model_radio_button_clicked();

    /*Edge Buttons*/
    void on_aperture_spin_box_valueChanged();

    void on_low_threshold_slider_valueChanged();

    void on_high_threshold_slider_valueChanged();

    void on_apply_all_edge_button_clicked();

    void on_reset_edge_button_clicked();

    /*MenuBar*/
    void on_actionSave_Pose_triggered();

    void on_actionSave_Kinematics_triggered();

    void on_actionLoad_Pose_triggered();

    void on_actionLoad_Kinematics_triggered();

    void on_actionAbout_JointTrack_Auto_triggered();

    void on_actionControls_triggered();

    void on_actionStop_Optimizer_triggered();

    void on_actionOptimizer_Settings_triggered();

    void on_actionDRR_Settings_triggered();

    void on_actionReset_View_triggered();

    void on_actionReset_Normal_Up_triggered();

    void on_actionModel_Interaction_Mode_triggered();

    void on_actionCamera_Interaction_Mode_triggered();

    void on_actionSegment_FemHR_triggered();

    void on_actionSegment_TibHR_triggered();

    void on_actionReset_Remove_All_Segmentation_triggered();

    void on_actionEstimate_Femoral_Implant_s_triggered();

    void on_actionEstimate_Tibial_Implant_s_triggered();

    void on_actionCopy_Next_Pose_triggered();

    void on_actionCopy_Previous_Pose_triggered();

    void on_actionAmbiguous_Pose_Processing_triggered();

    /*Optimization Buttons*/
    void on_optimize_button_clicked();

    void on_optimize_all_button_clicked();

    void on_optimize_each_button_clicked();

    void on_optimize_from_button_clicked();

    void on_actionOptimize_Backward_triggered();

    /*OPTIMIZATION SLOTS*/
    /*Update Blue Current Optimum*/
    void onUpdateOptimum(
        double,
        double,
        double,
        double,
        double,
        double,
        unsigned int);

    /*Finished Optimizing Frame, Send Optimum to MainScreen (the shared
     * controller's terminal-frame relay; the out-of-bounds status travels
     * on the relay so the view can box — L14). The controller already
     * persisted the pose at its tracked current frame.*/
    void onOptimizedFrame(
        double,
        double,
        double,
        double,
        double,
        double,
        bool,
        unsigned int,
        bool,
        QString,
        bool);

    /*The shared controller's severity-carrying message channel (L14): the
     * widgets preserves its box-type distinctions.*/
    void onControllerMessage(
        const QString& title,
        const QString& message,
        OptimizerRunController::Severity severity);

    /*Update Display with Speed, Cost Function Calls, Current Minimum*/
    void onUpdateDisplay(double, int, double, unsigned int);

    /*Update Dilation Background if Radio Button is on Dilation and Moving
     * Betweeen Trunks and Branches*/
    void onUpdateDilationBackground();

    void
    updateOrientationSymTrap_MS(double, double, double, double, double, double);

    /*On Optimizer Control Windows Save Setting*/
    void onSaveSettings(
        OptimizerSettings,
        jta_cost_function::CostFunctionManager,
        jta_cost_function::CostFunctionManager,
        jta_cost_function::CostFunctionManager);

protected:
    void resizeEvent(QResizeEvent* event) override;

    void keyPressEvent(QKeyEvent* event) override;
};

#endif /* MAINSCREEN_H */
