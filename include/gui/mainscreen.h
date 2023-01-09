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
#include <QtWidgets/QMainWindow>
#include <qactiongroup.h>
#include "ui_mainscreen.h"
#include <memory.h>
/*Font*/
#include <qfont.h>

/*Key Event*/
#include <QKeyEvent>

#include "nfd/nfd.h"

/*Direct Data Structures*/
#include "core/data_structures_6D.h"

/*Custom Calibration Struct (Used in CUDA GPU METRICS)*/
#include "core/calibration.h"

/*VTK*/
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


/*Frame and Model and Location Storage*/
#include "core/frame.h"
#include "core/model.h"
#include "core/location_storage.h"

/*Optimizer Settings*/
#include "core/optimizer_settings.h"

/*Optimizer Manager*/
#include "core/optimizer_manager.h"

/*Optimizer Settings Control Window*/
#include "gui/settings_control.h"

/*DRR Settings Control Window*/
#include "drr_tool.h"

/* Symmetry Trap Analysis Window*/

/*Cost Function Library*/
#include "cost_functions/CostFunctionManager.h"

/*CostFunctionTools*/
#include "camera_calibration.h"

/*machine_learning_tools*/
#include "core/machine_learning_tools.h"

#include "nfd/nfd.h"

#include "gui/viewer.h"

/**
 * @brief The MainScreen object that inherits the QMainWindow object type. This object serves as the class hosting all the items on the main window.
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

	/*Stop Optimizer*/
	void StopOptimizer();

	// [SYM TRAP] Send out optimizer time remaining
	void UpdateTimeRemaining(int);

private:
	Ui::MainScreenClass ui;

	double UF_BLUE[3] = {0, 82, 204};
	double UF_ORANGE[3] = {255, 77, 0};

	void print_selected_item();

	int curr_frame();

	float start_time;
	void remove_background_highlights_from_model_list_widget();

	/*GUI FUNCTIONS*/
	/*Arrange Layout (Do this in code so scales across different DPI monitors and handles weird fonts)*/
	void ArrangeMainScreenLayout(QFont application_font);

	/*Private Variables*/
	/*Original Sizes After Construction for Main Screen List Widgets, their Group Boxes and QVTK Widget*/
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

	/*VTK Variables*/
	std::vector<vtkSmartPointer<vtkActor>> model_actor_list;
	std::vector<vtkSmartPointer<vtkPolyDataMapper>> model_mapper_list;
	vtkSmartPointer<vtkRenderer> renderer;
	vtkSmartPointer<vtkImageData> current_background;
	vtkSmartPointer<vtkSTLReader> stl_reader;
	vtkSmartPointer<vtkDataSetMapper> image_mapper;
	vtkSmartPointer<vtkActor> actor_image;
	vtkSmartPointer<vtkTextActor> actor_text;
	vtkSmartPointer<vtkImageImport> importer;
	vtkSmartPointer<vtkInteractorStyleTrackballCamera> camera_style_interactor;


	std::shared_ptr<Viewer> vw = std::make_shared<Viewer>();


	/*View Menu Radio Button Container*/
	QActionGroup *alignmentGroup, *alignmentGroupSegment;

	/*Frame/Model Containers*/
	std::vector<Frame> loaded_frames;
	std::vector<Frame> loaded_frames_B; /*If Biplane mode, need second group of loaded frames for camera B*/
	std::vector<Model> loaded_models;
	/*Location Storage Class*/
	LocationStorage model_locations_;

	QModelIndexList selected_model_indices();

	/*Index of Previously Selected Frame/Models*/
	int previous_frame_index_;
	QModelIndexList previous_model_indices_;

	/*Save the Pose From The Last Selected Frame*/
	void SaveLastPose();

	/*Optimizer Settings That Must Be Set in Constructor and Changed on OSettings Update */
	OptimizerSettings optimizer_settings_;

	/*Copy of the Above Only Used While Optimizing to Display Output*/
	OptimizerSettings display_optimizer_settings_;

	/*Cost Function Managers (from JTA Cost Function Library) for each stage of DIRECT-JTA Optimizer*/
	jta_cost_function::CostFunctionManager trunk_manager_;
	jta_cost_function::CostFunctionManager branch_manager_;
	jta_cost_function::CostFunctionManager leaf_manager_; // For extra Z-translation usually (esp. when monoplane)

	/*Function That Saves Dilation as 0 if No Trunk Manager has a Dilation Int Parameter,
	else saves all the Dilation Images for Each Frame as the Dilation Constant*/
	void UpdateDilationFrames();


	/*Optimization Function: Packages Off The Optimization process in
	a new thread*/

	/*Launch Optimizer*/


	void
	LaunchOptimizer(QString directive); //Directive Says whether it is Optimize Single, From, All, or Each (or Sym_Trap)

	/*Optimizer Thread and Manager*/
	QThread* optimizer_thread;
	OptimizerManager* optimizer_manager;

	/*Disable and Enable MainScreen During and After Optimization*/
	void DisableAll();

	void EnableAll();

	/*Function That Loads Settings from Registry or (If First Time Loading
	Saves Default Settings*/
	void LoadSettingsBetweenSessions();

	/*Mat to Vtk*/
	void matToVTK(cv::Mat Input, vtkSmartPointer<vtkImageData> Output);

	/*Optimizer Window Control*/
	SettingsControl* settings_control;

	/*Sym Trap Window*/

	/*Calculate Viewing Angle (Accounts for Offsets)*/
	double CalculateViewingAngle(int width, int height, bool CameraA);

	/*Helper Function To Segment And Update Frames According to Model File*/
	void segmentHelperFunction(std::string pt_model_location, unsigned int input_width, unsigned int input_height);

	// Helper function for sym_trap to get information about the current pose
	Point6D copy_current_pose();

	bool sym_trap_running;

public Q_SLOTS:
	// Call Optimizer Launch
	void optimizer_launch_slot();

	/*Load Buttons*/
	void on_load_calibration_button_clicked(); /*Load Calibration Clicked*/
	void on_load_image_button_clicked(); /*Load Images*/
	void on_load_model_button_clicked(); /*Load Models*/


	/*Biplane View Button (Monoplane is Biplane A, Biplans is Biplane B*/
	void on_camera_A_radio_button_clicked();

	void on_camera_B_radio_button_clicked();

	/*List Widgets*/
	void on_image_list_widget_itemSelectionChanged(); /*Image List Widget Changed*/
	void on_model_list_widget_itemSelectionChanged(); /*Model List Widget Changed*/

	/*Multiple Selection For Models Radio buttons*/
	void on_single_model_radio_button_clicked();

	void on_multiple_model_radio_button_clicked();

	/*Radio Buttons*/
	/*Image Radio Buttons*/
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

	void on_actionNFD_Pose_Estimate_triggered();

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
	void onUpdateOptimum(double, double, double, double, double, double, unsigned int);

	/*Finished Optimizing Frame, Send Optimum to MainScreen*/
	void onOptimizedFrame(double, double, double, double, double, double, bool, unsigned int, bool, QString);

	/*Uh oh There was an Error. String contains the message
	*/
	void onOptimizerError(QString error_message);

	/*Update Display with Speed, Cost Function Calls, Current Minimum*/
	void onUpdateDisplay(double, int, double, unsigned int);

	/*Update Dilation Background if Radio Button is on Dilation and Moving Betweeen Trunks and Branches*/
	void onUpdateDilationBackground();

	void updateOrientationSymTrap_MS(double, double, double, double, double, double);

	/*On Optimizer Control Windows Save Setting*/
	void
	onSaveSettings(OptimizerSettings, jta_cost_function::CostFunctionManager, jta_cost_function::CostFunctionManager,
	               jta_cost_function::CostFunctionManager);

protected:
	void resizeEvent(QResizeEvent* event) override;

	void keyPressEvent(QKeyEvent* event) override;
};

#endif /* MAINSCREEN_H */
