/*Main Screen Header*/
#include "gui/mainscreen.h"

/*Font Manipulation*/
#include <qfontmetrics.h>

/*Settings Constants*/
#include "core/settings_constants.h"

/*Size Constants*/
#include "core/mainscreen_size_constants.h"

/*Process Events*/
#include <qapplication.h>

/*Settings*/
#include <qsettings.h>
#include <qdesktopwidget.h>

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
#include <torch/torch.h>
#include <torch/script.h>
#include <c10/cuda/CUDAMacros.h>

#include "core/machine_learning_tools.h"
#include "core/ambiguous_pose_processing.h"

using namespace std;

/*Temporary Functions to Ease VTK Interaction and STL Loading*/
/*Mat to VTK Function*/
void MainScreen::matToVTK(cv::Mat Input, vtkSmartPointer<vtkImageData> Output) {
	//assert(Input.data != NULL);
	//vtkImageImport *importer = vtkImageImport::New();
	if (Output) {
		importer->SetOutput(Output);
	}
	importer->SetDataSpacing(1, 1, 1);
	importer->SetDataOrigin(0, 0, 0);
	importer->SetWholeExtent(0, Input.size().width - 1, 0,
	                         Input.size().height - 1, 0, 0);
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
	//Used to Set Height/2 = To The Bigger of the Width/2 + X Offset vs Height/2 + Y Offset,
	// now just set to height/2 + y_offset
	if (CameraA) {
		double y = height * calibration_file_.camera_A_principal_.pixel_pitch_ / 2.0 +
			abs(calibration_file_.camera_A_principal_.principal_y_);
		return 180.0 / 3.1415926535897932384626433832795028841971693993751 * 2.0 *
			atan2(y, calibration_file_.camera_A_principal_.principal_distance_);
	}
	double y =height * calibration_file_.camera_B_principal_.pixel_pitch_ / 2.0 +
		abs(calibration_file_.camera_B_principal_.principal_y_);
	return 180.0 / 3.1415926535897932384626433832795028841971693993751 * 2.0 *
		atan2(y, calibration_file_.camera_B_principal_.principal_distance_);
}

/*Constructor*/
MainScreen::MainScreen(QWidget* parent)
	: QMainWindow(parent) {
	ui.setupUi(this);


	this->start_time = -1;
	sym_trap_running = false;

	/*Set Minimum and Maximum for Sliders*/
	ui.low_threshold_slider->setMinimum(0);
	ui.high_threshold_slider->setMinimum(0);
	ui.low_threshold_slider->setMaximum(800);
	ui.high_threshold_slider->setMaximum(800);

	/*Load Settings (THIS MUST BE DONE FIRST)*/
	LoadSettingsBetweenSessions();

	/*Set Label to Threshold Values*/
	ui.low_threshold_value->setText(QString::number(ui.low_threshold_slider->value()));
	ui.high_threshold_value->setText(QString::number(ui.high_threshold_slider->value()));

	/*Set Font, Font Size*/
	QFont application_font("Segoe UI", FONT_SIZE);
	QApplication::setFont(application_font);

	/*Set Up Settings Control Window*/
	settings_control = new SettingsControl(this);
	connect(settings_control,
	        SIGNAL(
		        SaveSettings(OptimizerSettings, jta_cost_function::CostFunctionManager, jta_cost_function::
			        CostFunctionManager, jta_cost_function::CostFunctionManager)),
	        this, SLOT(
		        onSaveSettings(OptimizerSettings, jta_cost_function::CostFunctionManager, jta_cost_function::
			        CostFunctionManager, jta_cost_function::CostFunctionManager)), Qt::DirectConnection);

	/* SYM TRAP */
	// Setup Sym Trap Window Obj
	//this->sym_trap_control = new sym_trap();
	// Connect signals for launching sym trap optimizer and updating progress bar
	//connect(sym_trap_control->ui.optimize, SIGNAL(clicked()), this, SLOT(optimizer_launch_slot()));
	//connect(this, SIGNAL(UpdateTimeRemaining(int)), sym_trap_control->ui.progressBar, SLOT(setValue(int)));


	/*Disable Stop Optimizer*/
	ui.actionStop_Optimizer->setDisabled(true);

	/*Set Up Icons For Files*/
	/*File*/
	ui.actionLoad_Pose->setIcon(QPixmap(":Menu_Icons/Resources/load_icon.png"));
	ui.actionSave_Pose->setIcon(QPixmap(":Menu_Icons/Resources/save_icon.png"));
	ui.actionQuit->setIcon(QPixmap(":Menu_Icons/Resources/quit_icon.ico"));

	/*View*/
	ui.actionReset_View->setIcon(QPixmap(":Menu_Icons/Resources/camerared.png"));
	ui.actionReset_Normal_Up->setIcon(QPixmap(":Menu_Icons/Resources/normaluppink.png"));

	/*Options*/
	ui.actionOptimizer_Settings->setIcon(QPixmap(":Menu_Icons/Resources/optimizer_settings_icon.png"));
	ui.actionRegion_Selection->setIcon(QPixmap(":Menu_Icons/Resources/selection_icon.ico"));
	ui.actionCenter_Placement->setIcon(QPixmap(":Menu_Icons/Resources/center_placement_icon.png"));
	ui.actionStop_Optimizer->setIcon(QPixmap(":Menu_Icons/Resources/stop_icon.png"));

	/*Help*/
	ui.actionAbout_JointTrack_Auto->setIcon(QPixmap(":Menu_Icons/Resources/help_icon.png"));
	ui.actionControls->setIcon(QPixmap(":Menu_Icons/Resources/controls_icon.png"));

	/*Set up RadioButton like functionality for View Menu Interaction Modes*/
	alignmentGroup = new QActionGroup(this);
	alignmentGroup->addAction(ui.actionModel_Interaction_Mode);
	alignmentGroup->addAction(ui.actionCamera_Interaction_Mode);
	ui.actionModel_Interaction_Mode->setChecked(true);

	/*Set up RadioButton like functionality for Segment Menu*/
	alignmentGroupSegment = new QActionGroup(this);
	alignmentGroupSegment->addAction(ui.actionBlack_Implant_Silhouettes_in_Original_Image_s);
	alignmentGroupSegment->addAction(ui.actionWhite_Implant_Silhouettes_in_Original_Image_s);
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
	QRect rec = QApplication::desktop()->availableGeometry();
	if (MINIMUM_WIDTH <= rec.width() && MINIMUM_HEIGHT <= rec.height()) {
		showMaximized();
	}

	/*INitialize Location Storage*/
	model_locations_ = LocationStorage();
	vw->initialize_vtk_pointers();
	vw->initialize_vtk_mappers();
	vw->initialize_vtk_renderers();
	/*Selection Model for Models*/
	ui.single_model_radio_button->setChecked(true);
	ui.model_list_widget->setSelectionMode(QAbstractItemView::SingleSelection);

	/*Have NOT Loaded Calibration Files Yet*/
	calibrated_for_monoplane_viewport_ = false;
	calibrated_for_biplane_viewport_ = false;

	/*Index of Previously Selected Frame/Models*/
	previous_frame_index_ = -1;
	///*Set up VTK*/
	vtkObject::GlobalWarningDisplayOff(); /*Turn off error display*/
	renderer = vw->get_renderer();
	actor_image = vw->get_actor_image();
	current_background = vw->get_current_background();
	stl_reader = vw->get_stl_reader();

	model_mapper_list = vw->get_model_mapper_list();
	model_actor_list = vw->get_model_actor_list();
	image_mapper = vw->get_image_mapper();
	actor_text = vw->get_actor_text();
	importer = vw->get_importer();
	key_press_vtk = vtkSmartPointer<KeyPressInteractorStyle>::New(); /*Custom Interactor from JTA*/
	key_press_vtk->initialize_MainScreen(this);
	key_press_vtk->initialize_viewer(vw);
	camera_style_interactor = vtkSmartPointer<CameraInteractorStyle>::New();
	//vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New(); /*Alternate Angled Interactor*/
	/*Text Actor Property*/
	actor_text->GetTextProperty()->SetFontSize(16);
	actor_text->GetTextProperty()->SetFontFamilyToCourier();
	actor_text->SetPosition2(0, 0);
	actor_text->GetTextProperty()->SetColor(214.0 / 255.0, 108.0 / 255.0, 35.0 / 255.0); //Earth Red

	/*Set Up Connections*/
	image_mapper->SetInputData(current_background);
	actor_image->SetPickable(0);
	actor_text->SetPickable(0);
	actor_image->SetMapper(image_mapper);
	vw->load_render_window(ui.qvtk_widget->renderWindow());
	//vw->load_renderers_into_render_window();
	ui.qvtk_widget->renderWindow()->Render();

	/*Interactor*/
	key_press_vtk->AutoAdjustCameraClippingRangeOff();
	vw->load_in_interactor_style(key_press_vtk);
	//ui.qvtk_widget->GetRenderWindow()->GetInteractor()->SetInteractorStyle(key_press_vtk);


	/*Pose Estimate Progress and Label Not Visible*/
	ui.pose_progress->setValue(0);
	ui.pose_progress->setVisible(false);
	ui.pose_label->setVisible(false);

	/*Update*/
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();

	/*Not Currently Optimizing*/
	currently_optimizing_ = false;

}

/*Destructor*/
MainScreen::~MainScreen() {
	/*Delete QAction Group for View Menu*/
	delete alignmentGroup;
}

/*Arrange Main Screen*/
void MainScreen::ArrangeMainScreenLayout(QFont application_font) {
	/*Resize Buttons based on Font Size In Order to be Compatible with High DPI Monitors*/
	QFontMetrics font_metrics(application_font);

	/*Adjust for Title Height*/
	this->setStyleSheet(
		this->styleSheet() += "QGroupBox { margin-top: " + QString::number(font_metrics.height() / 2) + "px; }");
	int group_box_to_top_button_y = font_metrics.height() / 2;

	/*Preprocessor Width*/
	/*Find Width of Buttons*/
	int preprocessor_button_width = font_metrics.width(ui.preprocessor_box->title());
	if (preprocessor_button_width < font_metrics.width(ui.load_calibration_button->text())) {
		preprocessor_button_width = font_metrics.width(ui.load_calibration_button->text());
	}
	if (preprocessor_button_width < font_metrics.width(ui.load_image_button->text())) {
		preprocessor_button_width = font_metrics.width(ui.load_image_button->text());
	}
	if (preprocessor_button_width < font_metrics.width(ui.load_model_button->text())) {
		preprocessor_button_width = font_metrics.width(ui.load_model_button->text());
	}
	/*Augment with Padding to find width of group_box*/
	preprocessor_button_width += INSIDE_BUTTON_PADDING_X;
	int preprocessor_group_box_width = preprocessor_button_width + 2 * GROUP_BOX_TO_BUTTON_PADDING_X;

	/*Optimization Directives Width*/
	int optimizer_button_width = font_metrics.width(ui.optimize_button->text());
	if (optimizer_button_width < font_metrics.width(ui.optimize_all_button->text())) {
		optimizer_button_width = font_metrics.width(ui.optimize_all_button->text());
	}
	if (optimizer_button_width < font_metrics.width(ui.optimize_each_button->text())) {
		optimizer_button_width = font_metrics.width(ui.optimize_each_button->text());
	}
	if (optimizer_button_width < font_metrics.width(ui.optimize_from_button->text())) {
		optimizer_button_width = font_metrics.width(ui.optimize_from_button->text());
	}
	/*Augment with padding to find width of group box*/
	optimizer_button_width += INSIDE_BUTTON_PADDING_X;
	int optimization_group_box_width = font_metrics.width(ui.optimization_box->title());
	if (optimization_group_box_width < 2 * optimizer_button_width + BUTTON_TO_BUTTON_PADDING_X + 2 *
		GROUP_BOX_TO_BUTTON_PADDING_X) {
		optimization_group_box_width = 2 * optimizer_button_width + BUTTON_TO_BUTTON_PADDING_X + 2 *
			GROUP_BOX_TO_BUTTON_PADDING_X;
	}

	/*image View Width*/
	int image_view_button_width = font_metrics.width(ui.original_image_radio_button->text());
	if (image_view_button_width < font_metrics.width(ui.inverted_image_radio_button->text())) {
		image_view_button_width = font_metrics.width(ui.inverted_image_radio_button->text());
	}
	if (image_view_button_width < font_metrics.width(ui.edges_image_radio_button->text())) {
		image_view_button_width = font_metrics.width(ui.edges_image_radio_button->text());
	}
	if (image_view_button_width < font_metrics.width(ui.dilation_image_radio_button->text())) {
		image_view_button_width = font_metrics.width(ui.dilation_image_radio_button->text());
	}
	/*Augment with Padding to find width of group_box*/
	image_view_button_width += INSIDE_RADIO_BUTTON_PADDING_X;
	int image_view_group_box_width = font_metrics.width(ui.image_view_box->title());
	if (image_view_group_box_width < 2 * image_view_button_width + BUTTON_TO_BUTTON_PADDING_X + 2 *
		GROUP_BOX_TO_BUTTON_PADDING_X) {
		image_view_group_box_width = 2 * image_view_button_width + BUTTON_TO_BUTTON_PADDING_X + 2 *
			GROUP_BOX_TO_BUTTON_PADDING_X;
	}

	/*image Selection Width*/
	int image_selection_button_width = font_metrics.width(ui.camera_A_radio_button->text());
	if (image_selection_button_width < font_metrics.width(ui.camera_B_radio_button->text())) {
		image_selection_button_width = font_metrics.width(ui.camera_B_radio_button->text());
	}
	/*Augment with Padding to find width of group_box*/
	image_selection_button_width += INSIDE_RADIO_BUTTON_PADDING_X;
	int image_selection_group_box_width = font_metrics.width(ui.image_selection_box->title());
	if (image_selection_group_box_width < 2 * image_selection_button_width + BUTTON_TO_BUTTON_PADDING_X + 2 *
		GROUP_BOX_TO_BUTTON_PADDING_X) {
		image_selection_group_box_width = 2 * image_selection_button_width + BUTTON_TO_BUTTON_PADDING_X + 2 *
			GROUP_BOX_TO_BUTTON_PADDING_X;
	}

	/*Get Largest Width Across Left Side Column*/
	int left_column_width = std::max(std::max(std::max(preprocessor_group_box_width,
	                                                   optimization_group_box_width),
	                                          image_view_group_box_width),
	                                 image_selection_group_box_width);

	/*Set Group Box Widths, Heights, and Member Objects Accordingly*/
	int text_height = font_metrics.height();
	/*Preprocessor*/
	ui.preprocessor_box->setGeometry(
		QRect(APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X,
		      APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y,
		      left_column_width,
		      2 * GROUP_BOX_TO_BUTTON_PADDING_Y + 2 * BUTTON_TO_BUTTON_PADDING_Y + 3 * (text_height +
			      INSIDE_BUTTON_PADDING_Y) + group_box_to_top_button_y));
	ui.load_calibration_button->setGeometry(
		QRect((left_column_width - preprocessor_button_width) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
		      preprocessor_button_width,
		      text_height + INSIDE_BUTTON_PADDING_Y));
	ui.load_image_button->setGeometry(
		QRect((left_column_width - preprocessor_button_width) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + (text_height + INSIDE_BUTTON_PADDING_Y + BUTTON_TO_BUTTON_PADDING_Y) +
		      group_box_to_top_button_y,
		      preprocessor_button_width,
		      text_height + INSIDE_BUTTON_PADDING_Y));
	ui.load_model_button->setGeometry(
		QRect((left_column_width - preprocessor_button_width) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + 2 * (text_height + INSIDE_BUTTON_PADDING_Y + BUTTON_TO_BUTTON_PADDING_Y) +
		      group_box_to_top_button_y,
		      preprocessor_button_width,
		      text_height + INSIDE_BUTTON_PADDING_Y));
	/*Optimization Directive*/
	ui.optimization_box->setGeometry(
		QRect(APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X,
		      GROUP_BOX_TO_GROUP_BOX_Y + ui.preprocessor_box->geometry().bottomLeft().y(),
		      left_column_width,
		      2 * GROUP_BOX_TO_BUTTON_PADDING_Y + 1 * BUTTON_TO_BUTTON_PADDING_Y + 2 * (text_height +
			      INSIDE_BUTTON_PADDING_Y) + group_box_to_top_button_y));
	ui.optimize_button->setGeometry(
		QRect((left_column_width - (2 * optimizer_button_width + BUTTON_TO_BUTTON_PADDING_X)) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
		      optimizer_button_width,
		      text_height + INSIDE_BUTTON_PADDING_Y));
	ui.optimize_all_button->setGeometry(
		QRect((left_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
		      optimizer_button_width,
		      text_height + INSIDE_BUTTON_PADDING_Y));
	ui.optimize_each_button->setGeometry(
		QRect((left_column_width - (2 * optimizer_button_width + BUTTON_TO_BUTTON_PADDING_X)) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + (text_height + INSIDE_BUTTON_PADDING_Y + BUTTON_TO_BUTTON_PADDING_Y) +
		      group_box_to_top_button_y,
		      optimizer_button_width,
		      text_height + INSIDE_BUTTON_PADDING_Y));
	ui.optimize_from_button->setGeometry(
		QRect((left_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + (text_height + INSIDE_BUTTON_PADDING_Y + BUTTON_TO_BUTTON_PADDING_Y) +
		      group_box_to_top_button_y,
		      optimizer_button_width,
		      text_height + INSIDE_BUTTON_PADDING_Y));
	/*image View*/
	ui.image_view_box->setGeometry(
		QRect(APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X,
		      GROUP_BOX_TO_GROUP_BOX_Y + ui.optimization_box->geometry().bottomLeft().y(),
		      left_column_width,
		      2 * GROUP_BOX_TO_BUTTON_PADDING_Y + 1 * BUTTON_TO_BUTTON_PADDING_Y + 2 * (text_height +
			      INSIDE_RADIO_BUTTON_PADDING_Y) + group_box_to_top_button_y));
	ui.original_image_radio_button->setGeometry(
		QRect((left_column_width - (2 * image_view_button_width + BUTTON_TO_BUTTON_PADDING_X)) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
		      image_view_button_width,
		      text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
	ui.inverted_image_radio_button->setGeometry(
		QRect((left_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
		      image_view_button_width,
		      text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
	ui.edges_image_radio_button->setGeometry(
		QRect((left_column_width - (2 * image_view_button_width + BUTTON_TO_BUTTON_PADDING_X)) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + (text_height + INSIDE_RADIO_BUTTON_PADDING_Y + BUTTON_TO_BUTTON_PADDING_Y)
		      + group_box_to_top_button_y,
		      image_view_button_width,
		      text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
	ui.dilation_image_radio_button->setGeometry(
		QRect((left_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + (text_height + INSIDE_RADIO_BUTTON_PADDING_Y + BUTTON_TO_BUTTON_PADDING_Y)
		      + group_box_to_top_button_y,
		      image_view_button_width,
		      text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
	/*image Selection*/
	/*Check Size of Application, If not big enough for listwidget, resize application*/
	int image_selection_box_height = 2 * GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y +
		text_height + INSIDE_RADIO_BUTTON_PADDING_Y +
		RADIO_BUTTON_TO_LIST_WIDGET_PADDING_Y + MINIMUM_LIST_WIDGET_SIZE;
	if (GROUP_BOX_TO_GROUP_BOX_Y + ui.image_view_box->geometry().bottomLeft().y() + image_selection_box_height +
		APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y + group_box_to_top_button_y + ui.menuBar->height()
		< this->height()) {
		image_selection_box_height = this->height() - (GROUP_BOX_TO_GROUP_BOX_Y + ui.image_view_box->geometry().
			bottomLeft().y() + APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y + group_box_to_top_button_y + ui.menuBar->
			height());
	}
	else {
		this->setMinimumHeight(
			GROUP_BOX_TO_GROUP_BOX_Y + ui.image_view_box->geometry().bottomLeft().y() + image_selection_box_height +
			APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y + group_box_to_top_button_y + ui.menuBar->height());
	}
	ui.image_selection_box->setGeometry(
		QRect(APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X,
		      GROUP_BOX_TO_GROUP_BOX_Y + ui.image_view_box->geometry().bottomLeft().y(),
		      left_column_width,
		      image_selection_box_height));
	ui.camera_A_radio_button->setGeometry(
		QRect((left_column_width - (2 * image_selection_button_width + BUTTON_TO_BUTTON_PADDING_X)) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
		      image_selection_button_width,
		      text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
	ui.camera_B_radio_button->setGeometry(
		QRect((left_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
		      image_selection_button_width,
		      text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
	ui.image_list_widget->setGeometry(
		QRect(GROUP_BOX_TO_BUTTON_PADDING_X,
		      ui.camera_A_radio_button->geometry().bottomLeft().y() + RADIO_BUTTON_TO_LIST_WIDGET_PADDING_Y,
		      left_column_width - 2 * GROUP_BOX_TO_BUTTON_PADDING_X,
		      ui.image_selection_box->height() - (ui.camera_A_radio_button->geometry().bottomLeft().y() +
			      RADIO_BUTTON_TO_LIST_WIDGET_PADDING_Y + GROUP_BOX_TO_BUTTON_PADDING_Y)));

	/*Dilation Box Width*/
	int right_button_bigger = font_metrics.width(ui.apply_all_edge_button->text());

	/*Edge Detection Box Width*/
	int edge_detection_box = font_metrics.width(ui.edge_detection_box->title());
	if (edge_detection_box < font_metrics.width(ui.aperture_label->text()) + LABEL_TO_SPIN_BOX_PADDING_X +
		INSIDE_RADIO_BUTTON_PADDING_X + font_metrics.width("888")) {
		edge_detection_box = font_metrics.width(ui.aperture_label->text()) + LABEL_TO_SPIN_BOX_PADDING_X +
			INSIDE_RADIO_BUTTON_PADDING_X + font_metrics.width("888") + SPIN_BOX_TO_GROUP_BOX_PADDING_X;
	}
	if (edge_detection_box < font_metrics.width(ui.low_threshold_label->text()) + LABEL_TO_SPIN_BOX_PADDING_X +
		INSIDE_RADIO_BUTTON_PADDING_X + font_metrics.width("888")) {
		edge_detection_box = font_metrics.width(ui.low_threshold_label->text()) + LABEL_TO_SPIN_BOX_PADDING_X +
			INSIDE_RADIO_BUTTON_PADDING_X + font_metrics.width("888") + SPIN_BOX_TO_GROUP_BOX_PADDING_X;
	}
	if (edge_detection_box < font_metrics.width(ui.high_threshold_label->text()) + LABEL_TO_SPIN_BOX_PADDING_X +
		INSIDE_RADIO_BUTTON_PADDING_X + font_metrics.width("888")) {
		edge_detection_box = font_metrics.width(ui.high_threshold_label->text()) + LABEL_TO_SPIN_BOX_PADDING_X +
			INSIDE_RADIO_BUTTON_PADDING_X + font_metrics.width("888") + SPIN_BOX_TO_GROUP_BOX_PADDING_X;
	}
	if (edge_detection_box < 2 * right_button_bigger + BUTTON_TO_BUTTON_PADDING_X + 2 *
		INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X) {
		edge_detection_box = 2 * right_button_bigger + BUTTON_TO_BUTTON_PADDING_X + 2 *
			INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X + 2 * GROUP_BOX_TO_BUTTON_PADDING_X;
	}

	/*model View Width*/
	int model_view_button_width = font_metrics.width(ui.original_model_radio_button->text());
	if (model_view_button_width < font_metrics.width(ui.solid_model_radio_button->text())) {
		model_view_button_width = font_metrics.width(ui.solid_model_radio_button->text());
	}
	if (model_view_button_width < font_metrics.width(ui.transparent_model_radio_button->text())) {
		model_view_button_width = font_metrics.width(ui.transparent_model_radio_button->text());
	}
	if (model_view_button_width < font_metrics.width(ui.wireframe_model_radio_button->text())) {
		model_view_button_width = font_metrics.width(ui.wireframe_model_radio_button->text());
	}
	/*Augment with Padding to find width of group_box*/
	model_view_button_width += INSIDE_RADIO_BUTTON_PADDING_X;
	int model_view_group_box_width = font_metrics.width(ui.model_view_box->title());
	if (model_view_group_box_width < 2 * model_view_button_width + BUTTON_TO_BUTTON_PADDING_X + 2 *
		GROUP_BOX_TO_BUTTON_PADDING_X) {
		model_view_group_box_width = 2 * model_view_button_width + BUTTON_TO_BUTTON_PADDING_X + 2 *
			GROUP_BOX_TO_BUTTON_PADDING_X;
	}

	/*model Selection Width*/
	int model_selection_button_width = font_metrics.width(ui.single_model_radio_button->text());
	if (model_selection_button_width < font_metrics.width(ui.multiple_model_radio_button->text())) {
		model_selection_button_width = font_metrics.width(ui.multiple_model_radio_button->text());
	}
	/*Augment with Padding to find width of group_box*/
	model_selection_button_width += INSIDE_RADIO_BUTTON_PADDING_X;
	int model_selection_group_box_width = font_metrics.width(ui.model_selection_box->title());
	if (model_selection_group_box_width < 2 * model_selection_button_width + BUTTON_TO_BUTTON_PADDING_X + 2 *
		GROUP_BOX_TO_BUTTON_PADDING_X) {
		model_selection_group_box_width = 2 * model_selection_button_width + BUTTON_TO_BUTTON_PADDING_X + 2 *
			GROUP_BOX_TO_BUTTON_PADDING_X;
	}

	/*Get Largest Width Across Right Side Column*/
	int right_column_width = std::max(std::max(
		                                  edge_detection_box,
		                                  model_view_group_box_width),
	                                  model_selection_group_box_width);

	/*Set Group Box Widths, Heights, and Member Objects Accordingly*/
	/*Edge Detection Box*/
	ui.edge_detection_box->setGeometry(
		QRect(this->width() - APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X - right_column_width,
		      APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y,
		      //GROUP_BOX_TO_GROUP_BOX_Y + ui.dilation_box->geometry().bottomLeft().y(),
		      right_column_width,
		      2 * GROUP_BOX_TO_BUTTON_PADDING_Y + 7 * text_height + 2 * INSIDE_BUTTON_PADDING_Y + 3 *
		      INSIDE_SPIN_BOX_PADDING_Y + group_box_to_top_button_y + 3 * SPIN_BOX_TO_SPIN_BOX_PADDING_Y +
		      BUTTON_TO_BUTTON_PADDING_Y));
	ui.aperture_spin_box->setGeometry(
		QRect(right_column_width - (SPIN_BOX_TO_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X + font_metrics.
			      width("888")),
		      GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
		      INSIDE_SPIN_BOX_PADDING_X + font_metrics.width("888"),
		      text_height + INSIDE_SPIN_BOX_PADDING_Y));
	ui.aperture_label->setGeometry(
		QRect(right_column_width - (font_metrics.width(ui.aperture_label->text()) + LABEL_TO_SPIN_BOX_PADDING_X +
			      SPIN_BOX_TO_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X + font_metrics.width("888")),
		      GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
		      font_metrics.width(ui.aperture_label->text()),
		      text_height + INSIDE_SPIN_BOX_PADDING_Y));

	ui.low_threshold_label->setGeometry(
		QRect(right_column_width - (font_metrics.width(ui.low_threshold_label->text()) + LABEL_TO_SPIN_BOX_PADDING_X +
			      SPIN_BOX_TO_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X + font_metrics.width("888")),
		      ui.aperture_label->geometry().bottom() + SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
		      font_metrics.width(ui.low_threshold_label->text()),
		      text_height + INSIDE_SPIN_BOX_PADDING_Y));

	ui.low_threshold_value->setGeometry(
		QRect(3 + right_column_width - (SPIN_BOX_TO_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X + font_metrics.
			      width("888")),
		      ui.aperture_label->geometry().bottom() + SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
		      font_metrics.width(ui.low_threshold_label->text()),
		      text_height + INSIDE_SPIN_BOX_PADDING_Y));

	ui.low_threshold_slider->setGeometry(
		QRect((right_column_width - BUTTON_TO_BUTTON_PADDING_X) / 2 - INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X -
		      right_button_bigger,
		      ui.low_threshold_label->geometry().bottom() + .5 * SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
		      2 * right_button_bigger + BUTTON_TO_BUTTON_PADDING_X + 2 * INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X,
		      1.5 * text_height));

	ui.high_threshold_label->setGeometry(
		QRect(3 + right_column_width - (font_metrics.width(ui.high_threshold_label->text()) +
			      LABEL_TO_SPIN_BOX_PADDING_X + SPIN_BOX_TO_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X +
			      font_metrics.width("888")),
		      ui.low_threshold_slider->geometry().bottom() + SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
		      font_metrics.width(ui.high_threshold_label->text()),
		      text_height + INSIDE_SPIN_BOX_PADDING_Y));

	ui.high_threshold_value->setGeometry(
		QRect(right_column_width - (SPIN_BOX_TO_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X + font_metrics.
			      width("888")),
		      ui.low_threshold_slider->geometry().bottom() + SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
		      font_metrics.width(ui.high_threshold_label->text()),
		      text_height + INSIDE_SPIN_BOX_PADDING_Y));

	ui.high_threshold_slider->setGeometry(
		QRect((right_column_width - BUTTON_TO_BUTTON_PADDING_X) / 2 - INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X -
		      right_button_bigger,
		      ui.high_threshold_label->geometry().bottom() + .5 * SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
		      2 * right_button_bigger + BUTTON_TO_BUTTON_PADDING_X + 2 * INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X,
		      1.5 * text_height));

	ui.apply_all_edge_button->setGeometry(
		QRect((right_column_width - BUTTON_TO_BUTTON_PADDING_X) / 2 - INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X -
		      right_button_bigger,
		      ui.high_threshold_slider->geometry().bottom() + GROUP_BOX_TO_BUTTON_PADDING_Y,
		      2 * right_button_bigger + BUTTON_TO_BUTTON_PADDING_X + 2 * INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X,
		      text_height + INSIDE_BUTTON_PADDING_Y));
	ui.reset_edge_button->setGeometry(
		QRect((right_column_width - BUTTON_TO_BUTTON_PADDING_X) / 2 - INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X -
		      right_button_bigger,
		      ui.apply_all_edge_button->geometry().bottom() + BUTTON_TO_BUTTON_PADDING_Y,
		      2 * right_button_bigger + BUTTON_TO_BUTTON_PADDING_X + 2 * INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X,
		      text_height + INSIDE_BUTTON_PADDING_Y));
	ui.edge_detection_box->setGeometry(
		QRect(this->width() - APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X - right_column_width,
		      APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y,
		      //GROUP_BOX_TO_GROUP_BOX_Y + ui.dilation_box->geometry().bottomLeft().y(),
		      right_column_width,
		      ui.reset_edge_button->geometry().bottom() + GROUP_BOX_TO_BUTTON_PADDING_Y));

	/*model View*/
	ui.model_view_box->setGeometry(
		QRect(this->width() - APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X - right_column_width,
		      GROUP_BOX_TO_GROUP_BOX_Y + ui.edge_detection_box->geometry().bottomLeft().y(),
		      right_column_width,
		      2 * GROUP_BOX_TO_BUTTON_PADDING_Y + 1 * BUTTON_TO_BUTTON_PADDING_Y + 2 * (text_height +
			      INSIDE_RADIO_BUTTON_PADDING_Y) + group_box_to_top_button_y));
	ui.original_model_radio_button->setGeometry(
		QRect((right_column_width - (2 * model_view_button_width + BUTTON_TO_BUTTON_PADDING_X)) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
		      model_view_button_width,
		      text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
	ui.solid_model_radio_button->setGeometry(
		QRect((right_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
		      model_view_button_width,
		      text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
	ui.transparent_model_radio_button->setGeometry(
		QRect((right_column_width - (2 * model_view_button_width + BUTTON_TO_BUTTON_PADDING_X)) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + (text_height + INSIDE_RADIO_BUTTON_PADDING_Y + BUTTON_TO_BUTTON_PADDING_Y)
		      + group_box_to_top_button_y,
		      model_view_button_width,
		      text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
	ui.wireframe_model_radio_button->setGeometry(
		QRect((right_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + (text_height + INSIDE_RADIO_BUTTON_PADDING_Y + BUTTON_TO_BUTTON_PADDING_Y)
		      + group_box_to_top_button_y,
		      model_view_button_width,
		      text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
	/*model Selection*/
	/*Check Size of Application, If not big enough for listwidget, resize application*/
	int model_selection_box_height = 2 * GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y +
		text_height + INSIDE_RADIO_BUTTON_PADDING_Y +
		RADIO_BUTTON_TO_LIST_WIDGET_PADDING_Y + MINIMUM_LIST_WIDGET_SIZE;
	if (GROUP_BOX_TO_GROUP_BOX_Y + ui.model_view_box->geometry().bottomLeft().y() + model_selection_box_height +
		APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y + group_box_to_top_button_y + ui.menuBar->height()
		< this->height()) {
		model_selection_box_height = this->height() - (GROUP_BOX_TO_GROUP_BOX_Y + ui.model_view_box->geometry().
			bottomLeft().y() + APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y + group_box_to_top_button_y + ui.menuBar->
			height());
	}
	else {
		int old_height = this->height();
		this->setMinimumHeight(
			GROUP_BOX_TO_GROUP_BOX_Y + ui.model_view_box->geometry().bottomLeft().y() + model_selection_box_height +
			APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y + group_box_to_top_button_y + ui.menuBar->height());
		QRect updated_image_selection_group_box_geometry = ui.image_selection_box->geometry();
		updated_image_selection_group_box_geometry.setHeight(
			updated_image_selection_group_box_geometry.height() + this->height() - old_height);
		QRect updated_image_list_widget_geometry = ui.image_list_widget->geometry();
		updated_image_list_widget_geometry.setHeight(
			updated_image_list_widget_geometry.height() + this->height() - old_height);
		ui.image_selection_box->setGeometry(updated_image_selection_group_box_geometry);
		ui.image_list_widget->setGeometry(updated_image_list_widget_geometry);

	}
	ui.model_selection_box->setGeometry(
		QRect(this->width() - APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X - right_column_width,
		      GROUP_BOX_TO_GROUP_BOX_Y + ui.model_view_box->geometry().bottomLeft().y(),
		      right_column_width,
		      model_selection_box_height));
	ui.single_model_radio_button->setGeometry(
		QRect((right_column_width - (2 * model_selection_button_width + BUTTON_TO_BUTTON_PADDING_X)) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
		      model_selection_button_width,
		      text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
	ui.multiple_model_radio_button->setGeometry(
		QRect((right_column_width + BUTTON_TO_BUTTON_PADDING_X) / 2,
		      GROUP_BOX_TO_BUTTON_PADDING_Y + group_box_to_top_button_y,
		      model_selection_button_width,
		      text_height + INSIDE_RADIO_BUTTON_PADDING_Y));
	ui.model_list_widget->setGeometry(
		QRect(GROUP_BOX_TO_BUTTON_PADDING_X,
		      ui.single_model_radio_button->geometry().bottomLeft().y() + RADIO_BUTTON_TO_LIST_WIDGET_PADDING_Y,
		      right_column_width - 2 * GROUP_BOX_TO_BUTTON_PADDING_X,
		      ui.model_selection_box->height() - (ui.single_model_radio_button->geometry().bottomLeft().y() +
			      RADIO_BUTTON_TO_LIST_WIDGET_PADDING_Y + GROUP_BOX_TO_BUTTON_PADDING_Y)));
	int qvtk_side_length;

	/*Arrange QVTK Widget*/
	/*Check if there is enough room*/
	if (ui.edge_detection_box->geometry().left() - ui.preprocessor_box->geometry().right() - 2 *
		GROUP_BOX_TO_QVTK_PADDING_X > MINIMUM_QVTK_WIDGET_WIDTH) {
		if ((ui.edge_detection_box->geometry().left() - ui.preprocessor_box->geometry().
		                                  right() - 2 * GROUP_BOX_TO_QVTK_PADDING_X) > (ui.model_selection_box->geometry().bottom() - (ui.preprocessor_box->geometry()
			                                  .top() + font_metrics.height() / 2) + 1)) {
			qvtk_side_length = ui.model_selection_box->geometry().bottom() - (ui.preprocessor_box->geometry()
				.top() + font_metrics.height() / 2) + 1; // original height
		}else {
			qvtk_side_length = (ui.edge_detection_box->geometry().left() - ui.preprocessor_box->geometry().
				right() - 2 * GROUP_BOX_TO_QVTK_PADDING_X); // original width
		}
		ui.qvtk_widget->setGeometry(QRect(ui.preprocessor_box->geometry().right() + GROUP_BOX_TO_QVTK_PADDING_X,
		                                  ui.preprocessor_box->geometry().top() + font_metrics.height() / 2,
		                                  qvtk_side_length,
		                                  qvtk_side_length));
	}
	else {
		if (MINIMUM_QVTK_WIDGET_WIDTH >	ui.model_selection_box->geometry().bottom() - (ui.preprocessor_box->geometry()
			                                  .top() + font_metrics.height() / 2) + 1) {
			qvtk_side_length = ui.model_selection_box->geometry().bottom() - (ui.preprocessor_box->geometry()
				.top() + font_metrics.height() / 2) + 1;
		} else {
			qvtk_side_length = MINIMUM_QVTK_WIDGET_WIDTH;
		}
		ui.qvtk_widget->setGeometry(QRect(ui.preprocessor_box->geometry().right() + GROUP_BOX_TO_QVTK_PADDING_X,
		                                  ui.preprocessor_box->geometry().top() + font_metrics.height() / 2,
		                                  qvtk_side_length,
		                                  qvtk_side_length));
		/*Shift Over Right Hand Column*/
		ui.edge_detection_box->move(QPoint(ui.qvtk_widget->geometry().right() + GROUP_BOX_TO_QVTK_PADDING_X,
		                                   ui.edge_detection_box->geometry().top()));
		ui.model_view_box->move(QPoint(ui.qvtk_widget->geometry().right() + GROUP_BOX_TO_QVTK_PADDING_X,
		                               ui.model_view_box->geometry().top()));
		ui.model_selection_box->move(QPoint(ui.qvtk_widget->geometry().right() + GROUP_BOX_TO_QVTK_PADDING_X,
		                                    ui.model_selection_box->geometry().top()));
		/*New Minimum*/
		this->setMinimumWidth(ui.edge_detection_box->geometry().right() + APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X);
	}

	/*Initialize the Starting heights and widths*/
	/*Original Sizes After Construction for Main Screen List Widgets, their Group Boxes and QVTK Widget*/
	image_list_widget_starting_height_ = ui.image_list_widget->geometry().height();
	image_selection_box_starting_height_ = ui.image_selection_box->geometry().height();
	model_list_widget_starting_height_ = ui.model_list_widget->geometry().height();
	model_selection_box_starting_height_ = ui.model_selection_box->geometry().height();
	qvtk_widget_starting_height_ = ui.qvtk_widget->geometry().height();
	qvtk_widget_starting_width_ = ui.qvtk_widget->geometry().width();
}

/*Handle Resize Event*/
void MainScreen::resizeEvent(QResizeEvent* event) {
	/*Process Events*/
	qApp->processEvents();

	/*Resize Event*/
	QMainWindow::resizeEvent(event);

	/*Expansion Constants*/
	int horizontal_expansion = this->width() - this->minimumWidth();
	int vertical_expansion = this->height() - this->minimumHeight();
	int total_expansion;

	if (horizontal_expansion > vertical_expansion) {
		total_expansion = vertical_expansion;
	}
	else {
		total_expansion = horizontal_expansion;
	}

	/*Expand QVTK Widget*/
	ui.qvtk_widget->resize(qvtk_widget_starting_width_ + total_expansion,
	                       qvtk_widget_starting_height_ + total_expansion);

	/*Extend List Widgets and Their Group Boxes*/
	ui.image_list_widget->resize(ui.image_list_widget->size().width(),
	                             image_list_widget_starting_height_ + vertical_expansion);
	ui.model_list_widget->resize(ui.model_list_widget->size().width(),
	                             model_list_widget_starting_height_ + vertical_expansion);
	ui.image_selection_box->resize(ui.image_selection_box->size().width(),
	                               image_selection_box_starting_height_ + vertical_expansion);
	ui.model_selection_box->resize(ui.model_selection_box->size().width(),
	                               model_selection_box_starting_height_ + vertical_expansion);

	/*Shift Over Right Column*/
	ui.edge_detection_box->move(QPoint(ui.qvtk_widget->geometry().right() + GROUP_BOX_TO_QVTK_PADDING_X,
	                                   ui.edge_detection_box->geometry().top()));
	ui.model_view_box->move(QPoint(ui.qvtk_widget->geometry().right() + GROUP_BOX_TO_QVTK_PADDING_X,
	                               ui.model_view_box->geometry().top()));
	ui.model_selection_box->move(QPoint(ui.qvtk_widget->geometry().right() + GROUP_BOX_TO_QVTK_PADDING_X,
	                                    ui.model_selection_box->geometry().top()));

}

/*MENU BAR BUTTONS*/
/*File Menu*/
/*Save Pose*/
void MainScreen::on_actionSave_Pose_triggered() {
	//Save Single Pose
	//Selection Check
	/*Load Models Selected Indices*/
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	if (ui.image_list_widget->currentRow() < 0 || selected.size() == 0) {
		QMessageBox::critical(this, "Error!", "Select Frame and Model First!", QMessageBox::Ok);
		return;
	}

	//Must be in Single Selection Mode to Load Pose
	if (ui.multiple_model_radio_button->isChecked()) {
		QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Save Pose!", QMessageBox::Ok);
		return;
	}

	/*Save Pose*/
	SaveLastPose();

	/*Get Pose to Save*/
	Point6D saved_pose = model_locations_.GetPose(ui.image_list_widget->currentRow(), selected[0].row());

	//Open Save File Dialogue
	QString SavePoseExtension = QFileDialog::getSaveFileName(this, tr("Save Pose"), ".",
	                                                         tr("JTA Pose File (*.jtap);; Pose File (*.txt)"));
	QFile file(SavePoseExtension);
	if (file.open(QIODevice::ReadWrite | QIODevice::Text)) {
		QTextStream stream(&file);
		stream << "JTA_EULER_POSE\nX_TRAN\t\tY_TRAN\t\tZ_TRAN\t\tZ_ROT\t\tX_ROT\t\tY_ROT\n";
		if (QString::number(saved_pose.x).length() < 7) {
			stream << saved_pose.x << ",\t\t";
		}
		else {
			stream << saved_pose.x << ",\t";
		}
		if (QString::number(saved_pose.y).length() < 7) {
			stream << saved_pose.y << ",\t\t";
		}
		else {
			stream << saved_pose.y << ",\t";
		}
		if (QString::number(saved_pose.z).length() < 7) {
			stream << saved_pose.z << ",\t\t";
		}
		else {
			stream << saved_pose.z << ",\t";
		}
		if (QString::number(saved_pose.za).length() < 7) {
			stream << saved_pose.za << ",\t\t";
		}
		else {
			stream << saved_pose.za << ",\t";
		}
		if (QString::number(saved_pose.xa).length() < 7) {
			stream << saved_pose.xa << ",\t\t";
		}
		else {
			stream << saved_pose.xa << ",\t";
		}
		if (QString::number(saved_pose.ya).length() < 7) {
			stream << saved_pose.ya << ",\t\t";
		}
		else {
			stream << saved_pose.ya << ",\n";
		}
	}
}

/*Save Kinematics*/
void MainScreen::on_actionSave_Kinematics_triggered() {
	//Save Single Pose
	/*Load Models Selected Indices*/
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	if (ui.image_list_widget->currentRow() < 0 || selected.size() == 0) {
		QMessageBox::critical(this, "Error!", "Select Model and Load Frames First!", QMessageBox::Ok);
		return;
	}

	//Must be in Single Selection Mode to Load Pose
	if (ui.multiple_model_radio_button->isChecked()) {
		QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Save Kinematics!",
		                      QMessageBox::Ok);
		return;
	}

	/*Save Pose*/
	SaveLastPose();

	//Open Save File Dialogue
	QString SavePoseExtension = QFileDialog::getSaveFileName(this, tr("Save Kinematics"), ".",
	                                                         tr(
		                                                         "JTA Kinematics File (*.jtak);; Kinematics File (*.txt)"));
	QFile file(SavePoseExtension);
	if (file.open(QIODevice::ReadWrite | QIODevice::Text)) {
		QTextStream stream(&file);
		stream << "JTA_EULER_KINEMATICS\nX_TRAN\t\tY_TRAN\t\tZ_TRAN\t\tZ_ROT\t\tX_ROT\t\tY_ROT\n";
		/*Get Pose to Save*/
		for (int i = 0; i < ui.image_list_widget->count(); i++) {
			Point6D saved_pose = model_locations_.GetPose(i, selected[0].row());

			if (QString::number(saved_pose.x).length() < 7) {
				stream << saved_pose.x << ",\t\t";
			}
			else {
				stream << saved_pose.x << ",\t";
			}
			if (QString::number(saved_pose.y).length() < 7) {
				stream << saved_pose.y << ",\t\t";
			}
			else {
				stream << saved_pose.y << ",\t";
			}
			if (QString::number(saved_pose.z).length() < 7) {
				stream << saved_pose.z << ",\t\t";
			}
			else {
				stream << saved_pose.z << ",\t";
			}
			if (QString::number(saved_pose.za).length() < 7) {
				stream << saved_pose.za << ",\t\t";
			}
			else {
				stream << saved_pose.za << ",\t";
			}
			if (QString::number(saved_pose.xa).length() < 7) {
				stream << saved_pose.xa << ",\t\t";
			}
			else {
				stream << saved_pose.xa << ",\t";
			}
			stream << saved_pose.ya << ",\n";
		}
	}
}

/*Load Pose*/
void MainScreen::on_actionLoad_Pose_triggered() {
	//Load Pose
	//Selection Check
	/*Load Models Selected Indices*/
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	if (ui.image_list_widget->currentRow() < 0 || selected.size() == 0) {
		QMessageBox::critical(this, "Error!", "Select Frame and Model First!", QMessageBox::Ok);
		return;
	}

	//Must be in Single Selection Mode to Load Pose
	if (ui.multiple_model_radio_button->isChecked()) {
		QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Load Pose!", QMessageBox::Ok);
		return;
	}

	//Load File Dialog
	QString LoadPoseExtension = QFileDialog::getOpenFileName(this, tr("Load Pose"), ".",
	                                                         tr(
		                                                         "JTA Pose File (*.jtap);; JointTrack Pose File (*.jtp);; Pose File (*.txt);;"));
	QFile inputFile(LoadPoseExtension);
	QFileInfo inputFileInfo(inputFile);
	if (inputFile.open(QIODevice::ReadOnly)) {
		QTextStream in(&inputFile);
		QStringList InputList = in.readAll().split(QRegExp("[\r\n]"), QString::SkipEmptyParts);
		if (InputList.size() == 0) {
			QMessageBox::critical(this, "Error!", "Invalid Pose File!", QMessageBox::Ok);
			inputFile.close();
			return;
		}
		if (inputFileInfo.suffix() == "jtp") {
			QStringList LineList = InputList[0].split(QRegExp("[,]"), QString::SkipEmptyParts);
			if (LineList.size() >= 6) {
				LineList[0].replace(" ", "");
				if (LineList[0] == "NOT_OPTIMIZED") {
					QMessageBox::critical(this, "Error!", "No Pose Exists!", QMessageBox::Ok);
					inputFile.close();
					return;
				}
				auto loaded_pose = Point6D(LineList[0].toDouble(), LineList[1].toDouble(), LineList[2].toDouble(),
				                           LineList[4].toDouble(), LineList[5].toDouble(), LineList[3].toDouble());
				model_locations_.SavePose(ui.image_list_widget->currentRow(), selected[0].row(), loaded_pose);
				vw->set_model_position_at_index(selected[0].row(), loaded_pose.x, loaded_pose.y, loaded_pose.z);
				vw->set_model_orientation_at_index(selected[0].row(), loaded_pose.xa, loaded_pose.ya, loaded_pose.za);
				ui.qvtk_widget->update();
				ui.qvtk_widget->renderWindow()->Render();
			}
			else {
				QMessageBox::critical(this, "Error!", "Invalid Pose!", QMessageBox::Ok);
				inputFile.close();
				return;
			}
		}
		else {
			if (InputList[0] == "JTA_EULER_POSE") {
				QStringList LineList = InputList[2].split(QRegExp("[,]"), QString::SkipEmptyParts);
				if (LineList.size() >= 6) {
					LineList[0].replace(" ", "");
					if (LineList[0] == "NOT_OPTIMIZED") {
						QMessageBox::critical(this, "Error!", "No Pose Exists!", QMessageBox::Ok);
						inputFile.close();
						return;
					}
					auto loaded_pose = Point6D(LineList[0].toDouble(), LineList[1].toDouble(), LineList[2].toDouble(),
					                           LineList[4].toDouble(), LineList[5].toDouble(), LineList[3].toDouble());
					model_locations_.SavePose(ui.image_list_widget->currentRow(), selected[0].row(), loaded_pose);
					vw->set_model_position_at_index(selected[0].row(), loaded_pose.x, loaded_pose.y, loaded_pose.z);
					vw->set_model_orientation_at_index(selected[0].row(), loaded_pose.xa, loaded_pose.ya, loaded_pose.za);
					ui.qvtk_widget->update();
					ui.qvtk_widget->renderWindow()->Render();
				}
				else {
					QMessageBox::critical(this, "Error!", "Invalid Pose!", QMessageBox::Ok);
					inputFile.close();
					return;
				}
			}
			else {
				QMessageBox::critical(this, "Error!", "Invalid Pose File!", QMessageBox::Ok);
				inputFile.close();
				return;
			}
		}

		inputFile.close();
	}
}

/*Copy Previous Pose*/
void MainScreen::on_actionCopy_Previous_Pose_triggered() {
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	if (ui.image_list_widget->currentRow() < 0 || selected.size() == 0) {
		QMessageBox::critical(this, "Error!", "Select Model and Load Frames First!", QMessageBox::Ok);
		return;
	}

	if (ui.multiple_model_radio_button->isChecked()) {
		QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Load Kinematics!",
		                      QMessageBox::Ok);
		return;
	}
	Point6D prev_pose = model_locations_.GetPose(ui.image_list_widget->currentRow() - 1, selected[0].row());
	model_locations_.SavePose(ui.image_list_widget->currentRow(), ui.model_list_widget->currentRow(), prev_pose);
	vw->set_model_position_at_index(selected[0].row(), prev_pose.x, prev_pose.y, prev_pose.z);
	vw->set_model_orientation_at_index(selected[0].row(), prev_pose.xa, prev_pose.ya, prev_pose.za);
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();
}

// For passing current pose into sym_trap window
Point6D MainScreen::copy_current_pose() {
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	if (ui.image_list_widget->currentRow() < 0 || selected.size() == 0) {
		QMessageBox::critical(this, "Error!", "Select Model and Load Frames First!", QMessageBox::Ok);
		return Point6D();
	}

	if (ui.multiple_model_radio_button->isChecked()) {
		QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Load Kinematics!",
		                      QMessageBox::Ok);
		return Point6D();
	}
	Point6D pose = model_locations_.GetPose(ui.image_list_widget->currentRow(), selected[0].row());
	return pose;
}

/*Copy Next Pose*/

void MainScreen::on_actionCopy_Next_Pose_triggered() {
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	if (ui.image_list_widget->currentRow() < 0 || selected.size() == 0) {
		QMessageBox::critical(this, "Error!", "Select Model and Load Frames First!", QMessageBox::Ok);
		return;
	}

	if (ui.multiple_model_radio_button->isChecked()) {
		QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Load Kinematics!",
		                      QMessageBox::Ok);
		return;
	}
	Point6D next_pose = model_locations_.GetPose(ui.image_list_widget->currentRow() + 1, selected[0].row());
	model_locations_.SavePose(ui.image_list_widget->currentRow(), ui.model_list_widget->currentRow(), next_pose);

	vw->set_model_position_at_index(selected[0].row(), next_pose.x, next_pose.y, next_pose.z);
	vw->set_model_orientation_at_index(selected[0].row(), next_pose.xa, next_pose.ya, next_pose.za);
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();

}

/*Load Kinematics*/
void MainScreen::on_actionLoad_Kinematics_triggered() {
	//Load Kinematics to Frames
	//Selection Check
	/*Load Models Selected Indices*/
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	if (ui.image_list_widget->currentRow() < 0 || selected.size() == 0) {
		QMessageBox::critical(this, "Error!", "Select Model and Load Frames First!", QMessageBox::Ok);
		return;
	}

	//Must be in Single Selection Mode to Load Pose
	if (ui.multiple_model_radio_button->isChecked()) {
		QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Load Kinematics!",
		                      QMessageBox::Ok);
		return;
	}

	//Load Frame Dialog
	QString LoadPoseExtension = QFileDialog::getOpenFileName(this, tr("Load Kinematics"), ".",
	                                                         tr(
		                                                         "JTA Kinematics File (*.jtak);; JointTrack Kinematics File (*.jts);; Kinematics File (*.txt)"));
	QFile inputFile(LoadPoseExtension);
	if (inputFile.open(QIODevice::ReadOnly)) {
		QTextStream in(&inputFile);
		QStringList InputList = in.readAll().split(QRegExp("[\r\n]"), QString::SkipEmptyParts);
		if (InputList.size() == 0) {
			QMessageBox::critical(this, "Error!", "Invalid Kinematics File!", QMessageBox::Ok);
			inputFile.close();
			return;
		}
		if (InputList[0] == "JTA_EULER_KINEMATICS" || InputList[0] == "JT_EULER_312") {
			for (int i = 2; i < InputList.length() && (i - 2) < ui.image_list_widget->count(); i++) {
				QStringList LineList = InputList[i].split(QRegExp("[,]"), QString::SkipEmptyParts);
				if (LineList.size() >= 6) {
					LineList[0].replace(" ", "");
					if (LineList[0] != "NOT_OPTIMIZED") {
						auto loaded_pose = Point6D(LineList[0].toDouble(), LineList[1].toDouble(),
						                           LineList[2].toDouble(),
						                           LineList[4].toDouble(), LineList[5].toDouble(),
						                           LineList[3].toDouble());
						model_locations_.SavePose(i - 2, ui.model_list_widget->currentRow(), loaded_pose);
					}
				}
			}
			if (ui.image_list_widget->currentRow() >= 0) {
				Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentRow(), selected[0].row());
				vw->set_model_position_at_index(selected[0].row(), loaded_pose.x, loaded_pose.y, loaded_pose.z);
				vw->set_model_orientation_at_index(selected[0].row(), loaded_pose.xa, loaded_pose.ya, loaded_pose.za);
				ui.qvtk_widget->update();
				ui.qvtk_widget->renderWindow()->Render();
			}
		}
		else {
			QMessageBox::critical(this, "Error!", "Invalid Kinematics File!", QMessageBox::Ok);
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
	if (ui.actionStop_Optimizer->isEnabled() == true) {
		emit StopOptimizer();
		QMessageBox::warning(this, "Warning!", "Optimizer stopped!", QMessageBox::Ok);
	}
}

/*View Menu*/
void MainScreen::on_actionReset_View_triggered() {
	/*Reset to Model Interaction Mode*/
	if (ui.actionModel_Interaction_Mode->isChecked()) {
		renderer->GetActiveCamera()->SetViewUp(0, 1, 0);
		renderer->GetActiveCamera()->SetPosition(0, 0, 0);
		renderer->GetActiveCamera()->SetFocalPoint(0, 0,
		                                           -1 * calibration_file_.camera_A_principal_.principal_distance_ /
		                                           calibration_file_.camera_A_principal_.pixel_pitch_);
		renderer->GetActiveCamera()->SetClippingRange(
			.1, 2.0 * calibration_file_.camera_A_principal_.principal_distance_ / calibration_file_.camera_A_principal_.
			pixel_pitch_);
		if (loaded_frames.size() > 0) {
			renderer->GetActiveCamera()->SetViewAngle(CalculateViewingAngle(
				loaded_frames[ui.image_list_widget->currentIndex().row()].GetOriginalImage().cols,
				loaded_frames[ui.image_list_widget->currentIndex().row()].GetOriginalImage().rows,
				true));
		}
		ui.qvtk_widget->renderWindow()->GetInteractor()->SetInteractorStyle(key_press_vtk);
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
	else {
		/*Rest to Camera Interaction Mode*/
		renderer->GetActiveCamera()->SetViewUp(0, 1, 0);
		renderer->GetActiveCamera()->SetPosition(0, 0, 0);
		renderer->GetActiveCamera()->SetFocalPoint(0, 0,
		                                           -1 * calibration_file_.camera_A_principal_.principal_distance_ /
		                                           calibration_file_.camera_A_principal_.pixel_pitch_);
		renderer->GetActiveCamera()->SetClippingRange(
			.1, 2.0 * calibration_file_.camera_A_principal_.principal_distance_ / calibration_file_.camera_A_principal_.
			pixel_pitch_);
		if (loaded_frames.size() > 0) {
			renderer->GetActiveCamera()->SetViewAngle(CalculateViewingAngle(
				loaded_frames[ui.image_list_widget->currentIndex().row()].GetOriginalImage().cols,
				loaded_frames[ui.image_list_widget->currentIndex().row()].GetOriginalImage().rows,
				true));
		}
		QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
		if (selected.size() > 0) {
			renderer->GetActiveCamera()->SetFocalPoint(0, 0, model_actor_list[selected[0].row()]->GetPosition()[2]);
		}
		else {
			renderer->GetActiveCamera()->SetFocalPoint(0, 0,
			                                           -1 * calibration_file_.camera_A_principal_.principal_distance_ /
			                                           calibration_file_.camera_A_principal_.pixel_pitch_);
		}
		ui.qvtk_widget->renderWindow()->GetInteractor()->SetInteractorStyle(camera_style_interactor);
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
};

void MainScreen::on_actionReset_Normal_Up_triggered() {
	renderer->GetActiveCamera()->SetViewUp(0, 1, 0);
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();
}

void MainScreen::on_actionModel_Interaction_Mode_triggered() {
	if (loaded_models.size() == 0 || loaded_frames.size() == 0) {
		QMessageBox::critical(this, "Error!",
		                      "Please load at least one model and one image before changing the interaction mode!",
		                      QMessageBox::Ok);
		ui.actionModel_Interaction_Mode->setChecked(true);
		return;
	}
	ui.qvtk_widget->renderWindow()->GetInteractor()->SetInteractorStyle(key_press_vtk);
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();
};


void MainScreen::on_actionCamera_Interaction_Mode_triggered() {
	if (loaded_models.size() == 0 || loaded_frames.size() == 0) {
		QMessageBox::critical(this, "Error!",
		                      "Please load at least one model and one image before changing the interaction mode!",
		                      QMessageBox::Ok);
		ui.actionModel_Interaction_Mode->setChecked(true);
		return;
	}
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	if (selected.size() > 0) {
		renderer->GetActiveCamera()->SetFocalPoint(0, 0, model_actor_list[selected[0].row()]->GetPosition()[2]);
	}
	else {
		renderer->GetActiveCamera()->SetFocalPoint(0, 0,
		                                           -1 * calibration_file_.camera_A_principal_.principal_distance_ /
		                                           calibration_file_.camera_A_principal_.pixel_pitch_);
	}
	ui.qvtk_widget->renderWindow()->GetInteractor()->SetInteractorStyle(camera_style_interactor);
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();
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
	NOTE: Because this is a traced model, it can only be used with a batch size of 1. To work around this, one must
	convert to Torch Script via Annotation.*/
	/* std::string pt_model_location = "C:/TorchScriptTrainedNetworks/HRNETSeg_BS6_LIMA1024Actual_070519_2_TORCH_SCRIPT.pt"; */
	QString pt_model_location = QFileDialog::getOpenFileName(this, tr("Load Trained Femoral Segmentation Architecture"),
	                                                         ".", tr("Torch File (*.pt)"));
	if (pt_model_location.toStdString() != "") {
		segmentHelperFunction(pt_model_location.toStdString(), 1024, 1024);
	}
}

void MainScreen::on_actionSegment_TibHR_triggered() {
	/*Deserialize the ScriptModule from a file using torch::jit::load().
	NOTE: Because this is a traced model, it can only be used with a batch size of 1. To work around this, one must
	convert to Torch Script via Annotation.*/

	/*Removed the code below to allow for the user to pick and choose which network they want to load instead of using hard-coded paths*/
	/* std::string pt_model_location = "C:/TorchScriptTrainedNetworks/HRNET_BS8_LIMA1024TibActual_070619_BS8_LIMA1024TibActual_070619_TORCH_SCRIPT.pt";*/
	QString pt_model_location = QFileDialog::getOpenFileName(this, tr("Load Trained Tibial Segmentation Architecture"),
	                                                         ".", tr("Torch File (*.pt)"));
	if (pt_model_location.toStdString() != "") {
		segmentHelperFunction(pt_model_location.toStdString(), 1024, 1024);
	}
}

void MainScreen::segmentHelperFunction(std::string pt_model_location, unsigned int input_width,
                                       unsigned int input_height) {
	// std::shared_ptr<torch::jit::script::Module> model(torch::jit::load(pt_model_location, torch::kCUDA));
	// Commented out above part to resolve error E0289+
	//   std::shared_ptr<torch::jit::Module> model(torch::jit::load(pt_model_location, torch::kCUDA));
	torch::jit::Module module(torch::jit::load(pt_model_location, torch::kCUDA));
	torch::jit::Module* model = &module; // would this work as a pointer
	/*object shared_ptr cannot be converted from _T to torch::jit::Module */
	/* typecasting might be a solution */

	/* try
	{
		model = torch::jit::load(pt_model_location, torch::kCUDA);
	} */
	/* catch (const c10::Error& e) {
		QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
	}*/
	if (model == nullptr) {
		QMessageBox::critical(this, "Error!",
		                      QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location),
		                      QMessageBox::Ok);
		return;
	}
	// Commented out the above to follow the general torchscript page for flagging an error

	ui.pose_progress->setValue(20);
	ui.pose_label->setText("Segmenting images...");
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();

	/*Send Each Image to GPU Tensor, Segment Via Model, Replace Inverted Image*/
	//torch::Tensor gpu_byte_placeholder(torch::zeros({ 1, 1, input_height, input_width }, torch::device(torch::kCUDA).dtype(torch::kByte)));
	bool black_sil_used = ui.actionBlack_Implant_Silhouettes_in_Original_Image_s->isChecked();
	for (int i = 0; i < ui.image_list_widget->count(); i++) {
		//cv::Mat correct_inversion = (255 * black_sil_used) + ((1 - 2 * black_sil_used) * loaded_frames[i].GetOriginalImage());
		//cv::Mat padded;
		//if (correct_inversion.cols > correct_inversion.rows)
		//	padded.create(correct_inversion.cols, correct_inversion.cols, correct_inversion.type());
		//else
		//	padded.create(correct_inversion.rows, correct_inversion.rows, correct_inversion.type());
		//unsigned int padded_width = padded.cols;
		//unsigned int padded_height = padded.rows;
		//padded.setTo(cv::Scalar::all(0));
		//correct_inversion.copyTo(padded(cv::Rect(0, 0, correct_inversion.cols, correct_inversion.rows)));
		//cv::resize(padded, padded, cv::Size(input_width, input_height));
		//cudaMemcpy(gpu_byte_placeholder.data_ptr(), padded.data,
		//	input_width * input_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
		//std::vector<torch::jit::IValue> inputs;
		//inputs.push_back(gpu_byte_placeholder.to(torch::dtype(torch::kFloat)).flip({ 2 })); // Must flip first
		//cudaMemcpy(padded.data, (255 * (model->forward(inputs).toTensor() > 0)).to(torch::dtype(torch::kByte)).flip({ 2 }).data_ptr(),
		//	input_width * input_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		//cv::resize(padded, padded, cv::Size(padded_width, padded_height));
		//cv::Mat unpadded = padded(cv::Rect(0, 0, correct_inversion.cols, correct_inversion.rows));

		cv::Mat unpadded = segment_image(loaded_frames[i].GetOriginalImage(), black_sil_used, model, input_width,
		                                 input_height);
		unpadded.copyTo(loaded_frames[i].GetInvertedImage());
		int dilation_val = 0;
		trunk_manager_.getActiveCostFunctionClass()->getIntParameterValue("Dilation", dilation_val);
		loaded_frames[i].SetEdgeImage(ui.aperture_spin_box->value(),
		                              ui.low_threshold_slider->value(),
		                              ui.high_threshold_slider->value(), true);
		loaded_frames[i].SetDilatedImage(dilation_val);
		if (calibrated_for_biplane_viewport_) {
			cv::Mat unpadded_biplane = segment_image(loaded_frames_B[i].GetOriginalImage(), black_sil_used, model,
			                                         input_width, input_height);
			unpadded_biplane.copyTo(loaded_frames_B[i].GetInvertedImage());
			loaded_frames_B[i].SetEdgeImage(ui.aperture_spin_box->value(),
			                                ui.low_threshold_slider->value(),
			                                ui.high_threshold_slider->value(), true);
			loaded_frames_B[i].SetDilatedImage(dilation_val);
		}

		ui.pose_progress->setValue(
			20 + 30 * static_cast<double>(i + 1) / static_cast<double>(ui.image_list_widget->count()));
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();


	}

	/*If Viewing Inverted Images Update*/
	if (ui.image_list_widget->currentIndex().row() >= 0 && ui.inverted_image_radio_button->isChecked() == true) {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_inverted_image(this->curr_frame(), true);

		}
		else {
			vw->update_display_background_to_inverted_image(this->curr_frame(), false);
		}
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
	/*If Viewing Edge Images Update*/
	else if (ui.image_list_widget->currentIndex().row() >= 0 && ui.edges_image_radio_button->isChecked() == true) {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_edge_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_edge_image(this->curr_frame(), false);
		}
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
	/*If Viewing Dilated Images Update*/
	else if (ui.image_list_widget->currentIndex().row() >= 0 && ui.dilation_image_radio_button->isChecked() == true) {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_dilation_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_dilation_image(this->curr_frame(), false);
		}
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
}

void MainScreen::on_actionReset_Remove_All_Segmentation_triggered() {
	for (int i = 0; i < ui.image_list_widget->count(); i++) {
		loaded_frames[i].ResetFromOriginal();
	}

	/*If Viewing Inverted Images Update*/
	if (ui.image_list_widget->currentIndex().row() >= 0 && ui.inverted_image_radio_button->isChecked() == true) {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_inverted_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_inverted_image(this->curr_frame(), false);
		}
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
	/*If Viewing Edge Images Update*/
	else if (ui.image_list_widget->currentIndex().row() >= 0 && ui.edges_image_radio_button->isChecked() == true) {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_edge_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_inverted_image(this->curr_frame(), false);
		}
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
	/*If Viewing Dilated Images Update*/
	else if (ui.image_list_widget->currentIndex().row() >= 0 && ui.dilation_image_radio_button->isChecked() == true) {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_dilation_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_dilation_image(this->curr_frame(), false);
		}
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
}

void MainScreen::on_actionEstimate_Femoral_Implant_s_triggered() {
	//Must be in Single Selection Mode to Load Pose
	if (ui.multiple_model_radio_button->isChecked()) {
		QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Estimate Kinematics!",
		                      QMessageBox::Ok);
		return;
	}

	//Must load a model
	if (loaded_models.size() < 1) {
		QMessageBox::critical(this, "Error!", "Must load a model!", QMessageBox::Ok);
		return;
	}

	//Must have loaded image
	if (loaded_frames.size() < 1) {
		QMessageBox::critical(this, "Error!", "Must load images!", QMessageBox::Ok);
		return;
	}

	/*Pose Estimate Progress and Label Visible*/
	ui.pose_progress->setValue(5);
	ui.pose_progress->setVisible(true);
	ui.pose_label->setText("Initializing high resolution segmentation...");
	ui.pose_label->setVisible(true);
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();
	qApp->processEvents();


	/*Segment*/
	this->on_actionSegment_FemHR_triggered();
	unsigned int input_height = 1024;
	unsigned int input_width = 1024;
	unsigned int orig_height = loaded_frames[0].GetInvertedImage().rows;
	unsigned int orig_width = loaded_frames[0].GetInvertedImage().cols;
	auto host_image = static_cast<unsigned char*>(malloc(input_width * input_height * sizeof(unsigned char)));
	ui.pose_label->setText("Initializing STL model on GPU...");
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();

	/*STL Information*/
	vector<vector<float>> triangle_information;
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	stl_reader_BIG::readAnySTL(QString::fromStdString(loaded_models[selected[0].row()].file_location_),
	                           triangle_information);

	/*GPU Models for the current Model*/
	auto gpu_mod = new GPUModel("model", true, orig_height, orig_width, 0, false,
	                            // switched cols and rows because the stored image is inverted?
	                            &(triangle_information[0])[0], &(triangle_information[1])[0],
	                            triangle_information[0].size() / 9, calibration_file_.camera_A_principal_);
	// BACKFACE CULLING APPEARS TO BE GIVING ERRORS

	ui.pose_progress->setValue(55);
	ui.pose_label->setText("Initializing femoral implant pose estimation...");
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();

	/*Load JIT Model*/
	/*std::string pt_model_location = "C:/TorchScriptTrainedNetworks/HRNETPR_BS6_dataLima1024_07192019_HRProcessed_Fem_07232019_1_TORCH_SCRIPT.pt"; */
	QString pt_model_location = QFileDialog::getOpenFileName(
		this, tr("Load Trained Femoral Pose Regression Architecture"), ".", tr("Torch File (*.pt)"));
	// std::shared_ptr<torch::jit::Module> model(torch::jit::load(pt_model_location.toStdString(), torch::kCUDA));
	torch::jit::Module module(torch::jit::load(pt_model_location.toStdString(), torch::kCUDA));
	torch::jit::Module* model = &module;
	if (model == nullptr) {
		// QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
		QMessageBox::critical(this, "Error!",
		                      QString("Cannot load PyTorch Torch Script model at: " + pt_model_location),
		                      QMessageBox::Ok);
		return;
	}

	/*Send Each Segmented Image to GPU Tensor, Predict Orientation, Then Z (From Area), then X,Y.
	After this, convert to non (0,0) centered orientation.
	Finally, update */
	ui.pose_progress->setValue(65);
	ui.pose_label->setText("Estimating femoral implant poses...");
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();
	auto orientation = new float[3];
	torch::Tensor gpu_byte_placeholder(torch::zeros({1, 1, input_height, input_width},
	                                                device(torch::kCUDA).dtype(torch::kByte)));
	for (int i = 0; i < ui.image_list_widget->count(); i++) {

		cv::Mat orig_inverted = loaded_frames[i].GetInvertedImage();
		cv::Mat padded;
		if (orig_inverted.cols > orig_inverted.rows) {
			padded.create(orig_inverted.cols, orig_inverted.cols, orig_inverted.type());
		}
		else {
			padded.create(orig_inverted.rows, orig_inverted.rows, orig_inverted.type());
		}
		unsigned int padded_width = padded.cols;
		unsigned int padded_height = padded.rows;
		padded.setTo(cv::Scalar::all(0));
		orig_inverted.copyTo(padded(cv::Rect(0, 0, orig_inverted.cols, orig_inverted.rows)));
		cv::resize(padded, padded, cv::Size(input_width, input_height));

		cudaMemcpy(gpu_byte_placeholder.data_ptr(), padded.data,
		           input_width * input_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(gpu_byte_placeholder.to(dtype(torch::kFloat)).flip({2})); // Must flip first
		cudaMemcpy(orientation, model->forward(inputs).toTensor().to(dtype(torch::kFloat)).data_ptr(),
		           3 * sizeof(float), cudaMemcpyDeviceToHost);
		/*Flip Segment*/
		auto output_mat_seg = cv::Mat(orig_inverted.rows, orig_inverted.cols, CV_8UC1);
		flip(orig_inverted, output_mat_seg, 0);

		/*Render*/
		gpu_mod->RenderPrimaryCamera(Pose(0, 0, -calibration_file_.camera_A_principal_.principal_distance_,
		                                  orientation[1], orientation[2], orientation[0]));

		/*Copy To Mat*/
		cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(),
		           orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		/*OpenCV Image Container/Write Function*/
		auto projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image); /*Reverse before flip*/
		auto output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
		flip(projection_mat, output_mat, 0);

		/*Get Scale*/
		double sum_seg = sum(sum(output_mat_seg))[0] / 255.0;
		double sum_proj = sum(sum(output_mat))[0] / 255.0;
		double z;
		/* Creating A check to ensure that the z translation is not greater than the principal distance */
		if (sum_proj / sum_seg > 1) {
			z = -calibration_file_.camera_A_principal_.principal_distance_;
		}
		else {
			z = -calibration_file_.camera_A_principal_.principal_distance_ * sqrt(sum_proj / sum_seg);
		}


		/*Reproject*/
		/*Render*/
		gpu_mod->RenderPrimaryCamera(Pose(0, 0, z, orientation[1], orientation[2], orientation[0]));
		cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(),
		           orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image);
		output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
		flip(projection_mat, output_mat, 0);

		/*cv::imwrite("C:/Users/pflood/Desktop/output_mat.png", output_mat);
		cv::imwrite("C:/Users/pflood/Desktop/output_mat_seg.png", output_mat_seg);*/

		/*Get X and Y*/
		cv::Mat proj64;
		output_mat.convertTo(proj64, CV_64FC1);
		cv::Mat seg64;
		output_mat_seg.convertTo(seg64, CV_64FC1);
		cv::Point2d x_y_point = phaseCorrelate(proj64, seg64) * (calibration_file_.camera_A_principal_.pixel_pitch_ * z
				* -1) /
			calibration_file_.camera_A_principal_.principal_distance_;
		double x = x_y_point.x;
		double y = -1 * x_y_point.y;

		//QMessageBox::critical(this, "Error!", QString::number(x) + ", " +
		//	QString::number(y) + ", " + 
		//	QString::number(z) + ", " + 
		//	QString::number(orientation[1]) + ", " +
		//	QString::number(orientation[2]) + ", " +
		//	QString::number(orientation[0]), QMessageBox::Ok);

		/*Convert from (0,0) Centered*/
		float za_rad = orientation[0] * 3.14159265358979323846 / 180.0;
		float xa_rad = orientation[1] * 3.14159265358979323846 / 180.0;
		float ya_rad = orientation[2] * 3.14159265358979323846 / 180.0;
		float cz = cos(za_rad);
		float sz = sin(za_rad);
		float cx = cos(xa_rad);
		float sx = sin(xa_rad);
		float cy = cos(ya_rad);
		float sy = sin(ya_rad);
		Matrix_3_3 R_g(
			cz * cy - sz * sx * sy, -1.0 * sz * cx, cz * sy + sz * cy * sx,
			sz * cy + cz * sx * sy, cz * cx, sz * sy - cz * cy * sx,
			-1.0 * cx * sy, sx, cx * cy);
		float theta_x = std::atan(-1.0 * y / z);
		float theta_y = std::asin(-1.0 * x / std::sqrt(x * x + y * y + z * z));
		Matrix_3_3 R_x(
			1, 0, 0,
			0, cos(theta_x), -sin(theta_x),
			0, sin(theta_x), cos(theta_x));
		Matrix_3_3 R_y(
			cos(theta_y), 0, sin(theta_y),
			0, 1, 0,
			-sin(theta_y), 0, cos(theta_y));
		Matrix_3_3 R_orig = calibration_file_.multiplication_mat_mat(
			R_y, calibration_file_.multiplication_mat_mat(R_x, R_g));
		/*Rot Mat To Eul ZXY*/
		/*Algorithm To Recover Z - X - Y Euler Angles*/
		float xa, ya, za;
		if (R_orig.A_32_ < 1) {
			if (R_orig.A_32_ > -1) {
				xa = asin(R_orig.A_32_);
				za = atan2(-1 * R_orig.A_12_, R_orig.A_22_);
				ya = atan2(-1 * R_orig.A_31_, R_orig.A_33_);

			}
			else {
				xa = -3.14159265358979323846 / 2.0;
				za = -1 * atan2(R_orig.A_13_, R_orig.A_11_);
				ya = 0;
			}
		}
		else {
			xa = 3.14159265358979323846 / 2.0;
			za = atan2(R_orig.A_13_, R_orig.A_11_);
			ya = 0;
		}

		xa = xa * 180.0 / 3.14159265358979323846;
		ya = ya * 180.0 / 3.14159265358979323846;
		za = za * 180.0 / 3.14159265358979323846;
		/*
				QMessageBox::critical(this, "Error!", QString::number(x) + ", " +
					QString::number(y) + ", " +
					QString::number(z) + ", " +
					QString::number(xa) + ", " +
					QString::number(ya) + ", " +
					QString::number(za), QMessageBox::Ok);*/
		/*Update Model Pose*/
		model_locations_.SavePose(i, ui.model_list_widget->currentRow(), Point6D(x, y, z, xa, ya, za));
		ui.pose_progress->setValue(
			65 + 30 * static_cast<double>(i + 1) / static_cast<double>(ui.image_list_widget->count()));
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		qApp->processEvents();
	}

	ui.pose_progress->setValue(98);
	ui.pose_label->setText("Deleting old models...");
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();

	/*Delete GPU Model*/
	delete gpu_mod;

	/*Free Array*/
	free(host_image);

	/*Update Model*/
	Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentRow(), selected[0].row());
	model_actor_list[selected[0].row()]->SetPosition(loaded_pose.x, loaded_pose.y, loaded_pose.z);
	model_actor_list[selected[0].row()]->SetOrientation(loaded_pose.xa, loaded_pose.ya, loaded_pose.za);
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();

	ui.pose_progress->setValue(100);
	ui.pose_label->setText("Finished!");
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();

	/*Pose Estimate Progress and Label Not Visible*/
	ui.pose_progress->setVisible(false);
	ui.pose_label->setVisible(false);

}

//void MainScreen::on_actionEstimate_Femoral_Implant_s_Algorithm_2_triggered() {
////Must be in Single Selection Mode to Load Pose
//if (ui.multiple_model_radio_button->isChecked()) {
//QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Estimate Kinematics!", QMessageBox::Ok);
//return;
//}

////Must load a model
//if (loaded_models.size() < 1) {
//QMessageBox::critical(this, "Error!", "Must load a model!", QMessageBox::Ok);
//return;
//}

////Must have loaded image
//if (loaded_frames.size() < 1) {
//QMessageBox::critical(this, "Error!", "Must load images!", QMessageBox::Ok);
//return;
//}

///*Pose Estimate Progress and Label Visible*/
//ui.pose_progress->setValue(5);
//ui.pose_progress->setVisible(1);
//ui.pose_label->setText("Initializing high resolution segmentation...");
//ui.pose_label->setVisible(1);
//ui.qvtk_widget->update();
//qApp->processEvents();


///*Segment*/
//this->on_actionSegment_FemHR_triggered();
//unsigned int input_height = 1024;
//unsigned int input_width = 1024;
//unsigned int orig_height = loaded_frames[0].GetInvertedImage().rows;
//unsigned int orig_width = loaded_frames[0].GetInvertedImage().cols;
//unsigned char* host_image = (unsigned char*)malloc(input_width * input_height * sizeof(unsigned char));
//ui.pose_label->setText("Initializing STL model on GPU...");
//ui.qvtk_widget->update();

///*STL Information*/
//vector<vector<float>> triangle_information;
//QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
//stl_reader_BIG::readAnySTL(QString::fromStdString(loaded_models[selected[0].row()].file_location_), triangle_information);

///*GPU Models for the current Model*/
//gpu_cost_function::GPUModel* gpu_mod = new gpu_cost_function::GPUModel("model", true, orig_height, orig_width, 0, false, // switched cols and rows because the stored image is inverted?
//&(triangle_information[0])[0], &(triangle_information[1])[0], triangle_information[0].size() / 9, calibration_file_.camera_A_principal_); // BACKFACE CULLING APPEARS TO BE GIVING ERRORS

//ui.pose_progress->setValue(55);
//ui.pose_label->setText("Initializing femoral implant pose estimation...");
//ui.qvtk_widget->update();

///*Load JIT Model*/
//std::string pt_model_location = "C:/TorchScriptTrainedNetworks/HRNETPR_BS6_dataLima1024_07192019_HRProcessed_Fem_07232019_1_TORCH_SCRIPT.pt";
//// std::shared_ptr<torch::jit::Module> model(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module module(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module* model = &module;
//if (model == nullptr)
//{
//QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
//return;
//}

///*Load JIT Z Model*/
//std::string pt_model_z_location = "C:/TorchScriptTrainedNetworks/HRNETPR_BS6_dataLima1024_07192019_HRProcessed_Fem_08012019_Z_1_TORCH_SCRIPT.pt";
//// std::shared_ptr<torch::jit::Module> model_z(torch::jit::load(pt_model_z_location, torch::kCUDA));
//torch::jit::Module module_z(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module* model_z = &module_z;
//if (model_z == nullptr)
//{
//QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
//return;
//}

///*Send Each Segmented Image to GPU Tensor, Predict Orientation, Then Z (From Area), then X,Y.
//After this, convert to non (0,0) centered orientation.
//Finally, update */
//ui.pose_progress->setValue(65);
//ui.pose_label->setText("Estimating femoral implant poses...");
//ui.qvtk_widget->update();
//float* orientation = new float[3];
//float* z_norm = new float[1];
//torch::Tensor gpu_byte_placeholder(torch::zeros({ 1, 1, input_height, input_width }, torch::device(torch::kCUDA).dtype(torch::kByte)));
//for (int i = 0; i < ui.image_list_widget->count(); i++) {

//cv::Mat orig_inverted = loaded_frames[i].GetInvertedImage();
//cv::Mat padded;
//if (orig_inverted.cols > orig_inverted.rows)
//padded.create(orig_inverted.cols, orig_inverted.cols, orig_inverted.type());
//else
//padded.create(orig_inverted.rows, orig_inverted.rows, orig_inverted.type());
//unsigned int padded_width = padded.cols;
//unsigned int padded_height = padded.rows;
//padded.setTo(cv::Scalar::all(0));
//orig_inverted.copyTo(padded(cv::Rect(0, 0, orig_inverted.cols, orig_inverted.rows)));
//cv::resize(padded, padded, cv::Size(input_width, input_height));

//cudaMemcpy(gpu_byte_placeholder.data_ptr(), padded.data,
//input_width * input_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
//std::vector<torch::jit::IValue> inputs;
//inputs.push_back(gpu_byte_placeholder.to(torch::dtype(torch::kFloat)).flip({ 2 })); // Must flip first
//cudaMemcpy(orientation, model->forward(inputs).toTensor().to(torch::dtype(torch::kFloat)).data_ptr(),
//3 * sizeof(float), cudaMemcpyDeviceToHost);
//cudaMemcpy(z_norm, model_z->forward(inputs).toTensor().to(torch::dtype(torch::kFloat)).data_ptr(),
//sizeof(float), cudaMemcpyDeviceToHost);

///*Flip Segment*/
//cv::Mat output_mat_seg = cv::Mat(orig_inverted.rows, orig_inverted.cols, CV_8UC1);
//cv::flip(orig_inverted, output_mat_seg, 0);

///*Render*/
//gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(0, 0, -calibration_file_.camera_A_principal_.principal_distance_, orientation[1], orientation[2], orientation[0]));

///*Copy To Mat*/
//cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(), orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

///*OpenCV Image Container/Write Function*/
//cv::Mat projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image); /*Reverse before flip*/
//cv::Mat output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
//cv::flip(projection_mat, output_mat, 0);

///*Get Scale*/
//double z = -calibration_file_.camera_A_principal_.principal_distance_ * z_norm[0];

///*Reproject*/
///*Render*/
//gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(0, 0, z, orientation[1], orientation[2], orientation[0]));
//cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(), orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image);
//output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
//cv::flip(projection_mat, output_mat, 0);

///*Get X and Y*/
//cv::Mat proj64;
//output_mat.convertTo(proj64, CV_64FC1);
//cv::Mat seg64;
//output_mat_seg.convertTo(seg64, CV_64FC1);
//cv::Point2d x_y_point = cv::phaseCorrelate(proj64, seg64) * (calibration_file_.camera_A_principal_.pixel_pitch_ * z * -1) /
//calibration_file_.camera_A_principal_.principal_distance_;
//double x = x_y_point.x;
//double y = -1 * x_y_point.y;


///*Convert from (0,0) Centered*/
//float za_rad = orientation[0] * 3.14159265358979323846 / 180.0;
//float xa_rad = orientation[1] * 3.14159265358979323846 / 180.0;
//float ya_rad = orientation[2] * 3.14159265358979323846 / 180.0;
//float cz = cos(za_rad);
//float sz = sin(za_rad);
//float cx = cos(xa_rad);
//float sx = sin(xa_rad);
//float cy = cos(ya_rad);
//float sy = sin(ya_rad);
//Matrix_3_3 R_g(
//cz * cy - sz * sx * sy, -1.0 * sz * cx, cz * sy + sz * cy * sx,
//sz * cy + cz * sx * sy, cz * cx, sz * sy - cz * cy * sx,
//-1.0 * cx * sy, sx, cx * cy);
//float theta_x = std::atan(-1.0 * y / z);
//float theta_y = std::asin(-1.0 * x / std::sqrt(x * x + y * y + z * z));
//Matrix_3_3 R_x(
//1, 0, 0,
//0, cos(theta_x), -sin(theta_x),
//0, sin(theta_x), cos(theta_x));
//Matrix_3_3 R_y(
//cos(theta_y), 0, sin(theta_y),
//0, 1, 0,
//-sin(theta_y), 0, cos(theta_y));
//Matrix_3_3 R_orig = calibration_file_.multiplication_mat_mat(R_y, calibration_file_.multiplication_mat_mat(R_x, R_g));
///*Rot Mat To Eul ZXY*/
///*Algorithm To Recover Z - X - Y Euler Angles*/
//float xa, ya, za;
//if (R_orig.A_32_ < 1) {
//if (R_orig.A_32_ > -1) {
//xa = asin(R_orig.A_32_);
//za = atan2(-1 * R_orig.A_12_, R_orig.A_22_);
//ya = atan2(-1 * R_orig.A_31_, R_orig.A_33_);

//}
//else {
//xa = -3.14159265358979323846 / 2.0;
//za = -1 * atan2(R_orig.A_13_, R_orig.A_11_);
//ya = 0;
//}
//}
//else {
//xa = 3.14159265358979323846 / 2.0;
//za = atan2(R_orig.A_13_, R_orig.A_11_);
//ya = 0;
//}

//xa = xa * 180.0 / 3.14159265358979323846;
//ya = ya * 180.0 / 3.14159265358979323846;
//za = za * 180.0 / 3.14159265358979323846;

///*Update Model Pose*/
//model_locations_.SavePose(i, ui.model_list_widget->currentRow(), Point6D(x, y, z, xa, ya, za));

//ui.pose_progress->setValue(65 + 30 * (double)(i + 1) / (double)ui.image_list_widget->count());
//ui.qvtk_widget->update();
//qApp->processEvents();
//}

//ui.pose_progress->setValue(98);
//ui.pose_label->setText("Deleting old models...");
//ui.qvtk_widget->update();

///*Delete GPU Model*/
//delete gpu_mod;

///*Free Array*/
//free(host_image);

///*Free Values*/
//delete orientation;
//delete z_norm;

///*Update Model*/
//Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentRow(), selected[0].row());
//model_actor_list[selected[0].row()]->SetPosition(loaded_pose.x, loaded_pose.y, loaded_pose.z);
//model_actor_list[selected[0].row()]->SetOrientation(loaded_pose.xa, loaded_pose.ya, loaded_pose.za);
//ui.qvtk_widget->update();

//ui.pose_progress->setValue(100);
//ui.pose_label->setText("Finished!");
//ui.qvtk_widget->update();

///*Pose Estimate Progress and Label Not Visible*/
//ui.pose_progress->setVisible(0);
//ui.pose_label->setVisible(0);

//}
void MainScreen::on_actionEstimate_Tibial_Implant_s_triggered() {
	//Must be in Single Selection Mode to Load Pose
	if (ui.multiple_model_radio_button->isChecked()) {
		QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Estimate Kinematics!",
		                      QMessageBox::Ok);
		return;
	}

	//Must load a model
	if (loaded_models.size() < 1) {
		QMessageBox::critical(this, "Error!", "Must load a model!", QMessageBox::Ok);
		return;
	}

	//Must have loaded image
	if (loaded_frames.size() < 1) {
		QMessageBox::critical(this, "Error!", "Must load images!", QMessageBox::Ok);
		return;
	}

	/*Pose Estimate Progress and Label Visible*/
	ui.pose_progress->setValue(5);
	ui.pose_progress->setVisible(true);
	ui.pose_label->setText("Initializing high resolution segmentation...");
	ui.pose_label->setVisible(true);
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();
	qApp->processEvents();


	/*Segment*/
	this->on_actionSegment_TibHR_triggered();
	unsigned int input_height = 1024;
	unsigned int input_width = 1024;
	unsigned int orig_height = loaded_frames[0].GetInvertedImage().rows;
	unsigned int orig_width = loaded_frames[0].GetInvertedImage().cols;
	auto host_image = static_cast<unsigned char*>(malloc(input_width * input_height * sizeof(unsigned char)));
	ui.pose_label->setText("Initializing STL model on GPU...");
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();

	/*STL Information*/
	vector<vector<float>> triangle_information;
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	stl_reader_BIG::readAnySTL(QString::fromStdString(loaded_models[selected[0].row()].file_location_),
	                           triangle_information);

	/*GPU Models for the current Model*/
	auto gpu_mod = new GPUModel("model", true, orig_height, orig_width, 0, false,
	                            // switched cols and rows because the stored image is inverted?
	                            &(triangle_information[0])[0], &(triangle_information[1])[0],
	                            triangle_information[0].size() / 9, calibration_file_.camera_A_principal_);
	// BACKFACE CULLING APPEARS TO BE GIVING ERRORS

	ui.pose_progress->setValue(55);
	ui.pose_label->setText("Initializing tibial implant pose estimation...");
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();

	/*Load JIT Model*/
	/* Removing the below code to allow for the user to select the model they wish to use*/
	/*std::string pt_model_location = "C:/TorchScriptTrainedNetworks/HRNETPR_BS6_dataLimaTib1024_08012019_HRProcessed_Tib_08022019_1_TORCH_SCRIPT.pt";*/
	QString pt_model_location = QFileDialog::getOpenFileName(
		this, tr("Load Trained Tibial Pose Estimation Architecture"), ".", tr("Torch File (*.pt)"));
	// std::shared_ptr<torch::jit::Module> model(torch::jit::load(pt_model_location.toStdString(), torch::kCUDA));
	torch::jit::Module module(torch::jit::load(pt_model_location.toStdString(), torch::kCUDA));
	torch::jit::Module* model = &module;
	if (model == nullptr) {
		QMessageBox::critical(this, "Error!",
		                      QString::fromStdString(
			                      "Cannot load PyTorch Torch Script model at: " + pt_model_location.toStdString()),
		                      QMessageBox::Ok);
		return;
	}

	/*Send Each Segmented Image to GPU Tensor, Predict Orientation, Then Z (From Area), then X,Y.
	After this, convert to non (0,0) centered orientation.
	Finally, update */
	ui.pose_progress->setValue(65);
	ui.pose_label->setText("Estimating tibial implant poses...");
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();
	auto orientation = new float[3];
	torch::Tensor gpu_byte_placeholder(torch::zeros({1, 1, input_height, input_width},
	                                                device(torch::kCUDA).dtype(torch::kByte)));
	for (int i = 0; i < ui.image_list_widget->count(); i++) {

		cv::Mat orig_inverted = loaded_frames[i].GetInvertedImage();
		cv::Mat padded;
		if (orig_inverted.cols > orig_inverted.rows) {
			padded.create(orig_inverted.cols, orig_inverted.cols, orig_inverted.type());
		}
		else {
			padded.create(orig_inverted.rows, orig_inverted.rows, orig_inverted.type());
		}
		unsigned int padded_width = padded.cols;
		unsigned int padded_height = padded.rows;
		padded.setTo(cv::Scalar::all(0));
		orig_inverted.copyTo(padded(cv::Rect(0, 0, orig_inverted.cols, orig_inverted.rows)));
		cv::resize(padded, padded, cv::Size(input_width, input_height));

		cudaMemcpy(gpu_byte_placeholder.data_ptr(), padded.data,
		           input_width * input_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(gpu_byte_placeholder.to(dtype(torch::kFloat)).flip({2})); // Must flip first
		cudaMemcpy(orientation, model->forward(inputs).toTensor().to(dtype(torch::kFloat)).data_ptr(),
		           3 * sizeof(float), cudaMemcpyDeviceToHost);
		/*Flip Segment*/
		auto output_mat_seg = cv::Mat(orig_inverted.rows, orig_inverted.cols, CV_8UC1);
		flip(orig_inverted, output_mat_seg, 0);

		/*Render*/
		gpu_mod->RenderPrimaryCamera(Pose(0, 0, -calibration_file_.camera_A_principal_.principal_distance_,
		                                  orientation[1], orientation[2], orientation[0]));

		/*Copy To Mat*/
		cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(),
		           orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		/*OpenCV Image Container/Write Function*/
		auto projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image); /*Reverse before flip*/
		auto output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
		flip(projection_mat, output_mat, 0);

		/*Get Scale*/
		double sum_seg = sum(sum(output_mat_seg))[0] / 255.0;
		double sum_proj = sum(sum(output_mat))[0] / 255.0;
		double z = -calibration_file_.camera_A_principal_.principal_distance_ * sqrt(sum_proj / sum_seg);
		/*Reproject*/
		/*Render*/
		gpu_mod->RenderPrimaryCamera(Pose(0, 0, z, orientation[1], orientation[2], orientation[0]));
		cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(),
		           orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image);
		output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
		flip(projection_mat, output_mat, 0);

		/*Get X and Y*/
		cv::Mat proj64;
		output_mat.convertTo(proj64, CV_64FC1);
		cv::Mat seg64;
		output_mat_seg.convertTo(seg64, CV_64FC1);
		cv::Point2d x_y_point = phaseCorrelate(proj64, seg64) * (calibration_file_.camera_A_principal_.pixel_pitch_ * z
				* -1) /
			calibration_file_.camera_A_principal_.principal_distance_;
		double x = x_y_point.x;
		double y = -1 * x_y_point.y;

		/*Convert from (0,0) Centered*/
		float za_rad = orientation[0] * 3.14159265358979323846 / 180.0;
		float xa_rad = orientation[1] * 3.14159265358979323846 / 180.0;
		float ya_rad = orientation[2] * 3.14159265358979323846 / 180.0;
		float cz = cos(za_rad);
		float sz = sin(za_rad);
		float cx = cos(xa_rad);
		float sx = sin(xa_rad);
		float cy = cos(ya_rad);
		float sy = sin(ya_rad);
		Matrix_3_3 R_g(
			cz * cy - sz * sx * sy, -1.0 * sz * cx, cz * sy + sz * cy * sx,
			sz * cy + cz * sx * sy, cz * cx, sz * sy - cz * cy * sx,
			-1.0 * cx * sy, sx, cx * cy);
		float theta_x = std::atan(-1.0 * y / z);
		float theta_y = std::asin(-1.0 * x / std::sqrt(x * x + y * y + z * z));
		Matrix_3_3 R_x(
			1, 0, 0,
			0, cos(theta_x), -sin(theta_x),
			0, sin(theta_x), cos(theta_x));
		Matrix_3_3 R_y(
			cos(theta_y), 0, sin(theta_y),
			0, 1, 0,
			-sin(theta_y), 0, cos(theta_y));
		Matrix_3_3 R_orig = calibration_file_.multiplication_mat_mat(
			R_y, calibration_file_.multiplication_mat_mat(R_x, R_g));
		/*Rot Mat To Eul ZXY*/
		/*Algorithm To Recover Z - X - Y Euler Angles*/
		float xa, ya, za;
		if (R_orig.A_32_ < 1) {
			if (R_orig.A_32_ > -1) {
				xa = asin(R_orig.A_32_);
				za = atan2(-1 * R_orig.A_12_, R_orig.A_22_);
				ya = atan2(-1 * R_orig.A_31_, R_orig.A_33_);

			}
			else {
				xa = -3.14159265358979323846 / 2.0;
				za = -1 * atan2(R_orig.A_13_, R_orig.A_11_);
				ya = 0;
			}
		}
		else {
			xa = 3.14159265358979323846 / 2.0;
			za = atan2(R_orig.A_13_, R_orig.A_11_);
			ya = 0;
		}

		xa = xa * 180.0 / 3.14159265358979323846;
		ya = ya * 180.0 / 3.14159265358979323846;
		za = za * 180.0 / 3.14159265358979323846;

		model_locations_.SavePose(i, ui.model_list_widget->currentRow(), Point6D(x, y, z, xa, ya, za));
		ui.pose_progress->setValue(
			65 + 30 * static_cast<double>(i + 1) / static_cast<double>(ui.image_list_widget->count()));
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		qApp->processEvents();
	}

	ui.pose_progress->setValue(98);
	ui.pose_label->setText("Deleting old models...");
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();

	/*Delete GPU Model*/
	delete gpu_mod;

	/*Free Array*/
	free(host_image);

	/*Update Model*/
	Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentRow(), selected[0].row());
	model_actor_list[selected[0].row()]->SetPosition(loaded_pose.x, loaded_pose.y, loaded_pose.z);
	model_actor_list[selected[0].row()]->SetOrientation(loaded_pose.xa, loaded_pose.ya, loaded_pose.za);
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();

	ui.pose_progress->setValue(100);
	ui.pose_label->setText("Finished!");
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();

	/*Pose Estimate Progress and Label Not Visible*/
	ui.pose_progress->setVisible(false);
	ui.pose_label->setVisible(false);
}

//void MainScreen::on_actionEstimate_Tibial_Implant_s_Alternative_Algorithm_triggered() {
////Must be in Single Selection Mode to Load Pose
//if (ui.multiple_model_radio_button->isChecked()) {
//QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Estimate Kinematics!", QMessageBox::Ok);
//return;
//}

////Must load a model
//if (loaded_models.size() < 1) {
//QMessageBox::critical(this, "Error!", "Must load a model!", QMessageBox::Ok);
//return;
//}

////Must have loaded image
//if (loaded_frames.size() < 1) {
//QMessageBox::critical(this, "Error!", "Must load images!", QMessageBox::Ok);
//return;
//}

///*Pose Estimate Progress and Label Visible*/
//ui.pose_progress->setValue(5);
//ui.pose_progress->setVisible(1);
//ui.pose_label->setText("Initializing high resolution segmentation...");
//ui.pose_label->setVisible(1);
//ui.qvtk_widget->update();
//qApp->processEvents();


///*Segment*/
//this->on_actionSegment_TibHR_triggered();
//unsigned int input_height = 1024;
//unsigned int input_width = 1024;
//unsigned int orig_height = loaded_frames[0].GetInvertedImage().rows;
//unsigned int orig_width = loaded_frames[0].GetInvertedImage().cols;
//unsigned char* host_image = (unsigned char*)malloc(input_width * input_height * sizeof(unsigned char));
//ui.pose_label->setText("Initializing STL model on GPU...");
//ui.qvtk_widget->update();

///*STL Information*/
//vector<vector<float>> triangle_information;
//QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
//stl_reader_BIG::readAnySTL(QString::fromStdString(loaded_models[selected[0].row()].file_location_), triangle_information);

///*GPU Models for the current Model*/
//gpu_cost_function::GPUModel* gpu_mod = new gpu_cost_function::GPUModel("model", true, orig_height, orig_width, 0, false, // switched cols and rows because the stored image is inverted?
//&(triangle_information[0])[0], &(triangle_information[1])[0], triangle_information[0].size() / 9, calibration_file_.camera_A_principal_); // BACKFACE CULLING APPEARS TO BE GIVING ERRORS

//ui.pose_progress->setValue(55);
//ui.pose_label->setText("Initializing tibial implant pose estimation...");
//ui.qvtk_widget->update();

///*Load JIT Model*/
//std::string pt_model_location = "C:/TorchScriptTrainedNetworks/HRNETPR_BS6_dataLimaTib1024_08012019_HRProcessed_Tib_08022019_1_TORCH_SCRIPT.pt";
//// std::shared_ptr<torch::jit::Module> model(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module module(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module* model = &module;
//if (model == nullptr)
//{
//QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
//return;
//}

///*Load JIT Z Model*/
//std::string pt_model_z_location = "C:/TorchScriptTrainedNetworks/HRNETPR_BS6_dataLimaTib1024_08012019_HRProcessed_Tib_08052019_Z_1_TORCH_SCRIPT.pt";
//// std::shared_ptr<torch::jit::Module> model_z(torch::jit::load(pt_model_z_location, torch::kCUDA));
//torch::jit::Module module_z(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module* model_z = &module_z;
//if (model_z == nullptr)
//{
//QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
//return;
//}

///*Send Each Segmented Image to GPU Tensor, Predict Orientation, Then Z (From Area), then X,Y.
//After this, convert to non (0,0) centered orientation.
//Finally, update */
//ui.pose_progress->setValue(65);
//ui.pose_label->setText("Estimating tibial implant poses...");
//ui.qvtk_widget->update();
//float* orientation = new float[3];
//float* z_norm = new float[1];
//torch::Tensor gpu_byte_placeholder(torch::zeros({ 1, 1, input_height, input_width }, torch::device(torch::kCUDA).dtype(torch::kByte)));
//for (int i = 0; i < ui.image_list_widget->count(); i++) {

//cv::Mat orig_inverted = loaded_frames[i].GetInvertedImage();
//cv::Mat padded;
//if (orig_inverted.cols > orig_inverted.rows)
//padded.create(orig_inverted.cols, orig_inverted.cols, orig_inverted.type());
//else
//padded.create(orig_inverted.rows, orig_inverted.rows, orig_inverted.type());
//unsigned int padded_width = padded.cols;
//unsigned int padded_height = padded.rows;
//padded.setTo(cv::Scalar::all(0));
//orig_inverted.copyTo(padded(cv::Rect(0, 0, orig_inverted.cols, orig_inverted.rows)));
//cv::resize(padded, padded, cv::Size(input_width, input_height));

//cudaMemcpy(gpu_byte_placeholder.data_ptr(), padded.data,
//input_width * input_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
//std::vector<torch::jit::IValue> inputs;
//inputs.push_back(gpu_byte_placeholder.to(torch::dtype(torch::kFloat)).flip({ 2 })); // Must flip first
//cudaMemcpy(orientation, model->forward(inputs).toTensor().to(torch::dtype(torch::kFloat)).data_ptr(),
//3 * sizeof(float), cudaMemcpyDeviceToHost);
//cudaMemcpy(z_norm, model_z->forward(inputs).toTensor().to(torch::dtype(torch::kFloat)).data_ptr(),
//sizeof(float), cudaMemcpyDeviceToHost);

///*Flip Segment*/
//cv::Mat output_mat_seg = cv::Mat(orig_inverted.rows, orig_inverted.cols, CV_8UC1);
//cv::flip(orig_inverted, output_mat_seg, 0);

///*Render*/
//gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(0, 0, -calibration_file_.camera_A_principal_.principal_distance_, orientation[1], orientation[2], orientation[0]));

///*Copy To Mat*/
//cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(), orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

///*OpenCV Image Container/Write Function*/
//cv::Mat projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image); /*Reverse before flip*/
//cv::Mat output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
//cv::flip(projection_mat, output_mat, 0);

///*Get Scale*/
//double z = -calibration_file_.camera_A_principal_.principal_distance_ * z_norm[0];

///*Reproject*/
///*Render*/
//gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(0, 0, z, orientation[1], orientation[2], orientation[0]));
//cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(), orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image);
//output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
//cv::flip(projection_mat, output_mat, 0);

///*Get X and Y*/
//cv::Mat proj64;
//output_mat.convertTo(proj64, CV_64FC1);
//cv::Mat seg64;
//output_mat_seg.convertTo(seg64, CV_64FC1);
//cv::Point2d x_y_point = cv::phaseCorrelate(proj64, seg64) * (calibration_file_.camera_A_principal_.pixel_pitch_ * z * -1) /
//calibration_file_.camera_A_principal_.principal_distance_;
//double x = x_y_point.x;
//double y = -1 * x_y_point.y;


///*Convert from (0,0) Centered*/
//float za_rad = orientation[0] * 3.14159265358979323846 / 180.0;
//float xa_rad = orientation[1] * 3.14159265358979323846 / 180.0;
//float ya_rad = orientation[2] * 3.14159265358979323846 / 180.0;
//float cz = cos(za_rad);
//float sz = sin(za_rad);
//float cx = cos(xa_rad);
//float sx = sin(xa_rad);
//float cy = cos(ya_rad);
//float sy = sin(ya_rad);
//Matrix_3_3 R_g(
//cz * cy - sz * sx * sy, -1.0 * sz * cx, cz * sy + sz * cy * sx,
//sz * cy + cz * sx * sy, cz * cx, sz * sy - cz * cy * sx,
//-1.0 * cx * sy, sx, cx * cy);
//float theta_x = std::atan(-1.0 * y / z);
//float theta_y = std::asin(-1.0 * x / std::sqrt(x * x + y * y + z * z));
//Matrix_3_3 R_x(
//1, 0, 0,
//0, cos(theta_x), -sin(theta_x),
//0, sin(theta_x), cos(theta_x));
//Matrix_3_3 R_y(
//cos(theta_y), 0, sin(theta_y),
//0, 1, 0,
//-sin(theta_y), 0, cos(theta_y));
//Matrix_3_3 R_orig = calibration_file_.multiplication_mat_mat(R_y, calibration_file_.multiplication_mat_mat(R_x, R_g));
///*Rot Mat To Eul ZXY*/
///*Algorithm To Recover Z - X - Y Euler Angles*/
//float xa, ya, za;
//if (R_orig.A_32_ < 1) {
//if (R_orig.A_32_ > -1) {
//xa = asin(R_orig.A_32_);
//za = atan2(-1 * R_orig.A_12_, R_orig.A_22_);
//ya = atan2(-1 * R_orig.A_31_, R_orig.A_33_);

//}
//else {
//xa = -3.14159265358979323846 / 2.0;
//za = -1 * atan2(R_orig.A_13_, R_orig.A_11_);
//ya = 0;
//}
//}
//else {
//xa = 3.14159265358979323846 / 2.0;
//za = atan2(R_orig.A_13_, R_orig.A_11_);
//ya = 0;
//}

//xa = xa * 180.0 / 3.14159265358979323846;
//ya = ya * 180.0 / 3.14159265358979323846;
//za = za * 180.0 / 3.14159265358979323846;

///*Update Model Pose*/
//model_locations_.SavePose(i, ui.model_list_widget->currentRow(), Point6D(x, y, z, xa, ya, za));

//ui.pose_progress->setValue(65 + 30 * (double)(i + 1) / (double)ui.image_list_widget->count());
//ui.qvtk_widget->update();
//qApp->processEvents();
//}

//ui.pose_progress->setValue(98);
//ui.pose_label->setText("Deleting old models...");
//ui.qvtk_widget->update();

///*Delete GPU Model*/
//delete gpu_mod;

///*Free Array*/
//free(host_image);

///*Free Values*/
//delete orientation;
//delete z_norm;

///*Update Model*/
//Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentRow(), selected[0].row());
//model_actor_list[selected[0].row()]->SetPosition(loaded_pose.x, loaded_pose.y, loaded_pose.z);
//model_actor_list[selected[0].row()]->SetOrientation(loaded_pose.xa, loaded_pose.ya, loaded_pose.za);
//ui.qvtk_widget->update();

//ui.pose_progress->setValue(100);
//ui.pose_label->setText("Finished!");
//ui.qvtk_widget->update();

///*Pose Estimate Progress and Label Not Visible*/
//ui.pose_progress->setVisible(0);
//ui.pose_label->setVisible(0);
//}
//void MainScreen::on_actionEstimate_Scapula_s_triggered() {
////Must be in Single Selection Mode to Load Pose
//if (ui.multiple_model_radio_button->isChecked()) {
//QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Estimate Kinematics!", QMessageBox::Ok);
//return;
//}

////Must load a model
//if (loaded_models.size() < 1) {
//QMessageBox::critical(this, "Error!", "Must load a model!", QMessageBox::Ok);
//return;
//}

////Must have loaded image
//if (loaded_frames.size() < 1) {
//QMessageBox::critical(this, "Error!", "Must load images!", QMessageBox::Ok);
//return;
//}

///*Pose Estimate Progress and Label Visible*/
//ui.pose_progress->setValue(5);
//ui.pose_progress->setVisible(1);
//ui.pose_label->setText("Initializing high resolution segmentation...");
//ui.pose_label->setVisible(1);
//ui.qvtk_widget->update();
//qApp->processEvents();


///*Segment*/
//segmentHelperFunction("C:/TorchScriptTrainedNetworks/HRNETSeg_BS24_dataAkiraMayPaperNoPatient18Scapula_512_072419_2_TORCH_SCRIPT.pt", 512, 512);
//unsigned int input_height = 512;
//unsigned int input_width = 512;
//unsigned int orig_height = loaded_frames[0].GetInvertedImage().rows;
//unsigned int orig_width = loaded_frames[0].GetInvertedImage().cols;
//unsigned char* host_image = (unsigned char*)malloc(orig_width * orig_height * sizeof(unsigned char));
//ui.pose_label->setText("Initializing STL model on GPU...");
//ui.qvtk_widget->update();

///*STL Information*/
//vector<vector<float>> triangle_information;
//QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
//stl_reader_BIG::readAnySTL(QString::fromStdString(loaded_models[selected[0].row()].file_location_), triangle_information);

///*GPU Models for the current Model*/
//gpu_cost_function::GPUModel* gpu_mod = new gpu_cost_function::GPUModel("model", true, orig_height, orig_width, 0, false, // switched cols and rows because the stored image is inverted?
//&(triangle_information[0])[0], &(triangle_information[1])[0], triangle_information[0].size() / 9, calibration_file_.camera_A_principal_); // BACKFACE CULLING APPEARS TO BE GIVING ERRORS

//ui.pose_progress->setValue(55);
//ui.pose_label->setText("Initializing scapula pose estimation...");
//ui.qvtk_widget->update();

///*Load JIT Model*/
//std::string pt_model_location = "C:/TorchScriptTrainedNetworks/HRNETPR_BS6_dataAkiraPt18_HRProcessed_Sca_08062019_1_TORCH_SCRIPT.pt";
//// std::shared_ptr<torch::jit::Module> model(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module module(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module* model = &module;
//if (model == nullptr)
//{
//QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
//return;
//}

///*Load JIT Z Model*/
//std::string pt_model_z_location = "C:/TorchScriptTrainedNetworks/HRNETPR_BS6_dataAkiraPt18_HRProcessed_Sca_08062019_Z_1_TORCH_SCRIPT.pt";
//// std::shared_ptr<torch::jit::Module> model_z(torch::jit::load(pt_model_z_location, torch::kCUDA));
//torch::jit::Module module_z(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module* model_z = &module_z;
//if (model_z == nullptr)
//{
//QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
//return;
//}

///*Send Each Segmented Image to GPU Tensor, Predict Orientation, Then Z (From Area), then X,Y.
//After this, convert to non (0,0) centered orientation.
//Finally, update */
//ui.pose_progress->setValue(65);
//ui.pose_label->setText("Estimating scapula poses...");
//ui.qvtk_widget->update();
//float* orientation = new float[3];
//float* z_norm = new float[1];
//torch::Tensor gpu_byte_placeholder(torch::zeros({ 1, 1, input_height, input_width }, torch::device(torch::kCUDA).dtype(torch::kByte)));
//for (int i = 0; i < ui.image_list_widget->count(); i++) {

//cv::Mat orig_inverted = loaded_frames[i].GetInvertedImage();
//cv::Mat padded;
//if (orig_inverted.cols > orig_inverted.rows)
//padded.create(orig_inverted.cols, orig_inverted.cols, orig_inverted.type());
//else
//padded.create(orig_inverted.rows, orig_inverted.rows, orig_inverted.type());
//unsigned int padded_width = padded.cols;
//unsigned int padded_height = padded.rows;
//padded.setTo(cv::Scalar::all(0));
//orig_inverted.copyTo(padded(cv::Rect(0, 0, orig_inverted.cols, orig_inverted.rows)));
//cv::resize(padded, padded, cv::Size(input_width, input_height));

//cudaMemcpy(gpu_byte_placeholder.data_ptr(), padded.data,
//input_width * input_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
//std::vector<torch::jit::IValue> inputs;
//inputs.push_back(gpu_byte_placeholder.to(torch::dtype(torch::kFloat)).flip({ 2 })); // Must flip first
//cudaMemcpy(orientation, model->forward(inputs).toTensor().to(torch::dtype(torch::kFloat)).data_ptr(),
//3 * sizeof(float), cudaMemcpyDeviceToHost);
//cudaMemcpy(z_norm, model_z->forward(inputs).toTensor().to(torch::dtype(torch::kFloat)).data_ptr(),
//sizeof(float), cudaMemcpyDeviceToHost);

///*Flip Segment*/
//cv::Mat output_mat_seg = cv::Mat(orig_inverted.rows, orig_inverted.cols, CV_8UC1);
//cv::flip(orig_inverted, output_mat_seg, 0);

///*Render*/
//gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(0, 0, -calibration_file_.camera_A_principal_.principal_distance_, orientation[1], orientation[2], orientation[0]));

///*Copy To Mat*/
//cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(), orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

///*OpenCV Image Container/Write Function*/
//cv::Mat projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image); /*Reverse before flip*/
//cv::Mat output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
//cv::flip(projection_mat, output_mat, 0);

///*Get Scale*/
//double z = -calibration_file_.camera_A_principal_.principal_distance_ * z_norm[0];

///*Reproject*/
///*Render*/
//gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(0, 0, z, orientation[1], orientation[2], orientation[0]));
//cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(), orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image);
//output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
//cv::flip(projection_mat, output_mat, 0);

///*Get X and Y*/
//cv::Mat proj64;
//output_mat.convertTo(proj64, CV_64FC1);
//cv::Mat seg64;
//output_mat_seg.convertTo(seg64, CV_64FC1);
//cv::Point2d x_y_point = cv::phaseCorrelate(proj64, seg64) * (calibration_file_.camera_A_principal_.pixel_pitch_ * z * -1) /
//calibration_file_.camera_A_principal_.principal_distance_;
//double x = x_y_point.x;
//double y = -1 * x_y_point.y;


///*Convert from (0,0) Centered*/
//float za_rad = orientation[0] * 3.14159265358979323846 / 180.0;
//float xa_rad = orientation[1] * 3.14159265358979323846 / 180.0;
//float ya_rad = orientation[2] * 3.14159265358979323846 / 180.0;
//float cz = cos(za_rad);
//float sz = sin(za_rad);
//float cx = cos(xa_rad);
//float sx = sin(xa_rad);
//float cy = cos(ya_rad);
//float sy = sin(ya_rad);
//Matrix_3_3 R_g(
//cz * cy - sz * sx * sy, -1.0 * sz * cx, cz * sy + sz * cy * sx,
//sz * cy + cz * sx * sy, cz * cx, sz * sy - cz * cy * sx,
//-1.0 * cx * sy, sx, cx * cy);
//float theta_x = std::atan(-1.0 * y / z);
//float theta_y = std::asin(-1.0 * x / std::sqrt(x * x + y * y + z * z));
//Matrix_3_3 R_x(
//1, 0, 0,
//0, cos(theta_x), -sin(theta_x),
//0, sin(theta_x), cos(theta_x));
//Matrix_3_3 R_y(
//cos(theta_y), 0, sin(theta_y),
//0, 1, 0,
//-sin(theta_y), 0, cos(theta_y));
//Matrix_3_3 R_orig = calibration_file_.multiplication_mat_mat(R_y, calibration_file_.multiplication_mat_mat(R_x, R_g));
///*Rot Mat To Eul ZXY*/
///*Algorithm To Recover Z - X - Y Euler Angles*/
//float xa, ya, za;
//if (R_orig.A_32_ < 1) {
//if (R_orig.A_32_ > -1) {
//xa = asin(R_orig.A_32_);
//za = atan2(-1 * R_orig.A_12_, R_orig.A_22_);
//ya = atan2(-1 * R_orig.A_31_, R_orig.A_33_);

//}
//else {
//xa = -3.14159265358979323846 / 2.0;
//za = -1 * atan2(R_orig.A_13_, R_orig.A_11_);
//ya = 0;
//}
//}
//else {
//xa = 3.14159265358979323846 / 2.0;
//za = atan2(R_orig.A_13_, R_orig.A_11_);
//ya = 0;
//}

//xa = xa * 180.0 / 3.14159265358979323846;
//ya = ya * 180.0 / 3.14159265358979323846;
//za = za * 180.0 / 3.14159265358979323846;

///*Update Model Pose*/
//model_locations_.SavePose(i, ui.model_list_widget->currentRow(), Point6D(x, y, z, xa, ya, za));

//ui.pose_progress->setValue(65 + 30 * (double)(i + 1) / (double)ui.image_list_widget->count());
//ui.qvtk_widget->update();
//qApp->processEvents();
//}
//QMessageBox::critical(this, "Error!", "zzzzzzzzzz!", QMessageBox::Ok);

//ui.pose_progress->setValue(98);
//ui.pose_label->setText("Deleting old models...");
//ui.qvtk_widget->update();

///*Delete GPU Model*/
//delete gpu_mod;

///*Free Array*/
//free(host_image);

///*Free Values*/
//delete orientation;
//delete z_norm;

///*Update Model*/
//Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentRow(), selected[0].row());
//model_actor_list[selected[0].row()]->SetPosition(loaded_pose.x, loaded_pose.y, loaded_pose.z);
//model_actor_list[selected[0].row()]->SetOrientation(loaded_pose.xa, loaded_pose.ya, loaded_pose.za);
//ui.qvtk_widget->update();

//ui.pose_progress->setValue(100);
//ui.pose_label->setText("Finished!");
//ui.qvtk_widget->update();

///*Pose Estimate Progress and Label Not Visible*/
//ui.pose_progress->setVisible(0);
//ui.pose_label->setVisible(0);
//}

void MainScreen::on_actionNFD_Pose_Estimate_triggered() {
	JTML_NFD nfd_obj;
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	if (selected.size() == 0 || previous_frame_index_ < 0 || ui.image_list_widget->currentIndex().row() !=
		previous_frame_index_ ||
		ui.image_list_widget->currentIndex().row() >= loaded_frames.size() ||
		ui.model_list_widget->currentIndex().row() >= loaded_models.size()) {
		QMessageBox::critical(this, "Error!", "Select Frame and Model First!", QMessageBox::Ok);
		return;
	}
	QString error_mess;
	nfd_obj.Initialize(calibration_file_, loaded_models, loaded_frames, selected,
	                   ui.image_list_widget->currentIndex().row(), error_mess);
	nfd_obj.Run();
}

/*Viewing Controls*/
void MainScreen::on_actionControls_triggered() {
	//Open Viewing Window Controls Window
	Controls cntrls;
	cntrls.exec();
}

/*Optimizer Window*/
void MainScreen::on_actionOptimizer_Settings_triggered() {
	/*Load the Optimizer Settings to the Window*/
	settings_control->LoadSettings(trunk_manager_, branch_manager_, leaf_manager_, optimizer_settings_);

	//Open Optimizer Settings Window
	settings_control->show();
}

/*Symmetry Trap Window*/


/*DRR Settings Window*/
void MainScreen::on_actionDRR_Settings_triggered() {
	/*CHeck if Loaded Models Yet*/
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	if (selected.size() > 0) {
		/*Open DRR Window*/
		DRRTool drt(loaded_models[selected[0].row()], calibration_file_.camera_A_principal_,
		            model_actor_list[selected[0].row()]->GetPosition()[2]);
		drt.exec();
	}
}

/*PREPROCESSOR BUTTONS*/
/*Load Calibration Button*/
void MainScreen::on_load_calibration_button_clicked() {
	/*Open File Search Dialogue*/
	QString calibration_file_extension = QFileDialog::getOpenFileName(this, tr("Load Calibration"), ".",
	                                                                  tr("Calibration File (*.txt)"));

	QFile inputFile(calibration_file_extension);
	if (inputFile.open(QIODevice::ReadOnly)) {
		QTextStream in(&inputFile);
		QStringList InputList = in.readAll().split(QRegExp("[\r\n]|,|\t| "), QString::SkipEmptyParts);

		/*Valid Code for Monoplane*/
		if (InputList[0] == "JT_INTCALIB" || InputList[0] == "JTA_INTCALIB") {

			/*Error Check*/
			if (InputList[4].toDouble() == 0) {
				QMessageBox::critical(this, "Error!",
				                      "Pixel size (the last number in the calibration file) is specified as 0! This is impossible.",
				                      QMessageBox::Ok);
				calibrated_for_monoplane_viewport_ = false;
				calibrated_for_biplane_viewport_ = false;
				inputFile.close();
				return;
			}

			/*Initialize Calibration*/
			calibrated_for_monoplane_viewport_ = true;
			calibrated_for_biplane_viewport_ = false;
			CameraCalibration principal_calibration_file(InputList[1].toDouble(), -1 * InputList[2].toDouble(),
			                                             //Negative For Offsets to make consistent with JointTrack
			                                             -1 * InputList[3].toDouble(), InputList[4].toDouble());
			float* prin_dist_ = &principal_calibration_file.principal_distance_;
			calibration_file_ = Calibration(principal_calibration_file);
			/*Update Interactor Calibration For Converting Text in Camera B View*/
			interactor_calibration = calibration_file_;
			Calibration* cal_pointer_ = &calibration_file_;

			// interactor_calibration.camera_A_principal_.principal_distance_ - should return 1198
			interactor_camera_B = false;
		}
		/*Valid Code for Biplane*/
		else if (InputList[0] == "JTA_INTCALIB_BIPLANE") {
			/*Convert and Do PIX MM Error CHECK*/
			/*Error Check*/
			if (InputList[4].toDouble() == 0 || InputList[8].toDouble() == 0) {
				QMessageBox::critical(this, "Error!",
				                      "Pixel size (the last number in the calibration file) is specified as 0! This is impossible.",
				                      QMessageBox::Ok);
				calibrated_for_monoplane_viewport_ = false;
				calibrated_for_biplane_viewport_ = false;
				inputFile.close();
				return;
			}
			/*Initialize Calibrations*/
			calibrated_for_monoplane_viewport_ = false;
			calibrated_for_biplane_viewport_ = true;
			/*Calibrate for Main View (A) and alternate view (B).
			Read in (x,y,z) displacement vector from origin (where A is) to origin of camera B.
			Read in othroogonal axis matrix for camera B (A is taken to be standard basis vectors)*/
			CameraCalibration principal_calibration_file_A(InputList[1].toDouble(), -1 * InputList[2].toDouble(),
			                                               -1 * InputList[3].toDouble(), InputList[4].toDouble());
			//Negatives to make consistent with JT
			CameraCalibration principal_calibration_file_B(InputList[5].toDouble(), -1 * InputList[6].toDouble(),
			                                               -1 * InputList[7].toDouble(), InputList[8].toDouble());
			Vect_3 origin_B(InputList[9].toDouble(), InputList[10].toDouble(), InputList[11].toDouble());
			Matrix_3_3 orthogonal_axes_B(InputList[12].toDouble(), InputList[13].toDouble(), InputList[14].toDouble(),
			                             InputList[15].toDouble(), InputList[16].toDouble(), InputList[17].toDouble(),
			                             InputList[18].toDouble(), InputList[19].toDouble(), InputList[20].toDouble());
			calibration_file_ = Calibration(principal_calibration_file_A, principal_calibration_file_B, origin_B,
			                                orthogonal_axes_B);

			/*Update Interactor Calibration For Converting Text in Camera B View*/
			interactor_calibration = calibration_file_;
		}
		else if (InputList[0] == "image") {
			CameraCalibration denver_calibration_A(InputList[6].toDouble(), InputList[7].toDouble(),
			                                       InputList[8].toDouble(), InputList[10].toDouble(),
			                                       InputList[11].toDouble());

			calibrated_for_monoplane_viewport_ = true;
			calibrated_for_biplane_viewport_ = false;
			calibration_file_ = Calibration(denver_calibration_A,"Denver");
			inputFile.close();
		}
		/*Invalid Code*/
		else {
			QMessageBox::critical(this, "Error!", "Invalid Configuration File!", QMessageBox::Ok);
			calibrated_for_monoplane_viewport_ = false;
			calibrated_for_biplane_viewport_ = false;
			inputFile.close();
			return;
		}
		inputFile.close();
	}
	/*Set Up QVTK Widget For Calibration*/
	/*Monoplane (Left Viewport)*/
	vw->load_renderers_into_render_window(calibration_file_);
	if (calibrated_for_monoplane_viewport_) {
		vw->setup_camera_calibration(calibration_file_);

		/*Set Checked To Monoplane but disable from further clicking*/
		ui.camera_A_radio_button->setChecked(true);
		ui.camera_A_radio_button->setDisabled(true);

		/*Disable Biplane*/
		ui.camera_B_radio_button->setDisabled(true);

		/*If Already loaded images CANT HAPPEN ANYMORE AS CALIBRATION IS ONE USE BUTTON*/
		if (ui.image_list_widget->currentIndex().row() >= 0) {
			/*Upload Image Data to Screen, Shift Image Location to Center In Middle of Screen and Adjust View Angle*/
			vw->place_image_actors_according_to_calibration(calibration_file_,
				loaded_frames[this->curr_frame()].GetOriginalImage().cols,
				loaded_frames[this->curr_frame()].GetOriginalImage().rows);
			renderer->GetActiveCamera()->SetViewAngle(CalculateViewingAngle(
				loaded_frames[ui.image_list_widget->currentIndex().row()].GetOriginalImage().cols,
				loaded_frames[ui.image_list_widget->currentIndex().row()].GetOriginalImage().rows,
				true));
		}

		/*Disable Reloading Calibration File*/
		ui.load_calibration_button->setDisabled(true);
		/*Enable View Menu Until Calbration Loaded*/
		ui.actionReset_View->setDisabled(false);
		ui.actionReset_Normal_Up->setDisabled(false);
		ui.actionModel_Interaction_Mode->setDisabled(false);
		ui.actionCamera_Interaction_Mode->setDisabled(false);
	}
	/*Biplane Calibration*/
	else if (calibrated_for_biplane_viewport_) {
		/*Set Up Calibration for Camera A to Home QVTKWidget*/
		renderer->GetActiveCamera()->SetFocalPoint(0, 0,
		                                           -1 * calibration_file_.camera_A_principal_.principal_distance_ /
		                                           calibration_file_.camera_A_principal_.pixel_pitch_);
		renderer->GetActiveCamera()->SetPosition(0, 0, 0);
		renderer->GetActiveCamera()->SetClippingRange(
			.1, 2.0 * calibration_file_.camera_A_principal_.principal_distance_ / calibration_file_.camera_A_principal_.
			pixel_pitch_);

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
	/*If TRUNK is Has Integer Parameter called Dilation, Update Dilation Values for Viewing Purposes*/
	int dilation_val = 0;
	std::vector<jta_cost_function::Parameter<int>> active_int_params = trunk_manager_.getActiveCostFunctionClass()->
		getIntParameters();
	for (int i = 0; i < active_int_params.size(); i++) {
		if (active_int_params[i].getParameterName() == "Dilation") {
			dilation_val = (trunk_manager_.getActiveCostFunctionClass())->getIntParameters().at(i).getParameterValue();
		}
	}
	if (dilation_val < 0) {
		dilation_val = 0;
	}

	/*Check to See if Calibration Loaded*/
	if (calibrated_for_monoplane_viewport_ == false && calibrated_for_biplane_viewport_ == false) {
		QMessageBox::critical(this, "Error!", "Load Calibration First!", QMessageBox::Ok);
		return;
	}

	/*If MONOPLANE Only*/
	if (calibrated_for_monoplane_viewport_) {
		//Load TIFF images
		QStringList TiffFileExtensions = QFileDialog::getOpenFileNames(this, tr("Load Image(s)"), ".",
		                                                               tr("Image File(s) (*.tif *.tiff)"));
		for (int i = 0; i < TiffFileExtensions.size(); i++) {
			auto new_frame = Frame(TiffFileExtensions[i].toStdString(), ui.aperture_spin_box->value(),
			                       ui.low_threshold_slider->value(),
			                       ui.high_threshold_slider->value(), dilation_val);
			/*Check That All Frames Are The Same Size and Not Empty*/
			int width = new_frame.GetEdgeImage().cols;
			int height = new_frame.GetEdgeImage().rows;
			for (int j = 0; j < loaded_frames.size(); j++) {
				if (width != loaded_frames[j].GetEdgeImage().cols ||
					height != loaded_frames[j].GetEdgeImage().rows) {
					QMessageBox::critical(this, "Error!", "Images Loaded Must Be The Same Size!", QMessageBox::Ok);
					goto stop;
				}
			}
			//Add to Loaded Frames
			loaded_frames.push_back(new_frame);
			//Populate Frame List Widget
			ui.image_list_widget->addItem(
				QFileInfo(QString::fromStdString(TiffFileExtensions[i].toStdString())).baseName());
			/*Add Blank Model Locations for Loaded Models*/
			model_locations_.LoadNewFrame();
		}
		/*Exit Label*/
	stop:;
		vw->set_loaded_frames(loaded_frames);

		//If No Loaded Frames, Default Select First
		if (ui.image_list_widget->currentRow() < 0 && loaded_frames.size() > 0) {
			ui.image_list_widget->setCurrentRow(0);
		}

		//this->vw->set_loaded_frames(loaded_frames);


	}
	else if (calibrated_for_biplane_viewport_) {
		//Load TIFF images for Camera A and Camera B - Must Be Same Amount or Error and None Will Load!
		QStringList TiffFileExtensionsCamera_A = QFileDialog::getOpenFileNames(
			this, tr("Load Image(s) for Camera A"), ".", tr("Image File(s) (*.tif *.tiff)"));
		QStringList TiffFileExtensionsCamera_B = QFileDialog::getOpenFileNames(
			this, tr("Load Image(s) for Camera B"), ".", tr("Image File(s) (*.tif *.tiff)"));

		/*Check Same Amount of Loaded Images*/
		if (TiffFileExtensionsCamera_A.size() != TiffFileExtensionsCamera_B.size()) {
			QMessageBox::critical(this, "Error!", "Please Load the Same Number of Images for Each Camera!",
			                      QMessageBox::Ok);
			return;
		}

		for (int i = 0; i < TiffFileExtensionsCamera_A.size(); i++) {
			auto new_frame_A = Frame(TiffFileExtensionsCamera_A[i].toStdString(), ui.aperture_spin_box->value(),
			                         ui.low_threshold_slider->value(),
			                         ui.high_threshold_slider->value(), dilation_val);
			auto new_frame_B = Frame(TiffFileExtensionsCamera_B[i].toStdString(), ui.aperture_spin_box->value(),
			                         ui.low_threshold_slider->value(),
			                         ui.high_threshold_slider->value(), dilation_val);
			/*Check That All Frames Are The Same Size and Not Empty*/
			int width = new_frame_A.GetEdgeImage().cols;
			int height = new_frame_A.GetEdgeImage().rows;
			for (int j = 0; j < loaded_frames.size(); j++) {
				if (width != loaded_frames[j].GetEdgeImage().cols ||
					height != loaded_frames[j].GetEdgeImage().rows) {
					QMessageBox::critical(this, "Error!", "Images Loaded Must Be The Same Size!", QMessageBox::Ok);
					goto stop_biplane;
				}
			}
			width = new_frame_B.GetEdgeImage().cols;
			height = new_frame_B.GetEdgeImage().rows;
			for (int j = 0; j < loaded_frames_B.size(); j++) {
				if (width != loaded_frames_B[j].GetEdgeImage().cols ||
					height != loaded_frames_B[j].GetEdgeImage().rows) {
					QMessageBox::critical(this, "Error!", "Images Loaded Must Be The Same Size!", QMessageBox::Ok);
					goto stop_biplane;
				}
			}

			//Add to Loaded Frames
			loaded_frames.push_back(new_frame_A);
			loaded_frames_B.push_back(new_frame_B);
			//Populate Frame List Widget
			ui.image_list_widget->addItem(
				"A: " + QFileInfo(QString::fromStdString(TiffFileExtensionsCamera_A[i].toStdString())).baseName() +
				"\nB: " + QFileInfo(QString::fromStdString(TiffFileExtensionsCamera_B[i].toStdString())).baseName());
			/*Add Blank Model Locations for Loaded Models*/
			model_locations_.LoadNewFrame();
		}
		/*Exit Label*/
	stop_biplane:;

		//If No Loaded Frames, Default Select First
		if (ui.image_list_widget->currentRow() < 0 && loaded_frames.size() > 0) {
			ui.image_list_widget->setCurrentRow(0);
		}
		vw->set_loaded_frames(loaded_frames);
		vw->set_loaded_frames_b(loaded_frames_B);
	}

}

/*Load Model Button*/
void MainScreen::on_load_model_button_clicked() {
	/*Check to See if Calibration Loaded*/
	if (calibrated_for_monoplane_viewport_ == false && calibrated_for_biplane_viewport_ == false) {
		QMessageBox::critical(this, "Error!", "Load Calibration First!", QMessageBox::Ok);
		return;
	}

	//Load CAD Model
	QStringList CADFileExtensions = QFileDialog::getOpenFileNames(this, tr("Load Implant Model(s)"), ".",
	                                                              tr("CAD File(s) (*.stl)"));

	/*For Each Cad File Extension Create Model Name*/
	QStringList CADModelNames, OldCADModelNames, LoadedCADModelNames;

	/*Initialize List With All CAD Model Names*/
	for (int i = 0; i < loaded_models.size(); i++) {
		OldCADModelNames.push_back(QString::fromStdString(loaded_models[i].model_name_));
	}
	for (int i = 0; i < CADFileExtensions.size(); i++) {
		LoadedCADModelNames.push_back(QFileInfo(QString::fromStdString(CADFileExtensions[i].toStdString())).baseName());
	}
	/*Check to See if Loaded Names are Unique*/
	for (int i = 0; i < LoadedCADModelNames.size(); i++) {
		QString temp_model_name = LoadedCADModelNames[i];
		int already_exists = 1;
		/*Search To See If Name Already Exists*/
		for (int j = 0; j < LoadedCADModelNames.size(); j++) {
			if (LoadedCADModelNames[j] == temp_model_name && j != i) {
				j = -1;
				already_exists++;
				temp_model_name = LoadedCADModelNames[i] + "(" + QString::number(already_exists) + ")";
			}
		}
		LoadedCADModelNames[i] = temp_model_name;
	}

	/*Check to See if Already Exists in Model List*/
	for (int i = 0; i < LoadedCADModelNames.size(); i++) {
		QString temp_model_name = LoadedCADModelNames[i];
		int already_exists = 1;
		/*Search To See If Name Already Exists*/
		for (int j = 0; j < OldCADModelNames.size(); j++) {
			if (OldCADModelNames[j] == temp_model_name) {
				j = -1;
				already_exists++;
				temp_model_name = LoadedCADModelNames[i] + "(" + QString::number(already_exists) + ")";
			}
		}
		CADModelNames.push_back(temp_model_name);
	}

	//for (int i = 0; i < CADFileExtensions.size(); i++) loaded_models.push_back(Model(CADFileExtensions[i].toStdString(), CADModelNames[i].toStdString(), "BLANK"));
	vw->load_models(CADFileExtensions, CADModelNames);
	for (int i = 0; i<CADFileExtensions.size(); i++) {
		loaded_models.push_back(Model(CADFileExtensions[i].toStdString(),
			CADModelNames[i].toStdString(),
			"BLANK"));
	}
	for (int i = 0; i < CADFileExtensions.size(); i++) {
		if (vw->are_models_loaded_incorrectly(i)) {
			QMessageBox::warning(this, "Warning!",
			                     "It is Possible that " + CADModelNames[i] + " (" + CADFileExtensions[i] + ") " +
			                     " is an invalid or corrupted STL file format. Proceed with caution!", QMessageBox::Ok);
		}
	}

	//Populate Model List Widget
	for (int i = 0; i < CADFileExtensions.size(); i++) {
		ui.model_list_widget->addItem(CADModelNames[i]);
	}

	/*Load Blank Poses for Available Frames (and Default Blank Poses even if no frames for viewing without frame)*/
	for (int i = 0; i < CADFileExtensions.size(); i++) {
		//model_locations_.LoadNewModel(calibration_file_.camera_A_principal_.principal_distance_,
		//`                              calibration_file_.camera_A_principal_.pixel_pitch_);
		model_locations_.LoadNewModel(calibration_file_);
	}
	vw->load_3d_models_into_actor_and_mapper_list();
	vw->load_model_actors_and_mappers_with_3d_data();
	//If No Loaded Models, Default Select First
	if (ui.model_list_widget->selectionModel()->selectedRows().size() == 0) {
		ui.model_list_widget->setCurrentRow(0);
	}
	if (calibration_file_.type_ == "UF") {
		vw->set_vtk_camera_from_calibration_and_image_size_if_jta(calibration_file_, 
			loaded_frames[0].GetOriginalImage().cols, 
			loaded_frames[0].GetOriginalImage().rows);
	} else if (calibration_file_.type_ == "Denver") {
		vw->set_vtk_camera_from_calibration_and_image_if_camera_matrix(calibration_file_,
			loaded_frames[0].GetOriginalImage().cols,
			loaded_frames[0].GetOriginalImage().rows);
	}

}

/*Biplane View Button (Camera A,Camera B*/
/*Biplane View A OR Monoplane*/
void MainScreen::on_camera_A_radio_button_clicked() {
	/*Interactor Boolean For Text Display (Convert to Camera A Coordinates)*/
	interactor_camera_B = false;

	/*Load Models Selected Indices*/
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();

	/*Make Sure A Row is Selected*/
	if (ui.image_list_widget->currentIndex().row() >= 0) {
		/*Disable Checking if biplane and save pose*/
		if (calibrated_for_biplane_viewport_ == true) {
			ui.camera_A_radio_button->setDisabled(true);
			ui.camera_B_radio_button->setDisabled(false);

			/*Save Last Pair Pose*/
			for (int r = 0; r < selected.size(); r++) {
				if (selected.size() != 0 && previous_frame_index_ != -1 && !currently_optimizing_) {
					double* position_curr = model_actor_list[selected[r].row()]->GetPosition();
					double* orientation_curr = model_actor_list[selected[r].row()]->GetOrientation();
					Point6D last_pose(position_curr[0], position_curr[1], position_curr[2],
					                  orientation_curr[0], orientation_curr[1], orientation_curr[2]);
					/*Camera A View, Save in Camera A coordinates by converting camera B*/
					model_locations_.SavePose(previous_frame_index_, selected[r].row(),
					                          calibration_file_.convert_Pose_B_to_Pose_A(last_pose));
				}
			}
		}

		/*Update to that frame's canny values*/
		ui.aperture_spin_box->setValue(loaded_frames[ui.image_list_widget->currentIndex().row()].GetAperture());
		ui.low_threshold_slider->setValue(loaded_frames[ui.image_list_widget->currentIndex().row()].GetLowThreshold());
		ui.high_threshold_slider->
		   setValue(loaded_frames[ui.image_list_widget->currentIndex().row()].GetHighThreshold());

		/*Display Corresponding Radio Button view to main QVTK widget*/
		if (ui.original_image_radio_button->isChecked() == true) {
			vw->update_display_background_to_original_image(this->curr_frame(), true);
		}
		else if (ui.inverted_image_radio_button->isChecked() == true) {
			vw->update_display_background_to_inverted_image(this->curr_frame(), true);
		}
		else if (ui.edges_image_radio_button->isChecked() == true) {
			vw->update_display_background_to_edge_image(this->curr_frame(), true);
		}
		else {
			/*Convert Selected Frame's Corresponding Picture to VTK Image Data*/
			vw->update_display_background_to_dilation_image(this->curr_frame(), true);
		}

		vw->place_image_actors_according_to_calibration(
			calibration_file_,
			loaded_frames[this->curr_frame()].GetOriginalImage().rows,
			loaded_frames[this->curr_frame()].GetOriginalImage().cols
		);
		//renderer->GetActiveCamera()->SetViewAngle(setAngle(renderer, loaded_frames[ui.image_list_widget->currentIndex().row()].GetOriginalImage().rows));
		renderer->GetActiveCamera()->SetViewAngle(CalculateViewingAngle(
			loaded_frames[ui.image_list_widget->currentIndex().row()].GetOriginalImage().cols,
			loaded_frames[ui.image_list_widget->currentIndex().row()].GetOriginalImage().rows,
			true));

		/*Set Model Pose if models are loaded*/
		if (selected.size() != 0 && !currently_optimizing_) {
			/*Save Last Pair Pose*/
			for (int r = 0; r < selected.size(); r++) {
				Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentIndex().row(),
				                                               selected[r].row());
				model_actor_list[selected[r].row()]->SetPosition(loaded_pose.x, loaded_pose.y, loaded_pose.z);
				model_actor_list[selected[r].row()]->SetOrientation(loaded_pose.xa, loaded_pose.ya, loaded_pose.za);

				/*Text Actor if On*/
				if (actor_text->GetTextProperty()->GetOpacity() > 0.5) {
					std::string infoText = "Location: <";
					infoText += std::to_string(
							static_cast<long double>(model_actor_list[selected[r].row()]->GetPosition()[0])) + ","
						+ std::to_string(
							static_cast<long double>(model_actor_list[selected[r].row()]->GetPosition()[1])) + ","
						+ std::to_string(
							static_cast<long double>(model_actor_list[selected[r].row()]->GetPosition()[2])) +
						">\nOrientation: <"
						+ std::to_string(
							static_cast<long double>(model_actor_list[selected[r].row()]->GetOrientation()[0])) + ","
						+ std::to_string(
							static_cast<long double>(model_actor_list[selected[r].row()]->GetOrientation()[1])) + ","
						+ std::to_string(
							static_cast<long double>(model_actor_list[selected[r].row()]->GetOrientation()[2])) + ">";
					actor_text->SetInput(infoText.c_str());
				}
			}
		}
		/*Allow For Updates on Screen if optimizing*/
		if (currently_optimizing_ && calibrated_for_biplane_viewport_) {
			/*Save Last Pair Pose*/
			for (int r = 0; r < selected.size(); r++) {
				auto current_pose = Point6D(model_actor_list[selected[r].row()]->GetPosition()[0],
				                            model_actor_list[selected[r].row()]->GetPosition()[1],
				                            model_actor_list[selected[r].row()]->GetPosition()[2],
				                            model_actor_list[selected[r].row()]->GetOrientation()[0],
				                            model_actor_list[selected[r].row()]->GetOrientation()[1],
				                            model_actor_list[selected[r].row()]->GetOrientation()[2]);
				current_pose = calibration_file_.convert_Pose_B_to_Pose_A(current_pose);
				model_actor_list[selected[r].row()]->SetPosition(current_pose.x, current_pose.y, current_pose.z);
				model_actor_list[selected[r].row()]->SetOrientation(current_pose.xa, current_pose.ya, current_pose.za);
			}
		}

		/*update qvtk widget*/
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}

}

/*Biplane View B*/
void MainScreen::on_camera_B_radio_button_clicked() {
	/*Interactor Boolean For Text Display (Convert to Camera A Coordinates)*/
	interactor_camera_B = true;

	/*Load Models Selected Indices*/
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();

	/*Make Sure A Row is Selected*/
	if (ui.image_list_widget->currentIndex().row() >= 0) {

		/*Disable Checking*/
		ui.camera_B_radio_button->setDisabled(true);
		ui.camera_A_radio_button->setDisabled(false);

		/*Save Last Pair Pose*/
		for (int r = 0; r < selected.size(); r++) {
			if (selected.size() != 0 && previous_frame_index_ != -1 && !currently_optimizing_) {
				double* position_curr = vw->get_model_position_at_index(selected[r].row());
				double* orientation_curr = vw->get_model_orientation_at_index(selected[r].row());
				Point6D last_pose(position_curr[0], position_curr[1], position_curr[2],
				                  orientation_curr[0], orientation_curr[1], orientation_curr[2]);
				/*If Camera B View, Save in Camera A coordinates*/
				model_locations_.SavePose(previous_frame_index_, selected[r].row(), last_pose);
			}
		}
		/*Update to that frame's canny values*/
		ui.aperture_spin_box->setValue(loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetAperture());
		ui.low_threshold_slider->
		   setValue(loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetLowThreshold());
		ui.high_threshold_slider->setValue(
			loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetHighThreshold());

		/*Display Corresponding Radio Button view to main QVTK widget*/
		if (ui.original_image_radio_button->isChecked() == true) {
			vw->update_display_background_to_original_image(this->curr_frame()
			                                                , false);
		}
		else if (ui.inverted_image_radio_button->isChecked() == true) {
			vw->update_display_background_to_inverted_image(this->curr_frame(), false);
		}
		else if (ui.edges_image_radio_button->isChecked() == true) {
			vw->update_display_background_to_edge_image(this->curr_frame(), false);
		}
		else {
			vw->update_display_background_to_dilation_image(this->curr_frame(), false);
		}

		vw->place_image_actors_according_to_calibration(calibration_file_.camera_B_principal_, loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetOriginalImage().rows,
			loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetOriginalImage().cols);
		renderer->GetActiveCamera()->SetViewAngle(CalculateViewingAngle(
			loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetOriginalImage().cols,
			loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetOriginalImage().rows,
			false));

		/*Set Model Pose if models are loaded*/
		if (selected.size() != 0 && !currently_optimizing_) {
			/*Save Last Pair Pose*/
			for (int r = 0; r < selected.size(); r++) {
				/*Convert To relative Camera B Pose as storage is done in camera A coordinates and rotations*/
				Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentIndex().row(),
				                                               selected[r].row());
				Point6D relative_B_pose = calibration_file_.convert_Pose_A_to_Pose_B(loaded_pose);
				vw->set_model_position_at_index(selected[r].row(), 
					relative_B_pose.x, 
					relative_B_pose.y, 
					relative_B_pose.z);
				vw->set_model_orientation_at_index(selected[r].row(),
					relative_B_pose.xa,
					relative_B_pose.ya,
					relative_B_pose.za);

				/*Text Actor if On*/
				if (actor_text->GetTextProperty()->GetOpacity() > 0.5) {
					/*Display In Terms of Camera A*/
					vw->set_actor_text(vw->print_location_and_orientation_of_model_at_index(ui.image_list_widget->currentIndex().row()));
					std::string infoText = "Location: <";
				}
			}
		}
		/*Allow For Updates on Screen if optimizing*/
		if (currently_optimizing_ && calibrated_for_biplane_viewport_) {
			/*Save Last Pair Pose*/
			for (int r = 0; r < selected.size(); r++) {
				auto curr_pos = vw->get_model_position_at_index(selected[r].row());
				auto curr_or = vw->get_model_orientation_at_index(selected[r].row());
				auto current_pose = Point6D(curr_pos[0], curr_pos[1], curr_pos[2], curr_or[0], curr_or[1], curr_or[2]);
				current_pose = calibration_file_.convert_Pose_A_to_Pose_B(current_pose);
				vw->set_model_position_at_index(selected[r].row(), current_pose.x, current_pose.y, current_pose.z);
				vw->set_model_orientation_at_index(selected[r].row(), current_pose.xa, current_pose.ya, current_pose.za);
			}
		}
		/*update qvtk widget*/
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
}

/*List Widgets (Model and Frame)*/
/*Frame Widget*/
/* this is where we are setting the current background */
void MainScreen::on_image_list_widget_itemSelectionChanged() {
	/*Make Sure A View is Selected*/
	if (ui.original_image_radio_button->isChecked() == false && ui.inverted_image_radio_button->isChecked() == false
		&& ui.edges_image_radio_button->isChecked() == false && ui.dilation_image_radio_button->isChecked() == false) {
		ui.original_image_radio_button->setChecked(true);
	}

	/*Save Last Pair Pose if not currently optimizing*/
	if (!currently_optimizing_) {
		SaveLastPose();
	}

	/*Update Last Viewed Index as This One*/
	previous_frame_index_ = ui.image_list_widget->currentIndex().row();

	/*Update to that frame's canny values*/
	if (ui.camera_A_radio_button->isChecked()) {
		ui.aperture_spin_box->setValue(loaded_frames[ui.image_list_widget->currentIndex().row()].GetAperture());
		ui.low_threshold_slider->setValue(loaded_frames[ui.image_list_widget->currentIndex().row()].GetLowThreshold());
		ui.high_threshold_slider->
		   setValue(loaded_frames[ui.image_list_widget->currentIndex().row()].GetHighThreshold());
	}
	else if (ui.camera_B_radio_button->isChecked() && calibrated_for_biplane_viewport_) {
		ui.aperture_spin_box->setValue(loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetAperture());
		ui.low_threshold_slider->
		   setValue(loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetLowThreshold());
		ui.high_threshold_slider->setValue(
			loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetHighThreshold());
	}


	/*Display Corresponding Radio Button view to main QVTK widget*/
	if (ui.original_image_radio_button->isChecked() == true) {
		/*Convert Selected Frame's Corresponding Picture to VTK Image Data*/
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_original_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_original_image(this->curr_frame(), false);
		}
	}
	else if (ui.inverted_image_radio_button->isChecked() == true) {
		/*Convert Selected Frame's Corresponding Picture to VTK Image Data*/
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_inverted_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_inverted_image(this->curr_frame(), false);
		}
	}
	else if (ui.edges_image_radio_button->isChecked() == true) {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_edge_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_edge_image(this->curr_frame(), false);
		}
	}
	else {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_dilation_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_dilation_image(this->curr_frame(), false);
		}
	}

	/*Upload Image Data to Screen, Shift Image Location to Center In Middle of Screen and Adjust View Angle*/
	if (ui.camera_A_radio_button->isChecked()) {
		vw->place_image_actors_according_to_calibration(
			calibration_file_,
			loaded_frames[this->curr_frame()].GetOriginalImage().rows,
			loaded_frames[this->curr_frame()].GetOriginalImage().cols
		);
		renderer->GetActiveCamera()->SetViewAngle(CalculateViewingAngle(
			loaded_frames[ui.image_list_widget->currentIndex().row()].GetOriginalImage().cols,
			loaded_frames[ui.image_list_widget->currentIndex().row()].GetOriginalImage().rows,
			true));
	}
	else {
		actor_image->SetPosition(
			-.5 * loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetOriginalImage().cols +
			calibration_file_.camera_B_principal_.principal_x_ / calibration_file_.camera_B_principal_.pixel_pitch_,
			-.5 * loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetOriginalImage().rows +
			calibration_file_.camera_B_principal_.principal_y_ / calibration_file_.camera_B_principal_.pixel_pitch_,
			-1 * calibration_file_.camera_B_principal_.principal_distance_ / calibration_file_.camera_B_principal_.
			pixel_pitch_);
		//renderer->GetActiveCamera()->SetViewAngle(setAngle(renderer, loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetOriginalImage().rows));
		renderer->GetActiveCamera()->SetViewAngle(CalculateViewingAngle(
			loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetOriginalImage().cols,
			loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetOriginalImage().rows,
			false));
	}

	/*Load Models to Screen*/
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	/*Hide Text if Nothing Selected*/
	if (selected.size() == 0) {
		actor_text->VisibilityOff();
	}
	else {
		actor_text->VisibilityOn();
	}
	/*Load Models*/
	for (int i = 0; i < selected.size(); i++) {

		/*Display Corresponding Radio Button view to main QVTK widget*/
		/*Original Model*/
		if (ui.original_model_radio_button->isChecked() == true) {
			vw->change_model_opacity_to_original(selected[i].row());
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
		/*Solid Color Model*/
		else if (ui.solid_model_radio_button->isChecked() == true) {
			vw->change_model_opacity_to_solid(selected[i].row());
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
		/*Transparent Model*/
		else if (ui.transparent_model_radio_button->isChecked() == true) {
			vw->change_model_opacity_to_transparent(selected[i].row());
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
		/*Wireframe Model*/
		else if (ui.wireframe_model_radio_button->isChecked()) {
			vw->change_model_opacity_to_wire_frame(selected[i].row());
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}

		/*If Camera A View*/
		if (ui.camera_A_radio_button->isChecked()) {
			/*Set Model Pose*/
			Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentIndex().row(),
			                                               selected[i].row());
			vw->set_model_position_at_index(selected[i].row(),
				loaded_pose.x,
				loaded_pose.y,
				loaded_pose.z);
			vw->set_model_orientation_at_index(selected[i].row(),
				loaded_pose.xa,
				loaded_pose.ya,
				loaded_pose.za);
			/*Text Actor if On*/
			if (actor_text->GetTextProperty()->GetOpacity() > 0.5) {
				vw->set_actor_text(vw->print_location_and_orientation_of_model_at_index(selected[i].row()));
				vw->set_actor_text_color_to_model_color_at_index(selected[i].row());
			}
		}
		else {
			/*Else, Camera B View*/
			/*Convert To relative Camera B Pose as storage is done in camera A coordinates and rotations*/
			Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentIndex().row(),
			                                               selected[i].row());
			Point6D relative_B_pose = calibration_file_.convert_Pose_A_to_Pose_B(loaded_pose);
			vw->set_model_position_at_index(selected[i].row(),
				relative_B_pose.x,
				relative_B_pose.y,
				relative_B_pose.z);
			vw->set_model_orientation_at_index(selected[i].row(),
				relative_B_pose.xa,
				relative_B_pose.ya,
				relative_B_pose.za);

			/*Text Actor if On*/
			if (actor_text->GetTextProperty()->GetOpacity() > 0.5) {
				vw->set_actor_text(vw->print_location_and_orientation_of_model_at_index(selected[i].row()));
				vw->set_actor_text_color_to_model_color_at_index(selected[i].row());
			}
		}
	}
	/*update qvtk widget*/
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();

}

QModelIndexList MainScreen::selected_model_indices() {
	return ui.model_list_widget->selectionModel()->selectedRows();
}
void MainScreen::remove_background_highlights_from_model_list_widget() {
	for (int i = 0; i < loaded_models.size();i++) {
		ui.model_list_widget->item(i)->setBackground(Qt::transparent);
	}
}
void MainScreen::print_selected_item() {
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	for (int i = 0; i<selected.size();i++) {
	}
}


/*Model Widget*/
void MainScreen::on_model_list_widget_itemSelectionChanged() {
	/*Save Last Pair Pose if not currently optimizing*/
	if (!currently_optimizing_) {
		SaveLastPose(); // Needs Work
	}
	/*Update Last Viewed Index as This One*/
	previous_model_indices_ = ui.model_list_widget->selectionModel()->selectedRows();

	/*Load Models to Screen*/
	vw->make_all_models_invisible();
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	/*Hide Text if Nothing Selected*/
	if (selected.size() == 0) {
		actor_text->VisibilityOff();
		if (ui.model_list_widget->currentIndex().row() >= 0) {
			ui.model_list_widget->item(ui.model_list_widget->currentIndex().row())->setSelected(true);
			return;
		}
	}
	else {
		actor_text->VisibilityOn();
	}

	/*Load Models and set their respective colors in the model list widget*/
	for (int i = 0; i < selected.size(); i++) {

		// Set a style sheet for selected items in the list widget, controls background colors for .stl model names
		ui.model_list_widget->setStyleSheet( 
			"QListView::item{background-color:rgb(0,33,165);}"
			"QListView::item:selected{background-color: rgb(250,70,22);}" );

		// If first selected item, make the model orange. Otherwise, make it blue.
		if (i == 0) {
			vw->set_3d_model_color(selected[i].row(), UF_ORANGE);
		}
		else  {
			//vw->make_model_visible_and_pickable_at_index(i);
			vw->set_3d_model_color(selected[i].row(), UF_BLUE);
		}

		if (ui.original_model_radio_button->isChecked() == true) {
			vw->change_model_opacity_to_original(selected[i].row());
			ui.qvtk_widget->update();
			ui.qvtk_widget->renderWindow()->Render();
		}
		else if (ui.solid_model_radio_button->isChecked() == true) {
			vw->change_model_opacity_to_solid(selected[i].row());
			ui.qvtk_widget->update();
			ui.qvtk_widget->renderWindow()->Render();
		}
		else if (ui.transparent_model_radio_button->isChecked() == true) {
			vw->change_model_opacity_to_transparent(selected[i].row());
			ui.qvtk_widget->update();
			ui.qvtk_widget->renderWindow()->Render();
		}
		/*Wireframe Model*/
		else if (ui.wireframe_model_radio_button->isChecked()) {
			vw->change_model_opacity_to_wire_frame(selected[i].row());
			ui.qvtk_widget->update();
			ui.qvtk_widget->renderWindow()->Render();
		}

		/*If Camera A View*/
		if (ui.camera_A_radio_button->isChecked()) {
			/*Set Model Pose*/
			Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentIndex().row(),
			                                               selected[i].row());
			vw->set_model_position_at_index(selected[i].row(), loaded_pose.x, loaded_pose.y, loaded_pose.z);
			vw->set_model_orientation_at_index(selected[i].row(), loaded_pose.xa, loaded_pose.ya, loaded_pose.za);
			
			/*Text Actor if On */
			if (actor_text->GetTextProperty()->GetOpacity() > 0.5) {
				vw->set_actor_text(vw->print_location_and_orientation_of_model_at_index(selected[i].row()));
				vw->set_actor_text_color_to_model_color_at_index(selected[i].row());
			}
		}
		else {
			/*Else, Camera B View*/
			/*Convert To relative Camera B Pose as storage is done in camera A coordinates and rotations*/
			Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentIndex().row(),
			                                               selected[i].row());
			Point6D relative_B_pose = calibration_file_.convert_Pose_A_to_Pose_B(loaded_pose);
			vw->set_model_position_at_index(selected[i].row(),
				relative_B_pose.x,
				relative_B_pose.y,
				relative_B_pose.z);
			vw->set_model_orientation_at_index(selected[i].row(),
				relative_B_pose.xa,
				relative_B_pose.ya,
				relative_B_pose.za);
				
			/*Text Actor if On*/
			if (actor_text->GetTextProperty()->GetOpacity() > 0.5) {
				vw->set_actor_text(vw->print_location_and_orientation_of_model_at_index(selected[i].row()));
				vw->set_actor_text_color_to_model_color_at_index(selected[i].row());
			}
		}
	}
	/*Update qvtkWidget*/
	ui.qvtk_widget->update();
	ui.qvtk_widget->renderWindow()->Render();
}

/*Make Selected Actor Principal from VTK*/
void MainScreen::VTKMakePrincipalSignal(vtkActor* new_principal_actor) {
	/*Get Model Actor*/
	int index_new_principal = -1;
	for (int i = 0; vw->model_actor_list_size(); i++) {
		if (vw->get_model_actor_at_index(i) == new_principal_actor) {
			index_new_principal = i;
			break;
		}
	}
	if (index_new_principal == -1) {
		QMessageBox::critical(this, "Error!", "Couldn't find model index to make principal!", QMessageBox::Ok);
		return;
	}

	/*Get Selected Indices on Model List Widget*/
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	/*No Point Swapping if just one*/
	if (selected.size() <= 1) {
		return;
	}

	for (int i = 0; i < selected.size(); i++) {
		if (selected[i].row() != index_new_principal) {
			ui.model_list_widget->item(selected[i].row())->setSelected(false);
		}
	}
	for (int i = 0; i < selected.size(); i++) {
		if (selected[i].row() != index_new_principal) {
			ui.model_list_widget->item(selected[i].row())->setSelected(true);
		}
	}
	if (ui.original_model_radio_button->isChecked() == true) {
		vw->change_model_opacity_to_original(index_new_principal);
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
	else if (ui.solid_model_radio_button->isChecked() == true) {
		vw->change_model_opacity_to_solid(index_new_principal);
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
	else if (ui.transparent_model_radio_button->isChecked() == true) {
		vw->change_model_opacity_to_transparent(index_new_principal);
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
	else if (ui.wireframe_model_radio_button->isChecked()) {
		vw->change_model_opacity_to_wire_frame(index_new_principal);
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}

	/*If Camera A View*/
	if (ui.camera_A_radio_button->isChecked()) {
		/*Set Model Pose*/
		Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentIndex().row(), index_new_principal);
		model_actor_list[index_new_principal]->SetPosition(loaded_pose.x, loaded_pose.y, loaded_pose.z);
		model_actor_list[index_new_principal]->SetOrientation(loaded_pose.xa, loaded_pose.ya, loaded_pose.za);

		/*Text Actor if On */
		if (actor_text->GetTextProperty()->GetOpacity() > 0.5) {
			std::string infoText = "Location: <";
			infoText += std::to_string(
					static_cast<long double>(model_actor_list[index_new_principal]->GetPosition()[0])) + ","
				+ std::to_string(static_cast<long double>(model_actor_list[index_new_principal]->GetPosition()[1])) +
				","
				+ std::to_string(static_cast<long double>(model_actor_list[index_new_principal]->GetPosition()[2])) +
				">\nOrientation: <"
				+ std::to_string(static_cast<long double>(model_actor_list[index_new_principal]->GetOrientation()[0])) +
				","
				+ std::to_string(static_cast<long double>(model_actor_list[index_new_principal]->GetOrientation()[1])) +
				","
				+ std::to_string(static_cast<long double>(model_actor_list[index_new_principal]->GetOrientation()[2])) +
				">";
			actor_text->SetInput(infoText.c_str());
			actor_text->GetTextProperty()->SetColor(model_actor_list[index_new_principal]->GetProperty()->GetColor());
		}
	}
	else {
		/*Else, Camera B View*/
		/*Convert To relative Camera B Pose as storage is done in camera A coordinates and rotations*/
		Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentIndex().row(), index_new_principal);
		Point6D relative_B_pose = calibration_file_.convert_Pose_A_to_Pose_B(loaded_pose);
		model_actor_list[index_new_principal]->SetPosition(relative_B_pose.x, relative_B_pose.y, relative_B_pose.z);
		model_actor_list[index_new_principal]->SetOrientation(relative_B_pose.xa, relative_B_pose.ya,
		                                                      relative_B_pose.za);

		/*Text Actor if On*/
		if (actor_text->GetTextProperty()->GetOpacity() > 0.5) {
			/*Display In Terms of Camera A*/
			std::string infoText = "Location: <";
			infoText += std::to_string(static_cast<long double>(loaded_pose.x)) + ","
				+ std::to_string(static_cast<long double>(loaded_pose.y)) + ","
				+ std::to_string(static_cast<long double>(loaded_pose.z)) + ">\nOrientation: <"
				+ std::to_string(static_cast<long double>(loaded_pose.xa)) + ","
				+ std::to_string(static_cast<long double>(loaded_pose.ya)) + ","
				+ std::to_string(static_cast<long double>(loaded_pose.za)) + ">";
			actor_text->SetInput(infoText.c_str());
		}
	}
	/*Update qvtkWidget*/
	ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
}

/*Multiple Selection For Models Radio buttons*/
void MainScreen::on_single_model_radio_button_clicked() {
	/*Change Selection Mode*/
	ui.model_list_widget->setSelectionMode(QAbstractItemView::SingleSelection);

	/*If Multiple Selections Choose First One Selected*/
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	if (selected.size() > 0) {
		ui.model_list_widget->setCurrentIndex(selected[0]);
	}
};

void MainScreen::on_multiple_model_radio_button_clicked() {
	ui.model_list_widget->setSelectionMode(QAbstractItemView::MultiSelection);
}

/*Image View Radio Buttons*/
/*Display Original Image*/
void MainScreen::on_original_image_radio_button_clicked() {
	/*Only Do Something if Loaded Frames, Skip*/
	if (ui.image_list_widget->currentRow() >= 0) {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_original_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_original_image(this->curr_frame(), false);
		}
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
}

/*Display Inverted Image*/
void MainScreen::on_inverted_image_radio_button_clicked() {
	/*Only Do Something if Loaded Frames, Skip*/
	if (ui.image_list_widget->currentRow() >= 0) {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_inverted_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_inverted_image(this->curr_frame(), false);
		}
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
}

/*Display Edge Detected Image*/
void MainScreen::on_edges_image_radio_button_clicked() {
	/*Only Do Something if Loaded Frames, Skip*/
	if (ui.image_list_widget->currentRow() >= 0) {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_edge_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_edge_image(this->curr_frame(), false);
		}
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
}

/*Display Dilated Image*/
void MainScreen::on_dilation_image_radio_button_clicked() {
	/*Only Do Something if Loaded Frames, Skip*/
	if (ui.image_list_widget->currentRow() >= 0) {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_dilation_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_dilation_image(this->curr_frame(), false);
		}
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
}

void MainScreen::on_original_model_radio_button_clicked() {
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	for (int i = 0; i < selected.size(); i++) {
		if (ui.original_model_radio_button->isChecked() == true) {
			vw->change_model_opacity_to_original(selected[i].row());
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
	}
}

void MainScreen::on_solid_model_radio_button_clicked() {
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	for (int i = 0; i < selected.size(); i++) {
		if (ui.solid_model_radio_button->isChecked() == true) {
			vw->change_model_opacity_to_solid(selected[i].row());
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
	}
}

void MainScreen::on_transparent_model_radio_button_clicked() {
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	for (int i = 0; i < selected.size(); i++) {
		if (ui.transparent_model_radio_button->isChecked() == true) {
			vw->change_model_opacity_to_transparent(selected[i].row());
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
	}
}

void MainScreen::on_wireframe_model_radio_button_clicked() {
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	Model model = loaded_models[selected[0].row()];
	int frame_idx = ui.image_list_widget->selectionModel()->selectedRows()[0].row();
	Point6D point6d = model_locations_.GetPose(frame_idx, selected[0].row());
	Pose pose = Pose(point6d.x, point6d.y, point6d.z, point6d.xa, point6d.ya, point6d.za);
	Frame frame = loaded_frames[frame_idx];
	auto gpu_frame = new GPUDilatedFrame(frame.GetDilationImage().cols,
		frame.GetDilationImage().cols,
		0,
		frame.GetDilationImage().data,
		4);

	GPUModel* gpu_mod = new GPUModel(model.model_name_,
		true,
		1024,
		1024,
		0,
		true,
		&model.triangle_vertices_[0],
		&model.triangle_normals_[0],
		model.triangle_vertices_.size() / 9,
		calibration_file_.camera_A_principal_);
	if (gpu_mod->IsInitializedCorrectly()) {
		gpu_mod->RenderPrimaryCamera(pose);
		if (gpu_mod->RenderPrimaryCamera(pose)) {
			std::cout << "Rendered okay" << std::endl;
			if (!gpu_mod->WritePrimaryCameraRenderedImage("test_image.tif")) {
				gpu_frame->GetGPUImage()->WriteImage("x_ray_img.tif");
				if (!gpu_frame->IsInitializedCorrectly()) {
					std::cout << "AYAYAYAYYYAY" << std::endl;
				}
			}

		}
	}
	delete gpu_frame;
	delete gpu_mod;

	std::cout << ui.qvtk_widget->width() << ", " << ui.qvtk_widget->height() << std::endl;
	for (int i = 0; i < selected.size(); i++) {
		if (ui.wireframe_model_radio_button->isChecked() == true) {
			vw->change_model_opacity_to_wire_frame(selected[i].row());
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
	}
}

/*KeyPress Event*/
void MainScreen::keyPressEvent(QKeyEvent* event) {
	/*Stop Optimizer*/
	if (event->key() == Qt::Key_Escape) {
		if (ui.actionStop_Optimizer->isEnabled() == true) {
			emit StopOptimizer();
			QMessageBox::warning(this, "Warning!", "Optimizer stopped!", QMessageBox::Ok);
		}
	}
	/*Toggle Information View*/
	if (event->key() == Qt::Key_I) {
		if (actor_text->GetTextProperty()->GetOpacity() > 0.5) {
			actor_text->GetTextProperty()->SetOpacity(0.0);
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
		else {
			actor_text->GetTextProperty()->SetOpacity(1.0);
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
	}
}

void MainScreen::VTKEscapeSignal() {
	if (ui.actionStop_Optimizer->isEnabled() == true) {
		emit StopOptimizer();
		QMessageBox::warning(this, "Warning!", "Optimizer stopped!", QMessageBox::Ok);
	}
}

/*Edge Detection Buttons*/
void MainScreen::on_aperture_spin_box_valueChanged() {
	/*Make Sure Images Loaded First*/
	if (loaded_frames.size() > 0) {

		/*Get High Value from Frame*/
		int low_val = LOW_THRESH;
		int high_val = HIGH_THRESH;
		if (ui.camera_A_radio_button->isChecked()) {
			low_val = loaded_frames[ui.image_list_widget->currentIndex().row()].GetLowThreshold();
			high_val = loaded_frames[ui.image_list_widget->currentIndex().row()].GetHighThreshold();
		}
		else if (ui.camera_B_radio_button->isChecked() && calibrated_for_biplane_viewport_) {
			low_val = loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetLowThreshold();
			high_val = loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetHighThreshold();
		}

		/*If TRUNK is Has Integer Parameter called Dilation, Update Dilation Values for Viewing Purposes*/
		int dilation_val = 0;
		std::vector<jta_cost_function::Parameter<int>> active_int_params = trunk_manager_.getActiveCostFunctionClass()->
			getIntParameters();
		for (int i = 0; i < active_int_params.size(); i++) {
			if (active_int_params[i].getParameterName() == "Dilation") {
				dilation_val = trunk_manager_.getActiveCostFunctionClass()->getIntParameters().at(i).
				                              getParameterValue();
			}
		}
		if (dilation_val < 0) {
			dilation_val = 0;
		}

		if (ui.image_list_widget->currentIndex().row() >= 0 && ui.image_list_widget->currentIndex().row() <
			loaded_frames.size()) {
			if (ui.camera_A_radio_button->isChecked()) {
				loaded_frames[ui.image_list_widget->currentIndex().row()].SetEdgeImage(ui.aperture_spin_box->value(),
					low_val,
					high_val);
				loaded_frames[ui.image_list_widget->currentIndex().row()].SetDilatedImage(dilation_val);
			}
			else if (ui.camera_B_radio_button->isChecked() && calibrated_for_biplane_viewport_) {
				loaded_frames_B[ui.image_list_widget->currentIndex().row()].SetEdgeImage(ui.aperture_spin_box->value(),
					low_val,
					high_val);
				loaded_frames_B[ui.image_list_widget->currentIndex().row()].SetDilatedImage(dilation_val);
			}
		}
		/*If Edge View Selected*/
		if (ui.image_list_widget->currentIndex().row() >= 0 && ui.edges_image_radio_button->isChecked() == true) {
			if (ui.camera_A_radio_button->isChecked()) {
				vw->update_display_background_to_edge_image(this->curr_frame(), true);
			}
			else {
				vw->update_display_background_to_edge_image(this->curr_frame(), false);
			}
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
		/*If Dilation View Selected*/
		if (ui.image_list_widget->currentIndex().row() >= 0 && ui.dilation_image_radio_button->isChecked() == true) {
			if (ui.camera_A_radio_button->isChecked()) {
				vw->update_display_background_to_dilation_image(this->curr_frame(), true);
			}
			else {
				vw->update_display_background_to_dilation_image(this->curr_frame(), false);
			}
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
		/*Save To Optimizer Settings and Registry*/
		QString Version = "Version" + QString::number(VER_FIRST_NUM) +
			QString::number(VER_MIDDLE_NUM) + QString::number(VER_LAST_NUM);
		QSettings setting("JointTrackAutoGPU", Version);
		setting.beginGroup("EdgeDetectionSettings");
		setting.setValue("APERTURE", ui.aperture_spin_box->value());
		setting.setValue("LOW_THRESH", low_val);
		setting.setValue("HIGH_THRESH", high_val);
		setting.endGroup();
	}
};

void MainScreen::on_low_threshold_slider_valueChanged() {
	/*Set Label Text*/
	ui.low_threshold_value->setText(QString::number(ui.low_threshold_slider->value()));

	/*Make Sure Images Loaded First*/
	if (loaded_frames.size() > 0) {

		/*Get High Value from Frame*/
		int aperture = APERTURE;
		int high_val = HIGH_THRESH;
		if (ui.camera_A_radio_button->isChecked()) {
			aperture = loaded_frames[ui.image_list_widget->currentIndex().row()].GetAperture();
			high_val = loaded_frames[ui.image_list_widget->currentIndex().row()].GetHighThreshold();
		}
		else if (ui.camera_B_radio_button->isChecked() && calibrated_for_biplane_viewport_) {
			aperture = loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetAperture();
			high_val = loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetHighThreshold();
		}

		/*If TRUNK is Has Integer Parameter called Dilation, Update Dilation Values for Viewing Purposes*/
		int dilation_val = 0;
		std::vector<jta_cost_function::Parameter<int>> active_int_params = trunk_manager_.getActiveCostFunctionClass()->
			getIntParameters();
		for (int i = 0; i < active_int_params.size(); i++) {
			if (active_int_params[i].getParameterName() == "Dilation") {
				dilation_val = trunk_manager_.getActiveCostFunctionClass()->getIntParameters().at(i).
				                              getParameterValue();
			}
		}
		if (dilation_val < 0) {
			dilation_val = 0;
		}

		if (ui.image_list_widget->currentIndex().row() >= 0 && ui.image_list_widget->currentIndex().row() <
			loaded_frames.size()) {
			if (ui.camera_A_radio_button->isChecked()) {
				loaded_frames[ui.image_list_widget->currentIndex().row()].SetEdgeImage(aperture,
					ui.low_threshold_slider->value(),
					high_val);
				loaded_frames[ui.image_list_widget->currentIndex().row()].SetDilatedImage(dilation_val);
			}
			else if (ui.camera_B_radio_button->isChecked() && calibrated_for_biplane_viewport_) {
				loaded_frames_B[ui.image_list_widget->currentIndex().row()].SetEdgeImage(aperture,
					ui.low_threshold_slider->value(),
					high_val);
				loaded_frames_B[ui.image_list_widget->currentIndex().row()].SetDilatedImage(dilation_val);
			}
		}
		/*If Edge View Selected*/
		if (ui.image_list_widget->currentIndex().row() >= 0 && ui.edges_image_radio_button->isChecked() == true) {
			if (ui.camera_A_radio_button->isChecked()) {
				vw->update_display_background_to_edge_image(this->curr_frame(), true);
			}
			else {
				vw->update_display_background_to_edge_image(this->curr_frame(), false);
			}
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
		/*If Dilation View Selected*/
		if (ui.image_list_widget->currentIndex().row() >= 0 && ui.dilation_image_radio_button->isChecked() == true) {
			if (ui.camera_A_radio_button->isChecked()) {
				vw->update_display_background_to_dilation_image(this->curr_frame(), true);
			}
			else {
				vw->update_display_background_to_dilation_image(this->curr_frame(), true);
			}
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
		/*Save To Optimizer Settings and Registry*/
		QString Version = "Version" + QString::number(VER_FIRST_NUM) +
			QString::number(VER_MIDDLE_NUM) + QString::number(VER_LAST_NUM);
		QSettings setting("JointTrackAutoGPU", Version);
		setting.beginGroup("EdgeDetectionSettings");
		setting.setValue("APERTURE", aperture);
		setting.setValue("LOW_THRESH", ui.low_threshold_slider->value());
		setting.setValue("HIGH_THRESH", high_val);
		setting.endGroup();
	}

};

void MainScreen::on_high_threshold_slider_valueChanged() {
	/*Set Label Text*/
	ui.high_threshold_value->setText(QString::number(ui.high_threshold_slider->value()));

	/*Make Sure Images Loaded First*/
	if (loaded_frames.size() > 0) {

		/*Get Low Value from Frame*/
		int aperture = APERTURE;
		int low_val = LOW_THRESH;
		if (ui.camera_A_radio_button->isChecked()) {
			aperture = loaded_frames[ui.image_list_widget->currentIndex().row()].GetAperture();
			low_val = loaded_frames[ui.image_list_widget->currentIndex().row()].GetLowThreshold();
		}
		else if (ui.camera_B_radio_button->isChecked() && calibrated_for_biplane_viewport_) {
			aperture = loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetAperture();
			low_val = loaded_frames_B[ui.image_list_widget->currentIndex().row()].GetLowThreshold();
		}

		/*If TRUNK is Has Integer Parameter called Dilation, Update Dilation Values for Viewing Purposes*/
		int dilation_val = 0;
		std::vector<jta_cost_function::Parameter<int>> active_int_params = trunk_manager_.getActiveCostFunctionClass()->
			getIntParameters();
		for (int i = 0; i < active_int_params.size(); i++) {
			if (active_int_params[i].getParameterName() == "Dilation") {
				dilation_val = trunk_manager_.getActiveCostFunctionClass()->getIntParameters().at(i).
				                              getParameterValue();
			}
		}
		if (dilation_val < 0) {
			dilation_val = 0;
		}

		if (ui.image_list_widget->currentIndex().row() >= 0 && ui.image_list_widget->currentIndex().row() <
			loaded_frames.size()) {
			if (ui.camera_A_radio_button->isChecked()) {
				loaded_frames[ui.image_list_widget->currentIndex().row()].SetEdgeImage(aperture,
					low_val,
					ui.high_threshold_slider->value());
				loaded_frames[ui.image_list_widget->currentIndex().row()].SetDilatedImage(dilation_val);
			}
			else if (ui.camera_B_radio_button->isChecked() && calibrated_for_biplane_viewport_) {
				loaded_frames_B[ui.image_list_widget->currentIndex().row()].SetEdgeImage(aperture,
					low_val,
					ui.high_threshold_slider->value());
				loaded_frames_B[ui.image_list_widget->currentIndex().row()].SetDilatedImage(dilation_val);
			}
		}
		/*If Edge View Selected*/
		if (ui.image_list_widget->currentIndex().row() >= 0 && ui.edges_image_radio_button->isChecked() == true) {
			if (ui.camera_A_radio_button->isChecked()) {
				vw->update_display_background_to_edge_image(this->curr_frame(), true);
			}
			else {
				vw->update_display_background_to_edge_image(this->curr_frame(), false);
			}
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
		/*If Dilation View Selected*/
		if (ui.image_list_widget->currentIndex().row() >= 0 && ui.dilation_image_radio_button->isChecked() == true) {
			if (ui.camera_A_radio_button->isChecked()) {
				vw->update_display_background_to_dilation_image(this->curr_frame(), true);
			}
			else {
				vw->update_display_background_to_dilation_image(this->curr_frame(), false);
			}
			ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
		}
		/*Save To Optimizer Settings and Registry*/
		QString Version = "Version" + QString::number(VER_FIRST_NUM) +
			QString::number(VER_MIDDLE_NUM) + QString::number(VER_LAST_NUM);
		QSettings setting("JointTrackAutoGPU", Version);
		setting.beginGroup("EdgeDetectionSettings");
		setting.setValue("APERTURE", aperture);
		setting.setValue("LOW_THRESH", low_val);
		setting.setValue("HIGH_THRESH", ui.high_threshold_slider->value());
		setting.endGroup();
	}
};
/*Apply All Edges*/
void MainScreen::on_apply_all_edge_button_clicked() {
	/*If TRUNK is Has Integer Parameter called Dilation, Update Dilation Values for Viewing Purposes*/
	int dilation_val = 0;
	std::vector<jta_cost_function::Parameter<int>> active_int_params = trunk_manager_.getActiveCostFunctionClass()->
		getIntParameters();
	for (int i = 0; i < active_int_params.size(); i++) {
		if (active_int_params[i].getParameterName() == "Dilation") {
			dilation_val = trunk_manager_.getActiveCostFunctionClass()->getIntParameters().at(i).getParameterValue();
		}
	}
	if (dilation_val < 0) {
		dilation_val = 0;
	}

	/*Apply Edge Detect to All Images*/
	for (int i = 0; i < loaded_frames.size(); i++) {
		loaded_frames[i].SetEdgeImage(ui.aperture_spin_box->value(),
		                              ui.low_threshold_slider->value(),
		                              ui.high_threshold_slider->value());
		loaded_frames[i].SetDilatedImage(dilation_val);
	}
	if (calibrated_for_biplane_viewport_) {
		for (int i = 0; i < loaded_frames_B.size(); i++) {
			loaded_frames_B[i].SetEdgeImage(ui.aperture_spin_box->value(),
			                                ui.low_threshold_slider->value(),
			                                ui.high_threshold_slider->value());
			loaded_frames_B[i].SetDilatedImage(dilation_val);
		}
	}
	/*If Edge View Selected*/
	if (ui.image_list_widget->currentIndex().row() >= 0 && ui.edges_image_radio_button->isChecked() == true) {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_edge_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_edge_image(this->curr_frame(), false);
		}
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
	/*If Dilation View Selected*/
	if (ui.image_list_widget->currentIndex().row() >= 0 && ui.dilation_image_radio_button->isChecked() == true) {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_dilation_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_dilation_image(this->curr_frame(), false);
		}
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
	/*Save To Optimizer Settings and Registry*/
	QString Version = "Version" + QString::number(VER_FIRST_NUM) +
		QString::number(VER_MIDDLE_NUM) + QString::number(VER_LAST_NUM);
	QSettings setting("JointTrackAutoGPU", Version);
	setting.beginGroup("EdgeDetectionSettings");
	setting.setValue("APERTURE", ui.aperture_spin_box->value());
	setting.setValue("LOW_THRESH", ui.low_threshold_slider->value());
	setting.setValue("HIGH_THRESH", ui.high_threshold_slider->value());
	setting.endGroup();
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
	/*Only Re-enable load calibration if for some reason neither are clibrated (don't know how this would ever happen)...*/
	if (calibrated_for_monoplane_viewport_ == false && calibrated_for_biplane_viewport_ == false) {
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
}

/*NON GUI FUNCTIONS*/
/*Save Last Pose (Do this when optimizing or when chaninging the list widgets*/
void MainScreen::SaveLastPose() {
	/*Save Last Pair Pose*/
	if (previous_model_indices_.size() > 0 && previous_frame_index_ != -1) {
		for (int i = 0; i < previous_model_indices_.size(); i++) {
			double* position_curr = vw->get_model_position_at_index(previous_model_indices_[i].row());
			double* orientation_curr = vw->get_model_orientation_at_index(previous_model_indices_[i].row());
			Point6D last_pose(position_curr[0], position_curr[1], position_curr[2],
			                  orientation_curr[0], orientation_curr[1], orientation_curr[2]);
			/*If Camera B View, Save in Camera A coordinates*/
			if (ui.camera_A_radio_button->isChecked()) {
				model_locations_.SavePose(previous_frame_index_, previous_model_indices_[i].row(), last_pose);
			}
			else {
				model_locations_.SavePose(previous_frame_index_, previous_model_indices_[i].row(),
				                          calibration_file_.convert_Pose_B_to_Pose_A(last_pose));
			}
		}
	}
}

/*Optimization Function: Packages Off The Optimization process in
a new thread*/
/*Launch Optimizer*/
void MainScreen::LaunchOptimizer(QString directive) {
	/*Save Last Pair Pose*/
	SaveLastPose();
	int iter_count;

	if (directive == "Sym_Trap") {
		sym_trap_running = true;
		//iter_count = sym_trap_control->getIterCount();
	}
	else {
		iter_count = 0;
	}
	/*Can Only Optimize If Chosen Frame and Model*/
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	if (selected.size() == 0 || previous_frame_index_ < 0 || ui.image_list_widget->currentIndex().row() !=
		previous_frame_index_ ||
		ui.image_list_widget->currentIndex().row() >= loaded_frames.size() ||
		ui.model_list_widget->currentIndex().row() >= loaded_models.size()) {
		QMessageBox::critical(this, "Error!", "Select Frame and Model First!", QMessageBox::Ok);
		return;
	}

	/*Check Frame List by Model List and Guess Matrix Dimensions are the Same Size*/
	if (model_locations_.GetFrameCount() != loaded_frames.size() ||
		model_locations_.GetModelCount() != loaded_models.size()) {
		QMessageBox::critical(this, "Critical Error!",
		                      "Pose Dimension Matrix Differs in Size from Frame and Models Loaded! Please Contact Support!",
		                      QMessageBox::Ok);
		return;
	}

	/*Initialize Thread*/
	/*Set Up Connections*/
	//Master Thread to Carry Optimizer
	optimizer_manager = new OptimizerManager(); //Create Master
	optimizer_thread = new QThread(); //Create QThread
	optimizer_manager->moveToThread(optimizer_thread); //Move Master to QThread

	/*Send the Following Information to the Optimizer Thread:
	0). Calibration Class
	1). Frame List(s)
	2). Selected Models List and Primary Model
	3). Current Pose Matrix
	4). Optimizer Settings Class
	5). The Three Cost Function Manager Classes
	6). Optimization Directives (All, From, Each, Single)
	7). Error Message Reference*/
	/*Initialze the Optimizer by Sending All the Previous Information and Setting Up/Checking the CUDA Connecitons*/
	QString error_mess;
	bool initialized_correctly = optimizer_manager->Initialize(
		*optimizer_thread,
		calibration_file_,
		loaded_frames, loaded_frames_B, ui.image_list_widget->currentIndex().row(),
		loaded_models, selected, selected[0].row(),
		model_locations_,
		optimizer_settings_,
		trunk_manager_, branch_manager_, leaf_manager_,
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
	connect(optimizer_manager, SIGNAL(UpdateDisplay(double, int, double, unsigned int)), this,
	        SLOT(onUpdateDisplay(double, int, double, unsigned int))); //Update Display
	connect(optimizer_manager, SIGNAL(OptimizerError(QString)), this, SLOT(onOptimizerError(QString)));
	//Optimizer Error Check
	connect(optimizer_manager, SIGNAL(UpdateOptimum(double, double, double, double, double, double, unsigned int)),
	        this, SLOT(onUpdateOptimum(double, double, double, double, double, double, unsigned int)));
	// Update Guess Connection
	connect(optimizer_manager,
	        SIGNAL(OptimizedFrame(double, double, double, double, double, double, bool, unsigned int, bool, QString)),
	        this, SLOT(
		        onOptimizedFrame(double, double, double, double, double, double, bool, unsigned int, bool, QString)));
	// Update Optimized Frame/View
	connect(this, SIGNAL(StopOptimizer()), optimizer_manager, SLOT(onStopOptimizer()), Qt::DirectConnection);
	/*Stops Optimizer*/
	connect(optimizer_manager, SIGNAL(UpdateDilationBackground()), this, SLOT(onUpdateDilationBackground()));
	/*UPDATE DILATION BACKGROUND	*/
	connect(optimizer_manager, SIGNAL(onUpdateOrientationSymTrap(double, double, double, double, double, double)), this,
	        SLOT(updateOrientationSymTrap_MS(double, double, double, double, double, double)));

	// Connect sym trap progress bar to thread
	//connect(optimizer_manager, SIGNAL(onProgressBarUpdate(int)), sym_trap_control->ui.progressBar, SLOT(setValue(int)));


	/*Start*/
	if (directive == "Each" || directive == "All") {
		ui.image_list_widget->setCurrentRow(0);
	}
	actor_text->GetTextProperty()->SetColor(214.0 / 255.0, 108.0 / 255.0, 35.0 / 255.0); // Set Orange;
	currently_optimizing_ = true;
	DisableAll();
	display_optimizer_settings_ = optimizer_settings_;
	optimizer_thread->start();
}

void MainScreen::updateOrientationSymTrap_MS(double x, double y, double z, double xa, double ya, double za) {
	//put the update logic here
	// look at loading kinematics for help
	// or copy pose
	// need to update the current model
	Point6D new_orientation(x, y, z, xa, ya, za);
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
	model_locations_.SavePose(ui.image_list_widget->currentRow(), ui.model_list_widget->currentRow(), new_orientation);
	vw->set_model_position_at_index(selected[0].row(), new_orientation.x, new_orientation.y, new_orientation.z);
	vw->set_model_orientation_at_index(selected[0].row(), new_orientation.xa, new_orientation.ya, new_orientation.za);
	ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
}

/*OPTIMIZATION
*/
/*Update Blue Current Optimum*/
void MainScreen::onUpdateOptimum(double x, double y, double z, double xa, double ya, double za,
                                 unsigned int primary_model_index) {
	/*Update Blue's Location*/
	auto CurrentPose = Point6D(x, y, z, xa, ya, za);
	if (ui.camera_B_radio_button->isChecked() == true) {
		CurrentPose = calibration_file_.convert_Pose_A_to_Pose_B(CurrentPose);
	}
	if (primary_model_index < loaded_models.size()) { //TODO: Find a better way to represent this
		vw->set_model_position_at_index(primary_model_index, CurrentPose.x, CurrentPose.y, CurrentPose.z);
		vw->set_model_orientation_at_index(primary_model_index, CurrentPose.xa, CurrentPose.ya, CurrentPose.za);
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}

}

/*Finished Optimizing Frame, Send Optimum to MainScreen*/
void MainScreen::onOptimizedFrame(double x, double y, double z, double xa, double ya, double za, bool move_next_frame,
                                  unsigned int primary_model_index, bool error_occurred, QString optimizer_directive) {
	/*Update Actor*/
	auto CurrentPose = Point6D(x, y, z, xa, ya, za);
	if (ui.camera_B_radio_button->isChecked() == true) {
		CurrentPose = calibration_file_.convert_Pose_A_to_Pose_B(CurrentPose);
	}
	if (primary_model_index < loaded_models.size()) { //todo: Find a better way to get size of model_actor_list
		vw->set_model_position_at_index(primary_model_index, CurrentPose.x, CurrentPose.y, CurrentPose.z);
		vw->set_model_orientation_at_index(primary_model_index, CurrentPose.xa, CurrentPose.ya, CurrentPose.za);
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
	else {
		/*Display Finished*/
		QMessageBox::critical(this, "Error!", "Model index out of bounds!", QMessageBox::Ok);
		/*Program Will Crash after this message but that is fine, this should never ever occurr...*/
	}

	/*Save Indices*/
	int current_frame_index = ui.image_list_widget->currentIndex().row();

	if (optimizer_directive == "Backward") {
		if (move_next_frame && current_frame_index > 0) {
			ui.image_list_widget->setCurrentRow(current_frame_index - 1);
			model_locations_.SavePose(current_frame_index, primary_model_index,
			                          Point6D(x, y, z, xa, ya, za));
		}
		else {
			/*Save Pose To Storage*/
			model_locations_.SavePose(current_frame_index, primary_model_index,
			                          Point6D(x, y, z, xa, ya, za));
			/*Not Currently Optimzing*/
			currently_optimizing_ = false;
			EnableAll();
			/*Display Finished*/
			if (!error_occurred) {
				QMessageBox::information(this, "Finished!", "All frames optimized!", QMessageBox::Ok);
			}
		}

	}
	else {
		/*If Commanded to Move To Next Frame Do So*/
		if (move_next_frame && current_frame_index + 1 < ui.image_list_widget->count()) {
			/*Bring Up Next Frame*/
			ui.image_list_widget->setCurrentRow(current_frame_index + 1);
			/*Save Pose To Storage*/
			model_locations_.SavePose(current_frame_index, primary_model_index,
			                          Point6D(x, y, z, xa, ya, za));
		}
		else {
			/*Save Pose To Storage*/
			model_locations_.SavePose(current_frame_index, primary_model_index,
			                          Point6D(x, y, z, xa, ya, za));
			/*Not Currently Optimzing*/
			currently_optimizing_ = false;
			EnableAll();
			/*Display Finished*/
			if (!error_occurred && !sym_trap_running) {
				QMessageBox::information(this, "Finished!", "All frames optimized!", QMessageBox::Ok);
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
void MainScreen::onUpdateDisplay(double iteration_speed, int current_iteration, double current_minimum,
                                 unsigned int primary_model_index) {

	div_t divresult;
	divresult = div(
		static_cast<int>(static_cast<double>(display_optimizer_settings_.trunk_budget + display_optimizer_settings_.
			enable_branch_ *
			display_optimizer_settings_.number_branches * display_optimizer_settings_.branch_budget +
			display_optimizer_settings_.enable_leaf_ * display_optimizer_settings_.leaf_budget
			- current_iteration) / (1000.0 / iteration_speed)), 60);
	std::stringstream ss;
	ss << std::setfill('0') << std::setw(2) << divresult.rem;
	std::string infoText = "Optimum Location: <";
	std::stringstream level;
	if (current_iteration < display_optimizer_settings_.trunk_budget) {
		level << "Trunk";
	}
	else if (current_iteration < display_optimizer_settings_.trunk_budget + display_optimizer_settings_.enable_branch_ *
		display_optimizer_settings_.number_branches * display_optimizer_settings_.branch_budget) {
		div_t divresultLevel;
		divresultLevel = div(current_iteration - display_optimizer_settings_.trunk_budget,
		                     display_optimizer_settings_.branch_budget);
		level << "Branch " << divresultLevel.quot + 1;
	}
	else if (current_iteration < display_optimizer_settings_.trunk_budget + display_optimizer_settings_.enable_branch_ *
		display_optimizer_settings_.number_branches * display_optimizer_settings_.branch_budget +
		display_optimizer_settings_.enable_leaf_ * display_optimizer_settings_.leaf_budget) {
		level << "Extra Z-Translation";
	}
	else {
		level << "Finished";
	}
	auto current_position = vw->get_model_position_at_index(primary_model_index);
	auto current_orientation = vw->get_model_orientation_at_index(primary_model_index);
	auto CurrentPose = Point6D(current_position[0], current_position[1], current_position[2],
		current_orientation[0], current_orientation[1], current_orientation[2]);
	if (ui.camera_B_radio_button->isChecked() == true) {
		CurrentPose = calibration_file_.convert_Pose_B_to_Pose_A(CurrentPose);
	}

	infoText += std::to_string(static_cast<long double>(CurrentPose.x)) + ","
		+ std::to_string(static_cast<long double>(CurrentPose.y)) + ","
		+ std::to_string(static_cast<long double>(CurrentPose.z)) + ">\nOptimum Orientation: <"
		+ std::to_string(static_cast<long double>(CurrentPose.xa)) + ","
		+ std::to_string(static_cast<long double>(CurrentPose.ya)) + ","
		+ std::to_string(static_cast<long double>(CurrentPose.za)) + ">\nMinimum Function Value: "
		+ std::to_string(static_cast<long double>(current_minimum)) + "\nIterations Per Second: "
		+ std::to_string(static_cast<long double>(1000.0) / iteration_speed) + "\nIteration Count: "
		+ std::to_string(static_cast<long long>(current_iteration)) + "\nSearch Level: "
		+ level.str() + "\nEstimated Time Remaining for Frame: <"
		+ std::to_string(static_cast<long long>(divresult.quot)) + ":" + ss.str() + ">";
	actor_text->SetInput(infoText.c_str());
	actor_text->GetTextProperty()->SetColor(214.0 / 255.0, 108.0 / 255.0, 35.0 / 255.0);

	/*update qvtk*/
	ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
}

/*Update Background if Dilation Selected and Moving From Trunk to Branch OR Branch to Z Search*/
void MainScreen::onUpdateDilationBackground() {
	if (ui.dilation_image_radio_button->isChecked() == true) {
		if (ui.camera_A_radio_button->isChecked() == true) {
			vw->update_display_background_to_dilation_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_dilation_image(this->curr_frame(), false);
		}
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
}


/*Function to load settings from registry and also check if First Time Loading*/
void MainScreen::LoadSettingsBetweenSessions() {
	/*Check if Loaded Before*/
	QString Version = "Version" + QString::number(VER_FIRST_NUM) +
		QString::number(VER_MIDDLE_NUM) + QString::number(VER_LAST_NUM);
	QSettings setting("JointTrackAutoGPU", Version);
	bool first_time_loading = false;
	QStringList groupList = setting.childGroups();
	if (groupList.size() == 0) {
		first_time_loading = true;
	}

	/*Not First Time Loading*/
	if (!first_time_loading) {
		/*Save Cost Function Settings*/
		setting.beginGroup("CostFunctionSettings");

		/*Get list of all keys and split into terminology*/
		QStringList cost_function_settings_keys = setting.allKeys();
		for (int i = 0; i < cost_function_settings_keys.size(); i++) {
			/*If 2 codes, should be the STAGE and ACTIVE_CF.
			If 4 codes, should be the STAGE, Cost Function Name, Parameter Name, Parameter Type*/
			QStringList key_codes = cost_function_settings_keys[i].split("@");
			if (key_codes.size() == 2 && key_codes[1] == "ACTIVE_CF") {
				if (key_codes[0] == "TRUNK") {
					trunk_manager_.setActiveCostFunction(
						setting.value(cost_function_settings_keys[i]).toString().toStdString());
				}
				else if (key_codes[0] == "BRANCH") {
					branch_manager_.setActiveCostFunction(
						setting.value(cost_function_settings_keys[i]).toString().toStdString());
				}
				else if (key_codes[0] == "LEAF") {
					leaf_manager_.setActiveCostFunction(
						setting.value(cost_function_settings_keys[i]).toString().toStdString());
				}
				else {
					QMessageBox::critical(this, "Error", "Error in key registry! Code A", QMessageBox::Ok);
				}
			}
			else if (key_codes.size() == 4) {
				if (key_codes[0] == "TRUNK") {
					if (key_codes[3] == "DOUBLE") {
						trunk_manager_.getCostFunctionClass(key_codes[1].toStdString())->setDoubleParameterValue(
							key_codes[2].toStdString(),
							setting.value(cost_function_settings_keys[i]).toDouble());
					}
					else if (key_codes[3] == "INT") {
						trunk_manager_.getCostFunctionClass(key_codes[1].toStdString())->setIntParameterValue(
							key_codes[2].toStdString(),
							setting.value(cost_function_settings_keys[i]).toInt());
					}
					else if (key_codes[3] == "BOOL") {
						trunk_manager_.getCostFunctionClass(key_codes[1].toStdString())->setBoolParameterValue(
							key_codes[2].toStdString(),
							setting.value(cost_function_settings_keys[i]).toBool());
					}
					else {
						QMessageBox::critical(this, "Error", "Error in key registry! Code D", QMessageBox::Ok);
					}
				}
				else if (key_codes[0] == "BRANCH") {
					if (key_codes[3] == "DOUBLE") {
						branch_manager_.getCostFunctionClass(key_codes[1].toStdString())->setDoubleParameterValue(
							key_codes[2].toStdString(),
							setting.value(cost_function_settings_keys[i]).toDouble());
					}
					else if (key_codes[3] == "INT") {
						branch_manager_.getCostFunctionClass(key_codes[1].toStdString())->setIntParameterValue(
							key_codes[2].toStdString(),
							setting.value(cost_function_settings_keys[i]).toInt());
					}
					else if (key_codes[3] == "BOOL") {
						branch_manager_.getCostFunctionClass(key_codes[1].toStdString())->setBoolParameterValue(
							key_codes[2].toStdString(),
							setting.value(cost_function_settings_keys[i]).toBool());
					}
					else {
						QMessageBox::critical(this, "Error", "Error in key registry! Code E", QMessageBox::Ok);
					}
				}
				else if (key_codes[0] == "LEAF") {
					if (key_codes[3] == "DOUBLE") {
						leaf_manager_.getCostFunctionClass(key_codes[1].toStdString())->setDoubleParameterValue(
							key_codes[2].toStdString(),
							setting.value(cost_function_settings_keys[i]).toDouble());
					}
					else if (key_codes[3] == "INT") {
						leaf_manager_.getCostFunctionClass(key_codes[1].toStdString())->setIntParameterValue(
							key_codes[2].toStdString(),
							setting.value(cost_function_settings_keys[i]).toInt());
					}
					else if (key_codes[3] == "BOOL") {
						leaf_manager_.getCostFunctionClass(key_codes[1].toStdString())->setBoolParameterValue(
							key_codes[2].toStdString(),
							setting.value(cost_function_settings_keys[i]).toBool());
					}
					else {
						QMessageBox::critical(this, "Error", "Error in key registry! Code F", QMessageBox::Ok);
					}
				}
				else {
					QMessageBox::critical(this, "Error", "Error in key registry! Code B", QMessageBox::Ok);
				}

			}
			else {
				QMessageBox::critical(this, "Error", "Error in key registry! Code C", QMessageBox::Ok);
			}
		}
		setting.endGroup();

		/*Load Optimizer Settings*/
		setting.beginGroup("OptimizerSettings");
		/*Variables*/
		/*Trunk*/
		optimizer_settings_.trunk_range = Point6D(setting.value("TRUNK@RANGE_X").toDouble(),
		                                          setting.value("TRUNK@RANGE_Y").toDouble(),
		                                          setting.value("TRUNK@RANGE_Z").toDouble(),
		                                          setting.value("TRUNK@RANGE_XA").toDouble(),
		                                          setting.value("TRUNK@RANGE_YA").toDouble(),
		                                          setting.value("TRUNK@RANGE_ZA").toDouble());
		optimizer_settings_.trunk_budget = setting.value("TRUNK@BUDGET").toInt();

		/*Branch*/
		optimizer_settings_.branch_range = Point6D(setting.value("BRANCH@RANGE_X").toDouble(),
		                                           setting.value("BRANCH@RANGE_Y").toDouble(),
		                                           setting.value("BRANCH@RANGE_Z").toDouble(),
		                                           setting.value("BRANCH@RANGE_XA").toDouble(),
		                                           setting.value("BRANCH@RANGE_YA").toDouble(),
		                                           setting.value("BRANCH@RANGE_ZA").toDouble());
		optimizer_settings_.number_branches = setting.value("BRANCH@NUMBER_BRANCHES").toInt();
		optimizer_settings_.enable_branch_ = setting.value("BRANCH@ENABLE").toBool();
		optimizer_settings_.branch_budget = setting.value("BRANCH@BUDGET").toInt();

		/*Leaf*/
		optimizer_settings_.leaf_range = Point6D(setting.value("LEAF@RANGE_X").toDouble(),
		                                         setting.value("LEAF@RANGE_Y").toDouble(),
		                                         setting.value("LEAF@RANGE_Z").toDouble(),
		                                         setting.value("LEAF@RANGE_XA").toDouble(),
		                                         setting.value("LEAF@RANGE_YA").toDouble(),
		                                         setting.value("LEAF@RANGE_ZA").toDouble());
		optimizer_settings_.enable_leaf_ = setting.value("LEAF@ENABLE").toBool();
		optimizer_settings_.leaf_budget = setting.value("LEAF@BUDGET").toInt();
		setting.endGroup();

		/*Edge Detection Settings*/
		setting.beginGroup("EdgeDetectionSettings");
		ui.aperture_spin_box->setValue(setting.value("APERTURE").toInt());
		ui.low_threshold_slider->setValue(setting.value("LOW_THRESH").toInt());
		ui.high_threshold_slider->setValue(setting.value("HIGH_THRESH").toInt());
		setting.endGroup();
	}
	else {
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
			if (properties.major != 9999 && properties.major >= 5) /* 9999 means emulation only */
			{
				++gpu_device_count;
			}
		}
		/*If no Cuda Compatitble Devices with Compute Capability Greater Than 5, Exit*/
		if (gpu_device_count == 0) {
			if (device_count == 0) {
				QMessageBox::critical(this, "Error!", "No CUDA capable GPU detected! Optimizer will not run!",
				                      QMessageBox::Ok);
			}
			else if (properties.major == 9999) {
				QMessageBox::critical(this, "Error!", "GPU is emulation only! Optimizer will not run!",
				                      QMessageBox::Ok);
			}
			else {
				QMessageBox::critical(this, "Error!",
				                      "GPU does not have high enough compute capability! Optimizer will not run!\nPlease upgrade to device with compute capability 5.0 or higher!",
				                      QMessageBox::Ok);
			}
		}
		else {
			/*First Time Loading Message Will Now Go Away By Marking in Registry*/
			/*DEPRECATED BUT STILL IN THE CODE - WHATEVER*/
			setting.beginGroup("FirstTime");
			setting.setValue("JTAFirstTime", false);
			setting.endGroup();
		}

		/*Save Default Settings*/
		/*Default Optimizer Settings*/
		optimizer_settings_ = OptimizerSettings();

		/*Default 3 Cost Function Managers*/
		trunk_manager_ = jta_cost_function::CostFunctionManager(Stage::Trunk);
		branch_manager_ = jta_cost_function::CostFunctionManager(Stage::Branch);
		leaf_manager_ = jta_cost_function::CostFunctionManager(Stage::Leaf);

		/*Change the Default Settings of Dilation for branch and leaf to 4 and 1 respectively*/
		branch_manager_.getCostFunctionClass("DIRECT_DILATION")->setIntParameterValue("Dilation", 4);
		leaf_manager_.getCostFunctionClass("DIRECT_DILATION")->setIntParameterValue("Dilation", 1);

		/*Save to Registry*/
		QString Version = "Version" + QString::number(VER_FIRST_NUM) +
			QString::number(VER_MIDDLE_NUM) + QString::number(VER_LAST_NUM);
		QSettings setting("JointTrackAutoGPU", Version);

		/*Save Cost Function Settings*/
		setting.beginGroup("CostFunctionSettings");
		/*Cost Function Managers (Save All Values for Parameters and Active Cost Function*/
		/*Trunk*/
		setting.setValue("TRUNK@ACTIVE_CF", QString::fromStdString(trunk_manager_.getActiveCostFunction()));
		std::vector<jta_cost_function::CostFunction> trunk_cost_functions = trunk_manager_.getAvailableCostFunctions();
		for (int i = 0; i < trunk_cost_functions.size(); i++) {
			std::vector<jta_cost_function::Parameter<double>> trunk_parameters_double = trunk_cost_functions[i].
				getDoubleParameters();
			for (int j = 0; j < trunk_parameters_double.size(); j++) {
				setting.setValue(QString::fromStdString(
					                 "TRUNK@" + trunk_cost_functions[i].getCostFunctionName() + "@" +
					                 trunk_parameters_double[j].getParameterName()
					                 + "@" + trunk_parameters_double[j].getParameterType()),
				                 trunk_parameters_double[j].getParameterValue());
			}
			std::vector<jta_cost_function::Parameter<int>> trunk_parameters_int = trunk_cost_functions[i].
				getIntParameters();
			for (int j = 0; j < trunk_parameters_int.size(); j++) {
				setting.setValue(QString::fromStdString(
					                 "TRUNK@" + trunk_cost_functions[i].getCostFunctionName() + "@" +
					                 trunk_parameters_int[j].getParameterName()
					                 + "@" + trunk_parameters_int[j].getParameterType()),
				                 trunk_parameters_int[j].getParameterValue());
			}
			std::vector<jta_cost_function::Parameter<bool>> trunk_parameters_bool = trunk_cost_functions[i].
				getBoolParameters();
			for (int j = 0; j < trunk_parameters_bool.size(); j++) {
				setting.setValue(QString::fromStdString(
					                 "TRUNK@" + trunk_cost_functions[i].getCostFunctionName() + "@" +
					                 trunk_parameters_bool[j].getParameterName()
					                 + "@" + trunk_parameters_bool[j].getParameterType()),
				                 trunk_parameters_bool[j].getParameterValue());
			}
		}

		/*Branch*/
		setting.setValue("BRANCH@ACTIVE_CF", QString::fromStdString(branch_manager_.getActiveCostFunction()));
		std::vector<jta_cost_function::CostFunction> branch_cost_functions = branch_manager_.
			getAvailableCostFunctions();
		for (int i = 0; i < branch_cost_functions.size(); i++) {
			std::vector<jta_cost_function::Parameter<double>> branch_parameters_double = branch_cost_functions[i].
				getDoubleParameters();
			for (int j = 0; j < branch_parameters_double.size(); j++) {
				setting.setValue(QString::fromStdString(
					                 "BRANCH@" + branch_cost_functions[i].getCostFunctionName() + "@" +
					                 branch_parameters_double[j].getParameterName()
					                 + "@" + branch_parameters_double[j].getParameterType()),
				                 branch_parameters_double[j].getParameterValue());
			}
			std::vector<jta_cost_function::Parameter<int>> branch_parameters_int = branch_cost_functions[i].
				getIntParameters();
			for (int j = 0; j < branch_parameters_int.size(); j++) {
				setting.setValue(QString::fromStdString(
					                 "BRANCH@" + branch_cost_functions[i].getCostFunctionName() + "@" +
					                 branch_parameters_int[j].getParameterName()
					                 + "@" + branch_parameters_int[j].getParameterType()),
				                 branch_parameters_int[j].getParameterValue());
			}
			std::vector<jta_cost_function::Parameter<bool>> branch_parameters_bool = branch_cost_functions[i].
				getBoolParameters();
			for (int j = 0; j < branch_parameters_bool.size(); j++) {
				setting.setValue(QString::fromStdString(
					                 "BRANCH@" + branch_cost_functions[i].getCostFunctionName() + "@" +
					                 branch_parameters_bool[j].getParameterName()
					                 + "@" + branch_parameters_bool[j].getParameterType()),
				                 branch_parameters_bool[j].getParameterValue());
			}
		}

		/*Leaf*/
		setting.setValue("LEAF@ACTIVE_CF", QString::fromStdString(leaf_manager_.getActiveCostFunction()));
		std::vector<jta_cost_function::CostFunction> leaf_cost_functions = leaf_manager_.getAvailableCostFunctions();
		for (int i = 0; i < leaf_cost_functions.size(); i++) {
			std::vector<jta_cost_function::Parameter<double>> leaf_parameters_double = leaf_cost_functions[i].
				getDoubleParameters();
			for (int j = 0; j < leaf_parameters_double.size(); j++) {
				setting.setValue(QString::fromStdString(
					                 "LEAF@" + leaf_cost_functions[i].getCostFunctionName() + "@" +
					                 leaf_parameters_double[j].getParameterName()
					                 + "@" + leaf_parameters_double[j].getParameterType()),
				                 leaf_parameters_double[j].getParameterValue());
			}
			std::vector<jta_cost_function::Parameter<int>> leaf_parameters_int = leaf_cost_functions[i].
				getIntParameters();
			for (int j = 0; j < leaf_parameters_int.size(); j++) {
				setting.setValue(QString::fromStdString(
					                 "LEAF@" + leaf_cost_functions[i].getCostFunctionName() + "@" + leaf_parameters_int[
						                 j].getParameterName()
					                 + "@" + leaf_parameters_int[j].getParameterType()),
				                 leaf_parameters_int[j].getParameterValue());
			}
			std::vector<jta_cost_function::Parameter<bool>> leaf_parameters_bool = leaf_cost_functions[i].
				getBoolParameters();
			for (int j = 0; j < leaf_parameters_bool.size(); j++) {
				setting.setValue(QString::fromStdString(
					                 "LEAF@" + leaf_cost_functions[i].getCostFunctionName() + "@" + leaf_parameters_bool
					                 [j].getParameterName()
					                 + "@" + leaf_parameters_bool[j].getParameterType()),
				                 leaf_parameters_bool[j].getParameterValue());
			}
		}

		setting.endGroup();

		/*Save Optimizer Settings*/
		setting.beginGroup("OptimizerSettings");
		/*Variables*/
		/*Trunk*/
		setting.setValue("TRUNK@RANGE_X", optimizer_settings_.trunk_range.x);
		setting.setValue("TRUNK@RANGE_Y", optimizer_settings_.trunk_range.y);
		setting.setValue("TRUNK@RANGE_Z", optimizer_settings_.trunk_range.z);
		setting.setValue("TRUNK@RANGE_XA", optimizer_settings_.trunk_range.xa);
		setting.setValue("TRUNK@RANGE_YA", optimizer_settings_.trunk_range.ya);
		setting.setValue("TRUNK@RANGE_ZA", optimizer_settings_.trunk_range.za);
		setting.setValue("TRUNK@BUDGET", optimizer_settings_.trunk_budget);


		/*Branch*/
		setting.setValue("BRANCH@RANGE_X", optimizer_settings_.branch_range.x);
		setting.setValue("BRANCH@RANGE_Y", optimizer_settings_.branch_range.y);
		setting.setValue("BRANCH@RANGE_Z", optimizer_settings_.branch_range.z);
		setting.setValue("BRANCH@RANGE_XA", optimizer_settings_.branch_range.xa);
		setting.setValue("BRANCH@RANGE_YA", optimizer_settings_.branch_range.ya);
		setting.setValue("BRANCH@RANGE_ZA", optimizer_settings_.branch_range.za);
		setting.setValue("BRANCH@NUMBER_BRANCHES", optimizer_settings_.number_branches);
		setting.setValue("BRANCH@ENABLE", optimizer_settings_.enable_branch_);
		setting.setValue("BRANCH@BUDGET", optimizer_settings_.branch_budget);

		/*Leaf*/
		setting.setValue("LEAF@RANGE_X", optimizer_settings_.leaf_range.x);
		setting.setValue("LEAF@RANGE_Y", optimizer_settings_.leaf_range.y);
		setting.setValue("LEAF@RANGE_Z", optimizer_settings_.leaf_range.z);
		setting.setValue("LEAF@RANGE_XA", optimizer_settings_.leaf_range.xa);
		setting.setValue("LEAF@RANGE_YA", optimizer_settings_.leaf_range.ya);
		setting.setValue("LEAF@RANGE_ZA", optimizer_settings_.leaf_range.za);
		setting.setValue("LEAF@ENABLE", optimizer_settings_.enable_leaf_);
		setting.setValue("LEAF@BUDGET", optimizer_settings_.leaf_budget);
		setting.endGroup();

		/*Edge Detection Settings*/
		setting.beginGroup("EdgeDetectionSettings");
		ui.aperture_spin_box->setValue(APERTURE);
		ui.low_threshold_slider->setValue(LOW_THRESH);
		ui.high_threshold_slider->setValue(HIGH_THRESH);
		setting.endGroup();

		/*Save*/
		setting.beginGroup("EdgeDetectionSettings");
		setting.setValue("APERTURE", ui.aperture_spin_box->value());
		setting.setValue("LOW_THRESH", ui.low_threshold_slider->value());
		setting.setValue("HIGH_THRESH", ui.high_threshold_slider->value());
		setting.endGroup();
	}
}

/*Function to Save Settings from Optimizer Control Window to both Registry and Optimizer Settings Class*/
/*On Optimizer Control Windows Save Setting*/
void MainScreen::onSaveSettings(OptimizerSettings opt_settings,
                                jta_cost_function::CostFunctionManager trunk_manager,
                                jta_cost_function::CostFunctionManager branch_manager,
                                jta_cost_function::CostFunctionManager leaf_manager) {

	/*Save to Optimizer Settings*/
	optimizer_settings_ = opt_settings;

	/*Save 3 Cost Function Managers*/
	trunk_manager_ = trunk_manager;
	branch_manager_ = branch_manager;
	leaf_manager_ = leaf_manager;

	/*Save to Registry*/
	QString Version = "Version" + QString::number(VER_FIRST_NUM) +
		QString::number(VER_MIDDLE_NUM) + QString::number(VER_LAST_NUM);
	QSettings setting("JointTrackAutoGPU", Version);

	/*Save Cost Function Settings*/
	setting.beginGroup("CostFunctionSettings");
	/*Cost Function Managers (Save All Values for Parameters and Active Cost Function*/
	/*Trunk*/
	setting.setValue("TRUNK@ACTIVE_CF", QString::fromStdString(trunk_manager_.getActiveCostFunction()));
	std::vector<jta_cost_function::CostFunction> trunk_cost_functions = trunk_manager_.getAvailableCostFunctions();
	for (int i = 0; i < trunk_cost_functions.size(); i++) {
		std::vector<jta_cost_function::Parameter<double>> trunk_parameters_double = trunk_cost_functions[i].
			getDoubleParameters();
		for (int j = 0; j < trunk_parameters_double.size(); j++) {
			setting.setValue(QString::fromStdString(
				                 "TRUNK@" + trunk_cost_functions[i].getCostFunctionName() + "@" +
				                 trunk_parameters_double[j].getParameterName()
				                 + "@" + trunk_parameters_double[j].getParameterType()),
			                 trunk_parameters_double[j].getParameterValue());
		}
		std::vector<jta_cost_function::Parameter<int>> trunk_parameters_int = trunk_cost_functions[i].
			getIntParameters();
		for (int j = 0; j < trunk_parameters_int.size(); j++) {
			setting.setValue(QString::fromStdString(
				                 "TRUNK@" + trunk_cost_functions[i].getCostFunctionName() + "@" + trunk_parameters_int[
					                 j].getParameterName()
				                 + "@" + trunk_parameters_int[j].getParameterType()),
			                 trunk_parameters_int[j].getParameterValue());
		}
		std::vector<jta_cost_function::Parameter<bool>> trunk_parameters_bool = trunk_cost_functions[i].
			getBoolParameters();
		for (int j = 0; j < trunk_parameters_bool.size(); j++) {
			setting.setValue(QString::fromStdString(
				                 "TRUNK@" + trunk_cost_functions[i].getCostFunctionName() + "@" + trunk_parameters_bool[
					                 j].getParameterName()
				                 + "@" + trunk_parameters_bool[j].getParameterType()),
			                 trunk_parameters_bool[j].getParameterValue());
		}
	}

	/*Branch*/
	setting.setValue("BRANCH@ACTIVE_CF", QString::fromStdString(branch_manager_.getActiveCostFunction()));
	std::vector<jta_cost_function::CostFunction> branch_cost_functions = branch_manager_.getAvailableCostFunctions();
	for (int i = 0; i < branch_cost_functions.size(); i++) {
		std::vector<jta_cost_function::Parameter<double>> branch_parameters_double = branch_cost_functions[i].
			getDoubleParameters();
		for (int j = 0; j < branch_parameters_double.size(); j++) {
			setting.setValue(QString::fromStdString(
				                 "BRANCH@" + branch_cost_functions[i].getCostFunctionName() + "@" +
				                 branch_parameters_double[j].getParameterName()
				                 + "@" + branch_parameters_double[j].getParameterType()),
			                 branch_parameters_double[j].getParameterValue());
		}
		std::vector<jta_cost_function::Parameter<int>> branch_parameters_int = branch_cost_functions[i].
			getIntParameters();
		for (int j = 0; j < branch_parameters_int.size(); j++) {
			setting.setValue(QString::fromStdString(
				                 "BRANCH@" + branch_cost_functions[i].getCostFunctionName() + "@" +
				                 branch_parameters_int[j].getParameterName()
				                 + "@" + branch_parameters_int[j].getParameterType()),
			                 branch_parameters_int[j].getParameterValue());
		}
		std::vector<jta_cost_function::Parameter<bool>> branch_parameters_bool = branch_cost_functions[i].
			getBoolParameters();
		for (int j = 0; j < branch_parameters_bool.size(); j++) {
			setting.setValue(QString::fromStdString(
				                 "BRANCH@" + branch_cost_functions[i].getCostFunctionName() + "@" +
				                 branch_parameters_bool[j].getParameterName()
				                 + "@" + branch_parameters_bool[j].getParameterType()),
			                 branch_parameters_bool[j].getParameterValue());
		}
	}

	/*Leaf*/
	setting.setValue("LEAF@ACTIVE_CF", QString::fromStdString(leaf_manager_.getActiveCostFunction()));
	std::vector<jta_cost_function::CostFunction> leaf_cost_functions = leaf_manager_.getAvailableCostFunctions();
	for (int i = 0; i < leaf_cost_functions.size(); i++) {
		std::vector<jta_cost_function::Parameter<double>> leaf_parameters_double = leaf_cost_functions[i].
			getDoubleParameters();
		for (int j = 0; j < leaf_parameters_double.size(); j++) {
			setting.setValue(QString::fromStdString(
				                 "LEAF@" + leaf_cost_functions[i].getCostFunctionName() + "@" + leaf_parameters_double[
					                 j].getParameterName()
				                 + "@" + leaf_parameters_double[j].getParameterType()),
			                 leaf_parameters_double[j].getParameterValue());
		}
		std::vector<jta_cost_function::Parameter<int>> leaf_parameters_int = leaf_cost_functions[i].getIntParameters();
		for (int j = 0; j < leaf_parameters_int.size(); j++) {
			setting.setValue(QString::fromStdString(
				                 "LEAF@" + leaf_cost_functions[i].getCostFunctionName() + "@" + leaf_parameters_int[j].
				                 getParameterName()
				                 + "@" + leaf_parameters_int[j].getParameterType()),
			                 leaf_parameters_int[j].getParameterValue());
		}
		std::vector<jta_cost_function::Parameter<bool>> leaf_parameters_bool = leaf_cost_functions[i].
			getBoolParameters();
		for (int j = 0; j < leaf_parameters_bool.size(); j++) {
			setting.setValue(QString::fromStdString(
				                 "LEAF@" + leaf_cost_functions[i].getCostFunctionName() + "@" + leaf_parameters_bool[j].
				                 getParameterName()
				                 + "@" + leaf_parameters_bool[j].getParameterType()),
			                 leaf_parameters_bool[j].getParameterValue());
		}
	}

	setting.endGroup();

	/*Save Optimizer Settings*/
	setting.beginGroup("OptimizerSettings");
	/*Variables*/
	/*Trunk*/
	setting.setValue("TRUNK@RANGE_X", optimizer_settings_.trunk_range.x);
	setting.setValue("TRUNK@RANGE_Y", optimizer_settings_.trunk_range.y);
	setting.setValue("TRUNK@RANGE_Z", optimizer_settings_.trunk_range.z);
	setting.setValue("TRUNK@RANGE_XA", optimizer_settings_.trunk_range.xa);
	setting.setValue("TRUNK@RANGE_YA", optimizer_settings_.trunk_range.ya);
	setting.setValue("TRUNK@RANGE_ZA", optimizer_settings_.trunk_range.za);
	setting.setValue("TRUNK@BUDGET", optimizer_settings_.trunk_budget);

	/*Branch*/
	setting.setValue("BRANCH@RANGE_X", optimizer_settings_.branch_range.x);
	setting.setValue("BRANCH@RANGE_Y", optimizer_settings_.branch_range.y);
	setting.setValue("BRANCH@RANGE_Z", optimizer_settings_.branch_range.z);
	setting.setValue("BRANCH@RANGE_XA", optimizer_settings_.branch_range.xa);
	setting.setValue("BRANCH@RANGE_YA", optimizer_settings_.branch_range.ya);
	setting.setValue("BRANCH@RANGE_ZA", optimizer_settings_.branch_range.za);
	setting.setValue("BRANCH@NUMBER_BRANCHES", optimizer_settings_.number_branches);
	setting.setValue("BRANCH@ENABLE", optimizer_settings_.enable_branch_);
	setting.setValue("BRANCH@BUDGET", optimizer_settings_.branch_budget);

	/*Leaf*/
	setting.setValue("LEAF@RANGE_X", optimizer_settings_.leaf_range.x);
	setting.setValue("LEAF@RANGE_Y", optimizer_settings_.leaf_range.y);
	setting.setValue("LEAF@RANGE_Z", optimizer_settings_.leaf_range.z);
	setting.setValue("LEAF@RANGE_XA", optimizer_settings_.leaf_range.xa);
	setting.setValue("LEAF@RANGE_YA", optimizer_settings_.leaf_range.ya);
	setting.setValue("LEAF@RANGE_ZA", optimizer_settings_.leaf_range.za);
	setting.setValue("LEAF@ENABLE", optimizer_settings_.enable_leaf_);
	setting.setValue("LEAF@BUDGET", optimizer_settings_.leaf_budget);
	setting.endGroup();

	/*Update Dilation Frames*/
	UpdateDilationFrames();

}

/*Function That Saves Dilation as 0 if No Trunk Manager has a Dilation Int Parameter,
else saves all the Dilation Images for Each Frame as the Dilation Constant*/
void MainScreen::UpdateDilationFrames() {
	/*If TRUNK is Has Integer Parameter called Dilation, Update Dilation Values for Viewing Purposes*/
	int dilation_val = 0;
	std::vector<jta_cost_function::Parameter<int>> active_int_params = trunk_manager_.getActiveCostFunctionClass()->
		getIntParameters();
	for (int i = 0; i < active_int_params.size(); i++) {
		if (active_int_params[i].getParameterName() == "Dilation") {
			dilation_val = trunk_manager_.getActiveCostFunctionClass()->getIntParameters().at(i).getParameterValue();
		}
	}
	if (dilation_val < 0) {
		dilation_val = 0;
	}
	/*Mahfouz Case*/
	if (trunk_manager_.getActiveCostFunction() == "DIRECT_MAHFOUZ") {
		dilation_val = 3;
	}

	/*Apply Dilation to All Images*/
	for (int i = 0; i < loaded_frames.size(); i++) {
		loaded_frames[i].SetDilatedImage(dilation_val);
		if (calibrated_for_biplane_viewport_) {
			loaded_frames_B[i].SetDilatedImage(dilation_val);
		}
	}
	/*If Dilation View Selected*/
	if (ui.image_list_widget->currentIndex().row() >= 0 && ui.dilation_image_radio_button->isChecked() == true) {
		if (ui.camera_A_radio_button->isChecked()) {
			vw->update_display_background_to_dilation_image(this->curr_frame(), true);
		}
		else {
			vw->update_display_background_to_dilation_image(this->curr_frame(), false);
		}
		ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();
	}
}

void MainScreen::on_actionAmbiguous_Pose_Processing_triggered() {
	if (ui.model_list_widget->selectionModel()->selectedRows().size() != 2) {
		QMessageBox::critical(this, "Error!",
		                      "Must Be in Multiple Model Selection Mode to Run Ambiguous Pose Analysis!",
		                      QMessageBox::Ok);
		return;
	}

	// Loop through each of the frames
	QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();

	// save the current location of the image


	for (int i = 0; i < ui.image_list_widget->count(); i++) {
		Point6D fem_pose = model_locations_.GetPose(i, selected[1].row());
		Point6D tib_pose_orig = model_locations_.GetPose(i, selected[0].row());

		Point6D tib_pose_final = tibial_pose_selector(fem_pose, tib_pose_orig);
		model_locations_.SavePose(i, selected[0].row(), tib_pose_final);

	}
	// Need to update the location of the frame that is currently on screen
	int selected_img_idx = ui.image_list_widget->selectionModel()->selectedRows()[0].row();
	Point6D current_img_pos = model_locations_.GetPose(selected_img_idx, selected[0].row());
	model_actor_list[selected[0].row()]->SetPosition(current_img_pos.x, current_img_pos.y, current_img_pos.z);
	model_actor_list[selected[0].row()]->SetOrientation(current_img_pos.xa, current_img_pos.ya, current_img_pos.za);

	ui.qvtk_widget->update();
		ui.qvtk_widget->renderWindow()->Render();

}
