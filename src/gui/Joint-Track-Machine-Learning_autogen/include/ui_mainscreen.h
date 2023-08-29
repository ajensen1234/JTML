/********************************************************************************
** Form generated from reading UI file 'mainscreen.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINSCREEN_H
#define UI_MAINSCREEN_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "QVTKOpenGLNativeWidget.h"

QT_BEGIN_NAMESPACE

class Ui_MainScreenClass
{
public:
    QAction *actionOptions;
    QAction *actionOptions_2;
    QAction *actionLoad_Pose;
    QAction *actionLoad_Kinematics;
    QAction *actionSave_Pose;
    QAction *actionSave_Kinematics;
    QAction *actionQuit;
    QAction *actionRegion_Selection;
    QAction *actionCenter_Placement;
    QAction *actionParallel_Tuner;
    QAction *actionStop_Optimizer;
    QAction *actionOptimizer_Settings;
    QAction *actionAbout_JointTrack_Auto;
    QAction *actionControls;
    QAction *actionReset_View;
    QAction *actionModel_Interaction_Mode;
    QAction *actionCamera_Interaction_Mode;
    QAction *actionReset_Normal_Up;
    QAction *actionDRR_Settings;
    QAction *actionBlack_Implant_Silhouettes_in_Original_Image_s;
    QAction *actionWhite_Implant_Silhouettes_in_Original_Image_s;
    QAction *actionReset_Remove_All_Segmentation;
    QAction *actionCustom_Segmentation;
    QAction *actionSegment_FemHR;
    QAction *actionSegment_TibHR;
    QAction *actionEstimate_Femoral_Implant_s;
    QAction *actionEstimate_Tibial_Implant_s;
    QAction *actionCopy_Next_Pose;
    QAction *actionCopy_Previous_Pose;
    QAction *actionLaunch_Tool;
    QAction *actionOptimize_Backward;
    QAction *actionNFD_Pose_Estimate;
    QAction *actionAmbiguous_Pose_Processing;
    QAction *actionOpen_Viewer_Window;
    QAction *actionCoronal_Plane_Viewer;
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout_4;
    QHBoxLayout *horizontalLayout_8;
    QVBoxLayout *Left;
    QGroupBox *preprocessor_box;
    QVBoxLayout *verticalLayout_11;
    QVBoxLayout *verticalLayout_5;
    QPushButton *load_calibration_button;
    QPushButton *load_image_button;
    QPushButton *load_model_button;
    QGroupBox *optimization_box;
    QVBoxLayout *verticalLayout_8;
    QGridLayout *gridLayout;
    QPushButton *optimize_button;
    QPushButton *optimize_all_button;
    QPushButton *optimize_each_button;
    QPushButton *optimize_from_button;
    QGroupBox *image_view_box;
    QVBoxLayout *verticalLayout_10;
    QGridLayout *gridLayout_2;
    QRadioButton *original_image_radio_button;
    QRadioButton *inverted_image_radio_button;
    QRadioButton *edges_image_radio_button;
    QRadioButton *dilation_image_radio_button;
    QGroupBox *image_selection_box;
    QVBoxLayout *verticalLayout_9;
    QVBoxLayout *verticalLayout_6;
    QHBoxLayout *horizontalLayout_7;
    QRadioButton *camera_A_radio_button;
    QRadioButton *camera_B_radio_button;
    QListWidget *image_list_widget;
    QGridLayout *gridLayout_3;
    QHBoxLayout *horizontalLayout_3;
    QProgressBar *pose_progress;
    QLabel *pose_label;
    QVTKOpenGLNativeWidget *qvtk_widget;
    QVBoxLayout *Right;
    QVTKOpenGLNativeWidget *qvtk_cpv;
    QGroupBox *edge_detection_box;
    QVBoxLayout *verticalLayout_13;
    QVBoxLayout *verticalLayout_3;
    QHBoxLayout *horizontalLayout_4;
    QLabel *aperture_label;
    QSpinBox *aperture_spin_box;
    QHBoxLayout *horizontalLayout_5;
    QLabel *low_threshold_label;
    QLabel *low_threshold_value;
    QSlider *low_threshold_slider;
    QHBoxLayout *horizontalLayout_6;
    QLabel *high_threshold_label;
    QLabel *high_threshold_value;
    QSlider *high_threshold_slider;
    QPushButton *apply_all_edge_button;
    QPushButton *reset_edge_button;
    QGroupBox *model_view_box;
    QVBoxLayout *verticalLayout_15;
    QHBoxLayout *horizontalLayout_2;
    QVBoxLayout *verticalLayout_12;
    QRadioButton *original_model_radio_button;
    QRadioButton *transparent_model_radio_button;
    QVBoxLayout *verticalLayout_2;
    QRadioButton *solid_model_radio_button;
    QRadioButton *wireframe_model_radio_button;
    QGroupBox *model_selection_box;
    QVBoxLayout *verticalLayout_14;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QRadioButton *single_model_radio_button;
    QRadioButton *multiple_model_radio_button;
    QListWidget *model_list_widget;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuOptions;
    QMenu *menuOptimizer;
    QMenu *menuHelp;
    QMenu *menuVIEW;
    QMenu *menuSegment;
    QMenu *menuPOSE_ESTIMATE;
    QMenu *menuSYMMETRY_TRAP_ANALYSIS;
    QMenu *menuOPTIMZER;

    void setupUi(QMainWindow *MainScreenClass)
    {
        if (MainScreenClass->objectName().isEmpty())
            MainScreenClass->setObjectName(QString::fromUtf8("MainScreenClass"));
        MainScreenClass->resize(1147, 787);
        QSizePolicy sizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(MainScreenClass->sizePolicy().hasHeightForWidth());
        MainScreenClass->setSizePolicy(sizePolicy);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/Desktop_Icon/Resources/jta_dime_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
        MainScreenClass->setWindowIcon(icon);
        MainScreenClass->setAutoFillBackground(false);
        MainScreenClass->setStyleSheet(QString::fromUtf8("/*\n"
"	Based on a template by Emanuel Claesson (c) 2013 with modifications by Paris Flood\n"
"\n"
"	Licensed under the Apache License, Version 2.0 (the \"License\");\n"
"	you may not use this file except in compliance with the License.\n"
"	You may obtain a copy of the License at\n"
"\n"
"		http://www.apache.org/licenses/LICENSE-2.0\n"
"\n"
"	Unless required by applicable law or agreed to in writing, software\n"
"	distributed under the License is distributed on an \"AS IS\" BASIS,\n"
"	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
"	See the License for the specific language governing permissions and\n"
"	limitations under the License.\n"
"*/\n"
"\n"
"/*\n"
"	COLOR_DARK     = #191919\n"
"	COLOR_MEDIUM   = #353535\n"
"	COLOR_MEDLIGHT = #5A5A5A\n"
"	COLOR_LIGHT    = #DDDDDD\n"
"	COLOR_BLUE  = #2158AA\n"
"	COLOR_ORANGE = #D66C23\n"
"*/\n"
"\n"
"* {\n"
"	background: #191919;\n"
"	color: #DDDDDD;\n"
"}\n"
"\n"
"QMessageBox  QPushButton{\n"
"border: 1px solid #5A5A5A;\n"
"   background:"
                        " #252525;\n"
"padding: 8px 45px;\n"
"}\n"
"\n"
" QMenuBar {\n"
"	background: #252525;\n"
"border: 1px solid #5A5A5A;\n"
" border-style: none none solid none;\n"
" }\n"
"\n"
" QMenuBar::item {\n"
"	 background: transparent;\n"
"     spacing: 3px; /* spacing between menu bar items */\n"
"     padding: 6px 25px;\n"
" }\n"
"\n"
" QMenuBar::item:selected{\n"
"	background: #353535;\n"
"  	border: 2px solid #D66C23;\n"
"    border-style: none none solid none;\n"
"}\n"
"\n"
" QMenuBar::item:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QCheckBox, QRadioButton {\n"
"	border: none;\n"
"}\n"
"\n"
"QRadioButton::indicator, QCheckBox::indicator {\n"
"	width: 13px;\n"
"	height: 13px;\n"
"}\n"
"\n"
"QRadioButton::indicator::unchecked, QCheckBox::indicator::unchecked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: none;\n"
"}\n"
"\n"
"QRadioButton::indicator:unchecked:hover, QCheckBox::indicator:unchecked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QRadioButton::indicator::ch"
                        "ecked, QCheckBox::indicator::checked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QRadioButton::indicator:checked:hover, QCheckBox::indicator:checked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"	subcontrol-origin: margin;\n"
"    subcontrol-position: top center;\n"
"}\n"
"\n"
"QScrollBar {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #191919;\n"
"}\n"
"\n"
"QScrollBar:horizontal {\n"
"	height: 15px;\n"
"	margin: 0px 16px 0px 16px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"	width: 15px;\n"
"	margin: 16px 0px 16px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle {\n"
"	background: #353535;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	border-width: 0px 1px 0px 1px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	border-width: 1px 0px 1px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	min-width: 20px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	"
                        "min-height: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-line, QScrollBar::sub-line {\n"
"	background:#353535;\n"
"	border: 1px solid #5A5A5A;\n"
"	subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::add-line {\n"
"	position: absolute;\n"
"}\n"
"\n"
"QScrollBar::add-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: right;\n"
"	left: 15px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: bottom;\n"
"	top: 15px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: left;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: top;\n"
"}\n"
"\n"
"QScrollBar:left-arrow, QScrollBar::right-arrow, QScrollBar::up-arrow, QScrollBar::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	width: 3px;\n"
"	height: 3px;\n"
"}\n"
"\n"
"QScrollBar::add-page, QScrollBar::sub-page {\n"
"	background: none;\n"
"}\n"
"\n"
"QAbstractButton {\n"
"	background: #252525;\n"
"}\n"
"\n"
"QAbstractButton:disabled {\n"
"	"
                        "background: #3F3F3F;\n"
"}\n"
"\n"
"QAbstractButton:hover {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"	background: #353535;\n"
"  	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QAbstractButton:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"/*QAbstractItemView {\n"
"	show-decoration-selected: 1;\n"
" 	outline: 0;\n"
"	selection-background-color: #2158AA;\n"
"	selection-color: #DDDDDD;\n"
"	alternate-background-color: #353535;\n"
"}*/\n"
"\n"
"QHeaderView {\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"	background: #191919;\n"
"	border: 1px solid #5A5A5A;\n"
"	padding: 4px;\n"
"}\n"
"\n"
"QHeaderView::section:selected, QHeaderView::section::checked {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QTableView {\n"
"	gridline-color: #5A5A5A;\n"
"}\n"
"\n"
"QTabBar {\n"
"	margin-left: 2px;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"	border-radius: 0px;\n"
"	padding: 4px;\n"
"	margin: 4px;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::do"
                        "wn-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox {\n"
"	padding-right: 15px;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"	subcontrol-origin: border;\n"
"}\n"
"QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"QAbstractSpinBox::up-arrow:hover, QAbstractSpinBox::down-arrow:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QSlider {\n"
"	border: none;\n"
""
                        "}\n"
"\n"
"QSlider::groove:horizontal {\n"
"	height: 5px;\n"
"	margin: 4px 0px 4px 0px;\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"	width: 5px;\n"
"	margin: 0px 4px 0px 4px;\n"
"}\n"
"\n"
"QSlider::handle {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"	width: 15px;\n"
"	margin: -4px 0px -4px 0px;\n"
"}\n"
"\n"
"QSlider::handle:vertical {\n"
"	height: 15px;\n"
"	margin: 0px -4px 0px -4px;\n"
"}\n"
"\n"
"QSlider::add-page:vertical, QSlider::sub-page:horizontal {\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QSlider::sub-page:vertical, QSlider::add-page:horizontal {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QLabel {\n"
"	border: none;\n"
"}\n"
"\n"
"QProgressBar {\n"
"	text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"	width: 1px;\n"
"	background-color: #2158AA;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QGroupBox {\n"
"	font: bold\n"
"}\n"
"\n"
"/*QListWidget::item:selected {\n"
"    background-color: #2158AA;\n"
""
                        "}*/\n"
"\n"
"QMenu::item:icon {\n"
"padding-left: 32px;\n"
"}\n"
"\n"
"QMenu::item {\n"
"padding-left: 32px;\n"
"padding-right: 32px;\n"
"padding-top: 5px;\n"
"padding-bottom: 5px;\n"
"}\n"
"\n"
" QMenu::item:selected{\n"
"	background: #353535;\n"
"	border: 2px solid #D66C23;\n"
"    border-style: none solid none none;\n"
"}\n"
"\n"
" QMenu::pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QMenu::item:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"    height: 5px;\n"
"	background: #2158AA;\n"
"}"));
        actionOptions = new QAction(MainScreenClass);
        actionOptions->setObjectName(QString::fromUtf8("actionOptions"));
        actionOptions_2 = new QAction(MainScreenClass);
        actionOptions_2->setObjectName(QString::fromUtf8("actionOptions_2"));
        actionLoad_Pose = new QAction(MainScreenClass);
        actionLoad_Pose->setObjectName(QString::fromUtf8("actionLoad_Pose"));
        QFont font;
        actionLoad_Pose->setFont(font);
        actionLoad_Pose->setVisible(true);
        actionLoad_Kinematics = new QAction(MainScreenClass);
        actionLoad_Kinematics->setObjectName(QString::fromUtf8("actionLoad_Kinematics"));
        actionSave_Pose = new QAction(MainScreenClass);
        actionSave_Pose->setObjectName(QString::fromUtf8("actionSave_Pose"));
        actionSave_Kinematics = new QAction(MainScreenClass);
        actionSave_Kinematics->setObjectName(QString::fromUtf8("actionSave_Kinematics"));
        actionQuit = new QAction(MainScreenClass);
        actionQuit->setObjectName(QString::fromUtf8("actionQuit"));
        actionRegion_Selection = new QAction(MainScreenClass);
        actionRegion_Selection->setObjectName(QString::fromUtf8("actionRegion_Selection"));
        actionRegion_Selection->setEnabled(false);
        actionRegion_Selection->setFont(font);
        actionCenter_Placement = new QAction(MainScreenClass);
        actionCenter_Placement->setObjectName(QString::fromUtf8("actionCenter_Placement"));
        actionCenter_Placement->setEnabled(false);
        actionParallel_Tuner = new QAction(MainScreenClass);
        actionParallel_Tuner->setObjectName(QString::fromUtf8("actionParallel_Tuner"));
        actionStop_Optimizer = new QAction(MainScreenClass);
        actionStop_Optimizer->setObjectName(QString::fromUtf8("actionStop_Optimizer"));
        actionOptimizer_Settings = new QAction(MainScreenClass);
        actionOptimizer_Settings->setObjectName(QString::fromUtf8("actionOptimizer_Settings"));
        actionOptimizer_Settings->setFont(font);
        actionAbout_JointTrack_Auto = new QAction(MainScreenClass);
        actionAbout_JointTrack_Auto->setObjectName(QString::fromUtf8("actionAbout_JointTrack_Auto"));
        actionControls = new QAction(MainScreenClass);
        actionControls->setObjectName(QString::fromUtf8("actionControls"));
        actionReset_View = new QAction(MainScreenClass);
        actionReset_View->setObjectName(QString::fromUtf8("actionReset_View"));
        actionModel_Interaction_Mode = new QAction(MainScreenClass);
        actionModel_Interaction_Mode->setObjectName(QString::fromUtf8("actionModel_Interaction_Mode"));
        actionModel_Interaction_Mode->setCheckable(true);
        actionModel_Interaction_Mode->setChecked(true);
        actionCamera_Interaction_Mode = new QAction(MainScreenClass);
        actionCamera_Interaction_Mode->setObjectName(QString::fromUtf8("actionCamera_Interaction_Mode"));
        actionCamera_Interaction_Mode->setCheckable(true);
        actionReset_Normal_Up = new QAction(MainScreenClass);
        actionReset_Normal_Up->setObjectName(QString::fromUtf8("actionReset_Normal_Up"));
        actionDRR_Settings = new QAction(MainScreenClass);
        actionDRR_Settings->setObjectName(QString::fromUtf8("actionDRR_Settings"));
        actionBlack_Implant_Silhouettes_in_Original_Image_s = new QAction(MainScreenClass);
        actionBlack_Implant_Silhouettes_in_Original_Image_s->setObjectName(QString::fromUtf8("actionBlack_Implant_Silhouettes_in_Original_Image_s"));
        actionBlack_Implant_Silhouettes_in_Original_Image_s->setCheckable(true);
        actionBlack_Implant_Silhouettes_in_Original_Image_s->setChecked(true);
        actionWhite_Implant_Silhouettes_in_Original_Image_s = new QAction(MainScreenClass);
        actionWhite_Implant_Silhouettes_in_Original_Image_s->setObjectName(QString::fromUtf8("actionWhite_Implant_Silhouettes_in_Original_Image_s"));
        actionWhite_Implant_Silhouettes_in_Original_Image_s->setCheckable(true);
        actionReset_Remove_All_Segmentation = new QAction(MainScreenClass);
        actionReset_Remove_All_Segmentation->setObjectName(QString::fromUtf8("actionReset_Remove_All_Segmentation"));
        actionCustom_Segmentation = new QAction(MainScreenClass);
        actionCustom_Segmentation->setObjectName(QString::fromUtf8("actionCustom_Segmentation"));
        actionSegment_FemHR = new QAction(MainScreenClass);
        actionSegment_FemHR->setObjectName(QString::fromUtf8("actionSegment_FemHR"));
        actionSegment_TibHR = new QAction(MainScreenClass);
        actionSegment_TibHR->setObjectName(QString::fromUtf8("actionSegment_TibHR"));
        actionEstimate_Femoral_Implant_s = new QAction(MainScreenClass);
        actionEstimate_Femoral_Implant_s->setObjectName(QString::fromUtf8("actionEstimate_Femoral_Implant_s"));
        actionEstimate_Tibial_Implant_s = new QAction(MainScreenClass);
        actionEstimate_Tibial_Implant_s->setObjectName(QString::fromUtf8("actionEstimate_Tibial_Implant_s"));
        actionCopy_Next_Pose = new QAction(MainScreenClass);
        actionCopy_Next_Pose->setObjectName(QString::fromUtf8("actionCopy_Next_Pose"));
        actionCopy_Previous_Pose = new QAction(MainScreenClass);
        actionCopy_Previous_Pose->setObjectName(QString::fromUtf8("actionCopy_Previous_Pose"));
        actionLaunch_Tool = new QAction(MainScreenClass);
        actionLaunch_Tool->setObjectName(QString::fromUtf8("actionLaunch_Tool"));
        actionOptimize_Backward = new QAction(MainScreenClass);
        actionOptimize_Backward->setObjectName(QString::fromUtf8("actionOptimize_Backward"));
        actionNFD_Pose_Estimate = new QAction(MainScreenClass);
        actionNFD_Pose_Estimate->setObjectName(QString::fromUtf8("actionNFD_Pose_Estimate"));
        actionAmbiguous_Pose_Processing = new QAction(MainScreenClass);
        actionAmbiguous_Pose_Processing->setObjectName(QString::fromUtf8("actionAmbiguous_Pose_Processing"));
        actionOpen_Viewer_Window = new QAction(MainScreenClass);
        actionOpen_Viewer_Window->setObjectName(QString::fromUtf8("actionOpen_Viewer_Window"));
        actionCoronal_Plane_Viewer = new QAction(MainScreenClass);
        actionCoronal_Plane_Viewer->setObjectName(QString::fromUtf8("actionCoronal_Plane_Viewer"));
        centralWidget = new QWidget(MainScreenClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        sizePolicy.setHeightForWidth(centralWidget->sizePolicy().hasHeightForWidth());
        centralWidget->setSizePolicy(sizePolicy);
        verticalLayout_4 = new QVBoxLayout(centralWidget);
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setSpacing(6);
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        Left = new QVBoxLayout();
        Left->setSpacing(6);
        Left->setObjectName(QString::fromUtf8("Left"));
        preprocessor_box = new QGroupBox(centralWidget);
        preprocessor_box->setObjectName(QString::fromUtf8("preprocessor_box"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Minimum);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(preprocessor_box->sizePolicy().hasHeightForWidth());
        preprocessor_box->setSizePolicy(sizePolicy1);
        preprocessor_box->setStyleSheet(QString::fromUtf8("/*\n"
"	Based on a template by Emanuel Claesson (c) 2013 with modifications by Paris Flood\n"
"\n"
"	Licensed under the Apache License, Version 2.0 (the \"License\");\n"
"	you may not use this file except in compliance with the License.\n"
"	You may obtain a copy of the License at\n"
"\n"
"		http://www.apache.org/licenses/LICENSE-2.0\n"
"\n"
"	Unless required by applicable law or agreed to in writing, software\n"
"	distributed under the License is distributed on an \"AS IS\" BASIS,\n"
"	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
"	See the License for the specific language governing permissions and\n"
"	limitations under the License.\n"
"*/\n"
"\n"
"/*\n"
"	COLOR_DARK     = #191919\n"
"	COLOR_MEDIUM   = #353535\n"
"	COLOR_MEDLIGHT = #5A5A5A\n"
"	COLOR_LIGHT    = #DDDDDD\n"
"	COLOR_BLUE  = #2158AA\n"
"	COLOR_ORANGE = #D66C23\n"
"*/\n"
"* {\n"
"	background: #191919;\n"
"	color: #DDDDDD;\n"
"border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
" QMenuBar {\n"
"	background: #191919;\n"
"	border"
                        "-style: none;\n"
" }\n"
"\n"
" QMenuBar::item {\n"
"	 background: transparent;\n"
"     spacing: 3px; /* spacing between menu bar items */\n"
"     padding: 5px 25px;\n"
" }\n"
"\n"
" QMenuBar::item:selected{\n"
"	background: #353535;\n"
"  	border: 2px solid #D66C23;\n"
"    border-style: none none solid none;\n"
"}\n"
"\n"
" QMenuBar::item:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QCheckBox, QRadioButton {\n"
"	border: none;\n"
"}\n"
"\n"
"QRadioButton::indicator, QCheckBox::indicator {\n"
"	width: 13px;\n"
"	height: 13px;\n"
"}\n"
"\n"
"QRadioButton::indicator::unchecked, QCheckBox::indicator::unchecked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: none;\n"
"}\n"
"\n"
"QRadioButton::indicator:unchecked:hover, QCheckBox::indicator:unchecked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QRadioButton::indicator::checked, QCheckBox::indicator::checked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QRadioButton::in"
                        "dicator:checked:hover, QCheckBox::indicator:checked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"	subcontrol-origin: margin;\n"
"    subcontrol-position: top center;\n"
"}\n"
"\n"
"QScrollBar {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #191919;\n"
"}\n"
"\n"
"QScrollBar:horizontal {\n"
"	height: 15px;\n"
"	margin: 0px 16px 0px 16px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"	width: 15px;\n"
"	margin: 16px 0px 16px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle {\n"
"	background: #353535;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	border-width: 0px 1px 0px 1px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	border-width: 1px 0px 1px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	min-width: 20px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	min-height: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-line, QScrollBar::sub-line {\n"
"	background:#353535;\n"
"	border: 1px solid #5A5A5A;\n"
"	subc"
                        "ontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::add-line {\n"
"	position: absolute;\n"
"}\n"
"\n"
"QScrollBar::add-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: right;\n"
"	left: 15px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: bottom;\n"
"	top: 15px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: left;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: top;\n"
"}\n"
"\n"
"QScrollBar:left-arrow, QScrollBar::right-arrow, QScrollBar::up-arrow, QScrollBar::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	width: 3px;\n"
"	height: 3px;\n"
"}\n"
"\n"
"QScrollBar::add-page, QScrollBar::sub-page {\n"
"	background: none;\n"
"}\n"
"\n"
"QAbstractButton {\n"
"	background: #252525;\n"
"}\n"
"\n"
"QAbstractButton:disabled {\n"
"	background: #3F3F3F;\n"
"}\n"
"\n"
"QAbstractButton:hover {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"	background: #35353"
                        "5;\n"
"  	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QAbstractButton:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QAbstractItemView {\n"
"	show-decoration-selected: 1;\n"
" 	outline: 0;\n"
"	selection-background-color: #2158AA;\n"
"	selection-color: #DDDDDD;\n"
"	alternate-background-color: #353535;\n"
"}\n"
"\n"
"QHeaderView {\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"	background: #191919;\n"
"	border: 1px solid #5A5A5A;\n"
"	padding: 4px;\n"
"}\n"
"\n"
"QHeaderView::section:selected, QHeaderView::section::checked {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QTableView {\n"
"	gridline-color: #5A5A5A;\n"
"}\n"
"\n"
"QTabBar {\n"
"	margin-left: 2px;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"	border-radius: 0px;\n"
"	padding: 4px;\n"
"	margin: 4px;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"	border: 1px solid #5A5A5A;\n"
"	back"
                        "ground: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox {\n"
"	padding-right: 15px;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"	subcontrol-origin: border;\n"
"}\n"
"QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"QAbstractSpinBox::up-arrow:hover, QAbstractSpinBox::down-arrow:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QSlider {\n"
"	border: none;\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"	height: 5px;\n"
"	margin: 4px 0px 4px 0px;\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"	width: 5px;"
                        "\n"
"	margin: 0px 4px 0px 4px;\n"
"}\n"
"\n"
"QSlider::handle {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"	width: 15px;\n"
"	margin: -4px 0px -4px 0px;\n"
"}\n"
"\n"
"QSlider::handle:vertical {\n"
"	height: 15px;\n"
"	margin: 0px -4px 0px -4px;\n"
"}\n"
"\n"
"QSlider::add-page:vertical, QSlider::sub-page:horizontal {\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QSlider::sub-page:vertical, QSlider::add-page:horizontal {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QLabel {\n"
"	border: none;\n"
"}\n"
"\n"
"QProgressBar {\n"
"	text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"	width: 1px;\n"
"	background-color: #2158AA;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QGroupBox {\n"
"	font: bold\n"
"}\n"
"\n"
"QListWidget::item:selected {\n"
"    background-color: #2158AA;\n"
"}\n"
"\n"
" QMenu::item:selected{\n"
"	background: #353535;\n"
"	border: 2px solid #D66C23;\n"
"    border-style: none solid none none;\n"
"}\n"
""
                        "\n"
" QMenu::pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"    height: 5px;\n"
"	background: #2158AA;\n"
"}"));
        preprocessor_box->setAlignment(Qt::AlignHCenter|Qt::AlignTop);
        preprocessor_box->setFlat(false);
        verticalLayout_11 = new QVBoxLayout(preprocessor_box);
        verticalLayout_11->setSpacing(6);
        verticalLayout_11->setContentsMargins(11, 11, 11, 11);
        verticalLayout_11->setObjectName(QString::fromUtf8("verticalLayout_11"));
        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        load_calibration_button = new QPushButton(preprocessor_box);
        load_calibration_button->setObjectName(QString::fromUtf8("load_calibration_button"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::MinimumExpanding);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(load_calibration_button->sizePolicy().hasHeightForWidth());
        load_calibration_button->setSizePolicy(sizePolicy2);
        load_calibration_button->setStyleSheet(QString::fromUtf8("QButton {    \n"
"       min-width: 80em;\n"
"       min-height: 40em;\n"
"       width: 1600em;\n"
"       height: 80em;\"    \n"
"}"));

        verticalLayout_5->addWidget(load_calibration_button);

        load_image_button = new QPushButton(preprocessor_box);
        load_image_button->setObjectName(QString::fromUtf8("load_image_button"));
        sizePolicy2.setHeightForWidth(load_image_button->sizePolicy().hasHeightForWidth());
        load_image_button->setSizePolicy(sizePolicy2);

        verticalLayout_5->addWidget(load_image_button);

        load_model_button = new QPushButton(preprocessor_box);
        load_model_button->setObjectName(QString::fromUtf8("load_model_button"));
        sizePolicy2.setHeightForWidth(load_model_button->sizePolicy().hasHeightForWidth());
        load_model_button->setSizePolicy(sizePolicy2);

        verticalLayout_5->addWidget(load_model_button);


        verticalLayout_11->addLayout(verticalLayout_5);


        Left->addWidget(preprocessor_box);

        optimization_box = new QGroupBox(centralWidget);
        optimization_box->setObjectName(QString::fromUtf8("optimization_box"));
        sizePolicy1.setHeightForWidth(optimization_box->sizePolicy().hasHeightForWidth());
        optimization_box->setSizePolicy(sizePolicy1);
        optimization_box->setStyleSheet(QString::fromUtf8("/*\n"
"	Based on a template by Emanuel Claesson (c) 2013 with modifications by Paris Flood\n"
"\n"
"	Licensed under the Apache License, Version 2.0 (the \"License\");\n"
"	you may not use this file except in compliance with the License.\n"
"	You may obtain a copy of the License at\n"
"\n"
"		http://www.apache.org/licenses/LICENSE-2.0\n"
"\n"
"	Unless required by applicable law or agreed to in writing, software\n"
"	distributed under the License is distributed on an \"AS IS\" BASIS,\n"
"	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
"	See the License for the specific language governing permissions and\n"
"	limitations under the License.\n"
"*/\n"
"\n"
"/*\n"
"	COLOR_DARK     = #191919\n"
"	COLOR_MEDIUM   = #353535\n"
"	COLOR_MEDLIGHT = #5A5A5A\n"
"	COLOR_LIGHT    = #DDDDDD\n"
"	COLOR_BLUE  = #2158AA\n"
"	COLOR_ORANGE = #D66C23\n"
"*/\n"
"* {\n"
"	background: #191919;\n"
"	color: #DDDDDD;\n"
"border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
" QMenuBar {\n"
"	background: #191919;\n"
"	border"
                        "-style: none;\n"
" }\n"
"\n"
" QMenuBar::item {\n"
"	 background: transparent;\n"
"     spacing: 3px; /* spacing between menu bar items */\n"
"     padding: 5px 25px;\n"
" }\n"
"\n"
" QMenuBar::item:selected{\n"
"	background: #353535;\n"
"  	border: 2px solid #D66C23;\n"
"    border-style: none none solid none;\n"
"}\n"
"\n"
" QMenuBar::item:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QCheckBox, QRadioButton {\n"
"	border: none;\n"
"}\n"
"\n"
"QRadioButton::indicator, QCheckBox::indicator {\n"
"	width: 13px;\n"
"	height: 13px;\n"
"}\n"
"\n"
"QRadioButton::indicator::unchecked, QCheckBox::indicator::unchecked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: none;\n"
"}\n"
"\n"
"QRadioButton::indicator:unchecked:hover, QCheckBox::indicator:unchecked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QRadioButton::indicator::checked, QCheckBox::indicator::checked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QRadioButton::in"
                        "dicator:checked:hover, QCheckBox::indicator:checked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"	subcontrol-origin: margin;\n"
"    subcontrol-position: top center;\n"
"}\n"
"\n"
"QScrollBar {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #191919;\n"
"}\n"
"\n"
"QScrollBar:horizontal {\n"
"	height: 15px;\n"
"	margin: 0px 16px 0px 16px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"	width: 15px;\n"
"	margin: 16px 0px 16px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle {\n"
"	background: #353535;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	border-width: 0px 1px 0px 1px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	border-width: 1px 0px 1px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	min-width: 20px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	min-height: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-line, QScrollBar::sub-line {\n"
"	background:#353535;\n"
"	border: 1px solid #5A5A5A;\n"
"	subc"
                        "ontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::add-line {\n"
"	position: absolute;\n"
"}\n"
"\n"
"QScrollBar::add-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: right;\n"
"	left: 15px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: bottom;\n"
"	top: 15px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: left;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: top;\n"
"}\n"
"\n"
"QScrollBar:left-arrow, QScrollBar::right-arrow, QScrollBar::up-arrow, QScrollBar::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	width: 3px;\n"
"	height: 3px;\n"
"}\n"
"\n"
"QScrollBar::add-page, QScrollBar::sub-page {\n"
"	background: none;\n"
"}\n"
"\n"
"QAbstractButton {\n"
"	background: #252525;\n"
"}\n"
"\n"
"QAbstractButton:disabled {\n"
"	background: #3F3F3F;\n"
"}\n"
"\n"
"QAbstractButton:hover {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"	background: #35353"
                        "5;\n"
"  	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QAbstractButton:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QAbstractItemView {\n"
"	show-decoration-selected: 1;\n"
" 	outline: 0;\n"
"	selection-background-color: #2158AA;\n"
"	selection-color: #DDDDDD;\n"
"	alternate-background-color: #353535;\n"
"}\n"
"\n"
"QHeaderView {\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"	background: #191919;\n"
"	border: 1px solid #5A5A5A;\n"
"	padding: 4px;\n"
"}\n"
"\n"
"QHeaderView::section:selected, QHeaderView::section::checked {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QTableView {\n"
"	gridline-color: #5A5A5A;\n"
"}\n"
"\n"
"QTabBar {\n"
"	margin-left: 2px;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"	border-radius: 0px;\n"
"	padding: 4px;\n"
"	margin: 4px;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"	border: 1px solid #5A5A5A;\n"
"	back"
                        "ground: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox {\n"
"	padding-right: 15px;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"	subcontrol-origin: border;\n"
"}\n"
"QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"QAbstractSpinBox::up-arrow:hover, QAbstractSpinBox::down-arrow:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QSlider {\n"
"	border: none;\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"	height: 5px;\n"
"	margin: 4px 0px 4px 0px;\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"	width: 5px;"
                        "\n"
"	margin: 0px 4px 0px 4px;\n"
"}\n"
"\n"
"QSlider::handle {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"	width: 15px;\n"
"	margin: -4px 0px -4px 0px;\n"
"}\n"
"\n"
"QSlider::handle:vertical {\n"
"	height: 15px;\n"
"	margin: 0px -4px 0px -4px;\n"
"}\n"
"\n"
"QSlider::add-page:vertical, QSlider::sub-page:horizontal {\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QSlider::sub-page:vertical, QSlider::add-page:horizontal {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QLabel {\n"
"	border: none;\n"
"}\n"
"\n"
"QProgressBar {\n"
"	text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"	width: 1px;\n"
"	background-color: #2158AA;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QGroupBox {\n"
"	font: bold\n"
"}\n"
"\n"
"QListWidget::item:selected {\n"
"    background-color: #2158AA;\n"
"}\n"
"\n"
" QMenu::item:selected{\n"
"	background: #353535;\n"
"	border: 2px solid #D66C23;\n"
"    border-style: none solid none none;\n"
"}\n"
""
                        "\n"
" QMenu::pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"    height: 5px;\n"
"	background: #2158AA;\n"
"}"));
        optimization_box->setAlignment(Qt::AlignHCenter|Qt::AlignTop);
        optimization_box->setFlat(false);
        verticalLayout_8 = new QVBoxLayout(optimization_box);
        verticalLayout_8->setSpacing(6);
        verticalLayout_8->setContentsMargins(11, 11, 11, 11);
        verticalLayout_8->setObjectName(QString::fromUtf8("verticalLayout_8"));
        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        optimize_button = new QPushButton(optimization_box);
        optimize_button->setObjectName(QString::fromUtf8("optimize_button"));
        sizePolicy2.setHeightForWidth(optimize_button->sizePolicy().hasHeightForWidth());
        optimize_button->setSizePolicy(sizePolicy2);
        optimize_button->setStyleSheet(QString::fromUtf8("QButton {    \n"
"       min-width: 80em;\n"
"       min-height: 40em;\n"
"       width: 1600em;\n"
"       height: 80em;\"    \n"
"}"));

        gridLayout->addWidget(optimize_button, 0, 0, 1, 1);

        optimize_all_button = new QPushButton(optimization_box);
        optimize_all_button->setObjectName(QString::fromUtf8("optimize_all_button"));
        sizePolicy2.setHeightForWidth(optimize_all_button->sizePolicy().hasHeightForWidth());
        optimize_all_button->setSizePolicy(sizePolicy2);
        optimize_all_button->setStyleSheet(QString::fromUtf8("QButton {    \n"
"       min-width: 80em;\n"
"       min-height: 40em;\n"
"       width: 1600em;\n"
"       height: 80em;\"    \n"
"}"));

        gridLayout->addWidget(optimize_all_button, 0, 1, 1, 1);

        optimize_each_button = new QPushButton(optimization_box);
        optimize_each_button->setObjectName(QString::fromUtf8("optimize_each_button"));
        sizePolicy2.setHeightForWidth(optimize_each_button->sizePolicy().hasHeightForWidth());
        optimize_each_button->setSizePolicy(sizePolicy2);
        optimize_each_button->setStyleSheet(QString::fromUtf8("QButton {    \n"
"       min-width: 80em;\n"
"       min-height: 40em;\n"
"       width: 1600em;\n"
"       height: 80em;\"    \n"
"}"));

        gridLayout->addWidget(optimize_each_button, 1, 0, 1, 1);

        optimize_from_button = new QPushButton(optimization_box);
        optimize_from_button->setObjectName(QString::fromUtf8("optimize_from_button"));
        sizePolicy2.setHeightForWidth(optimize_from_button->sizePolicy().hasHeightForWidth());
        optimize_from_button->setSizePolicy(sizePolicy2);
        optimize_from_button->setStyleSheet(QString::fromUtf8("QButton {    \n"
"       min-width: 80em;\n"
"       min-height: 40em;\n"
"       width: 1600em;\n"
"       height: 80em;\"    \n"
"}"));

        gridLayout->addWidget(optimize_from_button, 1, 1, 1, 1);


        verticalLayout_8->addLayout(gridLayout);


        Left->addWidget(optimization_box);

        image_view_box = new QGroupBox(centralWidget);
        image_view_box->setObjectName(QString::fromUtf8("image_view_box"));
        sizePolicy1.setHeightForWidth(image_view_box->sizePolicy().hasHeightForWidth());
        image_view_box->setSizePolicy(sizePolicy1);
        image_view_box->setStyleSheet(QString::fromUtf8("/*\n"
"	Based on a template by Emanuel Claesson (c) 2013 with modifications by Paris Flood\n"
"\n"
"	Licensed under the Apache License, Version 2.0 (the \"License\");\n"
"	you may not use this file except in compliance with the License.\n"
"	You may obtain a copy of the License at\n"
"\n"
"		http://www.apache.org/licenses/LICENSE-2.0\n"
"\n"
"	Unless required by applicable law or agreed to in writing, software\n"
"	distributed under the License is distributed on an \"AS IS\" BASIS,\n"
"	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
"	See the License for the specific language governing permissions and\n"
"	limitations under the License.\n"
"*/\n"
"\n"
"/*\n"
"	COLOR_DARK     = #191919\n"
"	COLOR_MEDIUM   = #353535\n"
"	COLOR_MEDLIGHT = #5A5A5A\n"
"	COLOR_LIGHT    = #DDDDDD\n"
"	COLOR_BLUE  = #2158AA\n"
"	COLOR_ORANGE = #D66C23\n"
"*/\n"
"* {\n"
"	background: #191919;\n"
"	color: #DDDDDD;\n"
"border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
" QMenuBar {\n"
"	background: #191919;\n"
"	border"
                        "-style: none;\n"
" }\n"
"\n"
" QMenuBar::item {\n"
"	 background: transparent;\n"
"     spacing: 3px; /* spacing between menu bar items */\n"
"     padding: 5px 25px;\n"
" }\n"
"\n"
" QMenuBar::item:selected{\n"
"	background: #353535;\n"
"  	border: 2px solid #D66C23;\n"
"    border-style: none none solid none;\n"
"}\n"
"\n"
" QMenuBar::item:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QCheckBox, QRadioButton {\n"
"	border: none;\n"
"}\n"
"\n"
"QRadioButton::indicator, QCheckBox::indicator {\n"
"	width: 13px;\n"
"	height: 13px;\n"
"}\n"
"\n"
"QRadioButton::indicator::unchecked, QCheckBox::indicator::unchecked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: none;\n"
"}\n"
"\n"
"QRadioButton::indicator:unchecked:hover, QCheckBox::indicator:unchecked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QRadioButton::indicator::checked, QCheckBox::indicator::checked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QRadioButton::in"
                        "dicator:checked:hover, QCheckBox::indicator:checked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"	subcontrol-origin: margin;\n"
"    subcontrol-position: top center;\n"
"}\n"
"\n"
"QScrollBar {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #191919;\n"
"}\n"
"\n"
"QScrollBar:horizontal {\n"
"	height: 15px;\n"
"	margin: 0px 16px 0px 16px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"	width: 15px;\n"
"	margin: 16px 0px 16px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle {\n"
"	background: #353535;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	border-width: 0px 1px 0px 1px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	border-width: 1px 0px 1px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	min-width: 20px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	min-height: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-line, QScrollBar::sub-line {\n"
"	background:#353535;\n"
"	border: 1px solid #5A5A5A;\n"
"	subc"
                        "ontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::add-line {\n"
"	position: absolute;\n"
"}\n"
"\n"
"QScrollBar::add-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: right;\n"
"	left: 15px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: bottom;\n"
"	top: 15px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: left;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: top;\n"
"}\n"
"\n"
"QScrollBar:left-arrow, QScrollBar::right-arrow, QScrollBar::up-arrow, QScrollBar::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	width: 3px;\n"
"	height: 3px;\n"
"}\n"
"\n"
"QScrollBar::add-page, QScrollBar::sub-page {\n"
"	background: none;\n"
"}\n"
"\n"
"QAbstractButton {\n"
"	background: #252525;\n"
"}\n"
"\n"
"QAbstractButton:disabled {\n"
"	background: #3F3F3F;\n"
"}\n"
"\n"
"QAbstractButton:hover {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"	background: #35353"
                        "5;\n"
"  	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QAbstractButton:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QAbstractItemView {\n"
"	show-decoration-selected: 1;\n"
" 	outline: 0;\n"
"	selection-background-color: #2158AA;\n"
"	selection-color: #DDDDDD;\n"
"	alternate-background-color: #353535;\n"
"}\n"
"\n"
"QHeaderView {\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"	background: #191919;\n"
"	border: 1px solid #5A5A5A;\n"
"	padding: 4px;\n"
"}\n"
"\n"
"QHeaderView::section:selected, QHeaderView::section::checked {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QTableView {\n"
"	gridline-color: #5A5A5A;\n"
"}\n"
"\n"
"QTabBar {\n"
"	margin-left: 2px;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"	border-radius: 0px;\n"
"	padding: 4px;\n"
"	margin: 4px;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"	border: 1px solid #5A5A5A;\n"
"	back"
                        "ground: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox {\n"
"	padding-right: 15px;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"	subcontrol-origin: border;\n"
"}\n"
"QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"QAbstractSpinBox::up-arrow:hover, QAbstractSpinBox::down-arrow:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QSlider {\n"
"	border: none;\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"	height: 5px;\n"
"	margin: 4px 0px 4px 0px;\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"	width: 5px;"
                        "\n"
"	margin: 0px 4px 0px 4px;\n"
"}\n"
"\n"
"QSlider::handle {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"	width: 15px;\n"
"	margin: -4px 0px -4px 0px;\n"
"}\n"
"\n"
"QSlider::handle:vertical {\n"
"	height: 15px;\n"
"	margin: 0px -4px 0px -4px;\n"
"}\n"
"\n"
"QSlider::add-page:vertical, QSlider::sub-page:horizontal {\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QSlider::sub-page:vertical, QSlider::add-page:horizontal {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QLabel {\n"
"	border: none;\n"
"}\n"
"\n"
"QProgressBar {\n"
"	text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"	width: 1px;\n"
"	background-color: #2158AA;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QGroupBox {\n"
"	font: bold\n"
"}\n"
"\n"
"QListWidget::item:selected {\n"
"    background-color: #2158AA;\n"
"}\n"
"\n"
" QMenu::item:selected{\n"
"	background: #353535;\n"
"	border: 2px solid #D66C23;\n"
"    border-style: none solid none none;\n"
"}\n"
""
                        "\n"
" QMenu::pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"    height: 5px;\n"
"	background: #2158AA;\n"
"}"));
        verticalLayout_10 = new QVBoxLayout(image_view_box);
        verticalLayout_10->setSpacing(6);
        verticalLayout_10->setContentsMargins(11, 11, 11, 11);
        verticalLayout_10->setObjectName(QString::fromUtf8("verticalLayout_10"));
        gridLayout_2 = new QGridLayout();
        gridLayout_2->setSpacing(6);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        original_image_radio_button = new QRadioButton(image_view_box);
        original_image_radio_button->setObjectName(QString::fromUtf8("original_image_radio_button"));
        QSizePolicy sizePolicy3(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(original_image_radio_button->sizePolicy().hasHeightForWidth());
        original_image_radio_button->setSizePolicy(sizePolicy3);
        original_image_radio_button->setChecked(true);

        gridLayout_2->addWidget(original_image_radio_button, 0, 0, 1, 1);

        inverted_image_radio_button = new QRadioButton(image_view_box);
        inverted_image_radio_button->setObjectName(QString::fromUtf8("inverted_image_radio_button"));
        sizePolicy3.setHeightForWidth(inverted_image_radio_button->sizePolicy().hasHeightForWidth());
        inverted_image_radio_button->setSizePolicy(sizePolicy3);

        gridLayout_2->addWidget(inverted_image_radio_button, 0, 1, 1, 1);

        edges_image_radio_button = new QRadioButton(image_view_box);
        edges_image_radio_button->setObjectName(QString::fromUtf8("edges_image_radio_button"));
        sizePolicy3.setHeightForWidth(edges_image_radio_button->sizePolicy().hasHeightForWidth());
        edges_image_radio_button->setSizePolicy(sizePolicy3);

        gridLayout_2->addWidget(edges_image_radio_button, 1, 0, 1, 1);

        dilation_image_radio_button = new QRadioButton(image_view_box);
        dilation_image_radio_button->setObjectName(QString::fromUtf8("dilation_image_radio_button"));
        sizePolicy3.setHeightForWidth(dilation_image_radio_button->sizePolicy().hasHeightForWidth());
        dilation_image_radio_button->setSizePolicy(sizePolicy3);

        gridLayout_2->addWidget(dilation_image_radio_button, 1, 1, 1, 1);


        verticalLayout_10->addLayout(gridLayout_2);


        Left->addWidget(image_view_box);

        image_selection_box = new QGroupBox(centralWidget);
        image_selection_box->setObjectName(QString::fromUtf8("image_selection_box"));
        QSizePolicy sizePolicy4(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(image_selection_box->sizePolicy().hasHeightForWidth());
        image_selection_box->setSizePolicy(sizePolicy4);
        image_selection_box->setStyleSheet(QString::fromUtf8("/*\n"
"	Based on a template by Emanuel Claesson (c) 2013 with modifications by Paris Flood\n"
"\n"
"	Licensed under the Apache License, Version 2.0 (the \"License\");\n"
"	you may not use this file except in compliance with the License.\n"
"	You may obtain a copy of the License at\n"
"\n"
"		http://www.apache.org/licenses/LICENSE-2.0\n"
"\n"
"	Unless required by applicable law or agreed to in writing, software\n"
"	distributed under the License is distributed on an \"AS IS\" BASIS,\n"
"	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
"	See the License for the specific language governing permissions and\n"
"	limitations under the License.\n"
"*/\n"
"\n"
"/*\n"
"	COLOR_DARK     = #191919\n"
"	COLOR_MEDIUM   = #353535\n"
"	COLOR_MEDLIGHT = #5A5A5A\n"
"	COLOR_LIGHT    = #DDDDDD\n"
"	COLOR_BLUE  = #2158AA\n"
"	COLOR_ORANGE = #D66C23\n"
"*/\n"
"* {\n"
"	background: #191919;\n"
"	color: #DDDDDD;\n"
"border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
" QMenuBar {\n"
"	background: #191919;\n"
"	border"
                        "-style: none;\n"
" }\n"
"\n"
" QMenuBar::item {\n"
"	 background: transparent;\n"
"     spacing: 3px; /* spacing between menu bar items */\n"
"     padding: 5px 25px;\n"
" }\n"
"\n"
" QMenuBar::item:selected{\n"
"	background: #353535;\n"
"  	border: 2px solid #D66C23;\n"
"    border-style: none none solid none;\n"
"}\n"
"\n"
" QMenuBar::item:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QCheckBox, QRadioButton {\n"
"	border: none;\n"
"}\n"
"\n"
"QRadioButton::indicator, QCheckBox::indicator {\n"
"	width: 13px;\n"
"	height: 13px;\n"
"}\n"
"\n"
"QRadioButton::indicator::unchecked, QCheckBox::indicator::unchecked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: none;\n"
"}\n"
"\n"
"QRadioButton::indicator:unchecked:hover, QCheckBox::indicator:unchecked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QRadioButton::indicator::checked, QCheckBox::indicator::checked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QRadioButton::in"
                        "dicator:checked:hover, QCheckBox::indicator:checked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"	subcontrol-origin: margin;\n"
"    subcontrol-position: top center;\n"
"}\n"
"\n"
"QScrollBar {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #191919;\n"
"}\n"
"\n"
"QScrollBar:horizontal {\n"
"	height: 15px;\n"
"	margin: 0px 16px 0px 16px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"	width: 15px;\n"
"	margin: 16px 0px 16px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle {\n"
"	background: #353535;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	border-width: 0px 1px 0px 1px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	border-width: 1px 0px 1px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	min-width: 20px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	min-height: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-line, QScrollBar::sub-line {\n"
"	background:#353535;\n"
"	border: 1px solid #5A5A5A;\n"
"	subc"
                        "ontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::add-line {\n"
"	position: absolute;\n"
"}\n"
"\n"
"QScrollBar::add-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: right;\n"
"	left: 15px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: bottom;\n"
"	top: 15px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: left;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: top;\n"
"}\n"
"\n"
"QScrollBar:left-arrow, QScrollBar::right-arrow, QScrollBar::up-arrow, QScrollBar::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	width: 3px;\n"
"	height: 3px;\n"
"}\n"
"\n"
"QScrollBar::add-page, QScrollBar::sub-page {\n"
"	background: none;\n"
"}\n"
"\n"
"QAbstractButton {\n"
"	background: #252525;\n"
"}\n"
"\n"
"QAbstractButton:disabled {\n"
"	background: #3F3F3F;\n"
"}\n"
"\n"
"QAbstractButton:hover {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"	background: #35353"
                        "5;\n"
"  	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QAbstractButton:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QAbstractItemView {\n"
"	show-decoration-selected: 1;\n"
" 	outline: 0;\n"
"	selection-background-color: #2158AA;\n"
"	selection-color: #DDDDDD;\n"
"	alternate-background-color: #353535;\n"
"}\n"
"\n"
"QHeaderView {\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"	background: #191919;\n"
"	border: 1px solid #5A5A5A;\n"
"	padding: 4px;\n"
"}\n"
"\n"
"QHeaderView::section:selected, QHeaderView::section::checked {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QTableView {\n"
"	gridline-color: #5A5A5A;\n"
"}\n"
"\n"
"QTabBar {\n"
"	margin-left: 2px;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"	border-radius: 0px;\n"
"	padding: 4px;\n"
"	margin: 4px;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"	border: 1px solid #5A5A5A;\n"
"	back"
                        "ground: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox {\n"
"	padding-right: 15px;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"	subcontrol-origin: border;\n"
"}\n"
"QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"QAbstractSpinBox::up-arrow:hover, QAbstractSpinBox::down-arrow:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QSlider {\n"
"	border: none;\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"	height: 5px;\n"
"	margin: 4px 0px 4px 0px;\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"	width: 5px;"
                        "\n"
"	margin: 0px 4px 0px 4px;\n"
"}\n"
"\n"
"QSlider::handle {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"	width: 15px;\n"
"	margin: -4px 0px -4px 0px;\n"
"}\n"
"\n"
"QSlider::handle:vertical {\n"
"	height: 15px;\n"
"	margin: 0px -4px 0px -4px;\n"
"}\n"
"\n"
"QSlider::add-page:vertical, QSlider::sub-page:horizontal {\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QSlider::sub-page:vertical, QSlider::add-page:horizontal {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QLabel {\n"
"	border: none;\n"
"}\n"
"\n"
"QProgressBar {\n"
"	text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"	width: 1px;\n"
"	background-color: #2158AA;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QGroupBox {\n"
"	font: bold\n"
"}\n"
"\n"
"QListWidget::item:selected {\n"
"    background-color: #2158AA;\n"
"}\n"
"\n"
" QMenu::item:selected{\n"
"	background: #353535;\n"
"	border: 2px solid #D66C23;\n"
"    border-style: none solid none none;\n"
"}\n"
""
                        "\n"
" QMenu::pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"    height: 5px;\n"
"	background: #2158AA;\n"
"}"));
        verticalLayout_9 = new QVBoxLayout(image_selection_box);
        verticalLayout_9->setSpacing(6);
        verticalLayout_9->setContentsMargins(11, 11, 11, 11);
        verticalLayout_9->setObjectName(QString::fromUtf8("verticalLayout_9"));
        verticalLayout_6 = new QVBoxLayout();
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(6);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        horizontalLayout_7->setContentsMargins(-1, 10, -1, 10);
        camera_A_radio_button = new QRadioButton(image_selection_box);
        camera_A_radio_button->setObjectName(QString::fromUtf8("camera_A_radio_button"));
        sizePolicy3.setHeightForWidth(camera_A_radio_button->sizePolicy().hasHeightForWidth());
        camera_A_radio_button->setSizePolicy(sizePolicy3);
        camera_A_radio_button->setChecked(true);

        horizontalLayout_7->addWidget(camera_A_radio_button);

        camera_B_radio_button = new QRadioButton(image_selection_box);
        camera_B_radio_button->setObjectName(QString::fromUtf8("camera_B_radio_button"));
        camera_B_radio_button->setEnabled(true);
        sizePolicy3.setHeightForWidth(camera_B_radio_button->sizePolicy().hasHeightForWidth());
        camera_B_radio_button->setSizePolicy(sizePolicy3);

        horizontalLayout_7->addWidget(camera_B_radio_button);


        verticalLayout_6->addLayout(horizontalLayout_7);

        image_list_widget = new QListWidget(image_selection_box);
        image_list_widget->setObjectName(QString::fromUtf8("image_list_widget"));
        QSizePolicy sizePolicy5(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(image_list_widget->sizePolicy().hasHeightForWidth());
        image_list_widget->setSizePolicy(sizePolicy5);

        verticalLayout_6->addWidget(image_list_widget);


        verticalLayout_9->addLayout(verticalLayout_6);


        Left->addWidget(image_selection_box);

        Left->setStretch(0, 1);
        Left->setStretch(1, 1);
        Left->setStretch(3, 2);

        horizontalLayout_8->addLayout(Left);

        gridLayout_3 = new QGridLayout();
        gridLayout_3->setSpacing(6);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        pose_progress = new QProgressBar(centralWidget);
        pose_progress->setObjectName(QString::fromUtf8("pose_progress"));
        QSizePolicy sizePolicy6(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
        sizePolicy6.setHorizontalStretch(0);
        sizePolicy6.setVerticalStretch(0);
        sizePolicy6.setHeightForWidth(pose_progress->sizePolicy().hasHeightForWidth());
        pose_progress->setSizePolicy(sizePolicy6);
        QPalette palette;
        QBrush brush(QColor(33, 88, 170, 255));
        brush.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::WindowText, brush);
        QBrush brush1(QColor(214, 108, 35, 255));
        brush1.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Button, brush1);
        palette.setBrush(QPalette::Active, QPalette::Text, brush);
        palette.setBrush(QPalette::Active, QPalette::ButtonText, brush);
        palette.setBrush(QPalette::Active, QPalette::Base, brush1);
        palette.setBrush(QPalette::Active, QPalette::Window, brush1);
        QBrush brush2(QColor(255, 170, 0, 255));
        brush2.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Highlight, brush2);
        QBrush brush3(QColor(33, 88, 170, 128));
        brush3.setStyle(Qt::NoBrush);
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette::Active, QPalette::PlaceholderText, brush3);
#endif
        palette.setBrush(QPalette::Inactive, QPalette::WindowText, brush);
        palette.setBrush(QPalette::Inactive, QPalette::Button, brush1);
        palette.setBrush(QPalette::Inactive, QPalette::Text, brush);
        palette.setBrush(QPalette::Inactive, QPalette::ButtonText, brush);
        palette.setBrush(QPalette::Inactive, QPalette::Base, brush1);
        palette.setBrush(QPalette::Inactive, QPalette::Window, brush1);
        palette.setBrush(QPalette::Inactive, QPalette::Highlight, brush2);
        QBrush brush4(QColor(33, 88, 170, 128));
        brush4.setStyle(Qt::NoBrush);
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette::Inactive, QPalette::PlaceholderText, brush4);
#endif
        palette.setBrush(QPalette::Disabled, QPalette::WindowText, brush);
        palette.setBrush(QPalette::Disabled, QPalette::Button, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Text, brush);
        palette.setBrush(QPalette::Disabled, QPalette::ButtonText, brush);
        palette.setBrush(QPalette::Disabled, QPalette::Base, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Window, brush1);
        QBrush brush5(QColor(0, 120, 215, 255));
        brush5.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Disabled, QPalette::Highlight, brush5);
        QBrush brush6(QColor(33, 88, 170, 128));
        brush6.setStyle(Qt::NoBrush);
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette::Disabled, QPalette::PlaceholderText, brush6);
#endif
        pose_progress->setPalette(palette);
        QFont font1;
        font1.setPointSize(12);
        font1.setBold(true);
        font1.setWeight(75);
        pose_progress->setFont(font1);
        pose_progress->setStyleSheet(QString::fromUtf8("QProgressBar {\n"
"/*\n"
"	COLOR_DARK     = #191919\n"
"	COLOR_MEDIUM   = #353535\n"
"	COLOR_MEDLIGHT = #5A5A5A\n"
"	COLOR_LIGHT    = #DDDDDD\n"
"	COLOR_BLUE  = #2158AA\n"
"	COLOR_ORANGE = #D66C23\n"
"*/\n"
" background-color: #D66C23;\n"
"text-color: #2158AA;\n"
"color: #2158AA;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"    background-color: #D66C23;\n"
"}"));
        pose_progress->setValue(24);

        horizontalLayout_3->addWidget(pose_progress);

        pose_label = new QLabel(centralWidget);
        pose_label->setObjectName(QString::fromUtf8("pose_label"));
        QSizePolicy sizePolicy7(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy7.setHorizontalStretch(0);
        sizePolicy7.setVerticalStretch(0);
        sizePolicy7.setHeightForWidth(pose_label->sizePolicy().hasHeightForWidth());
        pose_label->setSizePolicy(sizePolicy7);
        pose_label->setFont(font1);

        horizontalLayout_3->addWidget(pose_label);


        gridLayout_3->addLayout(horizontalLayout_3, 0, 0, 1, 1);

        qvtk_widget = new QVTKOpenGLNativeWidget(centralWidget);
        qvtk_widget->setObjectName(QString::fromUtf8("qvtk_widget"));
        QSizePolicy sizePolicy8(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy8.setHorizontalStretch(1);
        sizePolicy8.setVerticalStretch(1);
        sizePolicy8.setHeightForWidth(qvtk_widget->sizePolicy().hasHeightForWidth());
        qvtk_widget->setSizePolicy(sizePolicy8);
        qvtk_widget->setSizeIncrement(QSize(1, 1));
        qvtk_widget->setBaseSize(QSize(1, 1));

        gridLayout_3->addWidget(qvtk_widget, 1, 0, 1, 1);


        horizontalLayout_8->addLayout(gridLayout_3);

        Right = new QVBoxLayout();
        Right->setSpacing(6);
        Right->setObjectName(QString::fromUtf8("Right"));
        qvtk_cpv = new QVTKOpenGLNativeWidget(centralWidget);
        qvtk_cpv->setObjectName(QString::fromUtf8("qvtk_cpv"));
        QSizePolicy sizePolicy9(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
        sizePolicy9.setHorizontalStretch(1);
        sizePolicy9.setVerticalStretch(1);
        sizePolicy9.setHeightForWidth(qvtk_cpv->sizePolicy().hasHeightForWidth());
        qvtk_cpv->setSizePolicy(sizePolicy9);
        qvtk_cpv->setSizeIncrement(QSize(1, 1));
        qvtk_cpv->setBaseSize(QSize(1, 1));
        qvtk_cpv->setStyleSheet(QString::fromUtf8("*{border: 2px solid white}"));

        Right->addWidget(qvtk_cpv);

        edge_detection_box = new QGroupBox(centralWidget);
        edge_detection_box->setObjectName(QString::fromUtf8("edge_detection_box"));
        sizePolicy1.setHeightForWidth(edge_detection_box->sizePolicy().hasHeightForWidth());
        edge_detection_box->setSizePolicy(sizePolicy1);
        edge_detection_box->setStyleSheet(QString::fromUtf8("/*\n"
"	Based on a template by Emanuel Claesson (c) 2013 with modifications by Paris Flood\n"
"\n"
"	Licensed under the Apache License, Version 2.0 (the \"License\");\n"
"	you may not use this file except in compliance with the License.\n"
"	You may obtain a copy of the License at\n"
"\n"
"		http://www.apache.org/licenses/LICENSE-2.0\n"
"\n"
"	Unless required by applicable law or agreed to in writing, software\n"
"	distributed under the License is distributed on an \"AS IS\" BASIS,\n"
"	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
"	See the License for the specific language governing permissions and\n"
"	limitations under the License.\n"
"*/\n"
"\n"
"/*\n"
"	COLOR_DARK     = #191919\n"
"	COLOR_MEDIUM   = #353535\n"
"	COLOR_MEDLIGHT = #5A5A5A\n"
"	COLOR_LIGHT    = #DDDDDD\n"
"	COLOR_BLUE  = #2158AA\n"
"	COLOR_ORANGE = #D66C23\n"
"*/\n"
"* {\n"
"	background: #191919;\n"
"	color: #DDDDDD;\n"
"border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
" QMenuBar {\n"
"	background: #191919;\n"
"	border"
                        "-style: none;\n"
" }\n"
"\n"
" QMenuBar::item {\n"
"	 background: transparent;\n"
"     spacing: 3px; /* spacing between menu bar items */\n"
"     padding: 5px 25px;\n"
" }\n"
"\n"
" QMenuBar::item:selected{\n"
"	background: #353535;\n"
"  	border: 2px solid #D66C23;\n"
"    border-style: none none solid none;\n"
"}\n"
"\n"
" QMenuBar::item:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QCheckBox, QRadioButton {\n"
"	border: none;\n"
"}\n"
"\n"
"QRadioButton::indicator, QCheckBox::indicator {\n"
"	width: 13px;\n"
"	height: 13px;\n"
"}\n"
"\n"
"QRadioButton::indicator::unchecked, QCheckBox::indicator::unchecked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: none;\n"
"}\n"
"\n"
"QRadioButton::indicator:unchecked:hover, QCheckBox::indicator:unchecked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QRadioButton::indicator::checked, QCheckBox::indicator::checked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QRadioButton::in"
                        "dicator:checked:hover, QCheckBox::indicator:checked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"	subcontrol-origin: margin;\n"
"    subcontrol-position: top center;\n"
"}\n"
"\n"
"QScrollBar {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #191919;\n"
"}\n"
"\n"
"QScrollBar:horizontal {\n"
"	height: 15px;\n"
"	margin: 0px 16px 0px 16px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"	width: 15px;\n"
"	margin: 16px 0px 16px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle {\n"
"	background: #353535;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	border-width: 0px 1px 0px 1px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	border-width: 1px 0px 1px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	min-width: 20px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	min-height: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-line, QScrollBar::sub-line {\n"
"	background:#353535;\n"
"	border: 1px solid #5A5A5A;\n"
"	subc"
                        "ontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::add-line {\n"
"	position: absolute;\n"
"}\n"
"\n"
"QScrollBar::add-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: right;\n"
"	left: 15px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: bottom;\n"
"	top: 15px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: left;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: top;\n"
"}\n"
"\n"
"QScrollBar:left-arrow, QScrollBar::right-arrow, QScrollBar::up-arrow, QScrollBar::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	width: 3px;\n"
"	height: 3px;\n"
"}\n"
"\n"
"QScrollBar::add-page, QScrollBar::sub-page {\n"
"	background: none;\n"
"}\n"
"\n"
"QAbstractButton {\n"
"	background: #252525;\n"
"}\n"
"\n"
"QAbstractButton:disabled {\n"
"	background: #3F3F3F;\n"
"}\n"
"\n"
"QAbstractButton:hover {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"	background: #35353"
                        "5;\n"
"  	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QAbstractButton:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QAbstractItemView {\n"
"	show-decoration-selected: 1;\n"
" 	outline: 0;\n"
"	selection-background-color: #2158AA;\n"
"	selection-color: #DDDDDD;\n"
"	alternate-background-color: #353535;\n"
"}\n"
"\n"
"QHeaderView {\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"	background: #191919;\n"
"	border: 1px solid #5A5A5A;\n"
"	padding: 4px;\n"
"}\n"
"\n"
"QHeaderView::section:selected, QHeaderView::section::checked {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QTableView {\n"
"	gridline-color: #5A5A5A;\n"
"}\n"
"\n"
"QTabBar {\n"
"	margin-left: 2px;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"	border-radius: 0px;\n"
"	padding: 4px;\n"
"	margin: 4px;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"	border: 1px solid #5A5A5A;\n"
"	back"
                        "ground: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox {\n"
"	padding-right: 15px;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"	subcontrol-origin: border;\n"
"}\n"
"QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"QAbstractSpinBox::up-arrow:hover, QAbstractSpinBox::down-arrow:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QSlider {\n"
"	border: none;\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"	height: 5px;\n"
"	margin: 4px 0px 4px 0px;\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"	width: 5px;"
                        "\n"
"	margin: 0px 4px 0px 4px;\n"
"}\n"
"\n"
"QSlider::handle {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"	width: 15px;\n"
"	margin: -4px 0px -4px 0px;\n"
"}\n"
"\n"
"QSlider::handle:vertical {\n"
"	height: 15px;\n"
"	margin: 0px -4px 0px -4px;\n"
"}\n"
"\n"
"QSlider::add-page:vertical, QSlider::sub-page:horizontal {\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QSlider::sub-page:vertical, QSlider::add-page:horizontal {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QLabel {\n"
"	border: none;\n"
"}\n"
"\n"
"QProgressBar {\n"
"	text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"	width: 1px;\n"
"	background-color: #2158AA;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QGroupBox {\n"
"	font: bold\n"
"}\n"
"\n"
"QListWidget::item:selected {\n"
"    background-color: #2158AA;\n"
"}\n"
"\n"
" QMenu::item:selected{\n"
"	background: #353535;\n"
"	border: 2px solid #D66C23;\n"
"    border-style: none solid none none;\n"
"}\n"
""
                        "\n"
" QMenu::pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"    height: 5px;\n"
"	background: #2158AA;\n"
"}"));
        edge_detection_box->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        verticalLayout_13 = new QVBoxLayout(edge_detection_box);
        verticalLayout_13->setSpacing(6);
        verticalLayout_13->setContentsMargins(11, 11, 11, 11);
        verticalLayout_13->setObjectName(QString::fromUtf8("verticalLayout_13"));
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        aperture_label = new QLabel(edge_detection_box);
        aperture_label->setObjectName(QString::fromUtf8("aperture_label"));
        sizePolicy7.setHeightForWidth(aperture_label->sizePolicy().hasHeightForWidth());
        aperture_label->setSizePolicy(sizePolicy7);

        horizontalLayout_4->addWidget(aperture_label);

        aperture_spin_box = new QSpinBox(edge_detection_box);
        aperture_spin_box->setObjectName(QString::fromUtf8("aperture_spin_box"));
        aperture_spin_box->setMinimum(3);
        aperture_spin_box->setMaximum(7);
        aperture_spin_box->setSingleStep(2);

        horizontalLayout_4->addWidget(aperture_spin_box);


        verticalLayout_3->addLayout(horizontalLayout_4);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(6);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        low_threshold_label = new QLabel(edge_detection_box);
        low_threshold_label->setObjectName(QString::fromUtf8("low_threshold_label"));
        sizePolicy7.setHeightForWidth(low_threshold_label->sizePolicy().hasHeightForWidth());
        low_threshold_label->setSizePolicy(sizePolicy7);

        horizontalLayout_5->addWidget(low_threshold_label);

        low_threshold_value = new QLabel(edge_detection_box);
        low_threshold_value->setObjectName(QString::fromUtf8("low_threshold_value"));
        sizePolicy7.setHeightForWidth(low_threshold_value->sizePolicy().hasHeightForWidth());
        low_threshold_value->setSizePolicy(sizePolicy7);

        horizontalLayout_5->addWidget(low_threshold_value);

        low_threshold_slider = new QSlider(edge_detection_box);
        low_threshold_slider->setObjectName(QString::fromUtf8("low_threshold_slider"));
        low_threshold_slider->setMinimum(0);
        low_threshold_slider->setValue(40);
        low_threshold_slider->setOrientation(Qt::Horizontal);

        horizontalLayout_5->addWidget(low_threshold_slider);


        verticalLayout_3->addLayout(horizontalLayout_5);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setSpacing(6);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        high_threshold_label = new QLabel(edge_detection_box);
        high_threshold_label->setObjectName(QString::fromUtf8("high_threshold_label"));
        sizePolicy7.setHeightForWidth(high_threshold_label->sizePolicy().hasHeightForWidth());
        high_threshold_label->setSizePolicy(sizePolicy7);

        horizontalLayout_6->addWidget(high_threshold_label);

        high_threshold_value = new QLabel(edge_detection_box);
        high_threshold_value->setObjectName(QString::fromUtf8("high_threshold_value"));
        sizePolicy7.setHeightForWidth(high_threshold_value->sizePolicy().hasHeightForWidth());
        high_threshold_value->setSizePolicy(sizePolicy7);

        horizontalLayout_6->addWidget(high_threshold_value);

        high_threshold_slider = new QSlider(edge_detection_box);
        high_threshold_slider->setObjectName(QString::fromUtf8("high_threshold_slider"));
        high_threshold_slider->setMaximum(300);
        high_threshold_slider->setOrientation(Qt::Horizontal);

        horizontalLayout_6->addWidget(high_threshold_slider);


        verticalLayout_3->addLayout(horizontalLayout_6);

        apply_all_edge_button = new QPushButton(edge_detection_box);
        apply_all_edge_button->setObjectName(QString::fromUtf8("apply_all_edge_button"));
        QSizePolicy sizePolicy10(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy10.setHorizontalStretch(0);
        sizePolicy10.setVerticalStretch(0);
        sizePolicy10.setHeightForWidth(apply_all_edge_button->sizePolicy().hasHeightForWidth());
        apply_all_edge_button->setSizePolicy(sizePolicy10);

        verticalLayout_3->addWidget(apply_all_edge_button);

        reset_edge_button = new QPushButton(edge_detection_box);
        reset_edge_button->setObjectName(QString::fromUtf8("reset_edge_button"));

        verticalLayout_3->addWidget(reset_edge_button);


        verticalLayout_13->addLayout(verticalLayout_3);


        Right->addWidget(edge_detection_box);

        model_view_box = new QGroupBox(centralWidget);
        model_view_box->setObjectName(QString::fromUtf8("model_view_box"));
        sizePolicy1.setHeightForWidth(model_view_box->sizePolicy().hasHeightForWidth());
        model_view_box->setSizePolicy(sizePolicy1);
        model_view_box->setMaximumSize(QSize(16777215, 263));
        model_view_box->setStyleSheet(QString::fromUtf8("/*\n"
"	Based on a template by Emanuel Claesson (c) 2013 with modifications by Paris Flood\n"
"\n"
"	Licensed under the Apache License, Version 2.0 (the \"License\");\n"
"	you may not use this file except in compliance with the License.\n"
"	You may obtain a copy of the License at\n"
"\n"
"		http://www.apache.org/licenses/LICENSE-2.0\n"
"\n"
"	Unless required by applicable law or agreed to in writing, software\n"
"	distributed under the License is distributed on an \"AS IS\" BASIS,\n"
"	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
"	See the License for the specific language governing permissions and\n"
"	limitations under the License.\n"
"*/\n"
"\n"
"/*\n"
"	COLOR_DARK     = #191919\n"
"	COLOR_MEDIUM   = #353535\n"
"	COLOR_MEDLIGHT = #5A5A5A\n"
"	COLOR_LIGHT    = #DDDDDD\n"
"	COLOR_BLUE  = #2158AA\n"
"	COLOR_ORANGE = #D66C23\n"
"*/\n"
"* {\n"
"	background: #191919;\n"
"	color: #DDDDDD;\n"
"border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
" QMenuBar {\n"
"	background: #191919;\n"
"	border"
                        "-style: none;\n"
" }\n"
"\n"
" QMenuBar::item {\n"
"	 background: transparent;\n"
"     spacing: 3px; /* spacing between menu bar items */\n"
"     padding: 5px 25px;\n"
" }\n"
"\n"
" QMenuBar::item:selected{\n"
"	background: #353535;\n"
"  	border: 2px solid #D66C23;\n"
"    border-style: none none solid none;\n"
"}\n"
"\n"
" QMenuBar::item:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QCheckBox, QRadioButton {\n"
"	border: none;\n"
"}\n"
"\n"
"QRadioButton::indicator, QCheckBox::indicator {\n"
"	width: 13px;\n"
"	height: 13px;\n"
"}\n"
"\n"
"QRadioButton::indicator::unchecked, QCheckBox::indicator::unchecked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: none;\n"
"}\n"
"\n"
"QRadioButton::indicator:unchecked:hover, QCheckBox::indicator:unchecked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QRadioButton::indicator::checked, QCheckBox::indicator::checked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QRadioButton::in"
                        "dicator:checked:hover, QCheckBox::indicator:checked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"	subcontrol-origin: margin;\n"
"    subcontrol-position: top center;\n"
"}\n"
"\n"
"QScrollBar {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #191919;\n"
"}\n"
"\n"
"QScrollBar:horizontal {\n"
"	height: 15px;\n"
"	margin: 0px 16px 0px 16px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"	width: 15px;\n"
"	margin: 16px 0px 16px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle {\n"
"	background: #353535;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	border-width: 0px 1px 0px 1px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	border-width: 1px 0px 1px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	min-width: 20px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	min-height: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-line, QScrollBar::sub-line {\n"
"	background:#353535;\n"
"	border: 1px solid #5A5A5A;\n"
"	subc"
                        "ontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::add-line {\n"
"	position: absolute;\n"
"}\n"
"\n"
"QScrollBar::add-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: right;\n"
"	left: 15px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: bottom;\n"
"	top: 15px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: left;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: top;\n"
"}\n"
"\n"
"QScrollBar:left-arrow, QScrollBar::right-arrow, QScrollBar::up-arrow, QScrollBar::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	width: 3px;\n"
"	height: 3px;\n"
"}\n"
"\n"
"QScrollBar::add-page, QScrollBar::sub-page {\n"
"	background: none;\n"
"}\n"
"\n"
"QAbstractButton {\n"
"	background: #252525;\n"
"}\n"
"\n"
"QAbstractButton:disabled {\n"
"	background: #3F3F3F;\n"
"}\n"
"\n"
"QAbstractButton:hover {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"	background: #35353"
                        "5;\n"
"  	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QAbstractButton:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QAbstractItemView {\n"
"	show-decoration-selected: 1;\n"
" 	outline: 0;\n"
"	selection-background-color: #2158AA;\n"
"	selection-color: #DDDDDD;\n"
"	alternate-background-color: #353535;\n"
"}\n"
"\n"
"QHeaderView {\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"	background: #191919;\n"
"	border: 1px solid #5A5A5A;\n"
"	padding: 4px;\n"
"}\n"
"\n"
"QHeaderView::section:selected, QHeaderView::section::checked {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QTableView {\n"
"	gridline-color: #5A5A5A;\n"
"}\n"
"\n"
"QTabBar {\n"
"	margin-left: 2px;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"	border-radius: 0px;\n"
"	padding: 4px;\n"
"	margin: 4px;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"	border: 1px solid #5A5A5A;\n"
"	back"
                        "ground: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox {\n"
"	padding-right: 15px;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"	subcontrol-origin: border;\n"
"}\n"
"QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"QAbstractSpinBox::up-arrow:hover, QAbstractSpinBox::down-arrow:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QSlider {\n"
"	border: none;\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"	height: 5px;\n"
"	margin: 4px 0px 4px 0px;\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"	width: 5px;"
                        "\n"
"	margin: 0px 4px 0px 4px;\n"
"}\n"
"\n"
"QSlider::handle {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"	width: 15px;\n"
"	margin: -4px 0px -4px 0px;\n"
"}\n"
"\n"
"QSlider::handle:vertical {\n"
"	height: 15px;\n"
"	margin: 0px -4px 0px -4px;\n"
"}\n"
"\n"
"QSlider::add-page:vertical, QSlider::sub-page:horizontal {\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QSlider::sub-page:vertical, QSlider::add-page:horizontal {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QLabel {\n"
"	border: none;\n"
"}\n"
"\n"
"QProgressBar {\n"
"	text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"	width: 1px;\n"
"	background-color: #2158AA;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QGroupBox {\n"
"	font: bold\n"
"}\n"
"\n"
"QListWidget::item:selected {\n"
"    background-color: #2158AA;\n"
"}\n"
"\n"
" QMenu::item:selected{\n"
"	background: #353535;\n"
"	border: 2px solid #D66C23;\n"
"    border-style: none solid none none;\n"
"}\n"
""
                        "\n"
" QMenu::pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"    height: 5px;\n"
"	background: #2158AA;\n"
"}"));
        model_view_box->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        verticalLayout_15 = new QVBoxLayout(model_view_box);
        verticalLayout_15->setSpacing(6);
        verticalLayout_15->setContentsMargins(11, 11, 11, 11);
        verticalLayout_15->setObjectName(QString::fromUtf8("verticalLayout_15"));
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        verticalLayout_12 = new QVBoxLayout();
        verticalLayout_12->setSpacing(6);
        verticalLayout_12->setObjectName(QString::fromUtf8("verticalLayout_12"));
        original_model_radio_button = new QRadioButton(model_view_box);
        original_model_radio_button->setObjectName(QString::fromUtf8("original_model_radio_button"));
        sizePolicy2.setHeightForWidth(original_model_radio_button->sizePolicy().hasHeightForWidth());
        original_model_radio_button->setSizePolicy(sizePolicy2);
        original_model_radio_button->setLayoutDirection(Qt::LeftToRight);
        original_model_radio_button->setChecked(true);

        verticalLayout_12->addWidget(original_model_radio_button);

        transparent_model_radio_button = new QRadioButton(model_view_box);
        transparent_model_radio_button->setObjectName(QString::fromUtf8("transparent_model_radio_button"));
        sizePolicy2.setHeightForWidth(transparent_model_radio_button->sizePolicy().hasHeightForWidth());
        transparent_model_radio_button->setSizePolicy(sizePolicy2);

        verticalLayout_12->addWidget(transparent_model_radio_button);


        horizontalLayout_2->addLayout(verticalLayout_12);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        solid_model_radio_button = new QRadioButton(model_view_box);
        solid_model_radio_button->setObjectName(QString::fromUtf8("solid_model_radio_button"));
        sizePolicy2.setHeightForWidth(solid_model_radio_button->sizePolicy().hasHeightForWidth());
        solid_model_radio_button->setSizePolicy(sizePolicy2);

        verticalLayout_2->addWidget(solid_model_radio_button);

        wireframe_model_radio_button = new QRadioButton(model_view_box);
        wireframe_model_radio_button->setObjectName(QString::fromUtf8("wireframe_model_radio_button"));
        sizePolicy2.setHeightForWidth(wireframe_model_radio_button->sizePolicy().hasHeightForWidth());
        wireframe_model_radio_button->setSizePolicy(sizePolicy2);

        verticalLayout_2->addWidget(wireframe_model_radio_button);


        horizontalLayout_2->addLayout(verticalLayout_2);


        verticalLayout_15->addLayout(horizontalLayout_2);


        Right->addWidget(model_view_box);

        model_selection_box = new QGroupBox(centralWidget);
        model_selection_box->setObjectName(QString::fromUtf8("model_selection_box"));
        sizePolicy4.setHeightForWidth(model_selection_box->sizePolicy().hasHeightForWidth());
        model_selection_box->setSizePolicy(sizePolicy4);
        model_selection_box->setStyleSheet(QString::fromUtf8("/*\n"
"	Based on a template by Emanuel Claesson (c) 2013 with modifications by Paris Flood\n"
"\n"
"	Licensed under the Apache License, Version 2.0 (the \"License\");\n"
"	you may not use this file except in compliance with the License.\n"
"	You may obtain a copy of the License at\n"
"\n"
"		http://www.apache.org/licenses/LICENSE-2.0\n"
"\n"
"	Unless required by applicable law or agreed to in writing, software\n"
"	distributed under the License is distributed on an \"AS IS\" BASIS,\n"
"	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
"	See the License for the specific language governing permissions and\n"
"	limitations under the License.\n"
"*/\n"
"\n"
"/*\n"
"	COLOR_DARK     = #191919\n"
"	COLOR_MEDIUM   = #353535\n"
"	COLOR_MEDLIGHT = #5A5A5A\n"
"	COLOR_LIGHT    = #DDDDDD\n"
"	COLOR_BLUE  = #2158AA\n"
"	COLOR_ORANGE = #D66C23\n"
"*/\n"
"* {\n"
"	background: #191919;\n"
"	color: #DDDDDD;\n"
"border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
" QMenuBar {\n"
"	background: #191919;\n"
"	border"
                        "-style: none;\n"
" }\n"
"\n"
" QMenuBar::item {\n"
"	 background: transparent;\n"
"     spacing: 3px; /* spacing between menu bar items */\n"
"     padding: 5px 25px;\n"
" }\n"
"\n"
" QMenuBar::item:selected{\n"
"	background: #353535;\n"
"  	border: 2px solid #D66C23;\n"
"    border-style: none none solid none;\n"
"}\n"
"\n"
" QMenuBar::item:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QCheckBox, QRadioButton {\n"
"	border: none;\n"
"}\n"
"\n"
"QRadioButton::indicator, QCheckBox::indicator {\n"
"	width: 13px;\n"
"	height: 13px;\n"
"}\n"
"\n"
"QRadioButton::indicator::unchecked, QCheckBox::indicator::unchecked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: none;\n"
"}\n"
"\n"
"QRadioButton::indicator:unchecked:hover, QCheckBox::indicator:unchecked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QRadioButton::indicator::checked, QCheckBox::indicator::checked {\n"
"	left: 5px;\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QRadioButton::in"
                        "dicator:checked:hover, QCheckBox::indicator:checked:hover {\n"
"	left: 5px;\n"
"	border: 1px solid #D66C23;\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"	subcontrol-origin: margin;\n"
"    subcontrol-position: top center;\n"
"}\n"
"\n"
"QScrollBar {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #191919;\n"
"}\n"
"\n"
"QScrollBar:horizontal {\n"
"	height: 15px;\n"
"	margin: 0px 16px 0px 16px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"	width: 15px;\n"
"	margin: 16px 0px 16px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle {\n"
"	background: #353535;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	border-width: 0px 1px 0px 1px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	border-width: 1px 0px 1px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"	min-width: 20px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"	min-height: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-line, QScrollBar::sub-line {\n"
"	background:#353535;\n"
"	border: 1px solid #5A5A5A;\n"
"	subc"
                        "ontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::add-line {\n"
"	position: absolute;\n"
"}\n"
"\n"
"QScrollBar::add-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: right;\n"
"	left: 15px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: bottom;\n"
"	top: 15px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:horizontal {\n"
"	width: 15px;\n"
"	subcontrol-position: left;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"	height: 15px;\n"
"	subcontrol-position: top;\n"
"}\n"
"\n"
"QScrollBar:left-arrow, QScrollBar::right-arrow, QScrollBar::up-arrow, QScrollBar::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	width: 3px;\n"
"	height: 3px;\n"
"}\n"
"\n"
"QScrollBar::add-page, QScrollBar::sub-page {\n"
"	background: none;\n"
"}\n"
"\n"
"QAbstractButton {\n"
"	background: #252525;\n"
"}\n"
"\n"
"QAbstractButton:disabled {\n"
"	background: #3F3F3F;\n"
"}\n"
"\n"
"QAbstractButton:hover {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"	background: #35353"
                        "5;\n"
"  	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QAbstractButton:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"/*QAbstractItemView {\n"
"	show-decoration-selected: 1;\n"
" 	outline: 0;\n"
"	selection-background-color: #2158AA;\n"
"	selection-color: #DDDDDD;\n"
"	alternate-background-color: #353535;\n"
"}*/\n"
"\n"
"QHeaderView {\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"	background: #191919;\n"
"	border: 1px solid #5A5A5A;\n"
"	padding: 4px;\n"
"}\n"
"\n"
"QHeaderView::section:selected, QHeaderView::section::checked {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QTableView {\n"
"	gridline-color: #5A5A5A;\n"
"}\n"
"\n"
"QTabBar {\n"
"	margin-left: 2px;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"	border-radius: 0px;\n"
"	padding: 4px;\n"
"	margin: 4px;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"	border: 1px solid #5A5A5A;\n"
"	"
                        "background: #353535;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox {\n"
"	padding-right: 15px;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"	subcontrol-origin: border;\n"
"}\n"
"QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {\n"
"	width: 3px;\n"
"	height: 3px;\n"
"	border: 1px solid #5A5A5A;\n"
"}\n"
"QAbstractSpinBox::up-arrow:hover, QAbstractSpinBox::down-arrow:hover {\n"
"	border: 1px solid #D66C23;\n"
"}\n"
"\n"
"QSlider {\n"
"	border: none;\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"	height: 5px;\n"
"	margin: 4px 0px 4px 0px;\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"	width: "
                        "5px;\n"
"	margin: 0px 4px 0px 4px;\n"
"}\n"
"\n"
"QSlider::handle {\n"
"	border: 1px solid #5A5A5A;\n"
"	background: #353535;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"	width: 15px;\n"
"	margin: -4px 0px -4px 0px;\n"
"}\n"
"\n"
"QSlider::handle:vertical {\n"
"	height: 15px;\n"
"	margin: 0px -4px 0px -4px;\n"
"}\n"
"\n"
"QSlider::add-page:vertical, QSlider::sub-page:horizontal {\n"
"	background: #2158AA;\n"
"}\n"
"\n"
"QSlider::sub-page:vertical, QSlider::add-page:horizontal {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QLabel {\n"
"	border: none;\n"
"}\n"
"\n"
"QProgressBar {\n"
"	text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"	width: 1px;\n"
"	background-color: #2158AA;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QGroupBox {\n"
"	font: bold\n"
"}\n"
"\n"
"/*QListWidget::item:selected {\n"
"    background-color: #2158AA;\n"
"}*/\n"
"\n"
" QMenu::item:selected{\n"
"	background: #353535;\n"
"	border: 2px solid #D66C23;\n"
"    border-style: none solid none none;\n"
""
                        "}\n"
"\n"
" QMenu::pressed {\n"
"	background: #5A5A5A;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"    height: 5px;\n"
"	background: #2158AA;\n"
"}"));
        model_selection_box->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        verticalLayout_14 = new QVBoxLayout(model_selection_box);
        verticalLayout_14->setSpacing(6);
        verticalLayout_14->setContentsMargins(11, 11, 11, 11);
        verticalLayout_14->setObjectName(QString::fromUtf8("verticalLayout_14"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 10, -1, 10);
        single_model_radio_button = new QRadioButton(model_selection_box);
        single_model_radio_button->setObjectName(QString::fromUtf8("single_model_radio_button"));
        sizePolicy2.setHeightForWidth(single_model_radio_button->sizePolicy().hasHeightForWidth());
        single_model_radio_button->setSizePolicy(sizePolicy2);
        single_model_radio_button->setChecked(true);

        horizontalLayout->addWidget(single_model_radio_button);

        multiple_model_radio_button = new QRadioButton(model_selection_box);
        multiple_model_radio_button->setObjectName(QString::fromUtf8("multiple_model_radio_button"));
        multiple_model_radio_button->setEnabled(true);
        sizePolicy2.setHeightForWidth(multiple_model_radio_button->sizePolicy().hasHeightForWidth());
        multiple_model_radio_button->setSizePolicy(sizePolicy2);

        horizontalLayout->addWidget(multiple_model_radio_button);


        verticalLayout->addLayout(horizontalLayout);

        model_list_widget = new QListWidget(model_selection_box);
        model_list_widget->setObjectName(QString::fromUtf8("model_list_widget"));
        model_list_widget->setStyleSheet(QString::fromUtf8(""));

        verticalLayout->addWidget(model_list_widget);

        verticalLayout->setStretch(1, 1);

        verticalLayout_14->addLayout(verticalLayout);


        Right->addWidget(model_selection_box);

        Right->setStretch(0, 1);
        Right->setStretch(3, 2);

        horizontalLayout_8->addLayout(Right);

        horizontalLayout_8->setStretch(0, 2);
        horizontalLayout_8->setStretch(1, 9);
        horizontalLayout_8->setStretch(2, 2);

        verticalLayout_4->addLayout(horizontalLayout_8);

        MainScreenClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainScreenClass);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1147, 29));
        QFont font2;
        font2.setBold(true);
        font2.setWeight(75);
        menuBar->setFont(font2);
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        QFont font3;
        font3.setBold(false);
        font3.setWeight(50);
        menuFile->setFont(font3);
        menuOptions = new QMenu(menuBar);
        menuOptions->setObjectName(QString::fromUtf8("menuOptions"));
        menuOptions->setFont(font3);
        menuOptimizer = new QMenu(menuBar);
        menuOptimizer->setObjectName(QString::fromUtf8("menuOptimizer"));
        menuOptimizer->setFont(font3);
        menuHelp = new QMenu(menuBar);
        menuHelp->setObjectName(QString::fromUtf8("menuHelp"));
        menuHelp->setFont(font3);
        menuVIEW = new QMenu(menuBar);
        menuVIEW->setObjectName(QString::fromUtf8("menuVIEW"));
        menuSegment = new QMenu(menuBar);
        menuSegment->setObjectName(QString::fromUtf8("menuSegment"));
        menuPOSE_ESTIMATE = new QMenu(menuBar);
        menuPOSE_ESTIMATE->setObjectName(QString::fromUtf8("menuPOSE_ESTIMATE"));
        menuSYMMETRY_TRAP_ANALYSIS = new QMenu(menuBar);
        menuSYMMETRY_TRAP_ANALYSIS->setObjectName(QString::fromUtf8("menuSYMMETRY_TRAP_ANALYSIS"));
        menuOPTIMZER = new QMenu(menuBar);
        menuOPTIMZER->setObjectName(QString::fromUtf8("menuOPTIMZER"));
        MainScreenClass->setMenuBar(menuBar);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuVIEW->menuAction());
        menuBar->addAction(menuOptions->menuAction());
        menuBar->addAction(menuOptimizer->menuAction());
        menuBar->addAction(menuHelp->menuAction());
        menuBar->addAction(menuSegment->menuAction());
        menuBar->addAction(menuPOSE_ESTIMATE->menuAction());
        menuBar->addAction(menuSYMMETRY_TRAP_ANALYSIS->menuAction());
        menuBar->addAction(menuOPTIMZER->menuAction());
        menuFile->addAction(actionLoad_Pose);
        menuFile->addAction(actionLoad_Kinematics);
        menuFile->addSeparator();
        menuFile->addAction(actionSave_Pose);
        menuFile->addAction(actionSave_Kinematics);
        menuFile->addSeparator();
        menuFile->addAction(actionQuit);
        menuFile->addSeparator();
        menuFile->addAction(actionCopy_Next_Pose);
        menuFile->addAction(actionCopy_Previous_Pose);
        menuOptions->addAction(actionRegion_Selection);
        menuOptions->addAction(actionCenter_Placement);
        menuOptions->addAction(actionStop_Optimizer);
        menuOptions->addAction(actionDRR_Settings);
        menuOptions->addAction(actionCoronal_Plane_Viewer);
        menuOptimizer->addAction(actionOptimizer_Settings);
        menuHelp->addAction(actionAbout_JointTrack_Auto);
        menuHelp->addAction(actionControls);
        menuVIEW->addAction(actionReset_View);
        menuVIEW->addAction(actionReset_Normal_Up);
        menuVIEW->addSeparator();
        menuVIEW->addAction(actionModel_Interaction_Mode);
        menuVIEW->addAction(actionCamera_Interaction_Mode);
        menuSegment->addSeparator();
        menuSegment->addAction(actionSegment_TibHR);
        menuSegment->addSeparator();
        menuSegment->addAction(actionBlack_Implant_Silhouettes_in_Original_Image_s);
        menuSegment->addAction(actionWhite_Implant_Silhouettes_in_Original_Image_s);
        menuSegment->addSeparator();
        menuSegment->addAction(actionReset_Remove_All_Segmentation);
        menuSegment->addSeparator();
        menuPOSE_ESTIMATE->addAction(actionEstimate_Femoral_Implant_s);
        menuPOSE_ESTIMATE->addSeparator();
        menuPOSE_ESTIMATE->addAction(actionEstimate_Tibial_Implant_s);
        menuPOSE_ESTIMATE->addSeparator();
        menuPOSE_ESTIMATE->addAction(actionNFD_Pose_Estimate);
        menuSYMMETRY_TRAP_ANALYSIS->addAction(actionLaunch_Tool);
        menuSYMMETRY_TRAP_ANALYSIS->addAction(actionAmbiguous_Pose_Processing);
        menuOPTIMZER->addAction(actionOptimize_Backward);

        retranslateUi(MainScreenClass);
        QObject::connect(actionQuit, SIGNAL(triggered()), MainScreenClass, SLOT(close()));

        QMetaObject::connectSlotsByName(MainScreenClass);
    } // setupUi

    void retranslateUi(QMainWindow *MainScreenClass)
    {
        MainScreenClass->setWindowTitle(QCoreApplication::translate("MainScreenClass", "JointTrack Machine Learning", nullptr));
        actionOptions->setText(QCoreApplication::translate("MainScreenClass", "Options", nullptr));
        actionOptions_2->setText(QCoreApplication::translate("MainScreenClass", "Options", nullptr));
        actionLoad_Pose->setText(QCoreApplication::translate("MainScreenClass", "Load Pose", nullptr));
        actionLoad_Kinematics->setText(QCoreApplication::translate("MainScreenClass", "Load Kinematics", nullptr));
        actionSave_Pose->setText(QCoreApplication::translate("MainScreenClass", "Save Pose", nullptr));
        actionSave_Kinematics->setText(QCoreApplication::translate("MainScreenClass", "Save Kinematics", nullptr));
        actionQuit->setText(QCoreApplication::translate("MainScreenClass", "Quit", nullptr));
        actionRegion_Selection->setText(QCoreApplication::translate("MainScreenClass", "Region Selection", nullptr));
        actionCenter_Placement->setText(QCoreApplication::translate("MainScreenClass", "Center Placement", nullptr));
        actionParallel_Tuner->setText(QCoreApplication::translate("MainScreenClass", "Parallel Tuner", nullptr));
        actionStop_Optimizer->setText(QCoreApplication::translate("MainScreenClass", "Stop Optimizer", nullptr));
        actionOptimizer_Settings->setText(QCoreApplication::translate("MainScreenClass", "Optimizer Settings", nullptr));
        actionAbout_JointTrack_Auto->setText(QCoreApplication::translate("MainScreenClass", "About", nullptr));
        actionControls->setText(QCoreApplication::translate("MainScreenClass", "Main Window Controls", nullptr));
        actionReset_View->setText(QCoreApplication::translate("MainScreenClass", "Reset View", nullptr));
        actionModel_Interaction_Mode->setText(QCoreApplication::translate("MainScreenClass", "Model Interaction Mode", nullptr));
        actionCamera_Interaction_Mode->setText(QCoreApplication::translate("MainScreenClass", "Camera Interaction Mode", nullptr));
        actionReset_Normal_Up->setText(QCoreApplication::translate("MainScreenClass", "Reset Normal Up", nullptr));
        actionDRR_Settings->setText(QCoreApplication::translate("MainScreenClass", "DRR Calibration Tool", nullptr));
        actionBlack_Implant_Silhouettes_in_Original_Image_s->setText(QCoreApplication::translate("MainScreenClass", "Black Implant Silhouettes in Original Image(s)", nullptr));
        actionWhite_Implant_Silhouettes_in_Original_Image_s->setText(QCoreApplication::translate("MainScreenClass", "White Implant Silhouettes in Original Image(s)", nullptr));
        actionReset_Remove_All_Segmentation->setText(QCoreApplication::translate("MainScreenClass", "Reset/Remove All Segmentation", nullptr));
        actionCustom_Segmentation->setText(QCoreApplication::translate("MainScreenClass", "Custom Segmentation", nullptr));
        actionSegment_FemHR->setText(QCoreApplication::translate("MainScreenClass", "Segment Femoral Implant(s) in High Resolution", nullptr));
        actionSegment_TibHR->setText(QCoreApplication::translate("MainScreenClass", "Segment Images", nullptr));
        actionEstimate_Femoral_Implant_s->setText(QCoreApplication::translate("MainScreenClass", "Estimate Femoral Implant(s)", nullptr));
        actionEstimate_Tibial_Implant_s->setText(QCoreApplication::translate("MainScreenClass", "Estimate Tibial Implant(s)", nullptr));
        actionCopy_Next_Pose->setText(QCoreApplication::translate("MainScreenClass", "Copy Next Pose", nullptr));
        actionCopy_Previous_Pose->setText(QCoreApplication::translate("MainScreenClass", "Copy Previous Pose", nullptr));
        actionLaunch_Tool->setText(QCoreApplication::translate("MainScreenClass", "Launch Tool", nullptr));
        actionOptimize_Backward->setText(QCoreApplication::translate("MainScreenClass", "Optimize Backward", nullptr));
        actionNFD_Pose_Estimate->setText(QCoreApplication::translate("MainScreenClass", "NFD Pose Estimate", nullptr));
        actionAmbiguous_Pose_Processing->setText(QCoreApplication::translate("MainScreenClass", "Ambiguous Pose Processing", nullptr));
        actionOpen_Viewer_Window->setText(QCoreApplication::translate("MainScreenClass", "Open Viewer Window", nullptr));
        actionCoronal_Plane_Viewer->setText(QCoreApplication::translate("MainScreenClass", "Coronal Plane Viewer", nullptr));
        preprocessor_box->setTitle(QCoreApplication::translate("MainScreenClass", "PREPROCESSOR", nullptr));
        load_calibration_button->setText(QCoreApplication::translate("MainScreenClass", "Load Calibration File", nullptr));
        load_image_button->setText(QCoreApplication::translate("MainScreenClass", "Load Fluoroscopic Image(s)", nullptr));
        load_model_button->setText(QCoreApplication::translate("MainScreenClass", "Load Implant Models", nullptr));
        optimization_box->setTitle(QCoreApplication::translate("MainScreenClass", "OPTIMIZATION DIRECTIVES", nullptr));
        optimize_button->setText(QCoreApplication::translate("MainScreenClass", "Optimize", nullptr));
        optimize_all_button->setText(QCoreApplication::translate("MainScreenClass", "Optimize All", nullptr));
        optimize_each_button->setText(QCoreApplication::translate("MainScreenClass", "Optimize Each", nullptr));
        optimize_from_button->setText(QCoreApplication::translate("MainScreenClass", "Optimize From", nullptr));
        image_view_box->setTitle(QCoreApplication::translate("MainScreenClass", "IMAGE VIEW", nullptr));
        original_image_radio_button->setText(QCoreApplication::translate("MainScreenClass", "Original", nullptr));
        inverted_image_radio_button->setText(QCoreApplication::translate("MainScreenClass", "Inverted/Segmented", nullptr));
        edges_image_radio_button->setText(QCoreApplication::translate("MainScreenClass", "Edges", nullptr));
        dilation_image_radio_button->setText(QCoreApplication::translate("MainScreenClass", "Dilation", nullptr));
        image_selection_box->setTitle(QCoreApplication::translate("MainScreenClass", "IMAGE SELECTION", nullptr));
        camera_A_radio_button->setText(QCoreApplication::translate("MainScreenClass", "Camera A", nullptr));
        camera_B_radio_button->setText(QCoreApplication::translate("MainScreenClass", "Camera B", nullptr));
        pose_label->setText(QCoreApplication::translate("MainScreenClass", "TextLabel", nullptr));
        edge_detection_box->setTitle(QCoreApplication::translate("MainScreenClass", "EDGE DETECTION", nullptr));
        aperture_label->setText(QCoreApplication::translate("MainScreenClass", "Aperture:", nullptr));
        low_threshold_label->setText(QCoreApplication::translate("MainScreenClass", "Low Threshold:", nullptr));
        low_threshold_value->setText(QCoreApplication::translate("MainScreenClass", "388", nullptr));
        high_threshold_label->setText(QCoreApplication::translate("MainScreenClass", "High Threshold:", nullptr));
        high_threshold_value->setText(QCoreApplication::translate("MainScreenClass", "388", nullptr));
        apply_all_edge_button->setText(QCoreApplication::translate("MainScreenClass", "Apply All", nullptr));
        reset_edge_button->setText(QCoreApplication::translate("MainScreenClass", "Reset", nullptr));
        model_view_box->setTitle(QCoreApplication::translate("MainScreenClass", "MODEL VIEW", nullptr));
        original_model_radio_button->setText(QCoreApplication::translate("MainScreenClass", "Original", nullptr));
        transparent_model_radio_button->setText(QCoreApplication::translate("MainScreenClass", "Transparent", nullptr));
        solid_model_radio_button->setText(QCoreApplication::translate("MainScreenClass", "Solid", nullptr));
        wireframe_model_radio_button->setText(QCoreApplication::translate("MainScreenClass", "Wire-frame", nullptr));
        model_selection_box->setTitle(QCoreApplication::translate("MainScreenClass", "MODEL SELECTION", nullptr));
        single_model_radio_button->setText(QCoreApplication::translate("MainScreenClass", "Single", nullptr));
        multiple_model_radio_button->setText(QCoreApplication::translate("MainScreenClass", "Multiple", nullptr));
        menuFile->setTitle(QCoreApplication::translate("MainScreenClass", "FILE", nullptr));
        menuOptions->setTitle(QCoreApplication::translate("MainScreenClass", "TOOLS", nullptr));
        menuOptimizer->setTitle(QCoreApplication::translate("MainScreenClass", "SETTINGS", nullptr));
        menuHelp->setTitle(QCoreApplication::translate("MainScreenClass", "HELP", nullptr));
        menuVIEW->setTitle(QCoreApplication::translate("MainScreenClass", "VIEW", nullptr));
        menuSegment->setTitle(QCoreApplication::translate("MainScreenClass", "SEGMENT", nullptr));
        menuPOSE_ESTIMATE->setTitle(QCoreApplication::translate("MainScreenClass", "POSE ESTIMATE", nullptr));
        menuSYMMETRY_TRAP_ANALYSIS->setTitle(QCoreApplication::translate("MainScreenClass", "SYMMETRY TRAP ANALYSIS", nullptr));
        menuOPTIMZER->setTitle(QCoreApplication::translate("MainScreenClass", "OPTIMZER", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainScreenClass: public Ui_MainScreenClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINSCREEN_H
