/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/********************************************************************************
** Form generated from reading UI file 'settings_control.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SETTINGS_CONTROL_H
#define UI_SETTINGS_CONTROL_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSpinBox>

QT_BEGIN_NAMESPACE

class Ui_settings_control
{
public:
    QPushButton *save_button;
    QPushButton *reset_button;
    QGroupBox *general_options_groupBox;
    QGroupBox *range_groupBox;
    QLabel *x_translation_label;
    QLabel *y_translation_label;
    QLabel *z_translation_label;
    QLabel *x_rotation_label;
    QLabel *y_rotation_label;
    QLabel *z_rotation_label;
    QSpinBox *x_translation_spinBox;
    QSpinBox *y_translation_spinBox;
    QSpinBox *z_translation_spinBox;
    QSpinBox *x_rotation_spinBox;
    QSpinBox *y_rotation_spinBox;
    QSpinBox *z_rotation_spinBox;
    QLabel *stage_budget_label;
    QSpinBox *budget_spinBox;
    QCheckBox *stage_enabled_checkBox;
    QGroupBox *stage_specific_groupBox;
    QSpinBox *branch_count_spinBox;
    QLabel *branch_total_count_label;
    QGroupBox *cost_function_groupBox;
    QListWidget *cost_function_listWidget;
    QGroupBox *cost_function_parameters_groupBox;
    QListWidget *cost_function_parameters_listWidget;
    QGroupBox *parameter_value_groupBox;
    QDoubleSpinBox *double_parameter_spinBox;
    QRadioButton *bool_parameter_true_radioButton;
    QRadioButton *bool_parameter_false_radioButton;
    QSpinBox *int_parameter_spinBox;
    QGroupBox *optimization_search_stage_groupBox;
    QRadioButton *trunk_radioButton;
    QRadioButton *branch_radioButton;
    QRadioButton *leaf_radioButton;
    QPushButton *cancel_button;

    void setupUi(QDialog *settings_control)
    {
        if (settings_control->objectName().isEmpty())
            settings_control->setObjectName(QString::fromUtf8("settings_control"));
        settings_control->resize(1089, 750);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/Desktop_Icon/Resources/jta_dime_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
        settings_control->setWindowIcon(icon);
        settings_control->setStyleSheet(QString::fromUtf8("/*\n"
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
        save_button = new QPushButton(settings_control);
        save_button->setObjectName(QString::fromUtf8("save_button"));
        save_button->setGeometry(QRect(380, 670, 112, 34));
        reset_button = new QPushButton(settings_control);
        reset_button->setObjectName(QString::fromUtf8("reset_button"));
        reset_button->setGeometry(QRect(530, 670, 112, 34));
        general_options_groupBox = new QGroupBox(settings_control);
        general_options_groupBox->setObjectName(QString::fromUtf8("general_options_groupBox"));
        general_options_groupBox->setGeometry(QRect(20, 140, 1041, 491));
        range_groupBox = new QGroupBox(general_options_groupBox);
        range_groupBox->setObjectName(QString::fromUtf8("range_groupBox"));
        range_groupBox->setGeometry(QRect(10, 170, 351, 121));
        range_groupBox->setAlignment(Qt::AlignCenter);
        x_translation_label = new QLabel(range_groupBox);
        x_translation_label->setObjectName(QString::fromUtf8("x_translation_label"));
        x_translation_label->setGeometry(QRect(20, 24, 71, 16));
        y_translation_label = new QLabel(range_groupBox);
        y_translation_label->setObjectName(QString::fromUtf8("y_translation_label"));
        y_translation_label->setGeometry(QRect(20, 54, 71, 16));
        z_translation_label = new QLabel(range_groupBox);
        z_translation_label->setObjectName(QString::fromUtf8("z_translation_label"));
        z_translation_label->setGeometry(QRect(20, 84, 71, 16));
        x_rotation_label = new QLabel(range_groupBox);
        x_rotation_label->setObjectName(QString::fromUtf8("x_rotation_label"));
        x_rotation_label->setGeometry(QRect(174, 24, 61, 16));
        y_rotation_label = new QLabel(range_groupBox);
        y_rotation_label->setObjectName(QString::fromUtf8("y_rotation_label"));
        y_rotation_label->setGeometry(QRect(174, 54, 61, 16));
        z_rotation_label = new QLabel(range_groupBox);
        z_rotation_label->setObjectName(QString::fromUtf8("z_rotation_label"));
        z_rotation_label->setGeometry(QRect(174, 84, 61, 16));
        x_translation_spinBox = new QSpinBox(range_groupBox);
        x_translation_spinBox->setObjectName(QString::fromUtf8("x_translation_spinBox"));
        x_translation_spinBox->setGeometry(QRect(110, 20, 47, 22));
        x_translation_spinBox->setMaximum(1000000);
        x_translation_spinBox->setValue(40);
        y_translation_spinBox = new QSpinBox(range_groupBox);
        y_translation_spinBox->setObjectName(QString::fromUtf8("y_translation_spinBox"));
        y_translation_spinBox->setGeometry(QRect(110, 50, 47, 22));
        y_translation_spinBox->setMaximum(1000000);
        y_translation_spinBox->setValue(40);
        z_translation_spinBox = new QSpinBox(range_groupBox);
        z_translation_spinBox->setObjectName(QString::fromUtf8("z_translation_spinBox"));
        z_translation_spinBox->setGeometry(QRect(110, 80, 47, 22));
        z_translation_spinBox->setMaximum(1000000);
        z_translation_spinBox->setValue(40);
        x_rotation_spinBox = new QSpinBox(range_groupBox);
        x_rotation_spinBox->setObjectName(QString::fromUtf8("x_rotation_spinBox"));
        x_rotation_spinBox->setGeometry(QRect(270, 20, 47, 22));
        x_rotation_spinBox->setMaximum(180);
        x_rotation_spinBox->setValue(20);
        y_rotation_spinBox = new QSpinBox(range_groupBox);
        y_rotation_spinBox->setObjectName(QString::fromUtf8("y_rotation_spinBox"));
        y_rotation_spinBox->setGeometry(QRect(270, 50, 47, 22));
        y_rotation_spinBox->setMaximum(180);
        y_rotation_spinBox->setValue(20);
        z_rotation_spinBox = new QSpinBox(range_groupBox);
        z_rotation_spinBox->setObjectName(QString::fromUtf8("z_rotation_spinBox"));
        z_rotation_spinBox->setGeometry(QRect(270, 80, 47, 22));
        z_rotation_spinBox->setMaximum(180);
        z_rotation_spinBox->setValue(20);
        stage_budget_label = new QLabel(general_options_groupBox);
        stage_budget_label->setObjectName(QString::fromUtf8("stage_budget_label"));
        stage_budget_label->setGeometry(QRect(80, 100, 141, 16));
        budget_spinBox = new QSpinBox(general_options_groupBox);
        budget_spinBox->setObjectName(QString::fromUtf8("budget_spinBox"));
        budget_spinBox->setGeometry(QRect(230, 100, 61, 22));
        budget_spinBox->setMinimum(1);
        budget_spinBox->setMaximum(999999999);
        budget_spinBox->setValue(6000);
        stage_enabled_checkBox = new QCheckBox(general_options_groupBox);
        stage_enabled_checkBox->setObjectName(QString::fromUtf8("stage_enabled_checkBox"));
        stage_enabled_checkBox->setGeometry(QRect(130, 40, 111, 16));
        stage_enabled_checkBox->setChecked(true);
        stage_specific_groupBox = new QGroupBox(general_options_groupBox);
        stage_specific_groupBox->setObjectName(QString::fromUtf8("stage_specific_groupBox"));
        stage_specific_groupBox->setGeometry(QRect(40, 350, 301, 81));
        branch_count_spinBox = new QSpinBox(stage_specific_groupBox);
        branch_count_spinBox->setObjectName(QString::fromUtf8("branch_count_spinBox"));
        branch_count_spinBox->setGeometry(QRect(240, 40, 42, 22));
        branch_count_spinBox->setMaximum(999999999);
        branch_count_spinBox->setValue(5);
        branch_total_count_label = new QLabel(stage_specific_groupBox);
        branch_total_count_label->setObjectName(QString::fromUtf8("branch_total_count_label"));
        branch_total_count_label->setGeometry(QRect(70, 40, 141, 16));
        cost_function_groupBox = new QGroupBox(general_options_groupBox);
        cost_function_groupBox->setObjectName(QString::fromUtf8("cost_function_groupBox"));
        cost_function_groupBox->setGeometry(QRect(420, 40, 261, 441));
        cost_function_listWidget = new QListWidget(cost_function_groupBox);
        cost_function_listWidget->setObjectName(QString::fromUtf8("cost_function_listWidget"));
        cost_function_listWidget->setGeometry(QRect(20, 40, 221, 371));
        cost_function_parameters_groupBox = new QGroupBox(general_options_groupBox);
        cost_function_parameters_groupBox->setObjectName(QString::fromUtf8("cost_function_parameters_groupBox"));
        cost_function_parameters_groupBox->setGeometry(QRect(710, 40, 271, 271));
        cost_function_parameters_listWidget = new QListWidget(cost_function_parameters_groupBox);
        cost_function_parameters_listWidget->setObjectName(QString::fromUtf8("cost_function_parameters_listWidget"));
        cost_function_parameters_listWidget->setGeometry(QRect(20, 40, 231, 192));
        parameter_value_groupBox = new QGroupBox(general_options_groupBox);
        parameter_value_groupBox->setObjectName(QString::fromUtf8("parameter_value_groupBox"));
        parameter_value_groupBox->setGeometry(QRect(710, 340, 271, 141));
        double_parameter_spinBox = new QDoubleSpinBox(parameter_value_groupBox);
        double_parameter_spinBox->setObjectName(QString::fromUtf8("double_parameter_spinBox"));
        double_parameter_spinBox->setGeometry(QRect(90, 50, 101, 22));
        double_parameter_spinBox->setDecimals(5);
        double_parameter_spinBox->setMinimum(-9999999999999999827367757839185598317239782875580932278577147150336.000000000000000);
        double_parameter_spinBox->setMaximum(999999999999999929757289024535551219930759168.000000000000000);
        bool_parameter_true_radioButton = new QRadioButton(parameter_value_groupBox);
        bool_parameter_true_radioButton->setObjectName(QString::fromUtf8("bool_parameter_true_radioButton"));
        bool_parameter_true_radioButton->setGeometry(QRect(10, 60, 82, 17));
        bool_parameter_false_radioButton = new QRadioButton(parameter_value_groupBox);
        bool_parameter_false_radioButton->setObjectName(QString::fromUtf8("bool_parameter_false_radioButton"));
        bool_parameter_false_radioButton->setGeometry(QRect(180, 60, 82, 17));
        int_parameter_spinBox = new QSpinBox(parameter_value_groupBox);
        int_parameter_spinBox->setObjectName(QString::fromUtf8("int_parameter_spinBox"));
        int_parameter_spinBox->setGeometry(QRect(110, 60, 42, 22));
        int_parameter_spinBox->setMinimum(-999999999);
        int_parameter_spinBox->setMaximum(999999999);
        optimization_search_stage_groupBox = new QGroupBox(settings_control);
        optimization_search_stage_groupBox->setObjectName(QString::fromUtf8("optimization_search_stage_groupBox"));
        optimization_search_stage_groupBox->setGeometry(QRect(250, 10, 571, 111));
        trunk_radioButton = new QRadioButton(optimization_search_stage_groupBox);
        trunk_radioButton->setObjectName(QString::fromUtf8("trunk_radioButton"));
        trunk_radioButton->setEnabled(true);
        trunk_radioButton->setGeometry(QRect(60, 50, 82, 17));
        trunk_radioButton->setChecked(true);
        branch_radioButton = new QRadioButton(optimization_search_stage_groupBox);
        branch_radioButton->setObjectName(QString::fromUtf8("branch_radioButton"));
        branch_radioButton->setGeometry(QRect(250, 50, 82, 17));
        leaf_radioButton = new QRadioButton(optimization_search_stage_groupBox);
        leaf_radioButton->setObjectName(QString::fromUtf8("leaf_radioButton"));
        leaf_radioButton->setGeometry(QRect(440, 50, 82, 17));
        cancel_button = new QPushButton(settings_control);
        cancel_button->setObjectName(QString::fromUtf8("cancel_button"));
        cancel_button->setGeometry(QRect(670, 670, 112, 34));

        retranslateUi(settings_control);

        QMetaObject::connectSlotsByName(settings_control);
    } // setupUi

    void retranslateUi(QDialog *settings_control)
    {
        settings_control->setWindowTitle(QCoreApplication::translate("settings_control", "Optimizer Settings", nullptr));
        save_button->setText(QCoreApplication::translate("settings_control", "Save", nullptr));
        reset_button->setText(QCoreApplication::translate("settings_control", "Reset", nullptr));
        general_options_groupBox->setTitle(QCoreApplication::translate("settings_control", "GENERAL OPTIONS", nullptr));
        range_groupBox->setTitle(QCoreApplication::translate("settings_control", "RANGE (\302\261)", nullptr));
        x_translation_label->setText(QCoreApplication::translate("settings_control", "X Translation:", nullptr));
        y_translation_label->setText(QCoreApplication::translate("settings_control", "Y Translation:", nullptr));
        z_translation_label->setText(QCoreApplication::translate("settings_control", "Z Translation:", nullptr));
        x_rotation_label->setText(QCoreApplication::translate("settings_control", "X Rotation:", nullptr));
        y_rotation_label->setText(QCoreApplication::translate("settings_control", "Y Rotation:", nullptr));
        z_rotation_label->setText(QCoreApplication::translate("settings_control", "Z Rotation:", nullptr));
        stage_budget_label->setText(QCoreApplication::translate("settings_control", "Stage Budget:", nullptr));
        stage_enabled_checkBox->setText(QCoreApplication::translate("settings_control", "Stage Enabled", nullptr));
        stage_specific_groupBox->setTitle(QCoreApplication::translate("settings_control", "STAGE SPECIFIC OPTIONS", nullptr));
        branch_total_count_label->setText(QCoreApplication::translate("settings_control", "Branch Count:", nullptr));
        cost_function_groupBox->setTitle(QCoreApplication::translate("settings_control", "CURRENT COST FUNCTION", nullptr));
        cost_function_parameters_groupBox->setTitle(QCoreApplication::translate("settings_control", "COST FUNCTION PARAMETERS", nullptr));
        parameter_value_groupBox->setTitle(QCoreApplication::translate("settings_control", "PARAMETER VALUE", nullptr));
        bool_parameter_true_radioButton->setText(QCoreApplication::translate("settings_control", "True", nullptr));
        bool_parameter_false_radioButton->setText(QCoreApplication::translate("settings_control", "False", nullptr));
        optimization_search_stage_groupBox->setTitle(QCoreApplication::translate("settings_control", "OPTIMIZATION SEARCH STAGE", nullptr));
        trunk_radioButton->setText(QCoreApplication::translate("settings_control", "Trunk", nullptr));
        branch_radioButton->setText(QCoreApplication::translate("settings_control", "Branch", nullptr));
        leaf_radioButton->setText(QCoreApplication::translate("settings_control", "Leaf", nullptr));
        cancel_button->setText(QCoreApplication::translate("settings_control", "Cancel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class settings_control: public Ui_settings_control {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SETTINGS_CONTROL_H
