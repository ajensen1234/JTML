/********************************************************************************
** Form generated from reading UI file 'drr_tool.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DRR_TOOL_H
#define UI_DRR_TOOL_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSlider>

#include "QVTKOpenGLNativeWidget.h"

QT_BEGIN_NAMESPACE

class Ui_drrTool {
   public:
    QLabel *drr_image_label;
    QGroupBox *maxThreshGroupBox;
    QDoubleSpinBox *minUpperSpinBox;
    QDoubleSpinBox *minLowerSpinBox;
    QLabel *minLowerPrefix;
    QLabel *minUpperPrefix;
    QLabel *minValuePrefix;
    QLabel *minValue;
    QSlider *minSlider;
    QGroupBox *maxThreshGroupBox_2;
    QDoubleSpinBox *maxUpperSpinBox;
    QDoubleSpinBox *maxLowerSpinBox;
    QLabel *maxLowerPrefix;
    QLabel *maxUpperPrefix;
    QLabel *maxValuePrefix;
    QLabel *maxValue;
    QSlider *maxSlider;
    QVTKOpenGLNativeWidget *qvtkWidget;

    void setupUi(QDialog *drrTool) {
        if (drrTool->objectName().isEmpty())
            drrTool->setObjectName(QString::fromUtf8("drrTool"));
        drrTool->resize(1411, 966);
        drrTool->setStyleSheet(QString::fromUtf8(
            "/*\n"
            "	Based on a template by Emanuel Claesson (c) 2013 with "
            "modifications by Paris Flood\n"
            "\n"
            "	Licensed under the Apache License, Version 2.0 (the "
            "\"License\");\n"
            "	you may not use this file except in compliance with the "
            "License.\n"
            "	You may obtain a copy of the License at\n"
            "\n"
            "		http://www.apache.org/licenses/LICENSE-2.0\n"
            "\n"
            "	Unless required by applicable law or agreed to in writing, "
            "software\n"
            "	distributed under the License is distributed on an \"AS IS\" "
            "BASIS,\n"
            "	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express "
            "or implied.\n"
            "	See the License for the specific language governing "
            "permissions and\n"
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
            "QRadioButton::indicator::unchecked, "
            "QCheckBox::indicator::unchecked {\n"
            "	left: 5px;\n"
            "	border: 1px solid #5A5A5A;\n"
            "	background: none;\n"
            "}\n"
            "\n"
            "QRadioButton::indicator:unchecked:hover, "
            "QCheckBox::indicator:unchecked:hover {\n"
            "	left: 5px;\n"
            "	border: 1px solid #D66C23;\n"
            "}\n"
            "\n"
            "QRadioButton::indicator::checked, QCheckBox::indicator::checked "
            "{\n"
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
            "QScrollBar:left-arrow, QScrollBar::right-arrow, "
            "QScrollBar::up-arrow, QScrollBar::down-arrow {\n"
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
            "QAbstractSpinBox::up-button:hover, "
            "QAbstractSpinBox::down-button:hover {\n"
            "	border: 1px solid #D66C23;\n"
            "}\n"
            "QAbstractSpinBox::up-button:pressed, "
            "QAbstractSpinBox::down-button:pressed {\n"
            "	background: #5A5A5A;\n"
            "}\n"
            "\n"
            "QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {\n"
            "	width: 3px;\n"
            "	height: 3px;\n"
            "	border: 1px solid #5A5A5A;\n"
            "}\n"
            "QAbstractSpinBox::up-arrow:hover, "
            "QAbstractSpinBox::down-arrow:hover {\n"
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
        drr_image_label = new QLabel(drrTool);
        drr_image_label->setObjectName(QString::fromUtf8("drr_image_label"));
        drr_image_label->setGeometry(QRect(710, 30, 650, 650));
        maxThreshGroupBox = new QGroupBox(drrTool);
        maxThreshGroupBox->setObjectName(
            QString::fromUtf8("maxThreshGroupBox"));
        maxThreshGroupBox->setGeometry(QRect(160, 700, 931, 111));
        minUpperSpinBox = new QDoubleSpinBox(maxThreshGroupBox);
        minUpperSpinBox->setObjectName(QString::fromUtf8("minUpperSpinBox"));
        minUpperSpinBox->setGeometry(QRect(830, 60, 81, 22));
        minUpperSpinBox->setDecimals(3);
        minUpperSpinBox->setMinimum(-1000000000000000000.000000000000000);
        minUpperSpinBox->setMaximum(1000000000000000000.000000000000000);
        minLowerSpinBox = new QDoubleSpinBox(maxThreshGroupBox);
        minLowerSpinBox->setObjectName(QString::fromUtf8("minLowerSpinBox"));
        minLowerSpinBox->setGeometry(QRect(130, 60, 81, 22));
        minLowerSpinBox->setDecimals(3);
        minLowerSpinBox->setMinimum(-1000000000000000000.000000000000000);
        minLowerSpinBox->setMaximum(10000000000000000000.000000000000000);
        minLowerSpinBox->setValue(-100.000000000000000);
        minLowerPrefix = new QLabel(maxThreshGroupBox);
        minLowerPrefix->setObjectName(QString::fromUtf8("minLowerPrefix"));
        minLowerPrefix->setGeometry(QRect(40, 60, 81, 16));
        minUpperPrefix = new QLabel(maxThreshGroupBox);
        minUpperPrefix->setObjectName(QString::fromUtf8("minUpperPrefix"));
        minUpperPrefix->setGeometry(QRect(750, 60, 81, 16));
        minValuePrefix = new QLabel(maxThreshGroupBox);
        minValuePrefix->setObjectName(QString::fromUtf8("minValuePrefix"));
        minValuePrefix->setGeometry(QRect(370, 40, 131, 16));
        minValue = new QLabel(maxThreshGroupBox);
        minValue->setObjectName(QString::fromUtf8("minValue"));
        minValue->setGeometry(QRect(510, 40, 111, 16));
        minSlider = new QSlider(maxThreshGroupBox);
        minSlider->setObjectName(QString::fromUtf8("minSlider"));
        minSlider->setGeometry(QRect(260, 60, 461, 22));
        minSlider->setMaximum(1000);
        minSlider->setValue(1000);
        minSlider->setOrientation(Qt::Horizontal);
        maxThreshGroupBox_2 = new QGroupBox(drrTool);
        maxThreshGroupBox_2->setObjectName(
            QString::fromUtf8("maxThreshGroupBox_2"));
        maxThreshGroupBox_2->setGeometry(QRect(160, 830, 931, 111));
        maxUpperSpinBox = new QDoubleSpinBox(maxThreshGroupBox_2);
        maxUpperSpinBox->setObjectName(QString::fromUtf8("maxUpperSpinBox"));
        maxUpperSpinBox->setGeometry(QRect(850, 50, 81, 22));
        maxUpperSpinBox->setDecimals(3);
        maxUpperSpinBox->setMinimum(-1000000000000000000.000000000000000);
        maxUpperSpinBox->setMaximum(1000000000000000000.000000000000000);
        maxUpperSpinBox->setValue(100.000000000000000);
        maxLowerSpinBox = new QDoubleSpinBox(maxThreshGroupBox_2);
        maxLowerSpinBox->setObjectName(QString::fromUtf8("maxLowerSpinBox"));
        maxLowerSpinBox->setGeometry(QRect(140, 50, 81, 22));
        maxLowerSpinBox->setDecimals(3);
        maxLowerSpinBox->setMinimum(-1000000000000000000.000000000000000);
        maxLowerSpinBox->setMaximum(1000000000000000000.000000000000000);
        maxLowerPrefix = new QLabel(maxThreshGroupBox_2);
        maxLowerPrefix->setObjectName(QString::fromUtf8("maxLowerPrefix"));
        maxLowerPrefix->setGeometry(QRect(30, 50, 81, 16));
        maxUpperPrefix = new QLabel(maxThreshGroupBox_2);
        maxUpperPrefix->setObjectName(QString::fromUtf8("maxUpperPrefix"));
        maxUpperPrefix->setGeometry(QRect(770, 50, 81, 16));
        maxValuePrefix = new QLabel(maxThreshGroupBox_2);
        maxValuePrefix->setObjectName(QString::fromUtf8("maxValuePrefix"));
        maxValuePrefix->setGeometry(QRect(340, 30, 131, 16));
        maxValue = new QLabel(maxThreshGroupBox_2);
        maxValue->setObjectName(QString::fromUtf8("maxValue"));
        maxValue->setGeometry(QRect(480, 30, 101, 16));
        maxSlider = new QSlider(maxThreshGroupBox_2);
        maxSlider->setObjectName(QString::fromUtf8("maxSlider"));
        maxSlider->setGeometry(QRect(260, 50, 461, 22));
        maxSlider->setMaximum(1000);
        maxSlider->setValue(500);
        maxSlider->setOrientation(Qt::Horizontal);
        qvtkWidget = new QVTKOpenGLNativeWidget(drrTool);
        qvtkWidget->setObjectName(QString::fromUtf8("qvtkWidget"));
        qvtkWidget->setGeometry(QRect(10, 0, 681, 681));

        retranslateUi(drrTool);

        QMetaObject::connectSlotsByName(drrTool);
    }  // setupUi

    void retranslateUi(QDialog *drrTool) {
        drrTool->setWindowTitle(
            QCoreApplication::translate("drrTool", "Dialog", nullptr));
        drr_image_label->setText(QString());
        maxThreshGroupBox->setTitle(QCoreApplication::translate(
            "drrTool", "MINIMUM THRESHOLD", nullptr));
        minLowerPrefix->setText(
            QCoreApplication::translate("drrTool", "Lower Bound:", nullptr));
        minUpperPrefix->setText(
            QCoreApplication::translate("drrTool", "Upper Bound:", nullptr));
        minValuePrefix->setText(QCoreApplication::translate(
            "drrTool", "Minimum Threshold Value:", nullptr));
        minValue->setText(
            QCoreApplication::translate("drrTool", "VALUE", nullptr));
        maxThreshGroupBox_2->setTitle(QCoreApplication::translate(
            "drrTool", "MAXIMUM THRESHOLD", nullptr));
        maxLowerPrefix->setText(
            QCoreApplication::translate("drrTool", "Lower Bound:", nullptr));
        maxUpperPrefix->setText(
            QCoreApplication::translate("drrTool", "Upper Bound:", nullptr));
        maxValuePrefix->setText(QCoreApplication::translate(
            "drrTool", "Maximum Threshold Value:", nullptr));
        maxValue->setText(
            QCoreApplication::translate("drrTool", "VALUE", nullptr));
    }  // retranslateUi
};

namespace Ui {
class drrTool : public Ui_drrTool {};
}  // namespace Ui

QT_END_NAMESPACE

#endif  // UI_DRR_TOOL_H
