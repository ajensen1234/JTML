/********************************************************************************
** Form generated from reading UI file 'sym_trap.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SYM_TRAP_H
#define UI_SYM_TRAP_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_symTrap
{
public:
    QPushButton *Plot_3D;
    QPushButton *save_data;
    QPushButton *load_data;
    QPushButton *optimize;
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout;
    QSpinBox *iterBox;
    QProgressBar *progressBar;
    QPushButton *save_plot;
    QPushButton *Plot_2D;

    void setupUi(QDialog *symTrap)
    {
        if (symTrap->objectName().isEmpty())
            symTrap->setObjectName(QString::fromUtf8("symTrap"));
        symTrap->resize(1172, 789);
        symTrap->setStyleSheet(QString::fromUtf8("/*\n"
"	Based on a template by Emanuel Claesson (c) 2013 with modifications by Paris Flood\n"
"\n"
"	Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);\n"
"	you may not use this file except in compliance with the License.\n"
"	You may obtain a copy of the License at\n"
"\n"
"		http://www.apache.org/licenses/LICENSE-2.0\n"
"\n"
"	Unless required by applicable law or agreed to in writing, software\n"
"	distributed under the License is distributed on an &quot;AS IS&quot; BASIS,\n"
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
"	background: #191"
                        "919;\n"
"	border-style: none;\n"
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
""
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
"	min-height: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-line, QScrollBar::sub-line {\n"
"	background:#353535;\n"
"	border: 1px solid #5"
                        "A5A5A;\n"
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
"	background: #3F3F3F;\n"
"}\n"
"\n"
"QAbstractButton:hover {\n"
"	background: #353535;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"	ba"
                        "ckground: #353535;\n"
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
"	border: 1px solid #5"
                        "A5A5A;\n"
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
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"	height: 5px;\n"
"	margin: 4px 0px 4px 0px;\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
""
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
"QListWidget::item:selected {\n"
"    background-color: #2158AA;\n"
"}\n"
"\n"
" QMenu::item:selected{\n"
"	background: #353535;\n"
"	border: 2px solid #D66C23;\n"
"    border-style: none solid none none"
                        ";\n"
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
        Plot_3D = new QPushButton(symTrap);
        Plot_3D->setObjectName(QString::fromUtf8("Plot_3D"));
        Plot_3D->setGeometry(QRect(360, 20, 151, 61));
        Plot_3D->setMaximumSize(QSize(679, 16777215));
        QFont font;
        font.setPointSize(16);
        Plot_3D->setFont(font);
        save_data = new QPushButton(symTrap);
        save_data->setObjectName(QString::fromUtf8("save_data"));
        save_data->setGeometry(QRect(870, 20, 111, 61));
        save_data->setMaximumSize(QSize(679, 16777215));
        save_data->setFont(font);
        load_data = new QPushButton(symTrap);
        load_data->setObjectName(QString::fromUtf8("load_data"));
        load_data->setGeometry(QRect(1000, 20, 131, 61));
        load_data->setMaximumSize(QSize(679, 16777215));
        load_data->setFont(font);
        optimize = new QPushButton(symTrap);
        optimize->setObjectName(QString::fromUtf8("optimize"));
        optimize->setGeometry(QRect(20, 20, 231, 61));
        optimize->setMaximumSize(QSize(679, 16777215));
        optimize->setFont(font);
        optimize->setAutoFillBackground(false);
        optimize->setFlat(false);
        verticalLayoutWidget = new QWidget(symTrap);
        verticalLayoutWidget->setObjectName(QString::fromUtf8("verticalLayoutWidget"));
        verticalLayoutWidget->setGeometry(QRect(20, 140, 1131, 631));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setSizeConstraint(QLayout::SetDefaultConstraint);
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        iterBox = new QSpinBox(symTrap);
        iterBox->setObjectName(QString::fromUtf8("iterBox"));
        iterBox->setGeometry(QRect(270, 20, 71, 61));
        QFont font1;
        font1.setPointSize(14);
        iterBox->setFont(font1);
        iterBox->setAutoFillBackground(false);
        iterBox->setMinimum(5);
        iterBox->setMaximum(999);
        iterBox->setSingleStep(1);
        iterBox->setValue(20);
        progressBar = new QProgressBar(symTrap);
        progressBar->setObjectName(QString::fromUtf8("progressBar"));
        progressBar->setGeometry(QRect(20, 90, 1131, 31));
        progressBar->setValue(0);
        save_plot = new QPushButton(symTrap);
        save_plot->setObjectName(QString::fromUtf8("save_plot"));
        save_plot->setGeometry(QRect(700, 20, 101, 61));
        save_plot->setMaximumSize(QSize(679, 16777215));
        save_plot->setFont(font);
        Plot_2D = new QPushButton(symTrap);
        Plot_2D->setObjectName(QString::fromUtf8("Plot_2D"));
        Plot_2D->setGeometry(QRect(530, 20, 151, 61));
        Plot_2D->setMaximumSize(QSize(679, 16777215));
        Plot_2D->setFont(font);

        retranslateUi(symTrap);

        QMetaObject::connectSlotsByName(symTrap);
    } // setupUi

    void retranslateUi(QDialog *symTrap)
    {
        symTrap->setWindowTitle(QCoreApplication::translate("symTrap", "Symmetry Trap Tool", nullptr));
        Plot_3D->setText(QCoreApplication::translate("symTrap", "Plot Data 3D", nullptr));
        save_data->setText(QCoreApplication::translate("symTrap", "Save Data", nullptr));
        load_data->setText(QCoreApplication::translate("symTrap", "Load Data", nullptr));
        optimize->setText(QCoreApplication::translate("symTrap", "Analyze", nullptr));
#if QT_CONFIG(tooltip)
        iterBox->setToolTip(QCoreApplication::translate("symTrap", "How many intermediate iterations should the rotation have?", nullptr));
#endif // QT_CONFIG(tooltip)
        progressBar->setFormat(QCoreApplication::translate("symTrap", "%p% Analyzed", nullptr));
        save_plot->setText(QCoreApplication::translate("symTrap", "Save Plot", nullptr));
        Plot_2D->setText(QCoreApplication::translate("symTrap", "Plot Data 2D", nullptr));
    } // retranslateUi

};

namespace Ui {
    class symTrap: public Ui_symTrap {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SYM_TRAP_H
