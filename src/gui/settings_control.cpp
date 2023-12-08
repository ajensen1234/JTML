// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*Settings Control Window*/
#include "gui/settings_control.h"

/*Message Box*/
#include <qmessagebox.h>

/*Settings Constant*/
#include "core/settings_constants.h"

/*Spacing Constants*/
#include "core/settings_window_size_constants.h"

SettingsControl::SettingsControl(QWidget* parent, Qt::WindowFlags flags)
    : QDialog(parent, flags) {
    ui.setupUi(this);

    connect(this, SIGNAL(Done()), this, SLOT(close()));

    /*Set Layout of All the Differnt Buttons and Boxes etc*/
    /*Resize Buttons based on Font Size In Order to be Compatible with High DPI
     * Monitors*/
    QFontMetrics font_metrics(this->font());

    /*Adjust for Title Height*/
    this->setStyleSheet(this->styleSheet() +=
                        "QGroupBox { margin-top: " +
                        QString::number(font_metrics.height() / 2) + "px; }");
    int group_box_to_top_button_y = font_metrics.height() / 2;

    /*Make Sure Trunk is checked and by default so is stage enabled*/
    ui.trunk_radioButton->setChecked(true);
    ui.stage_enabled_checkBox->setChecked(true);

    /*Load Cost Function Names in Constructor as thie cannot change during a
     * session*/
    std::vector<jta_cost_function::CostFunction> available_cost_functions =
        jta_cost_function::CostFunctionManager().getAvailableCostFunctions();
    for (int i = 0; i < available_cost_functions.size(); i++) {
        ui.cost_function_listWidget->addItem(QString::fromStdString(
            available_cost_functions[i].getCostFunctionName()));
    }

    /*Set Up Stage Select Group Box Size */
    /*Get Size of Radio Button Widths*/
    int stage_radio_button_widths =
        INSIDE_RADIO_BUTTON_PADDING_X +
        std::max(std::max(font_metrics.horizontalAdvance("Trunk"),
                          font_metrics.horizontalAdvance("Branch")),
                 font_metrics.horizontalAdvance("Leaf"));
    /*Check if Horizontal Spacing is Big Enough for Title*/
    int safety_padding_x = 0;
    if (3 * stage_radio_button_widths + 2 * GROUP_BOX_TO_RADIO_BUTTON_X +
            2 * BUTTON_TO_BUTTON_PADDING_X >
        1.25 * font_metrics.horizontalAdvance(
                   ui.optimization_search_stage_groupBox->title())) {
        ui.trunk_radioButton->setGeometry(QRect(
            GROUP_BOX_TO_RADIO_BUTTON_X,
            GROUP_BOX_TO_RADIO_BUTTON_PADDING_Y + (font_metrics.height() / 2),
            stage_radio_button_widths,
            font_metrics.height() + INSIDE_RADIO_BUTTON_PADDING_Y));
        ui.branch_radioButton->setGeometry(QRect(
            ui.trunk_radioButton->geometry().right() +
                BUTTON_TO_BUTTON_PADDING_X,
            GROUP_BOX_TO_RADIO_BUTTON_PADDING_Y + (font_metrics.height() / 2),
            stage_radio_button_widths,
            font_metrics.height() + INSIDE_RADIO_BUTTON_PADDING_Y));
        ui.leaf_radioButton->setGeometry(QRect(
            ui.branch_radioButton->geometry().right() +
                BUTTON_TO_BUTTON_PADDING_X,
            GROUP_BOX_TO_RADIO_BUTTON_PADDING_Y + (font_metrics.height() / 2),
            stage_radio_button_widths,
            font_metrics.height() + INSIDE_RADIO_BUTTON_PADDING_Y));
    } else {
        safety_padding_x =
            (1.25 * font_metrics.horizontalAdvance(
                        ui.optimization_search_stage_groupBox->title()) -
             (3 * stage_radio_button_widths + 2 * GROUP_BOX_TO_RADIO_BUTTON_X +
              2 * BUTTON_TO_BUTTON_PADDING_X)) /
            2;
        ui.trunk_radioButton->setGeometry(QRect(
            safety_padding_x + GROUP_BOX_TO_RADIO_BUTTON_X,
            GROUP_BOX_TO_RADIO_BUTTON_PADDING_Y + (font_metrics.height() / 2),
            stage_radio_button_widths,
            font_metrics.height() + INSIDE_RADIO_BUTTON_PADDING_Y));
        ui.branch_radioButton->setGeometry(QRect(
            ui.trunk_radioButton->geometry().right() +
                BUTTON_TO_BUTTON_PADDING_X,
            GROUP_BOX_TO_RADIO_BUTTON_PADDING_Y + (font_metrics.height() / 2),
            stage_radio_button_widths,
            font_metrics.height() + INSIDE_RADIO_BUTTON_PADDING_Y));
        ui.leaf_radioButton->setGeometry(QRect(
            ui.branch_radioButton->geometry().right() +
                BUTTON_TO_BUTTON_PADDING_X,
            GROUP_BOX_TO_RADIO_BUTTON_PADDING_Y + (font_metrics.height() / 2),
            stage_radio_button_widths,
            font_metrics.height() + INSIDE_RADIO_BUTTON_PADDING_Y));
    }
    /*Set Dimensions of Box*/
    ui.optimization_search_stage_groupBox->setGeometry(
        QRect(0, 0,
              ui.leaf_radioButton->geometry().right() + safety_padding_x +
                  GROUP_BOX_TO_RADIO_BUTTON_X,
              ui.leaf_radioButton->geometry().bottom() +
                  +(font_metrics.height() / 2) +
                  GROUP_BOX_TO_RADIO_BUTTON_PADDING_Y));

    /*Set Up Dimensions for Range Group Box*/
    ui.range_groupBox->setGeometry(QRect(
        GROUP_BOX_TO_SMALL_GROUP_BOX_X,
        group_box_to_top_button_y + font_metrics.height() +
            2 * SMALL_GROUP_BOX_PADDING_Y + 7,
        font_metrics.horizontalAdvance(ui.x_translation_label->text()) +
            font_metrics.horizontalAdvance(ui.x_rotation_label->text()) +
            LABEL_TO_SPIN_BOX_PADDING_X * 2 + SPIN_BOX_TO_LABEL_PADDING_X +
            SMALL_GROUP_BOX_PADDING_X * 2 + INSIDE_SPIN_BOX_PADDING_X * 2 +
            2 * font_metrics.horizontalAdvance("XXX"),
        group_box_to_top_button_y + SMALL_GROUP_BOX_PADDING_Y * 2 +
            font_metrics.height() * 3 + 3 * INSIDE_SPIN_BOX_PADDING_Y +
            2 * SPIN_BOX_TO_SPIN_BOX_PADDING_Y));
    ui.x_translation_label->setGeometry(
        QRect(SMALL_GROUP_BOX_PADDING_X,
              group_box_to_top_button_y + SMALL_GROUP_BOX_PADDING_Y,
              font_metrics.horizontalAdvance(ui.x_translation_label->text()),
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.y_translation_label->setGeometry(
        QRect(SMALL_GROUP_BOX_PADDING_X,
              group_box_to_top_button_y + SMALL_GROUP_BOX_PADDING_Y +
                  font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y +
                  SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
              font_metrics.horizontalAdvance(ui.y_translation_label->text()),
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.z_translation_label->setGeometry(
        QRect(SMALL_GROUP_BOX_PADDING_X,
              group_box_to_top_button_y + SMALL_GROUP_BOX_PADDING_Y +
                  2 * (font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y +
                       SPIN_BOX_TO_SPIN_BOX_PADDING_Y),
              font_metrics.horizontalAdvance(ui.z_translation_label->text()),
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.x_rotation_label->setGeometry(
        QRect(font_metrics.horizontalAdvance(ui.x_translation_label->text()) +
                  SMALL_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X +
                  font_metrics.horizontalAdvance("XXX") +
                  LABEL_TO_SPIN_BOX_PADDING_X + SMALL_GROUP_BOX_PADDING_X,
              group_box_to_top_button_y + SMALL_GROUP_BOX_PADDING_Y,
              font_metrics.horizontalAdvance(ui.x_rotation_label->text()),
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.y_rotation_label->setGeometry(
        QRect(font_metrics.horizontalAdvance(ui.x_translation_label->text()) +
                  SMALL_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X +
                  font_metrics.horizontalAdvance("XXX") +
                  LABEL_TO_SPIN_BOX_PADDING_X + SMALL_GROUP_BOX_PADDING_X,
              group_box_to_top_button_y + SMALL_GROUP_BOX_PADDING_Y +
                  font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y +
                  SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
              font_metrics.horizontalAdvance(ui.y_rotation_label->text()),
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.z_rotation_label->setGeometry(
        QRect(font_metrics.horizontalAdvance(ui.x_translation_label->text()) +
                  SMALL_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X +
                  font_metrics.horizontalAdvance("XXX") +
                  LABEL_TO_SPIN_BOX_PADDING_X + SMALL_GROUP_BOX_PADDING_X,
              group_box_to_top_button_y + SMALL_GROUP_BOX_PADDING_Y +
                  2 * (font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y +
                       SPIN_BOX_TO_SPIN_BOX_PADDING_Y),
              font_metrics.horizontalAdvance(ui.z_rotation_label->text()),
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.x_translation_spinBox->setGeometry(
        QRect(font_metrics.horizontalAdvance(ui.x_translation_label->text()) +
                  LABEL_TO_SPIN_BOX_PADDING_X + SMALL_GROUP_BOX_PADDING_X,
              group_box_to_top_button_y + SMALL_GROUP_BOX_PADDING_Y,
              font_metrics.horizontalAdvance("XXX") + INSIDE_SPIN_BOX_PADDING_X,
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.y_translation_spinBox->setGeometry(
        QRect(font_metrics.horizontalAdvance(ui.x_translation_label->text()) +
                  LABEL_TO_SPIN_BOX_PADDING_X + SMALL_GROUP_BOX_PADDING_X,
              group_box_to_top_button_y + SMALL_GROUP_BOX_PADDING_Y +
                  font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y +
                  SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
              font_metrics.horizontalAdvance("XXX") + INSIDE_SPIN_BOX_PADDING_X,
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.z_translation_spinBox->setGeometry(
        QRect(font_metrics.horizontalAdvance(ui.x_translation_label->text()) +
                  LABEL_TO_SPIN_BOX_PADDING_X + SMALL_GROUP_BOX_PADDING_X,
              group_box_to_top_button_y + SMALL_GROUP_BOX_PADDING_Y +
                  2 * (font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y +
                       SPIN_BOX_TO_SPIN_BOX_PADDING_Y),
              font_metrics.horizontalAdvance("XXX") + INSIDE_SPIN_BOX_PADDING_X,
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.x_rotation_spinBox->setGeometry(
        QRect(font_metrics.horizontalAdvance(ui.x_translation_label->text()) +
                  LABEL_TO_SPIN_BOX_PADDING_X +
                  font_metrics.horizontalAdvance(ui.x_rotation_label->text()) +
                  SMALL_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X +
                  font_metrics.horizontalAdvance("XXX") +
                  SPIN_BOX_TO_LABEL_PADDING_X + LABEL_TO_SPIN_BOX_PADDING_X,
              group_box_to_top_button_y + SMALL_GROUP_BOX_PADDING_Y,
              font_metrics.horizontalAdvance("XXX") + INSIDE_SPIN_BOX_PADDING_X,
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.y_rotation_spinBox->setGeometry(
        QRect(font_metrics.horizontalAdvance(ui.x_translation_label->text()) +
                  LABEL_TO_SPIN_BOX_PADDING_X +
                  font_metrics.horizontalAdvance(ui.x_rotation_label->text()) +
                  SMALL_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X +
                  font_metrics.horizontalAdvance("XXX") +
                  SPIN_BOX_TO_LABEL_PADDING_X + LABEL_TO_SPIN_BOX_PADDING_X,
              group_box_to_top_button_y + SMALL_GROUP_BOX_PADDING_Y +
                  font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y +
                  SPIN_BOX_TO_SPIN_BOX_PADDING_Y,
              font_metrics.horizontalAdvance("XXX") + INSIDE_SPIN_BOX_PADDING_X,
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.z_rotation_spinBox->setGeometry(
        QRect(font_metrics.horizontalAdvance(ui.x_translation_label->text()) +
                  LABEL_TO_SPIN_BOX_PADDING_X +
                  font_metrics.horizontalAdvance(ui.x_rotation_label->text()) +
                  SMALL_GROUP_BOX_PADDING_X + INSIDE_SPIN_BOX_PADDING_X +
                  font_metrics.horizontalAdvance("XXX") +
                  SPIN_BOX_TO_LABEL_PADDING_X + LABEL_TO_SPIN_BOX_PADDING_X,
              group_box_to_top_button_y + SMALL_GROUP_BOX_PADDING_Y +
                  2 * (font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y +
                       SPIN_BOX_TO_SPIN_BOX_PADDING_Y),
              font_metrics.horizontalAdvance("XXX") + INSIDE_SPIN_BOX_PADDING_X,
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));

    /*Set the Y Positioning and Size of Everything in the General Options Left
     * Column*/
    ui.stage_enabled_checkBox->setGeometry(QRect(
        0, GROUP_BOX_TO_RADIO_BUTTON_PADDING_Y + (font_metrics.height() / 2),
        INSIDE_RADIO_BUTTON_PADDING_X +
            font_metrics.horizontalAdvance(ui.stage_enabled_checkBox->text()),
        font_metrics.height() + INSIDE_RADIO_BUTTON_PADDING_Y));
    ui.stage_budget_label->setGeometry(QRect(
        0, ui.stage_enabled_checkBox->geometry().bottom() + CHECKBOX_TO_LABEL_Y,
        font_metrics.horizontalAdvance(ui.stage_budget_label->text()),
        font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.budget_spinBox->setGeometry(QRect(
        font_metrics.horizontalAdvance(ui.stage_budget_label->text()) +
            LABEL_TO_SPIN_BOX_PADDING_X + SMALL_GROUP_BOX_PADDING_X,
        ui.stage_budget_label->geometry().top(),
        font_metrics.horizontalAdvance("XXXXXXX") + INSIDE_SPIN_BOX_PADDING_X,
        font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.range_groupBox->setGeometry(
        QRect(GROUP_BOX_TO_SMALL_GROUP_BOX_X,
              ui.stage_budget_label->geometry().bottom() +
                  GROUP_BOX_TO_LABEL_PADDING_Y,
              ui.range_groupBox->geometry().width(),
              ui.range_groupBox->geometry().height()));
    ui.branch_total_count_label->setGeometry(QRect(
        (ui.range_groupBox->geometry().width() -
         (font_metrics.horizontalAdvance(ui.branch_total_count_label->text()) +
          LABEL_TO_SPIN_BOX_PADDING_X + font_metrics.horizontalAdvance("XXX") +
          INSIDE_SPIN_BOX_PADDING_X)) /
            2,
        group_box_to_top_button_y + SMALL_GROUP_BOX_PADDING_Y,
        font_metrics.horizontalAdvance(ui.branch_total_count_label->text()),
        font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.branch_count_spinBox->setGeometry(
        QRect(ui.branch_total_count_label->geometry().right() +
                  LABEL_TO_SPIN_BOX_PADDING_X,
              ui.branch_total_count_label->geometry().top(),
              font_metrics.horizontalAdvance("XXX") + INSIDE_SPIN_BOX_PADDING_X,
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.stage_specific_groupBox->setGeometry(
        QRect(GROUP_BOX_TO_SMALL_GROUP_BOX_X,
              ui.range_groupBox->geometry().bottom() + GROUP_BOX_TO_GROUP_BOX_Y,
              ui.range_groupBox->geometry().width(),
              ui.branch_total_count_label->geometry().bottom() +
                  (ui.range_groupBox->geometry().height() -
                   ui.z_translation_label->geometry().bottom())));

    /*Shift the rest of the Left Column to the correct horizontal position*/
    ui.stage_enabled_checkBox->move(
        QPoint(ui.range_groupBox->geometry().left() +
                   ((ui.range_groupBox->geometry().width() -
                     ui.stage_enabled_checkBox->geometry().width()) /
                    2),
               ui.stage_enabled_checkBox->geometry().top()));
    ui.stage_budget_label->move(
        QPoint(ui.range_groupBox->geometry().left() +
                   ((ui.range_groupBox->geometry().width() -
                     (ui.stage_budget_label->geometry().width() +
                      LABEL_TO_SPIN_BOX_PADDING_X +
                      ui.budget_spinBox->geometry().width())) /
                    2),
               ui.stage_budget_label->geometry().top()));
    ui.budget_spinBox->move(QPoint(
        LABEL_TO_SPIN_BOX_PADDING_X + ui.stage_budget_label->geometry().right(),
        ui.budget_spinBox->geometry().top()));

    /*Central Column*/
    ui.cost_function_groupBox->setGeometry(QRect(
        ui.range_groupBox->geometry().right() + GROUP_BOX_TO_GROUP_BOX_X,
        ui.stage_enabled_checkBox->geometry().top() + group_box_to_top_button_y,
        ui.range_groupBox->geometry().width(),
        ui.stage_specific_groupBox->geometry().bottom() -
            (ui.stage_enabled_checkBox->geometry().top() +
             group_box_to_top_button_y)));
    ui.cost_function_listWidget->setGeometry(QRect(
        GROUP_BOX_TO_SMALL_GROUP_BOX_X,
        group_box_to_top_button_y + GROUP_BOX_TO_SMALL_GROUP_BOX_Y,
        ui.cost_function_groupBox->geometry().width() -
            2 * GROUP_BOX_TO_SMALL_GROUP_BOX_X,
        ui.cost_function_groupBox->geometry().height() -
            (group_box_to_top_button_y + 2 * GROUP_BOX_TO_SMALL_GROUP_BOX_Y)));

    /*Right Column*/
    ui.double_parameter_spinBox->setGeometry(
        QRect((ui.cost_function_groupBox->geometry().width() -
               (font_metrics.horizontalAdvance("XXXXXXXXXX") +
                INSIDE_SPIN_BOX_PADDING_X)) /
                  2,
              group_box_to_top_button_y + GROUP_BOX_TO_LABEL_PADDING_Y,
              font_metrics.horizontalAdvance("XXXXXXXXXX") +
                  INSIDE_SPIN_BOX_PADDING_X,
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    ui.int_parameter_spinBox->setGeometry(
        QRect((ui.cost_function_groupBox->geometry().width() -
               (font_metrics.horizontalAdvance("XXXXXXXXXX") +
                INSIDE_SPIN_BOX_PADDING_X)) /
                  2,
              group_box_to_top_button_y + GROUP_BOX_TO_LABEL_PADDING_Y,
              font_metrics.horizontalAdvance("XXXXXXXXXX") +
                  INSIDE_SPIN_BOX_PADDING_X,
              font_metrics.height() + INSIDE_SPIN_BOX_PADDING_Y));
    /*T/F Radio Button width*/
    int true_false_radiobutton_width =
        INSIDE_RADIO_BUTTON_PADDING_X +
        std::max(font_metrics.horizontalAdvance(
                     ui.bool_parameter_true_radioButton->text()),
                 font_metrics.horizontalAdvance(
                     ui.bool_parameter_false_radioButton->text()));
    ui.bool_parameter_true_radioButton->setGeometry(QRect(
        (ui.cost_function_groupBox->geometry().width() -
         (2 * true_false_radiobutton_width + BUTTON_TO_BUTTON_PADDING_X)) /
            2,
        2 + ui.int_parameter_spinBox->geometry().top(),
        true_false_radiobutton_width,
        font_metrics.height() + INSIDE_RADIO_BUTTON_PADDING_Y));
    ui.bool_parameter_false_radioButton->setGeometry(
        QRect(ui.bool_parameter_true_radioButton->geometry().right() +
                  BUTTON_TO_BUTTON_PADDING_X,
              2 + ui.int_parameter_spinBox->geometry().top(),
              true_false_radiobutton_width,
              font_metrics.height() + INSIDE_RADIO_BUTTON_PADDING_Y));
    ui.parameter_value_groupBox->setGeometry(
        QRect(ui.cost_function_groupBox->geometry().right() +
                  GROUP_BOX_TO_GROUP_BOX_X,
              ui.stage_specific_groupBox->geometry().top(),
              ui.cost_function_groupBox->geometry().width(),
              ui.stage_specific_groupBox->geometry().height()));
    ui.cost_function_parameters_groupBox->setGeometry(
        QRect(ui.cost_function_groupBox->geometry().right() +
                  GROUP_BOX_TO_GROUP_BOX_X,
              ui.cost_function_groupBox->geometry().top(),
              ui.cost_function_groupBox->geometry().width(),
              ui.range_groupBox->geometry().bottom() -
                  ui.cost_function_groupBox->geometry().top() + 1));
    ui.cost_function_parameters_listWidget->setGeometry(QRect(
        GROUP_BOX_TO_SMALL_GROUP_BOX_X,
        group_box_to_top_button_y + GROUP_BOX_TO_SMALL_GROUP_BOX_Y,
        ui.cost_function_parameters_groupBox->geometry().width() -
            2 * GROUP_BOX_TO_SMALL_GROUP_BOX_X,
        ui.cost_function_parameters_groupBox->geometry().height() -
            (group_box_to_top_button_y + 2 * GROUP_BOX_TO_SMALL_GROUP_BOX_Y)));

    /*Set Geometry of General Options*/
    ui.general_options_groupBox->setGeometry(
        QRect(APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X, 100,
              ui.parameter_value_groupBox->geometry().right() +
                  GROUP_BOX_TO_SMALL_GROUP_BOX_X,
              ui.parameter_value_groupBox->geometry().bottom() +
                  GROUP_BOX_TO_GROUP_BOX_Y));

    /*Move Search Stage Box in Both Direction*/
    ui.optimization_search_stage_groupBox->move(QPoint(
        ((ui.general_options_groupBox->geometry().width() -
          ui.optimization_search_stage_groupBox->geometry().width()) /
         2) +
            ui.general_options_groupBox->geometry().left(),
        APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y + group_box_to_top_button_y));

    /*Move General Options Vertically*/
    ui.general_options_groupBox->move(
        QPoint(ui.general_options_groupBox->geometry().left(),
               ui.optimization_search_stage_groupBox->geometry().bottom() +
                   GROUP_BOX_TO_GROUP_BOX_Y));

    /*Set Geometry of Bottom Three Buttons*/
    int opt_settings_button_width =
        INSIDE_BUTTON_PADDING_X +
        std::max(
            std::max(font_metrics.horizontalAdvance(ui.save_button->text()),
                     font_metrics.horizontalAdvance(ui.reset_button->text())),
            font_metrics.horizontalAdvance(ui.cancel_button->text()));
    ui.save_button->setGeometry(QRect(
        1 + ((ui.general_options_groupBox->geometry().right() +
              APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X) -
             (3 * opt_settings_button_width + 2 * BUTTON_TO_BUTTON_PADDING_X)) /
                2,
        ui.general_options_groupBox->geometry().bottom() +
            GROUP_BOX_TO_GROUP_BOX_Y,
        opt_settings_button_width,
        font_metrics.height() + INSIDE_BUTTON_PADDING_Y));
    ui.reset_button->setGeometry(
        QRect(ui.save_button->geometry().right() + BUTTON_TO_BUTTON_PADDING_X,
              ui.general_options_groupBox->geometry().bottom() +
                  GROUP_BOX_TO_GROUP_BOX_Y,
              opt_settings_button_width,
              font_metrics.height() + INSIDE_BUTTON_PADDING_Y));
    ui.cancel_button->setGeometry(
        QRect(ui.reset_button->geometry().right() + BUTTON_TO_BUTTON_PADDING_X,
              ui.general_options_groupBox->geometry().bottom() +
                  GROUP_BOX_TO_GROUP_BOX_Y,
              opt_settings_button_width,
              font_metrics.height() + INSIDE_BUTTON_PADDING_Y));

    /*Set Width and Height of Window*/
    setFixedSize(
        ui.general_options_groupBox->geometry().right() +
            APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X,
        ui.reset_button->geometry().bottom() + GROUP_BOX_TO_GROUP_BOX_Y);
}

SettingsControl::~SettingsControl() {}

/*Load Optimizer Settings from Main Window*/
void SettingsControl::LoadSettings(
    jta_cost_function::CostFunctionManager sc_trunk_manager,
    jta_cost_function::CostFunctionManager sc_branch_manager,
    jta_cost_function::CostFunctionManager sc_leaf_manager,
    OptimizerSettings opt_settings) {
    /*Load CFMs locally*/
    sc_trunk_manager_ = sc_trunk_manager;
    sc_branch_manager_ = sc_branch_manager;
    sc_leaf_manager_ = sc_leaf_manager;

    /*Load Optimizer Settings locally*/
    opt_settings_ = opt_settings;

    /*Get Rid of Parameters Index and Hide the Value View*/
    ui.cost_function_parameters_listWidget->clearSelection();
    ui.double_parameter_spinBox->setEnabled(false);
    ui.double_parameter_spinBox->setVisible(false);
    ui.int_parameter_spinBox->setEnabled(false);
    ui.int_parameter_spinBox->setVisible(false);
    ui.bool_parameter_true_radioButton->setEnabled(false);
    ui.bool_parameter_true_radioButton->setVisible(false);
    ui.bool_parameter_false_radioButton->setEnabled(false);
    ui.bool_parameter_false_radioButton->setVisible(false);

    /*Populate Available List Widget and Select the Current One*/
    std::vector<jta_cost_function::CostFunction> available_cost_functions =
        sc_trunk_manager_.getAvailableCostFunctions();
    ui.trunk_radioButton->setChecked(true);  // Always load to trunk
    if (ui.trunk_radioButton->isChecked()) {
        /*Load Optimizer Settings (non-cost function)*/
        ui.stage_enabled_checkBox->setChecked(true);  // ALWAYS TRUE FOR TRUNK
        ui.budget_spinBox->setValue(opt_settings_.trunk_budget);
        ui.x_translation_spinBox->setValue(opt_settings_.trunk_range.x);
        ui.y_translation_spinBox->setValue(opt_settings_.trunk_range.y);
        ui.z_translation_spinBox->setValue(opt_settings_.trunk_range.z);
        ui.x_rotation_spinBox->setValue(opt_settings_.trunk_range.xa);
        ui.y_rotation_spinBox->setValue(opt_settings_.trunk_range.ya);
        ui.z_rotation_spinBox->setValue(opt_settings_.trunk_range.za);
        /*Disable Branch options*/

        ui.branch_total_count_label->setVisible(false);
        ui.branch_count_spinBox->setVisible(false);
        ui.branch_count_spinBox->setEnabled(false);
        ui.branch_count_spinBox->setValue(opt_settings_.number_branches);

        /*Select Active Cost Function*/
        for (int i = 0; i < ui.cost_function_listWidget->count(); i++) {
            if (ui.cost_function_listWidget->item(i)->text() ==
                QString::fromStdString(
                    sc_trunk_manager_.getActiveCostFunction())) {
                ui.cost_function_listWidget->setCurrentRow(i);
                break;
            }
            if (i == (ui.cost_function_listWidget->count() - 1))
                QMessageBox::critical(this, "Error!",
                                      "Active cost function not found!",
                                      QMessageBox::Ok);
        }
    }
    /*Select the first Parameter if There are any Parameters*/
    if (ui.cost_function_parameters_listWidget->count() > 0)
        ui.cost_function_parameters_listWidget->setCurrentRow(0);
}

/*On List Widgets Changed*/
/*Cost Function Selection*/
void SettingsControl::on_cost_function_listWidget_itemSelectionChanged() {
    /*Clear Parameter List*/
    ui.cost_function_parameters_listWidget->clear();
    ui.cost_function_parameters_listWidget->clearSelection();
    /*Get Rid of Parameters Index and Hide the Value View*/
    ui.double_parameter_spinBox->setEnabled(false);
    ui.double_parameter_spinBox->setVisible(false);
    ui.int_parameter_spinBox->setEnabled(false);
    ui.int_parameter_spinBox->setVisible(false);
    ui.bool_parameter_true_radioButton->setEnabled(false);
    ui.bool_parameter_true_radioButton->setVisible(false);
    ui.bool_parameter_false_radioButton->setEnabled(false);
    ui.bool_parameter_false_radioButton->setVisible(false);

    /*Save Selection as Active Cost Function and update aparameter list*/
    if (ui.trunk_radioButton->isChecked()) {
        sc_trunk_manager_.setActiveCostFunction(
            ui.cost_function_listWidget->currentItem()->text().toStdString());
        std::vector<jta_cost_function::Parameter<double>>
            temp_double_params_vec =
                sc_trunk_manager_.getActiveCostFunctionClass()
                    ->getDoubleParameters();
        for (int i = 0; i < temp_double_params_vec.size(); i++) {
            ui.cost_function_parameters_listWidget->addItem(
                QString::fromStdString(
                    temp_double_params_vec[i].getParameterName()));
        }
        std::vector<jta_cost_function::Parameter<int>> temp_int_params_vec =
            sc_trunk_manager_.getActiveCostFunctionClass()->getIntParameters();
        for (int i = 0; i < temp_int_params_vec.size(); i++) {
            ui.cost_function_parameters_listWidget->addItem(
                QString::fromStdString(
                    temp_int_params_vec[i].getParameterName()));
        }
        std::vector<jta_cost_function::Parameter<bool>> temp_bool_params_vec =
            sc_trunk_manager_.getActiveCostFunctionClass()->getBoolParameters();
        for (int i = 0; i < temp_bool_params_vec.size(); i++) {
            ui.cost_function_parameters_listWidget->addItem(
                QString::fromStdString(
                    temp_bool_params_vec[i].getParameterName()));
        }
    } else if (ui.branch_radioButton->isChecked()) {
        sc_branch_manager_.setActiveCostFunction(
            ui.cost_function_listWidget->currentItem()->text().toStdString());
        std::vector<jta_cost_function::Parameter<double>>
            temp_double_params_vec =
                sc_branch_manager_.getActiveCostFunctionClass()
                    ->getDoubleParameters();
        for (int i = 0; i < temp_double_params_vec.size(); i++) {
            ui.cost_function_parameters_listWidget->addItem(
                QString::fromStdString(
                    temp_double_params_vec[i].getParameterName()));
        }
        std::vector<jta_cost_function::Parameter<int>> temp_int_params_vec =
            sc_branch_manager_.getActiveCostFunctionClass()->getIntParameters();
        for (int i = 0; i < temp_int_params_vec.size(); i++) {
            ui.cost_function_parameters_listWidget->addItem(
                QString::fromStdString(
                    temp_int_params_vec[i].getParameterName()));
        }
        std::vector<jta_cost_function::Parameter<bool>> temp_bool_params_vec =
            sc_branch_manager_.getActiveCostFunctionClass()
                ->getBoolParameters();
        for (int i = 0; i < temp_bool_params_vec.size(); i++) {
            ui.cost_function_parameters_listWidget->addItem(
                QString::fromStdString(
                    temp_bool_params_vec[i].getParameterName()));
        }
    } else {
        sc_leaf_manager_.setActiveCostFunction(
            ui.cost_function_listWidget->currentItem()->text().toStdString());
        std::vector<jta_cost_function::Parameter<double>>
            temp_double_params_vec =
                sc_leaf_manager_.getActiveCostFunctionClass()
                    ->getDoubleParameters();
        for (int i = 0; i < temp_double_params_vec.size(); i++) {
            ui.cost_function_parameters_listWidget->addItem(
                QString::fromStdString(
                    temp_double_params_vec[i].getParameterName()));
        }
        std::vector<jta_cost_function::Parameter<int>> temp_int_params_vec =
            sc_leaf_manager_.getActiveCostFunctionClass()->getIntParameters();
        for (int i = 0; i < temp_int_params_vec.size(); i++) {
            ui.cost_function_parameters_listWidget->addItem(
                QString::fromStdString(
                    temp_int_params_vec[i].getParameterName()));
        }
        std::vector<jta_cost_function::Parameter<bool>> temp_bool_params_vec =
            sc_leaf_manager_.getActiveCostFunctionClass()->getBoolParameters();
        for (int i = 0; i < temp_bool_params_vec.size(); i++) {
            ui.cost_function_parameters_listWidget->addItem(
                QString::fromStdString(
                    temp_bool_params_vec[i].getParameterName()));
        }
    }

    /*Select the first Parameter if There are any Parameters*/
    if (ui.cost_function_parameters_listWidget->count() > 0)
        ui.cost_function_parameters_listWidget->setCurrentRow(0);
};

void SettingsControl::
    on_cost_function_parameters_listWidget_itemSelectionChanged() {
    /*Get Rid of Parameters Index and Hide the Value View*/
    ui.double_parameter_spinBox->setEnabled(false);
    ui.double_parameter_spinBox->setVisible(false);
    ui.int_parameter_spinBox->setEnabled(false);
    ui.int_parameter_spinBox->setVisible(false);
    ui.bool_parameter_true_radioButton->setEnabled(false);
    ui.bool_parameter_true_radioButton->setVisible(false);
    ui.bool_parameter_false_radioButton->setEnabled(false);
    ui.bool_parameter_false_radioButton->setVisible(false);

    QString param_name_widg =
        ui.cost_function_parameters_listWidget->currentItem()->text();
    /*Get Parameter Value and Type*/
    if (ui.trunk_radioButton->isChecked()) {
        std::vector<jta_cost_function::Parameter<double>>
            temp_double_params_vec =
                sc_trunk_manager_.getActiveCostFunctionClass()
                    ->getDoubleParameters();
        for (int i = 0; i < temp_double_params_vec.size(); i++) {
            if (QString::fromStdString(
                    temp_double_params_vec[i].getParameterName()) ==
                param_name_widg) {
                ui.double_parameter_spinBox->setVisible(true);
                ui.double_parameter_spinBox->setEnabled(true);
                ui.double_parameter_spinBox->setValue(
                    temp_double_params_vec[i].getParameterValue());
                return;
            }
        }
        std::vector<jta_cost_function::Parameter<int>> temp_int_params_vec =
            sc_trunk_manager_.getActiveCostFunctionClass()->getIntParameters();
        for (int i = 0; i < temp_int_params_vec.size(); i++) {
            if (QString::fromStdString(
                    temp_int_params_vec[i].getParameterName()) ==
                param_name_widg) {
                ui.int_parameter_spinBox->setVisible(true);
                ui.int_parameter_spinBox->setEnabled(true);
                ui.int_parameter_spinBox->setValue(
                    temp_int_params_vec[i].getParameterValue());
                return;
            }
        }
        std::vector<jta_cost_function::Parameter<bool>> temp_bool_params_vec =
            sc_trunk_manager_.getActiveCostFunctionClass()->getBoolParameters();
        for (int i = 0; i < temp_bool_params_vec.size(); i++) {
            if (QString::fromStdString(
                    temp_bool_params_vec[i].getParameterName()) ==
                param_name_widg) {
                ui.bool_parameter_true_radioButton->setEnabled(true);
                ui.bool_parameter_true_radioButton->setVisible(true);
                ui.bool_parameter_false_radioButton->setEnabled(true);
                ui.bool_parameter_false_radioButton->setVisible(true);
                ui.bool_parameter_true_radioButton->setChecked(
                    temp_bool_params_vec[i].getParameterValue());
                ui.bool_parameter_false_radioButton->setChecked(
                    !temp_bool_params_vec[i].getParameterValue());
                return;
            }
        }

    } else if (ui.branch_radioButton->isChecked()) {
        std::vector<jta_cost_function::Parameter<double>>
            temp_double_params_vec =
                sc_branch_manager_.getActiveCostFunctionClass()
                    ->getDoubleParameters();
        for (int i = 0; i < temp_double_params_vec.size(); i++) {
            if (QString::fromStdString(
                    temp_double_params_vec[i].getParameterName()) ==
                param_name_widg) {
                ui.double_parameter_spinBox->setVisible(true);
                ui.double_parameter_spinBox->setEnabled(true);
                ui.double_parameter_spinBox->setValue(
                    temp_double_params_vec[i].getParameterValue());
                return;
            }
        }
        std::vector<jta_cost_function::Parameter<int>> temp_int_params_vec =
            sc_branch_manager_.getActiveCostFunctionClass()->getIntParameters();
        for (int i = 0; i < temp_int_params_vec.size(); i++) {
            if (QString::fromStdString(
                    temp_int_params_vec[i].getParameterName()) ==
                param_name_widg) {
                ui.int_parameter_spinBox->setVisible(true);
                ui.int_parameter_spinBox->setEnabled(true);
                ui.int_parameter_spinBox->setValue(
                    temp_int_params_vec[i].getParameterValue());
                return;
            }
        }
        std::vector<jta_cost_function::Parameter<bool>> temp_bool_params_vec =
            sc_branch_manager_.getActiveCostFunctionClass()
                ->getBoolParameters();
        for (int i = 0; i < temp_bool_params_vec.size(); i++) {
            if (QString::fromStdString(
                    temp_bool_params_vec[i].getParameterName()) ==
                param_name_widg) {
                ui.bool_parameter_true_radioButton->setEnabled(true);
                ui.bool_parameter_true_radioButton->setVisible(true);
                ui.bool_parameter_false_radioButton->setEnabled(true);
                ui.bool_parameter_false_radioButton->setVisible(true);
                ui.bool_parameter_true_radioButton->setChecked(
                    temp_bool_params_vec[i].getParameterValue());
                ui.bool_parameter_false_radioButton->setChecked(
                    !temp_bool_params_vec[i].getParameterValue());
                return;
            }
        }
    } else {
        std::vector<jta_cost_function::Parameter<double>>
            temp_double_params_vec =
                sc_leaf_manager_.getActiveCostFunctionClass()
                    ->getDoubleParameters();
        for (int i = 0; i < temp_double_params_vec.size(); i++) {
            if (QString::fromStdString(
                    temp_double_params_vec[i].getParameterName()) ==
                param_name_widg) {
                ui.double_parameter_spinBox->setVisible(true);
                ui.double_parameter_spinBox->setEnabled(true);
                ui.double_parameter_spinBox->setValue(
                    temp_double_params_vec[i].getParameterValue());
                return;
            }
        }
        std::vector<jta_cost_function::Parameter<int>> temp_int_params_vec =
            sc_leaf_manager_.getActiveCostFunctionClass()->getIntParameters();
        for (int i = 0; i < temp_int_params_vec.size(); i++) {
            if (QString::fromStdString(
                    temp_int_params_vec[i].getParameterName()) ==
                param_name_widg) {
                ui.int_parameter_spinBox->setVisible(true);
                ui.int_parameter_spinBox->setEnabled(true);
                ui.int_parameter_spinBox->setValue(
                    temp_int_params_vec[i].getParameterValue());
                return;
            }
        }
        std::vector<jta_cost_function::Parameter<bool>> temp_bool_params_vec =
            sc_leaf_manager_.getActiveCostFunctionClass()->getBoolParameters();
        for (int i = 0; i < temp_bool_params_vec.size(); i++) {
            if (QString::fromStdString(
                    temp_bool_params_vec[i].getParameterName()) ==
                param_name_widg) {
                ui.bool_parameter_true_radioButton->setEnabled(true);
                ui.bool_parameter_true_radioButton->setVisible(true);
                ui.bool_parameter_false_radioButton->setEnabled(true);
                ui.bool_parameter_false_radioButton->setVisible(true);
                ui.bool_parameter_true_radioButton->setChecked(
                    temp_bool_params_vec[i].getParameterValue());
                ui.bool_parameter_false_radioButton->setChecked(
                    !temp_bool_params_vec[i].getParameterValue());
                return;
            }
        }
    }
};

/*Radio buttons for stage*/
void SettingsControl::on_trunk_radioButton_clicked() {
    /*Get Rid of Parameters Index and Hide the Value View*/
    ui.cost_function_parameters_listWidget->clearSelection();
    ui.double_parameter_spinBox->setEnabled(false);
    ui.double_parameter_spinBox->setVisible(false);
    ui.int_parameter_spinBox->setEnabled(false);
    ui.int_parameter_spinBox->setVisible(false);
    ui.bool_parameter_true_radioButton->setEnabled(false);
    ui.bool_parameter_true_radioButton->setVisible(false);
    ui.bool_parameter_false_radioButton->setEnabled(false);
    ui.bool_parameter_false_radioButton->setVisible(false);

    /*Populate Available List Widget and Select the Current One*/
    std::vector<jta_cost_function::CostFunction> available_cost_functions =
        sc_trunk_manager_.getAvailableCostFunctions();

    /*Load Optimizer Settings (non-cost function)*/
    ui.stage_enabled_checkBox->setChecked(true);  // ALWAYS TRUE FOR TRUNK
    ui.budget_spinBox->setValue(opt_settings_.trunk_budget);
    ui.x_translation_spinBox->setValue(opt_settings_.trunk_range.x);
    ui.y_translation_spinBox->setValue(opt_settings_.trunk_range.y);
    ui.z_translation_spinBox->setValue(opt_settings_.trunk_range.z);
    ui.x_rotation_spinBox->setValue(opt_settings_.trunk_range.xa);
    ui.y_rotation_spinBox->setValue(opt_settings_.trunk_range.ya);
    ui.z_rotation_spinBox->setValue(opt_settings_.trunk_range.za);
    /*Disable Branch options*/
    ui.branch_total_count_label->setVisible(false);
    ui.branch_count_spinBox->setVisible(false);
    ui.branch_count_spinBox->setEnabled(false);

    /*Select Active Cost Function*/
    for (int i = 0; i < ui.cost_function_listWidget->count(); i++) {
        if (ui.cost_function_listWidget->item(i)->text() ==
            QString::fromStdString(sc_trunk_manager_.getActiveCostFunction())) {
            ui.cost_function_listWidget->setCurrentRow(i);
            break;
        }
        if (i == (ui.cost_function_listWidget->count() - 1))
            QMessageBox::critical(this, "Error!",
                                  "Active cost function not found!",
                                  QMessageBox::Ok);
    }

    /*Select the first Parameter if There are any Parameters*/
    if (ui.cost_function_parameters_listWidget->count() > 0)
        ui.cost_function_parameters_listWidget->setCurrentRow(0);
};

void SettingsControl::on_branch_radioButton_clicked() {
    /*Get Rid of Parameters Index and Hide the Value View*/
    ui.cost_function_parameters_listWidget->clearSelection();
    ui.double_parameter_spinBox->setEnabled(false);
    ui.double_parameter_spinBox->setVisible(false);
    ui.int_parameter_spinBox->setEnabled(false);
    ui.int_parameter_spinBox->setVisible(false);
    ui.bool_parameter_true_radioButton->setEnabled(false);
    ui.bool_parameter_true_radioButton->setVisible(false);
    ui.bool_parameter_false_radioButton->setEnabled(false);
    ui.bool_parameter_false_radioButton->setVisible(false);

    /*Populate Available List Widget and Select the Current One*/
    std::vector<jta_cost_function::CostFunction> available_cost_functions =
        sc_branch_manager_.getAvailableCostFunctions();

    /*Load Optimizer Settings (non-cost function)*/
    ui.stage_enabled_checkBox->setChecked(opt_settings_.enable_branch_);
    ui.budget_spinBox->setValue(opt_settings_.branch_budget);
    ui.x_translation_spinBox->setValue(opt_settings_.branch_range.x);
    ui.y_translation_spinBox->setValue(opt_settings_.branch_range.y);
    ui.z_translation_spinBox->setValue(opt_settings_.branch_range.z);
    ui.x_rotation_spinBox->setValue(opt_settings_.branch_range.xa);
    ui.y_rotation_spinBox->setValue(opt_settings_.branch_range.ya);
    ui.z_rotation_spinBox->setValue(opt_settings_.branch_range.za);
    /*Enable Branch options*/
    ui.branch_total_count_label->setVisible(true);
    ui.branch_count_spinBox->setVisible(true);
    ui.branch_count_spinBox->setEnabled(true);

    /*Select Active Cost Function*/
    for (int i = 0; i < ui.cost_function_listWidget->count(); i++) {
        if (ui.cost_function_listWidget->item(i)->text() ==
            QString::fromStdString(
                sc_branch_manager_.getActiveCostFunction())) {
            ui.cost_function_listWidget->setCurrentRow(i);
            break;
        }
        if (i == (ui.cost_function_listWidget->count() - 1))
            QMessageBox::critical(this, "Error!",
                                  "Active cost function not found!",
                                  QMessageBox::Ok);
    }

    /*Select the first Parameter if There are any Parameters*/
    if (ui.cost_function_parameters_listWidget->count() > 0)
        ui.cost_function_parameters_listWidget->setCurrentRow(0);
};

void SettingsControl::on_leaf_radioButton_clicked() {
    /*Get Rid of Parameters Index and Hide the Value View*/
    ui.cost_function_parameters_listWidget->clearSelection();
    ui.double_parameter_spinBox->setEnabled(false);
    ui.double_parameter_spinBox->setVisible(false);
    ui.int_parameter_spinBox->setEnabled(false);
    ui.int_parameter_spinBox->setVisible(false);
    ui.bool_parameter_true_radioButton->setEnabled(false);
    ui.bool_parameter_true_radioButton->setVisible(false);
    ui.bool_parameter_false_radioButton->setEnabled(false);
    ui.bool_parameter_false_radioButton->setVisible(false);

    /*Populate Available List Widget and Select the Current One*/
    std::vector<jta_cost_function::CostFunction> available_cost_functions =
        sc_leaf_manager_.getAvailableCostFunctions();

    /*Load Optimizer Settings (non-cost function)*/
    ui.stage_enabled_checkBox->setChecked(opt_settings_.enable_leaf_);
    ui.budget_spinBox->setValue(opt_settings_.leaf_budget);
    ui.x_translation_spinBox->setValue(opt_settings_.leaf_range.x);
    ui.y_translation_spinBox->setValue(opt_settings_.leaf_range.y);
    ui.z_translation_spinBox->setValue(opt_settings_.leaf_range.z);
    ui.x_rotation_spinBox->setValue(opt_settings_.leaf_range.xa);
    ui.y_rotation_spinBox->setValue(opt_settings_.leaf_range.ya);
    ui.z_rotation_spinBox->setValue(opt_settings_.leaf_range.za);
    /*Disable Branch options*/
    ui.branch_total_count_label->setVisible(false);
    ui.branch_count_spinBox->setVisible(false);
    ui.branch_count_spinBox->setEnabled(false);

    /*Select Active Cost Function*/
    for (int i = 0; i < ui.cost_function_listWidget->count(); i++) {
        if (ui.cost_function_listWidget->item(i)->text() ==
            QString::fromStdString(sc_leaf_manager_.getActiveCostFunction())) {
            ui.cost_function_listWidget->setCurrentRow(i);
            break;
        }
        if (i == (ui.cost_function_listWidget->count() - 1))
            QMessageBox::critical(this, "Error!",
                                  "Active cost function not found!",
                                  QMessageBox::Ok);
    }

    /*Select the first Parameter if There are any Parameters*/
    if (ui.cost_function_parameters_listWidget->count() > 0)
        ui.cost_function_parameters_listWidget->setCurrentRow(0);
};

/*Optimizer Settings Buttons Toggled*/
void SettingsControl::on_stage_enabled_checkBox_clicked() {
    if (ui.trunk_radioButton->isChecked()) {
        ui.stage_enabled_checkBox->setChecked(true);
        QMessageBox::critical(this, "Not allowed!",
                              "Optimizer must always have a trunk stage!",
                              QMessageBox::Ok);
    } else if (ui.branch_radioButton->isChecked()) {
        opt_settings_.enable_branch_ = ui.stage_enabled_checkBox->isChecked();
    } else {
        opt_settings_.enable_leaf_ = ui.stage_enabled_checkBox->isChecked();
    }
};

void SettingsControl::on_budget_spinBox_valueChanged() {
    if (ui.trunk_radioButton->isChecked()) {
        opt_settings_.trunk_budget = ui.budget_spinBox->value();
    } else if (ui.branch_radioButton->isChecked()) {
        opt_settings_.branch_budget = ui.budget_spinBox->value();
    } else {
        opt_settings_.leaf_budget = ui.budget_spinBox->value();
    }
};

void SettingsControl::on_x_translation_spinBox_valueChanged() {
    if (ui.trunk_radioButton->isChecked()) {
        opt_settings_.trunk_range.x = ui.x_translation_spinBox->value();
    } else if (ui.branch_radioButton->isChecked()) {
        opt_settings_.branch_range.x = ui.x_translation_spinBox->value();
    } else {
        opt_settings_.leaf_range.x = ui.x_translation_spinBox->value();
    }
};

void SettingsControl::on_y_translation_spinBox_valueChanged() {
    if (ui.trunk_radioButton->isChecked()) {
        opt_settings_.trunk_range.y = ui.y_translation_spinBox->value();
    } else if (ui.branch_radioButton->isChecked()) {
        opt_settings_.branch_range.y = ui.y_translation_spinBox->value();
    } else {
        opt_settings_.leaf_range.y = ui.y_translation_spinBox->value();
    }
};

void SettingsControl::on_z_translation_spinBox_valueChanged() {
    if (ui.trunk_radioButton->isChecked()) {
        opt_settings_.trunk_range.z = ui.z_translation_spinBox->value();
    } else if (ui.branch_radioButton->isChecked()) {
        opt_settings_.branch_range.z = ui.z_translation_spinBox->value();
    } else {
        opt_settings_.leaf_range.z = ui.z_translation_spinBox->value();
    }
};

void SettingsControl::on_x_rotation_spinBox_valueChanged() {
    if (ui.trunk_radioButton->isChecked()) {
        opt_settings_.trunk_range.xa = ui.x_rotation_spinBox->value();
    } else if (ui.branch_radioButton->isChecked()) {
        opt_settings_.branch_range.xa = ui.x_rotation_spinBox->value();
    } else {
        opt_settings_.leaf_range.xa = ui.x_rotation_spinBox->value();
    }
};

void SettingsControl::on_y_rotation_spinBox_valueChanged() {
    if (ui.trunk_radioButton->isChecked()) {
        opt_settings_.trunk_range.ya = ui.y_rotation_spinBox->value();
    } else if (ui.branch_radioButton->isChecked()) {
        opt_settings_.branch_range.ya = ui.y_rotation_spinBox->value();
    } else {
        opt_settings_.leaf_range.ya = ui.y_rotation_spinBox->value();
    }
};

void SettingsControl::on_z_rotation_spinBox_valueChanged() {
    if (ui.trunk_radioButton->isChecked()) {
        opt_settings_.trunk_range.za = ui.z_rotation_spinBox->value();
    } else if (ui.branch_radioButton->isChecked()) {
        opt_settings_.branch_range.za = ui.z_rotation_spinBox->value();
    } else {
        opt_settings_.leaf_range.za = ui.z_rotation_spinBox->value();
    }
};

void SettingsControl::on_branch_count_spinBox_valueChanged() {
    if (ui.trunk_radioButton->isChecked()) {
    } else if (ui.branch_radioButton->isChecked()) {
        opt_settings_.number_branches = ui.branch_count_spinBox->value();
    } else {
    }
};

void SettingsControl::on_double_parameter_spinBox_valueChanged() {
    if (ui.trunk_radioButton->isChecked()) {
        if (!sc_trunk_manager_.getActiveCostFunctionClass()
                 ->setDoubleParameterValue(
                     ui.cost_function_parameters_listWidget->currentItem()
                         ->text()
                         .toStdString(),
                     ui.double_parameter_spinBox->value())) {
            QMessageBox::critical(this, "Error!",
                                  "Could not update parameter value!",
                                  QMessageBox::Ok);
        }
    } else if (ui.branch_radioButton->isChecked()) {
        if (!sc_branch_manager_.getActiveCostFunctionClass()
                 ->setDoubleParameterValue(
                     ui.cost_function_parameters_listWidget->currentItem()
                         ->text()
                         .toStdString(),
                     ui.double_parameter_spinBox->value())) {
            QMessageBox::critical(this, "Error!",
                                  "Could not update parameter value!",
                                  QMessageBox::Ok);
        }
    } else {
        if (!sc_leaf_manager_.getActiveCostFunctionClass()
                 ->setDoubleParameterValue(
                     ui.cost_function_parameters_listWidget->currentItem()
                         ->text()
                         .toStdString(),
                     ui.double_parameter_spinBox->value())) {
            QMessageBox::critical(this, "Error!",
                                  "Could not update parameter value!",
                                  QMessageBox::Ok);
        }
    }
};

void SettingsControl::on_int_parameter_spinBox_valueChanged() {
    if (ui.trunk_radioButton->isChecked()) {
        if (!sc_trunk_manager_.getActiveCostFunctionClass()
                 ->setIntParameterValue(
                     ui.cost_function_parameters_listWidget->currentItem()
                         ->text()
                         .toStdString(),
                     ui.int_parameter_spinBox->value())) {
            QMessageBox::critical(this, "Error!",
                                  "Could not update parameter value!",
                                  QMessageBox::Ok);
        }
    } else if (ui.branch_radioButton->isChecked()) {
        if (!sc_branch_manager_.getActiveCostFunctionClass()
                 ->setIntParameterValue(
                     ui.cost_function_parameters_listWidget->currentItem()
                         ->text()
                         .toStdString(),
                     ui.int_parameter_spinBox->value())) {
            QMessageBox::critical(this, "Error!",
                                  "Could not update parameter value!",
                                  QMessageBox::Ok);
        }
    } else {
        if (!sc_leaf_manager_.getActiveCostFunctionClass()
                 ->setIntParameterValue(
                     ui.cost_function_parameters_listWidget->currentItem()
                         ->text()
                         .toStdString(),
                     ui.int_parameter_spinBox->value())) {
            QMessageBox::critical(this, "Error!",
                                  "Could not update parameter value!",
                                  QMessageBox::Ok);
        }
    }
};

void SettingsControl::on_bool_parameter_true_radioButton_clicked() {
    if (ui.trunk_radioButton->isChecked()) {
        if (!sc_trunk_manager_.getActiveCostFunctionClass()
                 ->setBoolParameterValue(
                     ui.cost_function_parameters_listWidget->currentItem()
                         ->text()
                         .toStdString(),
                     ui.bool_parameter_true_radioButton->isChecked())) {
            QMessageBox::critical(this, "Error!",
                                  "Could not update parameter value!",
                                  QMessageBox::Ok);
        }
    } else if (ui.branch_radioButton->isChecked()) {
        if (!sc_branch_manager_.getActiveCostFunctionClass()
                 ->setBoolParameterValue(
                     ui.cost_function_parameters_listWidget->currentItem()
                         ->text()
                         .toStdString(),
                     ui.bool_parameter_true_radioButton->isChecked())) {
            QMessageBox::critical(this, "Error!",
                                  "Could not update parameter value!",
                                  QMessageBox::Ok);
        }
    } else {
        if (!sc_leaf_manager_.getActiveCostFunctionClass()
                 ->setBoolParameterValue(
                     ui.cost_function_parameters_listWidget->currentItem()
                         ->text()
                         .toStdString(),
                     ui.bool_parameter_true_radioButton->isChecked())) {
            QMessageBox::critical(this, "Error!",
                                  "Could not update parameter value!",
                                  QMessageBox::Ok);
        }
    }
};

void SettingsControl::on_bool_parameter_false_radioButton_clicked() {
    if (ui.trunk_radioButton->isChecked()) {
        if (!sc_trunk_manager_.getActiveCostFunctionClass()
                 ->setBoolParameterValue(
                     ui.cost_function_parameters_listWidget->currentItem()
                         ->text()
                         .toStdString(),
                     ui.bool_parameter_true_radioButton->isChecked())) {
            QMessageBox::critical(this, "Error!",
                                  "Could not update parameter value!",
                                  QMessageBox::Ok);
        }
    } else if (ui.branch_radioButton->isChecked()) {
        if (!sc_branch_manager_.getActiveCostFunctionClass()
                 ->setBoolParameterValue(
                     ui.cost_function_parameters_listWidget->currentItem()
                         ->text()
                         .toStdString(),
                     ui.bool_parameter_true_radioButton->isChecked())) {
            QMessageBox::critical(this, "Error!",
                                  "Could not update parameter value!",
                                  QMessageBox::Ok);
        }
    } else {
        if (!sc_leaf_manager_.getActiveCostFunctionClass()
                 ->setBoolParameterValue(
                     ui.cost_function_parameters_listWidget->currentItem()
                         ->text()
                         .toStdString(),
                     ui.bool_parameter_true_radioButton->isChecked())) {
            QMessageBox::critical(this, "Error!",
                                  "Could not update parameter value!",
                                  QMessageBox::Ok);
        }
    }
};

/*Save Button*/
void SettingsControl::on_save_button_clicked() {
    /*Emit Save Settings*/
    emit SaveSettings(opt_settings_, sc_trunk_manager_, sc_branch_manager_,
                      sc_leaf_manager_);

    /*Close Window*/
    emit Done();
}

/*Reset Button*/
void SettingsControl::on_reset_button_clicked() {
    /*Reinitialize Optimizer Settings and 3 Cost Function Managers*/
    opt_settings_ = OptimizerSettings();
    sc_trunk_manager_ = jta_cost_function::CostFunctionManager(Stage::Trunk);
    sc_branch_manager_ = jta_cost_function::CostFunctionManager(Stage::Branch);
    sc_leaf_manager_ = jta_cost_function::CostFunctionManager(Stage::Leaf);

    /*Change the Default Settings of Dilation for branch and leaf to 4 and 1
     * respectively*/
    sc_branch_manager_.getCostFunctionClass("DIRECT_DILATION")
        ->setIntParameterValue("Dilation", 4);
    sc_leaf_manager_.getCostFunctionClass("DIRECT_DILATION")
        ->setIntParameterValue("Dilation", 1);

    /*Refresh Settings GUI*/
    /*Get Rid of Parameters Index and Hide the Value View*/
    ui.cost_function_parameters_listWidget->clearSelection();
    ui.double_parameter_spinBox->setEnabled(false);
    ui.double_parameter_spinBox->setVisible(false);
    ui.int_parameter_spinBox->setEnabled(false);
    ui.int_parameter_spinBox->setVisible(false);
    ui.bool_parameter_true_radioButton->setEnabled(false);
    ui.bool_parameter_true_radioButton->setVisible(false);
    ui.bool_parameter_false_radioButton->setEnabled(false);
    ui.bool_parameter_false_radioButton->setVisible(false);
    ui.cost_function_parameters_listWidget->clearSelection();

    /*Populate Available List Widget and Select the Current One*/
    std::vector<jta_cost_function::CostFunction> available_cost_functions =
        sc_trunk_manager_.getAvailableCostFunctions();
    ui.trunk_radioButton->setChecked(true);  // Always load to trunk
    if (ui.trunk_radioButton->isChecked()) {
        /*Load Optimizer Settings (non-cost function)*/
        ui.stage_enabled_checkBox->setChecked(true);  // ALWAYS TRUE FOR TRUNK
        ui.budget_spinBox->setValue(opt_settings_.trunk_budget);
        ui.x_translation_spinBox->setValue(opt_settings_.trunk_range.x);
        ui.y_translation_spinBox->setValue(opt_settings_.trunk_range.y);
        ui.z_translation_spinBox->setValue(opt_settings_.trunk_range.z);
        ui.x_rotation_spinBox->setValue(opt_settings_.trunk_range.xa);
        ui.y_rotation_spinBox->setValue(opt_settings_.trunk_range.ya);
        ui.z_rotation_spinBox->setValue(opt_settings_.trunk_range.za);
        /*Disable Branch options*/

        ui.branch_total_count_label->setVisible(false);
        ui.branch_count_spinBox->setVisible(false);
        ui.branch_count_spinBox->setEnabled(false);
        ui.branch_count_spinBox->setValue(opt_settings_.number_branches);

        /*Select Active Cost Function*/
        for (int i = 0; i < ui.cost_function_listWidget->count(); i++) {
            if (ui.cost_function_listWidget->item(i)->text() ==
                QString::fromStdString(
                    sc_trunk_manager_.getActiveCostFunction())) {
                ui.cost_function_listWidget->setCurrentRow(i);
            }
        }
    }
    /*Select the first Parameter if There are any Parameters*/
    if (ui.cost_function_parameters_listWidget->count() > 0)
        ui.cost_function_parameters_listWidget->setCurrentRow(0);
}

/*Cancel Button*/
void SettingsControl::on_cancel_button_clicked() {
    /*Close Window*/
    emit Done();
}
