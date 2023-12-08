/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#ifndef SETTINGS_CONTROL_H
#define SETTINGS_CONTROL_H

#include <qdialog.h>

#include "ui_settings_control.h"

/*Optimizer Settings Class*/
#include "core/optimizer_settings.h"

/*JTA Cost Function Class*/
#include "cost_functions/CostFunctionManager.h"

// About OptimizerSettings Popup Header
class SettingsControl : public QDialog {
    Q_OBJECT

   public:
    SettingsControl(QWidget* parent = 0,
                    Qt::WindowFlags flags = Qt::WindowFlags());
    ~SettingsControl() override;

    /*Load Optimizer Settings from Main Window*/
    void LoadSettings(jta_cost_function::CostFunctionManager sc_trunk_manager,
                      jta_cost_function::CostFunctionManager sc_branch_manager,
                      jta_cost_function::CostFunctionManager sc_leaf_manager,
                      OptimizerSettings opt_settings);

   private:
    Ui::settings_control ui;

    /*Local to Settings Control Cost Function Managers*/
    jta_cost_function::CostFunctionManager sc_trunk_manager_;
    jta_cost_function::CostFunctionManager sc_branch_manager_;
    jta_cost_function::CostFunctionManager
        sc_leaf_manager_;  // For extra Z-translation usually (esp. when
                           // monoplane)

    /*Optimizer Settings for Everything but the Cost Function Stuff*/
    OptimizerSettings opt_settings_;

   public slots:
    /*Save Button*/
    void on_save_button_clicked();

    /*Reset Button*/
    void on_reset_button_clicked();

    /*Cancel Button*/
    void on_cancel_button_clicked();

    /*Radio buttons for stage*/
    void on_trunk_radioButton_clicked();
    void on_branch_radioButton_clicked();
    void on_leaf_radioButton_clicked();

    /*List Widgets Changed*/
    void on_cost_function_listWidget_itemSelectionChanged();
    void on_cost_function_parameters_listWidget_itemSelectionChanged();

    /*Optimizer Settings Buttons Toggled*/
    void on_stage_enabled_checkBox_clicked();
    void on_budget_spinBox_valueChanged();
    void on_x_translation_spinBox_valueChanged();
    void on_y_translation_spinBox_valueChanged();
    void on_z_translation_spinBox_valueChanged();
    void on_x_rotation_spinBox_valueChanged();
    void on_y_rotation_spinBox_valueChanged();
    void on_z_rotation_spinBox_valueChanged();
    void on_branch_count_spinBox_valueChanged();
    void on_double_parameter_spinBox_valueChanged();
    void on_int_parameter_spinBox_valueChanged();
    void on_bool_parameter_true_radioButton_clicked();
    void on_bool_parameter_false_radioButton_clicked();

   signals:
    /*Saves the three Cost Function Manager Settings and the Optimizer Settings
    to:
    - the registry
    - their local class versions on the main window GUI*/
    void SaveSettings(OptimizerSettings, jta_cost_function::CostFunctionManager,
                      jta_cost_function::CostFunctionManager,
                      jta_cost_function::CostFunctionManager);  //
    /*Close Window*/
    void Done();  //
};

#endif /* SETTINGS_CONTROL_H */
