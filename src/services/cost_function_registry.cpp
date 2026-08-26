/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*Cost-function registry mapping (plan 006 U1 / R8): the raw
 * CostFunctionSettings entries builder shared by the widgets app (MainScreen)
 * and the QML app (SettingsBridge). Relocated verbatim from
 * MainScreen::BuildCostFunctionRegistryEntries (mainscreen.cpp:4895, plan
 * 004 U3) — the body below is byte-identical to the pre-extraction mapping
 * (R13); the golden 51-entry fixture (test/unit/cost_function_registry_test
 * .cpp + test/unit/experimental_settings_test.cpp) pins it. The module is a
 * compute-linked sibling of the QtCore-only SettingsService: it reads the
 * GPU-linked CostFunctionManager (jtml_services links jtml_compute PRIVATE).*/

#include "services/cost_function_registry.h"

#include "compute/CostFunctionManager.h"

namespace jta {

std::vector<RegistryEntry> BuildCostFunctionRegistryEntries(
    jta_cost_function::CostFunctionManager& trunk_manager,
    jta_cost_function::CostFunctionManager& branch_manager,
    jta_cost_function::CostFunctionManager& leaf_manager) {
    std::vector<jta::RegistryEntry> entries;

    /*Cost Function Managers (Save All Values for Parameters and Active Cost
     * Function*/
    /*Trunk*/
    entries.push_back(jta::RegistryEntry{
        QStringLiteral("TRUNK@ACTIVE_CF"),
        QString::fromStdString(trunk_manager.getActiveCostFunction())});
    std::vector<jta_cost_function::CostFunction> trunk_cost_functions =
        trunk_manager.getAvailableCostFunctions();
    for (int i = 0; i < trunk_cost_functions.size(); i++) {
        std::vector<jta_cost_function::Parameter<double>>
            trunk_parameters_double =
                trunk_cost_functions[i].getDoubleParameters();
        for (int j = 0; j < trunk_parameters_double.size(); j++) {
            entries.push_back(jta::RegistryEntry{
                QString::fromStdString(
                    "TRUNK@" + trunk_cost_functions[i].getCostFunctionName() +
                    "@" + trunk_parameters_double[j].getParameterName() + "@" +
                    trunk_parameters_double[j].getParameterType()),
                trunk_parameters_double[j].getParameterValue()});
        }
        std::vector<jta_cost_function::Parameter<int>> trunk_parameters_int =
            trunk_cost_functions[i].getIntParameters();
        for (int j = 0; j < trunk_parameters_int.size(); j++) {
            entries.push_back(jta::RegistryEntry{
                QString::fromStdString(
                    "TRUNK@" + trunk_cost_functions[i].getCostFunctionName() +
                    "@" + trunk_parameters_int[j].getParameterName() + "@" +
                    trunk_parameters_int[j].getParameterType()),
                trunk_parameters_int[j].getParameterValue()});
        }
        std::vector<jta_cost_function::Parameter<bool>> trunk_parameters_bool =
            trunk_cost_functions[i].getBoolParameters();
        for (int j = 0; j < trunk_parameters_bool.size(); j++) {
            entries.push_back(jta::RegistryEntry{
                QString::fromStdString(
                    "TRUNK@" + trunk_cost_functions[i].getCostFunctionName() +
                    "@" + trunk_parameters_bool[j].getParameterName() + "@" +
                    trunk_parameters_bool[j].getParameterType()),
                trunk_parameters_bool[j].getParameterValue()});
        }
    }

    /*Branch*/
    entries.push_back(jta::RegistryEntry{
        QStringLiteral("BRANCH@ACTIVE_CF"),
        QString::fromStdString(branch_manager.getActiveCostFunction())});
    std::vector<jta_cost_function::CostFunction> branch_cost_functions =
        branch_manager.getAvailableCostFunctions();
    for (int i = 0; i < branch_cost_functions.size(); i++) {
        std::vector<jta_cost_function::Parameter<double>>
            branch_parameters_double =
                branch_cost_functions[i].getDoubleParameters();
        for (int j = 0; j < branch_parameters_double.size(); j++) {
            entries.push_back(jta::RegistryEntry{
                QString::fromStdString(
                    "BRANCH@" + branch_cost_functions[i].getCostFunctionName() +
                    "@" + branch_parameters_double[j].getParameterName() + "@" +
                    branch_parameters_double[j].getParameterType()),
                branch_parameters_double[j].getParameterValue()});
        }
        std::vector<jta_cost_function::Parameter<int>> branch_parameters_int =
            branch_cost_functions[i].getIntParameters();
        for (int j = 0; j < branch_parameters_int.size(); j++) {
            entries.push_back(jta::RegistryEntry{
                QString::fromStdString(
                    "BRANCH@" + branch_cost_functions[i].getCostFunctionName() +
                    "@" + branch_parameters_int[j].getParameterName() + "@" +
                    branch_parameters_int[j].getParameterType()),
                branch_parameters_int[j].getParameterValue()});
        }
        std::vector<jta_cost_function::Parameter<bool>> branch_parameters_bool =
            branch_cost_functions[i].getBoolParameters();
        for (int j = 0; j < branch_parameters_bool.size(); j++) {
            entries.push_back(jta::RegistryEntry{
                QString::fromStdString(
                    "BRANCH@" + branch_cost_functions[i].getCostFunctionName() +
                    "@" + branch_parameters_bool[j].getParameterName() + "@" +
                    branch_parameters_bool[j].getParameterType()),
                branch_parameters_bool[j].getParameterValue()});
        }
    }

    /*Leaf*/
    entries.push_back(jta::RegistryEntry{
        QStringLiteral("LEAF@ACTIVE_CF"),
        QString::fromStdString(leaf_manager.getActiveCostFunction())});
    std::vector<jta_cost_function::CostFunction> leaf_cost_functions =
        leaf_manager.getAvailableCostFunctions();
    for (int i = 0; i < leaf_cost_functions.size(); i++) {
        std::vector<jta_cost_function::Parameter<double>>
            leaf_parameters_double =
                leaf_cost_functions[i].getDoubleParameters();
        for (int j = 0; j < leaf_parameters_double.size(); j++) {
            entries.push_back(jta::RegistryEntry{
                QString::fromStdString(
                    "LEAF@" + leaf_cost_functions[i].getCostFunctionName() +
                    "@" + leaf_parameters_double[j].getParameterName() + "@" +
                    leaf_parameters_double[j].getParameterType()),
                leaf_parameters_double[j].getParameterValue()});
        }
        std::vector<jta_cost_function::Parameter<int>> leaf_parameters_int =
            leaf_cost_functions[i].getIntParameters();
        for (int j = 0; j < leaf_parameters_int.size(); j++) {
            entries.push_back(jta::RegistryEntry{
                QString::fromStdString(
                    "LEAF@" + leaf_cost_functions[i].getCostFunctionName() +
                    "@" + leaf_parameters_int[j].getParameterName() + "@" +
                    leaf_parameters_int[j].getParameterType()),
                leaf_parameters_int[j].getParameterValue()});
        }
        std::vector<jta_cost_function::Parameter<bool>> leaf_parameters_bool =
            leaf_cost_functions[i].getBoolParameters();
        for (int j = 0; j < leaf_parameters_bool.size(); j++) {
            entries.push_back(jta::RegistryEntry{
                QString::fromStdString(
                    "LEAF@" + leaf_cost_functions[i].getCostFunctionName() +
                    "@" + leaf_parameters_bool[j].getParameterName() + "@" +
                    leaf_parameters_bool[j].getParameterType()),
                leaf_parameters_bool[j].getParameterValue()});
        }
    }

    return entries;
}

} /* namespace jta */
