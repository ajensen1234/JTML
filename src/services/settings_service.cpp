/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*SettingsService implementation (plan 004 U3 / R9): the QSettings round-trip
 * relocated verbatim from MainScreen::LoadSettingsBetweenSessions /
 * onSaveSettings / the four edge-slot writes. No behavior change: same
 * org/app/group/key names, same first-run detection, same value conversions
 * (missing keys read as 0/false).*/

#include "services/settings_service.h"

#include <QStringList>

namespace jta {

SettingsService::SettingsService(
    const QString& settings_path, QSettings::Format format)
    : settings_(settings_path.isEmpty()
                    ? QSettings(OrganizationName(), ApplicationName())
                    : QSettings(settings_path, format)) {}

QString SettingsService::OrganizationName() {
    return QStringLiteral("JointTrackAutoGPU");
}

QString SettingsService::ApplicationName() {
    return QStringLiteral("Version") + QString::number(VER_FIRST_NUM) +
           QString::number(VER_MIDDLE_NUM) + QString::number(VER_LAST_NUM);
}

bool SettingsService::IsFirstTime() const {
    return settings_.childGroups().size() == 0;
}

SettingsService::LoadResult SettingsService::LoadSettings() {
    LoadResult result;
    result.first_time = IsFirstTime();
    if (result.first_time) {
        /*Defaults (mirrors the first-run branch of the original code:
         * OptimizerSettings() plus the APERTURE/LOW_THRESH/HIGH_THRESH
         * constants).*/
        result.optimizer = OptimizerSettings();
        result.edge = EdgeSettings();
        return result;
    }

    /*Cost Function Settings: raw keys + values. The view applies them to the
     * CostFunctionManagers and owns the key-format error dialogs (codes
     * A-F), exactly as before.*/
    settings_.beginGroup("CostFunctionSettings");
    const QStringList cost_function_settings_keys = settings_.allKeys();
    result.cost_function_entries.reserve(cost_function_settings_keys.size());
    for (const QString& key : cost_function_settings_keys) {
        result.cost_function_entries.push_back({key, settings_.value(key)});
    }
    settings_.endGroup();

    /*Optimizer Settings (same keys + conversions as the original code;
     * missing keys read as 0/false).*/
    settings_.beginGroup("OptimizerSettings");
    /*Variables*/
    /*Trunk*/
    result.optimizer.trunk_range = Point6D(
        settings_.value("TRUNK@RANGE_X").toDouble(),
        settings_.value("TRUNK@RANGE_Y").toDouble(),
        settings_.value("TRUNK@RANGE_Z").toDouble(),
        settings_.value("TRUNK@RANGE_XA").toDouble(),
        settings_.value("TRUNK@RANGE_YA").toDouble(),
        settings_.value("TRUNK@RANGE_ZA").toDouble());
    result.optimizer.trunk_budget =
        settings_.value("TRUNK@BUDGET").toInt();

    /*Branch*/
    result.optimizer.branch_range = Point6D(
        settings_.value("BRANCH@RANGE_X").toDouble(),
        settings_.value("BRANCH@RANGE_Y").toDouble(),
        settings_.value("BRANCH@RANGE_Z").toDouble(),
        settings_.value("BRANCH@RANGE_XA").toDouble(),
        settings_.value("BRANCH@RANGE_YA").toDouble(),
        settings_.value("BRANCH@RANGE_ZA").toDouble());
    result.optimizer.number_branches =
        settings_.value("BRANCH@NUMBER_BRANCHES").toInt();
    result.optimizer.enable_branch_ =
        settings_.value("BRANCH@ENABLE").toBool();
    result.optimizer.branch_budget =
        settings_.value("BRANCH@BUDGET").toInt();

    /*Leaf*/
    result.optimizer.leaf_range = Point6D(
        settings_.value("LEAF@RANGE_X").toDouble(),
        settings_.value("LEAF@RANGE_Y").toDouble(),
        settings_.value("LEAF@RANGE_Z").toDouble(),
        settings_.value("LEAF@RANGE_XA").toDouble(),
        settings_.value("LEAF@RANGE_YA").toDouble(),
        settings_.value("LEAF@RANGE_ZA").toDouble());
    result.optimizer.enable_leaf_ =
        settings_.value("LEAF@ENABLE").toBool();
    result.optimizer.leaf_budget = settings_.value("LEAF@BUDGET").toInt();
    settings_.endGroup();

    /*Edge Detection Settings*/
    settings_.beginGroup("EdgeDetectionSettings");
    result.edge.aperture = settings_.value("APERTURE").toInt();
    result.edge.low_thresh = settings_.value("LOW_THRESH").toInt();
    result.edge.high_thresh = settings_.value("HIGH_THRESH").toInt();
    settings_.endGroup();

    return result;
}

void SettingsService::MarkFirstTimeDone() {
    settings_.beginGroup("FirstTime");
    settings_.setValue("JTAFirstTime", false);
    settings_.endGroup();
}

void SettingsService::SaveCostFunctionSettings(
    const std::vector<RegistryEntry>& entries) {
    settings_.beginGroup("CostFunctionSettings");
    for (const RegistryEntry& entry : entries) {
        settings_.setValue(entry.key, entry.value);
    }
    settings_.endGroup();
}

void SettingsService::SaveOptimizerSettings(
    const OptimizerSettings& settings) {
    settings_.beginGroup("OptimizerSettings");
    /*Variables*/
    /*Trunk*/
    settings_.setValue("TRUNK@RANGE_X", settings.trunk_range.x);
    settings_.setValue("TRUNK@RANGE_Y", settings.trunk_range.y);
    settings_.setValue("TRUNK@RANGE_Z", settings.trunk_range.z);
    settings_.setValue("TRUNK@RANGE_XA", settings.trunk_range.xa);
    settings_.setValue("TRUNK@RANGE_YA", settings.trunk_range.ya);
    settings_.setValue("TRUNK@RANGE_ZA", settings.trunk_range.za);
    settings_.setValue("TRUNK@BUDGET", settings.trunk_budget);

    /*Branch*/
    settings_.setValue("BRANCH@RANGE_X", settings.branch_range.x);
    settings_.setValue("BRANCH@RANGE_Y", settings.branch_range.y);
    settings_.setValue("BRANCH@RANGE_Z", settings.branch_range.z);
    settings_.setValue("BRANCH@RANGE_XA", settings.branch_range.xa);
    settings_.setValue("BRANCH@RANGE_YA", settings.branch_range.ya);
    settings_.setValue("BRANCH@RANGE_ZA", settings.branch_range.za);
    settings_.setValue(
        "BRANCH@NUMBER_BRANCHES", settings.number_branches);
    settings_.setValue("BRANCH@ENABLE", settings.enable_branch_);
    settings_.setValue("BRANCH@BUDGET", settings.branch_budget);

    /*Leaf*/
    settings_.setValue("LEAF@RANGE_X", settings.leaf_range.x);
    settings_.setValue("LEAF@RANGE_Y", settings.leaf_range.y);
    settings_.setValue("LEAF@RANGE_Z", settings.leaf_range.z);
    settings_.setValue("LEAF@RANGE_XA", settings.leaf_range.xa);
    settings_.setValue("LEAF@RANGE_YA", settings.leaf_range.ya);
    settings_.setValue("LEAF@RANGE_ZA", settings.leaf_range.za);
    settings_.setValue("LEAF@ENABLE", settings.enable_leaf_);
    settings_.setValue("LEAF@BUDGET", settings.leaf_budget);
    settings_.endGroup();
}

void SettingsService::SaveEdgeSettings(
    int aperture, int low_thresh, int high_thresh) {
    settings_.beginGroup("EdgeDetectionSettings");
    settings_.setValue("APERTURE", aperture);
    settings_.setValue("LOW_THRESH", low_thresh);
    settings_.setValue("HIGH_THRESH", high_thresh);
    settings_.endGroup();
}

} /* namespace jta */
