/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*SettingsService (plan 004 U3 / R9): widget-free QSettings persistence for the
 * settings previously written inline in MainScreen
 * (LoadSettingsBetweenSessions, onSaveSettings, the four edge-slot writes).
 * QtCore-only: takes/returns values, never reads ui.* widgets, never shows
 * dialogs.
 *
 * Registry parity is a hard contract (plan C4): org "JointTrackAutoGPU", app
 * "Version" + VER_FIRST_NUM+VER_MIDDLE_NUM+VER_LAST_NUM ("Version340"), groups
 * CostFunctionSettings / OptimizerSettings / EdgeDetectionSettings / FirstTime,
 * keys APERTURE / LOW_THRESH / HIGH_THRESH and STAGE@ACTIVE_CF /
 * STAGE@CFname@ParamName@TYPE; first-run detection via childGroups().size()==0.
 *
 * The cost-function settings round-trip as RAW (key, value) entries so the
 * service stays free of the GPU-linked CostFunctionManager; MainScreen maps the
 * managers <-> entries (it already links jtml_compute).*/

#ifndef SETTINGS_SERVICE_H
#define SETTINGS_SERVICE_H

#include <QSettings>
#include <QString>
#include <QVariant>
#include <vector>

#include "domain/settings_constants.h"
#include "services/optimizer_settings.h"

namespace jta {

/*Edge-detection persisted values (registry group EdgeDetectionSettings, keys
 * APERTURE / LOW_THRESH / HIGH_THRESH).*/
struct EdgeSettings {
    int aperture = APERTURE;
    int low_thresh = LOW_THRESH;
    int high_thresh = HIGH_THRESH;
};

/*One raw registry entry: key + stored value. Cost-function entries use the
 * STAGE@ACTIVE_CF / STAGE@CFname@ParamName@TYPE key formats.*/
struct RegistryEntry {
    QString key;
    QVariant value;
};

class SettingsService {
public:
    /*Default: the real registry (org "JointTrackAutoGPU", app "Version340").
     * Tests pass an ini path + QSettings::IniFormat to isolate to a temp file
     * -- they never touch the real registry.*/
    explicit SettingsService(
        const QString& settings_path = QString(),
        QSettings::Format format = QSettings::NativeFormat);
    ~SettingsService() = default;

    /*Registry parity contract (plan C4).*/
    static QString OrganizationName();
    static QString ApplicationName();

    /*First-run detection: no groups at all in the registry.*/
    bool IsFirstTime() const;

    struct LoadResult {
        bool first_time = false;
        /*Raw CostFunctionSettings keys + values (empty on first run).*/
        std::vector<RegistryEntry> cost_function_entries;
        /*Loaded OptimizerSettings; defaults on first run; zero-filled reads on
         * missing keys (QSettings semantics preserved).*/
        OptimizerSettings optimizer;
        /*Loaded edge values; defaults on first run; 0 on missing keys.*/
        EdgeSettings edge;
    };
    LoadResult LoadSettings();

    /*Writes group FirstTime, key JTAFirstTime=false. The view calls this only
     * after the CUDA probe passes (preserves the original gate).*/
    void MarkFirstTimeDone();

    /*Writes group CostFunctionSettings verbatim, in entry order.*/
    void SaveCostFunctionSettings(const std::vector<RegistryEntry>& entries);

    /*Writes group OptimizerSettings (the exact key set of the original code).*/
    void SaveOptimizerSettings(const OptimizerSettings& settings);

    /*Writes group EdgeDetectionSettings (keys APERTURE/LOW_THRESH/HIGH_THRESH).
     * Caller passes the per-site sourced values exactly as computed today.*/
    void SaveEdgeSettings(int aperture, int low_thresh, int high_thresh);

private:
    QSettings settings_;
};

} /* namespace jta */

#endif /* SETTINGS_SERVICE_H */
