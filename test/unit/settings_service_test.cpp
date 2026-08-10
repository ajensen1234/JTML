// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Deterministic Catch2 twin for the SettingsService (complements
// test_settings_service_properties.cpp; plan 004 U3 / R9 / R14). Pins the
// registry parity contract (org/app/group/key names), first-run detection +
// defaults, the edge-write value transport (the per-site widget/frame sourcing
// lives in the MainScreen slots until U5 -- this pins the service side: the
// three passed values land on the exact three keys), and the missing/corrupt
// group error path. All tests redirect QSettings to a temp ini file -- the
// real registry is never touched.

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include <QSettings>
#include <QString>
#include <QStringList>
#include <QTemporaryDir>

#include "services/settings_service.h"

namespace {

/*A temp ini file that is removed with the fixture.*/
struct TempSettings {
    QTemporaryDir dir;
    QString path;
    TempSettings() : path(dir.filePath("settings.ini")) {
        REQUIRE(dir.isValid());
    }
};

QStringList sorted(const QStringList& list) {
    QStringList copy = list;
    std::sort(copy.begin(), copy.end());
    return copy;
}

}  // namespace

TEST_CASE(
    "settings_service: registry parity contract (org/app/group/key names)",
    "[settings_service]") {
    REQUIRE(jta::SettingsService::OrganizationName() == "JointTrackAutoGPU");
    REQUIRE(jta::SettingsService::ApplicationName() == "Version340");

    TempSettings tmp;
    {
        jta::SettingsService svc(tmp.path, QSettings::IniFormat);
        /*Cost-function keys use the STAGE@ACTIVE_CF and
         * STAGE@CFname@ParamName@TYPE formats.*/
        svc.SaveCostFunctionSettings({
            {"TRUNK@ACTIVE_CF", QStringLiteral("DIRECT_DILATION")},
            {"TRUNK@DIRECT_DILATION@Dilation@INT", 5},
            {"TRUNK@DIRECT_MAHFOUZ@d1@DOUBLE", 1.5},
            {"BRANCH@ACTIVE_CF", QStringLiteral("DIRECT_MAHFOUZ")},
            {"LEAF@ACTIVE_CF", QStringLiteral("DIRECT_DILATION")},
        });
        OptimizerSettings opt;
        svc.SaveOptimizerSettings(opt);
        svc.SaveEdgeSettings(3, 40, 120);
        svc.MarkFirstTimeDone();
    }

    /*Read the ini back raw and check the exact group/key table.*/
    QSettings raw(tmp.path, QSettings::IniFormat);
    REQUIRE(sorted(raw.childGroups()) ==
            sorted(QStringList({"CostFunctionSettings",
                                "EdgeDetectionSettings",
                                "FirstTime",
                                "OptimizerSettings"})));

    raw.beginGroup("CostFunctionSettings");
    REQUIRE(sorted(raw.allKeys()) ==
            sorted(QStringList({"BRANCH@ACTIVE_CF",
                                "LEAF@ACTIVE_CF",
                                "TRUNK@ACTIVE_CF",
                                "TRUNK@DIRECT_DILATION@Dilation@INT",
                                "TRUNK@DIRECT_MAHFOUZ@d1@DOUBLE"})));
    REQUIRE(raw.value("TRUNK@ACTIVE_CF").toString() == "DIRECT_DILATION");
    REQUIRE(raw.value("TRUNK@DIRECT_DILATION@Dilation@INT").toInt() == 5);
    REQUIRE(raw.value("TRUNK@DIRECT_MAHFOUZ@d1@DOUBLE").toDouble() == 1.5);
    raw.endGroup();

    raw.beginGroup("OptimizerSettings");
    REQUIRE(sorted(raw.allKeys()) ==
            sorted(QStringList({"BRANCH@BUDGET",
                                "BRANCH@ENABLE",
                                "BRANCH@NUMBER_BRANCHES",
                                "BRANCH@RANGE_X",
                                "BRANCH@RANGE_XA",
                                "BRANCH@RANGE_Y",
                                "BRANCH@RANGE_YA",
                                "BRANCH@RANGE_Z",
                                "BRANCH@RANGE_ZA",
                                "LEAF@BUDGET",
                                "LEAF@ENABLE",
                                "LEAF@RANGE_X",
                                "LEAF@RANGE_XA",
                                "LEAF@RANGE_Y",
                                "LEAF@RANGE_YA",
                                "LEAF@RANGE_Z",
                                "LEAF@RANGE_ZA",
                                "TRUNK@BUDGET",
                                "TRUNK@RANGE_X",
                                "TRUNK@RANGE_XA",
                                "TRUNK@RANGE_Y",
                                "TRUNK@RANGE_YA",
                                "TRUNK@RANGE_Z",
                                "TRUNK@RANGE_ZA"})));
    raw.endGroup();

    raw.beginGroup("EdgeDetectionSettings");
    REQUIRE(sorted(raw.allKeys()) ==
            sorted(QStringList({"APERTURE", "HIGH_THRESH", "LOW_THRESH"})));
    REQUIRE(raw.value("APERTURE").toInt() == 3);
    REQUIRE(raw.value("LOW_THRESH").toInt() == 40);
    REQUIRE(raw.value("HIGH_THRESH").toInt() == 120);
    raw.endGroup();

    raw.beginGroup("FirstTime");
    REQUIRE(raw.value("JTAFirstTime").toBool() == false);
    raw.endGroup();
}

TEST_CASE(
    "settings_service: first-run detection returns defaults and creates "
    "FirstTime",
    "[settings_service]") {
    TempSettings tmp;
    {
        jta::SettingsService svc(tmp.path, QSettings::IniFormat);
        REQUIRE(svc.IsFirstTime());

        jta::SettingsService::LoadResult result = svc.LoadSettings();
        REQUIRE(result.first_time);
        REQUIRE(result.cost_function_entries.empty());
        /*Defaults: OptimizerSettings() + the edge constants.*/
        REQUIRE(result.optimizer.trunk_budget == TRUNK_BUDGET);
        REQUIRE(result.optimizer.trunk_range.x == TRUNK_RANGE.x);
        REQUIRE(result.optimizer.branch_budget == BRANCH_BUDGET);
        REQUIRE(result.optimizer.enable_branch_ == ENABLE_BRANCH);
        REQUIRE(result.optimizer.leaf_budget == Z_SEARCH_BUDGET);
        REQUIRE(result.optimizer.enable_leaf_ == ENABLE_Z);
        REQUIRE(result.edge.aperture == APERTURE);
        REQUIRE(result.edge.low_thresh == LOW_THRESH);
        REQUIRE(result.edge.high_thresh == HIGH_THRESH);

        /*The view calls MarkFirstTimeDone() only after the CUDA probe passes
         * (original gate preserved).*/
        svc.MarkFirstTimeDone();
    }

    jta::SettingsService svc(tmp.path, QSettings::IniFormat);
    REQUIRE_FALSE(svc.IsFirstTime());
}

TEST_CASE(
    "settings_service: fractional and negative values round-trip bit-exact",
    "[settings_service]") {
    TempSettings tmp;
    {
        jta::SettingsService svc(tmp.path, QSettings::IniFormat);
        svc.SaveCostFunctionSettings({
            {"TRUNK@f1@DOUBLE", 0.1},
            {"TRUNK@f2@DOUBLE", -2.25},
            {"TRUNK@f3@DOUBLE", 3.141592653589793},
            {"TRUNK@f4@DOUBLE", -0.0},
            {"TRUNK@f5@DOUBLE", 1e-300},
            {"TRUNK@i1@INT", -42},
            {"TRUNK@i2@INT", 0},
            {"TRUNK@b1@BOOL", true},
            {"TRUNK@b2@BOOL", false},
        });
        OptimizerSettings opt;
        opt.trunk_range =
            Point6D(0.1, -2.25, 3.141592653589793, -0.0, 1e-300, 0.0);
        opt.trunk_budget = -42;
        opt.enable_branch_ = true;
        svc.SaveOptimizerSettings(opt);
        svc.SaveEdgeSettings(-1, -2, -3);
    }

    jta::SettingsService svc(tmp.path, QSettings::IniFormat);
    jta::SettingsService::LoadResult loaded = svc.LoadSettings();
    REQUIRE_FALSE(loaded.first_time);
    /*Keys come back sorted (QSettings::allKeys() ordering) -- look up by key.*/
    auto entry = [&](const QString& key) -> const QVariant& {
        for (const jta::RegistryEntry& e : loaded.cost_function_entries) {
            if (e.key == key) return e.value;
        }
        FAIL("missing key: " + key.toStdString());
        static const QVariant dummy;
        return dummy;
    };
    REQUIRE(loaded.cost_function_entries.size() == 9);
    REQUIRE(entry("TRUNK@f1@DOUBLE").toDouble() == 0.1);
    REQUIRE(entry("TRUNK@f2@DOUBLE").toDouble() == -2.25);
    REQUIRE(entry("TRUNK@f3@DOUBLE").toDouble() ==
            3.141592653589793);
    REQUIRE(entry("TRUNK@f4@DOUBLE").toDouble() == -0.0);
    REQUIRE(entry("TRUNK@f5@DOUBLE").toDouble() == 1e-300);
    REQUIRE(entry("TRUNK@i1@INT").toInt() == -42);
    REQUIRE(entry("TRUNK@i2@INT").toInt() == 0);
    REQUIRE(entry("TRUNK@b1@BOOL").toBool() == true);
    REQUIRE(entry("TRUNK@b2@BOOL").toBool() == false);
    REQUIRE(loaded.optimizer.trunk_range.x == 0.1);
    REQUIRE(loaded.optimizer.trunk_range.y == -2.25);
    REQUIRE(loaded.optimizer.trunk_range.z == 3.141592653589793);
    REQUIRE(loaded.optimizer.trunk_range.xa == -0.0);
    REQUIRE(loaded.optimizer.trunk_range.ya == 1e-300);
    REQUIRE(loaded.optimizer.trunk_range.za == 0.0);
    REQUIRE(loaded.optimizer.trunk_budget == -42);
    REQUIRE(loaded.optimizer.enable_branch_ == true);
    REQUIRE(loaded.edge.aperture == -1);
    REQUIRE(loaded.edge.low_thresh == -2);
    REQUIRE(loaded.edge.high_thresh == -3);
}

TEST_CASE(
    "settings_service: SaveEdgeSettings transports each passed value to its "
    "exact key (per-site sourcing)",
    "[settings_service]") {
    TempSettings tmp;
    {
        jta::SettingsService svc(tmp.path, QSettings::IniFormat);
        /*The four MainScreen edge sites differ only in how they source the
         * three values (own widget vs frame); the service must write exactly
         * what it is given, with no cross-key derivation. Simulate the
         * slider-slot shape: own key from the widget, the other two from the
         * current frame. Apply-all's shape is the same call with all three
         * widget values -- identical transport.*/
        const int frame_aperture = 7;
        const int frame_low = 55;
        const int frame_high = 130;
        svc.SaveEdgeSettings(frame_aperture, frame_low, frame_high);
    }

    QSettings raw(tmp.path, QSettings::IniFormat);
    raw.beginGroup("EdgeDetectionSettings");
    REQUIRE(raw.value("APERTURE").toInt() == 7);
    REQUIRE(raw.value("LOW_THRESH").toInt() == 55);
    REQUIRE(raw.value("HIGH_THRESH").toInt() == 130);
    raw.endGroup();
}

TEST_CASE(
    "settings_service: missing/corrupt groups load as zero-fills, no crash",
    "[settings_service]") {
    TempSettings tmp;
    {
        /*A non-first-run registry (groups exist) with a corrupt value and
         * missing groups/keys. QSettings semantics preserved: reads on
         * missing keys yield 0/false (the plan's "defaults" wording; the
         * original code had no default-constant fallback here -- R13 keeps
         * that exactly).*/
        QSettings raw(tmp.path, QSettings::IniFormat);
        raw.beginGroup("OptimizerSettings");
        raw.setValue("TRUNK@BUDGET", QStringLiteral("not-an-int"));
        raw.endGroup();
        raw.beginGroup("EdgeDetectionSettings");
        raw.setValue("APERTURE", QStringLiteral("garbage"));
        raw.endGroup();
        /*No CostFunctionSettings group at all.*/
    }

    jta::SettingsService svc(tmp.path, QSettings::IniFormat);
    REQUIRE_FALSE(svc.IsFirstTime());
    jta::SettingsService::LoadResult result = svc.LoadSettings();
    REQUIRE_FALSE(result.first_time);
    REQUIRE(result.cost_function_entries.empty());
    REQUIRE(result.optimizer.trunk_budget == 0);
    REQUIRE(result.optimizer.trunk_range.x == 0.0);
    REQUIRE(result.optimizer.enable_branch_ == false);
    REQUIRE(result.edge.aperture == 0);
    REQUIRE(result.edge.low_thresh == 0);
    REQUIRE(result.edge.high_thresh == 0);
}
