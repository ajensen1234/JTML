// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 006 U1: the shared cost-function registry mapping pins (R8 / AE3).
// Deterministic Catch2 headless tests that direct-compile the shared
// jta::BuildCostFunctionRegistryEntries (services/cost_function_registry.cpp —
// the ONE mapping both front-ends call: the widgets MainScreen and the QML
// SettingsBridge) against the real SettingsService (settings_service.cpp +
// optimizer_settings.cpp) with QSettings redirected to a temp ini (the real
// registry is never touched) and link the real CostFunctionManager/CostFunction
// from jtml_compute (the constructor + parameter surface exercised here are
// CPU-only; no CUDA calls at runtime).
//
// Scenarios (plan 006 U1):
//  (a) happy: the shared function with the widgets first-run manager
//      configuration (fresh managers + branch/leaf DIRECT_DILATION Dilation
//      =4/1 overrides) produces the exact golden 51-entry table (17 per
//      stage, %.17g exact IEEE-754 round-trip) — the fixture below is a
//      verbatim copy of the one in experimental_settings_test.cpp (kept in
//      sync; the capture procedure is documented there) and the 51 VALUES are
//      the pre-extraction widgets output (R13: verbatim relocation, the
//      golden table is the spec);
//  (b) SettingsBridge::save() -> registry -> LoadSettings round-trip
//      unchanged — covered by experimental_settings_test.cpp (the bridge's
//      mapping is now this shared function; the parity pin + save-path tests
//      there re-run against it);
//  (c) edge: zero-stage (default-constructed) managers — the closest
//      constructible "empty" configuration — are tolerated: no crash, the
//      mapping stays well-defined (it never reads stage/GPU state), and a
//      manager with an empty available-cost-functions list would contribute
//      only its ACTIVE_CF entry (the parameter loops skip);
//  (d) integration: the widgets save path (managers -> shared mapping ->
//      SaveCostFunctionSettings -> LoadSettings) lands the identical golden
//      entry set on disk.

#include <catch2/catch_test_macros.hpp>

#include <QMetaType>
#include <QSettings>
#include <QString>
#include <QStringList>
#include <QTemporaryDir>
#include <QVariant>

#include <cstring>
#include <map>

#include "compute/CostFunctionManager.h"
#include "services/cost_function_registry.h"
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

/*One golden entry: (key, stored type, value). Doubles print at %.17g (exact
 * round-trip through QString::toDouble).*/
struct GoldenEntry {
    const char* key;
    const char* type;
    const char* value;
};

/*Verbatim copy of the golden fixture in experimental_settings_test.cpp
 * (capture procedure documented there — thrown-away friend-injection harness
 * over the REAL pre-extraction MainScreen method, 2026-08-11). The 51 VALUES
 * are the pinned spec of the shared mapping: 17 per stage — ACTIVE_CF, then
 * per available cost function the double/int/bool parameter groups, in
 * listCostFunctions order. Keep the two copies in sync.*/
const GoldenEntry kGoldenCostFunctionEntries[] = {
    {"TRUNK@ACTIVE_CF", "string", "DIRECT_DILATION"},
    {"TRUNK@sym_trap_function@PoleWeight@DOUBLE", "double", "75"},
    {"TRUNK@sym_trap_function@VVWeight@DOUBLE", "double", "500"},
    {"TRUNK@sym_trap_function@Dilation@INT", "int", "3"},
    {"TRUNK@DD_NEW_POLE_CONSTRAINT@PoleWeight@DOUBLE", "double", "75"},
    {"TRUNK@DD_NEW_POLE_CONSTRAINT@Dilation@INT", "int", "3"},
    {"TRUNK@DD_NEW_POLE_CONSTRAINT@X_TRANS@BOOL", "bool", "false"},
    {"TRUNK@DD_NEW_POLE_CONSTRAINT@Y_TRANS@BOOL", "bool", "false"},
    {"TRUNK@DD_NEW_POLE_CONSTRAINT@Z_TRANS@BOOL", "bool", "false"},
    {"TRUNK@DIRECT_DILATION_POLE_CONSTRAINT@PoleWeight@DOUBLE", "double", "1"},
    {"TRUNK@DIRECT_DILATION_POLE_CONSTRAINT@Pole_Weight@DOUBLE", "double", "1"},
    {"TRUNK@DIRECT_DILATION_POLE_CONSTRAINT@Dilation@INT", "int", "6"},
    {"TRUNK@DIRECT_DILATION_SAME_Z@Z_Weight@DOUBLE", "double", "1"},
    {"TRUNK@DIRECT_DILATION_SAME_Z@Dilation@INT", "int", "6"},
    {"TRUNK@DIRECT_DILATION_T1@Dilation@INT", "int", "6"},
    {"TRUNK@DIRECT_DILATION@Dilation@INT", "int", "6"},
    {"TRUNK@DIRECT_MAHFOUZ@Black_Silhouette@BOOL", "bool", "true"},
    {"BRANCH@ACTIVE_CF", "string", "DIRECT_DILATION"},
    {"BRANCH@sym_trap_function@PoleWeight@DOUBLE", "double", "75"},
    {"BRANCH@sym_trap_function@VVWeight@DOUBLE", "double", "500"},
    {"BRANCH@sym_trap_function@Dilation@INT", "int", "3"},
    {"BRANCH@DD_NEW_POLE_CONSTRAINT@PoleWeight@DOUBLE", "double", "75"},
    {"BRANCH@DD_NEW_POLE_CONSTRAINT@Dilation@INT", "int", "3"},
    {"BRANCH@DD_NEW_POLE_CONSTRAINT@X_TRANS@BOOL", "bool", "false"},
    {"BRANCH@DD_NEW_POLE_CONSTRAINT@Y_TRANS@BOOL", "bool", "false"},
    {"BRANCH@DD_NEW_POLE_CONSTRAINT@Z_TRANS@BOOL", "bool", "false"},
    {"BRANCH@DIRECT_DILATION_POLE_CONSTRAINT@PoleWeight@DOUBLE", "double", "1"},
    {"BRANCH@DIRECT_DILATION_POLE_CONSTRAINT@Pole_Weight@DOUBLE",
     "double",
     "1"},
    {"BRANCH@DIRECT_DILATION_POLE_CONSTRAINT@Dilation@INT", "int", "6"},
    {"BRANCH@DIRECT_DILATION_SAME_Z@Z_Weight@DOUBLE", "double", "1"},
    {"BRANCH@DIRECT_DILATION_SAME_Z@Dilation@INT", "int", "6"},
    {"BRANCH@DIRECT_DILATION_T1@Dilation@INT", "int", "6"},
    {"BRANCH@DIRECT_DILATION@Dilation@INT", "int", "4"},
    {"BRANCH@DIRECT_MAHFOUZ@Black_Silhouette@BOOL", "bool", "true"},
    {"LEAF@ACTIVE_CF", "string", "DIRECT_DILATION"},
    {"LEAF@sym_trap_function@PoleWeight@DOUBLE", "double", "75"},
    {"LEAF@sym_trap_function@VVWeight@DOUBLE", "double", "500"},
    {"LEAF@sym_trap_function@Dilation@INT", "int", "3"},
    {"LEAF@DD_NEW_POLE_CONSTRAINT@PoleWeight@DOUBLE", "double", "75"},
    {"LEAF@DD_NEW_POLE_CONSTRAINT@Dilation@INT", "int", "3"},
    {"LEAF@DD_NEW_POLE_CONSTRAINT@X_TRANS@BOOL", "bool", "false"},
    {"LEAF@DD_NEW_POLE_CONSTRAINT@Y_TRANS@BOOL", "bool", "false"},
    {"LEAF@DD_NEW_POLE_CONSTRAINT@Z_TRANS@BOOL", "bool", "false"},
    {"LEAF@DIRECT_DILATION_POLE_CONSTRAINT@PoleWeight@DOUBLE", "double", "1"},
    {"LEAF@DIRECT_DILATION_POLE_CONSTRAINT@Pole_Weight@DOUBLE", "double", "1"},
    {"LEAF@DIRECT_DILATION_POLE_CONSTRAINT@Dilation@INT", "int", "6"},
    {"LEAF@DIRECT_DILATION_SAME_Z@Z_Weight@DOUBLE", "double", "1"},
    {"LEAF@DIRECT_DILATION_SAME_Z@Dilation@INT", "int", "6"},
    {"LEAF@DIRECT_DILATION_T1@Dilation@INT", "int", "6"},
    {"LEAF@DIRECT_DILATION@Dilation@INT", "int", "1"},
    {"LEAF@DIRECT_MAHFOUZ@Black_Silhouette@BOOL", "bool", "true"},
};
constexpr size_t kGoldenEntryCount =
    sizeof(kGoldenCostFunctionEntries) / sizeof(kGoldenCostFunctionEntries[0]);

bool sameBits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

/*Compare one produced registry entry against one golden entry (key + stored
 * type + exact value — bit-exact for doubles).*/
void requireEntryMatches(
    const jta::RegistryEntry& produced, const GoldenEntry& golden) {
    INFO("key: " << produced.key.toStdString());
    REQUIRE(produced.key == QString::fromLatin1(golden.key));
    const QString type = QString::fromLatin1(golden.type);
    if (type == QLatin1String("string")) {
        REQUIRE(produced.value.typeId() == QMetaType::QString);
        REQUIRE(produced.value.toString() == QString::fromLatin1(golden.value));
    } else if (type == QLatin1String("int")) {
        REQUIRE(produced.value.typeId() == QMetaType::Int);
        REQUIRE(
            produced.value.toInt() ==
            QString::fromLatin1(golden.value).toInt());
    } else if (type == QLatin1String("bool")) {
        REQUIRE(produced.value.typeId() == QMetaType::Bool);
        REQUIRE(
            produced.value.toBool() ==
            (QString::fromLatin1(golden.value) == QLatin1String("true")));
    } else {
        REQUIRE(produced.value.typeId() == QMetaType::Double);
        const double expected = QString::fromLatin1(golden.value).toDouble();
        REQUIRE(sameBits(expected, produced.value.toDouble()));
    }
}

/*The widgets first-run manager configuration (the fixed configuration the
 * golden fixture was captured for): fresh managers + the branch/leaf
 * DIRECT_DILATION Dilation=4/1 overrides — exactly what
 * LoadSettingsBetweenSessions' first-run branch and
 * SettingsControl::on_reset_button_clicked apply (mainscreen.cpp:4836).*/
struct FirstRunManagers {
    FirstRunManagers() :
        trunk(Stage::Trunk), branch(Stage::Branch), leaf(Stage::Leaf) {
        branch.getCostFunctionClass("DIRECT_DILATION")
            ->setIntParameterValue("Dilation", 4);
        leaf.getCostFunctionClass("DIRECT_DILATION")
            ->setIntParameterValue("Dilation", 1);
    }
    jta_cost_function::CostFunctionManager trunk;
    jta_cost_function::CostFunctionManager branch;
    jta_cost_function::CostFunctionManager leaf;
};

} // namespace

/*---------------------------------------------------------------------------*/

TEST_CASE(
    "cost_function_registry: first-run manager configuration produces the "
    "exact golden 51-entry table",
    "[cost_function_registry]") {
    /*(a) The widgets first-run configuration -> the exact pre-extraction
     * widgets output: 51 entries (17 per stage), ACTIVE_CF first, then per
     * available cost function the double/int/bool parameter groups in
     * listCostFunctions order; doubles bit-exact (%.17g round-trip).*/
    FirstRunManagers managers;
    const std::vector<jta::RegistryEntry> entries =
        jta::BuildCostFunctionRegistryEntries(
            managers.trunk, managers.branch, managers.leaf);

    REQUIRE(entries.size() == kGoldenEntryCount);
    for (size_t i = 0; i < kGoldenEntryCount; ++i) {
        requireEntryMatches(entries[i], kGoldenCostFunctionEntries[i]);
    }

    /*Per-stage structure: 17 entries each, ACTIVE_CF first (the golden table
     * order is the pinned contract — a reorder would change registry keys).*/
    REQUIRE(
        entries[0].key == QStringLiteral("TRUNK@ACTIVE_CF"));
    REQUIRE(
        entries[17].key == QStringLiteral("BRANCH@ACTIVE_CF"));
    REQUIRE(
        entries[34].key == QStringLiteral("LEAF@ACTIVE_CF"));
}

TEST_CASE(
    "cost_function_registry: zero-stage managers are tolerated (empty "
    "configuration robustness)",
    "[cost_function_registry]") {
    /*(c) The closest constructible "empty" configuration: default-constructed
     * (no Stage argument) managers. The mapping never reads stage/GPU state —
     * it only walks getActiveCostFunction/getAvailableCostFunctions — so the
     * output stays well-defined and crash-free; a manager with an empty
     * available-cost-functions list would contribute only its ACTIVE_CF entry
     * (the parameter loops skip), which is not directly constructible via the
     * public CostFunctionManager API.*/
    jta_cost_function::CostFunctionManager trunk;
    jta_cost_function::CostFunctionManager branch;
    jta_cost_function::CostFunctionManager leaf;
    const std::vector<jta::RegistryEntry> entries =
        jta::BuildCostFunctionRegistryEntries(trunk, branch, leaf);

    REQUIRE(entries.size() == kGoldenEntryCount);
    REQUIRE(
        entries[0].key == QStringLiteral("TRUNK@ACTIVE_CF"));
    REQUIRE(
        entries[0].value.toString() == QStringLiteral("DIRECT_DILATION"));
    REQUIRE(
        entries[17].key == QStringLiteral("BRANCH@ACTIVE_CF"));
    REQUIRE(
        entries[34].key == QStringLiteral("LEAF@ACTIVE_CF"));

    /*Deterministic: repeated calls over the same managers produce the same
     * table (the mapping is a pure read of manager state).*/
    const std::vector<jta::RegistryEntry> again =
        jta::BuildCostFunctionRegistryEntries(trunk, branch, leaf);
    REQUIRE(again.size() == entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        REQUIRE(again[i].key == entries[i].key);
        REQUIRE(again[i].value == entries[i].value);
    }
}

TEST_CASE(
    "cost_function_registry: registry write/read-back lands the identical "
    "golden entry set (widgets save path)",
    "[cost_function_registry]") {
    /*(d) The widgets onSaveSettings path without MainScreen: managers ->
     * shared mapping -> SaveCostFunctionSettings -> LoadSettings. Every golden
     * entry must be present on disk with the identical value (the registry
     * contract both apps share).*/
    TempSettings tmp;
    jta::SettingsService svc(tmp.path, QSettings::IniFormat);
    FirstRunManagers managers;
    const std::vector<jta::RegistryEntry> entries =
        jta::BuildCostFunctionRegistryEntries(
            managers.trunk, managers.branch, managers.leaf);

    svc.SaveCostFunctionSettings(entries);
    const jta::SettingsService::LoadResult result = svc.LoadSettings();

    std::map<QString, QVariant> saved;
    for (const jta::RegistryEntry& entry : result.cost_function_entries) {
        saved[entry.key] = entry.value;
    }
    REQUIRE(saved.size() == kGoldenEntryCount);
    for (const GoldenEntry& golden : kGoldenCostFunctionEntries) {
        const auto it = saved.find(QString::fromLatin1(golden.key));
        REQUIRE(it != saved.end());
        requireEntryMatches(jta::RegistryEntry{it->first, it->second}, golden);
    }
}
