// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 005 U5: SettingsBridge + replicated registry mapping pins (R4, R5,
// R10, R17). Deterministic Catch2 headless tests that direct-compile the
// bridge (SettingsBridge) against the real SettingsService
// (settings_service.cpp + optimizer_settings.cpp) with QSettings redirected
// to a temp ini (the real registry is never touched) and link the real
// CostFunctionManager/CostFunction from jtml_compute (the constructor +
// parameter surface exercised here are CPU-only; no CUDA calls at runtime).
//
// Pins (plan 005 U5 test scenarios):
//  (a) happy: setting a cost variant per stage reflects in the managers
//      (getActiveCostFunction);
//  (b) happy: save -> load round-trip preserves every value (registry parity
//      contract: CostFunctionSettings/OptimizerSettings/EdgeDetectionSettings
//      groups);
//  (c) edge: fractional cost parameters round-trip bit-exact (double, no int
//      narrowing — the pinned invariant from the Parameter<double>
//      truncation bug);
//  (d) integration: the shared mapping (jta::BuildCostFunctionRegistryEntries
//      — the ONE mapping both front-ends call, plan 006 U1) produces
//      IDENTICAL registry entries to the pre-extraction widgets reference for
//      the same manager configuration (parity pin vs the golden fixture
//      below).
//
// Golden fixture capture (documented procedure, plan 005 U5 "Parity pin
// mechanism"): the widgets reference
// MainScreen::BuildCostFunctionRegistryEntries (mainscreen.cpp:4895) is a
// private jtml_view member — a live call from a headless test is impossible.
// The fixture below was captured on 2026-08-11 by a throwaway harness that
// called the REAL method (via the standard friend-injection template trick on a
// placement-new'd MainScreen — the mapping is const and never reads `this`),
// linked against libjtml_view.a + libjtml_compute.so + the widgets link set,
// for the widgets FIRST-RUN manager configuration: three fresh
// CostFunctionManager(Stage) instances plus the branch/leaf DIRECT_DILATION
// Dilation=4/1 overrides that LoadSettingsBetweenSessions' first-run branch and
// SettingsControl::on_reset_button_clicked apply (mainscreen.cpp:4836).
// The harness dumped the entries as (key, type, value) literals; the table
// below is verbatim (doubles at %.17g — exact IEEE-754 round-trip). The
// bridge's reset() reproduces that exact configuration, so test (d) compares
// the shared mapping's output against the captured widgets output for the
// same input. Since plan 006 U1 the mapping lives in jtml_services
// (cost_function_registry.cpp) and test (d) calls it directly instead of the
// bridge replication — the 51 VALUES are unchanged (verbatim relocation, R13).
// Regenerate the fixture after any cost-function parameter list or mapping
// change and re-run the pin.

#include <catch2/catch_test_macros.hpp>

#include <QMetaType>
#include <QSettings>
#include <QString>
#include <QStringList>
#include <QTemporaryDir>
#include <QVariant>

#include <cstring>
#include <map>

#include "SettingsBridge.h"
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

/*Verbatim capture of the widgets reference output for the first-run manager
 * configuration (51 entries: 17 per stage — ACTIVE_CF, then per available
 * cost function the double/int/bool parameter groups, in listCostFunctions
 * order). See the capture note above.*/
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
 * DIRECT_DILATION Dilation=4/1 overrides — exactly what the bridge's reset()
 * reproduces.*/
void requireWidgetsFirstRunConfiguration(SettingsBridge* bridge) {
    bridge->reset();
}

} // namespace

/*---------------------------------------------------------------------------*/

TEST_CASE(
    "settings_bridge: cost-variant selection per stage reflects in the "
    "managers",
    "[settings_bridge]") {
    TempSettings tmp;
    jta::SettingsService svc(tmp.path, QSettings::IniFormat);
    SettingsBridge bridge(&svc);

    /*Available cost functions, in listCostFunctions order (7 per stage).*/
    const QStringList expected_names = {
        "sym_trap_function",
        "DD_NEW_POLE_CONSTRAINT",
        "DIRECT_DILATION_POLE_CONSTRAINT",
        "DIRECT_DILATION_SAME_Z",
        "DIRECT_DILATION_T1",
        "DIRECT_DILATION",
        "DIRECT_MAHFOUZ",
    };
    REQUIRE(bridge.trunkCostFunctions() == expected_names);
    REQUIRE(bridge.branchCostFunctions() == expected_names);
    REQUIRE(bridge.leafCostFunctions() == expected_names);

    /*Default active variant is DIRECT_DILATION (index 5).*/
    REQUIRE(bridge.trunkCostFunctionIndex() == 5);
    REQUIRE(bridge.branchCostFunctionIndex() == 5);
    REQUIRE(bridge.leafCostFunctionIndex() == 5);
    REQUIRE(
        bridge.trunkManager()->getActiveCostFunction() == "DIRECT_DILATION");
    REQUIRE_FALSE(bridge.dirty());

    /*Per-stage selection reflects in the real managers.*/
    bridge.setTrunkCostFunctionIndex(0);  // sym_trap_function
    bridge.setBranchCostFunctionIndex(4); // DIRECT_DILATION_T1
    bridge.setLeafCostFunctionIndex(6);   // DIRECT_MAHFOUZ
    REQUIRE(
        bridge.trunkManager()->getActiveCostFunction() == "sym_trap_function");
    REQUIRE(
        bridge.branchManager()->getActiveCostFunction() ==
        "DIRECT_DILATION_T1");
    REQUIRE(bridge.leafManager()->getActiveCostFunction() == "DIRECT_MAHFOUZ");
    REQUIRE(bridge.trunkCostFunctionIndex() == 0);
    REQUIRE(bridge.branchCostFunctionIndex() == 4);
    REQUIRE(bridge.leafCostFunctionIndex() == 6);
    REQUIRE(bridge.dirty());

    /*Out-of-range indexes are ignored (no manager change, no extra dirty
     * transition).*/
    bridge.setTrunkCostFunctionIndex(99);
    bridge.setTrunkCostFunctionIndex(-1);
    REQUIRE(bridge.trunkCostFunctionIndex() == 0);

    /*Dilation follows the ACTIVE cost function's "Dilation" parameter;
     * variants without one (DIRECT_MAHFOUZ) expose hasDilation=false and
     * no-op setters with the per-stage default fallback.*/
    REQUIRE(bridge.trunkDilation() == 3);    // sym_trap_function default
    REQUIRE(bridge.branchDilation() == 6);   // DIRECT_DILATION_T1 default
    REQUIRE_FALSE(bridge.leafHasDilation()); // DIRECT_MAHFOUZ
    REQUIRE(bridge.leafDilation() == Z_SEARCH_DILATION);
    bridge.setLeafDilation(9); // no-op: no Dilation parameter
    REQUIRE(bridge.leafDilation() == Z_SEARCH_DILATION);

    bridge.setTrunkDilation(8);
    REQUIRE(
        bridge.trunkManager()
            ->getActiveCostFunctionClass()
            ->getIntParameters()[0]
            .getParameterValue() == 8);
}

TEST_CASE(
    "settings_bridge: save -> load round-trip preserves every value",
    "[settings_bridge]") {
    TempSettings tmp;
    jta::SettingsService svc(tmp.path, QSettings::IniFormat);
    SettingsBridge bridge(&svc);

    /*Touch every editable field with distinct values (fractional ranges
     * included — the range fields are doubles end to end).*/
    bridge.setTrunkRangeX(12.5);
    bridge.setTrunkRangeY(-3.25);
    bridge.setTrunkRangeZ(40);
    bridge.setTrunkRangeXA(7.75);
    bridge.setTrunkRangeYA(0.5);
    bridge.setTrunkRangeZA(-15);
    bridge.setTrunkBudget(12345);
    bridge.setBranchRangeX(9.5);
    bridge.setBranchRangeY(8);
    bridge.setBranchRangeZ(7.25);
    bridge.setBranchRangeXA(6);
    bridge.setBranchRangeYA(5.5);
    bridge.setBranchRangeZA(4);
    bridge.setBranchBudget(4321);
    bridge.setNumberBranches(3);
    bridge.setEnableBranch(false);
    bridge.setLeafRangeX(1.125);
    bridge.setLeafRangeY(2.25);
    bridge.setLeafRangeZ(3.375);
    bridge.setLeafRangeXA(4.5);
    bridge.setLeafRangeYA(5.625);
    bridge.setLeafRangeZA(6.75);
    bridge.setLeafBudget(999);
    bridge.setEnableLeaf(false);
    bridge.setTrunkCostFunctionIndex(1);  // DD_NEW_POLE_CONSTRAINT
    bridge.setBranchCostFunctionIndex(3); // DIRECT_DILATION_SAME_Z
    bridge.setLeafCostFunctionIndex(6);   // DIRECT_MAHFOUZ
    bridge.setTrunkDilation(9);
    bridge.setBranchDilation(2);
    bridge.setLeafDilation(5); // no-op (DIRECT_MAHFOUZ has no Dilation)

    REQUIRE(bridge.dirty());
    bridge.save();
    REQUIRE_FALSE(bridge.dirty());

    /*Registry parity contract: the three settings groups exist.*/
    QSettings raw(tmp.path, QSettings::IniFormat);
    const QStringList groups = raw.childGroups();
    REQUIRE(groups.contains("CostFunctionSettings"));
    REQUIRE(groups.contains("OptimizerSettings"));
    REQUIRE(groups.contains("EdgeDetectionSettings"));

    /*A fresh bridge over the same registry restores every value.*/
    SettingsBridge loaded(&svc);
    REQUIRE_FALSE(loaded.dirty());
    loaded.load();
    REQUIRE_FALSE(loaded.dirty());

    REQUIRE(loaded.trunkRangeX() == 12.5);
    REQUIRE(loaded.trunkRangeY() == -3.25);
    REQUIRE(loaded.trunkRangeZ() == 40);
    REQUIRE(loaded.trunkRangeXA() == 7.75);
    REQUIRE(loaded.trunkRangeYA() == 0.5);
    REQUIRE(loaded.trunkRangeZA() == -15);
    REQUIRE(loaded.trunkBudget() == 12345);
    REQUIRE(loaded.branchRangeX() == 9.5);
    REQUIRE(loaded.branchRangeY() == 8);
    REQUIRE(loaded.branchRangeZ() == 7.25);
    REQUIRE(loaded.branchRangeXA() == 6);
    REQUIRE(loaded.branchRangeYA() == 5.5);
    REQUIRE(loaded.branchRangeZA() == 4);
    REQUIRE(loaded.branchBudget() == 4321);
    REQUIRE(loaded.numberBranches() == 3);
    REQUIRE_FALSE(loaded.enableBranch());
    REQUIRE(loaded.leafRangeX() == 1.125);
    REQUIRE(loaded.leafRangeY() == 2.25);
    REQUIRE(loaded.leafRangeZ() == 3.375);
    REQUIRE(loaded.leafRangeXA() == 4.5);
    REQUIRE(loaded.leafRangeYA() == 5.625);
    REQUIRE(loaded.leafRangeZA() == 6.75);
    REQUIRE(loaded.leafBudget() == 999);
    REQUIRE_FALSE(loaded.enableLeaf());
    REQUIRE(loaded.trunkCostFunctionIndex() == 1);
    REQUIRE(loaded.branchCostFunctionIndex() == 3);
    REQUIRE(loaded.leafCostFunctionIndex() == 6);
    REQUIRE(
        loaded.trunkManager()->getActiveCostFunction() ==
        "DD_NEW_POLE_CONSTRAINT");
    REQUIRE(loaded.trunkDilation() == 9);
    REQUIRE(loaded.branchDilation() == 2);
    REQUIRE(loaded.leafDilation() == Z_SEARCH_DILATION);
    REQUIRE_FALSE(loaded.leafHasDilation());

    /*The exact persisted optimizer values (service contract, same keys as
     * the widgets first-run save).*/
    const jta::SettingsService::LoadResult result = svc.LoadSettings();
    REQUIRE(result.optimizer.trunk_budget == 12345);
    REQUIRE(result.optimizer.number_branches == 3);
    REQUIRE_FALSE(result.optimizer.enable_branch_);
    REQUIRE(result.optimizer.leaf_range.z == 3.375);
    REQUIRE(result.edge.aperture == APERTURE);
    REQUIRE(result.edge.low_thresh == LOW_THRESH);
    REQUIRE(result.edge.high_thresh == HIGH_THRESH);
}

TEST_CASE(
    "settings_bridge: fractional cost parameters round-trip bit-exact (no "
    "int narrowing)",
    "[settings_bridge]") {
    TempSettings tmp;
    jta::SettingsService svc(tmp.path, QSettings::IniFormat);
    SettingsBridge bridge(&svc);

    /*Fractional + negative + >6-significant-digit doubles on cost-function
     * parameters (the truncation-bug pin: Parameter<double> was once an int
     * member — a narrowing mapping would land 75.5 on 75.0). Non-active and
     * non-DIRECT_DILATION functions included: the mapping walks ALL
     * available cost functions.*/
    const double pole_weight = 75.123456789012345;
    const double vv_weight = -0.25;
    const double z_weight = 0.1 + 0.2; // 0.30000000000000004
    REQUIRE(bridge.trunkManager()
                ->getCostFunctionClass("sym_trap_function")
                ->setDoubleParameterValue("PoleWeight", pole_weight));
    REQUIRE(bridge.trunkManager()
                ->getCostFunctionClass("sym_trap_function")
                ->setDoubleParameterValue("VVWeight", vv_weight));
    REQUIRE(bridge.branchManager()
                ->getCostFunctionClass("DIRECT_DILATION_SAME_Z")
                ->setDoubleParameterValue("Z_Weight", z_weight));
    bridge.save();

    SettingsBridge loaded(&svc);
    loaded.load();

    double out = 0;
    REQUIRE(loaded.trunkManager()
                ->getCostFunctionClass("sym_trap_function")
                ->getDoubleParameterValue("PoleWeight", out));
    REQUIRE(sameBits(pole_weight, out));
    REQUIRE(loaded.trunkManager()
                ->getCostFunctionClass("sym_trap_function")
                ->getDoubleParameterValue("VVWeight", out));
    REQUIRE(sameBits(vv_weight, out));
    REQUIRE(loaded.branchManager()
                ->getCostFunctionClass("DIRECT_DILATION_SAME_Z")
                ->getDoubleParameterValue("Z_Weight", out));
    REQUIRE(sameBits(z_weight, out));

    /*And the same values survive a second save->load cycle (idempotent
     * registry round-trip).*/
    loaded.save();
    SettingsBridge reloaded(&svc);
    reloaded.load();
    REQUIRE(reloaded.trunkManager()
                ->getCostFunctionClass("sym_trap_function")
                ->getDoubleParameterValue("PoleWeight", out));
    REQUIRE(sameBits(pole_weight, out));
}

TEST_CASE(
    "settings_bridge: shared registry mapping matches the golden fixture "
    "(parity pin)",
    "[settings_bridge]") {
    TempSettings tmp;
    jta::SettingsService svc(tmp.path, QSettings::IniFormat);
    SettingsBridge bridge(&svc);
    requireWidgetsFirstRunConfiguration(&bridge);

    /*The shared mapping (jta::BuildCostFunctionRegistryEntries — the one
     * function both front-ends call, plan 006 U1) produces exactly the
     * pre-extraction widgets entries, in the widgets order (ACTIVE_CF first,
     * then per cost function the double/int/bool parameter groups).*/
    const std::vector<jta::RegistryEntry> entries =
        jta::BuildCostFunctionRegistryEntries(
            *bridge.trunkManager(),
            *bridge.branchManager(),
            *bridge.leafManager());
    REQUIRE(entries.size() == kGoldenEntryCount);
    for (size_t i = 0; i < kGoldenEntryCount; ++i) {
        requireEntryMatches(entries[i], kGoldenCostFunctionEntries[i]);
    }

    /*The full widgets save path (mapping -> SettingsService -> registry ->
     * LoadSettings) lands the identical entry set on disk: every golden
     * entry is present with the identical value after a save.*/
    bridge.save();
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

TEST_CASE(
    "settings_bridge: first run and reset semantics", "[settings_bridge]") {
    TempSettings tmp;
    jta::SettingsService svc(tmp.path, QSettings::IniFormat);

    /*First run: load() applies defaults and NEVER writes the registry (the
     * session stays clean until an explicit Save).*/
    {
        SettingsBridge bridge(&svc);
        bridge.load();
        REQUIRE(bridge.trunkBudget() == TRUNK_BUDGET);
        REQUIRE(bridge.trunkRangeX() == TRUNK_RANGE.x);
        REQUIRE(bridge.branchDilation() == 4); // widgets first-run override
        REQUIRE(bridge.leafDilation() == Z_SEARCH_DILATION);
        REQUIRE_FALSE(bridge.dirty());
        REQUIRE(svc.IsFirstTime()); // nothing written by load()
    }

    /*Reset: settings_constants.h defaults + fresh managers (the widgets
     * SettingsControl parity), and the session is dirty until Save.*/
    {
        SettingsBridge bridge(&svc);
        bridge.setTrunkBudget(1);
        bridge.setTrunkCostFunctionIndex(0);
        bridge.save();
        REQUIRE_FALSE(bridge.dirty());

        bridge.reset();
        REQUIRE(bridge.dirty()); // a session edit until Save
        REQUIRE(bridge.trunkBudget() == TRUNK_BUDGET);
        REQUIRE(bridge.trunkRangeX() == TRUNK_RANGE.x);
        REQUIRE(bridge.trunkRangeY() == TRUNK_RANGE.y);
        REQUIRE(bridge.trunkRangeZ() == TRUNK_RANGE.z);
        REQUIRE(bridge.trunkRangeXA() == TRUNK_RANGE.xa);
        REQUIRE(bridge.trunkRangeYA() == TRUNK_RANGE.ya);
        REQUIRE(bridge.trunkRangeZA() == TRUNK_RANGE.za);
        REQUIRE(bridge.branchRangeX() == BRANCH_RANGE.x);
        REQUIRE(bridge.branchBudget() == BRANCH_BUDGET);
        REQUIRE(bridge.numberBranches() == NUMBER_BRANCHES);
        REQUIRE(bridge.enableBranch() == ENABLE_BRANCH);
        REQUIRE(bridge.leafRangeX() == Z_SEARCH_RANGE.x);
        REQUIRE(bridge.leafBudget() == Z_SEARCH_BUDGET);
        REQUIRE(bridge.enableLeaf() == ENABLE_Z);
        REQUIRE(bridge.trunkCostFunctionIndex() == 5); // DIRECT_DILATION
        REQUIRE(bridge.trunkDilation() == TRUNK_DILATION);
        REQUIRE(bridge.branchDilation() == 4);
        REQUIRE(bridge.leafDilation() == Z_SEARCH_DILATION);
        REQUIRE(
            bridge.trunkManager()->getActiveCostFunction() ==
            "DIRECT_DILATION");

        /*Save after reset persists the defaults.*/
        bridge.save();
        REQUIRE_FALSE(bridge.dirty());
        SettingsBridge loaded(&svc);
        loaded.load();
        REQUIRE(loaded.trunkBudget() == TRUNK_BUDGET);
        REQUIRE(loaded.trunkCostFunctionIndex() == 5);
    }
}
