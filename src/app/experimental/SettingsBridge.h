// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U5: SettingsBridge — the thin settings adapter (R4, R5, R10, R17).
// Pass-through orchestration only (thinness rule): the bridge holds the
// session-local editor state — one OptimizerSettings + three per-stage
// CostFunctionManager(Stage) instances (the experiment knobs: per-stage
// ranges/budgets/dilation, branch count, enable toggles, and the per-stage
// cost-variant choice) — and delegates every semantic to the seams:
// OptimizerSettings/settings_constants.h for the defaults,
// CostFunctionManager for the cost-variant surface, SettingsService for the
// registry.
//
// Session-local edits with explicit save (review fix): property setters only
// mutate the in-memory objects and mark the session dirty; nothing touches
// the registry until Save(). Reset() restores the settings_constants.h
// defaults (mirroring the widgets SettingsControl reset: fresh managers +
// branch/leaf DIRECT_DILATION Dilation=4/1 overrides). Load() restores the
// persisted state (widgets LoadSettingsBetweenSessions semantics; malformed
// keys are skipped instead of the widgets error dialogs); on first run it
// applies defaults and does NOT write the registry (explicit-save contract —
// MarkFirstTimeDone stays gated on a future CUDA probe).
//
// Registry mapping (parity contract, plan C4): Save() writes the
// CostFunctionSettings entries produced by buildCostFunctionRegistryEntries()
// — a DELEGATE to the shared jta::BuildCostFunctionRegistryEntries
// (plan 006 U1: the widgets MainScreen calls the SAME shared function —
// there is no replication). Keys (STAGE@ACTIVE_CF /
// STAGE@CFname@ParamName@TYPE with DOUBLE/INT/BOOL type suffixes) and
// values (lossless doubles — the truncation-bug lesson) MUST stay
// identical to the widgets output or the two apps fight over the registry;
// the parity pin lives in test/unit/experimental_settings_test.cpp.
//
// Dilation semantics (widgets parity): per-stage dilation is the "Dilation"
// int parameter of the stage's ACTIVE cost function. The getters read the
// active CF's parameter (falling back to the per-stage default constant when
// the active CF has none — e.g. DIRECT_MAHFOUZ); the setters write the active
// CF and no-op when it has no Dilation parameter (the hasDilation properties
// let the panel disable the field, mirroring the widgets parameter list).

#pragma once

#include <QObject>
#include <QStringList>
#include <memory>
#include <vector>

#include "services/optimizer_settings.h"
#include "services/settings_service.h"

namespace jta_cost_function {
class CostFunctionManager;
}

class SettingsBridge : public QObject {
    Q_OBJECT

    // ---- Session editor state (QML form fields) --------------------------
    // Per-stage ranges: 6 components each (translations x/y/z, rotations
    // xa/ya/za). All setters mark the session dirty.
    Q_PROPERTY(double trunkRangeX READ trunkRangeX WRITE setTrunkRangeX NOTIFY
                   settingsEdited)
    Q_PROPERTY(double trunkRangeY READ trunkRangeY WRITE setTrunkRangeY NOTIFY
                   settingsEdited)
    Q_PROPERTY(double trunkRangeZ READ trunkRangeZ WRITE setTrunkRangeZ NOTIFY
                   settingsEdited)
    Q_PROPERTY(double trunkRangeXA READ trunkRangeXA WRITE setTrunkRangeXA
                   NOTIFY settingsEdited)
    Q_PROPERTY(double trunkRangeYA READ trunkRangeYA WRITE setTrunkRangeYA
                   NOTIFY settingsEdited)
    Q_PROPERTY(double trunkRangeZA READ trunkRangeZA WRITE setTrunkRangeZA
                   NOTIFY settingsEdited)
    Q_PROPERTY(int trunkBudget READ trunkBudget WRITE setTrunkBudget NOTIFY
                   settingsEdited)

    Q_PROPERTY(double branchRangeX READ branchRangeX WRITE setBranchRangeX
                   NOTIFY settingsEdited)
    Q_PROPERTY(double branchRangeY READ branchRangeY WRITE setBranchRangeY
                   NOTIFY settingsEdited)
    Q_PROPERTY(double branchRangeZ READ branchRangeZ WRITE setBranchRangeZ
                   NOTIFY settingsEdited)
    Q_PROPERTY(double branchRangeXA READ branchRangeXA WRITE setBranchRangeXA
                   NOTIFY settingsEdited)
    Q_PROPERTY(double branchRangeYA READ branchRangeYA WRITE setBranchRangeYA
                   NOTIFY settingsEdited)
    Q_PROPERTY(double branchRangeZA READ branchRangeZA WRITE setBranchRangeZA
                   NOTIFY settingsEdited)
    Q_PROPERTY(int branchBudget READ branchBudget WRITE setBranchBudget NOTIFY
                   settingsEdited)
    Q_PROPERTY(int numberBranches READ numberBranches WRITE setNumberBranches
                   NOTIFY settingsEdited)
    Q_PROPERTY(bool enableBranch READ enableBranch WRITE setEnableBranch NOTIFY
                   settingsEdited)

    Q_PROPERTY(double leafRangeX READ leafRangeX WRITE setLeafRangeX NOTIFY
                   settingsEdited)
    Q_PROPERTY(double leafRangeY READ leafRangeY WRITE setLeafRangeY NOTIFY
                   settingsEdited)
    Q_PROPERTY(double leafRangeZ READ leafRangeZ WRITE setLeafRangeZ NOTIFY
                   settingsEdited)
    Q_PROPERTY(double leafRangeXA READ leafRangeXA WRITE setLeafRangeXA NOTIFY
                   settingsEdited)
    Q_PROPERTY(double leafRangeYA READ leafRangeYA WRITE setLeafRangeYA NOTIFY
                   settingsEdited)
    Q_PROPERTY(double leafRangeZA READ leafRangeZA WRITE setLeafRangeZA NOTIFY
                   settingsEdited)
    Q_PROPERTY(int leafBudget READ leafBudget WRITE setLeafBudget NOTIFY
                   settingsEdited)
    Q_PROPERTY(bool enableLeaf READ enableLeaf WRITE setEnableLeaf NOTIFY
                   settingsEdited)

    // ---- Per-stage cost-variant choice -----------------------------------
    Q_PROPERTY(QStringList trunkCostFunctions READ trunkCostFunctions CONSTANT)
    Q_PROPERTY(
        QStringList branchCostFunctions READ branchCostFunctions CONSTANT)
    Q_PROPERTY(QStringList leafCostFunctions READ leafCostFunctions CONSTANT)
    Q_PROPERTY(int trunkCostFunctionIndex READ trunkCostFunctionIndex WRITE
                   setTrunkCostFunctionIndex NOTIFY settingsEdited)
    Q_PROPERTY(int branchCostFunctionIndex READ branchCostFunctionIndex WRITE
                   setBranchCostFunctionIndex NOTIFY settingsEdited)
    Q_PROPERTY(int leafCostFunctionIndex READ leafCostFunctionIndex WRITE
                   setLeafCostFunctionIndex NOTIFY settingsEdited)

    // ---- Per-stage dilation (active cost function's "Dilation" int param) -
    Q_PROPERTY(int trunkDilation READ trunkDilation WRITE setTrunkDilation
                   NOTIFY settingsEdited)
    Q_PROPERTY(int branchDilation READ branchDilation WRITE setBranchDilation
                   NOTIFY settingsEdited)
    Q_PROPERTY(int leafDilation READ leafDilation WRITE setLeafDilation NOTIFY
                   settingsEdited)
    Q_PROPERTY(
        bool trunkHasDilation READ trunkHasDilation NOTIFY settingsEdited)
    Q_PROPERTY(
        bool branchHasDilation READ branchHasDilation NOTIFY settingsEdited)
    Q_PROPERTY(bool leafHasDilation READ leafHasDilation NOTIFY settingsEdited)

    // ---- Session state ----------------------------------------------------
    Q_PROPERTY(bool dirty READ dirty NOTIFY dirtyChanged)

public:
    /*The SettingsService is injected (the hub owns the default real-registry
     * instance; tests pass an ini-backed one — the real registry is never
     * touched by tests). Not owned by the bridge.*/
    explicit SettingsBridge(
        jta::SettingsService* settings_service,
        QObject* parent = nullptr);
    ~SettingsBridge() override;

    // ---- Session actions (QML buttons) -----------------------------------
    Q_INVOKABLE void save();   // explicit save: registry write, clears dirty
    Q_INVOKABLE void load();   // restore persisted state (startup); defaults
                               // on first run; never writes
    Q_INVOKABLE void reset();  // settings_constants.h defaults + fresh
                               // managers (widgets SettingsControl parity);
                               // marks dirty (a session edit until Save)

    // ---- Range/budget/number/enable accessors ----------------------------
    double trunkRangeX() const;
    double trunkRangeY() const;
    double trunkRangeZ() const;
    double trunkRangeXA() const;
    double trunkRangeYA() const;
    double trunkRangeZA() const;
    int trunkBudget() const;
    void setTrunkRangeX(double v);
    void setTrunkRangeY(double v);
    void setTrunkRangeZ(double v);
    void setTrunkRangeXA(double v);
    void setTrunkRangeYA(double v);
    void setTrunkRangeZA(double v);
    void setTrunkBudget(int v);

    double branchRangeX() const;
    double branchRangeY() const;
    double branchRangeZ() const;
    double branchRangeXA() const;
    double branchRangeYA() const;
    double branchRangeZA() const;
    int branchBudget() const;
    int numberBranches() const;
    bool enableBranch() const;
    void setBranchRangeX(double v);
    void setBranchRangeY(double v);
    void setBranchRangeZ(double v);
    void setBranchRangeXA(double v);
    void setBranchRangeYA(double v);
    void setBranchRangeZA(double v);
    void setBranchBudget(int v);
    void setNumberBranches(int v);
    void setEnableBranch(bool v);

    double leafRangeX() const;
    double leafRangeY() const;
    double leafRangeZ() const;
    double leafRangeXA() const;
    double leafRangeYA() const;
    double leafRangeZA() const;
    int leafBudget() const;
    bool enableLeaf() const;
    void setLeafRangeX(double v);
    void setLeafRangeY(double v);
    void setLeafRangeZ(double v);
    void setLeafRangeXA(double v);
    void setLeafRangeYA(double v);
    void setLeafRangeZA(double v);
    void setLeafBudget(int v);
    void setEnableLeaf(bool v);

    // ---- Cost-variant accessors -------------------------------------------
    QStringList trunkCostFunctions() const;
    QStringList branchCostFunctions() const;
    QStringList leafCostFunctions() const;
    int trunkCostFunctionIndex() const;
    int branchCostFunctionIndex() const;
    int leafCostFunctionIndex() const;
    void setTrunkCostFunctionIndex(int index);
    void setBranchCostFunctionIndex(int index);
    void setLeafCostFunctionIndex(int index);

    // ---- Dilation accessors -----------------------------------------------
    int trunkDilation() const;
    int branchDilation() const;
    int leafDilation() const;
    void setTrunkDilation(int v);
    void setBranchDilation(int v);
    void setLeafDilation(int v);
    bool trunkHasDilation() const;
    bool branchHasDilation() const;
    bool leafHasDilation() const;

    // ---- C++ surface (tests + U6 optimizer wiring) ------------------------
    bool dirty() const;
    OptimizerSettings& optimizerSettings();
    jta_cost_function::CostFunctionManager* trunkManager() const;
    jta_cost_function::CostFunctionManager* branchManager() const;
    jta_cost_function::CostFunctionManager* leafManager() const;

    /*The shared registry mapping (parity pin, see file header): raw
     * CostFunctionSettings entries for the current manager state, in the
     * exact widgets order (ACTIVE_CF first, then per available cost function
     * the double/int/bool parameter groups) — a thin wrapper over
     * jta::BuildCostFunctionRegistryEntries (plan 006 U1), the SAME shared
     * function the widgets MainScreen calls.*/
    std::vector<jta::RegistryEntry> buildCostFunctionRegistryEntries() const;

signals:
    void settingsEdited();
    void dirtyChanged();

private:
    void markDirty();
    void applyCostFunctionEntries(
        const std::vector<jta::RegistryEntry>& entries);
    jta_cost_function::CostFunctionManager* managerForStage(
        const QString& stage) const;
    int costFunctionIndex(
        jta_cost_function::CostFunctionManager* manager) const;
    void setCostFunctionIndex(
        jta_cost_function::CostFunctionManager* manager,
        int index);
    QStringList costFunctionNames(
        jta_cost_function::CostFunctionManager* manager) const;
    int dilation(
        jta_cost_function::CostFunctionManager* manager,
        int fallback_default) const;
    void setDilation(jta_cost_function::CostFunctionManager* manager, int v);
    bool hasDilation(jta_cost_function::CostFunctionManager* manager) const;

    jta::SettingsService* settings_service_ = nullptr;
    OptimizerSettings optimizer_;
    std::unique_ptr<jta_cost_function::CostFunctionManager> trunk_manager_;
    std::unique_ptr<jta_cost_function::CostFunctionManager> branch_manager_;
    std::unique_ptr<jta_cost_function::CostFunctionManager> leaf_manager_;
    jta::EdgeSettings edge_;
    bool dirty_ = false;
};
