// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "SettingsBridge.h"

#include <QStringList>

#include "compute/CostFunction.h"
#include "compute/CostFunctionManager.h"
#include "domain/settings_constants.h"
#include "services/cost_function_registry.h"

namespace {

/*Per-stage fallback dilation defaults (used only when the ACTIVE cost
 * function has no "Dilation" parameter, e.g. DIRECT_MAHFOUZ): the
 * settings_constants per-stage defaults. Branch has no dedicated constant —
 * 4 is the widgets branch value (LoadSettingsBetweenSessions first-run /
 * SettingsControl reset, mainscreen.cpp:4836).*/
constexpr int kBranchDilationDefault = 4;

}  // namespace

SettingsBridge::SettingsBridge(
    jta::SettingsService* settings_service,
    QObject* parent) :
    QObject(parent), settings_service_(settings_service) {
    /*Defaults (settings_constants.h via the OptimizerSettings/EdgeSettings
     * ctors) + fresh managers; the widgets first-run branch/leaf dilation
     * overrides (mainscreen.cpp:4836).*/
    optimizer_ = OptimizerSettings();
    edge_ = jta::EdgeSettings();
    trunk_manager_ =
        std::make_unique<jta_cost_function::CostFunctionManager>(Stage::Trunk);
    branch_manager_ =
        std::make_unique<jta_cost_function::CostFunctionManager>(Stage::Branch);
    leaf_manager_ =
        std::make_unique<jta_cost_function::CostFunctionManager>(Stage::Leaf);
    branch_manager_->getCostFunctionClass("DIRECT_DILATION")
        ->setIntParameterValue("Dilation", 4);
    leaf_manager_->getCostFunctionClass("DIRECT_DILATION")
        ->setIntParameterValue("Dilation", 1);
}

SettingsBridge::~SettingsBridge() = default;

/*----------------------------------------------------------------------------
 * Session actions
 *----------------------------------------------------------------------------*/

void SettingsBridge::save() {
    /*Registry write (explicit save): the three settings groups the widgets
     * onSaveSettings + first-run write. The cost-function entries come from
     * the shared mapping (parity contract — the widgets MainScreen calls the
     * same services function, plan 006 U1).*/
    settings_service_->SaveCostFunctionSettings(
        buildCostFunctionRegistryEntries());
    settings_service_->SaveOptimizerSettings(optimizer_);
    settings_service_->SaveEdgeSettings(
        edge_.aperture, edge_.low_thresh, edge_.high_thresh);

    if (dirty_) {
        dirty_ = false;
        emit dirtyChanged();
    }
    emit settingsEdited();
}

void SettingsBridge::load() {
    jta::SettingsService::LoadResult result = settings_service_->LoadSettings();
    if (!result.first_time) {
        /*Widgets LoadSettingsBetweenSessions semantics (the non-first-time
         * branch): apply the raw cost-function entries, then the optimizer
         * and edge values. Malformed keys are skipped (the widgets shows
         * modal error dialogs; the QML app has no modal surface for these —
         * the registry is only written by the two apps' own mapping, so
         * malformed keys are not expected).*/
        applyCostFunctionEntries(result.cost_function_entries);
        optimizer_ = result.optimizer;
        edge_ = result.edge;
    }
    /*First run: keep the constructor defaults; do NOT write the registry
     * (session-local edits with explicit save; MarkFirstTimeDone stays gated
     * on a future CUDA probe).*/

    if (dirty_) {
        dirty_ = false;
        emit dirtyChanged();
    }
    emit settingsEdited();
}

void SettingsBridge::reset() {
    /*Widgets SettingsControl::on_reset_button_clicked parity: fresh
     * OptimizerSettings + fresh managers + the branch/leaf DIRECT_DILATION
     * dilation overrides (settings_control.cpp:1197). A reset is a session
     * edit until Save() — dirty stays true.*/
    optimizer_ = OptimizerSettings();
    edge_ = jta::EdgeSettings();
    trunk_manager_ =
        std::make_unique<jta_cost_function::CostFunctionManager>(Stage::Trunk);
    branch_manager_ =
        std::make_unique<jta_cost_function::CostFunctionManager>(Stage::Branch);
    leaf_manager_ =
        std::make_unique<jta_cost_function::CostFunctionManager>(Stage::Leaf);
    branch_manager_->getCostFunctionClass("DIRECT_DILATION")
        ->setIntParameterValue("Dilation", 4);
    leaf_manager_->getCostFunctionClass("DIRECT_DILATION")
        ->setIntParameterValue("Dilation", 1);

    markDirty();
}

/*----------------------------------------------------------------------------
 * Trunk
 *----------------------------------------------------------------------------*/

double SettingsBridge::trunkRangeX() const {
    return optimizer_.trunk_range.x;
}
double SettingsBridge::trunkRangeY() const {
    return optimizer_.trunk_range.y;
}
double SettingsBridge::trunkRangeZ() const {
    return optimizer_.trunk_range.z;
}
double SettingsBridge::trunkRangeXA() const {
    return optimizer_.trunk_range.xa;
}
double SettingsBridge::trunkRangeYA() const {
    return optimizer_.trunk_range.ya;
}
double SettingsBridge::trunkRangeZA() const {
    return optimizer_.trunk_range.za;
}
int SettingsBridge::trunkBudget() const {
    return optimizer_.trunk_budget;
}
void SettingsBridge::setTrunkRangeX(double v) {
    if (optimizer_.trunk_range.x == v) {
        return;
    }
    optimizer_.trunk_range.x = v;
    markDirty();
}
void SettingsBridge::setTrunkRangeY(double v) {
    if (optimizer_.trunk_range.y == v) {
        return;
    }
    optimizer_.trunk_range.y = v;
    markDirty();
}
void SettingsBridge::setTrunkRangeZ(double v) {
    if (optimizer_.trunk_range.z == v) {
        return;
    }
    optimizer_.trunk_range.z = v;
    markDirty();
}
void SettingsBridge::setTrunkRangeXA(double v) {
    if (optimizer_.trunk_range.xa == v) {
        return;
    }
    optimizer_.trunk_range.xa = v;
    markDirty();
}
void SettingsBridge::setTrunkRangeYA(double v) {
    if (optimizer_.trunk_range.ya == v) {
        return;
    }
    optimizer_.trunk_range.ya = v;
    markDirty();
}
void SettingsBridge::setTrunkRangeZA(double v) {
    if (optimizer_.trunk_range.za == v) {
        return;
    }
    optimizer_.trunk_range.za = v;
    markDirty();
}
void SettingsBridge::setTrunkBudget(int v) {
    if (optimizer_.trunk_budget == v) {
        return;
    }
    optimizer_.trunk_budget = v;
    markDirty();
}

/*----------------------------------------------------------------------------
 * Branch
 *----------------------------------------------------------------------------*/

double SettingsBridge::branchRangeX() const {
    return optimizer_.branch_range.x;
}
double SettingsBridge::branchRangeY() const {
    return optimizer_.branch_range.y;
}
double SettingsBridge::branchRangeZ() const {
    return optimizer_.branch_range.z;
}
double SettingsBridge::branchRangeXA() const {
    return optimizer_.branch_range.xa;
}
double SettingsBridge::branchRangeYA() const {
    return optimizer_.branch_range.ya;
}
double SettingsBridge::branchRangeZA() const {
    return optimizer_.branch_range.za;
}
int SettingsBridge::branchBudget() const {
    return optimizer_.branch_budget;
}
int SettingsBridge::numberBranches() const {
    return optimizer_.number_branches;
}
bool SettingsBridge::enableBranch() const {
    return optimizer_.enable_branch_;
}
void SettingsBridge::setBranchRangeX(double v) {
    if (optimizer_.branch_range.x == v) {
        return;
    }
    optimizer_.branch_range.x = v;
    markDirty();
}
void SettingsBridge::setBranchRangeY(double v) {
    if (optimizer_.branch_range.y == v) {
        return;
    }
    optimizer_.branch_range.y = v;
    markDirty();
}
void SettingsBridge::setBranchRangeZ(double v) {
    if (optimizer_.branch_range.z == v) {
        return;
    }
    optimizer_.branch_range.z = v;
    markDirty();
}
void SettingsBridge::setBranchRangeXA(double v) {
    if (optimizer_.branch_range.xa == v) {
        return;
    }
    optimizer_.branch_range.xa = v;
    markDirty();
}
void SettingsBridge::setBranchRangeYA(double v) {
    if (optimizer_.branch_range.ya == v) {
        return;
    }
    optimizer_.branch_range.ya = v;
    markDirty();
}
void SettingsBridge::setBranchRangeZA(double v) {
    if (optimizer_.branch_range.za == v) {
        return;
    }
    optimizer_.branch_range.za = v;
    markDirty();
}
void SettingsBridge::setBranchBudget(int v) {
    if (optimizer_.branch_budget == v) {
        return;
    }
    optimizer_.branch_budget = v;
    markDirty();
}
void SettingsBridge::setNumberBranches(int v) {
    if (optimizer_.number_branches == v) {
        return;
    }
    optimizer_.number_branches = v;
    markDirty();
}
void SettingsBridge::setEnableBranch(bool v) {
    if (optimizer_.enable_branch_ == v) {
        return;
    }
    optimizer_.enable_branch_ = v;
    markDirty();
}

/*----------------------------------------------------------------------------
 * Leaf
 *----------------------------------------------------------------------------*/

double SettingsBridge::leafRangeX() const {
    return optimizer_.leaf_range.x;
}
double SettingsBridge::leafRangeY() const {
    return optimizer_.leaf_range.y;
}
double SettingsBridge::leafRangeZ() const {
    return optimizer_.leaf_range.z;
}
double SettingsBridge::leafRangeXA() const {
    return optimizer_.leaf_range.xa;
}
double SettingsBridge::leafRangeYA() const {
    return optimizer_.leaf_range.ya;
}
double SettingsBridge::leafRangeZA() const {
    return optimizer_.leaf_range.za;
}
int SettingsBridge::leafBudget() const {
    return optimizer_.leaf_budget;
}
bool SettingsBridge::enableLeaf() const {
    return optimizer_.enable_leaf_;
}
void SettingsBridge::setLeafRangeX(double v) {
    if (optimizer_.leaf_range.x == v) {
        return;
    }
    optimizer_.leaf_range.x = v;
    markDirty();
}
void SettingsBridge::setLeafRangeY(double v) {
    if (optimizer_.leaf_range.y == v) {
        return;
    }
    optimizer_.leaf_range.y = v;
    markDirty();
}
void SettingsBridge::setLeafRangeZ(double v) {
    if (optimizer_.leaf_range.z == v) {
        return;
    }
    optimizer_.leaf_range.z = v;
    markDirty();
}
void SettingsBridge::setLeafRangeXA(double v) {
    if (optimizer_.leaf_range.xa == v) {
        return;
    }
    optimizer_.leaf_range.xa = v;
    markDirty();
}
void SettingsBridge::setLeafRangeYA(double v) {
    if (optimizer_.leaf_range.ya == v) {
        return;
    }
    optimizer_.leaf_range.ya = v;
    markDirty();
}
void SettingsBridge::setLeafRangeZA(double v) {
    if (optimizer_.leaf_range.za == v) {
        return;
    }
    optimizer_.leaf_range.za = v;
    markDirty();
}
void SettingsBridge::setLeafBudget(int v) {
    if (optimizer_.leaf_budget == v) {
        return;
    }
    optimizer_.leaf_budget = v;
    markDirty();
}
void SettingsBridge::setEnableLeaf(bool v) {
    if (optimizer_.enable_leaf_ == v) {
        return;
    }
    optimizer_.enable_leaf_ = v;
    markDirty();
}

/*----------------------------------------------------------------------------
 * Cost-variant choice
 *----------------------------------------------------------------------------*/

QStringList SettingsBridge::trunkCostFunctions() const {
    return costFunctionNames(trunk_manager_.get());
}
QStringList SettingsBridge::branchCostFunctions() const {
    return costFunctionNames(branch_manager_.get());
}
QStringList SettingsBridge::leafCostFunctions() const {
    return costFunctionNames(leaf_manager_.get());
}

int SettingsBridge::trunkCostFunctionIndex() const {
    return costFunctionIndex(trunk_manager_.get());
}
int SettingsBridge::branchCostFunctionIndex() const {
    return costFunctionIndex(branch_manager_.get());
}
int SettingsBridge::leafCostFunctionIndex() const {
    return costFunctionIndex(leaf_manager_.get());
}
void SettingsBridge::setTrunkCostFunctionIndex(int index) {
    setCostFunctionIndex(trunk_manager_.get(), index);
}
void SettingsBridge::setBranchCostFunctionIndex(int index) {
    setCostFunctionIndex(branch_manager_.get(), index);
}
void SettingsBridge::setLeafCostFunctionIndex(int index) {
    setCostFunctionIndex(leaf_manager_.get(), index);
}

/*----------------------------------------------------------------------------
 * Dilation (active cost function's "Dilation" int parameter)
 *----------------------------------------------------------------------------*/

int SettingsBridge::trunkDilation() const {
    return dilation(trunk_manager_.get(), TRUNK_DILATION);
}
int SettingsBridge::branchDilation() const {
    return dilation(branch_manager_.get(), kBranchDilationDefault);
}
int SettingsBridge::leafDilation() const {
    return dilation(leaf_manager_.get(), Z_SEARCH_DILATION);
}
void SettingsBridge::setTrunkDilation(int v) {
    setDilation(trunk_manager_.get(), v);
}
void SettingsBridge::setBranchDilation(int v) {
    setDilation(branch_manager_.get(), v);
}
void SettingsBridge::setLeafDilation(int v) {
    setDilation(leaf_manager_.get(), v);
}
bool SettingsBridge::trunkHasDilation() const {
    return hasDilation(trunk_manager_.get());
}
bool SettingsBridge::branchHasDilation() const {
    return hasDilation(branch_manager_.get());
}
bool SettingsBridge::leafHasDilation() const {
    return hasDilation(leaf_manager_.get());
}

/*----------------------------------------------------------------------------
 * C++ surface
 *----------------------------------------------------------------------------*/

bool SettingsBridge::dirty() const {
    return dirty_;
}

OptimizerSettings& SettingsBridge::optimizerSettings() {
    return optimizer_;
}

jta_cost_function::CostFunctionManager* SettingsBridge::trunkManager() const {
    return trunk_manager_.get();
}
jta_cost_function::CostFunctionManager* SettingsBridge::branchManager() const {
    return branch_manager_.get();
}
jta_cost_function::CostFunctionManager* SettingsBridge::leafManager() const {
    return leaf_manager_.get();
}

/*----------------------------------------------------------------------------
 * The shared registry mapping (parity contract).
 *
 * Delegates to the shared services function
 *jta::BuildCostFunctionRegistryEntries (plan 006 U1 — the widgets MainScreen
 *calls the same function; the golden fixture in experimental_settings_test.cpp
 *pins the output): same key formats (STAGE@ACTIVE_CF /
 *STAGE@CFname@ParamName@TYPE), same entry order (ACTIVE_CF first, then per
 *available cost function the double/int/bool parameter groups), same value
 *types (double/int/bool QVariants — lossless, no narrowing). This bridge method
 *is now a thin wrapper keeping the QML call site + the C++ test surface
 *unchanged.
 *----------------------------------------------------------------------------*/

std::vector<jta::RegistryEntry>
SettingsBridge::buildCostFunctionRegistryEntries() const {
    return jta::BuildCostFunctionRegistryEntries(
        *trunk_manager_, *branch_manager_, *leaf_manager_);
}

/*----------------------------------------------------------------------------
 * Private helpers
 *----------------------------------------------------------------------------*/

void SettingsBridge::markDirty() {
    if (!dirty_) {
        dirty_ = true;
        emit dirtyChanged();
    }
    emit settingsEdited();
}

void SettingsBridge::applyCostFunctionEntries(
    const std::vector<jta::RegistryEntry>& entries) {
    /*Widgets LoadSettingsBetweenSessions cost-function loop (mainscreen.cpp:
     * 4633), minus the modal error dialogs (malformed keys are skipped).*/
    for (const jta::RegistryEntry& entry : entries) {
        QStringList key_codes = entry.key.split(QStringLiteral("@"));
        if (key_codes.size() == 2 &&
            key_codes[1] == QStringLiteral("ACTIVE_CF")) {
            jta_cost_function::CostFunctionManager* manager =
                managerForStage(key_codes[0]);
            if (manager) {
                manager->setActiveCostFunction(
                    entry.value.toString().toStdString());
            }
        } else if (key_codes.size() == 4) {
            jta_cost_function::CostFunctionManager* manager =
                managerForStage(key_codes[0]);
            if (!manager) {
                continue;
            }
            jta_cost_function::CostFunction* cost_function =
                manager->getCostFunctionClass(key_codes[1].toStdString());
            const QString param_name = key_codes[2];
            const QString param_type = key_codes[3];
            if (param_type == QStringLiteral("DOUBLE")) {
                cost_function->setDoubleParameterValue(
                    param_name.toStdString(), entry.value.toDouble());
            } else if (param_type == QStringLiteral("INT")) {
                cost_function->setIntParameterValue(
                    param_name.toStdString(), entry.value.toInt());
            } else if (param_type == QStringLiteral("BOOL")) {
                cost_function->setBoolParameterValue(
                    param_name.toStdString(), entry.value.toBool());
            }
            /*Unknown type suffix: skipped (widgets error code D/E/F).*/
        }
        /*Wrong code count: skipped (widgets error code C).*/
    }
}

jta_cost_function::CostFunctionManager* SettingsBridge::managerForStage(
    const QString& stage) const {
    if (stage == QStringLiteral("TRUNK")) {
        return trunk_manager_.get();
    }
    if (stage == QStringLiteral("BRANCH")) {
        return branch_manager_.get();
    }
    if (stage == QStringLiteral("LEAF")) {
        return leaf_manager_.get();
    }
    /*Unknown stage: skipped (widgets error codes A/B).*/
    return nullptr;
}

int SettingsBridge::costFunctionIndex(
    jta_cost_function::CostFunctionManager* manager) const {
    const QStringList names = costFunctionNames(manager);
    const QString active =
        QString::fromStdString(manager->getActiveCostFunction());
    return names.indexOf(active);
}

void SettingsBridge::setCostFunctionIndex(
    jta_cost_function::CostFunctionManager* manager,
    int index) {
    const QStringList names = costFunctionNames(manager);
    if (index < 0 || index >= names.size()) {
        return;
    }
    if (QString::fromStdString(manager->getActiveCostFunction()) ==
        names[index]) {
        return;
    }
    manager->setActiveCostFunction(names[index].toStdString());
    markDirty();
}

int SettingsBridge::dilation(
    jta_cost_function::CostFunctionManager* manager,
    int fallback_default) const {
    /*Read the "Dilation" int parameter of the ACTIVE cost function (the
     * widgets UpdateDilationFrames search pattern, mainscreen.cpp:5033).*/
    std::vector<jta_cost_function::Parameter<int>> active_int_params =
        manager->getActiveCostFunctionClass()->getIntParameters();
    for (auto& param : active_int_params) {
        if (param.getParameterName() == "Dilation") {
            return param.getParameterValue();
        }
    }
    return fallback_default;
}

void SettingsBridge::setDilation(
    jta_cost_function::CostFunctionManager* manager,
    int v) {
    if (!hasDilation(manager)) {
        return;
    }
    if (dilation(manager, -1) == v) {
        return;
    }
    manager->getActiveCostFunctionClass()->setIntParameterValue("Dilation", v);
    markDirty();
}

bool SettingsBridge::hasDilation(
    jta_cost_function::CostFunctionManager* manager) const {
    std::vector<jta_cost_function::Parameter<int>> active_int_params =
        manager->getActiveCostFunctionClass()->getIntParameters();
    for (auto& param : active_int_params) {
        if (param.getParameterName() == "Dilation") {
            return true;
        }
    }
    return false;
}

QStringList SettingsBridge::costFunctionNames(
    jta_cost_function::CostFunctionManager* manager) const {
    QStringList names;
    std::vector<jta_cost_function::CostFunction> available =
        manager->getAvailableCostFunctions();
    for (auto& cost_function : available) {
        names.append(
            QString::fromStdString(cost_function.getCostFunctionName()));
    }
    return names;
}
