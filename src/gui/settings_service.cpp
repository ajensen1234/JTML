#include "gui/settings_service.h"

#include "core/settings_constants.h"

#include <QSettings>

#include <array>

namespace {

constexpr auto kOrganizationName = "JointTrackAutoGPU";
constexpr auto kRootKey = "JTML";
constexpr auto kSchemaVersionKey = "JTML/SchemaVersion";
constexpr auto kSchemaVersion = 1;

constexpr auto kLegacyCostFunctionGroup = "CostFunctionSettings";
constexpr auto kLegacyOptimizerGroup = "OptimizerSettings";
constexpr auto kLegacyEdgeGroup = "EdgeDetectionSettings";

QString BuildVersionString() {
    return "Version" + QString::number(VER_FIRST_NUM) +
           QString::number(VER_MIDDLE_NUM) + QString::number(VER_LAST_NUM);
}

QSettings BuildSettings() {
    return QSettings(kOrganizationName, BuildVersionString());
}

struct StageConfig {
    const char* standardized_stage;
    const char* legacy_stage;
    jta_cost_function::CostFunctionManager* manager;
};

std::array<StageConfig, 3>
MutableStageConfigs(jta_core::SessionContext& session) {
    return {{{"Trunk", "TRUNK", &session.trunk_manager_},
             {"Branch", "BRANCH", &session.branch_manager_},
             {"Leaf", "LEAF", &session.leaf_manager_}}};
}

std::array<StageConfig, 3>
ConstStageConfigs(const jta_core::SessionContext& session) {
    auto& mutable_session = const_cast<jta_core::SessionContext&>(session);
    return MutableStageConfigs(mutable_session);
}

QString StagePrefix(const char* stage_name) {
    return QString("%1/%2").arg(kRootKey, stage_name);
}

QString OptimizerPrefix(const char* stage_name) {
    return QString("%1/Optimizer/%2").arg(kRootKey, stage_name);
}

QString StageActiveCostFunctionKey(const char* stage_name) {
    return QString("%1/ActiveCostFunction").arg(StagePrefix(stage_name));
}

QString StageCostFunctionParameterKey(
    const char* stage_name,
    const QString& cost_function_name,
    const char* type_name,
    const QString& parameter_name) {
    return QString("%1/CostFunctions/%2/%3/%4")
        .arg(StagePrefix(stage_name), cost_function_name, type_name, parameter_name);
}

void SaveStageCostFunctionSettings(
    QSettings& settings,
    const char* stage_name,
    jta_cost_function::CostFunctionManager& manager) {
    settings.setValue(
        StageActiveCostFunctionKey(stage_name),
        QString::fromStdString(manager.getActiveCostFunction()));

    std::vector<jta_cost_function::CostFunction> available_cost_functions =
        manager.getAvailableCostFunctions();
    for (auto& cost_function : available_cost_functions) {
        const QString cost_function_name =
            QString::fromStdString(cost_function.getCostFunctionName());

        auto double_parameters = cost_function.getDoubleParameters();
        for (auto& parameter : double_parameters) {
            settings.setValue(
                StageCostFunctionParameterKey(
                    stage_name,
                    cost_function_name,
                    "Double",
                    QString::fromStdString(parameter.getParameterName())),
                parameter.getParameterValue());
        }

        auto int_parameters = cost_function.getIntParameters();
        for (auto& parameter : int_parameters) {
            settings.setValue(
                StageCostFunctionParameterKey(
                    stage_name,
                    cost_function_name,
                    "Int",
                    QString::fromStdString(parameter.getParameterName())),
                parameter.getParameterValue());
        }

        auto bool_parameters = cost_function.getBoolParameters();
        for (auto& parameter : bool_parameters) {
            settings.setValue(
                StageCostFunctionParameterKey(
                    stage_name,
                    cost_function_name,
                    "Bool",
                    QString::fromStdString(parameter.getParameterName())),
                parameter.getParameterValue());
        }
    }
}

void SaveOptimizerSettings(QSettings& settings, const OptimizerSettings& optimizer_settings) {
    const QString trunk_prefix = OptimizerPrefix("Trunk");
    settings.setValue(
        QString("%1/Range/X").arg(trunk_prefix), optimizer_settings.trunk_range.x);
    settings.setValue(
        QString("%1/Range/Y").arg(trunk_prefix), optimizer_settings.trunk_range.y);
    settings.setValue(
        QString("%1/Range/Z").arg(trunk_prefix), optimizer_settings.trunk_range.z);
    settings.setValue(
        QString("%1/Range/XA").arg(trunk_prefix), optimizer_settings.trunk_range.xa);
    settings.setValue(
        QString("%1/Range/YA").arg(trunk_prefix), optimizer_settings.trunk_range.ya);
    settings.setValue(
        QString("%1/Range/ZA").arg(trunk_prefix), optimizer_settings.trunk_range.za);
    settings.setValue(QString("%1/Budget").arg(trunk_prefix), optimizer_settings.trunk_budget);

    const QString branch_prefix = OptimizerPrefix("Branch");
    settings.setValue(
        QString("%1/Range/X").arg(branch_prefix), optimizer_settings.branch_range.x);
    settings.setValue(
        QString("%1/Range/Y").arg(branch_prefix), optimizer_settings.branch_range.y);
    settings.setValue(
        QString("%1/Range/Z").arg(branch_prefix), optimizer_settings.branch_range.z);
    settings.setValue(
        QString("%1/Range/XA").arg(branch_prefix), optimizer_settings.branch_range.xa);
    settings.setValue(
        QString("%1/Range/YA").arg(branch_prefix), optimizer_settings.branch_range.ya);
    settings.setValue(
        QString("%1/Range/ZA").arg(branch_prefix), optimizer_settings.branch_range.za);
    settings.setValue(
        QString("%1/NumberBranches").arg(branch_prefix),
        optimizer_settings.number_branches);
    settings.setValue(
        QString("%1/Enable").arg(branch_prefix), optimizer_settings.enable_branch_);
    settings.setValue(
        QString("%1/Budget").arg(branch_prefix), optimizer_settings.branch_budget);

    const QString leaf_prefix = OptimizerPrefix("Leaf");
    settings.setValue(
        QString("%1/Range/X").arg(leaf_prefix), optimizer_settings.leaf_range.x);
    settings.setValue(
        QString("%1/Range/Y").arg(leaf_prefix), optimizer_settings.leaf_range.y);
    settings.setValue(
        QString("%1/Range/Z").arg(leaf_prefix), optimizer_settings.leaf_range.z);
    settings.setValue(
        QString("%1/Range/XA").arg(leaf_prefix), optimizer_settings.leaf_range.xa);
    settings.setValue(
        QString("%1/Range/YA").arg(leaf_prefix), optimizer_settings.leaf_range.ya);
    settings.setValue(
        QString("%1/Range/ZA").arg(leaf_prefix), optimizer_settings.leaf_range.za);
    settings.setValue(
        QString("%1/Enable").arg(leaf_prefix), optimizer_settings.enable_leaf_);
    settings.setValue(
        QString("%1/Budget").arg(leaf_prefix), optimizer_settings.leaf_budget);
}

void PersistEdgeDetectionSettings(
    QSettings& settings,
    const jta_gui::EdgeDetectionSettings& edge_detection_settings) {
    settings.setValue(
        QString("%1/EdgeDetection/Aperture").arg(kRootKey),
        edge_detection_settings.aperture);
    settings.setValue(
        QString("%1/EdgeDetection/LowThreshold").arg(kRootKey),
        edge_detection_settings.low_threshold);
    settings.setValue(
        QString("%1/EdgeDetection/HighThreshold").arg(kRootKey),
        edge_detection_settings.high_threshold);

    settings.beginGroup(kLegacyEdgeGroup);
    settings.setValue("APERTURE", edge_detection_settings.aperture);
    settings.setValue("LOW_THRESH", edge_detection_settings.low_threshold);
    settings.setValue("HIGH_THRESH", edge_detection_settings.high_threshold);
    settings.endGroup();
}

bool HasStandardizedSettings(const QSettings& settings) {
    return settings.contains(kSchemaVersionKey) ||
           settings.contains(StageActiveCostFunctionKey("Trunk"));
}

bool HasLegacySettings(QSettings& settings) {
    const QStringList groups = settings.childGroups();
    return groups.contains(kLegacyCostFunctionGroup) ||
           groups.contains(kLegacyOptimizerGroup);
}

jta_cost_function::CostFunctionManager* LegacyStageToManager(
    const QString& stage_name,
    jta_core::SessionContext& session) {
    if (stage_name == "TRUNK") {
        return &session.trunk_manager_;
    }
    if (stage_name == "BRANCH") {
        return &session.branch_manager_;
    }
    if (stage_name == "LEAF") {
        return &session.leaf_manager_;
    }
    return nullptr;
}

void LoadStandardizedStageCostFunctionSettings(
    const QSettings& settings,
    const char* stage_name,
    jta_cost_function::CostFunctionManager& manager) {
    const QString active_cf_key = StageActiveCostFunctionKey(stage_name);
    if (settings.contains(active_cf_key)) {
        manager.setActiveCostFunction(settings.value(active_cf_key).toString().toStdString());
    }

    auto available_cost_functions = manager.getAvailableCostFunctions();
    for (auto& cost_function : available_cost_functions) {
        const std::string cost_function_name_std = cost_function.getCostFunctionName();
        const QString cost_function_name = QString::fromStdString(cost_function_name_std);

        auto double_parameters = cost_function.getDoubleParameters();
        for (auto& parameter : double_parameters) {
            const QString key = StageCostFunctionParameterKey(
                stage_name,
                cost_function_name,
                "Double",
                QString::fromStdString(parameter.getParameterName()));
            if (settings.contains(key)) {
                manager.getCostFunctionClass(cost_function_name_std)
                    ->setDoubleParameterValue(
                        parameter.getParameterName(), settings.value(key).toDouble());
            }
        }

        auto int_parameters = cost_function.getIntParameters();
        for (auto& parameter : int_parameters) {
            const QString key = StageCostFunctionParameterKey(
                stage_name,
                cost_function_name,
                "Int",
                QString::fromStdString(parameter.getParameterName()));
            if (settings.contains(key)) {
                manager.getCostFunctionClass(cost_function_name_std)
                    ->setIntParameterValue(
                        parameter.getParameterName(), settings.value(key).toInt());
            }
        }

        auto bool_parameters = cost_function.getBoolParameters();
        for (auto& parameter : bool_parameters) {
            const QString key = StageCostFunctionParameterKey(
                stage_name,
                cost_function_name,
                "Bool",
                QString::fromStdString(parameter.getParameterName()));
            if (settings.contains(key)) {
                manager.getCostFunctionClass(cost_function_name_std)
                    ->setBoolParameterValue(
                        parameter.getParameterName(), settings.value(key).toBool());
            }
        }
    }
}

void LoadLegacyCostFunctionSettings(
    QSettings& settings,
    jta_core::SessionContext& session) {
    settings.beginGroup(kLegacyCostFunctionGroup);

    const QStringList keys = settings.allKeys();
    for (const auto& key : keys) {
        const QStringList key_codes = key.split("@");
        if (key_codes.size() == 2 && key_codes[1] == "ACTIVE_CF") {
            if (auto* manager = LegacyStageToManager(key_codes[0], session); manager != nullptr) {
                manager->setActiveCostFunction(settings.value(key).toString().toStdString());
            }
            continue;
        }

        if (key_codes.size() != 4) {
            continue;
        }

        auto* manager = LegacyStageToManager(key_codes[0], session);
        if (manager == nullptr) {
            continue;
        }

        const std::string cost_function_name = key_codes[1].toStdString();
        const std::string parameter_name = key_codes[2].toStdString();
        const QString parameter_type = key_codes[3];
        if (parameter_type == "DOUBLE") {
            manager->getCostFunctionClass(cost_function_name)
                ->setDoubleParameterValue(parameter_name, settings.value(key).toDouble());
        } else if (parameter_type == "INT") {
            manager->getCostFunctionClass(cost_function_name)
                ->setIntParameterValue(parameter_name, settings.value(key).toInt());
        } else if (parameter_type == "BOOL") {
            manager->getCostFunctionClass(cost_function_name)
                ->setBoolParameterValue(parameter_name, settings.value(key).toBool());
        }
    }

    settings.endGroup();
}

void LoadStandardizedOptimizerSettings(
    const QSettings& settings,
    OptimizerSettings& optimizer_settings) {
    const QString trunk_prefix = OptimizerPrefix("Trunk");
    optimizer_settings.trunk_range = Point6D(
        settings.value(
                    QString("%1/Range/X").arg(trunk_prefix),
                    optimizer_settings.trunk_range.x)
            .toDouble(),
        settings.value(
                    QString("%1/Range/Y").arg(trunk_prefix),
                    optimizer_settings.trunk_range.y)
            .toDouble(),
        settings.value(
                    QString("%1/Range/Z").arg(trunk_prefix),
                    optimizer_settings.trunk_range.z)
            .toDouble(),
        settings.value(
                    QString("%1/Range/XA").arg(trunk_prefix),
                    optimizer_settings.trunk_range.xa)
            .toDouble(),
        settings.value(
                    QString("%1/Range/YA").arg(trunk_prefix),
                    optimizer_settings.trunk_range.ya)
            .toDouble(),
        settings.value(
                    QString("%1/Range/ZA").arg(trunk_prefix),
                    optimizer_settings.trunk_range.za)
            .toDouble());
    optimizer_settings.trunk_budget =
        settings.value(QString("%1/Budget").arg(trunk_prefix), optimizer_settings.trunk_budget)
            .toInt();

    const QString branch_prefix = OptimizerPrefix("Branch");
    optimizer_settings.branch_range = Point6D(
        settings.value(
                    QString("%1/Range/X").arg(branch_prefix),
                    optimizer_settings.branch_range.x)
            .toDouble(),
        settings.value(
                    QString("%1/Range/Y").arg(branch_prefix),
                    optimizer_settings.branch_range.y)
            .toDouble(),
        settings.value(
                    QString("%1/Range/Z").arg(branch_prefix),
                    optimizer_settings.branch_range.z)
            .toDouble(),
        settings.value(
                    QString("%1/Range/XA").arg(branch_prefix),
                    optimizer_settings.branch_range.xa)
            .toDouble(),
        settings.value(
                    QString("%1/Range/YA").arg(branch_prefix),
                    optimizer_settings.branch_range.ya)
            .toDouble(),
        settings.value(
                    QString("%1/Range/ZA").arg(branch_prefix),
                    optimizer_settings.branch_range.za)
            .toDouble());
    optimizer_settings.number_branches =
        settings
            .value(
                QString("%1/NumberBranches").arg(branch_prefix),
                optimizer_settings.number_branches)
            .toInt();
    optimizer_settings.enable_branch_ =
        settings
            .value(
                QString("%1/Enable").arg(branch_prefix),
                optimizer_settings.enable_branch_)
            .toBool();
    optimizer_settings.branch_budget =
        settings
            .value(
                QString("%1/Budget").arg(branch_prefix),
                optimizer_settings.branch_budget)
            .toInt();

    const QString leaf_prefix = OptimizerPrefix("Leaf");
    optimizer_settings.leaf_range = Point6D(
        settings.value(
                    QString("%1/Range/X").arg(leaf_prefix),
                    optimizer_settings.leaf_range.x)
            .toDouble(),
        settings.value(
                    QString("%1/Range/Y").arg(leaf_prefix),
                    optimizer_settings.leaf_range.y)
            .toDouble(),
        settings.value(
                    QString("%1/Range/Z").arg(leaf_prefix),
                    optimizer_settings.leaf_range.z)
            .toDouble(),
        settings.value(
                    QString("%1/Range/XA").arg(leaf_prefix),
                    optimizer_settings.leaf_range.xa)
            .toDouble(),
        settings.value(
                    QString("%1/Range/YA").arg(leaf_prefix),
                    optimizer_settings.leaf_range.ya)
            .toDouble(),
        settings.value(
                    QString("%1/Range/ZA").arg(leaf_prefix),
                    optimizer_settings.leaf_range.za)
            .toDouble());
    optimizer_settings.enable_leaf_ =
        settings
            .value(
                QString("%1/Enable").arg(leaf_prefix),
                optimizer_settings.enable_leaf_)
            .toBool();
    optimizer_settings.leaf_budget =
        settings
            .value(QString("%1/Budget").arg(leaf_prefix), optimizer_settings.leaf_budget)
            .toInt();
}

void LoadLegacyOptimizerSettings(QSettings& settings, OptimizerSettings& optimizer_settings) {
    settings.beginGroup(kLegacyOptimizerGroup);

    optimizer_settings.trunk_range = Point6D(
        settings.value("TRUNK@RANGE_X", optimizer_settings.trunk_range.x).toDouble(),
        settings.value("TRUNK@RANGE_Y", optimizer_settings.trunk_range.y).toDouble(),
        settings.value("TRUNK@RANGE_Z", optimizer_settings.trunk_range.z).toDouble(),
        settings.value("TRUNK@RANGE_XA", optimizer_settings.trunk_range.xa).toDouble(),
        settings.value("TRUNK@RANGE_YA", optimizer_settings.trunk_range.ya).toDouble(),
        settings.value("TRUNK@RANGE_ZA", optimizer_settings.trunk_range.za).toDouble());
    optimizer_settings.trunk_budget =
        settings.value("TRUNK@BUDGET", optimizer_settings.trunk_budget).toInt();

    optimizer_settings.branch_range = Point6D(
        settings.value("BRANCH@RANGE_X", optimizer_settings.branch_range.x).toDouble(),
        settings.value("BRANCH@RANGE_Y", optimizer_settings.branch_range.y).toDouble(),
        settings.value("BRANCH@RANGE_Z", optimizer_settings.branch_range.z).toDouble(),
        settings.value("BRANCH@RANGE_XA", optimizer_settings.branch_range.xa).toDouble(),
        settings.value("BRANCH@RANGE_YA", optimizer_settings.branch_range.ya).toDouble(),
        settings.value("BRANCH@RANGE_ZA", optimizer_settings.branch_range.za).toDouble());
    optimizer_settings.number_branches =
        settings
            .value("BRANCH@NUMBER_BRANCHES", optimizer_settings.number_branches)
            .toInt();
    optimizer_settings.enable_branch_ =
        settings.value("BRANCH@ENABLE", optimizer_settings.enable_branch_).toBool();
    optimizer_settings.branch_budget =
        settings.value("BRANCH@BUDGET", optimizer_settings.branch_budget).toInt();

    optimizer_settings.leaf_range = Point6D(
        settings.value("LEAF@RANGE_X", optimizer_settings.leaf_range.x).toDouble(),
        settings.value("LEAF@RANGE_Y", optimizer_settings.leaf_range.y).toDouble(),
        settings.value("LEAF@RANGE_Z", optimizer_settings.leaf_range.z).toDouble(),
        settings.value("LEAF@RANGE_XA", optimizer_settings.leaf_range.xa).toDouble(),
        settings.value("LEAF@RANGE_YA", optimizer_settings.leaf_range.ya).toDouble(),
        settings.value("LEAF@RANGE_ZA", optimizer_settings.leaf_range.za).toDouble());
    optimizer_settings.enable_leaf_ =
        settings.value("LEAF@ENABLE", optimizer_settings.enable_leaf_).toBool();
    optimizer_settings.leaf_budget =
        settings.value("LEAF@BUDGET", optimizer_settings.leaf_budget).toInt();

    settings.endGroup();
}

void LoadEdgeDetectionSettings(
    QSettings& settings,
    jta_gui::EdgeDetectionSettings& edge_detection_settings) {
    settings.beginGroup(kLegacyEdgeGroup);
    const bool has_legacy_edge_keys = settings.contains("APERTURE") ||
                                      settings.contains("LOW_THRESH") ||
                                      settings.contains("HIGH_THRESH");
    if (has_legacy_edge_keys) {
        edge_detection_settings.aperture =
            settings.value("APERTURE", APERTURE).toInt();
        edge_detection_settings.low_threshold =
            settings.value("LOW_THRESH", LOW_THRESH).toInt();
        edge_detection_settings.high_threshold =
            settings.value("HIGH_THRESH", HIGH_THRESH).toInt();
    }
    settings.endGroup();

    if (!has_legacy_edge_keys) {
        edge_detection_settings.aperture =
            settings.value(QString("%1/EdgeDetection/Aperture").arg(kRootKey), APERTURE)
                .toInt();
        edge_detection_settings.low_threshold =
            settings
                .value(
                    QString("%1/EdgeDetection/LowThreshold").arg(kRootKey),
                    LOW_THRESH)
                .toInt();
        edge_detection_settings.high_threshold =
            settings
                .value(
                    QString("%1/EdgeDetection/HighThreshold").arg(kRootKey),
                    HIGH_THRESH)
                .toInt();
    }
}

void SetDefaultSessionSettings(jta_core::SessionContext& session) {
    session.optimizer_settings_ = OptimizerSettings();
    session.trunk_manager_ = jta_cost_function::CostFunctionManager(Stage::Trunk);
    session.branch_manager_ = jta_cost_function::CostFunctionManager(Stage::Branch);
    session.leaf_manager_ = jta_cost_function::CostFunctionManager(Stage::Leaf);

    session.branch_manager_.getCostFunctionClass("DIRECT_DILATION")
        ->setIntParameterValue("Dilation", 4);
    session.leaf_manager_.getCostFunctionClass("DIRECT_DILATION")
        ->setIntParameterValue("Dilation", 1);
}

} // namespace

namespace jta_gui {

void SettingsService::LoadSettings(jta_core::SessionContext& session) {
    QSettings settings = BuildSettings();
    first_time_loading_ = false;

    if (HasStandardizedSettings(settings)) {
        for (const auto& stage : MutableStageConfigs(session)) {
            LoadStandardizedStageCostFunctionSettings(
                settings, stage.standardized_stage, *stage.manager);
        }
        LoadStandardizedOptimizerSettings(settings, session.optimizer_settings_);
        LoadEdgeDetectionSettings(settings, edge_detection_settings_);
        return;
    }

    if (HasLegacySettings(settings)) {
        LoadLegacyCostFunctionSettings(settings, session);
        LoadLegacyOptimizerSettings(settings, session.optimizer_settings_);
        LoadEdgeDetectionSettings(settings, edge_detection_settings_);

        SaveSettings(session);
        return;
    }

    first_time_loading_ = true;
    edge_detection_settings_ = {APERTURE, LOW_THRESH, HIGH_THRESH};
    SetDefaultSessionSettings(session);
    SaveSettings(session);
}

void SettingsService::SaveSettings(const jta_core::SessionContext& session) {
    QSettings settings = BuildSettings();

    settings.remove(kRootKey);
    settings.setValue(kSchemaVersionKey, kSchemaVersion);

    for (const auto& stage : ConstStageConfigs(session)) {
        SaveStageCostFunctionSettings(
            settings, stage.standardized_stage, *stage.manager);
    }
    SaveOptimizerSettings(settings, session.optimizer_settings_);
    PersistEdgeDetectionSettings(settings, edge_detection_settings_);
}

void SettingsService::SaveEdgeDetectionSettings(
    int aperture, int low_threshold, int high_threshold) {
    edge_detection_settings_ = {aperture, low_threshold, high_threshold};

    QSettings settings = BuildSettings();
    PersistEdgeDetectionSettings(settings, edge_detection_settings_);
}

const EdgeDetectionSettings& SettingsService::GetEdgeDetectionSettings() const {
    return edge_detection_settings_;
}

bool SettingsService::WasFirstTimeLoading() const {
    return first_time_loading_;
}

} // namespace jta_gui
