#include "gui/settings_service.h"

#include <QSettings>
#include <QTemporaryDir>

#include <iostream>

namespace {

int fail(const char* message) {
    std::cerr << message << std::endl;
    return 1;
}

} // namespace

int main() {
    QTemporaryDir temp_dir;
    if (!temp_dir.isValid()) {
        return fail("temporary settings directory unavailable");
    }

    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, temp_dir.path());

    jta_gui::SettingsService service;

    jta_core::SessionContext source;
    source.optimizer_settings_.trunk_budget = 1234;
    source.optimizer_settings_.branch_budget = 4321;
    source.optimizer_settings_.leaf_budget = 2468;
    source.optimizer_settings_.enable_branch_ = false;
    source.optimizer_settings_.enable_leaf_ = true;
    source.optimizer_settings_.number_branches = 7;
    source.optimizer_settings_.trunk_range = Point6D(11, 12, 13, 14, 15, 16);
    source.optimizer_settings_.branch_range = Point6D(21, 22, 23, 24, 25, 26);
    source.optimizer_settings_.leaf_range = Point6D(31, 32, 33, 34, 35, 36);
    source.trunk_manager_.setActiveCostFunction("DIRECT_DILATION");
    source.trunk_manager_.getCostFunctionClass("DIRECT_DILATION")
        ->setIntParameterValue("Dilation", 9);

    service.SaveEdgeDetectionSettings(5, 55, 155);
    service.SaveSettings(source);

    jta_core::SessionContext loaded;
    service.LoadSettings(loaded);

    if (loaded.optimizer_settings_.trunk_budget != source.optimizer_settings_.trunk_budget ||
        loaded.optimizer_settings_.branch_budget != source.optimizer_settings_.branch_budget ||
        loaded.optimizer_settings_.leaf_budget != source.optimizer_settings_.leaf_budget) {
        return fail("optimizer budgets did not round-trip");
    }

    if (loaded.optimizer_settings_.number_branches !=
            source.optimizer_settings_.number_branches ||
        loaded.optimizer_settings_.enable_branch_ !=
            source.optimizer_settings_.enable_branch_ ||
        loaded.optimizer_settings_.enable_leaf_ != source.optimizer_settings_.enable_leaf_) {
        return fail("optimizer toggles did not round-trip");
    }

    if (loaded.optimizer_settings_.trunk_range.x != source.optimizer_settings_.trunk_range.x ||
        loaded.optimizer_settings_.branch_range.ya !=
            source.optimizer_settings_.branch_range.ya ||
        loaded.optimizer_settings_.leaf_range.za != source.optimizer_settings_.leaf_range.za) {
        return fail("optimizer ranges did not round-trip");
    }

    int loaded_dilation = -1;
    if (!loaded.trunk_manager_.getCostFunctionClass("DIRECT_DILATION")
             ->getIntParameterValue("Dilation", loaded_dilation) ||
        loaded_dilation != 9) {
        return fail("cost function parameters did not round-trip");
    }

    const auto edge_settings = service.GetEdgeDetectionSettings();
    if (edge_settings.aperture != 5 || edge_settings.low_threshold != 55 ||
        edge_settings.high_threshold != 155) {
        return fail("edge detection settings did not round-trip");
    }

    return 0;
}
