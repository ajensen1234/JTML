/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*BuildCostFunctionRegistryEntries (plan 006 U1 / R8): the raw
 * CostFunctionSettings registry mapping shared by BOTH front-ends — the
 * widgets MainScreen and the QML SettingsBridge call this one function, and
 * the golden 51-entry fixture (test/unit/cost_function_registry_test.cpp +
 * test/unit/experimental_settings_test.cpp) pins its output. The mapping
 * lives in a sibling services module, NOT inside SettingsService (which
 * stays QtCore-only and compute-free): it reads the GPU-linked
 * CostFunctionManager, and jtml_services already links jtml_compute PRIVATE.
 *
 * Key formats STAGE@ACTIVE_CF / STAGE@CFname@ParamName@TYPE; entry order
 * ACTIVE_CF first, then per available cost function (listCostFunctions
 * order) the double/int/bool parameter groups. Relocated verbatim from
 * MainScreen::BuildCostFunctionRegistryEntries (mainscreen.cpp:4895, plan
 * 004 U3) — R13: byte-identical relocation, the golden table is the spec.*/

#ifndef COST_FUNCTION_REGISTRY_H
#define COST_FUNCTION_REGISTRY_H

#include <vector>

#include "services/settings_service.h"

namespace jta_cost_function {
class CostFunctionManager;
}

namespace jta {

/*Builds the raw CostFunctionSettings registry entries from the three cost
 * function managers (trunk/branch/leaf). The SettingsService round-trips the
 * raw entries; the managers are read-only here (the GPU-linked
 * CostFunctionManager stays out of the QtCore-only SettingsService).*/
std::vector<RegistryEntry> BuildCostFunctionRegistryEntries(
    jta_cost_function::CostFunctionManager& trunk_manager,
    jta_cost_function::CostFunctionManager& branch_manager,
    jta_cost_function::CostFunctionManager& leaf_manager);

} /* namespace jta */

#endif /* COST_FUNCTION_REGISTRY_H */
