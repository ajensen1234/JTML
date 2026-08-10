// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel PBT for the SettingsService (plan 004 U3 / R9 / R15). Locks the
// save -> load round-trip invariant: every key/group survives exactly, and
// fractional / negative / +/-0.0 values survive bit-exact -- the
// silent-narrowing invariant shape from
// docs/solutions/logic-errors/cost-function-parameter-double-truncation-2026-08-08.md.
//
// QtCore-only: QSettings is redirected to a fresh temp ini file per case via
// the service's path/format override; the real registry is never touched.
// The writer service is destroyed (block scope) before the reader is created,
// so the ini file is flushed and the reload is a genuine restart-equivalent
// round-trip.

#include <algorithm>
#include <cmath>

#include <hegel/hegel.h>

#include <catch2/catch_test_macros.hpp>

#include <QString>
#include <QTemporaryDir>
#include <QVariant>

#include "services/settings_service.h"

namespace gs = hegel::generators;

namespace {

/*Drawn value plus its QVariant, so the reload can be compared per type.*/
struct DrawnEntry {
    QString key;
    QVariant value;
    double as_double;
    int as_int;
    bool as_bool;
    bool is_double;
    bool is_int;
};

/*A drawn double that exercises the narrowing boundaries: +/-0.0, fractionals,
 * negatives, plus uniform floats across the range.*/
double drawDouble(hegel::TestCase& tc) {
    return tc.draw(gs::one_of({gs::sampled_from<double>({-0.0, 0.0, 0.1, -2.25, 3.141592653589793}),
                                gs::floats<double>({.min_value = -100.0, .max_value = 100.0})}));
}

}  // namespace

TEST_CASE(
    "settings_service[PBT]: save -> load round-trip preserves every key/group "
    "exactly",
    "[settings_service][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            QTemporaryDir dir;
            REQUIRE(dir.isValid());
            const QString path = dir.filePath("settings.ini");

            /*Draw unique registry keys (the STAGE@NAME@PARAM@TYPE alphabet;
             * no '/' so QSettings treats each as a flat key under the group).*/
            auto key_gen = gs::text({
                .min_size = 1,
                .max_size = 24,
                .alphabet =
                    "TRUNKBRANCHLEAF@_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            });
            std::vector<std::string> keys = tc.draw(
                gs::vectors(key_gen, {.min_size = 0,
                                      .max_size = 10,
                                      .unique = true}));
            /*QSettings reserves '/' as a path separator (a '/' key is
             * normalized to an empty key and cannot round-trip). App registry
             * keys never contain '/' (they use '@'), so exclude it.*/
            keys.erase(
                std::remove_if(
                    keys.begin(), keys.end(),
                    [](const std::string& k) {
                        return k.find('/') != std::string::npos;
                    }),
                keys.end());

            /*Draw the settings state.*/
            std::vector<DrawnEntry> drawn_entries;
            for (const std::string& key : keys) {
                DrawnEntry e;
                e.key = QString::fromStdString(key);
                int type = tc.draw(gs::sampled_from<int>({0, 1, 2}));
                e.is_double = (type == 0);
                e.is_int = (type == 1);
                if (type == 0) {
                    e.as_double = drawDouble(tc);
                    e.value = e.as_double;
                    e.as_int = 0;
                    e.as_bool = false;
                } else if (type == 1) {
                    e.as_int = tc.draw(
                        gs::integers<int>({.min_value = -100000,
                                           .max_value = 100000}));
                    e.value = e.as_int;
                    e.as_double = 0.0;
                    e.as_bool = false;
                } else {
                    e.as_bool = tc.draw(gs::booleans());
                    e.value = e.as_bool;
                    e.as_double = 0.0;
                    e.as_int = 0;
                }
                drawn_entries.push_back(e);
            }

            OptimizerSettings opt;
            auto coord =
                gs::floats<double>({.min_value = -100.0, .max_value = 100.0});
            opt.trunk_range =
                Point6D(tc.draw(coord), tc.draw(coord), tc.draw(coord),
                        tc.draw(coord), tc.draw(coord), tc.draw(coord));
            opt.trunk_budget = tc.draw(
                gs::integers<int>({.min_value = -100000,
                                   .max_value = 100000}));
            opt.branch_range =
                Point6D(tc.draw(coord), tc.draw(coord), tc.draw(coord),
                        tc.draw(coord), tc.draw(coord), tc.draw(coord));
            opt.number_branches = tc.draw(
                gs::integers<int>({.min_value = -100,
                                   .max_value = 100}));
            opt.enable_branch_ = tc.draw(gs::booleans());
            opt.branch_budget = tc.draw(
                gs::integers<int>({.min_value = -100000,
                                   .max_value = 100000}));
            opt.leaf_range =
                Point6D(tc.draw(coord), tc.draw(coord), tc.draw(coord),
                        tc.draw(coord), tc.draw(coord), tc.draw(coord));
            opt.enable_leaf_ = tc.draw(gs::booleans());
            opt.leaf_budget = tc.draw(
                gs::integers<int>({.min_value = -100000,
                                   .max_value = 100000}));
            const int aperture = tc.draw(
                gs::integers<int>({.min_value = -100, .max_value = 100}));
            const int low_thresh = tc.draw(
                gs::integers<int>({.min_value = -100, .max_value = 100}));
            const int high_thresh = tc.draw(
                gs::integers<int>({.min_value = -100, .max_value = 100}));

            /*First run on a fresh registry: defaults, no groups.*/
            {
                jta::SettingsService svc(path, QSettings::IniFormat);
                REQUIRE(svc.IsFirstTime());
                jta::SettingsService::LoadResult first = svc.LoadSettings();
                REQUIRE(first.first_time);
                REQUIRE(first.cost_function_entries.empty());
                REQUIRE(first.optimizer.trunk_budget == TRUNK_BUDGET);
                REQUIRE(first.edge.aperture == APERTURE);
                REQUIRE(first.edge.low_thresh == LOW_THRESH);
                REQUIRE(first.edge.high_thresh == HIGH_THRESH);

                std::vector<jta::RegistryEntry> entries;
                entries.reserve(drawn_entries.size());
                for (const DrawnEntry& e : drawn_entries) {
                    entries.push_back({e.key, e.value});
                }
                svc.SaveCostFunctionSettings(entries);
                svc.SaveOptimizerSettings(opt);
                svc.SaveEdgeSettings(aperture, low_thresh, high_thresh);
            }

            /*Restart-equivalent reload through a fresh service.*/
            {
                jta::SettingsService svc(path, QSettings::IniFormat);
                REQUIRE_FALSE(svc.IsFirstTime());
                jta::SettingsService::LoadResult loaded = svc.LoadSettings();
                REQUIRE_FALSE(loaded.first_time);

                /*Every cost-function key preserved, values bit-exact. Keys
                 * come back sorted (QSettings::allKeys() ordering); the view
                 * applies each entry independently, so the invariant is the
                 * key SET + value exactness, not insertion order.*/
                REQUIRE(loaded.cost_function_entries.size() ==
                        drawn_entries.size());
                for (const DrawnEntry& drawn : drawn_entries) {
                    auto it = std::find_if(
                        loaded.cost_function_entries.begin(),
                        loaded.cost_function_entries.end(),
                        [&](const jta::RegistryEntry& loaded_entry) {
                            return loaded_entry.key == drawn.key;
                        });
                    REQUIRE(it != loaded.cost_function_entries.end());
                    const QVariant& got = it->value;
                    if (drawn.is_double) {
                        const double expected = drawn.as_double;
                        REQUIRE(got.toDouble() == expected);
                        /*+/-0.0 sign preservation (the +/-0.0 draw edge).*/
                        if (expected == 0.0) {
                            REQUIRE(std::signbit(got.toDouble()) ==
                                    std::signbit(expected));
                        }
                    } else if (drawn.is_int) {
                        REQUIRE(got.toInt() == drawn.as_int);
                    } else {
                        REQUIRE(got.toBool() == drawn.as_bool);
                    }
                }

                /*Optimizer settings bit-exact (no silent narrowing).*/
                REQUIRE(loaded.optimizer.trunk_range.x == opt.trunk_range.x);
                REQUIRE(loaded.optimizer.trunk_range.y == opt.trunk_range.y);
                REQUIRE(loaded.optimizer.trunk_range.z == opt.trunk_range.z);
                REQUIRE(loaded.optimizer.trunk_range.xa ==
                        opt.trunk_range.xa);
                REQUIRE(loaded.optimizer.trunk_range.ya ==
                        opt.trunk_range.ya);
                REQUIRE(loaded.optimizer.trunk_range.za ==
                        opt.trunk_range.za);
                REQUIRE(loaded.optimizer.trunk_budget == opt.trunk_budget);
                REQUIRE(loaded.optimizer.branch_range.x ==
                        opt.branch_range.x);
                REQUIRE(loaded.optimizer.branch_range.y ==
                        opt.branch_range.y);
                REQUIRE(loaded.optimizer.branch_range.z ==
                        opt.branch_range.z);
                REQUIRE(loaded.optimizer.branch_range.xa ==
                        opt.branch_range.xa);
                REQUIRE(loaded.optimizer.branch_range.ya ==
                        opt.branch_range.ya);
                REQUIRE(loaded.optimizer.branch_range.za ==
                        opt.branch_range.za);
                REQUIRE(loaded.optimizer.number_branches ==
                        opt.number_branches);
                REQUIRE(loaded.optimizer.enable_branch_ ==
                        opt.enable_branch_);
                REQUIRE(loaded.optimizer.branch_budget == opt.branch_budget);
                REQUIRE(loaded.optimizer.leaf_range.x == opt.leaf_range.x);
                REQUIRE(loaded.optimizer.leaf_range.y == opt.leaf_range.y);
                REQUIRE(loaded.optimizer.leaf_range.z == opt.leaf_range.z);
                REQUIRE(loaded.optimizer.leaf_range.xa == opt.leaf_range.xa);
                REQUIRE(loaded.optimizer.leaf_range.ya == opt.leaf_range.ya);
                REQUIRE(loaded.optimizer.leaf_range.za == opt.leaf_range.za);
                REQUIRE(loaded.optimizer.enable_leaf_ == opt.enable_leaf_);
                REQUIRE(loaded.optimizer.leaf_budget == opt.leaf_budget);

                /*Edge settings exact.*/
                REQUIRE(loaded.edge.aperture == aperture);
                REQUIRE(loaded.edge.low_thresh == low_thresh);
                REQUIRE(loaded.edge.high_thresh == high_thresh);
            }
        },
        hegel::Settings{.test_cases = 300});
}
