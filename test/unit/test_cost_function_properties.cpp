// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel PBT for the cost-function parameter registry (jta_cost_function::CostFunction,
// merged into jtml_compute at 003 U4 - a pre-existing pure surface that had ZERO
// headless coverage). Locks the invariants the JTA client's cost-tuning UI relies on:
//   - typed round-trip: addParameter(Parameter<double>), setDoubleParameterValue,
//     getDoubleParameterValue recovers the value bit-exact (same for int/bool),
//   - unknown-name rejection: set*/get* with an unregistered name returns false and
//     leaves the out-param untouched,
//   - cross-type rejection: a name registered as double rejects int/bool accessors,
//   - count/enumeration: get*Parameters() sizes match the number of addParameter calls.
// R2-safe: round-trip + enumeration only, never re-deriving the storage semantics.

#include <string>
#include <vector>

#include <hegel/hegel.h>

#include <catch2/catch_test_macros.hpp>

#include "compute/CostFunction.h"

namespace gs = hegel::generators;
using jta_cost_function::CostFunction;
using jta_cost_function::Parameter;

namespace {

// A bounded name; the draw loop appends a unique-index suffix so registrations
// are collision-free for the round-trip probe.
auto NameGen() {
    return gs::sampled_from<std::string>({"dilate", "range", "z_open", "angle", "tol"});
}

}  // namespace

TEST_CASE("CostFunction[PBT]: typed parameter round-trip is bit-exact",
          "[cost_function][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 8}));
            auto dval = gs::floats<double>({.min_value = -1e6, .max_value = 1e6});
            auto ival = gs::integers<int>({.min_value = -1e6, .max_value = 1e6});
            CostFunction cf;
            std::vector<std::string> dnames, inames, bnames;
            for (int i = 0; i < n; ++i) {
                std::string base = tc.draw(NameGen());
                dnames.push_back(base + ":d" + std::to_string(i));
                cf.addParameter(Parameter<double>(dnames.back(), tc.draw(dval)));
                inames.push_back(base + ":i" + std::to_string(i));
                cf.addParameter(Parameter<int>(inames.back(), tc.draw(ival)));
                bnames.push_back(base + ":b" + std::to_string(i));
                cf.addParameter(Parameter<bool>(bnames.back(), tc.draw(gs::booleans())));
            }
            // Re-set each registered param and recover it bit-exact.
            for (const auto& nm : dnames) {
                double v = tc.draw(dval);
                REQUIRE(cf.setDoubleParameterValue(nm, v));
                double got = -999.0;
                REQUIRE(cf.getDoubleParameterValue(nm, got));
                REQUIRE(got == v);
            }
            for (const auto& nm : inames) {
                int v = tc.draw(ival);
                REQUIRE(cf.setIntParameterValue(nm, v));
                int got = -999;
                REQUIRE(cf.getIntParameterValue(nm, got));
                REQUIRE(got == v);
            }
            for (const auto& nm : bnames) {
                bool v = tc.draw(gs::booleans());
                REQUIRE(cf.setBoolParameterValue(nm, v));
                bool got = false;
                REQUIRE(cf.getBoolParameterValue(nm, got));
                REQUIRE(got == v);
            }
            // Count invariants: one entry per addParameter call, type-separated.
            REQUIRE(cf.getDoubleParameters().size() == dnames.size());
            REQUIRE(cf.getIntParameters().size() == inames.size());
            REQUIRE(cf.getBoolParameters().size() == bnames.size());
        },
        hegel::Settings{.test_cases = 300});
}

TEST_CASE("CostFunction[PBT]: unknown/cross-type names are rejected cleanly",
          "[cost_function][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto n = tc.draw(gs::integers<int>({.min_value = 1, .max_value = 5}));
            CostFunction cf;
            for (int i = 0; i < n; ++i) {
                cf.addParameter(Parameter<double>("d" + std::to_string(i), 0.0));
                cf.addParameter(Parameter<int>("i" + std::to_string(i), 0));
            }
            // Names that do not exist as doubles (int names + random garbage).
            auto junk = tc.draw(gs::text({.min_size = 1, .max_size = 12}));
            double out = 42.25;
            REQUIRE_FALSE(cf.getDoubleParameterValue("i0", out));
            REQUIRE(out == 42.25);  // out-param untouched on failure
            REQUIRE_FALSE(cf.setDoubleParameterValue("i0", 1.0));
            REQUIRE_FALSE(cf.getDoubleParameterValue(junk, out));
            REQUIRE_FALSE(cf.setDoubleParameterValue(junk, 1.0));
            int iout = 7;
            REQUIRE_FALSE(cf.getIntParameterValue("d0", iout));
            REQUIRE(iout == 7);
            REQUIRE_FALSE(cf.setIntParameterValue("d0", 1));
        },
        hegel::Settings{.test_cases = 300});
}
