// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Deterministic Catch2 twin for the cost-function parameter registry
// (complements the hegel PBT in test_cost_function_properties.cpp per the house
// rule: PBT guards invariants, deterministic cases pin the specific edges).

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "compute/CostFunction.h"

using jta_cost_function::CostFunction;
using jta_cost_function::Parameter;

TEST_CASE("CostFunction: typed round-trip + name/metadata edges", "[cost_function]") {
    CostFunction cf("MyCost");
    REQUIRE(cf.getCostFunctionName() == "MyCost");

    cf.addParameter(Parameter<double>("range", 15.0));
    cf.addParameter(Parameter<int>("iters", 5000));
    cf.addParameter(Parameter<bool>("z_open", true));

    double d = 0.0;
    REQUIRE(cf.getDoubleParameterValue("range", d));
    REQUIRE(d == 15.0);
    REQUIRE(cf.setDoubleParameterValue("range", 20.0));
    REQUIRE(cf.getDoubleParameterValue("range", d));
    REQUIRE(d == 20.0);

    int iv = 0;
    REQUIRE(cf.getIntParameterValue("iters", iv));
    REQUIRE(iv == 5000);
    REQUIRE(cf.setIntParameterValue("iters", 6000));
    REQUIRE(cf.getIntParameterValue("iters", iv));
    REQUIRE(iv == 6000);

    bool b = false;
    REQUIRE(cf.getBoolParameterValue("z_open", b));
    REQUIRE(b == true);
    REQUIRE(cf.setBoolParameterValue("z_open", false));
    REQUIRE(cf.getBoolParameterValue("z_open", b));
    REQUIRE(b == false);

    // No cross-type aliasing: an int name is not a double name.
    REQUIRE_FALSE(cf.getDoubleParameterValue("iters", d));
    REQUIRE_FALSE(cf.setDoubleParameterValue("iters", 1.0));
    // Unknown names rejected, out-param untouched.
    double sentinel = 99.0;
    REQUIRE_FALSE(cf.getDoubleParameterValue("nope", sentinel));
    REQUIRE(sentinel == 99.0);
    REQUIRE_FALSE(cf.setDoubleParameterValue("nope", 1.0));

    // Type-separated enumeration.
    REQUIRE(cf.getDoubleParameters().size() == 1);
    REQUIRE(cf.getIntParameters().size() == 1);
    REQUIRE(cf.getBoolParameters().size() == 1);
}

TEST_CASE("CostFunction: default parameter + empty registry", "[cost_function]") {
    CostFunction cf;
    REQUIRE(cf.getCostFunctionName() == "Nameless_Cost_Function");
    REQUIRE(cf.getDoubleParameters().empty());
    double d = 1.0;
    REQUIRE_FALSE(cf.getDoubleParameterValue("x", d));
}
