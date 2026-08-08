// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// CUDA-free unit tests for the pure 6D data structures used by the DIRECT
// optimizer. These compile WITHOUT any GPU/.cu source, which is only possible
// because data_structures_6D.h / direct_data_storage.h no longer pull in
// gpu/render_engine.cuh (see plan U3).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "domain/data_structures_6D.h"
#include "domain/direct_data_storage.h"

using Catch::Approx;

TEST_CASE("Point6D constructors initialize fields", "[data_structures]") {
    SECTION("default constructor zeroes all six DOF") {
        Point6D p;
        REQUIRE(p.x == 0.0);
        REQUIRE(p.y == 0.0);
        REQUIRE(p.z == 0.0);
        REQUIRE(p.xa == 0.0);
        REQUIRE(p.ya == 0.0);
        REQUIRE(p.za == 0.0);
    }
    SECTION("six-arg constructor assigns each field") {
        Point6D p(1, 2, 3, 4, 5, 6);
        REQUIRE(p.x == 1.0);
        REQUIRE(p.y == 2.0);
        REQUIRE(p.z == 3.0);
        REQUIRE(p.xa == 4.0);
        REQUIRE(p.ya == 5.0);
        REQUIRE(p.za == 6.0);
    }
}

TEST_CASE("Point6D accessors and updates", "[data_structures]") {
    Point6D p(1, 2, 3, 4, 5, 6);
    SECTION("GetDirection returns the requested field") {
        REQUIRE(p.GetDirection(X_DIRECTION) == 1.0);
        REQUIRE(p.GetDirection(Y_DIRECTION) == 2.0);
        REQUIRE(p.GetDirection(Z_DIRECTION) == 3.0);
        REQUIRE(p.GetDirection(XA_DIRECTION) == 4.0);
        REQUIRE(p.GetDirection(YA_DIRECTION) == 5.0);
        REQUIRE(p.GetDirection(ZA_DIRECTION) == 6.0);
    }
    SECTION("UpdateDirection changes only the requested field") {
        p.UpdateDirection(Z_DIRECTION, 99.0);
        REQUIRE(p.z == 99.0);
        REQUIRE(p.x == 1.0);
        REQUIRE(p.y == 2.0);
    }
    SECTION("GetLargestDirection returns the dominant axis") {
        Point6D big_za(1, 2, 3, 4, 5, 60);
        REQUIRE(big_za.GetLargestDirection() == ZA_DIRECTION);
        Point6D big_x(60, 1, 1, 1, 1, 1);
        REQUIRE(big_x.GetLargestDirection() == X_DIRECTION);
    }
}

TEST_CASE("Point6D::GetDistanceFrom is Euclidean in 6D", "[data_structures]") {
    Point6D a(0, 0, 0, 0, 0, 0);
    Point6D b(3, 0, 0, 0, 0, 4);  // 3-4-5 in the x/za plane -> distance 5
    REQUIRE(a.GetDistanceFrom(b) == Approx(5.0));
}

TEST_CASE("HyperBox6D construction and geometry", "[data_structures]") {
    SECTION("size_ is the L2 norm of sides") {
        HyperBox6D box(0.0, Point6D(0, 0, 0, 0, 0, 0), Point6D(2, 2, 0, 0, 0, 0));
        REQUIRE(box.size_ == Approx(std::sqrt(8.0)));  // sqrt(2^2 + 2^2)
        REQUIRE(box.value_ == 0.0);
    }
    SECTION("SetSides recomputes size_") {
        HyperBox6D box;
        box.SetSides(Point6D(3, 0, 0, 0, 0, 0));
        REQUIRE(box.size_ == Approx(3.0));
    }
    SECTION("containsPoint true at center and boundary, false outside") {
        HyperBox6D box(0.0, Point6D(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                       Point6D(1, 1, 1, 1, 1, 1));
        REQUIRE(box.containsPoint(box.GetCenter()));
        REQUIRE(box.containsPoint(Point6D(1, 1, 1, 1, 1, 1)));   // boundary
        REQUIRE_FALSE(box.containsPoint(Point6D(2, 0.5, 0.5, 0.5, 0.5, 0.5)));
    }
    SECTION("TrisectSide divides the chosen side by three") {
        HyperBox6D box(0.0, Point6D(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                       Point6D(1, 1, 1, 1, 1, 1));
        box.TrisectSide(X_DIRECTION);
        REQUIRE(box.GetSides().x == Approx(1.0 / 3.0));
        REQUIRE(box.GetSides().y == Approx(1.0));
        REQUIRE(box.size_ == Approx(std::sqrt(5.0 + (1.0 / 9.0))));
    }
}

TEST_CASE("DirectDataStorage basics", "[data_structures]") {
    SECTION("constructor seeds one unit column with the initial value") {
        DirectDataStorage d(-1.0);
        REQUIRE(d.GetNumberColumns() == 1u);
        REQUIRE(d.GetMinimumHyperboxValue(0) == Approx(-1.0));
    }
    SECTION("AddHyperBox adds a new size-column when the box differs in size") {
        DirectDataStorage d(0.0);
        REQUIRE(d.GetNumberColumns() == 1u);  // seed unit-size column
        // A box with sides (1/3) has a different size (L2 norm) than the unit
        // seed column, so it opens a second column rather than joining the first.
        auto* box = new HyperBox6D(5.0, Point6D(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                   Point6D(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
                                           1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0));
        d.AddHyperBox(box);
        REQUIRE(d.GetNumberColumns() == 2u);
    }
    SECTION("AddHyperBox with the same size joins the existing column") {
        DirectDataStorage d(0.0);
        auto* box = new HyperBox6D(5.0, Point6D(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                   Point6D(1, 1, 1, 1, 1, 1));
        d.AddHyperBox(box);
        REQUIRE(d.GetNumberColumns() == 1u);  // same unit size -> same column
    }
}
