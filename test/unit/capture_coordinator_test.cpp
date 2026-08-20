/* Plan 012 U3 — CaptureCoordinator: process-wide exclusive capture lock + park registry.
 * CUDA-free, headless. Test-first expects capture_coordinator.h with the contract
 * defined in task.
 */
#include <catch2/catch_test_macros.hpp>

#include "compute/capture_coordinator.h"

#include <chrono>

using gpu_cost_function::CaptureCoordinator;

TEST_CASE("park failure is NotSubmitted-like (tryAcquireExclusive returns false)", "[capture_coordinator]") {
    CaptureCoordinator coord;
    coord.registerProducer("vtk", []() { return false; }, []() {});
    REQUIRE_FALSE(coord.tryAcquireExclusive(std::chrono::milliseconds(10)));
    REQUIRE(coord.parkedCount() == 0);
    REQUIRE_FALSE(coord.isOwner());
}

TEST_CASE("all parks succeed -> exclusive owned + all producers parked", "[capture_coordinator]") {
    CaptureCoordinator coord;
    coord.registerProducer("p1", []() { return true; }, []() {});
    coord.registerProducer("p2", []() { return true; }, []() {});
    REQUIRE(coord.tryAcquireExclusive(std::chrono::milliseconds(10)));
    REQUIRE(coord.isOwner());
    REQUIRE(coord.parkedCount() == 2);
    coord.release();
}

TEST_CASE("release unparks and clears ownership", "[capture_coordinator]") {
    CaptureCoordinator coord;
    bool unparked1 = false, unparked2 = false;
    coord.registerProducer("p1", []() { return true; }, [&]() { unparked1 = true; });
    coord.registerProducer("p2", []() { return true; }, [&]() { unparked2 = true; });
    REQUIRE(coord.tryAcquireExclusive(std::chrono::milliseconds(10)));
    REQUIRE(coord.isOwner());
    coord.release();
    REQUIRE_FALSE(coord.isOwner());
    REQUIRE(coord.parkedCount() == 0);
    REQUIRE(unparked1);
    REQUIRE(unparked2);
}

TEST_CASE("producer registered twice by same name is idempotent", "[capture_coordinator]") {
    CaptureCoordinator coord;
    coord.registerProducer("p", []() { return true; }, []() {});
    coord.registerProducer("p", []() { return true; }, []() {});
    REQUIRE(coord.tryAcquireExclusive(std::chrono::milliseconds(10)));
    REQUIRE(coord.parkedCount() == 1);
    coord.release();
    REQUIRE(coord.parkedCount() == 0);
}

TEST_CASE("reentrant acquire on owner thread succeeds", "[capture_coordinator]") {
    CaptureCoordinator coord;
    coord.registerProducer("p1", []() { return true; }, []() {});
    REQUIRE(coord.tryAcquireExclusive(std::chrono::milliseconds(10)));
    REQUIRE(coord.isOwner());
    // Reentrant
    REQUIRE(coord.tryAcquireExclusive(std::chrono::milliseconds(10)));
    REQUIRE(coord.isOwner());
    REQUIRE(coord.parkedCount() == 1);
    coord.release();
    REQUIRE_FALSE(coord.isOwner());
}

TEST_CASE("park failure unparks already-parked producers", "[capture_coordinator]") {
    CaptureCoordinator coord;
    bool unparked_p1 = false;
    coord.registerProducer("p1", []() { return true; }, [&]() { unparked_p1 = true; });
    bool unparked_p2 = false;
    coord.registerProducer("p2", []() { return false; }, [&]() { unparked_p2 = true; });
    REQUIRE_FALSE(coord.tryAcquireExclusive(std::chrono::milliseconds(10)));
    REQUIRE_FALSE(coord.isOwner());
    REQUIRE(coord.parkedCount() == 0);
    // p1 was parked then must be un-parked on the failure path; p2 was never
    // parked (park returned false) so its unpark must NOT run.
    REQUIRE(unparked_p1);
    REQUIRE_FALSE(unparked_p2);
}

TEST_CASE("tryAcquireExclusive without producers succeeds", "[capture_coordinator]") {
    CaptureCoordinator coord;
    REQUIRE(coord.tryAcquireExclusive(std::chrono::milliseconds(10)));
    REQUIRE(coord.isOwner());
    coord.release();
    REQUIRE_FALSE(coord.isOwner());
}
