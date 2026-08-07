// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Headless lifecycle tests for OptimizeCoordinator (plan U4, R4/R5/R7, AE1).
//
// These run a real QThread + worker with an injected stub cost (zero GPU), and
// observe the coordinator's signals (on the test/main thread) via QSignalSpy,
// exactly the seam the "hangs after optimizer finishes" regressions live in.

#include <QSignalSpy>
#include <QtTest/QtTest>
#include <stdexcept>

#include "core/optimize_coordinator.h"

namespace {

// Deterministic stub cost: a convex quadratic centered at the origin.
double StubCost(const Point6D& p) {
    return p.x * p.x + p.y * p.y + p.z * p.z + p.xa * p.xa + p.ya * p.ya +
           p.za * p.za;
}

Point6D Range(double r) {
    return Point6D(r, r, r, r, r, r);
}

Point6D Zero() {
    return Point6D(0, 0, 0, 0, 0, 0);
}

bool IsIdle(const OptimizeCoordinator& c) {
    return c.GetState() == OptimizeCoordinator::State::Idle;
}

}  // namespace

class CoordinatorTest : public QObject {
    Q_OBJECT
private slots:
    void TransitionsThroughRunAndBackToIdle();
    void RefusesDoubleStartWhileRunning();
    void CooperativeStopReturnsToIdle();
    void ThrowingCostEmitsErrorAndRecovers();
    void EmitsRunningThenIdleStateChanged();
};

void CoordinatorTest::TransitionsThroughRunAndBackToIdle() {
    OptimizeCoordinator c(StubCost, Range(10), Zero(), 1000);
    QSignalSpy finished(&c, &OptimizeCoordinator::Finished);
    QVERIFY(IsIdle(c));
    QVERIFY(c.Start());
    QVERIFY(c.GetState() == OptimizeCoordinator::State::Running);
    QVERIFY(finished.wait(5000));  // pumps a nested event loop; fails on hang
    QVERIFY(IsIdle(c));

    // Re-launchable after completion:
    QVERIFY(c.Start());
    QVERIFY(finished.wait(5000));
    QVERIFY(IsIdle(c));
}

void CoordinatorTest::RefusesDoubleStartWhileRunning() {
    OptimizeCoordinator c(StubCost, Range(10), Zero(), 10000);
    QSignalSpy finished(&c, &OptimizeCoordinator::Finished);
    QVERIFY(c.Start());
    QVERIFY(c.GetState() == OptimizeCoordinator::State::Running);
    QVERIFY(!c.Start());  // double-start refused
    QVERIFY(finished.wait(5000));  // let the run complete / clean up
    QVERIFY(IsIdle(c));
}

void CoordinatorTest::CooperativeStopReturnsToIdle() {
    OptimizeCoordinator c(StubCost, Range(10), Zero(), 1000000u);
    QSignalSpy finished(&c, &OptimizeCoordinator::Finished);
    QVERIFY(c.Start());
    c.Stop();  // cooperative; checked between DIRECT iterations
    QVERIFY(finished.wait(5000));
    QVERIFY(IsIdle(c));
}

void CoordinatorTest::ThrowingCostEmitsErrorAndRecovers() {
    auto throwing_cost = [](const Point6D&) -> double {
        throw std::runtime_error("simulated init/cost failure");
    };
    OptimizeCoordinator c(throwing_cost, Range(10), Zero(), 1000);
    QSignalSpy errors(&c, &OptimizeCoordinator::ErrorOccurred);
    QSignalSpy finished(&c, &OptimizeCoordinator::Finished);
    QVERIFY(c.Start());
    QVERIFY(finished.wait(5000));
    QCOMPARE(errors.count(), 1);      // error surfaced to the caller
    QCOMPARE(finished.count(), 1);
    QVERIFY(IsIdle(c));               // back to a re-launchable state, no hang

    // A fresh coordinator with a good cost runs cleanly after the failure path.
    OptimizeCoordinator good(StubCost, Range(10), Zero(), 1000);
    QSignalSpy good_finished(&good, &OptimizeCoordinator::Finished);
    QVERIFY(good.Start());
    QVERIFY(good_finished.wait(5000));
    QVERIFY(IsIdle(good));
}

void CoordinatorTest::EmitsRunningThenIdleStateChanged() {
    OptimizeCoordinator c(StubCost, Range(10), Zero(), 1000);
    QSignalSpy states(&c, &OptimizeCoordinator::StateChanged);
    QSignalSpy finished(&c, &OptimizeCoordinator::Finished);
    QVERIFY(c.Start());
    QVERIFY(finished.wait(5000));
    // Running (1) then Idle (0).
    QCOMPARE(states.count(), 2);
    QCOMPARE(states.at(0).at(0).toInt(),
             static_cast<int>(OptimizeCoordinator::State::Running));
    QCOMPARE(states.at(1).at(0).toInt(),
             static_cast<int>(OptimizeCoordinator::State::Idle));
}

QTEST_GUILESS_MAIN(CoordinatorTest)
#include "coordinator_test.moc"
