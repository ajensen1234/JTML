// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 006 U6 lifecycle tests: SessionStateController (R5/R6/R10, AE2).
//
// QtTest over the QObject notification shell — scenarios (a)-(d):
//  (a) dataset load -> datasetChanged; selection change -> selectionChanged
//      with the correct rows/primary/mirrors;
//  (b) no actual value change -> no signal (diff emission; repeated syncs
//      don't spam);
//  (c) mirrors consistent at emission time (M9/H2) — the state's mirrors
//      are advanced before the emit (previous == current), and the signal
//      payload's previous values are the PRE-CHANGE selection;
//  (d) ResetForDatasetClear -> mirrors reset to -1/empty, seed cleared,
//      datasetChanged emitted; a pending deferred emission is dropped.
// Plus the M7 run-in-flight probe and the selection-only signal split.
//
// All emissions are same-thread direct connections (no worker thread), so
// QSignalSpy observes the controller directly (QTBUG-2842 does not apply).

#include <QSignalSpy>
#include <QtTest/QtTest>

#include "coordinator/session_state_controller.h"

namespace {

std::vector<int> Rows(const QVariant& v) {
    return v.value<std::vector<int>>();
}

}  // namespace

class SessionStateControllerTest : public QObject {
    Q_OBJECT
private slots:
    void initTestCase();
    void DatasetLoadEmitsDatasetChanged();
    void SelectionChangeEmitsSelectionChangedWithRowsPrimaryAndMirrors();
    void SelectionOnlyChangeSkipsDatasetSignal();
    void NoActualChangeEmitsNothing();
    void MirrorsConsistentAtEmissionTime();
    void ResetForDatasetClearResetsMirrorsClearsSeedAndEmits();
    void ResetForDatasetClearDropsPendingDeferredEmission();
    void RunInFlightProbeReflectsInjectedSource();
};

void SessionStateControllerTest::initTestCase() {
    /*QSignalSpy records the std::vector<int> payloads as QVariants — the
     * metatype must be registered up front.*/
    qRegisterMetaType<std::vector<int>>("std::vector<int>");
}

void SessionStateControllerTest::DatasetLoadEmitsDatasetChanged() {
    jta::SessionState state;
    SessionStateController ctrl(&state);
    QSignalSpy dataset(&ctrl, &SessionStateController::datasetChanged);
    QSignalSpy selection(&ctrl, &SessionStateController::selectionChanged);

    /*Dataset load: counts change -> datasetChanged fires from the sync
     * (dataset facts are independent of the selection mirrors); the
     * selection signal is DEFERRED to CommitSelection (M9).*/
    ctrl.UpdateSession(10, 2, 0, {0, 1});
    QCOMPARE(dataset.count(), 1);
    QCOMPARE(selection.count(), 0);
    QCOMPARE(state.GetFrameCount(), 10);
    QCOMPARE(state.GetModelCount(), 2);

    ctrl.CommitSelection();
    QCOMPARE(selection.count(), 1);
    QCOMPARE(state.GetCurrentFrame(), 0);
    QCOMPARE(state.GetSelectedModels(), std::vector<int>({0, 1}));
}

void SessionStateControllerTest::SelectionChangeEmitsSelectionChangedWithRowsPrimaryAndMirrors() {
    jta::SessionState state;
    SessionStateController ctrl(&state);
    QSignalSpy selection(&ctrl, &SessionStateController::selectionChanged);

    /*Steady state first: frame 3, rows {0, 1} (primary = first = 0).*/
    ctrl.UpdateSession(10, 5, 3, {1, 0});  // unsorted input normalizes
    ctrl.CommitSelection();
    selection.clear();

    /*Selection change: frame 5, rows {0, 2}.*/
    ctrl.UpdateSession(10, 5, 5, {2, 0});
    ctrl.CommitSelection();

    QCOMPARE(selection.count(), 1);
    const auto args = selection.at(0);
    QCOMPARE(args.at(0).toInt(), 5);                      // current frame
    QCOMPARE(args.at(1).toInt(), 0);                      // primary = first row
    QVERIFY(Rows(args.at(2)) == std::vector<int>({0, 2}));  // rows (sorted)
    QCOMPARE(args.at(3).toInt(), 3);                      // previous frame
    QVERIFY(Rows(args.at(4)) == std::vector<int>({0, 1}));  // previous rows
    /*State after commit: mirrors == current (H2 steady state).*/
    QCOMPARE(state.GetPreviousFrame(), 5);
    QVERIFY(state.GetPreviousModelRows() == std::vector<int>({0, 2}));
}

void SessionStateControllerTest::SelectionOnlyChangeSkipsDatasetSignal() {
    jta::SessionState state;
    SessionStateController ctrl(&state);
    QSignalSpy dataset(&ctrl, &SessionStateController::datasetChanged);
    QSignalSpy selection(&ctrl, &SessionStateController::selectionChanged);

    ctrl.UpdateSession(10, 5, 3, {0});
    ctrl.CommitSelection();
    dataset.clear();
    selection.clear();

    /*Counts unchanged: a selection change must NOT raise datasetChanged.*/
    ctrl.UpdateSession(10, 5, 4, {1});
    QCOMPARE(dataset.count(), 0);
    ctrl.CommitSelection();
    QCOMPARE(selection.count(), 1);
}

void SessionStateControllerTest::NoActualChangeEmitsNothing() {
    jta::SessionState state;
    SessionStateController ctrl(&state);
    QSignalSpy dataset(&ctrl, &SessionStateController::datasetChanged);
    QSignalSpy selection(&ctrl, &SessionStateController::selectionChanged);

    ctrl.UpdateSession(10, 2, 3, {1});
    ctrl.CommitSelection();
    QCOMPARE(dataset.count(), 1);
    QCOMPARE(selection.count(), 1);

    /*Repeated syncs of identical facts: silent (diff emission, M9).*/
    ctrl.UpdateSession(10, 2, 3, {1});
    ctrl.CommitSelection();
    ctrl.UpdateSession(10, 2, 3, {1});
    ctrl.CommitSelection();
    QCOMPARE(dataset.count(), 1);
    QCOMPARE(selection.count(), 1);

    /*Raw-but-normalizing input (out-of-range rows) maps to the same stored
     * values -> still silent.*/
    ctrl.UpdateSession(10, 2, 3, {1, 7, -1});
    ctrl.CommitSelection();
    QCOMPARE(dataset.count(), 1);
    QCOMPARE(selection.count(), 1);
    QVERIFY(state.GetSelectedModels() == std::vector<int>({1}));
}

void SessionStateControllerTest::MirrorsConsistentAtEmissionTime() {
    jta::SessionState state;
    SessionStateController ctrl(&state);
    QSignalSpy selection(&ctrl, &SessionStateController::selectionChanged);

    /*Steady state: frame 3, rows {0, 1}.*/
    ctrl.UpdateSession(10, 5, 3, {0, 1});
    ctrl.CommitSelection();
    selection.clear();
    QCOMPARE(state.GetPreviousFrame(), 3);

    /*Observe the state AT emission: the mirrors must already be advanced
     * (M9 — an emit inside the sync would notify with stale previous).*/
    int observed_previous_frame = -999;
    std::vector<int> observed_previous_rows;
    QObject::connect(
        &ctrl,
        &SessionStateController::selectionChanged,
        [&](int, int, const std::vector<int>&, int,
            const std::vector<int>&) {
            observed_previous_frame = state.GetPreviousFrame();
            observed_previous_rows = state.GetPreviousModelRows();
        });

    ctrl.UpdateSession(10, 5, 5, {0, 2});
    ctrl.CommitSelection();

    /*State at emission: previous == current (consistent, H2).*/
    QCOMPARE(observed_previous_frame, 5);
    QVERIFY(observed_previous_rows == std::vector<int>({0, 2}));
    /*Payload: previous == the PRE-CHANGE current (what save-last-pose would
     * have read).*/
    QCOMPARE(selection.count(), 1);
    const auto args = selection.at(0);
    QCOMPARE(args.at(3).toInt(), 3);
    QVERIFY(Rows(args.at(4)) == std::vector<int>({0, 1}));
}

void SessionStateControllerTest::ResetForDatasetClearResetsMirrorsClearsSeedAndEmits() {
    jta::SessionState state;
    bool seed_cleared = false;
    SessionStateController ctrl(&state, {}, [&] { seed_cleared = true; });
    QSignalSpy dataset(&ctrl, &SessionStateController::datasetChanged);

    ctrl.UpdateSession(10, 5, 3, {0, 1});
    ctrl.CommitSelection();
    QVERIFY(state.HasPreviousSelection());
    dataset.clear();

    ctrl.ResetForDatasetClear();
    QCOMPARE(state.GetPreviousFrame(), -1);
    QVERIFY(state.GetPreviousModelRows().empty());
    QVERIFY(!state.HasPreviousSelection());
    QVERIFY(seed_cleared);  // the optimizer's pending seed is dropped (H5/M10b)
    QCOMPARE(dataset.count(), 1);
}

void SessionStateControllerTest::ResetForDatasetClearDropsPendingDeferredEmission() {
    jta::SessionState state;
    SessionStateController ctrl(&state);
    QSignalSpy selection(&ctrl, &SessionStateController::selectionChanged);

    /*A deferred selection change pending from the sync names the wiped
     * dataset — ResetForDatasetClear must drop it, not emit it.*/
    ctrl.UpdateSession(10, 5, 3, {0, 1});
    QCOMPARE(selection.count(), 0);
    ctrl.ResetForDatasetClear();
    ctrl.CommitSelection();
    QCOMPARE(selection.count(), 0);
    QCOMPARE(state.GetPreviousFrame(), -1);
}

void SessionStateControllerTest::RunInFlightProbeReflectsInjectedSource() {
    jta::SessionState state;
    bool in_flight = false;
    SessionStateController ctrl(&state, [&] { return in_flight; });
    QVERIFY(!ctrl.runInFlight());
    in_flight = true;
    QVERIFY(ctrl.runInFlight());
    in_flight = false;
    QVERIFY(!ctrl.runInFlight());
}

QTEST_GUILESS_MAIN(SessionStateControllerTest)
#include "session_state_controller_test.moc"
