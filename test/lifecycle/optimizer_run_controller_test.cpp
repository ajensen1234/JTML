// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 006 U5: OptimizerRunController lifecycle tests (R7, R13, R15, R16;
// F1; AE1) — the fake-driver drive-sequence tripwire written FIRST (the
// plan's execution note: gate-first). QtTest with a fake OptimizerRunDriver
// implementing the drive seam: it records Initialize args + drive calls and
// exposes the manager's 7+1 signals for the test to emit, so the
// controller's relays are observed on the controller (test) thread via
// QSignalSpy exactly like the real worker-thread emissions (QTBUG-2842 —
// the controller re-emits by-value).
//
// Scenarios (plan 006 U5 test list):
//  - happy path drive sequence: SaveLastPose mirror (before the gate) ->
//    gate pass -> seed applied -> Initialize args by value -> 8 binds ->
//    start; terminal OptimizedFrame -> Completed;
//  - H2 gate-input rule: previous == current regardless of the mirrors; a
//    frame jump between sync and run still passes with a valid selection;
//  - progress mapping via the UpdateDisplay relay (widgets level math);
//  - H1: re-run rejected while threadActive (incl. the widgets terminal-
//    frame -> thread-death window); stale-epoch relays from run 1 arriving
//    during run 2 are dropped; the re-run opens after the ghost's finished;
//  - M6: Initialize failure — ghost termination observed via the
//    pre-Initialize finished bind (incl. finished emitted DURING
//    Initialize); threadActive clears; re-run opens;
//  - M10a: Initialize failure after the seed was applied -> the pre-seed
//    snapshot is restored; a subsequent run starts from the restored pose;
//  - gate rejections: both statuses surface with the widgets' distinct
//    message texts + Critical severity; state unchanged; seed NOT consumed;
//  - OptimizerError -> terminal OptimizedFrame (error bit): state stays
//    Error, the relay still fires (the widgets unlock-on-relay pin, M8);
//  - H3: controller destroyed mid-run -> cooperative stop + bounded wait,
//    no crash/hang;
//  - cooperative stop: Stopping + the reverse bind fires; the run completes
//    through the normal terminal frame.

#include <QSignalSpy>
#include <QtTest/QtTest>

#include "coordinator/optimizer_run_controller.h"

namespace {

using jta::OptimizerRunControllerCore;

/*The fake driver: implements the drive seam, records Initialize args + the
 * drive calls, and exposes the manager's 7+1 signals for the test to emit
 * (the controller's binds are string-based against the polymorphic manager
 * QObject, so the fake declares the same signals/slot). ThreadActive is
 * driven by Start()/Wait() exactly like the real thread lifecycle.*/
class FakeDriver : public QObject, public jta::OptimizerRunDriver {
    Q_OBJECT
public:
    explicit FakeDriver(int tag) : tag_(tag) {}

    QObject* Manager() override { return this; }
    bool ThreadActive() const override { return thread_active_; }

    bool Initialize(const jta::OptimizerRunLaunch& launch,
                    QString& error_message) override {
        initialize_calls_++;
        last_launch_ = launch;
        error_message = QStringLiteral("simulated Initialize failure");
        /*Ghost pin: when set, the fake posts finished() QUEUED from the top
         * of Initialize (production-shaped: the real manager's ghost emits
         * from its worker thread, so the controller's finished bind — made
         * BEFORE Initialize — observes the termination on the event loop,
         * never re-entrantly inside start()).*/
        if (emit_finished_during_initialize_) {
            QMetaObject::invokeMethod(
                this, "finished", Qt::QueuedConnection);
        }
        return initialize_result_;
    }
    void Start() override {
        thread_active_ = true;
        start_calls_++;
    }
    void Stop() override { stop_calls_++; }
    void Wait() override {
        thread_active_ = false;
        wait_calls_++;
    }

    int tag_ = 0;
    bool initialize_result_ = true;
    bool emit_finished_during_initialize_ = false;
    bool thread_active_ = false;
    int start_calls_ = 0;
    int stop_calls_ = 0;
    int wait_calls_ = 0;
    int initialize_calls_ = 0;
    int stop_slot_calls_ = 0;
    jta::OptimizerRunLaunch last_launch_;

signals:
    void UpdateDisplay(double, int, double, unsigned int);
    void OptimizerError(QString);
    void UpdateOptimum(double, double, double, double, double, double,
                       unsigned int);
    void OptimizedFrame(double, double, double, double, double, double, bool,
                        unsigned int, bool, QString);
    void UpdateDilationBackground();
    void onUpdateOrientationSymTrap(double, double, double, double, double,
                                    double);
    void finished();

public slots:
    /*The reverse-bind target (SLOT(onStopOptimizer()) — the widgets'
     * DirectConnection StopOptimizer analog).*/
    void onStopOptimizer() { stop_slot_calls_++; }
};

/*A runnable request over a real (3 frames x 2 models) LocationStorage. The
 * launch containers are intentionally minimal — the fake records them as
 * passed (the by-value args pin), it never consumes them.*/
struct RunFixture {
    LocationStorage storage;

    RunFixture() {
        storage.LoadNewFrame();
        storage.LoadNewFrame();
        storage.LoadNewFrame();
        storage.LoadNewModel(1.0, 1.0);
        storage.LoadNewModel(1.0, 1.0);
    }

    OptimizerRunRequest MakeRequest(int current_frame = 0) const {
        OptimizerRunRequest req;
        req.directive = OptimizerRunController::Directive::Single;
        /*SaveLastPose mirror (widgets canonical row shape).*/
        req.save_frame = current_frame;
        req.save_rows = {0};
        req.save_pose_source = [](int) { return Point6D(7.0, 0, 0, 0, 0, 0); };
        req.camera_is_a = true;
        req.save_convert_rule = jta::SavePoseConvertRule::NeverConvert;
        /*Gate input.*/
        req.selected_model_rows = {0};
        req.current_frame = current_frame;
        req.frame_count = 3;
        req.model_current_index = 0;
        req.model_count = 2;
        req.pose_frame_count = 3;
        req.pose_model_count = 2;
        req.storage = const_cast<LocationStorage*>(&storage);
        /*Initialize payload: the storage by value + minimal containers (the
         * fake records them).*/
        req.launch.pose_matrix = storage;
        req.launch.settings.trunk_budget = 1000;
        req.launch.settings.branch_budget = 500;
        req.launch.settings.number_branches = 2;
        req.launch.settings.enable_branch_ = true;
        req.launch.settings.leaf_budget = 200;
        req.launch.settings.enable_leaf_ = true;
        req.launch.iter_count = 0;
        return req;
    }
};

/*A rig handing out sequential fakes: the test PREPARES each fake before the
 * controller's start() (the factory is invoked lazily inside start()), so a
 * fake's failure mode is configured before Initialize. Shared ownership:
 * the rig keeps every fake alive for the test's lifetime (emitting stale
 * relays from a finished run's fake is exactly how the epoch + sender guard
 * is exercised — H1), while the controller holds its own reference per
 * run.*/
struct FakeDriverRig {
    std::vector<std::shared_ptr<FakeDriver>> fakes;
    size_t next = 0;

    FakeDriver* Prepare() {
        auto fake =
            std::make_shared<FakeDriver>(static_cast<int>(fakes.size()));
        fakes.push_back(fake);
        return fake.get();
    }
    std::shared_ptr<jta::OptimizerRunDriver> MakeDriver() {
        Q_ASSERT(next < fakes.size());
        return fakes[next++];
    }
};

Point6D P6(double x, double y, double z, double xa, double ya, double za) {
    return Point6D(x, y, z, xa, ya, za);
}

bool SamePose(const Point6D& a, const Point6D& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.xa == b.xa &&
           a.ya == b.ya && a.za == b.za;
}

/*Message recorder (severity-carrying channel; a lambda keeps the enum arg
 * type exact instead of QSignalSpy's QVariant).*/
struct MessageRecorder {
    QStringList titles;
    QStringList texts;
    QList<int> severities;
};

}  // namespace

class OptimizerRunControllerTest : public QObject {
    Q_OBJECT
private slots:
    void HappyPathDriveSequence();
    void SaveLastPoseMirrorRunsBeforeGate();
    void GateRejectionsSurfaceTypedMessages();
    void H2GateInputUsesPreviousEqualsCurrent();
    void ProgressMappingViaUpdateDisplayRelay();
    void CooperativeStopCompletesThroughTerminalFrame();
    void OptimizerErrorThenTerminalFrameKeepsError();
    void GhostInitializeFailureReapsAndReopens();
    void FinishedEmittedDuringInitializeIsObserved();
    void ThreadDeathWindowRerunRejectedThenOpens();
    void StaleEpochRelaysAreDropped();
    void SeedRestoredOnInitializeFailure();
    void DestructorMidRunStopsAndWaits();
    void AllEachTrackedFrameStartsAtZero();
    void SeedReachesLaunchPayload();
    void StoppingStateRerunRejectedWithDistinctMessage();
    void OutOfBoundsTerminalFrameSkipsStorage();
    void MoveNextFrameAdvancesTrackedFrame();
};

void OptimizerRunControllerTest::HappyPathDriveSequence() {
    /*The full drive sequence: save mirror -> gate pass -> seed applied ->
     * Initialize args by value -> 8 binds -> start; terminal OptimizedFrame
     * -> Completed; finished -> driver reaped.*/
    RunFixture f;
    OptimizerRunRequest req = f.MakeRequest();
    FakeDriverRig rig;
    FakeDriver* fake = rig.Prepare();
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    QSignalSpy states(&c, &OptimizerRunController::runStateChanged);
    QSignalSpy relays(&c, &OptimizerRunController::optimizedFrameRelayed);
    QSignalSpy poses(&c, &OptimizerRunController::poseUpdated);
    QSignalSpy dilations(
        &c, &OptimizerRunController::dilationBackgroundRequested);
    QSignalSpy symtraps(&c, &OptimizerRunController::orientationSymTrapUpdated);
    QSignalSpy progress(&c, &OptimizerRunController::progressChanged);
    QSignalSpy seeds(&c, &OptimizerRunController::seedApplied);

    c.setSeedPose(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0, 0);

    QVERIFY(c.start(req));
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Running);
    QVERIFY(c.running());
    QVERIFY(!c.canRun());

    /*Seed applied AFTER the gate (one-shot) — the storage holds the seed.*/
    QCOMPARE(seeds.count(), 1);
    QVERIFY(SamePose(f.storage.GetPose(0, 0), P6(1, 2, 3, 4, 5, 6)));
    QVERIFY(!c.hasSeedPose());

    /*Initialize args by value: the fake recorded the launch the controller
     * forwarded (the gate intent packs primary + current frame; the typed
     * directive maps to the widgets' string).*/
    QCOMPARE(fake->initialize_calls_, 1);
    QCOMPARE(fake->start_calls_, 1);
    QVERIFY(fake->thread_active_);
    QCOMPARE(fake->last_launch_.directive, QStringLiteral("Single"));
    QCOMPARE(fake->last_launch_.primary_model_index, 0u);
    QCOMPARE(fake->last_launch_.current_frame_index, 0u);
    QCOMPARE(fake->last_launch_.iter_count, 0);
    QCOMPARE(fake->last_launch_.selected_model_indexes.size(),
             req.launch.selected_model_indexes.size());  // by-value pass-through
    QCOMPARE(fake->last_launch_.pose_matrix.GetFrameCount(), 3);
    QCOMPARE(fake->last_launch_.camera_a_frames.size(), 0u);  // as built

    /*The 8 binds: 6 relays fire from the fake's emissions (the reverse bind
     * + finished are pinned in the other cases).*/
    emit fake->UpdateDisplay(100.0, 500, -0.5, 0);
    QCOMPARE(progress.count(), 1);
    QCOMPARE(relays.count(), 0);
    emit fake->UpdateOptimum(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0);
    QCOMPARE(poses.count(), 1);
    emit fake->UpdateDilationBackground();
    QCOMPARE(dilations.count(), 1);
    emit fake->onUpdateOrientationSymTrap(0.1, 0.2, 0.3, 0.0, 0.0, 0.0);
    QCOMPARE(symtraps.count(), 1);

    /*Terminal OptimizedFrame -> relay + storage SavePose + Completed.*/
    emit fake->OptimizedFrame(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, false, 0, false,
                              QStringLiteral("Single"));
    QCOMPARE(relays.count(), 1);
    QVERIFY(SamePose(f.storage.GetPose(0, 0), P6(1, 2, 3, 4, 5, 6)));
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Completed);
    QVERIFY(!c.running());
    QVERIFY(c.canRun());
    QCOMPARE(states.count(), 2);  // Running then Completed

    /*finished -> the driver is waited + reaped (a re-run never races it).*/
    emit fake->finished();
    QCOMPARE(fake->wait_calls_, 1);
}

void OptimizerRunControllerTest::SaveLastPoseMirrorRunsBeforeGate() {
    /*Pinned order (mainscreen.cpp:4137 / OptimizerBridge.cpp:114): the
     * SaveLastPose mirror runs BEFORE the gate — a gate REJECTION has
     * already persisted the scene poses.*/
    RunFixture f;
    OptimizerRunRequest req = f.MakeRequest();
    req.selected_model_rows = {};  // gate rejects (no selection)
    req.model_current_index = -1;
    FakeDriverRig rig;
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });

    QVERIFY(!c.start(req));

    /*The mirror wrote (save_frame, save_rows) with the pose-source value.*/
    QVERIFY(SamePose(f.storage.GetPose(0, 0), P6(7, 0, 0, 0, 0, 0)));
    /*No driver was ever created (the gate is before the manager).*/
    QVERIFY(rig.fakes.empty());
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Idle);
}

void OptimizerRunControllerTest::GateRejectionsSurfaceTypedMessages() {
    /*Both gate statuses surface with the widgets' distinct message texts
     * (severity-carrying channel, L14); state unchanged; seed NOT consumed
     * (the seed is taken after the gate).*/
    RunFixture f;
    FakeDriverRig rig;
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    MessageRecorder messages;
    connect(&c, &OptimizerRunController::messageRequested,
            [&messages](const QString& title, const QString& message,
                        OptimizerRunController::Severity severity) {
                messages.titles.push_back(title);
                messages.texts.push_back(message);
                messages.severities.push_back(static_cast<int>(severity));
            });

    c.setSeedPose(9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0);

    /*SelectFrameAndModel.*/
    OptimizerRunRequest req = f.MakeRequest();
    req.selected_model_rows = {};
    req.model_current_index = -1;
    QVERIFY(!c.start(req));
    QCOMPARE(messages.titles.size(), 1);
    QCOMPARE(messages.titles.front(), QStringLiteral("Error!"));
    QCOMPARE(messages.texts.front(),
             QStringLiteral("Select Frame and Model First!"));
    QCOMPARE(messages.severities.front(),
             static_cast<int>(OptimizerRunController::Severity::Critical));
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Idle);
    QVERIFY(rig.fakes.empty());
    QVERIFY(c.hasSeedPose());  // seed NOT consumed on a rejected run

    /*PoseMatrixDimensionMismatch.*/
    req = f.MakeRequest();
    req.pose_frame_count = 2;  // != frame_count
    QVERIFY(!c.start(req));
    QCOMPARE(messages.titles.size(), 2);
    QCOMPARE(messages.titles.back(), QStringLiteral("Critical Error!"));
    QCOMPARE(messages.texts.back(),
             QStringLiteral("Pose Dimension Matrix Differs in Size from "
                            "Frame and Models Loaded! Please Contact "
                            "Support!"));
    QCOMPARE(messages.severities.back(),
             static_cast<int>(OptimizerRunController::Severity::Critical));
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Idle);
    QVERIFY(rig.fakes.empty());
    QVERIFY(c.hasSeedPose());

    /*And the pending seed still applies on the next accepted run (the
     * one-shot survived the rejections).*/
    rig.Prepare();
    req = f.MakeRequest();
    QVERIFY(c.start(req));
    QVERIFY(SamePose(f.storage.GetPose(0, 0), P6(9, 0, 0, 0, 0, 0)));
}

void OptimizerRunControllerTest::H2GateInputUsesPreviousEqualsCurrent() {
    /*H2: the gate Input is assembled with previous == current regardless of
     * the session mirrors — a frame jump between sync and run still passes
     * when the selection is valid, and Initialize receives the run's
     * current frame.*/
    RunFixture f;
    OptimizerRunRequest req = f.MakeRequest(/*current_frame=*/2);
    FakeDriverRig rig;
    FakeDriver* fake = rig.Prepare();
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });

    QVERIFY(c.start(req));
    QCOMPARE(fake->last_launch_.current_frame_index, 2u);
    QCOMPARE(fake->last_launch_.primary_model_index, 0u);
}

void OptimizerRunControllerTest::ProgressMappingViaUpdateDisplayRelay() {
    /*The UpdateDisplay bind: stage/calls/min/progress surface mirrors the
     * widgets onUpdateDisplay arithmetic against the cumulative budgets
     * (trunk 1000 + 2 x 500 branch + 200 leaf = 2200).*/
    RunFixture f;
    OptimizerRunRequest req = f.MakeRequest();
    FakeDriverRig rig;
    FakeDriver* fake = rig.Prepare();
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    QSignalSpy relays(&c, &OptimizerRunController::updateDisplayRelayed);
    QSignalSpy progress(&c, &OptimizerRunController::progressChanged);
    QVERIFY(c.start(req));

    emit fake->UpdateDisplay(100.0, 500, -0.25, 0);
    QCOMPARE(relays.count(), 1);
    QCOMPARE(c.stageText(), QStringLiteral("Trunk"));
    QCOMPARE(c.costCalls(), 500);
    QCOMPARE(c.currentMinimum(), -0.25);
    QCOMPARE(c.progress(), 500.0 / 2200.0);
    QCOMPARE(progress.count(), 1);

    emit fake->UpdateDisplay(100.0, 1001, 0.0, 0);
    QCOMPARE(c.stageText(), QStringLiteral("Branch 1"));
    emit fake->UpdateDisplay(100.0, 1501, 0.0, 0);
    QCOMPARE(c.stageText(), QStringLiteral("Branch 2"));
    emit fake->UpdateDisplay(100.0, 2001, 0.0, 0);
    QCOMPARE(c.stageText(), QStringLiteral("Extra Z-Translation"));

    emit fake->UpdateDisplay(100.0, 2200, -1.0, 0);
    QCOMPARE(c.stageText(), QStringLiteral("Finished"));
    QCOMPARE(c.progress(), 1.0);
    emit fake->UpdateDisplay(100.0, 4400, -1.0, 0);
    QCOMPARE(c.progress(), 1.0);
}

void OptimizerRunControllerTest::CooperativeStopCompletesThroughTerminalFrame() {
    /*stop() -> Stopping + the DirectConnection reverse bind fires (the
     * fake's onStopOptimizer slot); the run completes through the normal
     * terminal-frame path.*/
    RunFixture f;
    OptimizerRunRequest req = f.MakeRequest();
    FakeDriverRig rig;
    FakeDriver* fake = rig.Prepare();
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    QSignalSpy states(&c, &OptimizerRunController::runStateChanged);
    QVERIFY(c.start(req));

    c.stop();
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Stopping);
    QCOMPARE(fake->stop_slot_calls_, 1);  // the reverse bind
    QCOMPARE(states.count(), 2);

    /*A second stop is a no-op (no second reverse-bind emission).*/
    c.stop();
    QCOMPARE(fake->stop_slot_calls_, 1);
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Stopping);

    /*The run completes through the normal terminal-frame path.*/
    emit fake->OptimizedFrame(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0, false,
                              QStringLiteral("Single"));
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Completed);
    QVERIFY(c.canRun());
    emit fake->finished();
}

void OptimizerRunControllerTest::OptimizerErrorThenTerminalFrameKeepsError() {
    /*OptimizerError -> Error + the message channel; the terminal
     * OptimizedFrame (error bit set) keeps the state at Error but the relay
     * STILL fires with the error bit — the pinned widgets unlock-on-relay
     * behavior (M8): the view unlocks at the terminal frame even with the
     * error bit set (the controller side; the view mapping is widgets-side).*/
    RunFixture f;
    OptimizerRunRequest req = f.MakeRequest();
    FakeDriverRig rig;
    FakeDriver* fake = rig.Prepare();
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    QSignalSpy relays(&c, &OptimizerRunController::optimizedFrameRelayed);
    MessageRecorder messages;
    connect(&c, &OptimizerRunController::messageRequested,
            [&messages](const QString& title, const QString& message,
                        OptimizerRunController::Severity severity) {
                messages.titles.push_back(title);
                messages.texts.push_back(message);
                messages.severities.push_back(static_cast<int>(severity));
            });
    QVERIFY(c.start(req));

    emit fake->OptimizerError(QStringLiteral("boom"));
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Error);
    QCOMPARE(messages.titles.size(), 1);
    QCOMPARE(messages.titles.front(), QStringLiteral("Error!"));
    QCOMPARE(messages.texts.front(), QStringLiteral("boom"));
    QCOMPARE(messages.severities.front(),
             static_cast<int>(OptimizerRunController::Severity::Critical));

    emit fake->OptimizedFrame(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, false, 0, true,
                              QStringLiteral("Single"));
    QCOMPARE(relays.count(), 1);
    QCOMPARE(relays.at(0).at(8).toBool(), true);  // error bit on the relay
    /*State stays Error (pinned QML semantics); the pose still persisted.*/
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Error);
    QVERIFY(SamePose(f.storage.GetPose(0, 0), P6(1, 2, 3, 4, 5, 6)));
    QVERIFY(c.canRun());  // the widgets unlock on the relay, not the state
}

void OptimizerRunControllerTest::GhostInitializeFailureReapsAndReopens() {
    /*M6: Initialize failure — the quirk starts the thread (the ghost); its
     * termination is observed via the finished bind placed BEFORE
     * Initialize; threadActive clears; a re-run opens.*/
    RunFixture f;
    FakeDriverRig rig;
    FakeDriver* fake1 = rig.Prepare();
    fake1->initialize_result_ = false;
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    MessageRecorder messages;
    connect(&c, &OptimizerRunController::messageRequested,
            [&messages](const QString& title, const QString& message,
                        OptimizerRunController::Severity) {
                messages.titles.push_back(title);
                messages.texts.push_back(message);
            });

    /*Run 1: Initialize fails -> the quirk starts the ghost thread BEFORE
     * the error message (state Error + the failure text).*/
    QVERIFY(!c.start(f.MakeRequest()));
    QCOMPARE(fake1->start_calls_, 1);  // the quirk's thread start
    QVERIFY(fake1->thread_active_);
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Error);
    QCOMPARE(messages.titles.size(), 1);
    QCOMPARE(messages.titles.front(), QStringLiteral("Error!"));
    QCOMPARE(messages.texts.front(),
             QStringLiteral("simulated Initialize failure"));

    /*While the ghost is alive a re-run is rejected (threadActive).*/
    QVERIFY(!c.start(f.MakeRequest()));
    QCOMPARE(rig.fakes.size(), 1u);

    /*The ghost's finished (observed via the pre-Initialize bind) clears
     * threadActive + reaps the driver; the re-run opens.*/
    emit fake1->finished();
    QCOMPARE(fake1->wait_calls_, 1);
    QVERIFY(!fake1->thread_active_);

    FakeDriver* fake2 = rig.Prepare();
    QVERIFY(c.start(f.MakeRequest()));
    QCOMPARE(fake2->initialize_calls_, 1);
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Running);
}

void OptimizerRunControllerTest::FinishedEmittedDuringInitializeIsObserved() {
    /*M6 pin: the finished bind exists BEFORE Initialize — a ghost posting
     * finished() from the top of Initialize (queued, production-shaped) is
     * observed once the event loop delivers it: the driver is waited,
     * threadActive clears, and the re-run opens. start() returns false
     * with the failure message + Error state.*/
    RunFixture f;
    FakeDriverRig rig;
    FakeDriver* fake1 = rig.Prepare();
    fake1->initialize_result_ = false;
    fake1->emit_finished_during_initialize_ = true;
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    MessageRecorder messages;
    connect(&c, &OptimizerRunController::messageRequested,
            [&messages](const QString& title, const QString& message,
                        OptimizerRunController::Severity) {
                messages.titles.push_back(title);
                messages.texts.push_back(message);
            });

    QVERIFY(!c.start(f.MakeRequest()));
    /*The quirk started the ghost thread before the error box...*/
    QCOMPARE(fake1->start_calls_, 1);
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Error);
    QCOMPARE(messages.titles.size(), 1);
    QCOMPARE(messages.texts.front(),
             QStringLiteral("simulated Initialize failure"));

    /*...and the ghost's queued finished (posted during Initialize, before
     * any other bind could exist) is observed on the event loop: waited,
     * threadActive cleared.*/
    QTRY_VERIFY_WITH_TIMEOUT(fake1->wait_calls_ == 1, 5000);
    QVERIFY(!fake1->thread_active_);

    /*The re-run opens (nothing is alive).*/
    FakeDriver* fake2 = rig.Prepare();
    QVERIFY(c.start(f.MakeRequest()));
    QCOMPARE(fake2->initialize_calls_, 1);
}

void OptimizerRunControllerTest::ThreadDeathWindowRerunRejectedThenOpens() {
    /*The acknowledged widgets delta: a re-run click in the terminal-frame
     * -> thread-death window (EnableAll fired, thread still finishing) is
     * rejected by the Start gate with a message; after the ghost's finished
     * the re-run opens.*/
    RunFixture f;
    FakeDriverRig rig;
    FakeDriver* fake1 = rig.Prepare();
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    MessageRecorder messages;
    connect(&c, &OptimizerRunController::messageRequested,
            [&messages](const QString& title, const QString& message,
                        OptimizerRunController::Severity) {
                messages.titles.push_back(title);
                messages.texts.push_back(message);
            });

    /*Run 1 completes at the terminal frame; the thread is still alive.*/
    QVERIFY(c.start(f.MakeRequest()));
    emit fake1->OptimizedFrame(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0, false,
                               QStringLiteral("Single"));
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Completed);
    QVERIFY(fake1->thread_active_);  // the thread has not died yet

    /*Re-run rejected by the Start gate with a message.*/
    QVERIFY(!c.start(f.MakeRequest()));
    QCOMPARE(rig.fakes.size(), 1u);
    QCOMPARE(messages.titles.back(), QStringLiteral("Warning!"));
    QCOMPARE(messages.texts.back(),
             QStringLiteral("Optimizer is still finishing the previous "
                            "run!"));

    /*The finished relay clears threadActive + reaps the driver; the re-run
     * opens.*/
    emit fake1->finished();
    QCOMPARE(fake1->wait_calls_, 1);
    QVERIFY(!fake1->thread_active_);

    FakeDriver* fake2 = rig.Prepare();
    QVERIFY(c.start(f.MakeRequest()));
    QCOMPARE(rig.fakes.size(), 2u);
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Running);
    QVERIFY(fake2->thread_active_);
}

void OptimizerRunControllerTest::StaleEpochRelaysAreDropped() {
    /*H1: relays from run 1 arriving during run 2 are dropped (sender +
     * epoch guard) — run 1's terminal frame must never corrupt run 2.*/
    RunFixture f;
    FakeDriverRig rig;
    FakeDriver* fake1 = rig.Prepare();
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    QSignalSpy relays(&c, &OptimizerRunController::optimizedFrameRelayed);
    QSignalSpy displays(&c, &OptimizerRunController::updateDisplayRelayed);
    QSignalSpy poses(&c, &OptimizerRunController::poseUpdated);

    /*Run 1: full completion (terminal frame + finished -> reaped).*/
    QVERIFY(c.start(f.MakeRequest()));
    emit fake1->OptimizedFrame(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0, false,
                               QStringLiteral("Single"));
    emit fake1->finished();
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Completed);

    /*Run 2 starts on a fresh driver.*/
    FakeDriver* fake2 = rig.Prepare();
    QVERIFY(c.start(f.MakeRequest()));
    QCOMPARE(rig.fakes.size(), 2u);
    QCOMPARE(relays.count(), 1);  // run 1's terminal frame only

    /*Run 1's stragglers (UpdateDisplay/OptimizedFrame/UpdateOptimum) are
     * dropped — the state + storage stay run 2's.*/
    emit fake1->UpdateDisplay(100.0, 500, 0.0, 0);
    QCOMPARE(displays.count(), 0);
    emit fake1->UpdateOptimum(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0);
    QCOMPARE(poses.count(), 0);
    emit fake1->OptimizedFrame(9.0, 9.0, 9.0, 9.0, 9.0, 9.0, false, 0, false,
                               QStringLiteral("Single"));
    QCOMPARE(relays.count(), 1);
    QVERIFY(!SamePose(f.storage.GetPose(0, 0), P6(9, 9, 9, 9, 9, 9)));
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Running);
    emit fake1->finished();  // stale finished: must not reap run 2's driver
    QCOMPARE(fake2->wait_calls_, 0);

    /*Run 2's own signals still flow.*/
    emit fake2->UpdateDisplay(100.0, 100, 0.0, 0);
    QCOMPARE(displays.count(), 1);
    emit fake2->OptimizedFrame(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0, false,
                               QStringLiteral("Single"));
    QCOMPARE(relays.count(), 2);
    QVERIFY(SamePose(f.storage.GetPose(0, 0), P6(2, 0, 0, 0, 0, 0)));
    emit fake2->finished();
    QCOMPARE(fake2->wait_calls_, 1);
}

void OptimizerRunControllerTest::SeedRestoredOnInitializeFailure() {
    /*M10a: the pre-seed storage snapshot is restored on Initialize failure
     * (the estimate is not silently kept — the storage reverts to the
     * estimate's own direct write, which here equals the SaveLastPose
     * mirror's scene source), and a subsequent run starts from the
     * restored pose.*/
    RunFixture f;
    const Point6D pre_seed = P6(4, 4, 4, 4, 4, 4);  // the estimate's write
    f.storage.SavePose(0, 0, pre_seed);

    FakeDriverRig rig;
    FakeDriver* fake1 = rig.Prepare();
    fake1->initialize_result_ = false;
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    QSignalSpy restored(&c, &OptimizerRunController::seedRestored);
    c.setSeedPose(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0, 0);

    /*The SaveLastPose mirror (pinned before the gate) persists the scene
     * poses — in the real ML flow the scene holds the estimate, so the
     * mirror writes the same value as the estimate's own storage write.*/
    OptimizerRunRequest req = f.MakeRequest();
    req.save_pose_source = [](int) { return P6(4, 4, 4, 4, 4, 4); };

    /*Run 1: Initialize fails AFTER the seed was applied -> the snapshot is
     * restored (seedRestored emitted); the ghost is reaped via its
     * finished.*/
    QVERIFY(!c.start(req));
    QVERIFY(SamePose(f.storage.GetPose(0, 0), pre_seed));
    QCOMPARE(restored.count(), 1);
    QCOMPARE(restored.at(0).at(0).toInt(), 0);
    QCOMPARE(restored.at(0).at(1).toInt(), 0);
    emit fake1->finished();

    /*A subsequent successful run starts from the restored (estimate) pose
     * — with the P1-2 payload refresh the launch matrix is re-copied AFTER
     * the SaveLastPose mirror (the pre-refactor order), so run 2's mirror
     * source (the scene, which holds the estimate) is what Initialize
     * consumes.*/
    FakeDriver* fake2 = rig.Prepare();
    OptimizerRunRequest req2 = f.MakeRequest();
    req2.save_pose_source = [](int) { return P6(4, 4, 4, 4, 4, 4); };
    QVERIFY(c.start(req2));
    QVERIFY(SamePose(
        fake2->last_launch_.pose_matrix.GetPose(0, 0), pre_seed));
}

void OptimizerRunControllerTest::DestructorMidRunStopsAndWaits() {
    /*H3: the controller destroyed mid-run requests the cooperative stop +
     * bounded wait — no crash/hang (the fake records the calls).*/
    RunFixture f;
    FakeDriverRig rig;
    {
        FakeDriver* fake = rig.Prepare();
        OptimizerRunController c([&rig] { return rig.MakeDriver(); });
        QVERIFY(c.start(f.MakeRequest()));
        QVERIFY(fake->thread_active_);
        /*Destroyed mid-run: no terminal frame, no finished.*/
    }
    FakeDriver* fake = rig.fakes.front().get();
    QVERIFY(fake->stop_calls_ >= 1);  // cooperative stop requested
    QVERIFY(fake->wait_calls_ >= 1);  // bounded wait
    QVERIFY(!fake->thread_active_);
}

void OptimizerRunControllerTest::AllEachTrackedFrameStartsAtZero() {
    /*P1-1: for All/Each the manager's frame sequence starts at 0
     * (optimizer_manager start_frame_index_ = 0) and the widgets view
     * reset its selection to 0 BEFORE start() (M11 two-phase), so the
     * controller's tracked frame must start at 0 — NOT the pre-reset
     * current_frame — or the terminal OptimizedFrame results persist at
     * the wrong LocationStorage rows (past-end rows silently dropped).*/
    RunFixture f;
    OptimizerRunRequest req = f.MakeRequest(/*current_frame=*/2);
    req.directive = OptimizerRunController::Directive::All;
    FakeDriverRig rig;
    FakeDriver* fake = rig.Prepare();
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    QSignalSpy relays(&c, &OptimizerRunController::optimizedFrameRelayed);
    QVERIFY(c.start(req));

    /*The manager emits one OptimizedFrame per frame 0..2; each result must
     * persist at the row it was optimized for. The OLD behavior started
     * the tracked frame at 2 — rows 2/3 would hold the results and rows
     * 0/1 would stay at the mirror value (the test then fails).*/
    emit fake->OptimizedFrame(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, true, 0, false,
                              QStringLiteral("All"));
    emit fake->OptimizedFrame(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, true, 0, false,
                              QStringLiteral("All"));
    emit fake->OptimizedFrame(3.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0, false,
                              QStringLiteral("All"));
    QCOMPARE(relays.count(), 3);

    QVERIFY(SamePose(f.storage.GetPose(0, 0), P6(1, 0, 0, 0, 0, 0)));
    QVERIFY(SamePose(f.storage.GetPose(1, 0), P6(2, 0, 0, 0, 0, 0)));
    QVERIFY(SamePose(f.storage.GetPose(2, 0), P6(3, 0, 0, 0, 0, 0)));
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Completed);
    emit fake->finished();
}

void OptimizerRunControllerTest::SeedReachesLaunchPayload() {
    /*P1-2: the seed (and the SaveLastPose mirror) must reach the by-value
     * launch payload the manager's Initialize consumes — the view's
     * snapshot predates start()'s mirror + seed writes. A pre-run storage
     * drift must be overridden by the seed in the PAYLOAD, not only in the
     * live storage.*/
    RunFixture f;
    const Point6D drift = P6(9, 9, 9, 9, 9, 9);  // stale/drifted pose
    f.storage.SavePose(0, 0, drift);
    OptimizerRunRequest req = f.MakeRequest();
    /*Mirror target row 1, so the mirror's value stays distinguishable from
     * the seed's (both must land in the payload).*/
    req.save_frame = 1;
    req.save_rows = {1};
    req.save_pose_source = [](int) { return P6(7, 0, 0, 0, 0, 0); };
    FakeDriverRig rig;
    FakeDriver* fake = rig.Prepare();
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    c.setSeedPose(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0, 0);

    QVERIFY(c.start(req));

    /*The payload reflects the mirror (row 1) AND the seed (row 0) — the
     * old flow's payload predated both (the seed never reached the run).*/
    QVERIFY(SamePose(
        fake->last_launch_.pose_matrix.GetPose(0, 0), P6(1, 2, 3, 4, 5, 6)));
    QVERIFY(SamePose(
        fake->last_launch_.pose_matrix.GetPose(1, 1), P6(7, 0, 0, 0, 0, 0)));
    /*The drift was overridden in the payload (estimate wins over drift).*/
    QVERIFY(!SamePose(
        fake->last_launch_.pose_matrix.GetPose(0, 0), drift));
    /*And the live storage agrees.*/
    QVERIFY(SamePose(f.storage.GetPose(0, 0), P6(1, 2, 3, 4, 5, 6)));
    emit fake->OptimizedFrame(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0, false,
                              QStringLiteral("Single"));
    emit fake->finished();
}

void OptimizerRunControllerTest::StoppingStateRerunRejectedWithDistinctMessage() {
    /*S1: a Run click while the cooperative stop drains (Stopping) is
     * rejected with the DISTINCT still-stopping message — not the generic
     * re-run-rejected text — and no driver is created; once the run
     * completes through the terminal frame, the re-run opens.*/
    RunFixture f;
    FakeDriverRig rig;
    FakeDriver* fake1 = rig.Prepare();
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    MessageRecorder messages;
    connect(&c, &OptimizerRunController::messageRequested,
            [&messages](const QString& title, const QString& message,
                        OptimizerRunController::Severity) {
                messages.titles.push_back(title);
                messages.texts.push_back(message);
            });

    QVERIFY(c.start(f.MakeRequest()));
    c.stop();
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Stopping);

    /*Re-run rejected while stopping: distinct message, no driver.*/
    QVERIFY(!c.start(f.MakeRequest()));
    QCOMPARE(rig.fakes.size(), 1u);
    QCOMPARE(messages.titles.back(), QStringLiteral("Warning!"));
    QCOMPARE(messages.texts.back(),
             QStringLiteral("Optimizer is still stopping..."));

    /*The run completes through the normal terminal-frame path; the
     * re-run then opens.*/
    emit fake1->OptimizedFrame(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0, false,
                               QStringLiteral("Single"));
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Completed);
    emit fake1->finished();
    FakeDriver* fake2 = rig.Prepare();
    QVERIFY(c.start(f.MakeRequest()));
    QCOMPARE(fake2->initialize_calls_, 1);
}

void OptimizerRunControllerTest::OutOfBoundsTerminalFrameSkipsStorage() {
    /*S5: an out-of-bounds terminal OptimizedFrame (primary_model_index
     * past the loaded model count) must not write the storage (the old
     * unconditional SavePose on an out-of-range row was the documented
     * never-occur crash path — LocationStorage drops it, but the skip is
     * the contract), the relay carries the OOB status, and nothing
     * crashes.*/
    RunFixture f;
    OptimizerRunRequest req = f.MakeRequest();
    FakeDriverRig rig;
    FakeDriver* fake = rig.Prepare();
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    QSignalSpy relays(&c, &OptimizerRunController::optimizedFrameRelayed);
    QVERIFY(c.start(req));

    /*The mirror wrote the pre-run pose at (0,0).*/
    QVERIFY(SamePose(f.storage.GetPose(0, 0), P6(7, 0, 0, 0, 0, 0)));

    emit fake->OptimizedFrame(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, false,
                              /*primary_model_index=*/5, false,
                              QStringLiteral("Single"));
    QCOMPARE(relays.count(), 1);
    /*The relay carries the out-of-bounds status (the widgets view boxes).*/
    QCOMPARE(relays.at(0).at(9).toBool(), true);
    /*No storage write (the tracked row keeps the mirror's value).*/
    QVERIFY(SamePose(f.storage.GetPose(0, 0), P6(7, 0, 0, 0, 0, 0)));
    /*Terminal state still advances normally.*/
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Completed);
    emit fake->finished();
}

void OptimizerRunControllerTest::MoveNextFrameAdvancesTrackedFrame() {
    /*S6: the tracked frame advances exactly per the :270-276 condition —
     * move_next_frame AND in-bounds in the directive's direction. Forward
     * (All, tracked starts at 0): advances 0->1->2, blocked at the last
     * frame. Backward (From-style keep of current_frame): 2->1->0,
     * blocked at frame 0. move_next_frame=false never advances.*/
    RunFixture f;
    OptimizerRunRequest req = f.MakeRequest();
    req.directive = OptimizerRunController::Directive::All;
    FakeDriverRig rig;
    FakeDriver* fake = rig.Prepare();
    OptimizerRunController c([&rig] { return rig.MakeDriver(); });
    QSignalSpy relays(&c, &OptimizerRunController::optimizedFrameRelayed);
    QVERIFY(c.start(req));

    /*Forward: advance on move_next_frame, stop at the last frame. The
     * no-advance emission comes LAST at its tracked row — the write
     * always lands at the TRACKED frame, so the final value at each row
     * proves the advance decisions (the blocked last-frame advance and the
     * move_next_frame=false write both stay at the same tracked row).*/
    emit fake->OptimizedFrame(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, true, 0, false,
                              QStringLiteral("All"));
    emit fake->OptimizedFrame(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, true, 0, false,
                              QStringLiteral("All"));
    emit fake->OptimizedFrame(3.0, 0.0, 0.0, 0.0, 0.0, 0.0, true, 0, false,
                              QStringLiteral("All"));  // blocked: last frame
    emit fake->OptimizedFrame(4.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0, false,
                              QStringLiteral("All"));  // no advance
    QCOMPARE(relays.count(), 4);
    /*Row 0 = frame 0 (advance from the first frame); row 1 = frame 1;
     * row 2 = the LAST emission — both the blocked last-frame advance
     * (3.0) and the no-advance write (4.0) landed at tracked frame 2 (a
     * buggy advance would have written row 3, which LocationStorage
     * silently drops — the assertion would fail).*/
    QVERIFY(SamePose(f.storage.GetPose(0, 0), P6(1, 0, 0, 0, 0, 0)));
    QVERIFY(SamePose(f.storage.GetPose(1, 0), P6(2, 0, 0, 0, 0, 0)));
    QVERIFY(SamePose(f.storage.GetPose(2, 0), P6(4, 0, 0, 0, 0, 0)));
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Completed);
    emit fake->finished();

    /*Backward: advances 2->1->0, blocked at frame 0.*/
    OptimizerRunRequest req_b = f.MakeRequest(/*current_frame=*/2);
    req_b.directive = OptimizerRunController::Directive::Backward;
    FakeDriver* fake_b = rig.Prepare();
    QVERIFY(c.start(req_b));
    emit fake_b->OptimizedFrame(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, true, 0, false,
                                QStringLiteral("Backward"));
    emit fake_b->OptimizedFrame(11.0, 0.0, 0.0, 0.0, 0.0, 0.0, true, 0, false,
                                QStringLiteral("Backward"));
    emit fake_b->OptimizedFrame(12.0, 0.0, 0.0, 0.0, 0.0, 0.0, true, 0, false,
                                QStringLiteral("Backward"));  // blocked at 0
    emit fake_b->OptimizedFrame(13.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0, false,
                                QStringLiteral("Backward"));  // no advance
    QCOMPARE(relays.count(), 8);
    QVERIFY(SamePose(f.storage.GetPose(2, 0), P6(10, 0, 0, 0, 0, 0)));
    QVERIFY(SamePose(f.storage.GetPose(1, 0), P6(11, 0, 0, 0, 0, 0)));
    QVERIFY(SamePose(f.storage.GetPose(0, 0), P6(13, 0, 0, 0, 0, 0)));
    QCOMPARE(c.runState(), OptimizerRunController::RunState::Completed);
    emit fake_b->finished();
}

QTEST_GUILESS_MAIN(OptimizerRunControllerTest)
#include "optimizer_run_controller_test.moc"
