// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 005 U8: PoseBridge pins (R9, R10). Deterministic Catch2 headless
// tests that direct-compile the bridge (PoseBridge + AppBridge/StudyBridge/
// SettingsBridge/OptimizerBridge/MlBridge/DelegateSelection/
// ExperimentalScene) and link jtml_coordinator for the REAL seam symbols
// (LocationStorage, pose_copy, pose_file_io). No Qt event loop, no GPU —
// the synthetic dataset (storage + list models driven directly) keeps the
// suite fast, and the selection contract is driven through StudyBridge's
// public delegate API.
//
// Pins (plan 005 U8 test scenarios a-e, review-fixed semantics):
//  - (a) edit a cell -> immediate SavePose -> read back shows the edited
//    value (single-axis read-modify-write, dirty set, table role re-reads);
//  - (b) copy-prev at frame 0 / copy-next at the last frame behave exactly
//    like the pinned pose_copy semantics (the bridge delegates: raw-index
//    GetPose/SavePose chain, no-image default fallback overwriting the
//    boundary frame; multi-select writes the PRIMARY row — the v1
//    single-model mapping);
//  - (c) non-numeric / NaN / infinite input is rejected with the inline
//    validation message, stored state unchanged;
//  - (d) WritePoseFile / WriteKinematicsFile false returns (unwritable
//    path) surface a message and keep the in-memory state + dirty flag;
//  - (e) save pose file -> reload -> identical (round-trip via the bridge,
//    plus the real golden kinematics fixture test/golden/
//    fem_oracle_captured.jtak read through loadKinematics);
//  - guard + selection semantics: no frame/model -> typed message, nothing
//    changes; successful saves clear the dirty flag.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <QFile>
#include <QFileInfo>
#include <QSettings>
#include <QTemporaryDir>

#include "AppBridge.h"
#include "ExperimentalScene.h"
#include "ExperimentalSession.h"
#include "PoseBridge.h"
#include "StudyBridge.h"
#include "domain/data_structures_6D.h"
#include "services/settings_service.h"
#include "view/frame_list_model.h"
#include "view/model_list_model.h"

using Catch::Approx;

namespace {

/*Records messageRequested emissions (the single QML Dialog analog).*/
struct MessageRecorder {
    QStringList titles;
    QStringList texts;
};

void connect_messages(PoseBridge* bridge, MessageRecorder* recorder) {
    QObject::connect(
        bridge, &PoseBridge::messageRequested,
        [recorder](const QString& title, const QString& message) {
            recorder->titles.push_back(title);
            recorder->texts.push_back(message);
        });
}

bool SamePose(const Point6D& a, const Point6D& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.xa == b.xa &&
           a.ya == b.ya && a.za == b.za;
}

/*The U8 fixture: a hub wired to an app-owned scene with an ini-backed
 * SettingsService (isolated — the real registry is never touched by headless
 * tests) and a SYNTHETIC dataset (2 models x 3 frames): the LocationStorage
 * matrix is built directly (LoadNewModel/LoadNewFrame — the same default
 * poses the real load path produces) and the frame LIST model is populated
 * through its public AppendFrame so StudyBridge's frameCount/selection
 * mirrors agree with the storage (the bridge reads the plan's frame count
 * from the list model, like the widgets' rowCount). No images/STLs are
 * parsed — the pose surface never touches Frame/Model contents.*/

struct PoseFixture {
    QTemporaryDir dir;
    jta::SettingsService settings_service{
        dir.filePath("settings.ini"), QSettings::IniFormat};
    ExperimentalScene scene;
    AppBridge hub{&scene, &settings_service};
    MessageRecorder messages;

    PoseFixture() {
        REQUIRE(dir.isValid());
        connect_messages(hub.poseBridge(), &messages);

        /*Dataset: 2 models x 3 frames (default pose z = -2500 with
         * principal_distance 1000 / pixel_pitch 0.1).*/
        ExperimentalSession* session = hub.session();
        session->model_locations.LoadNewModel(1000.0, 0.1);
        session->model_locations.LoadNewModel(1000.0, 0.1);
        session->model_locations.LoadNewFrame();
        session->model_locations.LoadNewFrame();
        session->model_locations.LoadNewFrame();
        /*Frame list rows so StudyBridge's frameCount matches the storage,
         * and model list rows so the selection contract accepts model
         * rows (toggleModelSelected ignores out-of-range rows).*/
        auto* frame_list =
            static_cast<FrameListModel*>(hub.studyBridge()->frameListModel());
        frame_list->AppendFrame(QStringLiteral("F0"));
        frame_list->AppendFrame(QStringLiteral("F1"));
        frame_list->AppendFrame(QStringLiteral("F2"));
        auto* model_list =
            static_cast<ModelListModel*>(hub.studyBridge()->modelListModel());
        model_list->AppendModels({QStringLiteral("M0"),
                                  QStringLiteral("M1")});
        /*The app's SyncSessionState tail mirrors the counts into the
         * session state.*/
        hub.studyBridge()->setCurrentFrame(0);
        hub.studyBridge()->toggleModelSelected(0);

        /*Two scene models (the render seam's view of the dataset) so the
         * scene-pose sync tail is observable.*/
        scene.setModels({
            SceneModel{"a.stl", "A", Point6D()},
            SceneModel{"b.stl", "B", Point6D()},
        });
    }

    StudyBridge* study() { return hub.studyBridge(); }
    ExperimentalSession* session() { return hub.session(); }
    PoseBridge* poses() { return hub.poseBridge(); }
    OptimizerBridge* optimizer() { return hub.optimizerBridge(); }
    QAbstractItemModel* table() {
        return static_cast<QAbstractItemModel*>(
            hub.poseBridge()->tableModel());
    }

    Point6D stored(int frame, int model) {
        return session()->model_locations.GetPose(frame, model);
    }
};

}  // namespace

TEST_CASE("pose_bridge: edit a cell -> immediate SavePose -> read back "
          "shows the edited value (scenario a)",
          "[pose_bridge]") {
    PoseFixture f;

    /*Single-axis read-modify-write: only Z (axis 2) of (frame 1, model 0)
     * changes; the other five axes stay at the default.*/
    REQUIRE(f.poses()->setPoseValue(1, 0, 2, QStringLiteral("12.5")));

    const Point6D pose = f.stored(1, 0);
    REQUIRE(pose.z == Approx(12.5));
    REQUIRE(pose.x == 0.0);
    REQUIRE(pose.y == 0.0);
    REQUIRE(pose.xa == 0.0);
    REQUIRE(pose.ya == 0.0);
    REQUIRE(pose.za == 0.0);
    /*The untouched cells keep their default poses.*/
    REQUIRE(f.stored(0, 0).z == Approx(-2500.0));

    /*Dirty set by the edit; the validation message stays clear.*/
    REQUIRE(f.poses()->dirty());
    REQUIRE(f.poses()->validationMessage().isEmpty());
    /*The table model re-reads the stored value (the QML binding surface).*/
    REQUIRE(f.poses()->poseValue(1, 0, 2) == Approx(12.5));

    /*The scene sync targets the CURRENT frame's cell only: frame 1 is not
     * the current frame (frame 0 is), so the scene pose is untouched; an
     * edit on the current frame lands in the scene (the widgets
     * viewport-update tail).*/
    REQUIRE(f.scene.models()[0].pose.z == 0.0);
    REQUIRE(f.poses()->setPoseValue(0, 0, 2, QStringLiteral("-4.0")));
    REQUIRE(f.scene.models()[0].pose.z == Approx(-4.0));
    REQUIRE(f.scene.models()[1].pose.z == 0.0);
}

TEST_CASE("pose_bridge: copy-prev at frame 0 / copy-next at the last frame "
          "match the pinned pose_copy semantics (scenario b)",
          "[pose_bridge]") {
    PoseFixture f;

    /*Frame 0 holds a real (non-default) pose; copy-previous onto it must
     * replace it with the no-image DEFAULT pose, not with frame 0's own
     * pose (the pinned boundary fallback: GetPose(-1, ...) resolves to the
     * model's initial pose, stored verbatim).*/
    f.poses()->setPoseValue(0, 0, 0, QStringLiteral("9.0"));
    REQUIRE(f.stored(0, 0).x == Approx(9.0));
    const Point6D default_pose = f.stored(-1, 0);
    REQUIRE(default_pose.z == Approx(-2500.0));

    f.poses()->copyPrevious();
    REQUIRE(SamePose(f.stored(0, 0), default_pose));
    REQUIRE(f.stored(1, 0).z == Approx(-2500.0));  // only frame 0 touched

    /*Copy-next at the LAST frame: the read is GetPose(3, ...) — the no-
     * image default, overwriting frame 2.*/
    f.poses()->setPoseValue(2, 0, 1, QStringLiteral("7.0"));
    REQUIRE(f.stored(2, 0).y == Approx(7.0));
    f.study()->setCurrentFrame(2);

    f.poses()->copyNext();
    REQUIRE(SamePose(f.stored(2, 0), f.stored(-1, 0)));

    /*Interior copy-next moves the next frame's pose onto the current
     * frame (the raw chain at non-boundary indices).*/
    f.poses()->setPoseValue(1, 0, 3, QStringLiteral("45.0"));
    f.study()->setCurrentFrame(0);
    f.poses()->copyNext();
    REQUIRE(f.stored(0, 0).xa == Approx(45.0));
}

TEST_CASE("pose_bridge: multi-select copy writes the PRIMARY row (v1 "
          "single-model mapping)",
          "[pose_bridge]") {
    PoseFixture f;

    /*v1 pose ops are primary-model-only: with models 0 and 1 selected, the
     * copy writes the primary row (0) — the widgets' current-row slot has no
     * QML analog. The seam still receives both indices (pinned in
     * pose_copy_test.cpp); the bridge passes the primary for both.*/
    f.study()->toggleModelSelected(1);
    REQUIRE(f.study()->primaryModelIndex() == 0);
    REQUIRE(f.study()->selectedModelCount() == 2);

    f.poses()->setPoseValue(0, 0, 4, QStringLiteral("30.0"));
    f.study()->setCurrentFrame(1);
    f.poses()->copyPrevious();

    REQUIRE(f.stored(1, 0).ya == Approx(30.0));   // write at primary
    REQUIRE(f.stored(1, 1).ya == 0.0);            // model 1 untouched
    REQUIRE(f.stored(0, 0).ya == Approx(30.0));   // read at primary
}

TEST_CASE("pose_bridge: non-numeric / NaN / infinite input rejected, state "
          "unchanged (scenario c)",
          "[pose_bridge]") {
    PoseFixture f;

    const Point6D before = f.stored(0, 0);

    /*Non-numeric.*/
    REQUIRE_FALSE(
        f.poses()->setPoseValue(0, 0, 0, QStringLiteral("abc")));
    /*The inline validation message surface carried the failure.*/
    REQUIRE_FALSE(f.poses()->validationMessage().isEmpty());
    REQUIRE(
        f.poses()->validationMessage().startsWith(
            QStringLiteral("Invalid pose value")));
    /*NaN / infinite spellings (Qt's toDouble accepts them — the finite
     * check must close the hole).*/
    REQUIRE_FALSE(
        f.poses()->setPoseValue(0, 0, 0, QStringLiteral("nan")));
    REQUIRE_FALSE(
        f.poses()->setPoseValue(0, 0, 0, QStringLiteral("inf")));
    REQUIRE_FALSE(
        f.poses()->setPoseValue(0, 0, 0, QStringLiteral("-inf")));
    /*Empty string.*/
    REQUIRE_FALSE(f.poses()->setPoseValue(0, 0, 0, QStringLiteral("")));
    /*Out-of-range cells.*/
    REQUIRE_FALSE(f.poses()->setPoseValue(0, 0, 6, QStringLiteral("1.0")));
    REQUIRE_FALSE(f.poses()->setPoseValue(3, 0, 0, QStringLiteral("1.0")));
    REQUIRE_FALSE(f.poses()->setPoseValue(0, 2, 0, QStringLiteral("1.0")));

    /*Every rejection left the storage untouched and set no dirty flag.*/
    REQUIRE(SamePose(f.stored(0, 0), before));
    REQUIRE_FALSE(f.poses()->dirty());

    /*A successful commit clears the message.*/
    REQUIRE(f.poses()->setPoseValue(0, 0, 0, QStringLiteral("1.5")));
    REQUIRE(f.poses()->validationMessage().isEmpty());
}

TEST_CASE("pose_bridge: save/load guard — no frame or model -> typed "
          "message, nothing changes",
          "[pose_bridge]") {
    PoseFixture f;
    f.study()->clearModelSelection();
    f.study()->setCurrentFrame(-1);

    const int messages_before = f.messages.titles.size();
    f.poses()->copyPrevious();
    f.poses()->copyNext();
    f.poses()->savePoseFile(QStringLiteral("/tmp/pose.jtap"));
    f.poses()->loadPoseFile(QStringLiteral("/tmp/pose.jtap"));
    f.poses()->saveKinematics(QStringLiteral("/tmp/kin.jtak"));
    f.poses()->loadKinematics(QStringLiteral("/tmp/kin.jtak"));

    REQUIRE(f.messages.titles.size() == messages_before + 6);
    /*The copy + kinematics slots use the copy slots' message; the pose
     * save/load slots use theirs (widgets strings byte-identical). Call
     * order: copyPrevious, copyNext, savePoseFile, loadPoseFile,
     * saveKinematics, loadKinematics.*/
    for (int i : {0, 1, 4, 5}) {
        REQUIRE(f.messages.titles[messages_before + i] ==
                QStringLiteral("Error!"));
        REQUIRE(
            f.messages.texts[messages_before + i] ==
            QStringLiteral("Select Model and Load Frames First!"));
    }
    for (int i : {2, 3}) {
        REQUIRE(f.messages.titles[messages_before + i] ==
                QStringLiteral("Error!"));
        REQUIRE(
            f.messages.texts[messages_before + i] ==
            QStringLiteral("Select Frame and Model First!"));
    }
    REQUIRE_FALSE(f.poses()->dirty());
    /*No file was written (the guard rejects before any seam call).*/
    REQUIRE(!QFileInfo::exists(f.dir.filePath("pose.jtap")));
}

TEST_CASE("pose_bridge: WritePoseFile / WriteKinematicsFile false return -> "
          "message + in-memory state kept (scenario d)",
          "[pose_bridge]") {
    PoseFixture f;

    /*Seed an edit so the dirty flag is on.*/
    REQUIRE(f.poses()->setPoseValue(1, 0, 2, QStringLiteral("12.5")));
    REQUIRE(f.poses()->dirty());

    /*An unwritable path (empty string — ofstream can never open it)
     * makes WritePoseFile return false: message surfaces, the in-memory
     * state AND the dirty flag are kept.*/
    const Point6D before = f.stored(1, 0);
    const int messages_before = f.messages.titles.size();
    f.poses()->savePoseFile(QStringLiteral(""));

    REQUIRE(f.messages.titles.size() == messages_before + 1);
    REQUIRE(f.messages.titles.back() == QStringLiteral("Error!"));
    REQUIRE(
        f.messages.texts.back() ==
        QStringLiteral("Failed to write pose file!"));
    REQUIRE(SamePose(f.stored(1, 0), before));
    REQUIRE(f.poses()->dirty());

    /*Same contract for the kinematics path.*/
    f.poses()->saveKinematics(QStringLiteral(""));
    REQUIRE(f.messages.titles.size() == messages_before + 2);
    REQUIRE(
        f.messages.texts.back() ==
        QStringLiteral("Failed to write kinematics file!"));
    REQUIRE(SamePose(f.stored(1, 0), before));
    REQUIRE(f.poses()->dirty());
}

TEST_CASE("pose_bridge: save pose file -> reload -> identical (scenario e, "
          "round-trip)",
          "[pose_bridge]") {
    PoseFixture f;

    /*Seed a few edits across frames.*/
    REQUIRE(f.poses()->setPoseValue(0, 0, 0, QStringLiteral("18.4")));
    REQUIRE(f.poses()->setPoseValue(1, 0, 2, QStringLiteral("-1022.93")));
    REQUIRE(f.poses()->setPoseValue(2, 0, 4, QStringLiteral("-7.79")));
    REQUIRE(f.poses()->dirty());

    /*Save the current frame's pose (frame 0, primary model).*/
    const QString pose_path = f.dir.filePath("pose.jtap");
    f.poses()->savePoseFile(pose_path);
    REQUIRE(f.messages.titles.isEmpty());
    /*A successful save clears the dirty flag.*/
    REQUIRE_FALSE(f.poses()->dirty());

    /*Mutate the cell, then reload the saved file — the value must come
     * back bit-identical.*/
    REQUIRE(f.poses()->setPoseValue(0, 0, 0, QStringLiteral("99.0")));
    f.poses()->loadPoseFile(pose_path);
    REQUIRE(f.messages.titles.isEmpty());
    REQUIRE(f.stored(0, 0).x == Approx(18.4));
    REQUIRE(f.stored(0, 0).z == Approx(-2500.0));  // untouched by the load
    /*A load is an in-memory mutation: dirty set again.*/
    REQUIRE(f.poses()->dirty());

    /*Kinematics round-trip: all frames, then reload into a fresh fixture —
     * every frame matches (bit-exact doubles from the file rows).*/
    const QString kin_path = f.dir.filePath("kin.jtak");
    f.poses()->saveKinematics(kin_path);
    REQUIRE_FALSE(f.poses()->dirty());

    PoseFixture g;
    g.poses()->loadKinematics(kin_path);
    REQUIRE(g.messages.titles.isEmpty());
    REQUIRE(g.stored(0, 0).x == Approx(18.4));
    REQUIRE(g.stored(0, 0).z == Approx(-2500.0));
    REQUIRE(g.stored(1, 0).z == Approx(-1022.93));
    REQUIRE(g.stored(2, 0).ya == Approx(-7.79));
    REQUIRE(g.poses()->dirty());
}

TEST_CASE("pose_bridge: real golden kinematics fixture reads through "
          "loadKinematics (test/golden style)",
          "[pose_bridge]") {
    /*The oracle's captured kinematics file (test/golden/
     * fem_oracle_captured.jtak, JTA_EULER_KINEMATICS, 3 data rows) loads
     * position-preserving into the session's 3 frames; the poses land in
     * LocationStorage and the scene sync targets the current frame.*/
    PoseFixture f;
    f.poses()->loadKinematics(
        QStringLiteral("test/golden/fem_oracle_captured.jtak"));
    REQUIRE(f.messages.titles.isEmpty());

    REQUIRE(f.stored(0, 0).x == Approx(18.3984));
    REQUIRE(f.stored(0, 0).y == Approx(19.6297));
    REQUIRE(f.stored(0, 0).z == Approx(-1022.93));
    REQUIRE(f.stored(1, 0).x == Approx(18.8612));
    REQUIRE(f.poses()->dirty());
}

TEST_CASE("pose_bridge: invalid pose/kinematics files surface the widgets' "
          "typed messages",
          "[pose_bridge]") {
    PoseFixture f;

    /*A garbage file: "Invalid Pose File!" (ok=false, not_optimized=false).*/
    const QString bad = f.dir.filePath("bad.txt");
    {
        QFile file(bad);
        REQUIRE(file.open(QIODevice::WriteOnly));
        file.write("not a pose file\n");
    }
    f.poses()->loadPoseFile(bad);
    REQUIRE(f.messages.titles.size() == 1);
    REQUIRE(f.messages.texts.back() == QStringLiteral("Invalid Pose File!"));
    REQUIRE_FALSE(f.poses()->dirty());

    /*A NOT_OPTIMIZED pose file: "No Pose Exists!" (valid-but-empty).*/
    const QString not_opt = f.dir.filePath("not_optimized.jtap");
    {
        QFile file(not_opt);
        REQUIRE(file.open(QIODevice::WriteOnly));
        file.write("JTA_EULER_POSE\n"
                   "X_TRAN\t\tY_TRAN\t\tZ_TRAN\t\tZ_ROT\t\tX_ROT\t\tY_ROT\n"
                   "NOT_OPTIMIZED,\t0,\t0,\t0,\t0,\t0,\n");
    }
    f.poses()->loadPoseFile(not_opt);
    REQUIRE(f.messages.titles.size() == 2);
    REQUIRE(f.messages.texts.back() == QStringLiteral("No Pose Exists!"));
    REQUIRE_FALSE(f.poses()->dirty());

    /*A garbage kinematics file: "Invalid Kinematics File!".*/
    f.poses()->loadKinematics(bad);
    REQUIRE(f.messages.titles.size() == 3);
    REQUIRE(
        f.messages.texts.back() == QStringLiteral("Invalid Kinematics File!"));
    REQUIRE_FALSE(f.poses()->dirty());

    /*State untouched throughout.*/
    REQUIRE(f.stored(0, 0).z == Approx(-2500.0));
}

/*Plan 007 U3 (D3): the pose table re-reads storage after a viewer drag
 * applied a pose. The refresh is wired in the hub (viewerPoseApplied ->
 * PoseBridge::refreshTable); applyViewerPose writes storage + emits, and
 * the table model must show the dragged value — without the hub wiring
 * the table would keep the stale value (QQC2 Dialog keeps its contentItem
 * across open/close, U1 review D-05).*/
TEST_CASE("pose_bridge: a viewer drag refreshes the table (D3 wiring)",
          "[pose_bridge]") {
    PoseFixture f;

    /*The observable of the refresh relay is the modelReset notification —
     * the QML bindings re-read the roles only when the model announces a
     * change (the model's data() reads storage live; the reset is what
     * makes the table re-evaluate).*/
    int resets = 0;
    QObject::connect(f.table(), &QAbstractItemModel::modelReset,
                     [&resets]() { ++resets; });

    /*A model-centric drag end on the current frame + primary model.*/
    f.study()->applyViewerPose(0, 42.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    REQUIRE(f.stored(0, 0).x == Approx(42.0));
    /*The hub wiring (viewerPoseApplied -> refreshTable) fired exactly one
     * reset — without it the table would keep showing the pre-drag value
     * (QQC2 Dialog keeps its contentItem across open/close, U1 D-05).*/
    REQUIRE(resets == 1);
}

/*Plan 007 U3 (D3): the refresh relay itself — an external storage write
 * becomes visible after refreshTable() (the run-state leg of D3 calls the
 * same relay; the terminal-state transition itself is not drivable
 * headlessly — the GPU run is oracle/manual-visual).*/
TEST_CASE("pose_bridge: refreshTable announces a model reset (D3 relay)",
          "[pose_bridge]") {
    PoseFixture f;
    int resets = 0;
    QObject::connect(f.table(), &QAbstractItemModel::modelReset,
                     [&resets]() { ++resets; });

    /*The relay announces the change so QML re-reads the roles (the model's
     * data() reads storage live — the reset is the notification, and the
     * run-state leg of D3 calls the same relay; the terminal-state
     * transition itself is not drivable headlessly — the GPU run is
     * oracle/manual-visual).*/
    f.poses()->refreshTable();
    REQUIRE(resets == 1);

    /*An external storage write is visible after the reset re-evaluation
     * (the run's terminal-frame SavePose path).*/
    f.session()->model_locations.SavePose(1, 0, Point6D(1, 2, 3, 4, 5, 6));
    f.poses()->refreshTable();
    REQUIRE(resets == 2);
    REQUIRE(f.table()->data(f.table()->index(1, 0), PoseTableModel::XRole)
                .toDouble() == Approx(1.0));
    REQUIRE(f.table()->data(f.table()->index(1, 0), PoseTableModel::ZaRole)
                .toDouble() == Approx(6.0));
}

/*Plan 007 U3 (D4): every manual pose write drops the pending ML seed —
 * viewer drags, pose-table edits, copy-prev/next, and pose/kinematics
 * loads (wired in the hub: viewerPoseApplied / poseTableChanged ->
 * OptimizerBridge::clearSeedPose). Without this the next run() would
 * silently apply the estimate over the user's arrangement (I3). A SAVE is
 * not a pose write — the seed survives it (only mutations clear).*/
TEST_CASE("pose_bridge: manual pose writes drop the pending ML seed (D4)",
          "[pose_bridge]") {
    PoseFixture f;

    /*Baseline: no seed -> clears are no-ops.*/
    REQUIRE_FALSE(f.optimizer()->hasSeedPose());
    f.optimizer()->clearSeedPose();
    REQUIRE_FALSE(f.optimizer()->hasSeedPose());

    /*Drag path: viewerPoseApplied -> clearSeedPose.*/
    f.optimizer()->setSeedPose(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    REQUIRE(f.optimizer()->hasSeedPose());
    f.study()->applyViewerPose(0, 42.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    REQUIRE_FALSE(f.optimizer()->hasSeedPose());

    /*Table-edit path: poseTableChanged -> clearSeedPose.*/
    f.optimizer()->setSeedPose(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    REQUIRE(f.poses()->setPoseValue(0, 0, 2, QStringLiteral("-4.0")));
    REQUIRE_FALSE(f.optimizer()->hasSeedPose());

    /*Copy path: copyNext writes the current frame + emits poseTableChanged.*/
    f.optimizer()->setSeedPose(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    f.poses()->copyNext();
    REQUIRE_FALSE(f.optimizer()->hasSeedPose());

    /*Load-pose path: loadPoseFile emits poseTableChanged.*/
    f.optimizer()->setSeedPose(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    const QString pose_path = f.dir.filePath("pose.jtap");
    f.poses()->savePoseFile(pose_path);
    /*A SAVE is not a pose write — the seed survives.*/
    REQUIRE(f.optimizer()->hasSeedPose());
    f.poses()->loadPoseFile(pose_path);
    REQUIRE_FALSE(f.optimizer()->hasSeedPose());

    /*Load-kinematics path (the golden fixture).*/
    f.optimizer()->setSeedPose(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    f.poses()->loadKinematics(
        QStringLiteral("test/golden/fem_oracle_captured.jtak"));
    REQUIRE_FALSE(f.optimizer()->hasSeedPose());

    /*A fresh seed still applies normally after all the clears.*/
    f.optimizer()->setSeedPose(9.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    REQUIRE(f.optimizer()->hasSeedPose());
}
