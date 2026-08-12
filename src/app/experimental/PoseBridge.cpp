// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U8: PoseBridge implementation — see the header for the contract.
// Everything here is orchestration order around the seams; the widgets
// precedents it mirrors (MainScreen::on_actionCopy_Previous_Pose_triggered /
// on_actionCopy_Next_Pose_triggered / Save/Load Pose / Save/Load Kinematics,
// mainscreen.cpp:1068-1442) are called out per site.

#include "PoseBridge.h"

// Qt
#include <QUrl>

// The seams + the app-owned dataset + the selection contract.
#include "AppBridge.h"
#include "ExperimentalScene.h"
#include "ExperimentalSession.h"
#include "StudyBridge.h"
#include "domain/data_structures_6D.h"
#include "domain/pose_copy.h"
#include "domain/pose_file_io.h"
#include "services/location_storage.h"

#include <cmath>
#include <vector>

namespace {

/*QML FileDialog yields file:// URLs; the pose_file_io seams want local
 * paths (the widgets QFileDialog returned plain paths). QUrl handles the
 * scheme stripping + percent-decoding; plain paths pass through untouched
 * (the headless tests call with plain paths). Mirrors StudyBridge's local
 * helper.*/
QString LocalPath(const QString& path) {
    if (path.startsWith(QStringLiteral("file://"))) {
        return QUrl(path).toLocalFile();
    }
    return path;
}

/*The widgets' save/load slots' guard message strings (byte-identical).
 * (The widgets' third guard message — "Must Be in Single Model Selection
 * Mode to Save Pose!" — belongs to the MultiModelMode branch, which cannot
 * trigger in the QML app: v1 pose ops are primary-model-only and the app
 * has no multi-model radio.)*/
const char* kSelectFrameAndModel =
    "Select Frame and Model First!";
const char* kSelectModelAndLoadFrames =
    "Select Model and Load Frames First!";

/*axis (0..5 = x, y, z, xa, ya, za) -> the table model role for that
 * cell; -1 for an out-of-range axis (the caller falls back to a full-row
 * notify).*/
int RoleForAxis(int axis) {
    switch (axis) {
        case 0:
            return PoseTableModel::XRole;
        case 1:
            return PoseTableModel::YRole;
        case 2:
            return PoseTableModel::ZRole;
        case 3:
            return PoseTableModel::XaRole;
        case 4:
            return PoseTableModel::YaRole;
        case 5:
            return PoseTableModel::ZaRole;
        default:
            return -1;
    }
}

}  // namespace

/*---- PoseTableModel ----*/

PoseTableModel::PoseTableModel(ExperimentalSession* session, QObject* parent)
    : QAbstractListModel(parent), session_(session) {}

void PoseTableModel::setModelRow(int row) {
    if (model_row_ == row) {
        return;
    }
    model_row_ = row;
    refresh();
}

int PoseTableModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) {
        return 0;
    }
    return session_->model_locations.GetFrameCount();
}

QVariant PoseTableModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.row() < 0 ||
        index.row() >= rowCount()) {
        return {};
    }
    /*GetPose is bounds-safe: an out-of-range model row falls back to the
     * no-image default (model_row_ < 0 renders nothing — QML hides the
     * table without a primary model).*/
    const Point6D pose =
        session_->model_locations.GetPose(index.row(), model_row_);
    switch (role) {
        case FrameIndexRole:
            return index.row();
        case XRole:
            return pose.x;
        case YRole:
            return pose.y;
        case ZRole:
            return pose.z;
        case XaRole:
            return pose.xa;
        case YaRole:
            return pose.ya;
        case ZaRole:
            return pose.za;
        default:
            return {};
    }
}

QHash<int, QByteArray> PoseTableModel::roleNames() const {
    return {
        {FrameIndexRole, "frameIndex"},
        {XRole, "x"},
        {YRole, "y"},
        {ZRole, "z"},
        {XaRole, "xa"},
        {YaRole, "ya"},
        {ZaRole, "za"},
    };
}

void PoseTableModel::refresh() {
    beginResetModel();
    endResetModel();
}

void PoseTableModel::notifyCellChanged(int frame, int axis) {
    const QModelIndex cell = index(frame, 0);
    if (!cell.isValid()) {
        return;
    }
    /*Review fix (ce-code-review 2026-08-12): emit the SINGLE role for the
     * edited axis — Qt 6 supports role-filtered dataChanged. The old
     * no-roles emit made all 6 cells + the frame label re-read per
     * single-axis edit (the U7 profile candidate). Behavior-preserving:
     * only the edited cell's binding re-evaluates; an out-of-range axis
     * falls back to a full-row notify.*/
    const int role = RoleForAxis(axis);
    if (role == -1) {
        emit dataChanged(cell, cell);
    } else {
        emit dataChanged(cell, cell, {role});
    }
}

/*---- PoseBridge ----*/

PoseBridge::PoseBridge(
    AppBridge* hub,
    ExperimentalSession* session,
    ExperimentalScene* scene,
    StudyBridge* study_bridge,
    QObject* parent)
    : QObject(parent),
      hub_(hub),
      session_(session),
      scene_(scene),
      study_bridge_(study_bridge),
      table_model_(new PoseTableModel(session, this)) {
    /*Table -> selection/dataset mirrors: the table always shows the PRIMARY
     * model's poses (v1 single-model pose ops); a dataset replace changes
     * the row count.*/
    connect(study_bridge_, &StudyBridge::selectionChanged,
            this, &PoseBridge::onSelectionChanged);
    connect(study_bridge_, &StudyBridge::datasetChanged,
            this, &PoseBridge::onDatasetChanged);
    onSelectionChanged();
}

PoseBridge::~PoseBridge() = default;

double PoseBridge::poseValue(int frame, int model, int axis) const {
    const Point6D pose = session_->model_locations.GetPose(frame, model);
    switch (axis) {
        case 0:
            return pose.x;
        case 1:
            return pose.y;
        case 2:
            return pose.z;
        case 3:
            return pose.xa;
        case 4:
            return pose.ya;
        case 5:
            return pose.za;
        default:
            return 0.0;
    }
}

bool PoseBridge::setPoseValue(int frame, int model, int axis,
                              const QString& text) {
    /*Validation (review fix): non-numeric / NaN / infinite input is
     * rejected with an inline message and the stored state is left
     * unchanged (no SavePose). Qt's toDouble accepts "nan"/"inf" spellings,
     * so the finite check closes the NaN hole explicitly.*/
    bool parsed = false;
    const double value = text.trimmed().toDouble(&parsed);
    if (!parsed || !std::isfinite(value)) {
        setValidation(
            QStringLiteral("Invalid pose value \"%1\" — enter a finite "
                           "number.")
                .arg(text));
        return false;
    }
    if (axis < 0 || axis > 5) {
        setValidation(QStringLiteral("Invalid pose axis %1 (0..5).").arg(axis));
        return false;
    }
    if (frame < 0 || frame >= session_->model_locations.GetFrameCount() ||
        model < 0 || model >= session_->model_locations.GetModelCount()) {
        setValidation(
            QStringLiteral("Pose cell out of range (frame %1, model %2).")
                .arg(frame)
                .arg(model));
        return false;
    }

    /*Immediate per-cell SavePose (net-new table semantics): read-modify-
     * write the single axis, leaving the other five untouched.*/
    Point6D pose = session_->model_locations.GetPose(frame, model);
    switch (axis) {
        case 0:
            pose.x = value;
            break;
        case 1:
            pose.y = value;
            break;
        case 2:
            pose.z = value;
            break;
        case 3:
            pose.xa = value;
            break;
        case 4:
            pose.ya = value;
            break;
        default:
            pose.za = value;
            break;
    }
    session_->model_locations.SavePose(frame, model, pose);

    markDirty();
    setValidation(QString());
    table_model_->notifyCellChanged(frame, axis);
    emit poseTableChanged();
    syncScenePose(frame, model);
    return true;
}

void PoseBridge::copyPrevious() {
    copyPose(false);
}

void PoseBridge::copyNext() {
    copyPose(true);
}

void PoseBridge::copyPose(bool next) {
    /*Guard decision from the pure seam (widgets copy slots). The QML app has
     * no multi-model radio (v1 pose ops are primary-model-only), so the
     * MultiModelMode branch cannot trigger; NoFrameOrModel maps to the
     * widgets copy slots' message.*/
    if (!guardSelection(QString::fromLatin1(kSelectModelAndLoadFrames))) {
        return;
    }
    const int frame = study_bridge_->currentFrame();
    const int primary = study_bridge_->primaryModelIndex();

    /*R13 index split + boundary rule owned by the seam (plan 004 U4): READ
     * at (frame ± 1, PRIMARY model), WRITE at (frame, CURRENT model row).
     * The QML app is v1 single-model for pose ops, so the current row IS the
     * primary row; the seam still receives both indices and never aligns
     * them (pinned by test/unit/pose_copy_test.cpp).*/
    const jta::pose_copy::CopyPlan plan =
        next ? jta::pose_copy::NextPose(
                   frame, primary, primary, study_bridge_->frameCount())
             : jta::pose_copy::PreviousPose(
                   frame, primary, primary, study_bridge_->frameCount());

    /*The raw-index view chain (widgets mirror, mainscreen.cpp:1251-1256 /
     * 1347-1352): GetPose at the plan's read cell, SavePose at the plan's
     * write cell — no clamping, no conversion. At frame 0 / the last frame
     * the read resolves to the no-image default pose (the model's initial
     * pose) and overwrites the boundary frame, exactly like the widgets.*/
    const Point6D pose = session_->model_locations.GetPose(
        plan.read_frame, plan.read_model);
    session_->model_locations.SavePose(
        plan.write_frame, plan.write_model, pose);

    /*The write row is always the current frame — the scene shows it.*/
    syncScenePose(plan.write_frame, plan.write_model);
    markDirty();
    table_model_->refresh();
    emit poseTableChanged();
}

void PoseBridge::savePoseFile(const QString& path) {
    /*Guard (widgets Save Pose slot).*/
    if (!guardSelection(QString::fromLatin1(kSelectFrameAndModel))) {
        return;
    }
    const Point6D pose = session_->model_locations.GetPose(
        study_bridge_->currentFrame(), study_bridge_->primaryModelIndex());
    /*Deviations documented in the header: no SaveLastPose mirror — the QML
     * pose table is storage-authoritative (scene drift is persisted by
     * OptimizerBridge's SaveLastPose mirror at run time).*/
    if (!jta::pose_file::WritePoseFile(
            LocalPath(path).toStdString(), pose)) {
        /*Review fix: a false return surfaces a message and keeps the
         * in-memory state (+ dirty flag) unchanged.*/
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("Failed to write pose file!"));
        return;
    }
    clearDirtyInternal();
}

void PoseBridge::loadPoseFile(const QString& path) {
    /*Guard (widgets Load Pose slot).*/
    if (!guardSelection(QString::fromLatin1(kSelectFrameAndModel))) {
        return;
    }
    Point6D loaded_pose;
    const jta::pose_file::LoadResult res = jta::pose_file::ReadPoseFile(
        LocalPath(path).toStdString(), loaded_pose);
    if (!res.ok) {
        /*NOT_OPTIMIZED is a valid-but-empty pose ("No Pose Exists!"); any
         * other parse failure is an invalid file (widgets mirror).*/
        emit messageRequested(
            QStringLiteral("Error!"),
            res.not_optimized
                ? QStringLiteral("No Pose Exists!")
                : QStringLiteral("Invalid Pose File!"));
        return;
    }
    const int frame = study_bridge_->currentFrame();
    const int model = study_bridge_->primaryModelIndex();
    session_->model_locations.SavePose(frame, model, loaded_pose);
    syncScenePose(frame, model);
    markDirty();
    table_model_->refresh();
    emit poseTableChanged();
}

void PoseBridge::saveKinematics(const QString& path) {
    /*Guard (widgets Save Kinematics slot).*/
    if (!guardSelection(QString::fromLatin1(kSelectModelAndLoadFrames))) {
        return;
    }
    /*One row per frame for the primary model (widgets mirror).*/
    const int model = study_bridge_->primaryModelIndex();
    std::vector<Point6D> all_poses;
    all_poses.reserve(
        static_cast<size_t>(session_->model_locations.GetFrameCount()));
    for (int i = 0; i < session_->model_locations.GetFrameCount(); ++i) {
        all_poses.push_back(session_->model_locations.GetPose(i, model));
    }
    if (!jta::pose_file::WriteKinematicsFile(
            LocalPath(path).toStdString(), all_poses)) {
        /*Review fix: false return -> message + in-memory state kept.*/
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("Failed to write kinematics file!"));
        return;
    }
    clearDirtyInternal();
}

void PoseBridge::loadKinematics(const QString& path) {
    /*Guard (widgets Load Kinematics slot).*/
    if (!guardSelection(QString::fromLatin1(kSelectModelAndLoadFrames))) {
        return;
    }
    std::vector<std::optional<Point6D>> loaded_poses;
    const jta::pose_file::LoadResult res =
        jta::pose_file::ReadKinematicsFile(
            LocalPath(path).toStdString(), loaded_poses);
    if (!res.ok) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("Invalid Kinematics File!"));
        return;
    }
    /*Apply up to the number of loaded frames (excess rows are ignored).
     * loaded_poses is position-preserving: index i = frame i; NOT_OPTIMIZED
     * / malformed rows are std::nullopt and leave that frame unset, so the
     * remaining frames keep their original alignment (widgets mirror). v1:
     * the write model is the primary (the widgets writes the current model
     * row — no distinct current row in the QML selection contract).*/
    const int model = study_bridge_->primaryModelIndex();
    const int frame_count = session_->model_locations.GetFrameCount();
    for (size_t i = 0; i < loaded_poses.size() &&
                        static_cast<int>(i) < frame_count;
         ++i) {
        if (loaded_poses[i].has_value()) {
            session_->model_locations.SavePose(
                static_cast<int>(i), model, *loaded_poses[i]);
        }
    }
    /*Widgets tail: refresh the viewport at the current frame.*/
    syncScenePose(study_bridge_->currentFrame(), model);
    markDirty();
    table_model_->refresh();
    emit poseTableChanged();
}

void PoseBridge::clearDirty() {
    clearDirtyInternal();
}

void PoseBridge::refreshTable() {
    /*D3 (plan 007 U3): full reset — small tables, the write-once list
     * models' full-reset style. The QML bindings re-read every row's
     * roles, so values changed by a run or a viewer drag become visible
     * without reopening the dialog.*/
    table_model_->refresh();
}

int PoseBridge::rowCount() const {
    return table_model_->rowCount();
}

QObject* PoseBridge::tableModel() const {
    return table_model_;
}

bool PoseBridge::dirty() const {
    return dirty_;
}

QString PoseBridge::validationMessage() const {
    return validation_message_;
}

/*---- Private mirrors ----*/

void PoseBridge::onSelectionChanged() {
    table_model_->setModelRow(study_bridge_->primaryModelIndex());
}

void PoseBridge::onDatasetChanged() {
    /*A dataset replace changes the frame count; the primary model may be
     * gone. The table re-reads everything (the write-once list models'
     * full-reset style).*/
    table_model_->refresh();
    onSelectionChanged();
}

/*The widgets viewport-update tail (vw->set_model_position_at_index + Render,
 * mainscreen.cpp:1256-1264): the scene renders the CURRENT frame, so a
 * mutation on another frame changes nothing visible. Bounds are guarded by
 * ExperimentalScene::setModelPose and QmlVtkRenderer::updatePose.*/
void PoseBridge::syncScenePose(int frame, int model) {
    if (frame != study_bridge_->currentFrame()) {
        return;
    }
    scene_->setModelPose(
        model, session_->model_locations.GetPose(frame, model));
    emit scenePoseChanged(model);
}

bool PoseBridge::guardSelection(const QString& reject_message) {
    const jta::pose_copy::SelectionGuard guard =
        jta::pose_copy::CheckSelection(
            study_bridge_->currentFrame(),
            study_bridge_->selectedModelCount(),
            /*multi_model_radio_checked=*/false);
    if (guard == jta::pose_copy::SelectionGuard::NoFrameOrModel) {
        emit messageRequested(QStringLiteral("Error!"), reject_message);
        return false;
    }
    /*MultiModelMode cannot trigger (no radio in the QML app; v1 pose ops are
     * primary-model-only).*/
    return true;
}

void PoseBridge::markDirty() {
    if (dirty_) {
        return;
    }
    dirty_ = true;
    emit dirtyChanged();
}

void PoseBridge::clearDirtyInternal() {
    if (!dirty_) {
        return;
    }
    dirty_ = false;
    emit dirtyChanged();
}

void PoseBridge::setValidation(const QString& message) {
    if (validation_message_ == message) {
        return;
    }
    validation_message_ = message;
    emit validationChanged();
}
