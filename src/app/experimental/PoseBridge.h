// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U8: PoseBridge — the thin pose-editing adapter (R9, R10). Pass-through
// orchestration only (thinness rule): the pose table rows come from
// LocationStorage::GetPose, cell edits commit through LocationStorage::
// SavePose, the copy-prev/next slots delegate to the pure pose_copy seam
// (PreviousPose/NextPose with the widgets slots' exact index semantics), and
// the save/load actions wrap pose_file_io byte-for-byte. No behavior lives
// here beyond the orchestration order; every semantic (boundary fallback,
// primary-vs-current split, format acceptance, position-preserving loads)
// comes from the seams.
//
// Pose table semantics (net-new UI — the widgets app never numerically edits
// poses; plan 005 U8 review fix):
//  - immediate per-cell SavePose on commit (no OK/cancel staging);
//  - non-numeric / NaN / infinite input is rejected with an inline
//    validation message (validationMessage) and the stored state is left
//    unchanged;
//  - a dirty flag tracks unsaved in-memory edits (set on edit/copy/load,
//    cleared by a successful savePoseFile/saveKinematics);
//  - table + save/load/copy controls are disabled during an optimizer run
//    (the U6 locking — QML binds them to optimizerBridge.running; the Run
//    button also closes the dialogs).
//
// Table surface: PoseTableModel (QAbstractListModel, rows = frames) exposes
// the primary model's 6 pose values (roles x/y/z/xa/ya/za + frameIndex) to
// QML; the bridge refreshes it on every storage mutation and re-points it at
// the primary model whenever the selection changes (v1: pose ops are
// primary-model-only — the OptimizerBridge single-model rule; the widgets
// current-row concept has no QML analog, so the copy write row is the
// primary row; the seam still receives BOTH indices and never aligns them,
// pinned in test/unit/pose_copy_test.cpp).
//
// File actions (pose_file_io wrappers, review fix): a false return from
// WritePoseFile/WriteKinematicsFile surfaces a message and keeps the
// in-memory state (and the dirty flag) unchanged; a successful save clears
// the dirty flag. Loads surface the widgets' typed messages ("No Pose
// Exists!" / "Invalid Pose File!" / "Invalid Kinematics File!") and apply
// the widgets semantics: load-pose writes the primary model at the current
// frame; load-kinematics writes per frame up to the loaded frame count
// (position-preserving — NOT_OPTIMIZED / malformed rows leave that frame
// untouched, excess rows ignored).
//
// Deviations (documented, v1): the widgets save slots call SaveLastPose()
// (scene drift -> storage) before writing; the QML pose table is
// storage-authoritative (the table shows and saves what LocationStorage
// holds — model-mode scene drift is persisted by OptimizerBridge's
// SaveLastPose mirror at run time, not by the pose file actions).

#pragma once

#include <QAbstractListModel>
#include <QObject>
#include <QString>

class AppBridge;
class ExperimentalScene;
class ExperimentalSession;
class StudyBridge;

// The table model: one row per loaded frame, 6 pose values for the model
// row the bridge points it at (primary model). Plain data() reads of
// LocationStorage::GetPose — no logic beyond the role mapping. The bridge
// calls setModelRow() on selection changes and refresh()/notifyCellChanged()
// after storage mutations; QML binds the Repeater/ListView to this model.
class PoseTableModel : public QAbstractListModel {
    Q_OBJECT
public:
    enum Roles {
        FrameIndexRole = Qt::UserRole + 1,
        XRole,
        YRole,
        ZRole,
        XaRole,
        YaRole,
        ZaRole,
    };

    explicit PoseTableModel(ExperimentalSession* session,
                            QObject* parent = nullptr);

    // The model row whose poses the table displays (-1 = none).
    void setModelRow(int row);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role) const override;
    QHash<int, QByteArray> roleNames() const override;

    // Full reset (copy/load/dataset/selection changes — small tables, the
    // write-once list models' full-reset style) and single-cell notify (per-
    // cell commits keep the delegate alive so an in-flight edit is never
    // disrupted).
    void refresh();
    void notifyCellChanged(int frame, int axis);

private:
    ExperimentalSession* session_ = nullptr;
    int model_row_ = -1;
};

class PoseBridge : public QObject {
    Q_OBJECT

    // Table surface: row count (frames) for the pose table header + hint
    // bindings.
    Q_PROPERTY(int rowCount READ rowCount NOTIFY poseTableChanged)
    // The table model (rows = frames, roles = the 6 pose values + frame
    // index for the primary model).
    Q_PROPERTY(QObject* tableModel READ tableModel CONSTANT)
    // Dirty state: in-memory pose edits not yet persisted to a
    // pose/kinematics file. Set by every successful mutation (cell edit,
    // copy, load), cleared by a successful savePoseFile/saveKinematics.
    Q_PROPERTY(bool dirty READ dirty NOTIFY dirtyChanged)
    // Last cell-validation failure (non-numeric / NaN / out-of-range cell):
    // the inline message surface for the table. Cleared by the next
    // successful commit.
    Q_PROPERTY(QString validationMessage READ validationMessage NOTIFY
                   validationChanged)

public:
    explicit PoseBridge(
        AppBridge* hub,
        ExperimentalSession* session,
        ExperimentalScene* scene,
        StudyBridge* study_bridge,
        QObject* parent = nullptr);
    ~PoseBridge() override;

    // ---- Table reads -----------------------------------------------------
    // Pose value for (frame, model, axis): axis 0..5 = x, y, z, xa, ya, za.
    // Out-of-range cells return 0 (LocationStorage::GetPose's safe fallback
    // semantics); the table binds its rows to rowCount + primaryModelIndex.
    Q_INVOKABLE double poseValue(int frame, int model, int axis) const;

    // ---- Cell edit: immediate SavePose per cell commit -------------------
    // Parses the QML TextField text (the table commits on editingFinished).
    // Non-numeric / NaN / infinite values and out-of-range cells are
    // rejected: validationMessage is set, nothing is stored (state
    // unchanged), false is returned. A valid value is saved immediately to
    // LocationStorage::SavePose (dirty set; the scene pose refreshed when
    // the cell is on the current frame). Returns true when the value was
    // saved.
    Q_INVOKABLE bool setPoseValue(int frame, int model, int axis,
                                  const QString& text);

    // ---- Copy prev/next (widgets slots' exact index semantics) -----------
    // Guard via the seam's CheckSelection (the QML app has no multi-model
    // radio — v1 pose ops are primary-model-only — so MultiModelMode cannot
    // trigger), then the seam's PreviousPose/NextPose plan and the raw view
    // chain: GetPose at (read_frame, read_model), SavePose at (write_frame,
    // write_model) — no clamping, no conversion (the no-image fallback at
    // frame 0 / the last frame comes from GetPose(-1 / count) resolving to
    // the model's initial pose). Dirty set; the scene pose refreshed (the
    // write row is always the current frame).
    Q_INVOKABLE void copyPrevious();
    Q_INVOKABLE void copyNext();

    // ---- File actions (pose_file_io wrappers) ----------------------------
    // Save the primary model's pose at the current frame (widgets Save Pose
    // slot: GetPose(current, primary) -> WritePoseFile). A false return
    // surfaces a message and keeps the in-memory state + dirty flag; a
    // successful write clears the dirty flag.
    Q_INVOKABLE void savePoseFile(const QString& path);
    // Load a single pose into the primary model at the current frame
    // (widgets Load Pose slot). !ok surfaces "No Pose Exists!" (NOT_
    // OPTIMIZED row) / "Invalid Pose File!"; success marks dirty.
    Q_INVOKABLE void loadPoseFile(const QString& path);
    // Save the primary model's per-frame poses, one row per frame (widgets
    // Save Kinematics slot). Same false-return contract as savePoseFile.
    Q_INVOKABLE void saveKinematics(const QString& path);
    // Load per-frame poses for the primary model, up to the loaded frame
    // count (widgets Load Kinematics slot; position-preserving rows — a
    // NOT_OPTIMIZED / malformed row leaves that frame untouched). !ok
    // surfaces "Invalid Kinematics File!"; success marks dirty.
    Q_INVOKABLE void loadKinematics(const QString& path);

    // Clear the unsaved-edits flag (the QML dirty badge; the settings
    // panel's explicit-save pattern).
    Q_INVOKABLE void clearDirty();

    // D3 (plan 007 U3): the single pose-table refresh owner's relay — the
    // table re-reads storage. Called by the hub (AppBridge) when a run
    // reaches a terminal state (Completed/Error) and when a viewer drag
    // applied a pose (viewerPoseApplied). QQC2 Dialog never destroys its
    // contentItem on close, so without this the table would keep showing
    // stale values after runs and drags (U1 review D-05 confirmed the
    // premise structurally). Relay plumbing only — no policy.
    void refreshTable();

    // ---- Reads -----------------------------------------------------------
    int rowCount() const;
    QObject* tableModel() const;
    bool dirty() const;
    QString validationMessage() const;

signals:
    void poseTableChanged();
    void dirtyChanged();
    void validationChanged();
    // The single QML Dialog mechanism (same channel as the other bridges):
    // guard rejections + file-action failures.
    void messageRequested(const QString& title, const QString& message);
    // Scene -> renderer chain (mirror of the other bridges): emitted after
    // a mutation lands on the CURRENT frame's cell; QML glue calls
    // viewport.updatePose(modelIndex).
    void scenePoseChanged(int modelIndex);

private slots:
    void onSelectionChanged();
    void onDatasetChanged();

private:
    void copyPose(bool next);
    // Refresh the scene pose for (frame, model) only when it is the current
    // view cell (the scene renders the current frame; the widgets'
    // viewport-update tail).
    void syncScenePose(int frame, int model);
    bool guardSelection(const QString& reject_message);
    void markDirty();
    void clearDirtyInternal();
    void setValidation(const QString& message);

    AppBridge* hub_ = nullptr;
    ExperimentalSession* session_ = nullptr;
    ExperimentalScene* scene_ = nullptr;
    StudyBridge* study_bridge_ = nullptr;
    PoseTableModel* table_model_ = nullptr;
    bool dirty_ = false;
    QString validation_message_;
};
