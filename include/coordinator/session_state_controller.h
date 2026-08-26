// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 006 U6: SessionStateController — the QObject notification shell over
// jta::SessionState (R5/R6/R10, AE2). BOTH front-ends write through it:
// MainScreen's SyncSessionState + previous-frame/model bookkeeping relocate
// here, and StudyBridge's syncSessionState thins onto it. The controller
// DIFFS the incoming plain values against the wrapped state and emits
// datasetChanged / selectionChanged only on actual change (M9), so repeated
// syncs do not spam.
//
// View-agnostic (R15): the views compute raw facts from their own selection
// mechanisms (the widgets' QItemSelectionModel, the QML DelegateSelection)
// and call UpdateSession with PLAIN VALUES — no widgets/QML/scene/renderer
// pointers anywhere. The run-in-flight probe (M7) and the seed-clear on
// dataset clear (H5/M10b) are injected std::function sources wired by the
// composition roots.
//
// Mirror semantics (H2, pinned): the previous-frame / previous-model-rows
// mirrors mean "the selection the user is leaving" during the view's
// save-last-pose step and "the current selection" in steady state — exactly
// the widgets' previous_frame_index_ / previous_model_indices_ semantics.
// The widgets selection handlers' order (sync -> save-last-pose -> mirrors)
// is reproduced by the two-phase API:
//   UpdateSession  — writes the new facts WITHOUT touching the mirrors, so
//                    the view's save-last-pose (called between the two
//                    phases) still reads the pre-change selection; defers
//                    selectionChanged (datasetChanged fires immediately when
//                    a count changed).
//   CommitSelection — advances the mirrors to the now-current selection (H2
//                    steady state) and emits the deferred selectionChanged
//                    (M9: emitted only AFTER the mirrors are consistent —
//                    an emit inside the sync would notify with stale
//                    previous). The QML side calls the two back-to-back (no
//                    save-last-pose between them).
//
// Ownership: the controller WRAPS (non-owning pointer) the SessionState the
// views share — MainScreen's member on the widgets side,
// ExperimentalSession::session_state on the QML side. The composition root
// owns the state; the controller is the diff + notification layer around it.

#ifndef SESSION_STATE_CONTROLLER_H
#define SESSION_STATE_CONTROLLER_H

#include <QObject>
#include <functional>
#include <vector>

#include "domain/session_state.h"

class SessionStateController : public QObject {
    Q_OBJECT

public:
    /*Wraps `state` (non-owning; must outlive the controller). The optional
     * callbacks are wired by the composition roots:
     *  - run_in_flight: the M7 probe consulted by runInFlight() (default:
     *    no run in flight — the views wire the OptimizerRunController's
     *    running() when the guard cuts land);
     *  - clear_seed: the H5/M10b dataset-clear seed drop invoked by
     *    ResetForDatasetClear (default: no-op — the widgets app has no
     *    dataset-clear path; AppBridge wires OptimizerBridge::clearSeedPose).
     * Neither callback is invoked during construction.*/
    explicit SessionStateController(
        jta::SessionState* state,
        std::function<bool()> run_in_flight = {},
        std::function<void()> clear_seed = {},
        QObject* parent = nullptr);

    /*The one write path. Diffs the four facts (normalized exactly like the
     * domain setters: out-of-range current frame -> -1, rows pruned to
     * [0, model_count) + sorted) against the wrapped state. On change:
     * writes the new facts, emits datasetChanged when a count changed, and
     * DEFERS selectionChanged to CommitSelection. The previous mirrors are
     * deliberately NOT touched here — the view's save-last-pose reads them
     * as the pre-change selection. No change -> no writes, no emissions.*/
    void UpdateSession(
        int frame_count,
        int model_count,
        int current_frame,
        const std::vector<int>& selected_rows);

    /*Mirror advance + deferred emission (call AFTER the view's
     * save-last-pose — the widgets handler order). Moves the previous
     * mirrors to the now-current selection (H2 steady state) and emits the
     * deferred selectionChanged; the signal payload's previous values are
     * the PRE-CHANGE selection (what save-last-pose would have read). No-op
     * (no emission) when no current-frame/selection change is pending.*/
    void CommitSelection();

    /*Dataset clear (H5/M10b): resets the previous mirrors to -1/empty,
     * invokes the injected seed-clear (the OptimizerRunController's pending
     * seed must not leak across datasets), drops any deferred pending
     * emission, and emits datasetChanged. The CURRENT values (counts,
     * current frame, selection) are the views' to reset — the QML
     * ExperimentalSession::ClearDataset wipes them before the call; the
     * widgets app has no clear path today.*/
    void ResetForDatasetClear();

    /*M7: true while an optimizer run is in flight — the probe for the
     * follow-up menu-guard and study-load-guard cuts. Defaults to false
     * until a composition root injects the run controller probe.*/
    bool runInFlight() const;

    /*---- Reads (thin pass-through to the wrapped state) ------------------*/
    const jta::SessionState& sessionState() const {
        return *state_;
    }

signals:
    /*Emitted when the frame/model counts changed (dataset facts). Also
     * emitted by ResetForDatasetClear after the mirrors reset.*/
    void datasetChanged();
    /*Emitted by CommitSelection on an actual current-frame / selection
     * change, AFTER the mirrors advanced to the now-current selection (M9).
     * Payload: the new selection (current_frame, primary = first row or -1,
     * selected_rows) + the PRE-CHANGE mirrors (previous_frame,
     * previous_model_rows — the selection the consumer would have saved
     * last).*/
    void selectionChanged(
        int current_frame,
        int primary_model_index,
        const std::vector<int>& selected_rows,
        int previous_frame,
        const std::vector<int>& previous_model_rows);

private:
    jta::SessionState* state_ = nullptr;
    std::function<bool()> run_in_flight_;
    std::function<void()> clear_seed_;
    /*Deferred selectionChanged (M9): set by UpdateSession when the current
     * frame or selection changed, consumed by CommitSelection. The pending
     * previous values are the pre-change selection captured at the sync —
     * the payload must carry what save-last-pose would have read.*/
    bool pending_selection_change_ = false;
    int pending_previous_frame_ = -1;
    std::vector<int> pending_previous_rows_;
};

#endif /* SESSION_STATE_CONTROLLER_H */
