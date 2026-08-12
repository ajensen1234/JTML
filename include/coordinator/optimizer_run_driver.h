// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 006 U5: OptimizerRunDriver — the narrow drive seam behind
// OptimizerRunController (M12/Q8). OptimizerManager::Initialize is
// NON-VIRTUAL, so a test subclass cannot intercept it; the seam is the
// driver, not the manager. The controller depends on this interface (a
// fresh driver per run, obtained from an injected factory); a production
// adapter (optimizer_run_driver.cpp) wraps OptimizerManager UNTOUCHED.
// Tests implement the interface and emit the manager's 7+1 signals,
// recording Initialize args + bind order. This preserves the "NOT touching
// OptimizerManager internals" boundary and is the multi-stage oracle's
// future entry point.
//
// Thread ownership lives in the driver: the production adapter creates a
// fresh OptimizerManager + QThread per instance and implements the
// destructor contract (H3 — never delete a running thread; cooperative
// stop -> quit + bounded wait, warn + keep waiting on expiry).

#ifndef OPTIMIZER_RUN_DRIVER_H
#define OPTIMIZER_RUN_DRIVER_H

#include <memory>

#include <QModelIndexList>
#include <QObject>
#include <QString>

/*The full Initialize surface (Calibration, Frame/Model containers,
 * LocationStorage, OptimizerSettings, the three CostFunctionManagers, the
 * QModelIndexList). Included first: the header pulls CostFunctionManager.h
 * (torch ATen headers) — the established include-order rule.*/
#include "coordinator/optimizer_manager.h"

namespace jta {

/*Everything OptimizerManager::Initialize consumes, packaged by the view
 * (BY VALUE — the manager works on copies) and forwarded by the controller
 * through the driver. `current_frame_index` / `primary_model_index` /
 * `directive` are filled by the controller from the gate intent + the typed
 * directive (the view leaves them at their defaults).*/
struct OptimizerRunLaunch {
    Calibration calibration;
    std::vector<Frame> camera_a_frames;
    std::vector<Frame> camera_b_frames;
    unsigned int current_frame_index = 0;
    std::vector<Model> models;
    QModelIndexList selected_model_indexes;
    unsigned int primary_model_index = 0;
    LocationStorage pose_matrix;
    OptimizerSettings settings;
    /*Plan 008 U8: the per-stage optimizer-variant slot (origin R3). Defaults
     * to the classic-DIRECT search (bit-identical); the production adapter
     * forwards it verbatim into OptimizerManager::Initialize ->
     * RunDirectStage's DirectOptimizer ctor. Non-default fields are plan-008
     * fail-fast stubs (std::invalid_argument at DirectOptimizer construction).*/
    DirectOptimizer::Options direct_options;
    jta_cost_function::CostFunctionManager trunk_manager;
    jta_cost_function::CostFunctionManager branch_manager;
    jta_cost_function::CostFunctionManager leaf_manager;
    QString directive;
    int iter_count = 0;
};

/*The drive seam. A fresh instance per run; the controller binds the 8
 * connects (finished FIRST — M6 — then the 7, L13) against Manager() before
 * calling Initialize.*/
class OptimizerRunDriver {
public:
    virtual ~OptimizerRunDriver() = default;

    /*The manager QObject whose signals the controller binds. Available
     * immediately after construction — BEFORE Initialize — so the finished
     * bind precedes Initialize and a failed Initialize's ghost thread
     * termination is observed (M6).*/
    virtual QObject* Manager() = 0;

    /*True while the run thread is alive. The controller's Start gate rejects
     * a re-run while a previous thread is alive (covers the Initialize-
     * failure ghost, H1/M6).*/
    virtual bool ThreadActive() const = 0;

    /*Mirrors OptimizerManager::Initialize's surface (by-value containers +
     * plain rows); the adapter forwards untouched.*/
    virtual bool Initialize(
        const OptimizerRunLaunch& launch, QString& error_message) = 0;

    /*Thread start (the quirk path starts the thread even when Initialize
     * failed — the ghost, preserved verbatim).*/
    virtual void Start() = 0;
    /*Cooperative emergency stop (the manager's onStopOptimizer flips the
     * worker flag; the run completes through the normal terminal-frame/
     * finished path).*/
    virtual void Stop() = 0;
    /*Wait the run thread out (quit + bounded wait; on expiry warn + keep
     * waiting — never delete a running thread, H3).*/
    virtual void Wait() = 0;
};

/*Production factory: a fresh OptimizerManagerRunDriver (new manager + new
 * QThread, moveToThread applied). Shared ownership: the controller releases
 * a finished run's driver at the next start(), while a finished driver's
 * connections stay alive until then (the epoch + sender guard drops any
 * straggler relays — H1).*/
std::shared_ptr<OptimizerRunDriver> CreateOptimizerManagerRunDriver();

}  // namespace jta

#endif /* OPTIMIZER_RUN_DRIVER_H */
