// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#pragma once

#include <QObject>
#include <QThread>
#include <QString>
#include <atomic>
#include <functional>

#include "domain/data_structures_6D.h"
#include "domain/direct_optimizer.h"

// Headless optimize coordinator (plan U4, R4-R7).
//
// OptimizeCoordinator owns the optimize state machine and a worker thread, with
// no GUI/widget dependencies. The GUI binds to it as a thin caller (R6). The
// actual optimization runs the pure DirectOptimizer on a worker QThread; the
// cost function is injected, so tests drive it with a stub (zero GPU).
//
// The coordinator re-emits progress/result signals on the coordinator's (test /
// main) thread, which is what makes it spy-able with QSignalSpy without the
// worker-thread crash (QTBUG-2842).

class OptimizeWorker : public QObject {
    Q_OBJECT
public:
    using CostFunction = std::function<double(const Point6D&)>;

    OptimizeWorker(CostFunction cost, Point6D range, Point6D starting_point,
                   unsigned int budget);
    ~OptimizeWorker() override;

    // Cooperative stop requested from another thread.
    void RequestStop();

public slots:
    // Runs DirectOptimizer on this worker's thread. Emits exactly one terminal
    // signal (succeeded or failed).
    void Run();

signals:
    // Optimum location (6 DOF) and value, in physical/denormalized space.
    void Succeeded(double x, double y, double z, double xa, double ya,
                   double za, double value);
    void Failed(QString message);

private:
    CostFunction cost_;
    Point6D range_;
    Point6D starting_point_;
    unsigned int budget_;

    std::atomic<bool> stop_requested_{false};
    DirectOptimizer* active_opt_ = nullptr;  // owned by Run(), worker thread only
};

class OptimizeCoordinator : public QObject {
    Q_OBJECT
public:
    using CostFunction = std::function<double(const Point6D&)>;

    enum class State { Idle = 0, Running = 1 };

    OptimizeCoordinator(CostFunction cost, Point6D range, Point6D starting_point,
                        unsigned int budget, QObject* parent = nullptr);
    ~OptimizeCoordinator() override;

    // Begin an optimize run. Returns false (and does nothing) if already
    // running. Emits stateChanged(Running) on a successful start.
    bool Start();

    // Cooperative stop of the current run (returns to Idle on completion).
    void Stop();

    State GetState() const;

signals:
    void StateChanged(int state);  // OptimizeCoordinator::State as int
    void Finished(bool ok);
    void ErrorOccurred(QString message);
    void RunRequested();  // internal: queued to the worker

private slots:
    void OnSucceeded(double x, double y, double z, double xa, double ya,
                     double za, double value);
    void OnFailed(QString message);

private:
    CostFunction cost_;
    Point6D range_;
    Point6D starting_point_;
    unsigned int budget_;

    State state_ = State::Idle;
    QThread worker_thread_;     // single persistent worker thread
    OptimizeWorker* worker_ = nullptr;  // owned; moved to worker_thread_
};
