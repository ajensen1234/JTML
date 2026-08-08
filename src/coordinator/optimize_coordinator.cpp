// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "coordinator/optimize_coordinator.h"

#include <utility>

// ---------------------------------------------------------------------------
// OptimizeWorker
// ---------------------------------------------------------------------------

OptimizeWorker::OptimizeWorker(CostFunction cost, Point6D range,
                               Point6D starting_point, unsigned int budget)
    : cost_(std::move(cost)),
      range_(range),
      starting_point_(starting_point),
      budget_(budget) {}

OptimizeWorker::~OptimizeWorker() {
    delete active_opt_;
}

void OptimizeWorker::RequestStop() {
    stop_requested_.store(true);
    if (active_opt_) {
        active_opt_->Stop();
    }
}

void OptimizeWorker::Run() {
    if (stop_requested_.load()) {
        emit Failed(QStringLiteral("Optimization stopped before start."));
        return;
    }
    try {
        auto* opt = new DirectOptimizer(cost_, range_, starting_point_, budget_);
        active_opt_ = opt;
        bool ok = opt->Run();
        Point6D loc = opt->GetOptimumLocation();
        double value = opt->GetOptimumValue();
        active_opt_ = nullptr;
        delete opt;

        if (!ok) {
            emit Failed(QStringLiteral("DIRECT optimizer error."));
            return;
        }
        // Completed (possibly after a cooperative stop): report the optimum found.
        emit Succeeded(loc.x, loc.y, loc.z, loc.xa, loc.ya, loc.za, value);
    } catch (const std::exception& e) {
        // A throwing cost function models an init/failure path headlessly (R7).
        if (active_opt_) {
            delete active_opt_;
            active_opt_ = nullptr;
        }
        emit Failed(QString::fromStdString(e.what()));
    } catch (...) {
        if (active_opt_) {
            delete active_opt_;
            active_opt_ = nullptr;
        }
        emit Failed(QStringLiteral("Unknown optimizer error."));
    }
}

// ---------------------------------------------------------------------------
// OptimizeCoordinator
// ---------------------------------------------------------------------------

OptimizeCoordinator::OptimizeCoordinator(CostFunction cost, Point6D range,
                                         Point6D starting_point,
                                         unsigned int budget, QObject* parent)
    : QObject(parent),
      cost_(std::move(cost)),
      range_(range),
      starting_point_(starting_point),
      budget_(budget) {
    // A single persistent worker thread + worker, reused across runs via the
    // RunRequested signal. Avoids per-run thread deleteLater races.
    worker_ = new OptimizeWorker(cost_, range_, starting_point_, budget_);
    worker_->moveToThread(&worker_thread_);

    connect(this, &OptimizeCoordinator::RunRequested, worker_,
            &OptimizeWorker::Run, Qt::QueuedConnection);
    connect(worker_, &OptimizeWorker::Succeeded, this,
            &OptimizeCoordinator::OnSucceeded, Qt::QueuedConnection);
    connect(worker_, &OptimizeWorker::Failed, this,
            &OptimizeCoordinator::OnFailed, Qt::QueuedConnection);

    worker_thread_.start();
}

OptimizeCoordinator::~OptimizeCoordinator() {
    if (worker_) worker_->RequestStop();
    worker_thread_.quit();
    worker_thread_.wait(5000);
    delete worker_;
    worker_ = nullptr;
}

bool OptimizeCoordinator::Start() {
    if (state_ != State::Idle) return false;

    state_ = State::Running;
    emit StateChanged(static_cast<int>(State::Running));
    emit RunRequested();  // queued -> runs on the worker thread
    return true;
}

void OptimizeCoordinator::Stop() {
    if (worker_) worker_->RequestStop();
}

OptimizeCoordinator::State OptimizeCoordinator::GetState() const {
    return state_;
}

void OptimizeCoordinator::OnSucceeded(double, double, double, double, double,
                                      double, double) {
    state_ = State::Idle;
    emit Finished(true);
    emit StateChanged(static_cast<int>(State::Idle));
}

void OptimizeCoordinator::OnFailed(QString message) {
    state_ = State::Idle;
    emit ErrorOccurred(message);
    emit Finished(false);
    emit StateChanged(static_cast<int>(State::Idle));
}
