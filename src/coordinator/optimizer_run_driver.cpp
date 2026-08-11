// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 006 U5: the production OptimizerRunDriver adapter — wraps
// OptimizerManager UNTOUCHED (M12/Q8): a fresh manager + thread per run,
// the drive calls delegated verbatim, and the destructor contract (H3).
// The manager's own Initialize() wires started->Optimize and the finished->
// quit/deleteLater cleanup chain internally; the adapter's QPointers simply
// observe that chain (never double-delete the deleteLater'd objects).

#include "coordinator/optimizer_run_driver.h"

#include <QPointer>
#include <QThread>
#include <QtGlobal>

namespace {

/*H3 bounded-wait budget: DIRECT iterations are bounded (20k/25k/30k calls),
 * so an expiry here is a diagnostic, not a branch — warn and keep waiting,
 * never delete a running thread.*/
constexpr int kRunWaitTimeoutMs = 5000;

}  // namespace

namespace jta {

class OptimizerManagerRunDriver : public OptimizerRunDriver {
public:
    OptimizerManagerRunDriver() {
        manager_ = new OptimizerManager();
        thread_ = new QThread();
        manager_->moveToThread(thread_);
    }

    ~OptimizerManagerRunDriver() override {
        /*H3 destructor contract: request the cooperative stop, then quit +
         * bounded wait; on expiry NEVER delete a running thread (that
         * reproduces the app-close-mid-run crash) — warn and keep waiting.*/
        if (thread_ && thread_->isRunning()) {
            if (manager_) {
                manager_->onStopOptimizer();
            }
            Wait();
        }
        /*After a normal run the manager's internal finished chain has
         * deleteLater'd both objects (QPointer goes null); a failed-
         * Initialize ghost cleans up through the same chain once its thread
         * runs. Nothing to delete here — the chain owns them.*/
    }

    QObject* Manager() override { return manager_; }

    bool ThreadActive() const override {
        return thread_ && thread_->isRunning();
    }

    bool Initialize(const OptimizerRunLaunch& launch, QString& error_message)
        override {
        return manager_->Initialize(
            *thread_,
            launch.calibration,
            launch.camera_a_frames,
            launch.camera_b_frames,
            launch.current_frame_index,
            launch.models,
            launch.selected_model_indexes,
            launch.primary_model_index,
            launch.pose_matrix,
            launch.settings,
            launch.trunk_manager,
            launch.branch_manager,
            launch.leaf_manager,
            launch.directive,
            error_message,
            launch.iter_count);
    }

    void Start() override {
        if (thread_) {
            thread_->start();
        }
    }

    void Stop() override {
        /*Direct call of the public slot — the same effect as the widgets'
         * DirectConnection reverse bind (onStopOptimizer flips a worker
         * flag).*/
        if (manager_) {
            manager_->onStopOptimizer();
        }
    }

    void Wait() override {
        if (!thread_) {
            return;
        }
        thread_->quit();
        if (!thread_->wait(kRunWaitTimeoutMs)) {
            qWarning().noquote()
                << "OptimizerManagerRunDriver: run thread did not finish"
                   " within" << kRunWaitTimeoutMs
                << "ms; keeping wait (never delete a running thread)";
            thread_->wait();
        }
    }

private:
    QPointer<OptimizerManager> manager_;
    QPointer<QThread> thread_;
};

std::shared_ptr<OptimizerRunDriver> CreateOptimizerManagerRunDriver() {
    return std::make_shared<OptimizerManagerRunDriver>();
}

}  // namespace jta
