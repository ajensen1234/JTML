#include "gui/worker_orchestrator.h"

#include <utility>

namespace {

std::vector<cv::Mat> CollectOriginalImages(std::vector<Frame>& frames) {
    std::vector<cv::Mat> images;
    images.reserve(frames.size());

    for (auto& frame : frames) {
        images.push_back(frame.GetOriginalImage());
    }

    return images;
}

std::vector<cv::Mat> CollectInvertedImages(std::vector<Frame>& frames) {
    std::vector<cv::Mat> images;
    images.reserve(frames.size());

    for (auto& frame : frames) {
        images.push_back(frame.GetInvertedImage());
    }

    return images;
}

} // namespace

namespace jta_gui {

WorkerOrchestrator::WorkerOrchestrator(QObject* parent) : QObject(parent) {}

WorkerOrchestrator::~WorkerOrchestrator() {
    CleanupSegmentationWorker();
    CleanupEstimationWorker();
}

void WorkerOrchestrator::ConfigureSegmentation(SegmentationConfig config) {
    segmentation_config_ = std::move(config);
}

void WorkerOrchestrator::StartSegmentation(jta_core::SessionContext& session) {
    if (!segmentation_config_.has_value()) {
        emit segmentationFinished(false, "Segmentation configuration not set.");
        return;
    }

    CleanupSegmentationWorker();

    const auto& config = *segmentation_config_;
    std::vector<cv::Mat> images_A = CollectOriginalImages(session.loaded_frames);
    std::vector<cv::Mat> images_B;
    if (session.calibrated_for_biplane_viewport_) {
        images_B = CollectOriginalImages(session.loaded_frames_B);
    }

    segmentation_worker_ = new jta_gui::SegmentationWorker(
        config.pt_model_location,
        config.input_width,
        config.input_height,
        images_A,
        images_B,
        session.calibrated_for_biplane_viewport_,
        config.black_sil_used,
        config.dilation_val,
        config.aperture,
        config.low_threshold,
        config.high_threshold);

    segmentation_thread_ = new QThread();
    segmentation_worker_->moveToThread(segmentation_thread_);

    connect(segmentation_thread_, &QThread::started, segmentation_worker_, &jta_gui::SegmentationWorker::process);
    connect(segmentation_worker_, &jta_gui::SegmentationWorker::imageSegmented, this, &WorkerOrchestrator::imageSegmented);
    connect(segmentation_worker_, &jta_gui::SegmentationWorker::progressUpdated, this, &WorkerOrchestrator::progressUpdated);
    connect(
        segmentation_worker_,
        &jta_gui::SegmentationWorker::finished,
        this,
        [this](bool success, const QString& error_message) {
            emit segmentationFinished(success, error_message);
            CleanupSegmentationWorker();
        });

    segmentation_thread_->start();
}

void WorkerOrchestrator::ConfigureEstimation(EstimationConfig config) {
    estimation_config_ = std::move(config);
}

void WorkerOrchestrator::StartEstimation(jta_core::SessionContext& session) {
    if (!estimation_config_.has_value()) {
        emit estimationFinished(false, "Estimation configuration not set.");
        return;
    }

    CleanupEstimationWorker();

    const auto& config = *estimation_config_;
    std::vector<cv::Mat> images_A = CollectInvertedImages(session.loaded_frames);
    std::vector<cv::Mat> images_B;
    if (session.calibrated_for_biplane_viewport_) {
        images_B = CollectInvertedImages(session.loaded_frames_B);
    }

    estimation_worker_ = new jta_gui::EstimationWorker(
        config.stl_path,
        config.pt_model_location,
        images_A,
        images_B,
        session.calibrated_for_biplane_viewport_,
        config.black_sil_used,
        session.calibration_file_);

    estimation_thread_ = new QThread();
    estimation_worker_->moveToThread(estimation_thread_);

    connect(estimation_thread_, &QThread::started, estimation_worker_, &jta_gui::EstimationWorker::process);
    connect(estimation_worker_, &jta_gui::EstimationWorker::poseEstimated, this, &WorkerOrchestrator::poseEstimated);
    connect(estimation_worker_, &jta_gui::EstimationWorker::progressUpdated, this, &WorkerOrchestrator::progressUpdated);
    connect(
        estimation_worker_,
        &jta_gui::EstimationWorker::finished,
        this,
        [this](bool success, const QString& error_message) {
            emit estimationFinished(success, error_message);
            CleanupEstimationWorker();
        });

    estimation_thread_->start();
}

void WorkerOrchestrator::CleanupSegmentationWorker() {
    if (segmentation_thread_ != nullptr) {
        segmentation_thread_->quit();
        segmentation_thread_->wait();
        delete segmentation_thread_;
        segmentation_thread_ = nullptr;
    }

    if (segmentation_worker_ != nullptr) {
        delete segmentation_worker_;
        segmentation_worker_ = nullptr;
    }
}

void WorkerOrchestrator::CleanupEstimationWorker() {
    if (estimation_thread_ != nullptr) {
        estimation_thread_->quit();
        estimation_thread_->wait();
        delete estimation_thread_;
        estimation_thread_ = nullptr;
    }

    if (estimation_worker_ != nullptr) {
        delete estimation_worker_;
        estimation_worker_ = nullptr;
    }
}

} // namespace jta_gui
