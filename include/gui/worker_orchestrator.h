#pragma once

#include <QObject>
#include <QThread>

#include <optional>
#include <string>

#include "core/session_context.h"
#include "gui/estimation_worker.h"
#include "gui/segmentation_worker.h"

namespace jta_gui {

struct SegmentationConfig {
    std::string pt_model_location;
    unsigned int input_width = 1024;
    unsigned int input_height = 1024;
    bool black_sil_used = true;
    int dilation_val = 0;
    int aperture = 3;
    int low_threshold = 0;
    int high_threshold = 150;
};

struct EstimationConfig {
    std::string stl_path;
    std::string pt_model_location;
    bool black_sil_used = true;
};

class WorkerOrchestrator : public QObject {
    Q_OBJECT

public:
    explicit WorkerOrchestrator(QObject* parent = nullptr);
    ~WorkerOrchestrator() override;

    void ConfigureSegmentation(SegmentationConfig config);
    void StartSegmentation(jta_core::SessionContext& session);

    void ConfigureEstimation(EstimationConfig config);
    void StartEstimation(jta_core::SessionContext& session);

Q_SIGNALS:
    void progressUpdated(int value, QString status);
    void imageSegmented(int index, cv::Mat segmented, bool isBiplane);
    void segmentationFinished(bool success, QString errorMessage);
    void poseEstimated(int index, Point6D pose);
    void estimationFinished(bool success, QString errorMessage);

private:
    void CleanupSegmentationWorker();
    void CleanupEstimationWorker();

    std::optional<SegmentationConfig> segmentation_config_;
    std::optional<EstimationConfig> estimation_config_;

    QThread* segmentation_thread_ = nullptr;
    jta_gui::SegmentationWorker* segmentation_worker_ = nullptr;

    QThread* estimation_thread_ = nullptr;
    jta_gui::EstimationWorker* estimation_worker_ = nullptr;
};

} // namespace jta_gui
