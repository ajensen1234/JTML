#pragma once
#include <QObject>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include "core/data_structures_6D.h"
#include "core/calibration.h"

namespace jta_gui {

class EstimationWorker : public QObject {
    Q_OBJECT
public:
    EstimationWorker(
        const std::string& stl_path,
        const std::string& pt_model_location,
        const std::vector<cv::Mat>& images_A,
        const std::vector<cv::Mat>& images_B,
        bool is_biplane,
        bool black_sil_used,
        Calibration calibration_file);

    ~EstimationWorker() override = default;

public Q_SLOTS:
    void process();

signals:
    void progressUpdated(int value, QString status);
    void poseEstimated(int index, Point6D pose);
    void finished(bool success, QString errorMessage);

private:
    std::string stl_path_;
    std::string pt_model_location_;
    std::vector<cv::Mat> images_A_;
    std::vector<cv::Mat> images_B_;
    bool is_biplane_;
    bool black_sil_used_;
    Calibration calibration_file_;
};

} // namespace jta_gui
