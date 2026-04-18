#pragma once
#include <QObject>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include "core/frame.h"

namespace jta_gui {

class SegmentationWorker : public QObject {
    Q_OBJECT
public:
    SegmentationWorker(
        const std::string& pt_model_location,
        unsigned int input_width,
        unsigned int input_height,
        const std::vector<cv::Mat>& images_A,
        const std::vector<cv::Mat>& images_B,
        bool is_biplane,
        bool black_sil_used,
        int dilation_val,
        int aperture,
        int low_threshold,
        int high_threshold);

    ~SegmentationWorker() override = default;

public Q_SLOTS:
    void process();

signals:
    void progressUpdated(int value, QString status);
    void imageSegmented(int index, cv::Mat segmented, bool isBiplane);
    void finished(bool success, QString errorMessage);

private:
    std::string pt_model_location_;
    unsigned int input_width_;
    unsigned int input_height_;
    std::vector<cv::Mat> images_A_;
    std::vector<cv::Mat> images_B_;
    bool is_biplane_;
    bool black_sil_used_;
    int dilation_val_;
    int aperture_;
    int low_threshold_;
    int high_threshold_;
};

} // namespace jta_gui
