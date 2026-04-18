#include "gui/segmentation_worker.h"
#include "core/machine_learning_tools.h"
#include <torch/csrc/api/include/torch/cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace jta_gui {

SegmentationWorker::SegmentationWorker(
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
    int high_threshold) :
    pt_model_location_(pt_model_location),
    input_width_(input_width),
    input_height_(input_height),
    images_A_(images_A),
    images_B_(images_B),
    is_biplane_(is_biplane),
    black_sil_used_(black_sil_used),
    dilation_val_(dilation_val),
    aperture_(aperture),
    low_threshold_(low_threshold),
    high_threshold_(high_threshold) {}

void SegmentationWorker::process() {
    try {
        torch::jit::Module module = torch::jit::load(pt_model_location_, torch::kCUDA);
        torch::jit::Module* model = &module;

        int total_images = images_A_.size();
        for (int i = 0; i < total_images; ++i) {
            // Segment image A
            cv::Mat segmented_A = segment_image(
                images_A_[i],
                black_sil_used_,
                model,
                input_width_,
                input_height_);
            emit imageSegmented(i, segmented_A, false);
            
            c10::cuda::CUDACachingAllocator::emptyCache();

            if (is_biplane_) {
                cv::Mat segmented_B = segment_image(
                    images_B_[i],
                    black_sil_used_,
                    model,
                    input_width_,
                    input_height_);
                emit imageSegmented(i, segmented_B, true);
                
                c10::cuda::CUDACachingAllocator::emptyCache();
            }

            int progress = static_cast<int>(100.0 * (i + 1) / total_images);
            emit progressUpdated(progress, QString("Segmenting frame %1 of %2...").arg(i + 1).arg(total_images));
        }

        emit finished(true, "");
    } catch (const c10::Error& e) {
        emit finished(false, QString::fromStdString(e.msg()));
    } catch (const std::exception& e) {
        emit finished(false, QString::fromStdString(e.what()));
    } catch (...) {
        emit finished(false, "Unknown error during segmentation");
    }
}

} // namespace jta_gui
