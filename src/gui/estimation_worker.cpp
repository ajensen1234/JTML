#include "gui/estimation_worker.h"
#include "core/machine_learning_tools.h"
#include "core/STLReader.h"
#include "gpu/gpu_model.cuh"
#include <torch/csrc/api/include/torch/cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace jta_gui {

EstimationWorker::EstimationWorker(
    const std::string& stl_path,
    const std::string& pt_model_location,
    const std::vector<cv::Mat>& images_A,
    const std::vector<cv::Mat>& images_B,
    bool is_biplane,
    bool black_sil_used,
    Calibration calibration_file) :
    stl_path_(stl_path),
    pt_model_location_(pt_model_location),
    images_A_(images_A),
    images_B_(images_B),
    is_biplane_(is_biplane),
    black_sil_used_(black_sil_used),
    calibration_file_(calibration_file) {}

void EstimationWorker::process() {
    try {
        emit progressUpdated(10, "Reading STL model...");
        std::vector<std::vector<float>> triangle_information;
        stl_reader_BIG::readAnySTL(QString::fromStdString(stl_path_), triangle_information);

        if (triangle_information.empty()) {
            emit finished(false, "Could not read STL file.");
            return;
        }

        emit progressUpdated(20, "Initializing GPU Model...");
        unsigned int orig_height = images_A_[0].rows;
        unsigned int orig_width = images_A_[0].cols;

        auto gpu_mod = std::make_unique<gpu_cost_function::GPUModel>(
            "model",
            true,
            orig_height,
            orig_width,
            0,
            false,
            &(triangle_information[0])[0],
            &(triangle_information[1])[0],
            triangle_information[0].size() / 9,
            calibration_file_.camera_A_principal_);

        emit progressUpdated(30, "Loading Torch Model...");
        torch::jit::Module module = torch::jit::load(pt_model_location_, torch::kCUDA);
        torch::jit::Module* model = &module;

        unsigned int input_height = 1024;
        unsigned int input_width = 1024;

        torch::Tensor gpu_byte_placeholder(
            torch::zeros(
                {1, 1, input_height, input_width},
                torch::device(torch::kCUDA).dtype(torch::kByte)));

        float orientation[3];
        int total_frames = images_A_.size();

        for (int i = 0; i < total_frames; ++i) {
            emit progressUpdated(30 + 70 * (i + 1) / total_frames, QString("Estimating pose for frame %1...").arg(i + 1));

            cv::Mat orig_inverted = images_A_[i]; // In MainScreen, these are already inverted
            cv::Mat padded;
            if (orig_inverted.cols > orig_inverted.rows) {
                padded.create(orig_inverted.cols, orig_inverted.cols, orig_inverted.type());
            } else {
                padded.create(orig_inverted.rows, orig_inverted.rows, orig_inverted.type());
            }
            unsigned int padded_width = padded.cols;
            unsigned int padded_height = padded.rows;
            padded.setTo(cv::Scalar::all(0));
            orig_inverted.copyTo(padded(cv::Rect(0, 0, orig_inverted.cols, orig_inverted.rows)));
            cv::resize(padded, padded, cv::Size(input_width, input_height));

            cudaMemcpy(
                gpu_byte_placeholder.data_ptr(),
                padded.data,
                input_width * input_height * sizeof(unsigned char),
                cudaMemcpyHostToDevice);

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(gpu_byte_placeholder.to(torch::dtype(torch::kFloat)).flip({2}));

            cudaMemcpy(
                orientation,
                model->forward(inputs).toTensor().to(torch::dtype(torch::kFloat)).data_ptr(),
                3 * sizeof(float),
                cudaMemcpyDeviceToHost);

            cv::Mat output_mat_seg = cv::Mat(orig_inverted.rows, orig_inverted.cols, CV_8UC1);
            cv::flip(orig_inverted, output_mat_seg, 0);

            gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(
                0, 0, -calibration_file_.camera_A_principal_.principal_distance_,
                orientation[1], orientation[2], orientation[0]));

            std::vector<unsigned char> host_image_vec(orig_width * orig_height);
            cudaMemcpy(
                host_image_vec.data(),
                gpu_mod->GetPrimaryCameraRenderedImagePointer(),
                orig_width * orig_height * sizeof(unsigned char),
                cudaMemcpyDeviceToHost);

            cv::Mat projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image_vec.data());
            cv::Mat output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
            cv::flip(projection_mat, output_mat, 0);

            double sum_seg = cv::sum(output_mat_seg)[0] / 255.0;
            double sum_proj = cv::sum(output_mat)[0] / 255.0;
            double z;

            if (sum_proj / sum_seg > 1) {
                z = -calibration_file_.camera_A_principal_.principal_distance_;
            } else {
                z = -calibration_file_.camera_A_principal_.principal_distance_ * sqrt(sum_proj / sum_seg);
            }

            Point6D estimated_pose;
            estimated_pose.x = 0;
            estimated_pose.y = 0;
            estimated_pose.z = z;
            estimated_pose.xa = orientation[1];
            estimated_pose.ya = orientation[2];
            estimated_pose.za = orientation[0];

            emit poseEstimated(i, estimated_pose);
            
            c10::cuda::CUDACachingAllocator::emptyCache();
        }

        emit finished(true, "");
    } catch (const std::exception& e) {
        emit finished(false, QString::fromStdString(e.what()));
    } catch (...) {
        emit finished(false, "Unknown error during pose estimation");
    }
}

} // namespace jta_gui
