#include "throughput_serial_pipeline.h"

#include <opencv2/imgcodecs.hpp>
#include "coordinator/optimizer_manager.h"
#include "compute/frame.h"
#include "services/model.h"
#include "services/calibration.h"
#include "compute/CostFunctionManager.h"
#include "compute/camera_calibration.h"
#include "compute/gpu_metrics.cuh"
#include "compute/gpu_model.cuh"
#include "compute/pose_matrix.h"
#include "compute/cost_capacity_service.cuh"
#include "compute/evaluation_context.h"
#include "compute/evaluation_executor.h"
#include "compute/graph_recipe.h"
#include "compute/bank_state.cuh"

#include <vector>
#include <string>

namespace throughput_serial {

namespace {

const std::string kStudyDir = "example_studies/Kneel_1/";
const std::string kBaseImage = kStudyDir + "1024/2806.tif";
const std::string kFemStl = kStudyDir + "KR_right_7_fem.stl";
const int kWidth = 1024;
const int kHeight = 1024;
const int kDevice = 0;

std::vector<unsigned char> MatToUchar(const cv::Mat& m) {
    std::vector<unsigned char> buf((size_t)m.rows * m.cols);
    for (int y = 0; y < m.rows; ++y) {
        const unsigned char* row = m.ptr<unsigned char>(y);
        std::copy(row, row + m.cols, buf.begin() + (size_t)y * m.cols);
    }
    return buf;
}

struct Pipeline {
    gpu_cost_function::GPUMetrics* metrics = nullptr;
    PoseMatrix* pose_storage = nullptr;
    gpu_cost_function::GPUModel* model = nullptr;
    Calibration calibration;
    jta_cost_function::CostFunctionManager* trunk = nullptr;
    std::vector<GPUEdgeFrame*> edge;
    std::vector<GPUDilatedFrame*> dilated;
    std::vector<GPUIntensityFrame*> intensity;
    std::vector<GPUFrame*> distance_maps;
    std::vector<GPUHeatmap*> heatmaps;
    std::vector<gpu_cost_function::GPUModel*> non_principal;
    Pipeline() = default;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&) noexcept = default;
    Pipeline& operator=(Pipeline&&) noexcept = default;
    ~Pipeline() {
        delete trunk;
        delete pose_storage;
        delete metrics;
        delete model;
        for (auto* p : edge) delete p;
        for (auto* p : dilated) delete p;
        for (auto* p : intensity) delete p;
        for (auto* p : distance_maps) delete p;
        for (auto* p : heatmaps) delete p;
    }
};

Pipeline BuildPipeline(const gpu_cost_function::CostCapacityService* cap = nullptr) {
    Frame frame(kBaseImage, 3, 0, 150, 6);
    frame.setCurvatureHeatmaps();
    Pipeline p;
    p.metrics = new gpu_cost_function::GPUMetrics();
    if (!p.metrics->IsInitializedCorrectly()) throw std::runtime_error("GPUMetrics failed");
    p.pose_storage = new PoseMatrix();
    auto edge_upload = MatToUchar(frame.GetEdgeImage());
    auto* edge = new GPUEdgeFrame(kWidth, kHeight, kDevice, edge_upload.data(), frame.GetHighThreshold(), frame.GetLowThreshold(), frame.GetAperture());
    if (!edge->IsInitializedCorrectly()) throw std::runtime_error("GPUEdgeFrame failed");
    p.edge.push_back(edge);
    auto dil_upload = MatToUchar(frame.GetDilationImage());
    auto* dilated = new GPUDilatedFrame(kWidth, kHeight, kDevice, dil_upload.data(), 6);
    if (!dilated->IsInitializedCorrectly()) throw std::runtime_error("GPUDilatedFrame failed");
    p.dilated.push_back(dilated);
    auto orig_upload = MatToUchar(frame.GetOriginalImage());
    auto inv_upload = MatToUchar(frame.GetInvertedImage());
    auto* intensity = new GPUIntensityFrame(kWidth, kHeight, kDevice, orig_upload.data(), false, inv_upload.data());
    if (!intensity->IsInitializedCorrectly()) throw std::runtime_error("GPUIntensityFrame failed");
    p.intensity.push_back(intensity);
    auto dist_upload = MatToUchar(frame.GetDistanceMap());
    auto* dm = new GPUFrame(kWidth, kHeight, kDevice, dist_upload.data());
    if (!dm->IsInitializedCorrectly()) throw std::runtime_error("GPUFrame failed");
    p.distance_maps.push_back(dm);
    auto* hm = new GPUHeatmap(kWidth, kHeight, kDevice, frame.GetNumCurvatureKeypoints(), frame.getCurvatureHeatmaps().data());
    if (!hm->IsInitializedCorrectly()) throw std::runtime_error("GPUHeatmap failed");
    p.heatmaps.push_back(hm);
    Model femur(kFemStl, "femur", "femur");
    if (!femur.initialized_correctly_) throw std::runtime_error("Model failed");
    int tri = static_cast<int>(femur.triangle_vertices_.size() / 9);
    if (tri <= 0) throw std::runtime_error("tri 0");
    CameraCalibration cam(1198.0f, -1.0f * 0.0f, -1.0f * 0.0f, 0.373f);
    Calibration calib(cam);
    p.calibration = calib;
    p.model = new gpu_cost_function::GPUModel("femur", true, kWidth, kHeight, kDevice, false, &femur.triangle_vertices_[0], &femur.triangle_normals_[0], tri, calib.camera_A_principal_, cap);
    if (!p.model->IsInitializedCorrectly()) throw std::runtime_error("GPUModel failed");
    p.trunk = new jta_cost_function::CostFunctionManager(Stage::Trunk);
    p.trunk->setActiveCostFunction("DIRECT_DILATION");
    p.trunk->updateCostFunctionParameterValues("DIRECT_DILATION", "Dilation", 6);
    if (p.trunk->getActiveCostFunctionClass()) {
        bool ok = p.trunk->getActiveCostFunctionClass()->setIntParameterValue("Dilation", 6);
        (void)ok;
    }
    p.trunk->UploadData(&p.edge, &p.dilated, &p.intensity, &p.edge, &p.dilated, &p.intensity, p.model, &p.non_principal, p.metrics, p.pose_storage, false);
    p.trunk->UploadDistanceMap(&p.distance_maps, &p.heatmaps);
    p.trunk->setCurrentFrameIndex(0);
    std::string err;
    if (!p.trunk->InitializeActiveCostFunction(err)) throw std::runtime_error("InitializeActiveCostFunction failed: " + err);
    return p;
}

} // anonymous

SerialContext::~SerialContext() {
    if (pipeline) {
        delete static_cast<Pipeline*>(pipeline);
    }
}
SerialContext::SerialContext(SerialContext&& o) noexcept : pipeline(o.pipeline), cost(std::move(o.cost)) { o.pipeline = nullptr; }
SerialContext& SerialContext::operator=(SerialContext&& o) noexcept {
    if (this != &o) {
        if (pipeline) delete static_cast<Pipeline*>(pipeline);
        pipeline = o.pipeline; o.pipeline = nullptr;
        cost = std::move(o.cost);
    }
    return *this;
}

SerialContext CreateSerialContext() {
    Pipeline* p = new Pipeline(BuildPipeline(nullptr));
    auto fn = jta::BuildGpuCostAdapter(p->model, p->calibration, *p->trunk);
    return SerialContext(static_cast<void*>(p), std::move(fn));
}

}
