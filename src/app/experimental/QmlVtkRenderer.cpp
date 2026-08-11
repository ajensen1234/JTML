// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U3: QmlVtkRenderer implementation — see the header for the threading
// contract and the Viewer mirror table. All VTK state lives in QmlVtkData
// (the vtkUserData returned by initializeVTK) and is touched only inside
// initializeVTK / destroyingVTK / dispatch_async bodies.

#include "QmlVtkRenderer.h"

// Qt
#include <QDebug>
#include <QString>

// VTK
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkDataSetMapper.h>
#include <vtkImageData.h>
#include <vtkImageImport.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSTLReader.h>

// OpenCV (inversion of the background frame)
#include <opencv2/imgproc.hpp>

// services (existing STL load path: Model wraps vtkSTLReader, stl_reader
// validates binary/ascii — the widgets app loads models the same way).
#include "services/model.h"

namespace {

// The vtkUserData returned by initializeVTK: owns every VTK object in the
// pipeline, all of it render-thread-only. The QML SceneGraph can delete the
// underlying QSGNode at any moment (in which case initializeVTK runs again
// with a fresh QmlVtkData built from the app-thread mirror).
struct QmlVtkData : vtkObject {
    static QmlVtkData* New();
    vtkTypeMacro(QmlVtkData, vtkObject);

    // Background chain — Viewer::initialize_vtk_mappers mirror:
    // vtkImageImport -> vtkImageData -> vtkDataSetMapper -> vtkActor
    // (the widgets app uses vtkActor+vtkDataSetMapper, NOT vtkImageActor).
    vtkNew<vtkRenderer> backgroundRenderer;
    vtkNew<vtkRenderer> sceneRenderer;
    vtkNew<vtkImageImport> importer;
    vtkNew<vtkImageData> background;
    vtkNew<vtkDataSetMapper> imageMapper;
    vtkNew<vtkActor> imageActor;
    // Keeps the imported pixel buffer alive: vtkImageImport wraps the Mat
    // data without copying (mainscreen matToVTK semantics), so the buffer
    // must outlive the vtkImageData that points at it. Replaced only inside
    // ApplyBackground (always followed by importer->Update() in the same
    // dispatch body).
    cv::Mat backgroundMat;

    // Model chain — Viewer::load_3d_models_into_actor_and_mapper_list
    // mirror: vtkSTLReader -> vtkPolyDataMapper -> vtkActor.
    struct ModelActor {
        vtkSmartPointer<vtkSTLReader> reader;
        vtkSmartPointer<vtkPolyDataMapper> mapper;
        vtkSmartPointer<vtkActor> actor;
    };
    std::vector<ModelActor> models;
};

vtkStandardNewMacro(QmlVtkData);

// matToVTK semantics copied from mainscreen.cpp:72 (the helper is
// replicated in the experimental tree by design; mainscreen.cpp is
// untouched). `src` is the effective (possibly inverted) frame image;
// `data->backgroundMat` keeps the buffer alive for the importer.
void ApplyBackground(
    QmlVtkData* data, const cv::Mat& src, BackgroundMode mode) {
    if (src.empty()) {
        return;  // no frame yet: keep the current background
    }
    cv::Mat effective;
    if (mode == BackgroundMode::Inverted) {
        cv::bitwise_not(src, effective);
    } else {
        effective = src;
    }
    data->backgroundMat = effective;
    data->importer->SetDataSpacing(1, 1, 1);
    data->importer->SetDataOrigin(0, 0, 0);
    data->importer->SetWholeExtent(
        0, effective.cols - 1, 0, effective.rows - 1, 0, 0);
    data->importer->SetDataExtentToWholeExtent();
    data->importer->SetDataScalarTypeToUnsignedChar();
    data->importer->SetNumberOfScalarComponents(effective.channels());
    data->importer->SetImportVoidPointer(effective.data);
    data->importer->Modified();
    data->importer->Update();
}

// Viewer::place_image_actors_according_to_calibration mirror: the image is
// centered on the origin. Z = -focalLengthPx (the widgets app uses
// -fy*pixel_pitch): under the parallel background camera the projection does
// not depend on Z, but the image must sit inside the (0.1, 2*fy) clipping
// range rather than at the camera origin (near-plane clipped). Parallel
// scale = half the image height so the image fills the viewport height.
void ApplyCameraPlacement(
    QmlVtkData* data, const cv::Mat& src, double focalLengthPx) {
    if (src.empty()) {
        return;
    }
    data->imageActor->SetPosition(
        -0.5 * src.cols, -0.5 * src.rows, -focalLengthPx);
    vtkCamera* bgCam = data->backgroundRenderer->GetActiveCamera();
    bgCam->ParallelProjectionOn();
    bgCam->SetParallelScale(0.5 * src.rows);
}

// Viewer::setup_camera_calibration + load_renderers_into_render_window +
// calculate_and_set_viewing_angle_from_calibration mirror (single
// viewport; window-center/aspect calibration plumbing lands with U4).
void ApplyCameraParams(
    QmlVtkData* data, double viewAngleDeg, double focalLengthPx) {
    vtkCamera* bgCam = data->backgroundRenderer->GetActiveCamera();
    bgCam->SetFocalPoint(0, 0, -1);
    bgCam->SetPosition(0, 0, 0);
    bgCam->SetClippingRange(0.1, 2.0 * focalLengthPx);

    vtkCamera* sceneCam = data->sceneRenderer->GetActiveCamera();
    sceneCam->SetPosition(0, 0, 0);
    sceneCam->SetFocalPoint(0, 0, -1);
    sceneCam->SetViewAngle(viewAngleDeg);
    sceneCam->SetClippingRange(
        0.1 * focalLengthPx, 1.75 * focalLengthPx);
}

// Viewer::load_3d_models_into_actor_and_mapper_list mirror: rebuild the
// model actor list from the scene descriptors (STL path + pose). Called
// from initializeVTK and from the updateModels dispatch body — both on the
// render thread, so the Model parse (vtkSTLReader) happens there too.
void RebuildModels(QmlVtkData* data, const std::vector<SceneModel>& models) {
    data->sceneRenderer->RemoveAllViewProps();
    data->models.clear();
    data->models.reserve(models.size());
    for (const SceneModel& scene_model : models) {
        QmlVtkData::ModelActor ma;
        Model model(scene_model.path, scene_model.name, "BLANK");
        if (!model.initialized_correctly_) {
            qWarning() << "[qml-renderer] STL load FAILED:"
                       << QString::fromStdString(scene_model.path);
            continue;
        }
        ma.reader = model.cad_reader_;
        ma.mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        ma.mapper->SetInputConnection(ma.reader->GetOutputPort());
        ma.actor = vtkSmartPointer<vtkActor>::New();
        ma.actor->SetMapper(ma.mapper);
        ma.actor->GetProperty()->SetColor(0.93, 0.86, 0.67);  // Bisque
        ma.actor->SetPosition(
            scene_model.pose.x, scene_model.pose.y, scene_model.pose.z);
        ma.actor->SetOrientation(
            scene_model.pose.xa, scene_model.pose.ya, scene_model.pose.za);
        ma.actor->PickableOff();
        data->sceneRenderer->AddActor(ma.actor);
        data->models.push_back(std::move(ma));
    }
}

}  // namespace

QmlVtkRenderer::QmlVtkRenderer(QQuickItem* parent)
    : QQuickVTKItem(parent) {}

QQuickVTKItem::vtkUserData QmlVtkRenderer::initializeVTK(
    vtkRenderWindow* renderWindow) {
    vtkNew<QmlVtkData> data;

    // Background chain (Viewer::initialize_vtk_mappers).
    data->importer->SetOutput(data->background);
    data->imageMapper->SetInputData(data->background);
    data->imageActor->SetMapper(data->imageMapper);
    data->imageActor->SetPickable(0);
    data->backgroundRenderer->AddActor(data->imageActor);

    // Layered renderers (Viewer::load_renderers_into_render_window):
    // layer 0 = background (interactive off), layer 1 = scene (models).
    data->backgroundRenderer->SetLayer(0);
    data->backgroundRenderer->InteractiveOff();
    data->sceneRenderer->SetLayer(1);
    data->sceneRenderer->InteractiveOn();
    renderWindow->SetNumberOfLayers(2);
    renderWindow->AddRenderer(data->backgroundRenderer);
    renderWindow->AddRenderer(data->sceneRenderer);
    renderWindow->SetMultiSamples(0);

    // Full pipeline from the app-thread mirror (safe: the GUI thread is
    // blocked at the scene-graph sync point while updatePaintNode runs).
    ApplyCameraParams(
        data, scene_mirror_.cameraViewAngle(), scene_mirror_.focalLengthPx());
    ApplyBackground(
        data, scene_mirror_.backgroundImage(), scene_mirror_.backgroundMode());
    ApplyCameraPlacement(
        data, scene_mirror_.backgroundImage(), scene_mirror_.focalLengthPx());
    RebuildModels(data, scene_mirror_.models());

    return data;
}

void QmlVtkRenderer::destroyingVTK(
    vtkRenderWindow* renderWindow, vtkUserData userData) {
    Q_UNUSED(renderWindow);
    auto* data = QmlVtkData::SafeDownCast(userData);
    if (!data) {
        return;
    }
    // Drop the props before the refcount release (the QSGVtkObjectNode
    // nulls vtkUserData after destroyingVTK). Explicit for the record: the
    // rest of the pipeline dies with the userData refcount.
    data->backgroundRenderer->RemoveAllViewProps();
    data->sceneRenderer->RemoveAllViewProps();
}

void QmlVtkRenderer::setScene(ExperimentalScene* scene) {
    bound_scene_ = scene;
    copySceneMirror();
    refreshPoseReadout();
}

ExperimentalScene* QmlVtkRenderer::scene() const {
    return bound_scene_;
}

void QmlVtkRenderer::applyScene() {
    if (!bound_scene_) {
        return;
    }
    copySceneMirror();
    refreshPoseReadout();
    const cv::Mat bg = scene_mirror_.backgroundImage();
    const BackgroundMode mode = scene_mirror_.backgroundMode();
    const std::vector<SceneModel> models = scene_mirror_.models();
    const double viewAngle = scene_mirror_.cameraViewAngle();
    const double focal = scene_mirror_.focalLengthPx();
    dispatch_async(
        [bg, mode, models, viewAngle, focal](
            vtkRenderWindow* renderWindow, vtkUserData userData) {
            auto* data = QmlVtkData::SafeDownCast(userData);
            if (!data) {
                return;
            }
            ApplyBackground(data, bg, mode);
            ApplyCameraPlacement(data, bg, focal);
            ApplyCameraParams(data, viewAngle, focal);
            RebuildModels(data, models);
            renderWindow->Render();
        });
}

void QmlVtkRenderer::updateBackground() {
    if (!bound_scene_) {
        return;
    }
    copySceneMirror();
    const cv::Mat bg = scene_mirror_.backgroundImage();
    const BackgroundMode mode = scene_mirror_.backgroundMode();
    const double scene_mirror_focal = scene_mirror_.focalLengthPx();
    dispatch_async(
        [bg, mode, scene_mirror_focal](
            vtkRenderWindow* renderWindow, vtkUserData userData) {
            auto* data = QmlVtkData::SafeDownCast(userData);
            if (!data) {
                return;
            }
            ApplyBackground(data, bg, mode);
            ApplyCameraPlacement(data, bg, scene_mirror_focal);
            renderWindow->Render();
        });
}

void QmlVtkRenderer::updatePose(int modelIndex) {
    if (!bound_scene_) {
        return;
    }
    copySceneMirror();
    const std::vector<SceneModel> models = scene_mirror_.models();
    if (modelIndex < 0 || modelIndex >= static_cast<int>(models.size())) {
        return;
    }
    const Point6D pose = models[static_cast<size_t>(modelIndex)].pose;
    refreshPoseReadout();
    dispatch_async(
        [modelIndex, pose](vtkRenderWindow* renderWindow, vtkUserData userData) {
            auto* data = QmlVtkData::SafeDownCast(userData);
            if (!data || modelIndex < 0 ||
                modelIndex >= static_cast<int>(data->models.size())) {
                return;
            }
            vtkActor* actor = data->models[static_cast<size_t>(modelIndex)].actor;
            actor->SetPosition(pose.x, pose.y, pose.z);
            actor->SetOrientation(pose.xa, pose.ya, pose.za);
            renderWindow->Render();
        });
}

void QmlVtkRenderer::updateModels() {
    if (!bound_scene_) {
        return;
    }
    copySceneMirror();
    refreshPoseReadout();
    const std::vector<SceneModel> models = scene_mirror_.models();
    dispatch_async(
        [models](vtkRenderWindow* renderWindow, vtkUserData userData) {
            auto* data = QmlVtkData::SafeDownCast(userData);
            if (!data) {
                return;
            }
            RebuildModels(data, models);
            renderWindow->Render();
        });
}

void QmlVtkRenderer::updateCamera() {
    if (!bound_scene_) {
        return;
    }
    copySceneMirror();
    const double viewAngle = scene_mirror_.cameraViewAngle();
    const double focal = scene_mirror_.focalLengthPx();
    dispatch_async(
        [viewAngle, focal](vtkRenderWindow* renderWindow, vtkUserData userData) {
            auto* data = QmlVtkData::SafeDownCast(userData);
            if (!data) {
                return;
            }
            ApplyCameraParams(data, viewAngle, focal);
            renderWindow->Render();
        });
}

QString QmlVtkRenderer::poseReadout() const {
    return pose_readout_;
}

void QmlVtkRenderer::copySceneMirror() {
    if (bound_scene_) {
        scene_mirror_ = *bound_scene_;
    }
}

void QmlVtkRenderer::refreshPoseReadout() {
    const std::vector<SceneModel> models = scene_mirror_.models();
    if (models.empty()) {
        pose_readout_ = QString();
    } else {
        const Point6D& p = models[0].pose;
        pose_readout_ = QStringLiteral("x=%1 y=%2 z=%3 xa=%4 ya=%5 za=%6")
                            .arg(p.x)
                            .arg(p.y)
                            .arg(p.z)
                            .arg(p.xa)
                            .arg(p.ya)
                            .arg(p.za);
    }
    emit sceneChanged();
}
