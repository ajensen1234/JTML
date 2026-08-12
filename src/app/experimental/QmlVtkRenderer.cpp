// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U3: QmlVtkRenderer implementation — see the header for the threading
// contract and the Viewer mirror table. All VTK state lives in QmlVtkData
// (the vtkUserData returned by initializeVTK) and is touched only inside
// initializeVTK / destroyingVTK / dispatch_async bodies.

#include "QmlVtkRenderer.h"

// Qt
#include <QDebug>
#include <QMetaObject>
#include <QString>
#include <QThread>

// VTK
#include <vtkActor.h>
#include <vtkCallbackCommand.h>  // complete type for GrabFocus (EventCallbackCommand is vtkCallbackCommand*)
#include <vtkCommand.h>
#include <vtkDataSetMapper.h>
#include <vtkImageData.h>
#include <vtkImageImport.h>
#include <vtkInteractorStyleTrackballActor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSTLReader.h>

// OpenCV (inversion of the background frame)
#include <opencv2/imgproc.hpp>

// services (existing STL load path: Model wraps vtkSTLReader, stl_reader
// validates binary/ascii — the widgets app loads models the same way) +
// the shared widget-free pipeline recipe (plan 006 U4: the Viewer 1:1
// mirror chains below now call these free functions; the render-thread
// contract is unchanged — all VTK objects stay QmlVtkData-owned and are
// touched only inside initializeVTK / destroyingVTK / dispatch_async).
#include "services/model.h"
#include "services/render_pipeline_builder.h"

namespace {

// Model-centric interaction (plan 005 feedback): a trackball-actor style
// whose picked actor IS the primary model — no hardware picking (the
// QQuickVTKItem pick-position bug tail: devicePixelRatio is never set on
// the QVTKInteractorAdapter in the pinned 9.3 integration, so pick
// positions can mis-locate). The widgets app's trackball-actor mode
// rotates the picked actor about its center; here the primary model is the
// implicit pick (FindPickedActor is non-virtual in VTK 9.3, so
// OnLeftButtonDown is overridden to mimic the base's successful-pick path).
class PrimaryModelStyle : public vtkInteractorStyleTrackballActor {
public:
    static PrimaryModelStyle* New();
    vtkTypeMacro(PrimaryModelStyle, vtkInteractorStyleTrackballActor);

    void SetPrimaryActor(vtkActor* actor, int sceneIndex) {
        primary_actor_ = actor;
        primary_actor_index_ = sceneIndex;
    }
    vtkActor* primaryActor() const { return primary_actor_; }
    int primaryActorIndex() const { return primary_actor_index_; }

protected:
    void OnLeftButtonDown() override {
        if (!primary_actor_) {
            return;
        }
        this->FindPokedRenderer(
            this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);
        this->InteractionProp = primary_actor_;  // the implicit pick
        if (this->CurrentRenderer == nullptr) {
            return;
        }
        this->GrabFocus(this->EventCallbackCommand);
        if (this->Interactor->GetShiftKey()) {
            this->StartPan();
        } else if (this->Interactor->GetControlKey()) {
            this->StartSpin();
        } else {
            this->StartRotate();
        }
    }

private:
    vtkActor* primary_actor_ = nullptr;
    /*Owner fix (2026-08-12): the EndInteraction observer reports the moved
     * actor's SCENE INDEX with its pose — the previous hardcoded 0 wrote
     * the drag into whichever model was scene index 0 (dragging the femur
     * visibly moved the tibia when the tibia was index 0). Kept in sync
     * with the actor by every SetPrimaryActor caller.*/
    int primary_actor_index_ = -1;
};

vtkStandardNewMacro(PrimaryModelStyle);

// Model-style EndInteraction observer (render thread): read the primary
// actor's transform and report it via the direct by-value signal emit
// (reportModelPoseAdjusted -> modelPoseAdjusted; AutoConnection queues
// delivery to GUI-thread receivers). The queued-invokeMethod variant is
// documented-broken on this stack (see
// docs/solutions/ui-bugs/jtml-qml-model-pose-sync-queued-functor-never-delivered-2026-08-11.md)
// — do NOT reintroduce it. Never touches app state here.
void OnModelStyleEndInteraction(
    vtkObject* caller, unsigned long, void* clientData, void*) {
    auto* style = static_cast<PrimaryModelStyle*>(caller);
    auto* renderer = static_cast<QmlVtkRenderer*>(clientData);
    vtkActor* actor = style->primaryActor();
    if (!actor || !renderer) {
        return;
    }
    double pos[3];
    double orient[3];
    actor->GetPosition(pos);
    actor->GetOrientation(orient);
    // Emit from whichever thread the observer runs on: the connections have
    // GUI-thread affinity, so AutoConnection queues the delivery.
    renderer->reportModelPoseAdjusted(
        style->primaryActorIndex(), pos[0], pos[1], pos[2], orient[0],
        orient[1], orient[2]);
}

// The vtkUserData returned by initializeVTK: owns every VTK object in the
// pipeline, all of it render-thread-only. The QML SceneGraph can delete the
// underlying QSGNode at any moment (in which case initializeVTK runs again
// with a fresh QmlVtkData built from the app-thread mirror).
struct QmlVtkData : vtkObject {
    static QmlVtkData* New();
    vtkTypeMacro(QmlVtkData, vtkObject);

    // Background chain — the shared jta::render_pipeline recipe
    // (ConfigureBackgroundChain): vtkImageImport -> vtkImageData ->
    // vtkDataSetMapper -> vtkActor (vtkActor+vtkDataSetMapper, NOT
    // vtkImageActor — matches the widgets app).
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

    // Model chain — the shared jta::render_pipeline recipe
    // (BuildModelActor): vtkSTLReader -> vtkPolyDataMapper -> vtkActor.
    struct ModelActor {
        vtkSmartPointer<vtkSTLReader> reader;
        vtkSmartPointer<vtkPolyDataMapper> mapper;
        vtkSmartPointer<vtkActor> actor;
    };
    std::vector<ModelActor> models;

    // Interactor styles (plan 005 feedback): camera-centric trackball
    // (QQuickVTKItem's default) and model-centric (rotates the primary
    // model). Swapped by setInteractionMode on the render thread.
    vtkNew<vtkInteractorStyleTrackballCamera> cameraStyle;
    vtkSmartPointer<PrimaryModelStyle> modelStyle;
    // The scene index the model-centric style moves (setActiveModel); the
    // camera-mode focal pivot follows it too. Survives RebuildModels.
    int activeModelIndex = 0;
    // EndInteraction observer on the model style: reads the primary actor's
    // transform (render thread) and posts the queued pose sync to the GUI
    // thread (plan-005 feedback #2).
    vtkNew<vtkCallbackCommand> styleEndObserver;
};

vtkStandardNewMacro(QmlVtkData);

/*The active-model clamp (review fix P2-5): every consumer (the camera
 * focal pivot, the model style's implicit pick, the pose-update pivot
 * compare) needs a VALID index, but a stale out-of-range activeModelIndex
 * can outlive a RebuildModels that shrank the model list. Invariant: a
 * negative index means "no primary selection" (the model style's implicit
 * pick is cleared) and is preserved; any out-of-range POSITIVE index
 * clamps to 0 (the pre-feedback default). RebuildModels writes the clamp
 * back so the stored index always satisfies the invariant.*/
int ClampedActiveIndex(const QmlVtkData* data, int size) {
    if (data->activeModelIndex >= 0 && data->activeModelIndex < size) {
        return data->activeModelIndex;
    }
    return data->activeModelIndex < 0 ? -1 : 0;
}

// Shared jta::render_pipeline::RefreshBackgroundImport (the matToVTK
// semantics — zero-copy import config). `src` is the effective (possibly
// inverted) frame image; `data->backgroundMat` keeps the buffer alive for
// the importer (the builder wraps it without copying). The empty guard is
// load-bearing HERE: with no frame it must keep the PREVIOUS background
// (buffer + importer untouched), not swap in an empty Mat.
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
    jta::render_pipeline::RefreshBackgroundImport(data->importer, effective);
}

// Shared jta::render_pipeline::PlaceBackgroundImage: the image is centered
// on the origin at Z = -focalLengthPx (the widgets app passes
// -fy*pixel_pitch — camera divergence (a) is the builder's zPlacement
// parameter). Under the parallel background camera the projection does not
// depend on Z, but the image must sit inside the (0.1, 2*fy) clipping range
// rather than at the camera origin (near-plane clipped). Parallel scale =
// half the image height so the image fills the viewport height.
void ApplyCameraPlacement(
    QmlVtkData* data, const cv::Mat& src, double focalLengthPx) {
    if (src.empty()) {
        return;
    }
    jta::render_pipeline::PlaceBackgroundImage(
        data->imageActor, data->backgroundRenderer, src.cols, src.rows,
        -focalLengthPx);
}

// Shared jta::render_pipeline camera recipe (widgets setup_camera_
// calibration + load_renderers_into_render_window + the scene view-angle/
// clipping from set_vtk_camera_from_calibration_*; single viewport). The
// window-center/aspect calibration plumbing is a QML-side camera-
// calibration concern, NOT shared-builder scope — the pre-U4 "lands with
// U4" expectation is STALE by design (camera divergence (c) stays
// view-side).
void ApplyCameraParams(
    QmlVtkData* data, double viewAngleDeg, double focalLengthPx) {
    jta::render_pipeline::SetupBackgroundCamera(
        data->backgroundRenderer, focalLengthPx);
    jta::render_pipeline::SetupSceneCameraFocal(data->sceneRenderer, -1.0);
    jta::render_pipeline::ApplySceneCameraProjection(
        data->sceneRenderer, viewAngleDeg, focalLengthPx);
}

// Camera-centric rotation pivots at the scene camera's focal point — pin it
// at the PRIMARY model so the view rotates around the model (the widgets
// app's camera mode uses a near-origin focal — camera divergence (b) is
// the builder's focalZ parameter; the plan-005 feedback made the
// off-model pivot explicit: rotate about the model).
void ApplyCameraFocus(QmlVtkData* data, const std::vector<SceneModel>& models) {
    /*The camera pivot follows the ACTIVE model (owner feedback 2026-08-11):
     * with several models loaded, rotating the view around the model you
     * selected beats always orbiting model 0.*/
    const int active = ClampedActiveIndex(data, static_cast<int>(models.size()));
    const double z = (models.empty() || active < 0)
                         ? -1.0
                         : models[static_cast<size_t>(active)].pose.z;
    jta::render_pipeline::SetupSceneCameraFocal(data->sceneRenderer, z);
}

// Shared jta::render_pipeline model chain (widgets
// load_3d_models_into_actor_and_mapper_list): rebuild the model actor list
// from the scene descriptors (STL path + pose). Called from initializeVTK
// and from the updateModels dispatch body — both on the render thread, so
// the Model parse (vtkSTLReader) happens there too. The reader/mapper/
// actor wiring is BuildModelActor; color + pickable stay view-side.
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
        ma.actor = vtkSmartPointer<vtkActor>::New();
        jta::render_pipeline::BuildModelActor(
            ma.mapper, ma.actor, ma.reader->GetOutputPort(),
            data->sceneRenderer);
        ma.actor->GetProperty()->SetColor(0.93, 0.86, 0.67);  // Bisque
        jta::render_pipeline::ApplyActorPose(ma.actor, scene_model.pose);
        ma.actor->PickableOff();
        data->models.push_back(std::move(ma));
    }
    // Keep the model-centric style's implicit pick in sync (harmless in
    // camera mode). The active model follows the session's PRIMARY
    // selection (setActiveModel) and survives rebuilds; falls back to
    // model 0 (the pre-feedback default) until the app selects something.
    // The clamp is WRITTEN BACK (review fix P2-5): a stale out-of-range
    // index from a rebuild that shrank the model list is persisted as the
    // clamped value, so every later consumer (updatePose's camera-pivot
    // compare included) sees a valid index.
    if (data->modelStyle) {
        const int active =
            ClampedActiveIndex(data, static_cast<int>(data->models.size()));
        data->activeModelIndex = active;  // pin the invariant
        data->modelStyle->SetPrimaryActor(
            (active < 0 || data->models.empty())
                ? nullptr
                : data->models[static_cast<size_t>(active)].actor,
            active);
    }
}

}  // namespace

QmlVtkRenderer::QmlVtkRenderer(QQuickItem* parent)
    : QQuickVTKItem(parent) {}

QQuickVTKItem::vtkUserData QmlVtkRenderer::initializeVTK(
    vtkRenderWindow* renderWindow) {
    vtkNew<QmlVtkData> data;

    // Background chain via the shared builder (widgets
    // initialize_vtk_mappers).
    jta::render_pipeline::ConfigureBackgroundChain(
        data->importer, data->background, data->imageMapper,
        data->imageActor, data->backgroundRenderer);

    // Layered renderers via the shared builder (widgets
    // load_renderers_into_render_window): layer 0 = background (interactive
    // off), layer 1 = scene (models). The QML-only window setting stays.
    jta::render_pipeline::SetupLayeredRenderers(
        renderWindow, data->backgroundRenderer, data->sceneRenderer);
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
    ApplyCameraFocus(data, scene_mirror_.models());

    // Interactor styles (plan 005 feedback): QQuickVTKItem's own wrapper
    // created the QVTKInteractor + a default trackball-camera style before
    // this override ran — swap in our two styles and apply the current
    // mode (RebuildModels above already re-pointed the model style's
    // implicit pick).
    data->modelStyle = vtkSmartPointer<PrimaryModelStyle>::New();
    // Pose sync observer (plan-005 feedback #2): EndInteraction on the
    // model style → render-thread transform read → queued GUI-thread emit.
    data->styleEndObserver->SetCallback(OnModelStyleEndInteraction);
    data->styleEndObserver->SetClientData(this);
    data->modelStyle->AddObserver(
        vtkCommand::EndInteractionEvent, data->styleEndObserver);
    vtkRenderWindowInteractor* iren = renderWindow->GetInteractor();
    if (iren) {
        iren->SetInteractorStyle(
            interaction_mode_ == ModelMode
                ? static_cast<vtkInteractorStyle*>(
                      data->modelStyle.Get())
                : static_cast<vtkInteractorStyle*>(data->cameraStyle.Get()));
    }

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
            ApplyCameraFocus(data, models);
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
            jta::render_pipeline::ApplyActorPose(actor, pose);
            // The rotation pivot follows the ACTIVE model (camera mode).
            if (modelIndex == data->activeModelIndex) {
                jta::render_pipeline::SetupSceneCameraFocal(
                    data->sceneRenderer, pose.z);
            }
            renderWindow->Render();
        });
}

void QmlVtkRenderer::setActiveModel(int sceneIndex) {
    /*Owner feedback 2026-08-11: the model-centric interactor moves the
     * session's PRIMARY model, not scene model 0. Render-thread hop via
     * dispatch_async (the style + actors are render-thread owned); a
     * negative index clears the implicit pick (nothing movable — the app
     * surfaces the selection state). The in-range selection goes through
     * the shared clamp helper (review fix P2-5) and is written back so the
     * stored index satisfies the invariant (the pick then always matches
     * the stored index).*/
    dispatch_async(
        [sceneIndex](vtkRenderWindow* renderWindow, vtkUserData userData) {
            auto* data = QmlVtkData::SafeDownCast(userData);
            if (!data || !data->modelStyle) {
                return;
            }
            data->activeModelIndex = sceneIndex;
            vtkActor* actor = nullptr;
            int active = -1;  // cleared pick: no movable model
            if (sceneIndex >= 0) {
                active = ClampedActiveIndex(
                    data, static_cast<int>(data->models.size()));
                data->activeModelIndex = active;  // pin the invariant
                if (!data->models.empty()) {
                    actor = data->models[static_cast<size_t>(active)].actor;
                }
            }
            data->modelStyle->SetPrimaryActor(actor, active);
            (void)renderWindow;
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
            ApplyCameraFocus(data, models);
            renderWindow->Render();
        });
}

void QmlVtkRenderer::setInteractionMode(int mode) {
    const int clamped =
        (mode == ModelMode) ? ModelMode : CameraMode;
    if (clamped == interaction_mode_) {
        return;
    }
    interaction_mode_ = clamped;
    emit interactionModeChanged();
    dispatch_async(
        [clamped](vtkRenderWindow* renderWindow, vtkUserData userData) {
            auto* data = QmlVtkData::SafeDownCast(userData);
            if (!data) {
                return;
            }
            vtkRenderWindowInteractor* iren = renderWindow->GetInteractor();
            if (!iren) {
                return;
            }
            iren->SetInteractorStyle(
                clamped == ModelMode
                    ? static_cast<vtkInteractorStyle*>(
                          data->modelStyle.Get())
                    : static_cast<vtkInteractorStyle*>(
                          data->cameraStyle.Get()));
            renderWindow->Render();
        });
}

int QmlVtkRenderer::interactionMode() const {
    return interaction_mode_;
}

void QmlVtkRenderer::reportModelPoseAdjusted(int sceneModelIndex, double x,
                                              double y, double z, double xa,
                                              double ya, double za) {
    // May be called from the render thread (the style's EndInteraction
    // observer): emit with by-value data; receivers with GUI-thread affinity
    // get queued delivery via AutoConnection. No VTK state is touched.
    emit modelPoseAdjusted(sceneModelIndex, x, y, z, xa, ya, za);
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
