/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*RenderPipelineBuilder (plan 006 U4 / R9, AE4): the widget-free,
 * value-parameterized VTK pipeline recipe shared by BOTH front-ends — the
 * widgets Viewer (src/view/viewer.cpp) and the QML QmlVtkRenderer
 * (src/app/experimental/QmlVtkRenderer.cpp) drive the SAME free functions,
 * removing the QML 1:1 mirror (QmlVtkRenderer.h:21-37 pipeline table) and
 * absorbing the dead MainScreen::matToVTK semantics (mainscreen.cpp:72).
 *
 * Shape: stateless free functions taking VTK objects BY POINTER — never a
 * QObject owning VTK state, never holding app-owned state. The QML side
 * calls them inside initializeVTK / dispatch_async bodies with its
 * render-thread-owned objects (the render-thread contract stays QML-side);
 * the widgets side calls them from the GUI thread. Interactor styles stay
 * view-side (they differ: widgets trackball+picking vs the QML
 * PrimaryModelStyle implicit-pick workaround).
 *
 * Recipe:
 *   ① background chain construction + zero-copy import configuration —
 *      ConfigureBackgroundChain (importer -> imageData -> mapper -> actor ->
 *      renderer) + RefreshBackgroundImport (spacing/origin/extent/scalar
 *      type/channels/SetImportVoidPointer/Update — vtkImageImport wraps the
 *      Mat buffer WITHOUT copying, so callers must keep the buffer alive
 *      for as long as the imported image is displayed);
 *   ② model chain — BuildModelActor (reader output -> mapper -> actor ->
 *      scene renderer);
 *   ③ camera setup — SetupBackgroundCamera (parallel bg focal/position/
 *      clipping), PlaceBackgroundImage (bg actor placement + parallel
 *      scale), SetupSceneCameraFocal (perspective scene position + focal
 *      pivot), ApplySceneCameraProjection (scene view angle + clipping);
 *   ④ background refresh — RefreshBackgroundImport;
 *   ⑤ actor pose apply — ApplyActorPose (SetPosition + SetOrientation).
 *
 * Camera divergences between the views are PARAMETERS, not unified
 * behavior:
 *   (a) background Z placement: widgets -fy*pixel_pitch, QML
 *       -focalLengthPx — the zPlacement argument of PlaceBackgroundImage;
 *   (b) scene-camera focal pivot: widgets near-origin (0,0,±1 by
 *       calibration type), QML primary-model z (0,0,models[0].pose.z) —
 *       the focalZ argument of SetupSceneCameraFocal;
 *   (c) window-center/aspect calibration plumbing stays VIEW-side (it is a
 *       per-view camera-calibration concern, not shared-recipe scope).
 *
 * The pure mat -> import-parameter derivation (DeriveImageImportParams) is
 * a standalone function pinned by test/unit/render_pipeline_builder_test.cpp
 * (gray vs color, channel count, extent, spacing/origin, empty/zero-size
 * degenerates); the VTK-touching calls are exercised by the render smokes
 * (jtml.render_smoke / jtml.qml_render_smoke).*/

#ifndef RENDER_PIPELINE_BUILDER_H
#define RENDER_PIPELINE_BUILDER_H

#include <opencv2/core.hpp>

#include "domain/data_structures_6D.h"

/*VTK classes are touched by pointer only; complete types live in the .cpp
 * (and in the views' own TUs, which include the VTK headers they own).*/
class vtkActor;
class vtkAlgorithmOutput;
class vtkDataSetMapper;
class vtkImageData;
class vtkImageImport;
class vtkPolyDataMapper;
class vtkRenderWindow;
class vtkRenderer;

namespace jta {
namespace render_pipeline {

/*The import parameters DERIVED from a background Mat (pure — no VTK
 * touched). SetDataSpacing(1,1,1) / SetDataOrigin(0,0,0) /
 * SetDataScalarTypeToUnsignedChar are constants of the recipe; extent and
 * channel count come from the Mat. For an empty/zero-size Mat the extent
 * degenerates to (0,-1,0,-1) and channels follows the Mat's type — valid,
 * well-defined values, no crash (the refresh function skips the importer
 * for empty Mats).*/
struct ImageImportParams {
    int extentMinX = 0;
    int extentMaxX = -1; /* mat.cols - 1; -1 for an empty Mat */
    int extentMinY = 0;
    int extentMaxY = -1; /* mat.rows - 1; -1 for an empty Mat */
    int channels = 1;    /* mat.channels() */
    double spacing[3] = {1.0, 1.0, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};
};

/*Pure mat -> import-parameter derivation (the standalone, headless-tested
 * part of the recipe).*/
ImageImportParams DeriveImageImportParams(const cv::Mat& mat);

/*① Background chain construction: wires importer -> imageData -> mapper ->
 * actor -> backgroundRenderer (SetOutput, SetInputData, SetMapper,
 * SetPickable(0), AddActor) exactly as both views did inline. Call once at
 * pipeline construction (the widgets Viewer ctor / QML initializeVTK).*/
void ConfigureBackgroundChain(
    vtkImageImport* importer,
    vtkImageData* imageData,
    vtkDataSetMapper* mapper,
    vtkActor* actor,
    vtkRenderer* backgroundRenderer);

/*④ Background refresh: zero-copy import of `mat` into the configured
 * pipeline (spacing/origin/extent/scalar-type/channels/
 * SetImportVoidPointer(mat.data)/Modified/Update — the dead
 * MainScreen::matToVTK semantics, see mainscreen.cpp:72). The importer
 * wraps the buffer WITHOUT copying: the caller keeps `mat`'s buffer alive
 * for as long as the imported image is displayed (the QML side holds it in
 * QmlVtkData::backgroundMat). Empty Mat -> no-op (keeps the current
 * background).*/
void RefreshBackgroundImport(vtkImageImport* importer, const cv::Mat& mat);

/*Layered renderer setup shared by both views (widgets
 * load_renderers_into_render_window / QML initializeVTK): background on
 * layer 0 (interactive off), scene on layer 1 (interactive on),
 * SetNumberOfLayers(2), AddRenderer x2. View-only window settings
 * (e.g. QML's SetMultiSamples(0)) stay at the call sites.*/
void SetupLayeredRenderers(
    vtkRenderWindow* renderWindow,
    vtkRenderer* backgroundRenderer,
    vtkRenderer* sceneRenderer);

/*③ Camera setup — background renderer (widgets setup_camera_calibration /
 * QML ApplyCameraParams): focal (0,0,-1), position (0,0,0), clipping
 * (0.1, 2.0 * focalLengthPx).*/
void SetupBackgroundCamera(
    vtkRenderer* backgroundRenderer,
    double focalLengthPx);

/*③ Camera setup — background image placement (widgets
 * place_image_actors_according_to_calibration / QML ApplyCameraPlacement):
 * image actor at (-0.5*w, -0.5*h, zPlacement), parallel projection on,
 * parallel scale 0.5*h. zPlacement is camera divergence (a): the widgets
 * pass -fy*pixel_pitch, the QML side passes -focalLengthPx.*/
void PlaceBackgroundImage(
    vtkActor* imageActor,
    vtkRenderer* backgroundRenderer,
    int imgW,
    int imgH,
    double zPlacement);

/*③ Camera setup — scene-camera position + focal pivot (widgets
 * load_renderers_into_render_window / QML ApplyCameraParams + the QML
 * primary-model focus follow): position (0,0,0), focal (0,0,focalZ).
 * focalZ is camera divergence (b): the widgets pass the near-origin ±1
 * focal direction from the calibration type, the QML side passes the
 * primary model's z (-1 when no model).*/
void SetupSceneCameraFocal(vtkRenderer* sceneRenderer, double focalZ);

/*③ Camera setup — perspective scene projection (widgets
 * set_vtk_camera_from_calibration_and_image*_if_* / QML ApplyCameraParams):
 * view angle + clipping (0.1 * focalLengthPx, 1.75 * focalLengthPx). The
 * VIEW derives the angle (the widgets' atan2 formula is view-side math;
 * the QML scene carries it precomputed); window-center/aspect calibration
 * plumbing stays view-side (divergence (c)).*/
void ApplySceneCameraProjection(
    vtkRenderer* sceneRenderer,
    double viewAngleDeg,
    double focalLengthPx);

/*② Model chain: reader output -> mapper -> actor -> scene renderer (widgets
 * load_3d_models_into_actor_and_mapper_list / QML RebuildModels). View-side
 * extras (color, visibility, pickable, pose) stay at the call sites.*/
void BuildModelActor(
    vtkPolyDataMapper* mapper,
    vtkActor* actor,
    vtkAlgorithmOutput* readerOutput,
    vtkRenderer* sceneRenderer);

/*⑤ Actor pose apply: SetPosition + SetOrientation from a Point6D (the QML
 * updatePose/RebuildModels pose write; the widgets' separate per-index
 * position/orientation setters keep their call-site semantics).*/
void ApplyActorPose(vtkActor* actor, const Point6D& pose);

} /* namespace render_pipeline */
} /* namespace jta */

#endif /* RENDER_PIPELINE_BUILDER_H */
