// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 006 U4: RenderPipelineBuilder implementation — the shared, widget-free VTK
// pipeline recipe (see the header for the full contract). Stateless free
// functions; every VTK object is caller-owned and passed by pointer, so the
// QML render-thread contract (all VTK objects created + touched inside
// initializeVTK / dispatch_async bodies) is preserved by construction.

#include "services/render_pipeline_builder.h"

#include <vtkActor.h>
#include <vtkAlgorithmOutput.h>
#include <vtkCamera.h>
#include <vtkDataSetMapper.h>
#include <vtkImageData.h>
#include <vtkImageImport.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>

namespace jta {
namespace render_pipeline {

ImageImportParams DeriveImageImportParams(const cv::Mat& mat) {
    ImageImportParams params;
    if (mat.empty()) {
        /* Degenerate but valid: extent (0,-1,0,-1), channels follows the
         * Mat's type (1 for the default Mat). Callers guard on empty before
         * touching the importer; this only has to be well-defined. */
        params.channels = mat.channels();
        return params;
    }
    params.extentMaxX = mat.cols - 1;
    params.extentMaxY = mat.rows - 1;
    params.channels = mat.channels();
    return params;
}

void ConfigureBackgroundChain(
    vtkImageImport* importer,
    vtkImageData* imageData,
    vtkDataSetMapper* mapper,
    vtkActor* actor,
    vtkRenderer* backgroundRenderer) {
    importer->SetOutput(imageData);
    mapper->SetInputData(imageData);
    actor->SetMapper(mapper);
    actor->SetPickable(0);
    backgroundRenderer->AddActor(actor);
}

void RefreshBackgroundImport(vtkImageImport* importer, const cv::Mat& mat) {
    if (importer == nullptr || mat.empty()) {
        /* No frame yet: keep the current background (the QML side relies on
         * this to keep its previous buffer alive; the widgets side never
         * passes an empty Mat). */
        return;
    }
    const ImageImportParams params = DeriveImageImportParams(mat);
    importer->SetDataSpacing(
        params.spacing[0], params.spacing[1], params.spacing[2]);
    importer->SetDataOrigin(
        params.origin[0], params.origin[1], params.origin[2]);
    importer->SetWholeExtent(
        params.extentMinX,
        params.extentMaxX,
        params.extentMinY,
        params.extentMaxY,
        0,
        0);
    importer->SetDataExtentToWholeExtent();
    importer->SetDataScalarTypeToUnsignedChar();
    importer->SetNumberOfScalarComponents(params.channels);
    importer->SetImportVoidPointer(mat.data);
    importer->Modified();
    importer->Update();
}

void SetupLayeredRenderers(
    vtkRenderWindow* renderWindow,
    vtkRenderer* backgroundRenderer,
    vtkRenderer* sceneRenderer) {
    backgroundRenderer->SetLayer(0);
    backgroundRenderer->InteractiveOff();
    sceneRenderer->SetLayer(1);
    sceneRenderer->InteractiveOn();
    renderWindow->SetNumberOfLayers(2);
    renderWindow->AddRenderer(backgroundRenderer);
    renderWindow->AddRenderer(sceneRenderer);
}

void SetupBackgroundCamera(
    vtkRenderer* backgroundRenderer,
    double focalLengthPx) {
    vtkCamera* camera = backgroundRenderer->GetActiveCamera();
    camera->SetFocalPoint(0, 0, -1);
    camera->SetPosition(0, 0, 0);
    camera->SetClippingRange(0.1, 2.0 * focalLengthPx);
}

void PlaceBackgroundImage(
    vtkActor* imageActor,
    vtkRenderer* backgroundRenderer,
    int imgW,
    int imgH,
    double zPlacement) {
    imageActor->SetPosition(-0.5 * imgW, -0.5 * imgH, zPlacement);
    vtkCamera* camera = backgroundRenderer->GetActiveCamera();
    camera->ParallelProjectionOn();
    camera->SetParallelScale(0.5 * imgH);
}

void SetupSceneCameraFocal(vtkRenderer* sceneRenderer, double focalZ) {
    vtkCamera* camera = sceneRenderer->GetActiveCamera();
    camera->SetPosition(0, 0, 0);
    camera->SetFocalPoint(0, 0, focalZ);
}

void ApplySceneCameraProjection(
    vtkRenderer* sceneRenderer,
    double viewAngleDeg,
    double focalLengthPx) {
    vtkCamera* camera = sceneRenderer->GetActiveCamera();
    camera->SetViewAngle(viewAngleDeg);
    camera->SetClippingRange(0.1 * focalLengthPx, 1.75 * focalLengthPx);
}

void BuildModelActor(
    vtkPolyDataMapper* mapper,
    vtkActor* actor,
    vtkAlgorithmOutput* readerOutput,
    vtkRenderer* sceneRenderer) {
    mapper->SetInputConnection(readerOutput);
    actor->SetMapper(mapper);
    sceneRenderer->AddActor(actor);
}

void ApplyActorPose(vtkActor* actor, const Point6D& pose) {
    actor->SetPosition(pose.x, pose.y, pose.z);
    actor->SetOrientation(pose.xa, pose.ya, pose.za);
}

} /* namespace render_pipeline */
} /* namespace jta */
