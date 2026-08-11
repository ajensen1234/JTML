// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "SpikeVtkItem.h"

// Qt
#include <QCoreApplication>
#include <QDebug>

// VTK
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSTLReader.h>

// services (existing STL load path: Model wraps vtkSTLReader, stl_reader
// validates binary/ascii).
#include "services/model.h"

#include <filesystem>

namespace {

// The vtkUserData returned by initializeVTK: owns every VTK object in the
// pipeline, all of it render-thread-only (the QML SceneGraph can delete the
// underlying QSGNode at any moment, in which case initializeVTK runs again).
struct SpikeData : vtkObject {
    static SpikeData* New();
    vtkTypeMacro(SpikeData, vtkObject);

    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkPolyDataMapper> mapper;
    vtkNew<vtkActor> actor;
    vtkSmartPointer<vtkSTLReader> reader;  // transferred from Model
};

vtkStandardNewMacro(SpikeData);

}  // namespace

SpikeVtkItem::SpikeVtkItem(QQuickItem* parent)
    : QQuickVTKItem(parent), stl_path_(ResolveFemStlPath()) {}

QQuickVTKItem::vtkUserData SpikeVtkItem::initializeVTK(
    vtkRenderWindow* renderWindow) {
    vtkNew<SpikeData> data;

    // Load the Kneel_1 femur STL through the existing services path. The
    // reader is transferred into the render-thread-owned user data so it
    // outlives this method (the mapper keeps a reference to the data object
    // either way; keeping the pipeline whole is the documented contract).
    Model fem(stl_path_, "fem", "BLANK");
    if (!fem.initialized_correctly_) {
        qWarning() << "[spike] STL load FAILED:" << QString::fromStdString(stl_path_);
    } else {
        data->reader = fem.cad_reader_;
        data->mapper->SetInputConnection(data->reader->GetOutputPort());
        data->actor->SetMapper(data->mapper);
        data->actor->GetProperty()->SetColor(0.93, 0.86, 0.67);  // Bisque
        data->renderer->AddActor(data->actor);
    }

    data->renderer->SetBackground(0.09, 0.10, 0.13);
    data->renderer->ResetCamera();

    renderWindow->AddRenderer(data->renderer);
    renderWindow->SetMultiSamples(0);
    return data;
}

void SpikeVtkItem::setActorOffset(double x, double y, double z) {
    // Copy on the GUI thread, capture by value: the lambda runs on the Qt
    // Quick render thread inside updatePaintNode.
    dispatch_async(
        [x, y, z](vtkRenderWindow* renderWindow, vtkUserData userData) {
            auto* data = SpikeData::SafeDownCast(userData);
            if (!data) {
                return;
            }
            data->actor->SetPosition(x, y, z);
            renderWindow->Render();
        });
}

void SpikeVtkItem::sampleCamera() {
    dispatch_async(
        [this](vtkRenderWindow* renderWindow, vtkUserData userData) {
            Q_UNUSED(renderWindow);
            auto* data = SpikeData::SafeDownCast(userData);
            if (!data) {
                return;
            }
            camera_pos_x_.store(data->renderer->GetActiveCamera()->GetPosition()[0]);
        });
}

std::string SpikeVtkItem::ResolveFemStlPath() {
    const std::string rel = "example_studies/Kneel_1/KR_right_7_fem.stl";
    if (std::filesystem::exists(rel)) {
        return rel;  // ctest runs from the repo root
    }
    // Manual runs from .build/bin: climb to the repo root.
    const std::string fromApp =
        std::filesystem::path(QCoreApplication::applicationDirPath().toStdString())
            .parent_path()
            .parent_path()
            .string() +
        "/" + rel;
    if (std::filesystem::exists(fromApp)) {
        return fromApp;
    }
    qWarning() << "[spike] STL not found at" << QString::fromStdString(rel)
               << "nor" << QString::fromStdString(fromApp);
    return rel;  // let the load fail loudly with the canonical relative path
}
