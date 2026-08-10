// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Minimal VTK standalone-render probe (post-003 debug).
//
// Tests whether VTK's OWN GL backend can render + capture on this machine,
// independent of the Viewer/app: an explicit vtkOpenGLRenderWindow (not via the
// object factory) + a vtkSphereSource actor, rendered and saved to
// render-smoke-output/probe-sphere.png.
//
// This isolates VTK+GL+W2I from our viewer state. If this renders but the
// Viewer smoke doesn't, the problem is in our wiring; if this also fails, VTK's
// standalone backend (X11/GLX) is the problem (this VTK is built with
// VTK_USE_X=OFF / EGL=OFF / OSMesa=OFF, so standalone windows may not work even
// though the app's QVTK (Qt-context) path does).

#include <filesystem>
#include <iostream>
#include <string>

#include <vtkActor.h>
#include <vtkNew.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkPNGWriter.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSphereSource.h>
#include <vtkSmartPointer.h>
#include <vtkWindowToImageFilter.h>

int main() {
    const std::string out_dir = "render-smoke-output";
    std::filesystem::create_directories(out_dir);

    auto rw = vtkSmartPointer<vtkRenderWindow>(vtkOpenGLRenderWindow::New());
    // ^ New() returns the base pointer via the factory; GetClassName() below
    // reveals the runtime type (vtkOpenGLRenderWindow = real GL backend, base
    // vtkRenderWindow = factory not registered).
    rw->OffScreenRenderingOn();
    rw->SetSize(512, 512);
    std::cout << "[probe] window class: " << rw->GetClassName() << "\n";

    auto ren = vtkSmartPointer<vtkRenderer>::New();
    auto src = vtkSmartPointer<vtkSphereSource>::New();
    auto map = vtkSmartPointer<vtkPolyDataMapper>::New();
    map->SetInputConnection(src->GetOutputPort());
    auto act = vtkSmartPointer<vtkActor>::New();
    act->SetMapper(map);
    ren->AddActor(act);
    rw->AddRenderer(ren);

    rw->Render();
    std::cout << "[probe] render returned; size=" << rw->GetSize()[0] << "x"
              << rw->GetSize()[1] << "\n";

    auto w2i = vtkSmartPointer<vtkWindowToImageFilter>::New();
    w2i->SetInput(rw);
    w2i->SetInputBufferTypeToRGB();
    w2i->ReadFrontBufferOff();
    w2i->Update();
    auto png = vtkSmartPointer<vtkPNGWriter>::New();
    png->SetFileName((out_dir + "/probe-sphere.png").c_str());
    png->SetInputConnection(w2i->GetOutputPort());
    png->Write();
    std::cout << "[probe] wrote " << out_dir << "/probe-sphere.png\n";
    std::cout << "[probe] DONE\n";
    return 0;
}
