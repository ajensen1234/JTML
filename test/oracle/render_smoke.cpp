// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Offscreen VTK rendering smoke for the Viewer display path (post-003 debug).
//
// MIRRORS THE APP'S ACTUAL MECHANISM: QVTKOpenGLNativeWidget + a
// vtkGenericOpenGLRenderWindow (the same path MainScreen uses - the app renders
// via Qt's GL context, NOT a standalone vtkRenderWindow). It drives the real
// Viewer exactly as MainScreen does after calibration (load_renderers_into_
// render_window -> setup_camera_calibration -> set_loaded_frames ->
// update_display_background_to_* -> place_image_actors_according_to_calibration
// -> render), then adds the fem/tib STL models. Renders and saves PNGs via
// Qt's widget->grab() (captures exactly what the app would display):
//   render-smoke-output/frame0-original.png
//   render-smoke-output/frame0-edge.png
//   render-smoke-output/frame0-original-models.png
//
// Run: ctest --test-dir .build -L render --output-on-failure
// The test env forces QT_QPA_PLATFORM=xcb (Wayland EGL is broken on this box).

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <QApplication>
#include <QSurfaceFormat>
#include <QVTKOpenGLNativeWidget.h>
#include <vtkGenericOpenGLRenderWindow.h>

#include "compute/frame.h"
#include "services/calibration.h"
#include "view/viewer.h"

namespace {

// Mirrors MainScreen's monoplane calibration parse (JT_INTCALIB: principal
// distance, px, py (negated for JointTrack convention), pixel pitch).
Calibration LoadMonoplaneCalibration(const std::string& path) {
    std::ifstream f(path);
    std::string all((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());
    std::vector<std::string> toks;
    std::string cur;
    auto flush = [&]() {
        if (!cur.empty()) toks.push_back(cur);
        cur.clear();
    };
    for (char c : all) {
        if (c == '\n' || c == '\r' || c == ',' || c == '\t' || c == ' ')
            flush();
        else
            cur.push_back(c);
    }
    flush();
    if (toks.empty() ||
        (toks[0] != "JT_INTCALIB" && toks[0] != "JTA_INTCALIB")) {
        std::cerr << "[render-smoke] bad calibration header: "
                  << (toks.empty() ? "<empty>" : toks[0]) << "\n";
        std::exit(1);
    }
    if (std::stod(toks[4]) == 0) {
        std::cerr << "[render-smoke] pixel pitch is 0\n";
        std::exit(1);
    }
    CameraCalibration cc(std::stod(toks[1]), -1.0 * std::stod(toks[2]),
                         -1.0 * std::stod(toks[3]), std::stod(toks[4]));
    return Calibration(cc);  // monoplane, type "UF"
}

}  // namespace

int main(int argc, char** argv) {
    // Qt6/VTK: the QVTK widget's default surface format must be set before
    // QApplication exists (same as the app's main.cpp).
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    QApplication app(argc, argv);

    const std::string study = "example_studies/Kneel_1";
    const std::string out_dir = "render-smoke-output";
    std::filesystem::create_directories(out_dir);

    // 1. Calibration (the app's monoplane path).
    Calibration cal = LoadMonoplaneCalibration(study + "/calibration.txt");
    std::cout << "[render-smoke] calibration: UF, principal_distance="
              << cal.camera_A_principal_.principal_distance_
              << ", pixel_pitch=" << cal.camera_A_principal_.pixel_pitch_
              << "\n";

    // 2. Top-level images -> Frames (Canny config matching the Tier-2 oracle).
    std::vector<std::string> tifs;
    for (const auto& e : std::filesystem::directory_iterator(study)) {
        if (e.path().extension() == ".tif" &&
            e.path().parent_path() == std::filesystem::path(study)) {
            tifs.push_back(e.path().string());
        }
    }
    std::sort(tifs.begin(), tifs.end());
    std::vector<Frame> frames;
    for (const auto& t : tifs) frames.emplace_back(t, 3, 0, 150, 6);
    std::cout << "[render-smoke] loaded " << frames.size() << " frames\n";
    if (frames.empty()) {
        std::cerr << "[render-smoke] no top-level .tif found in " << study
                  << "\n";
        return 1;
    }

    // 3. The APP'S mechanism: QVTKOpenGLNativeWidget + vtkGenericOpenGLRenderWindow.
    auto rw = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    auto widget = new QVTKOpenGLNativeWidget();
    widget->setRenderWindow(rw);
    widget->resize(1024, 1024);
    widget->show();  // realize the Qt GL context (under QT_QPA_PLATFORM=xcb)
    app.processEvents();

    // 4. Wire the real Viewer exactly as MainScreen does post-calibration.
    Viewer vw;
    vw.load_render_window(rw);
    vw.load_renderers_into_render_window(cal);
    vw.setup_camera_calibration(cal);
    vw.set_loaded_frames(frames);

    // 5. Display frame 0 original image (the load->select->display path).
    vw.update_display_background_to_original_image(0, /*CameraA=*/true);
    vw.place_image_actors_according_to_calibration(
        cal, frames[0].GetOriginalImage().cols,
        frames[0].GetOriginalImage().rows);
    rw->Render();
    app.processEvents();
    widget->grab().save(QString::fromStdString(out_dir + "/frame0-original.png"));
    std::cout << "[render-smoke] wrote " << out_dir
              << "/frame0-original.png\n";

    // 6. Edge image (alternate display mode).
    vw.update_display_background_to_edge_image(0, true);
    rw->Render();
    app.processEvents();
    widget->grab().save(QString::fromStdString(out_dir + "/frame0-edge.png"));
    std::cout << "[render-smoke] wrote " << out_dir << "/frame0-edge.png\n";

    // 7. Add the fem/tib STL models to the scene layer and re-render.
    vw.get_loaded_models()->emplace_back(study + "/KR_right_7_fem.stl", "fem",
                                         "BLANK");
    vw.get_loaded_models()->emplace_back(study + "/KR_right_6_tib.stl", "tib",
                                         "BLANK");
    vw.load_3d_models_into_actor_and_mapper_list();
    for (int i = 0; i < vw.get_number_of_model_actors(); ++i) {
        vw.change_model_opacity_to_original(i);
    }
    vw.update_display_background_to_original_image(0, true);
    rw->Render();
    app.processEvents();
    widget->grab().save(
        QString::fromStdString(out_dir + "/frame0-original-models.png"));
    std::cout << "[render-smoke] wrote " << out_dir
              << "/frame0-original-models.png\n";

    std::cout << "[render-smoke] frames=" << frames.size()
              << " model_actors=" << vw.get_number_of_model_actors()
              << " render_size=" << rw->GetSize()[0] << "x" << rw->GetSize()[1]
              << "\n";
    std::cout << "[render-smoke] DONE\n";
    return 0;
}
