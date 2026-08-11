// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// QML render smoke (plan 005 U3 render seam / R15): loads renderer.qml
// through QQmlApplicationEngine (the REAL QML path: qmlRegisterType
// resolution, qrc/import resolution, QQuickVTKItem ownership inside a QML
// scene) and verifies QmlVtkRenderer renders models at pose over the fluoro
// background under the render-thread contract.
//
// Capture is QQuickWindow::grabWindow (vtkWindowToImageFilter is a documented
// segfault on this build - see docs/solutions/tooling-decisions/
// jtml-rendering-runtime-xcb-qvtk-2026-08-10.md); the first grab is gated on
// rendered frames (grabWindow before the first frame returns an empty image).
//
// The app-owned ExperimentalScene is the driver (R7/R11): the smoke mutates
// it on the GUI thread and calls the renderer's GUI-thread slots
// (applyScene/updatePose/updateBackground), which copy scene state to locals
// and dispatch by-value lambdas to the Qt Quick render thread.
//
// Checks (plan 005 U3 test scenarios):
//   a. background frame renders non-blank; adding one model at a known pose
//      over it renders the silhouette (captures differ);
//   b. pose update via the slot re-renders at the new pose (captures
//      differ) — the dynamic dispatch_async leg;
//   c. background swap (original/inverted) re-renders correctly;
//   d. destroying the item mid-update does not crash (dispatch_async after
//      destruction is safe — the scene state is app-owned, outliving the
//      item).
//
// PNG artifacts go to qml-render-smoke-output/ for human inspection.
//
// Run: ctest --test-dir .build -R qml_render_smoke --output-on-failure
// The test env forces QT_QPA_PLATFORM=xcb (the Wayland/EGL stack on this box
// cannot give a GL context - blank render; xcb is the app's working platform).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <QCoreApplication>
#include <QEventLoop>
#include <QGuiApplication>
#include <QImage>
#include <QPointer>
#include <QQmlApplicationEngine>
#include <QQmlEngine>
#include <QQuickWindow>
#include <QSurfaceFormat>
#include <QTimer>
#include <QThread>
#include <QUrl>
#include <QtTest/QTest>
#include <QVTKOpenGLNativeWidget.h>
#include <QQuickVTKItem.h>
#include <vtkNew.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include "ExperimentalScene.h"
#include "QmlVtkRenderer.h"

namespace {

struct ImageStats {
    double mean = 0.0;
    double stddev = 0.0;
};

ImageStats GrayStats(const QImage& img) {
    const QImage gray = img.convertToFormat(QImage::Format_Grayscale8);
    const auto* data = gray.constBits();
    const qsizetype n = gray.sizeInBytes();
    double sum = 0.0;
    for (qsizetype i = 0; i < n; ++i) {
        sum += data[i];
    }
    const double mean = n > 0 ? sum / n : 0.0;
    double sq = 0.0;
    for (qsizetype i = 0; i < n; ++i) {
        const double d = data[i] - mean;
        sq += d * d;
    }
    return {mean, n > 0 ? std::sqrt(sq / n) : 0.0};
}

// Fraction of pixels whose gray level differs by more than `thresh` between
// two frames (0..1).
double DiffFraction(const QImage& a, const QImage& b, int thresh) {
    const QImage ga = a.convertToFormat(QImage::Format_Grayscale8);
    const QImage gb = b.convertToFormat(QImage::Format_Grayscale8);
    const int w = std::min(ga.width(), gb.width());
    const int h = std::min(ga.height(), gb.height());
    if (w <= 0 || h <= 0) {
        return 1.0;
    }
    long changed = 0;
    for (int y = 0; y < h; ++y) {
        const auto* ra = ga.constScanLine(y);
        const auto* rb = gb.constScanLine(y);
        for (int x = 0; x < w; ++x) {
            if (std::abs(int(ra[x]) - int(rb[x])) > thresh) {
                ++changed;
            }
        }
    }
    return double(changed) / double(w * h);
}

// Waits until at least `frames` afterRendering signals have fired (delivered
// to the GUI thread via QueuedConnection; the QQuickVTKItem drives a
// self-sustaining render loop via the VTK WindowFrameEvent -> scheduleRender
// chain). Returns false on timeout. This is the expose/afterRendering gate:
// grabWindow before the first rendered frame returns an empty image.
bool WaitForRenderedFrames(QQuickWindow* window, int frames, int timeoutMs = 20000) {
    QEventLoop loop;
    int seen = 0;
    QObject::connect(window, &QQuickWindow::afterRendering, &loop,
        [&]() {
            if (++seen >= frames) {
                loop.quit();
            }
        },
        Qt::QueuedConnection);
    QTimer::singleShot(timeoutMs, &loop, &QEventLoop::quit);
    if (seen < frames) {
        loop.exec();
    }
    return seen >= frames;
}

bool SavePng(const QImage& img, const std::string& path) {
    if (!img.save(QString::fromStdString(path))) {
        std::cerr << "[qml-render-smoke] failed to write " << path << "\n";
        return false;
    }
    return true;
}

// Same resolution logic as the U1 spike: ctest runs from the repo root;
// manual runs from .build/bin climb to the repo root.
std::string ResolveFemStlPath() {
    const std::string rel = "example_studies/Kneel_1/KR_right_7_fem.stl";
    if (std::filesystem::exists(rel)) {
        return rel;
    }
    const std::string fromApp =
        std::filesystem::path(QCoreApplication::applicationDirPath().toStdString())
            .parent_path()
            .parent_path()
            .string() +
        "/" + rel;
    return std::filesystem::exists(fromApp) ? fromApp : rel;
}

// Synthetic fluoro stand-in: 480x480 vertical gray gradient (high stddev so
// the blank-render gate is meaningful).
cv::Mat MakeSyntheticBackground() {
    cv::Mat bg(480, 480, CV_8UC1);
    for (int y = 0; y < bg.rows; ++y) {
        const uchar v = static_cast<uchar>((y * 255) / (bg.rows - 1));
        for (int x = 0; x < bg.cols; ++x) {
            bg.at<uchar>(y, x) = v;
        }
    }
    return bg;
}

}  // namespace

int main(int argc, char** argv) {
    // Same pre-app GL setup as the spike (and the widgets app): default
    // surface format + setGraphicsApi (OpenGLRhi scenegraph) before
    // QGuiApplication.
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    QQuickVTKItem::setGraphicsApi();
    QGuiApplication app(argc, argv);

    qmlRegisterType<QmlVtkRenderer>("jtml.experimental", 1, 0, "QmlVtkRenderer");

    // Heap-owned engine: leg 4 tears the whole QML scene down mid-update
    // (the renderer item is JavaScriptOwnership — QML-created — so its
    // destruction is the engine's job, see leg 4). Declared before the
    // scene so the scene (app-owned) outlives it.
    auto engine = std::make_unique<QQmlApplicationEngine>();
    engine->load(QUrl(QStringLiteral("qrc:/renderer.qml")));
    if (engine->rootObjects().isEmpty()) {
        std::cerr << "[qml-render-smoke] FAIL: QQmlApplicationEngine could not "
                     "load qrc:/renderer.qml (type registration / qrc / import "
                     "resolution)\n";
        return 1;
    }
    auto* window = qobject_cast<QQuickWindow*>(engine->rootObjects().first());
    auto* renderer = engine->rootObjects().first()->findChild<QmlVtkRenderer*>();
    if (!window || !renderer) {
        std::cerr << "[qml-render-smoke] FAIL: QML root is not a QQuickWindow "
                     "or QmlVtkRenderer not found in the scene\n";
        return 1;
    }
    std::cout << "[qml-render-smoke] QML scene OK: root="
              << engine->rootObjects().first()->metaObject()->className()
              << " rendererItem=found\n";

    // Autoinit/factory diagnostic (for the record). On this build VTK_USE_X=OFF
    // and EGL/OSMesa are off: vtkRenderWindow::New() returning the base class
    // is EXPECTED even with autoinit - the QML path renders through Qt's GL
    // stack (vtkGenericOpenGLRenderWindow inside QQuickVTKItem). A missing
    // vtk_module_autoinit manifests as factory errors at render time, which
    // the render legs below would catch.
    vtkNew<vtkRenderWindow> rwProbe;
    vtkNew<vtkRenderWindowInteractor> irenProbe;
    std::cout << "[qml-render-smoke] autoinit probe: "
              << "vtkRenderWindow::New() -> " << rwProbe->GetClassName()
              << ", vtkRenderWindowInteractor::New() -> "
              << irenProbe->GetClassName() << "\n";

    const std::string out_dir = "qml-render-smoke-output";
    std::filesystem::create_directories(out_dir);

    // App-owned scene (R7/R11): outlives the renderer item on purpose — the
    // destroy-mid-update leg relies on the scene state being app-owned.
    const std::string fem_path = ResolveFemStlPath();
    if (!std::filesystem::exists(fem_path)) {
        std::cerr << "[qml-render-smoke] FAIL: femur STL not found at "
                  << fem_path << "\n";
        return 1;
    }
    ExperimentalScene scene;
    scene.setBackgroundImage(MakeSyntheticBackground());
    scene.setBackgroundMode(BackgroundMode::Original);
    scene.setCameraViewAngle(25.0);
    scene.setFocalLengthPx(1198.0);  // Kneel_1 fy

    window->show();
    window->requestActivate();
    renderer->setScene(&scene);
    renderer->applyScene();  // background only, no models yet
    app.processEvents();

    // Edge-case documentation: grab before the first rendered frame.
    const QImage preGrab = window->grabWindow();
    std::cout << "[qml-render-smoke] grabWindow before first frame: "
              << (preGrab.isNull() ? "null" : "non-null") << " "
              << preGrab.width() << "x" << preGrab.height() << "\n";

    // ---- Leg 1a: background frame renders non-blank ----------------------
    if (!WaitForRenderedFrames(window, 3)) {
        std::cerr << "[qml-render-smoke] FAIL: no rendered frame within "
                     "timeout (render loop dead?)\n";
        return 1;
    }
    app.processEvents();
    const QImage frameBg = window->grabWindow();
    if (frameBg.isNull()) {
        std::cerr << "[qml-render-smoke] FAIL: grabWindow returned null after "
                     "rendered frames\n";
        return 1;
    }
    SavePng(frameBg, out_dir + "/frame0-background.png");
    const ImageStats sBg = GrayStats(frameBg);
    std::cout << "[qml-render-smoke] leg1a background: " << frameBg.width()
              << "x" << frameBg.height() << " gray mean=" << sBg.mean
              << " stddev=" << sBg.stddev << "\n";
    if (sBg.stddev < 20.0) {
        std::cerr << "[qml-render-smoke] FAIL: blank render (stddev="
                  << sBg.stddev << ") - background frame missing\n";
        return 1;
    }
    std::cout << "[qml-render-smoke] leg1a background: NON-BLANK (ok)\n";

    // ---- Leg 1b: model at a known pose over the background ----------------
    // The femur STL is a small thin bone (~67x60x72 units): at z=-500 the
    // projected silhouette covers ~2-3% of the frame (well above the 0.005
    // diff threshold; the bbox matches the camera-math prediction), and the
    // crescent shape IS the solid bone (layered-rendering contract: the
    // layer-1 scene renderer preserves the layer-0 background color and
    // clears depth - vtkRenderer::SetLayer sets PreserveColorBuffer).
    scene.setModels({SceneModel{fem_path, "fem", Point6D(0.0, 0.0, -500.0, 0.0, 0.0, 0.0)}});
    renderer->applyScene();
    if (!WaitForRenderedFrames(window, 2)) {
        std::cerr << "[qml-render-smoke] FAIL: no frame after applyScene "
                     "(model) within timeout\n";
        return 1;
    }
    app.processEvents();
    const QImage frameModel = window->grabWindow();
    SavePng(frameModel, out_dir + "/frame1-model-pose1.png");
    const double diffModel = DiffFraction(frameBg, frameModel, 8);
    std::cout << "[qml-render-smoke] leg1b model-at-pose: diff-fraction="
              << diffModel << " poseReadout="
              << renderer->poseReadout().toStdString() << "\n";
    if (diffModel < 0.005) {
        std::cerr << "[qml-render-smoke] FAIL: adding the model did not change "
                     "the render (silhouette missing; STL load failed?)\n";
        return 1;
    }
    std::cout << "[qml-render-smoke] leg1b model-at-pose: SILHOUETTE (ok)\n";

    // ---- Leg 2: pose update via the slot re-renders ----------------------
    scene.setModelPose(0, Point6D(60.0, 0.0, -500.0, 0.0, 0.0, 0.0));
    renderer->updatePose(0);
    if (!WaitForRenderedFrames(window, 2)) {
        std::cerr << "[qml-render-smoke] FAIL: no frame after dispatch_async "
                     "pose update within timeout\n";
        return 1;
    }
    app.processEvents();
    const QImage framePose = window->grabWindow();
    SavePng(framePose, out_dir + "/frame2-model-pose2.png");
    const double diffPose = DiffFraction(frameModel, framePose, 8);
    std::cout << "[qml-render-smoke] leg2 pose update: diff-fraction="
              << diffPose << " poseReadout="
              << renderer->poseReadout().toStdString() << "\n";
    if (diffPose < 0.005) {
        std::cerr << "[qml-render-smoke] FAIL: dispatch_async pose update did "
                     "not re-render (captures identical)\n";
        return 1;
    }
    std::cout << "[qml-render-smoke] leg2 pose update: RE-RENDERED (ok)\n";

    // ---- Leg 3: background swap (original/inverted) re-renders -----------
    scene.setBackgroundMode(BackgroundMode::Inverted);
    renderer->updateBackground();
    if (!WaitForRenderedFrames(window, 2)) {
        std::cerr << "[qml-render-smoke] FAIL: no frame after background swap "
                     "within timeout\n";
        return 1;
    }
    app.processEvents();
    const QImage frameInv = window->grabWindow();
    SavePng(frameInv, out_dir + "/frame3-background-inverted.png");
    const double diffInv = DiffFraction(framePose, frameInv, 8);
    std::cout << "[qml-render-smoke] leg3 background swap: diff-fraction="
              << diffInv << "\n";
    if (diffInv < 0.1) {
        std::cerr << "[qml-render-smoke] FAIL: inverted background did not "
                     "re-render (diff-fraction=" << diffInv << ")\n";
        return 1;
    }
    std::cout << "[qml-render-smoke] leg3 background swap: RE-RENDERED (ok)\n";

    // ---- Leg 4: destroying the item mid-update does not crash -------------
    // The renderer is a QML-created item (JavaScriptOwnership — the QML
    // engine's GC owns it, so C++-side deleteLater never completes in this
    // no-top-level-loop harness). Destruction therefore goes through the
    // engine — the app's real window-close path: one more dispatch is
    // queued, then the whole scene is torn down. The queued lambda is
    // dropped with the item, the SG node teardown releases the VTK pipeline
    // (destroyingVTK) on the render thread, and the app-owned scene
    // outlives it all. No crash == pass.
    QPointer<QmlVtkRenderer> guard(renderer);
    std::cout << "[qml-render-smoke] leg4 item thread=" << renderer->thread()
              << " gui thread=" << QThread::currentThread() << " ownership="
              << (QQmlEngine::objectOwnership(renderer) == QQmlEngine::CppOwnership
                          ? "CppOwnership"
                          : "JavaScriptOwnership")
              << "\n";
    renderer->updatePose(0);
    engine.reset();
    if (guard) {
        std::cerr << "[qml-render-smoke] FAIL: renderer item survived the "
                     "engine teardown\n";
        return 1;
    }
    for (int i = 0; i < 20; ++i) {  // let any in-flight render settle
        app.processEvents();
        QTest::qWait(10);
    }
    std::cout << "[qml-render-smoke] leg4 destroy-mid-update: NO CRASH (ok)\n";

    std::cout << "[qml-render-smoke] DONE: background + silhouette + pose "
                 "update + background swap + destroy-mid-update all OK\n";
    return 0;
}
