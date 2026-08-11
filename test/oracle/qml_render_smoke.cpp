// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// QML render smoke (plan 005 U1 spike gate / R15): loads spike.qml through
// QQmlApplicationEngine (the REAL QML path: qmlRegisterType resolution,
// qrc/import resolution, QQuickVTKItem ownership inside a QML scene) and
// verifies QQuickVTKItem renders + re-renders + interacts on this box's
// xcb + Qt 6.7.2 + VTK 9.3.
//
// Capture is QQuickWindow::grabWindow (vtkWindowToImageFilter is a documented
// segfault on this build - see docs/solutions/tooling-decisions/
// jtml-rendering-runtime-xcb-qvtk-2026-08-10.md); the first grab is gated on
// rendered frames (grabWindow before the first frame returns an empty image).
//
// Checks:
//   1. the QML scene renders a non-blank frame (STL silhouette => gray
//      stddev + center-region variance above thresholds; a failed STL load
//      leaves only the small banner and fails the check);
//   2. a pose update dispatched from the app thread via dispatch_async
//      re-renders (captures differ - the dynamic render-thread leg is
//      retired at the gate, not at U3);
//   3. mouse drag (QTest through the Qt Quick delivery pipeline) rotates the
//      trackball camera (camera position read back via dispatch_async +
//      pixel diff);
//   4. no factory/autoinit errors (vtk_module_autoinit is linked; class
//      names are logged for the record).
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
#include <string>

#include <QEventLoop>
#include <QGuiApplication>
#include <QImage>
#include <QQmlApplicationEngine>
#include <QQuickWindow>
#include <QSurfaceFormat>
#include <QTimer>
#include <QUrl>
#include <QtTest/QTest>
#include <QVTKOpenGLNativeWidget.h>
#include <QQuickVTKItem.h>
#include <vtkNew.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include "SpikeVtkItem.h"

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

}  // namespace

int main(int argc, char** argv) {
    // Same pre-app GL setup as the spike (and the widgets app): default
    // surface format + setGraphicsApi (OpenGLRhi scenegraph) before
    // QGuiApplication.
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    QQuickVTKItem::setGraphicsApi();
    QGuiApplication app(argc, argv);

    qmlRegisterType<SpikeVtkItem>("jtml.experimental", 1, 0, "SpikeVtkItem");

    QQmlApplicationEngine engine;
    engine.load(QUrl(QStringLiteral("qrc:/spike.qml")));
    if (engine.rootObjects().isEmpty()) {
        std::cerr << "[qml-render-smoke] FAIL: QQmlApplicationEngine could not "
                     "load qrc:/spike.qml (type registration / qrc / import "
                     "resolution)\n";
        return 1;
    }
    auto* window = qobject_cast<QQuickWindow*>(engine.rootObjects().first());
    auto* spike = engine.rootObjects().first()->findChild<SpikeVtkItem*>();
    if (!window || !spike) {
        std::cerr << "[qml-render-smoke] FAIL: QML root is not a QQuickWindow "
                     "or SpikeVtkItem not found in the scene\n";
        return 1;
    }
    std::cout << "[qml-render-smoke] QML scene OK: root="
              << engine.rootObjects().first()->metaObject()->className()
              << " spikeItem=found stl=" << spike->stlPath() << "\n";

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

    window->show();
    window->requestActivate();
    app.processEvents();

    // Edge-case documentation: grab before the first rendered frame.
    const QImage preGrab = window->grabWindow();
    std::cout << "[qml-render-smoke] grabWindow before first frame: "
              << (preGrab.isNull() ? "null" : "non-null") << " "
              << preGrab.width() << "x" << preGrab.height() << "\n";

    // ---- Leg 1: renders non-blank under xcb --------------------------------
    if (!WaitForRenderedFrames(window, 3)) {
        std::cerr << "[qml-render-smoke] FAIL: no rendered frame within "
                     "timeout (render loop dead?)\n";
        return 1;
    }
    app.processEvents();
    QImage frame1 = window->grabWindow();
    if (frame1.isNull()) {
        std::cerr << "[qml-render-smoke] FAIL: grabWindow returned null after "
                     "rendered frames\n";
        return 1;
    }
    SavePng(frame1, out_dir + "/frame0-vtk.png");
    const ImageStats s1 = GrayStats(frame1);
    const int cw = frame1.width() / 2;
    const int ch = frame1.height() / 2;
    const ImageStats sCenter =
        GrayStats(frame1.copy(cw / 2, ch / 2, cw, ch));
    std::cout << "[qml-render-smoke] frame0: " << frame1.width() << "x"
              << frame1.height() << " gray mean=" << s1.mean
              << " stddev=" << s1.stddev
              << " center-stddev=" << sCenter.stddev << "\n";
    if (s1.stddev < 20.0 || sCenter.stddev < 10.0) {
        std::cerr << "[qml-render-smoke] FAIL: blank render (stddev="
                  << s1.stddev << ", center=" << sCenter.stddev
                  << ") - VTK silhouette missing\n";
        return 1;
    }
    std::cout << "[qml-render-smoke] leg1 render: NON-BLANK (ok)\n";

    // ---- Leg 2: dispatch_async pose update re-renders ----------------------
    spike->setActorOffset(30.0, 0.0, 0.0);
    if (!WaitForRenderedFrames(window, 2)) {
        std::cerr << "[qml-render-smoke] FAIL: no frame after dispatch_async "
                     "update within timeout\n";
        return 1;
    }
    app.processEvents();
    const QImage frame2 = window->grabWindow();
    SavePng(frame2, out_dir + "/frame1-offset.png");
    const double diff2 = DiffFraction(frame1, frame2, 8);
    std::cout << "[qml-render-smoke] leg2 dispatch_async: diff-fraction="
              << diff2 << "\n";
    if (diff2 < 0.005) {
        std::cerr << "[qml-render-smoke] FAIL: dispatch_async pose update did "
                     "not re-render (captures identical)\n";
        return 1;
    }
    std::cout << "[qml-render-smoke] leg2 dispatch_async: RE-RENDERED (ok)\n";

    // ---- Leg 3: mouse interaction rotates the trackball camera -------------
    spike->sampleCamera();
    if (!WaitForRenderedFrames(window, 1)) {
        return 1;
    }
    const double camBefore = spike->lastCameraPositionX();
    const QPoint dragStart(frame1.width() / 2, frame1.height() / 2);
    QTest::mousePress(window, Qt::LeftButton, Qt::NoModifier, dragStart);
    for (int i = 1; i <= 6; ++i) {
        QTest::mouseMove(window,
            dragStart + QPoint(12 * i, 5 * i));
        QTest::qWait(16);
    }
    QTest::mouseRelease(window, Qt::LeftButton, Qt::NoModifier,
        dragStart + QPoint(72, 30));
    if (!WaitForRenderedFrames(window, 2)) {
        return 1;
    }
    app.processEvents();
    spike->sampleCamera();
    if (!WaitForRenderedFrames(window, 1)) {
        return 1;
    }
    const double camAfter = spike->lastCameraPositionX();
    const QImage frame3 = window->grabWindow();
    SavePng(frame3, out_dir + "/frame2-after-drag.png");
    const double diff3 = DiffFraction(frame2, frame3, 8);
    std::cout << "[qml-render-smoke] leg3 interaction: camera x "
              << camBefore << " -> " << camAfter << " (delta "
              << (camAfter - camBefore) << "), diff-fraction=" << diff3 << "\n";
    if (std::abs(camAfter - camBefore) < 10.0 && diff3 < 0.005) {
        std::cerr << "[qml-render-smoke] FAIL: mouse drag did not move the "
                     "camera nor change pixels (interaction dead)\n";
        return 1;
    }
    std::cout << "[qml-render-smoke] leg3 interaction: CAMERA MOVED (ok)\n";

    std::cout << "[qml-render-smoke] DONE: render + dispatch_async + "
                 "interaction all OK\n";
    return 0;
}
