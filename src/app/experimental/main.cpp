// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U2/U4: jtml_experimental main — the QML composition root.
//
// Repo pattern (spike + widgets app): the default GL surface format must be
// set before the app exists. setGraphicsApi() (which MUST also run before
// QGuiApplication) sets the Qt Quick scenegraph to OpenGLRhi + re-applies
// VTK's QML surface format on top.
//
// The qrc-embedded shell (qrc:/main.qml) is loaded through
// QQmlApplicationEngine. The QML-exposed surface:
//  - "jtml.experimental" QML module: QmlVtkRenderer (the U3 viewport
//    render seam — models at pose over the fluoro background under
//    QQuickVTKItem's render-thread contract);
//  - root-context properties: appBridge (the hub — owns the app dataset
//    ExperimentalSession + the thin per-seam adapters; U2 exposes counts +
//    placeholder signals, StudyBridge lands in U4, OptimizerBridge/MlBridge/
//    PoseBridge in U6/U7/U8), studyBridge (the U4 study-load adapter + the
//    delegate selection contract). The list models are NOT context
//    properties: StudyBridge owns them (fresh instances on dataset replace)
//    and main.qml binds studyBridge.frameListModel / modelListModel.

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickVTKItem.h>
#include <QSurfaceFormat>
#include <QUrl>
#include <QVTKOpenGLNativeWidget.h>

#include "AppBridge.h"
#include "ExperimentalScene.h"
#include "QmlVtkRenderer.h"
#include "StudyBridge.h"  // complete type: the setContextProperty QObject* overload needs it

int main(int argc, char* argv[]) {
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    QQuickVTKItem::setGraphicsApi();

    QGuiApplication app(argc, argv);

    qmlRegisterType<QmlVtkRenderer>("jtml.experimental", 1, 0, "QmlVtkRenderer");

    // App-owned scene (R7/R11): outlives the engine; the QML-created
    // renderer binds to it after load (U4).
    ExperimentalScene scene;

    // The QML-exposed hub. U2: dataset counts + placeholder signals only;
    // all behavior stays in the seams it delegates to (thinness rule).
    // U4: the hub owns the app dataset (ExperimentalSession — frames/models/
    // LocationStorage/calibration, R3) + the StudyBridge adapter.
    AppBridge app_bridge(&scene);

    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("appBridge", &app_bridge);
    engine.rootContext()->setContextProperty(
        "studyBridge", app_bridge.studyBridge());

    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty()) {
        return -1;
    }

    // Bind the QML-created viewport renderer to the app-owned scene (U3
    // setScene: the renderer copies the scene state at the scene-graph sync
    // point; later updates arrive via the StudyBridge scene signals).
    if (auto* renderer =
            engine.rootObjects().first()->findChild<QmlVtkRenderer*>()) {
        renderer->setScene(&scene);
    }

    return app.exec();
}
