// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U2: jtml_experimental main — the QML composition root.
//
// Repo pattern (spike + widgets app): the default GL surface format must be
// set before the app exists. setGraphicsApi() (which MUST also run before
// QGuiApplication) sets the Qt Quick scenegraph to OpenGLRhi + re-applies
// VTK's QML surface format on top.
//
// The qrc-embedded shell (qrc:/main.qml) is loaded through
// QQmlApplicationEngine. The QML-exposed surface:
//  - "jtml.experimental" QML module: SpikeVtkItem (the U2 viewport
//    placeholder; U3 replaces it with QmlVtkRenderer);
//  - root-context properties: appBridge (the hub — session/settings/pose
//    surfaces; U2 exposes counts + placeholder signals, the thin per-seam
//    adapters StudyBridge/OptimizerBridge/MlBridge/PoseBridge land in
//    U4/U6/U7/U8), frameListModel/modelListModel (the direct-compiled
//    widget-free list models; empty until U4's StudyBridge populates them).

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickVTKItem.h>
#include <QSurfaceFormat>
#include <QUrl>
#include <QVTKOpenGLNativeWidget.h>

#include "AppBridge.h"
#include "SpikeVtkItem.h"
#include "view/frame_list_model.h"
#include "view/model_list_model.h"

int main(int argc, char* argv[]) {
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    QQuickVTKItem::setGraphicsApi();

    QGuiApplication app(argc, argv);

    qmlRegisterType<SpikeVtkItem>("jtml.experimental", 1, 0, "SpikeVtkItem");

    // App-owned dataset view-models (R3): direct-compiled, jtml_view NOT
    // linked (R1). They outlive the engine (declared before it).
    FrameListModel frame_list_model;
    ModelListModel model_list_model;

    // The QML-exposed hub. U2: dataset counts + placeholder signals only;
    // all behavior stays in the seams it will delegate to (thinness rule).
    AppBridge app_bridge;

    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("appBridge", &app_bridge);
    engine.rootContext()->setContextProperty("frameListModel", &frame_list_model);
    engine.rootContext()->setContextProperty("modelListModel", &model_list_model);

    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty()) {
        return -1;
    }

    return app.exec();
}
