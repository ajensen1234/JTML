// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U1 spike main: minimal QML window hosting SpikeVtkItem (QQuickVTKItem).
//
// Run (repo root, so example_studies/ resolves):
//   QT_QPA_PLATFORM=xcb .build/bin/jtml_experimental_spike
// Headless-capture twin: test/oracle/qml_render_smoke.cpp (ctest -R qml_render_smoke).

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQuickVTKItem.h>
#include <QSurfaceFormat>
#include <QTimer>
#include <QUrl>
#include <QVTKOpenGLNativeWidget.h>

#include "SpikeVtkItem.h"

int main(int argc, char* argv[]) {
    // Repo pattern: the default GL surface format must be set before the app
    // exists. setGraphicsApi() (which MUST also run before QGuiApplication)
    // sets the Qt Quick scenegraph to OpenGLRhi + re-applies VTK's QML
    // surface format on top.
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    QQuickVTKItem::setGraphicsApi();

    QGuiApplication app(argc, argv);

    qmlRegisterType<SpikeVtkItem>("jtml.experimental", 1, 0, "SpikeVtkItem");

    QQmlApplicationEngine engine;
    engine.load(QUrl(QStringLiteral("qrc:/spike.qml")));
    if (engine.rootObjects().isEmpty()) {
        return -1;
    }

    // Dynamic render-thread leg (manual verification): nudge the actor from
    // the GUI thread via dispatch_async so a human can confirm app-thread ->
    // render-thread updates re-render live.
    auto* spike = engine.rootObjects().first()->findChild<SpikeVtkItem*>();
    if (spike) {
        QTimer::singleShot(3000, spike, [spike]() { spike->setActorOffset(15.0, 0.0, 0.0); });
        QTimer::singleShot(5000, spike, [spike]() { spike->setActorOffset(0.0, 12.0, 0.0); });
        QTimer::singleShot(7000, spike, [spike]() { spike->setActorOffset(0.0, 0.0, 0.0); });
    }

    return app.exec();
}
