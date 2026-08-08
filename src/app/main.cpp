// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include <QtWidgets/QApplication>
#include <QSurfaceFormat>
#include <QVTKOpenGLNativeWidget.h>

#include "view/mainscreen.h"

int main(int argc, char* argv[]) {
    /*QVTKOpenGLNativeWidget must have its default surface format set before any
     * QApplication exists (required for correct OpenGL context on Qt6).*/
    QSurfaceFormat::setDefaultFormat(
        QVTKOpenGLNativeWidget::defaultFormat());

    /*Otherwise Cant See TEXT*/
    QApplication a(argc, argv);
    MainScreen w;
    w.show();
    return a.exec();
}
