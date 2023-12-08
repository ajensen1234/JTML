// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include <QtWidgets/QApplication>

#include "gui/mainscreen.h"

int main(int argc, char* argv[]) {
    /*Otherwise Cant See TEXT*/
    QApplication a(argc, argv);
    MainScreen w;
    w.show();
    return a.exec();
}
