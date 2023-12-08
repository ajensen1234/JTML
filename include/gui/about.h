/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#ifndef ABOUT_H
#define ABOUT_H

#include <qdialog.h>

#include "ui_about.h"

// About JTA Popup Header
class About : public QDialog {
    Q_OBJECT

   public:
    About(QWidget* parent = 0, Qt::WindowFlags flags = Qt::WindowFlags());
    ~About() override;
    void setVersion(int A, int B, int C);  // Sets Version Number Label

   private:
    Ui::aboutJTA ui;
};

#endif  // ABOUT_H
