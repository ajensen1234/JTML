/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#ifndef CONTROLS_H
#define CONTROLS_H

#include <qdialog.h>
#include <qevent.h>
#include <qgraphicsscene.h>
#include <qgraphicsview.h>
#include <qimage.h>
#include <qnamespace.h>
#include <qscrollbar.h>

#include <QGraphicsPixmapItem>

#include "ui_controls.h"

// Controls JTA Popup Header
class Controls : public QDialog {
    Q_OBJECT

   public:
    Controls(QWidget* parent = 0, Qt::WindowFlags flags = Qt::WindowFlags());
    ~Controls() override;

   private:
    Ui::controls ui;
    QGraphicsScene* center_scene;
    QGraphicsView* center_graph;
    QGraphicsPixmapItem* center_item;
};

#endif  // CONTROLS_H
