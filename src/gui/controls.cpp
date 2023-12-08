// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "gui/controls.h"

// Controls JTA Popup CPP
Controls::Controls(QWidget* parent, Qt::WindowFlags flags)
    : QDialog(parent, flags) {
    ui.setupUi(this);
    setFixedSize(520, 500);

    center_scene = new QGraphicsScene;
    center_scene->setSceneRect(0, 0, 500, 1620);

    center_graph = new QGraphicsView();
    center_graph->setParent(this);
    center_graph->setScene(center_scene);
    center_graph->setGeometry(0, 0, 521, 520);
    center_graph->setFrameStyle(QFrame::NoFrame);
    center_graph->viewport()->setAutoFillBackground(false);
    center_graph->verticalScrollBar()->setValue(22);

    center_scene->clear();
    center_item = new QGraphicsPixmapItem(
        QPixmap(":/Controls/Resources/control_list_new.png"));
    center_scene->addItem(center_item);
}

Controls::~Controls() {
    delete center_graph;
    delete center_scene;
}
