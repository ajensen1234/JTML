#ifndef CONTROLS_H
#define CONTROLS_H

#include <qdialog.h>
#include <qevent.h>
#include <qgraphicsscene.h>
#include <qgraphicsview.h>
#include <qimage.h>
#include <qscrollbar.h>

#include <QGraphicsPixmapItem>

#include "ui_controls.h"

// Controls JTA Popup Header
class Controls : public QDialog {
    Q_OBJECT

   public:
    Controls(QWidget* parent = 0, Qt::WindowFlags flags = 0);
    ~Controls() override;

   private:
    Ui::controls ui;
    QGraphicsScene* center_scene;
    QGraphicsView* center_graph;
    QGraphicsPixmapItem* center_item;
};

#endif  // CONTROLS_H
