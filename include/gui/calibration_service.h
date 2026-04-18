#ifndef CALIBRATION_SERVICE_H
#define CALIBRATION_SERVICE_H

#pragma once

#include <QObject>
#include <QString>
#include <QStringList>

#include "core/calibration.h"
#include "core/session_context.h"

class QListWidget;
class QWidget;
class vtkRenderer;
class Viewer;

namespace jta_gui {

class CalibrationService : public QObject {
    Q_OBJECT

public:
    CalibrationService(
        Viewer* primary_viewer,
        Viewer* coronal_viewer,
        vtkRenderer* renderer,
        QListWidget* image_list_widget,
        QObject* parent = nullptr);

    void LoadCalibration(QWidget* parent, jta_core::SessionContext& context);

Q_SIGNALS:
    void calibrationLoaded(Calibration calibration);
    void error(QString message);

private:
    void ApplyCalibrationToViewers(jta_core::SessionContext& context);
    [[nodiscard]] static double CalculateViewingAngle(
        const jta_core::SessionContext& context,
        int height,
        bool camera_a);
    static bool TryLoadUfMonoplane(
        const QStringList& input_list,
        jta_core::SessionContext& context,
        QString& error_message);
    static bool TryLoadUfBiplane(
        const QStringList& input_list,
        jta_core::SessionContext& context,
        QString& error_message);
    static bool TryLoadDenver(
        const QStringList& input_list,
        jta_core::SessionContext& context,
        QString& error_message);

    Viewer* primary_viewer_;
    Viewer* coronal_viewer_;
    vtkRenderer* renderer_;
    QListWidget* image_list_widget_;
};

} // namespace jta_gui

#endif /* CALIBRATION_SERVICE_H */
