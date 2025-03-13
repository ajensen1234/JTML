#pragma once

#include <QObject>
#include <QString>
#include "core/calibration.h"

class CalibrationManager : public QObject {
    Q_OBJECT

public:
    explicit CalibrationManager(QObject* parent = nullptr);
    
    bool loadCalibration(const QString& path, bool isDenver = false);
    bool loadDenverCalibration(const QString& cal1Path, const QString& cal2Path);
    const Calibration& getCurrentCalibration() const { return calibration_; }
    bool isBiplaneCalibrated() const { return calibrated_for_biplane_viewport_; }
    bool isMonoplaneCalibrated() const { return calibrated_for_monoplane_viewport_; }
    
    // Helper functions
    CameraCalibration getCameraA() const { return calibration_.camera_A_principal_; }
    CameraCalibration getCameraB() const { return calibration_.camera_B_principal_; }
    const std::string& getCalibrationType() const { return calibration_.type_; }

signals:
    void calibrationLoaded();
    void calibrationError(const QString& message);

private:
    bool loadUFCalibration(const QString& path);
    bool parseDenverCalibration(const QString& path, DenverCameraCalibration& cal);
    
    Calibration calibration_;
    bool calibrated_for_monoplane_viewport_ = false;
    bool calibrated_for_biplane_viewport_ = false;
};