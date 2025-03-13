#include "gui/calibration_manager.h"
#include <QFile>
#include <QTextStream>
#include <QStringList>

CalibrationManager::CalibrationManager(QObject* parent) 
    : QObject(parent) 
{
}

bool CalibrationManager::loadCalibration(const QString& path, bool isDenver) {
    if (isDenver) {
        emit calibrationError("Denver calibration requires two files");
        return false;
    }
    return loadUFCalibration(path);
}

bool CalibrationManager::loadUFCalibration(const QString& path) {
    QFile inputFile(path);
    if (!inputFile.open(QIODevice::ReadOnly)) {
        emit calibrationError("Could not open calibration file");
        return false;
    }

    QTextStream in(&inputFile);
    QStringList InputList = in.readAll().split(QRegExp("[\r\n]|,|\t| "), Qt::SkipEmptyParts);
    inputFile.close();

    if (InputList[0] == "JT_INTCALIB" || InputList[0] == "JTA_INTCALIB") {
        if (InputList[4].toDouble() == 0) {
            emit calibrationError("Pixel size (the last number in the calibration file) is specified as 0! This is impossible.");
            return false;
        }

        calibrated_for_monoplane_viewport_ = true;
        calibrated_for_biplane_viewport_ = false;
        
        CameraCalibration principal_calibration_file(
            InputList[1].toDouble(),
            -1 * InputList[2].toDouble(),  // Negative for offsets to make consistent with JointTrack
            -1 * InputList[3].toDouble(),
            InputList[4].toDouble());
        
        calibration_ = Calibration(principal_calibration_file, "UF");
        emit calibrationLoaded();
        return true;
    } else if (InputList[0] == "JTA_INTCALIB_BIPLANE") {
        if (InputList[4].toDouble() == 0 || InputList[8].toDouble() == 0) {
            emit calibrationError("Pixel size is specified as 0!");
            return false;
        }

        calibrated_for_monoplane_viewport_ = false;
        calibrated_for_biplane_viewport_ = true;

        CameraCalibration principal_calibration_file_A(
            InputList[1].toDouble(),
            -1 * InputList[2].toDouble(),
            -1 * InputList[3].toDouble(),
            InputList[4].toDouble());

        CameraCalibration principal_calibration_file_B(
            InputList[5].toDouble(),
            -1 * InputList[6].toDouble(),
            -1 * InputList[7].toDouble(),
            InputList[8].toDouble());

        Vect_3 origin_B(
            InputList[9].toDouble(),
            InputList[10].toDouble(),
            InputList[11].toDouble());

        Matrix_3_3 orthogonal_axes_B(
            InputList[12].toDouble(),
            InputList[13].toDouble(),
            InputList[14].toDouble(),
            InputList[15].toDouble(),
            InputList[16].toDouble(),
            InputList[17].toDouble(),
            InputList[18].toDouble(),
            InputList[19].toDouble(),
            InputList[20].toDouble());

        calibration_ = Calibration(
            principal_calibration_file_A,
            principal_calibration_file_B,
            origin_B,
            orthogonal_axes_B);

        emit calibrationLoaded();
        return true;
    }

    emit calibrationError("Invalid calibration file format");
    return false;
}

bool CalibrationManager::parseDenverCalibration(const QString& path, DenverCameraCalibration& cal) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        emit calibrationError("Could not open calibration file: " + path);
        return false;
    }
    
    QTextStream in(&file);
    QString line;
    
    // Skip "image" line
    line = in.readLine();
    
    // Skip blank line
    line = in.readLine();
    
    // Read image size
    line = in.readLine();
    QStringList sizes = line.split(",");
    if (sizes.size() != 2) {
        emit calibrationError("Invalid image size format in file: " + path);
        return false;
    }
    cal.width = sizes[0].toInt();
    cal.height = sizes[1].toInt();
    
    // Skip blank line and "camera matrix" line
    in.readLine();
    in.readLine();
    
    // Read camera matrix
    QStringList mat1 = in.readLine().split(",");
    QStringList mat2 = in.readLine().split(",");
    QStringList mat3 = in.readLine().split(",");
    
    if (mat1.size() != 3 || mat2.size() != 3 || mat3.size() != 3) {
        emit calibrationError("Invalid camera matrix format in file: " + path);
        return false;
    }
    
    cal.fx = mat1[0].toDouble();
    cal.fy = mat2[1].toDouble();
    cal.cx = mat1[2].toDouble();
    cal.cy = mat2[2].toDouble();
    
    // Skip blank line and "rotation" line
    in.readLine();
    in.readLine();
    
    // Read rotation matrix
    QList<double> rotVals;
    for (int i = 0; i < 3; i++) {
        QStringList rots = in.readLine().split(",");
        if (rots.size() != 3) {
            emit calibrationError("Invalid rotation matrix format in file: " + path);
            return false;
        }
        for (int j = 0; j < 3; j++) {
            rotVals.append(rots[j].toDouble());
        }
    }
    
    cal.rotation = Matrix_3_3(
        rotVals[0], rotVals[1], rotVals[2],
        rotVals[3], rotVals[4], rotVals[5],
        rotVals[6], rotVals[7], rotVals[8]
    );
    
    // Skip blank line and "translation" line
    in.readLine();
    in.readLine();
    
    // Read translation
    QList<double> transVals;
    for (int i = 0; i < 3; i++) {
        line = in.readLine();
        if (line.isEmpty()) {
            emit calibrationError("Invalid translation vector format in file: " + path);
            return false;
        }
        transVals.append(line.toDouble());
    }
    
    cal.translation = Vect_3(transVals[0], transVals[1], transVals[2]);
    
    file.close();
    return true;
}

bool CalibrationManager::loadDenverCalibration(const QString& cal1Path, const QString& cal2Path) {
    DenverCameraCalibration dcal1, dcal2;
    
    if (!parseDenverCalibration(cal1Path, dcal1)) {
        return false;
    }
    
    if (!parseDenverCalibration(cal2Path, dcal2)) {
        return false;
    }
    
    // Create calibration from Denver format
    calibration_ = Calibration(dcal1, dcal2);
    calibrated_for_monoplane_viewport_ = false;
    calibrated_for_biplane_viewport_ = true;
    
    emit calibrationLoaded();
    return true;
}