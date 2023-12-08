// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*Basic Linear Algebra Structures*/
#include "BasicLinearAlgebraStructures.h"

namespace basic_la {
XYPoint::XYPoint(double X, double Y) {
    X_ = X;
    Y_ = Y;
}

XYZPoint::XYZPoint(double X, double Y, double Z) {
    X_ = X;
    Y_ = Y;
    Z_ = Z;
}

RotationMatrixZXY::RotationMatrixZXY(double Z_angle, double X_angle,
                                     double Y_angle) {
    float cz = cos(Z_angle * 3.14159265358979323846 / 180.0);
    float sz = sin(Z_angle * 3.14159265358979323846 / 180.0);
    float cx = cos(X_angle * 3.14159265358979323846 / 180.0);
    float sx = sin(X_angle * 3.14159265358979323846 / 180.0);
    float cy = cos(Y_angle * 3.14159265358979323846 / 180.0);
    float sy = sin(Y_angle * 3.14159265358979323846 / 180.0);

    /* R*v = RzRxRy*v */
    rotation_00_ = cz * cy - sz * sx * sy;
    rotation_01_ = -1.0 * sz * cx;
    rotation_02_ = cz * sy + sz * cy * sx;
    rotation_10_ = sz * cy + cz * sx * sy;
    rotation_11_ = cz * cx;
    rotation_12_ = sz * sy - cz * cy * sx;
    rotation_20_ = -1.0 * cx * sy;
    rotation_21_ = sx;
    rotation_22_ = cx * cy;
}

XYZPoint RotationMatrixZXY::RotatePoint(XYZPoint point) {
    double rX = rotation_00_ * point.X_ + rotation_01_ * point.Y_ +
                rotation_02_ * point.Z_;
    double rY = rotation_10_ * point.X_ + rotation_11_ * point.Y_ +
                rotation_12_ * point.Z_;
    double rZ = rotation_20_ * point.X_ + rotation_21_ * point.Y_ +
                rotation_22_ * point.Z_;

    return XYZPoint(rX, rY, rZ);
}
}  // namespace basic_la