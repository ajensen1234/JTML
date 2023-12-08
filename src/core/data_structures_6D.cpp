// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "core/data_structures_6D.h"

/*Standard*/
#include <cmath>
#include <iostream>

Point6D::Point6D(double xval, double yval, double zval, double xaval,
                 double yaval, double zaval) {
    x = xval;
    y = yval;
    z = zval;
    xa = xaval;
    ya = yaval;
    za = zaval;
}

Point6D::Point6D() {
    x = 0;
    y = 0;
    z = 0;
    xa = 0;
    ya = 0;
    za = 0;
}

Point6D::Point6D(gpu_cost_function::Pose p) {
    this->x = p.x_location_;
    this->y = p.y_location_;
    this->z = p.z_location_;
    this->xa = p.x_angle_;
    this->ya = p.y_angle_;
    this->xa = p.z_angle_;
}

double Point6D::GetDistanceFrom(Point6D otherPoint) {
    return std::sqrt((otherPoint.x - x) * (otherPoint.x - x) +
                     (otherPoint.y - y) * (otherPoint.y - y) +
                     (otherPoint.z - z) * (otherPoint.z - z) +
                     (otherPoint.xa - xa) * (otherPoint.xa - xa) +
                     (otherPoint.ya - ya) * (otherPoint.ya - ya) +
                     (otherPoint.za - za) * (otherPoint.za - za));
}

Direction Point6D::GetLargestDirection() {
    double array_directions[6] = {x, y, z, xa, ya, za};
    double max_element =
        *std::max_element(array_directions, array_directions + 6);
    if (max_element == x) return X_DIRECTION;
    if (max_element == y) return Y_DIRECTION;
    if (max_element == z) return Z_DIRECTION;
    if (max_element == xa) return XA_DIRECTION;
    if (max_element == ya) return YA_DIRECTION;
    return ZA_DIRECTION;
}

double Point6D::GetDirection(Direction direction) {
    switch (direction) {
        case X_DIRECTION:
            return x;
            break;
        case Y_DIRECTION:
            return y;
            break;
        case Z_DIRECTION:
            return z;
            break;
        case XA_DIRECTION:
            return xa;
            break;
        case YA_DIRECTION:
            return ya;
            break;
        case ZA_DIRECTION:
            return za;
            break;
    }
}

void Point6D::UpdateDirection(Direction direction, double updated_value) {
    switch (direction) {
        case X_DIRECTION:
            x = updated_value;
            break;
        case Y_DIRECTION:
            y = updated_value;
            break;
        case Z_DIRECTION:
            z = updated_value;
            break;
        case XA_DIRECTION:
            xa = updated_value;
            break;
        case YA_DIRECTION:
            ya = updated_value;
            break;
        case ZA_DIRECTION:
            za = updated_value;
            break;
    }
}

HyperBox6D::HyperBox6D(double value, Point6D center, Point6D sides) {
    value_ = value;
    center_ = center;
    sides_ = sides;
    size_ =
        std::sqrt(sides_.x * sides_.x + sides_.y * sides_.y +
                  sides_.z * sides_.z + sides_.xa * sides_.xa +
                  sides_.ya * sides_.ya + sides_.za * sides_.za); /*L2 Norm*/
}

HyperBox6D::HyperBox6D() {
    value_ = -1;
    center_ = Point6D(0, 0, 0, 0, 0, 0);
    sides_ = Point6D(0, 0, 0, 0, 0, 0);
    size_ =
        std::sqrt(sides_.x * sides_.x + sides_.y * sides_.y +
                  sides_.z * sides_.z + sides_.xa * sides_.xa +
                  sides_.ya * sides_.ya + sides_.za * sides_.za); /*L2 Norm*/
}

void HyperBox6D::SetSides(Point6D new_sides) {
    sides_ = new_sides;
    size_ =
        std::sqrt(sides_.x * sides_.x + sides_.y * sides_.y +
                  sides_.z * sides_.z + sides_.xa * sides_.xa +
                  sides_.ya * sides_.ya + sides_.za * sides_.za); /*L2 Norm*/
}

Point6D HyperBox6D::GetSides() { return sides_; }

bool HyperBox6D::containsPoint(Point6D point) {
    if (((center_.x - 0.5 * sides_.x) <= point.x &&
         point.x <= (center_.x + 0.5 * sides_.x)) &&
        ((center_.y - 0.5 * sides_.y) <= point.y &&
         point.y <= (center_.y + 0.5 * sides_.y)) &&
        ((center_.z - 0.5 * sides_.z) <= point.z &&
         point.z <= (center_.z + 0.5 * sides_.z)) &&
        ((center_.xa - 0.5 * sides_.xa) <= point.xa &&
         point.xa <= (center_.xa + 0.5 * sides_.xa)) &&
        ((center_.ya - 0.5 * sides_.ya) <= point.ya &&
         point.ya <= (center_.ya + 0.5 * sides_.ya)) &&
        ((center_.za - 0.5 * sides_.za) <= point.za &&
         point.za <= (center_.za + 0.5 * sides_.za)))
        return true;
    return false;
}

void HyperBox6D::SetCenter(Point6D new_center) { center_ = new_center; }

Point6D HyperBox6D::GetCenter() { return center_; }

void HyperBox6D::TrisectSide(Direction trisect_side) {
    switch (trisect_side) {
        case X_DIRECTION:
            sides_.x = sides_.x / 3.0;
            break;
        case Y_DIRECTION:
            sides_.y = sides_.y / 3.0;
            break;
        case Z_DIRECTION:
            sides_.z = sides_.z / 3.0;
            break;
        case XA_DIRECTION:
            sides_.xa = sides_.xa / 3.0;
            break;
        case YA_DIRECTION:
            sides_.ya = sides_.ya / 3.0;
            break;
        case ZA_DIRECTION:
            sides_.za = sides_.za / 3.0;
            break;
    }
    size_ =
        std::sqrt(sides_.x * sides_.x + sides_.y * sides_.y +
                  sides_.z * sides_.z + sides_.xa * sides_.xa +
                  sides_.ya * sides_.ya + sides_.za * sides_.za); /*L2 Norm*/
}

void HyperBox6D::PrintCenter() {
    std::cout << "\nHyperBox Center: [\t" << center_.x << ",\t" << center_.y
              << ",\t" << center_.z << ",\t" << center_.xa << ",\t"
              << center_.ya << ",\t" << center_.za << "]";
}
