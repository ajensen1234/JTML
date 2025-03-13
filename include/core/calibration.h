/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#ifndef CALIBRATION_H
#define CALIBRATION_H

/*Includes*/
#include "camera_calibration.h" //*Camera Calibration For Renderer (principal distance, principal x/y, pix pitch)
#include <cmath>

/*Vec 3*/
#include "data_structures_6D.h"
struct Vect_3 {
    Vect_3(float v_1, float v_2, float v_3) {
        v_1_ = v_1;
        v_2_ = v_2;
        v_3_ = v_3;
    }
    Vect_3() {
        v_1_ = 0;
        v_2_ = 0;
        v_3_ = 0;
    }

    /*Storage*/
    float v_1_;
    float v_2_;
    float v_3_;
};

/*3 by 3 Matrix*/
struct Matrix_3_3 {
    Matrix_3_3(
        float A_11,
        float A_12,
        float A_13,
        float A_21,
        float A_22,
        float A_23,
        float A_31,
        float A_32,
        float A_33) {
        A_11_ = A_11;
        A_12_ = A_12;
        A_13_ = A_13;
        A_21_ = A_21;
        A_22_ = A_22;
        A_23_ = A_23;
        A_31_ = A_31;
        A_32_ = A_32;
        A_33_ = A_33;
    };
    Matrix_3_3() {
        A_11_ = 0;
        A_12_ = 0;
        A_13_ = 0;
        A_21_ = 0;
        A_22_ = 0;
        A_23_ = 0;
        A_31_ = 0;
        A_32_ = 0;
        A_33_ = 0;
    }
    /*Storage*/
    float A_11_;
    float A_12_;
    float A_13_;
    float A_21_;
    float A_22_;
    float A_23_;
    float A_31_;
    float A_32_;
    float A_33_;

    /*Perform Transpose*/
    Matrix_3_3 tranpose() const {
        return Matrix_3_3(
            A_11_, A_21_, A_31_, A_12_, A_22_, A_32_, A_13_, A_23_, A_33_);
    };

    /*Matrix inverse for 3x3*/
    Matrix_3_3 inverse() const {
        float det = A_11_ * (A_22_ * A_33_ - A_23_ * A_32_) -
                    A_12_ * (A_21_ * A_33_ - A_23_ * A_31_) +
                    A_13_ * (A_21_ * A_32_ - A_22_ * A_31_);
        
        float invdet = 1 / det;
        
        return Matrix_3_3(
            (A_22_ * A_33_ - A_23_ * A_32_) * invdet,
            (A_13_ * A_32_ - A_12_ * A_33_) * invdet,
            (A_12_ * A_23_ - A_13_ * A_22_) * invdet,
            (A_23_ * A_31_ - A_21_ * A_33_) * invdet,
            (A_11_ * A_33_ - A_13_ * A_31_) * invdet,
            (A_13_ * A_21_ - A_11_ * A_23_) * invdet,
            (A_21_ * A_32_ - A_22_ * A_31_) * invdet,
            (A_12_ * A_31_ - A_11_ * A_32_) * invdet,
            (A_11_ * A_22_ - A_12_ * A_21_) * invdet
        );
    }

    // Matrix multiplication operator
    Matrix_3_3 operator*(const Matrix_3_3& other) const {
        return Matrix_3_3(
            A_11_ * other.A_11_ + A_12_ * other.A_21_ + A_13_ * other.A_31_,
            A_11_ * other.A_12_ + A_12_ * other.A_22_ + A_13_ * other.A_32_,
            A_11_ * other.A_13_ + A_12_ * other.A_23_ + A_13_ * other.A_33_,

            A_21_ * other.A_11_ + A_22_ * other.A_21_ + A_23_ * other.A_31_,
            A_21_ * other.A_12_ + A_22_ * other.A_22_ + A_23_ * other.A_32_,
            A_21_ * other.A_13_ + A_22_ * other.A_23_ + A_23_ * other.A_33_,

            A_31_ * other.A_11_ + A_32_ * other.A_21_ + A_33_ * other.A_31_,
            A_31_ * other.A_12_ + A_32_ * other.A_22_ + A_33_ * other.A_32_,
            A_31_ * other.A_13_ + A_32_ * other.A_23_ + A_33_ * other.A_33_);
    }

    // Vector multiplication operator
    Vect_3 operator*(const Vect_3& v) const {
        return Vect_3(
            A_11_ * v.v_1_ + A_12_ * v.v_2_ + A_13_ * v.v_3_,
            A_21_ * v.v_1_ + A_22_ * v.v_2_ + A_23_ * v.v_3_,
            A_31_ * v.v_1_ + A_32_ * v.v_2_ + A_33_ * v.v_3_
        );
    }
};

/*Denver Camera Calibration Structure*/
struct DenverCameraCalibration {
    DenverCameraCalibration() {
        width = height = 0;
        fx = fy = cx = cy = 0.0;
    }
    
    // Image properties
    int width;
    int height;
    
    // Camera intrinsics
    double fx;  // Focal length x
    double fy;  // Focal length y
    double cx;  // Principal point x
    double cy;  // Principal point y
    
    // Extrinsics 
    Matrix_3_3 rotation;    // 3x3 rotation matrix
    Vect_3 translation;     // 3D translation vector
};

struct Calibration {
    /* Constructors for Monoplane and Biplane*/
    Calibration(
        CameraCalibration monoplane_principal, std::string type = "UF") {
        biplane_calibration = false;
        camera_A_principal_ = monoplane_principal;
        type_ = type;
    };
    
    /**
     * @brief Constructor for UF style biplane calibration
     * @param biplane_A_principal Camera A calibration
     * @param biplane_B_principal Camera B calibration
     * @param origin_B Origin of Camera B relative to A
     * @param axes_B Rotation matrix for Camera B
     */
    Calibration(
        CameraCalibration biplane_A_principal,
        CameraCalibration biplane_B_principal,
        Vect_3 origin_B,
        Matrix_3_3 axes_B) {
        biplane_calibration = true;
        camera_A_principal_ = biplane_A_principal;
        camera_B_principal_ = biplane_B_principal;
        origin_B_ = origin_B;
        axes_B_ = axes_B;
        type_ = "UF";
    };

    /**
     * @brief Constructor for Denver style biplane calibration
     * @param cam1 First camera calibration
     * @param cam2 Second camera calibration 
     */
    Calibration(const DenverCameraCalibration& cam1, 
                const DenverCameraCalibration& cam2) {
        biplane_calibration = true;
        type_ = "Denver";
        
        // Convert camera 1 (reference camera)
        camera_A_principal_ = CameraCalibration(
            cam1.fx,    // principal_distance
            cam1.cx,    // principal_x 
            cam1.cy,    // principal_y
            1.0         // pixel_pitch (normalized)
        );
        
        // Convert camera 2 
        camera_B_principal_ = CameraCalibration(
            cam2.fx,
            cam2.cx,
            cam2.cy,
            1.0
        );
        
        // Calculate relative transformation between cameras
        Matrix_3_3 R1_inv = cam1.rotation.inverse();
        axes_B_ = cam2.rotation * R1_inv;  // Relative rotation
        
        // T2 - R2*R1^-1*T1 gives translation from cam1 to cam2  
        Vect_3 T1(cam1.translation.v_1_, cam1.translation.v_2_, cam1.translation.v_3_);
        origin_B_ = Vect_3(
            cam2.translation.v_1_ - (axes_B_ * T1).v_1_,
            cam2.translation.v_2_ - (axes_B_ * T1).v_2_, 
            cam2.translation.v_3_ - (axes_B_ * T1).v_3_
        );
    }
    
    Calibration() {
        biplane_calibration = false;
    };

    /*Calibrated For Biplane?*/
    bool biplane_calibration;

    /*Which group? (helps determine z-axis direction*/
    std::string type_;

    /*Storage*/
    CameraCalibration
        camera_A_principal_; /*used for both monoplane and biplane*/
    CameraCalibration camera_B_principal_; /*only used for biplane*/
    Vect_3 origin_B_; /*Origin of Camera B with respect o A which is assumed to
                         be at (0,0,0) */
    Matrix_3_3 axes_B_; /*Orthogonal Coordinate System of B where A is assumed
                           to have standard unit system*/

    /*Camera A Pose to Camera B Pose*/
    Point6D convert_Pose_A_to_Pose_B(Point6D poseA) {
        if (biplane_calibration) {
            /*Deal with Location*/
            Vect_3 location_B = axes_B_.tranpose() * 
                Vect_3(
                    poseA.x - origin_B_.v_1_,
                    poseA.y - origin_B_.v_2_,
                    poseA.z - origin_B_.v_3_);

            /*Deal with Orientation*/
            /*Construct ROtation Matrices for A: Rz, Rx, Ry
            Then Find R = Rz*Rx*Ry
            Then Tranform as R_B = Q'*R where Q is the axes_B_ matrix
            Then recover theta_x,y, and z for camera B (may not be unique)*/
            /*Convert To Rads*/
            float PI = 3.141592653589793238462643383279502884;
            float theta_x_A = poseA.xa * (PI / 180.0);
            float theta_y_A = poseA.ya * (PI / 180.0);
            float theta_z_A = poseA.za * (PI / 180.0);
            Matrix_3_3 R_x(
                1,
                0,
                0,
                0,
                cos(theta_x_A),
                -1 * sin(theta_x_A),
                0,
                sin(theta_x_A),
                cos(theta_x_A));
            Matrix_3_3 R_y(
                cos(theta_y_A),
                0,
                sin(theta_y_A),
                0,
                1,
                0,
                -1 * sin(theta_y_A),
                0,
                cos(theta_y_A));
            Matrix_3_3 R_z(
                cos(theta_z_A),
                -1 * sin(theta_z_A),
                0,
                sin(theta_z_A),
                cos(theta_z_A),
                0,
                0,
                0,
                1);
            Matrix_3_3 R = R_z * (R_x * R_y);
            Matrix_3_3 R_B = axes_B_.tranpose() * R;

            /*Algorithm To Recover Z - X - Y Euler Angles*/
            float theta_x_B, theta_y_B, theta_z_B;
            if (R_B.A_32_ < 1) {
                if (R_B.A_32_ > -1) {
                    theta_x_B = asin(R_B.A_32_);
                    theta_z_B = atan2(-1 * R_B.A_12_, R_B.A_22_);
                    theta_y_B = atan2(-1 * R_B.A_31_, R_B.A_33_);

                } else {
                    theta_x_B = -PI / 2.0;
                    theta_z_B = -1 * atan2(R_B.A_13_, R_B.A_11_);
                    theta_y_B = 0;
                }
            } else {
                theta_x_B = PI / 2.0;
                theta_z_B = atan2(R_B.A_13_, R_B.A_11_);
                theta_y_B = 0;
            }

            /*Return New Pose*/
            return Point6D(
                location_B.v_1_,
                location_B.v_2_,
                location_B.v_3_,
                theta_x_B * (180.0 / PI),
                theta_y_B * (180.0 / PI),
                theta_z_B * (180.0 / PI));
        } else
            return poseA; // Just return the same.
    };

    /*Camera B Pose to Camera A Pose*/
    Point6D convert_Pose_B_to_Pose_A(Point6D poseA) {
        if (biplane_calibration) {
            /*Deal with Location*/
            Vect_3 location_B = axes_B_ * Vect_3(poseA.x, poseA.y, poseA.z);
            location_B = Vect_3(
                location_B.v_1_ + origin_B_.v_1_,
                location_B.v_2_ + origin_B_.v_2_,
                location_B.v_3_ + origin_B_.v_3_);

            /*Deal with Orientation*/
            /*Construct ROtation Matrices for B: Rz, Rx, Ry
            Then Find R = Rz*Rx*Ry
            Then Tranform as R_B = Q'*R*Q where Q is the axes_B_ matrix
            Then recover theta_x,y, and z for camera B (may not be unique)*/
            /*Convert To Rads*/
            float PI = 3.141592653589793238462643383279502884;
            float theta_x_A = poseA.xa * (PI / 180.0);
            float theta_y_A = poseA.ya * (PI / 180.0);
            float theta_z_A = poseA.za * (PI / 180.0);
            Matrix_3_3 R_x(
                1,
                0,
                0,
                0,
                cos(theta_x_A),
                -1 * sin(theta_x_A),
                0,
                sin(theta_x_A),
                cos(theta_x_A));
            Matrix_3_3 R_y(
                cos(theta_y_A),
                0,
                sin(theta_y_A),
                0,
                1,
                0,
                -1 * sin(theta_y_A),
                0,
                cos(theta_y_A));
            Matrix_3_3 R_z(
                cos(theta_z_A),
                -1 * sin(theta_z_A),
                0,
                sin(theta_z_A),
                cos(theta_z_A),
                0,
                0,
                0,
                1);
            Matrix_3_3 R = R_z * (R_x * R_y);
            Matrix_3_3 R_B = axes_B_ * R;

            /*Algorithm To Recover Z - X - Y Euler Angles*/
            float theta_x_B, theta_y_B, theta_z_B;
            if (R_B.A_32_ < 1) {
                if (R_B.A_32_ > -1) {
                    theta_x_B = asin(R_B.A_32_);
                    theta_z_B = atan2(-1 * R_B.A_12_, R_B.A_22_);
                    theta_y_B = atan2(-1 * R_B.A_31_, R_B.A_33_);

                } else {
                    theta_x_B = -PI / 2.0;
                    theta_z_B = -1 * atan2(R_B.A_13_, R_B.A_11_);
                    theta_y_B = 0;
                }
            } else {
                theta_x_B = PI / 2.0;
                theta_z_B = atan2(R_B.A_13_, R_B.A_11_);
                theta_y_B = 0;
            }

            /*Return New Pose*/
            return Point6D(
                location_B.v_1_,
                location_B.v_2_,
                location_B.v_3_,
                theta_x_B * (180.0 / PI),
                theta_y_B * (180.0 / PI),
                theta_z_B * (180.0 / PI));
        } else
            return poseA; // Just return the same.
    };
};
#endif /* CALIBRATION_H */