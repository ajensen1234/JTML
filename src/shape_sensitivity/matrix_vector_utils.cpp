#include "matrix_vector_utils.h"

RotationMatrix rotation_nudge(Pose input_pose, float theta, std::string axis) {
    float cz = cos(input_pose.z_angle_ * 3.14159265358979323846f / 180.0f);
    float sz = sin(input_pose.z_angle_ * 3.14159265358979323846f / 180.0f);
    float cx = cos(input_pose.x_angle_ * 3.14159265358979323846f / 180.0f);
    float sx = sin(input_pose.x_angle_ * 3.14159265358979323846f / 180.0f);
    float cy = cos(input_pose.y_angle_ * 3.14159265358979323846f / 180.0f);
    float sy = sin(input_pose.y_angle_ * 3.14159265358979323846f / 180.0f);

    /* R*v = RzRxRy*v */
    RotationMatrix model_rotation_mat_ =
        RotationMatrix(cz * cy - sz * sx * sy, -1.0 * sz * cx,
                       cz * sy + sz * cy * sx, sz * cy + cz * sx * sy, cz * cx,
                       sz * sy - cz * cy * sx, -1.0 * cx * sy, sx, cx * cy);

    // Now, we need a match statement to determine which of the axis we are
    // rotation about

    float cx_nudge, sx_nudge, cy_nudge, sy_nudge, cz_nudge, sz_nudge;

    if (axis == "x") {
        cx_nudge = cos(theta * 3.14159265358979323846f / 180.0f);
        sx_nudge = sin(theta * 3.14159265358979323846f / 180.0f);
        cy_nudge = cos(0);
        sy_nudge = sin(0);
        cz_nudge = cos(0);
        sz_nudge = sin(0);

    } else if (axis == "y") {
        cx_nudge = cos(0);
        sx_nudge = sin(0);
        cy_nudge = cos(theta * 3.14159265358979323846f / 180.0f);
        sy_nudge = sin(theta * 3.14159265358979323846f / 180.0f);
        cz_nudge = cos(0);
        sz_nudge = sin(0);
    } else if (axis == "z") {
        cx_nudge = cos(0);
        sx_nudge = sin(0);
        cy_nudge = cos(0);
        sy_nudge = sin(0);
        cz_nudge = cos(theta * 3.14159265358979323846f / 180.0f);
        sz_nudge = sin(theta * 3.14159265358979323846f / 180.0f);
    } else {
        cx_nudge = cos(0);
        sx_nudge = sin(0);
        cy_nudge = cos(0);
        sy_nudge = sin(0);
        cz_nudge = cos(0);
        sz_nudge = sin(0);
    }
    RotationMatrix mat_nudge = RotationMatrix(
        cz_nudge * cy_nudge - sz_nudge * sx_nudge * sy_nudge,
        -1.0 * sz_nudge * cx_nudge,
        cz_nudge * sy_nudge + sz_nudge * cy_nudge * sx_nudge,
        sz_nudge * cy_nudge + cz_nudge * sx_nudge * sy_nudge,
        cz_nudge * cx_nudge,
        sz_nudge * sy_nudge - cz_nudge * cy_nudge * sx_nudge,
        -1.0 * cx_nudge * sy_nudge, sx_nudge, cx_nudge * cy_nudge);

    return matmul(model_rotation_mat_, mat_nudge);
};

RotationMatrix matmul(RotationMatrix A, RotationMatrix B) {
    RotationMatrix result;
    result.rotation_00_ = A.rotation_00_ * B.rotation_00_ +
                          A.rotation_01_ * B.rotation_10_ +
                          A.rotation_02_ * B.rotation_20_;
    result.rotation_01_ = A.rotation_00_ * B.rotation_01_ +
                          A.rotation_01_ * B.rotation_11_ +
                          A.rotation_02_ * B.rotation_21_;
    result.rotation_02_ = A.rotation_00_ * B.rotation_02_ +
                          A.rotation_01_ * B.rotation_12_ +
                          A.rotation_02_ * B.rotation_22_;

    result.rotation_10_ = A.rotation_10_ * B.rotation_00_ +
                          A.rotation_11_ * B.rotation_10_ +
                          A.rotation_12_ * B.rotation_20_;
    result.rotation_11_ = A.rotation_10_ * B.rotation_01_ +
                          A.rotation_11_ * B.rotation_11_ +
                          A.rotation_12_ * B.rotation_21_;
    result.rotation_12_ = A.rotation_10_ * B.rotation_02_ +
                          A.rotation_11_ * B.rotation_12_ +
                          A.rotation_12_ * B.rotation_22_;

    result.rotation_20_ = A.rotation_20_ * B.rotation_00_ +
                          A.rotation_21_ * B.rotation_10_ +
                          A.rotation_22_ * B.rotation_20_;
    result.rotation_21_ = A.rotation_20_ * B.rotation_01_ +
                          A.rotation_21_ * B.rotation_11_ +
                          A.rotation_22_ * B.rotation_21_;
    result.rotation_22_ = A.rotation_20_ * B.rotation_02_ +
                          A.rotation_21_ * B.rotation_12_ +
                          A.rotation_22_ * B.rotation_22_;

    return result;
}

std::vector<float> vector_differece(std::vector<float> vec1,
                                    std::vector<float> vec2) {
    // Check to make sure that the vectors are the same size
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument(
            "Error: input vectors are not the same size");
    }
    // If here, vectors are same size
    std::vector<float> result(vec1.size());
    for (size_t i = 0; i < vec1.size(); i++) {
        result[i] = vec1[i] - vec2[i];
    }

    return result;
}

float vector_norm(std::vector<float> vec) {
    float sum;
    for (auto e : vec) {
        sum += e;
    }
    return sqrt(sum);
}
