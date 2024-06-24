#pragma once

#include <string>
#include <vector>

#include "gpu/render_engine.cuh"

using namespace gpu_cost_function;

RotationMatrix rotation_nudge(Pose input_pose, float theta, std::string axis);
RotationMatrix matmul(RotationMatrix A, RotationMatrix B);
std::vector<float> vector_differece(std::vector<float> vec1,
                                    std::vector<float> vec2);
float vector_norm(std::vector<float> vec);
