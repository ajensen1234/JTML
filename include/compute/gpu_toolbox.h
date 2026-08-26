/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#ifndef GPU_TOOLBOX_H
#define GPU_TOOLBOX_H

/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {
/*This class is a toolbox for users looking to write their own
cost functions, and uses GPU computing. Users will be provided
with several resources (stored on the GPU) and functions
(computed on the GPU):
        Resources (Stored on GPU Memory During Initialization of
        CostFunctionToolboxGPU Class):
                - Every image uploaded to JTA
                - Edge detected version of every image uploaded to JTA
                - Dilated version of every image uploaded to JTA
                (dilation value is same as "Dilation" int parameter
                in Cost Function chosen for stage. If such a parameter
                does not exist dilation is 0 and this image is a copy
                of the edge detected version).
                - GPU model classes (one for each model). The GPU model
                for the primary model will be stored seperately
                from the list of GPU models for the non-primary
                models.
                        - GPU model class also includes information
                        about the model such as the name, file location,
                        model type, and STL style triangle information.
                        This last piece of information is stored on the GPU.
                - Render Engine Class. As of 4/11/2018 this will have
                to be modified to accept device pointers for the triangle
                normals and vertices.


        */
class CostFunctionToolboxGPU {}
}  // namespace gpu_cost_function

#endif /* GPU_TOOLBOX_H */
