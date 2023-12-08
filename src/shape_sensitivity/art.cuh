#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>

#include <iostream>
#include <vector>

#include "gpu/gpu_image.cuh"

class img_desc {
   public:
    img_desc(int height, int width, int gpu_device);
    ~img_desc();
    bool good_to_go();

    std::complex<float> art_n_p(int n, int p,
                                gpu_cost_function::GPUImage* dev_image);

    std::vector<float> hu_moments(gpu_cost_function::GPUImage* dev_image);
    int height();
    int width();

   private:
    float* dev_Fnp_re;
    float* dev_Fnp_imag;
    float* Fnp_re;
    float* Fnp_imag;
    float* dev_raw_img_moments_;
    float* raw_img_moments_;
    int width_;
    int height_;
    bool init_;
};
