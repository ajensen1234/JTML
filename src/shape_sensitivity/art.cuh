#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>

#include <iostream>
#include <vector>

class img_desc {
   public:
    img_desc(int height, int width, int gpu_device);
    ~img_desc();
    bool good_to_go();

    std::complex<double> art_n_p(int n, int p, unsigned char* dev_image);

    std::vector<double> hu_moments(unsigned char* dev_image);
    int height();
    int width();

   private:
    double* dev_Fnp_re;
    double* dev_Fnp_imag;
    double* Fnp_re;
    double* Fnp_imag;
    double* dev_raw_img_moments_;
    double* raw_img_moments_;
    int width_;
    int height_;
    bool init_;
};
