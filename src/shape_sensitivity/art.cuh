#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>

#include <iostream>

class art {
   public:
    art(int height, int width, int gpu_device, unsigned char* host_image);
    ~art();
    bool good_to_go();

    std::complex<double> art_n_p(int n, int p);

   private:
    double* dev_Fnp_re;
    double* dev_Fnp_imag;
    double* Fnp_re;
    double* Fnp_imag;
    unsigned char* dev_image;
    int width_;
    int height_;
    bool init_;
};
