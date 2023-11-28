#include "art.cuh"

__global__ void art_np_kernel(int height, int width, int n, int p,
                              unsigned char* image, double* dev_fnp_re,
                              double* dev_fnp_imag) {
    // thread values
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int orig_loc = x + width * y;

    if (x < width - 1) {
        if (y < height - 1) {
            double x_vec = x - width / 2;
            double y_vec = y - height / 2;
            double rho = sqrt(x_vec * x_vec + y_vec * y_vec) / (width / 2);
            double theta = atan2(y_vec, x_vec);
            if (rho <= 1) {
                double R = (n == 0) ? 1.0 : 2.0 * cos(3.1415928 * n * rho);
                thrust::complex<double> A =
                    (1 / (2 * 3.1415928)) *
                    exp(thrust::complex<double>(0.0, p * theta));
                thrust::complex<double> fnp_complex =
                    image[orig_loc] * A * R * rho;
                atomicAdd(&dev_fnp_re[0], fnp_complex.real());
                atomicAdd(&dev_fnp_imag[0], fnp_complex.imag());
            }
        }
    }
}

__global__ void reset_vars(double* dev_fnp_re, double* dev_fnp_imag) {
    dev_fnp_re[0] = 0;
    dev_fnp_imag[0] = 0;
}
art::art(int height, int width, int gpu_device, unsigned char* host_image) {
    init_ = true;
    height_ = height;
    width_ = width;
    cudaHostAlloc((void**)&Fnp_re, sizeof(double), cudaHostAllocDefault);
    if (cudaGetLastError() != cudaSuccess) {
        init_ = false;
    }
    cudaHostAlloc((void**)&Fnp_imag, sizeof(double), cudaHostAllocDefault);
    if (cudaGetLastError() != cudaSuccess) {
        init_ = false;
    }
    cudaMalloc((void**)&dev_Fnp_imag, sizeof(double));
    if (cudaGetLastError() != cudaSuccess) {
        init_ = false;
    }
    cudaMalloc((void**)&dev_Fnp_re, sizeof(double));
    if (cudaGetLastError() != cudaSuccess) {
        init_ = false;
    }
    cudaMalloc((void**)&dev_image, width * height * sizeof(unsigned char));
    if (cudaGetLastError() != cudaSuccess) {
        init_ = false;
    }
    cudaMemcpy(dev_image, host_image, width * height * sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess) {
        init_ = false;
    }
};

art::~art() {
    cudaFree(dev_image);
    cudaFree(dev_Fnp_imag);
    cudaFree(dev_Fnp_re);
    cudaFreeHost(Fnp_re);
    cudaFreeHost(Fnp_imag);
    std::cout << "ART GPU DELETED" << std::endl;
};
bool art::good_to_go() { return init_; }
std::complex<double> art::art_n_p(int n, int p) {
    auto dim_grid = dim3(ceil(static_cast<double>(width_) / sqrt(256)),
                         ceil(static_cast<double>(height_) / sqrt(256)));
    auto block_grid = dim3(16, 16);

    reset_vars<<<1, 1>>>(dev_Fnp_re, dev_Fnp_imag);
    art_np_kernel<<<dim_grid, block_grid>>>(height_, width_, n, p, dev_image,
                                            dev_Fnp_re, dev_Fnp_imag);

    cudaMemcpy(Fnp_re, dev_Fnp_re, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Fnp_imag, dev_Fnp_imag, sizeof(double), cudaMemcpyDeviceToHost);
    std::complex<double> fnp(Fnp_re[0], Fnp_imag[0]);
    return fnp;
};
