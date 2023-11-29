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
            // Define some vectors that will be used to construct rho in polar
            // coords
            double x_vec = x - width / 2;
            double y_vec = y - height / 2;
            // We normalize rho to have a diameter equal to the width of the
            // image
            //  This prevents the corners from having some of the "basis"
            //  functions, but this is the way that the paper presents it
            double rho = sqrt(x_vec * x_vec + y_vec * y_vec) / (width / 2);

            // Theta in polar coords based on where we are in the image
            double theta = atan2(y_vec, x_vec);
            // We Are only looking at a normalized rho of 1
            // This is part of the integration
            if (rho <= 1) {
                // This is the R-cos functon that is used to derive some of the
                // angular invariance (Eq 7)
                double R = (n == 0) ? 1.0 : 2.0 * cos(3.1415928 * n * rho);
                // This is defining A, which gives rotation invariance (Eq 6)
                thrust::complex<double> A =
                    (1 / (2 * 3.1415928)) *
                    exp(thrust::complex<double>(0.0, p * theta));
                // This is defining the integration over the whole image, and
                // constructiong the full value of F_np (Eq 4)
                thrust::complex<double> fnp_complex =
                    image[orig_loc] * A * R * rho;
                atomicAdd(&dev_fnp_re[0], fnp_complex.real());
                atomicAdd(&dev_fnp_imag[0], fnp_complex.imag());
            }
        }
    }
}

__global__ void reset_vars(double* dev_fnp_re, double* dev_fnp_imag) {
    // Resetting the values of the kernels on the GPU
    dev_fnp_re[0] = 0;
    dev_fnp_imag[0] = 0;
}
art::art(int height, int width, int gpu_device, unsigned char* host_image) {
    // We start with the assumption that things go according the plan
    init_ = true;
    height_ = height;
    width_ = width;

    // Allocating all of the memory that we have onto the GPU and CPU
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
};
// Our "good to go" function that tells us if everything went according to plan
bool art::good_to_go() { return init_; }

// Our function that actually calls the kernel
// TODO: We will need to update this to accept the dev_image from some of our
// projections, eventually
// That will also include some bounding box stuff (which should actually speed
// things up considerably)
std::complex<double> art::art_n_p(int n, int p) {
    // Standard defintion for creating our work groups
    auto dim_grid = dim3(ceil(static_cast<double>(width_) / sqrt(256)),
                         ceil(static_cast<double>(height_) / sqrt(256)));
    auto block_grid = dim3(16, 16);

    // Reset the variables that we are storing
    reset_vars<<<1, 1>>>(dev_Fnp_re, dev_Fnp_imag);
    // Run the kernel
    art_np_kernel<<<dim_grid, block_grid>>>(height_, width_, n, p, dev_image,
                                            dev_Fnp_re, dev_Fnp_imag);

    // Copying everything back to host (CPU)
    cudaMemcpy(Fnp_re, dev_Fnp_re, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Fnp_imag, dev_Fnp_imag, sizeof(double), cudaMemcpyDeviceToHost);

    // Returning the value of the complex function that we have calculated.
    std::complex<double> fnp(Fnp_re[0], Fnp_imag[0]);
    return fnp;
};
