#include "art.cuh"

__global__ void art_np_kernel(int height, int width, int n, int p,
                              unsigned char* image, double* dev_fnp_re,
                              double* dev_fnp_imag, int left_x, int bottom_y) {
    // thread values
    int thread_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int thread_y = (blockIdx.y * blockDim.y) + threadIdx.y;

    int x = thread_x + left_x;
    int y = thread_y + bottom_y;

    int orig_loc = x + width * y;

    if (x < width && y < height) {
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
            thrust::complex<double> fnp_complex = image[orig_loc] * A * R * rho;
            atomicAdd(&dev_fnp_re[0], fnp_complex.real());
            atomicAdd(&dev_fnp_imag[0], fnp_complex.imag());
        }
    }
}

__global__ void reset_vars(double* dev_fnp_re, double* dev_fnp_imag) {
    // Resetting the values of the kernels on the GPU
    dev_fnp_re[0] = 0;
    dev_fnp_imag[0] = 0;
}

__global__ void raw_image_moments_kernel(double* img_moments, int height,
                                         int width, unsigned char* image,
                                         int left_x, int bottom_y) {
    int thread_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int thread_y = (blockIdx.y * blockDim.y) + threadIdx.y;

    int x = thread_x + left_x;
    int y = thread_y + bottom_y;

    int orig_loc = x + width * y;

    if (x < width && y < height) {
        // M_00
        atomicAdd(&img_moments[0], image[orig_loc]);

        // M01
        atomicAdd(&img_moments[1], image[orig_loc] * y);

        // M10
        atomicAdd(&img_moments[2], image[orig_loc] * x);

        // M11
        atomicAdd(&img_moments[3], image[orig_loc] * x * y);

        // M02
        atomicAdd(&img_moments[4], image[orig_loc] * y * y);

        // M20
        atomicAdd(&img_moments[5], image[orig_loc] * x * x);

        // M12
        atomicAdd(&img_moments[6], image[orig_loc] * x * y * y);

        // M21
        atomicAdd(&img_moments[7], image[orig_loc] * x * x * y);

        // M22
        atomicAdd(&img_moments[8], image[orig_loc] * x * x * y * y);

        // M03
        atomicAdd(&img_moments[9], image[orig_loc] * y * y * y);

        // M30
        atomicAdd(&img_moments[10], image[orig_loc] * x * x * x);
    }
}

__global__ void clear_img_moments(double* hu) {
    hu[0] = 0;
    hu[1] = 0;
    hu[2] = 0;
    hu[3] = 0;
    hu[4] = 0;
    hu[5] = 0;
    hu[6] = 0;
    hu[7] = 0;
    hu[8] = 0;
    hu[9] = 0;
    hu[10] = 0;
}

img_desc::img_desc(int height, int width, int gpu_device) {
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

    cudaHostAlloc((void**)&raw_img_moments_, 11 * sizeof(double),
                  cudaHostAllocDefault);
    if (cudaGetLastError() != cudaSuccess) {
        init_ = false;
    }
    cudaMalloc((void**)&dev_raw_img_moments_, 11 * sizeof(double));
    if (cudaGetLastError() != cudaSuccess) {
        init_ = false;
    }
};

img_desc::~img_desc() {
    cudaFree(dev_Fnp_imag);
    cudaFree(dev_Fnp_re);
    cudaFreeHost(Fnp_re);
    cudaFreeHost(Fnp_imag);
    cudaFreeHost(raw_img_moments_);
    cudaFree(dev_raw_img_moments_);
};
// Our "good to go" function that tells us if everything went according to plan
bool img_desc::good_to_go() { return init_; }

// Our function that actually calls the kernel
// TODO: We will need to update this to accept the dev_image from some of our
// projections, eventually
// That will also include some bounding box stuff (which should actually speed
// things up considerably)
std::complex<double> img_desc::art_n_p(int n, int p,
                                       gpu_cost_function::GPUImage* dev_image) {
    // Standard defintion for creating our work groups
    const int threads_per_block = 256;
    int* bounding_box = dev_image->GetBoundingBox();
    int left_x = max(bounding_box[0], 0);
    int bottom_y = max(bounding_box[1], 0);
    int right_x = min(bounding_box[2], width_ - 1);
    int top_y = min(bounding_box[3], height_ - 1);
    int diff_cropped_width = right_x - left_x - 1;
    int diff_cropped_height = top_y - bottom_y + 1;

    dim3 dim_grid_bounding_box =
        dim3(ceil(static_cast<double>(diff_cropped_width) /
                  sqrt(static_cast<double>(threads_per_block))),
             ceil(static_cast<double>(diff_cropped_height) /
                  sqrt(static_cast<double>(threads_per_block))));

    dim3 dim_block = dim3(ceil(sqrt(static_cast<double>(threads_per_block))),
                          ceil(sqrt(static_cast<double>(threads_per_block))));

    // Reset the variables that we are storing
    reset_vars<<<1, 1>>>(dev_Fnp_re, dev_Fnp_imag);
    // Run the kernel
    art_np_kernel<<<dim_grid_bounding_box, dim_block>>>(
        height_, width_, n, p, dev_image->GetDeviceImagePointer(), dev_Fnp_re,
        dev_Fnp_imag, left_x, bottom_y);

    // Copying everything back to host (CPU)
    cudaMemcpy(Fnp_re, dev_Fnp_re, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Fnp_imag, dev_Fnp_imag, sizeof(double), cudaMemcpyDeviceToHost);

    // Returning the value of the complex function that we have calculated.
    std::complex<double> fnp(Fnp_re[0], Fnp_imag[0]);
    return fnp;
};
int img_desc::height() { return height_; };
int img_desc::width() { return width_; };

std::vector<double> img_desc::hu_moments(
    gpu_cost_function::GPUImage* dev_image) {
    const int threads_per_block = 256;
    int* bounding_box = dev_image->GetBoundingBox();
    int left_x = max(bounding_box[0], 0);
    int bottom_y = max(bounding_box[1], 0);
    int right_x = min(bounding_box[2], width_ - 1);
    int top_y = min(bounding_box[3], height_ - 1);
    int diff_cropped_width = right_x - left_x - 1;
    int diff_cropped_height = top_y - bottom_y + 1;

    dim3 dim_grid_bounding_box =
        dim3(ceil(static_cast<double>(diff_cropped_width) /
                  sqrt(static_cast<double>(threads_per_block))),
             ceil(static_cast<double>(diff_cropped_height) /
                  sqrt(static_cast<double>(threads_per_block))));

    dim3 dim_block = dim3(ceil(sqrt(static_cast<double>(threads_per_block))),
                          ceil(sqrt(static_cast<double>(threads_per_block))));

    clear_img_moments<<<1, 1>>>(dev_raw_img_moments_);
    raw_image_moments_kernel<<<dim_grid_bounding_box, dim_block>>>(
        dev_raw_img_moments_, height_, width_,
        dev_image->GetDeviceImagePointer(), left_x, bottom_y);

    // copy raw image moments back to host
    cudaMemcpy(raw_img_moments_, dev_raw_img_moments_, 11 * sizeof(double),
               cudaMemcpyDeviceToHost);

    // Here, we use those moments to calculate the values of the kernel;
    // Storing values as variables to make life a LOT easier

    double m00 = raw_img_moments_[0];
    double m01 = raw_img_moments_[1];
    double m10 = raw_img_moments_[2];
    double m11 = raw_img_moments_[3];
    double m02 = raw_img_moments_[4];
    double m20 = raw_img_moments_[5];
    double m12 = raw_img_moments_[6];
    double m21 = raw_img_moments_[7];
    double m22 = raw_img_moments_[8];
    double m03 = raw_img_moments_[9];
    double m30 = raw_img_moments_[10];

    double xbar = m10 / m00;
    double ybar = m01 / m00;
    double mu[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    double eta[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    mu[0][0] = m00;
    mu[0][1] = 0;
    mu[1][0] = 0;
    mu[1][1] = m11 - xbar * m01;
    mu[2][0] = m20 - xbar * m10;
    mu[0][2] = m02 - ybar * m01;
    mu[2][1] = m21 - 2 * xbar * m11 - ybar * m20 + 2 * xbar * xbar * m01;
    mu[1][2] = m12 - 2 * ybar * m11 - xbar * m02 + 2 * ybar * ybar * m10;
    mu[3][0] = m30 - 3 * xbar * m20 + 2 * xbar * xbar * m10;
    mu[0][3] = m03 - 3 * ybar * m02 + 2 * ybar * ybar * m01;

    // Now onto the hu moments
    // (aka moment invariants)
    // First we calculate the eta invariants
    for (int i = 0; i <= 3; i++) {
        for (int j = 0; j <= 3; j++) {
            eta[i][j] = mu[i][j] / (pow(mu[0][0], (1 + ((i + j) / 2))));
        }
    }

    // Hu moments dun dun dunnnnn
    // (vector index is off by 1 because of zero-based indexing)
    std::vector<double> hu(8);
    hu[0] = eta[2][0] + eta[0][2];

    hu[1] = pow((eta[2][0] - eta[0][2]), 2) + 4 * pow(eta[1][1], 2);

    hu[2] =
        pow(eta[3][0] - 3 * eta[1][2], 2) + pow(3 * eta[2][1] - eta[0][3], 2);

    hu[3] = pow(eta[3][0] + eta[1][2], 2) + pow(eta[2][1] + eta[0][3], 2);

    hu[4] =
        (eta[3][0] - 3 * eta[1][2]) * (eta[3][0] + eta[1][2]) *
            (pow(eta[3][0] + eta[1][2], 2) -
             3 * pow(eta[2][1] + eta[0][3], 2)) +
        (3 * eta[2][1] - eta[0][3]) * (eta[2][1] + eta[0][3]) *
            (3 * pow(eta[3][0] + eta[1][2], 2) - pow(eta[2][1] + eta[0][3], 2));

    hu[5] =
        (eta[1][2] - eta[0][3]) *
            (pow(eta[3][0] + eta[1][2], 2) - pow(eta[2][1] + eta[0][3], 2)) +
        (eta[2][1] + eta[0][3]) *
            (3 * pow(eta[3][0] + eta[1][2], 2) - pow(eta[2][1] + eta[0][3], 2));

    hu[6] = (eta[2][0] - eta[0][2]) * (pow(eta[3][0] + eta[1][2], 2) -
                                       pow(eta[2][1] + eta[0][3], 2)) +
            4 * eta[1][1] * (eta[3][0] + eta[1][2]) * (eta[2][1] + eta[0][3]);

    hu[7] =
        (3 * eta[2][1] - eta[0][3]) * (eta[3][0] + eta[1][2]) *
            (pow(eta[3][0] + eta[1][2], 2) -
             3 * pow(eta[2][1] + eta[0][3], 2)) -
        (eta[3][0] - 3 * eta[1][2]) * (eta[2][1] + eta[0][3]) *
            (3 * pow(eta[3][0] + eta[1][2], 2) - pow(eta[2][1] + eta[0][3], 2));
    return hu;
}
