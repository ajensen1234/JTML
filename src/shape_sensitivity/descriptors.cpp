#include "descriptors.h"

std::vector<double> calculateIARTD(img_desc* img_desc_gpu) {
    /**
     * This is a function to calculate the Invariant Angular Radtial Transform
     Descriptor.
     * See: J.-M. Lee and W.-Y. Kim, “A New Shape Description Method Using
     Angular Radial Transform,” IEICE Trans. Inf. & Syst., vol. E95.D, no. 6,
     pp. 1628–1635, 2012, doi: 10.1587/transinf.E95.D.1628.
     * The input is a binary image (either a segmentation or a projected image),
     and the output is the vector containing the descriptor variables.
     */
    const int MAX_P =
        8;  // Setting max values for the number of "rings" and "angles"
    const int MAX_N = 3;
    double phase_n_1[MAX_N + 1];  // Creating array for phase correction term
    // (Eqs 15, 16)
    int H = img_desc_gpu->height();
    int W = img_desc_gpu->width();
    std::vector<double> iartd(2 * (MAX_N + 1) * (MAX_P + 1));
    std::vector<double> iartd_gpu(2 * (MAX_N + 1) * (MAX_P));
    // int H = binary_image.rows;
    // int W = binary_image.cols;
    // auto img_desc_gpu = new img_desc(H, W, 0, binary_image.data);

    auto idx = [](int n, int p) -> int {
        return (n * MAX_P + p - 1) * 2;
    };  // Lambda for easy indexing
    for (int n = 0; n <= MAX_N; n++) {
        for (int p = 0; p <= MAX_P; p++) {
            std::complex<double> fnp = img_desc_gpu->art_n_p(n, p);
            if (p > 1) {
                iartd[idx(n, p)] = abs(fnp) / (double)(H * W);
                iartd[idx(n, p) + 1] = arg(fnp) / (double)(H * W);

            } else if (p == 1) {
                // But, we want to keep values at p=1 for the normalization
                // procedure
                phase_n_1[n] = arg(fnp);
            }
        }
    }
    // Phase Correction using values from p = 1 (Eq 15, 16)
    for (int n = 0; n <= MAX_N; n++) {
        for (int p = 2; p <= MAX_P; p++) {
            std::complex<double> fnp_prime =
                std::complex<double>(iartd[idx(n, p)], iartd[idx(n, p) + 1]) *
                exp(std::complex<double>(0.0, -p * phase_n_1[n]));
            iartd[idx(n, p)] = abs(fnp_prime);
            iartd[idx(n, p) + 1] -= phase_n_1[n];
        }
    }
    // delete (img_desc_gpu);
    return iartd;
};
