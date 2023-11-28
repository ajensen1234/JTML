#include "descriptors.h"

std::vector<double> calculateIARTD(cv::Mat binary_image) {
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
    std::vector<double> iartd(2 * (MAX_N + 1) * (MAX_P + 1));
    std::vector<double> iartd_gpu(2 * (MAX_N + 1) * (MAX_P));
    int H = binary_image.rows;
    int W = binary_image.cols;
    auto art_gpu = new art(H, W, 0, binary_image.data);
    if (art_gpu->good_to_go()) {
        std::cout
            << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            << std::endl;
    } else {
        std::cout
            << "---------------------------------------------------------------"
            << std::endl;
        delete (art_gpu);
    }

    auto idx = [](int n, int p) -> int {
        return (n * MAX_P + p - 1) * 2;
    };  // Lambda for easy indexing
    for (int n = 0; n <= MAX_N; n++) {
        std::cout << "n = " << n << std::endl;
        for (int p = 0; p <= MAX_P; p++) {
            std::complex<double> Fnp(0.0, 0.0);  // F_np declaration (Eq 4)
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    // Rho is the magnitude of the polar coordinates over which
                    // we are integrating (Eq 4).
                    // We normalize rho to 1 for the outer ring of the image
                    // (some of the corners will be lopped off with this cutoff)
                    double rho =
                        sqrt(pow((x - W / 2), 2) + pow((y - H / 2), 2));
                    rho /= ((double)(W) / 2);
                    if (rho > 1) continue;
                    double theta = atan2((double)y - (double)H / 2,
                                         (double)x - (double)W / 2);
                    // A helps achieve invariance wrt magnitude (Eq 7)
                    // We use the cosine version
                    std::complex<double> A =
                        (1 / (2 * CV_PI)) *
                        exp(std::complex<double>(0.0, p * theta));
                    // V allows it to be separable along radial direction
                    // Eq 5
                    double R = (n == 0) ? 1.0 : 2.0 * cos(CV_PI * n * rho);
                    std::complex<double> V = A * R;
                    // We do an iterative summation for the input binary image
                    // To do so discretely, we are simply adding at each
                    // iteration of n/p Eq 4
                    Fnp += (double)binary_image.at<uchar>(y, x) * V * rho;
                }  // end for (x)
            }  // end for (y)
            // At this point, we have done the summation for the entire image,
            // so we normalize by the total number of pixels in the input image
            Fnp /= (double)(H * W);
            std::cout << "CPU: " << Fnp << std::endl;
            std::cout << "GPU: " << art_gpu->art_n_p(n, p) / (double)(H * W)
                      << "\n---------------------" << std::endl;
            // Because we don't gain any information for p<2, we filter out
            // the values that we save
            if (p > 1) {
                // IARTD.at<double>(idx(n, p)) = abs(Fnp);
                // IARTD.at<double>(idx(n, p) + 1) = arg(Fnp);
                iartd_gpu[idx(n, p)] =
                    abs(art_gpu->art_n_p(n, p)) / (double)(H * W);
                iartd_gpu[idx(n, p) + 1] =
                    arg(art_gpu->art_n_p(n, p)) / (double)(H * W);

                iartd[idx(n, p)] = abs(Fnp);
                iartd[idx(n, p) + 1] = arg(Fnp);
            } else if (p == 1) {
                // But, we want to keep values at p=1 for the normalization
                // procedure
                phase_n_1[n] = arg(Fnp);
            }
        }
        std::cout << "End for N" << std::endl;
    }
    // Phase Correction using values from p = 1 (Eq 15, 16)
    for (int n = 0; n <= MAX_N; n++) {
        std::cout << "Phase correction N = " << n << std::endl;
        for (int p = 2; p <= MAX_P; p++) {
            std::complex<double> fnp_prime =
                std::complex<double>(iartd[idx(n, p)], iartd[idx(n, p) + 1]) *
                exp(std::complex<double>(0.0, -p * phase_n_1[n]));
            iartd[idx(n, p)] = abs(fnp_prime);
            iartd[idx(n, p) + 1] -= phase_n_1[n];
        }
    }
    std::cout << "End IARTD calc" << std::endl;
    delete (art_gpu);
    return iartd;
};
