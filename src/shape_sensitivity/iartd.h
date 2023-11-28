#ifndef IARTD_H
#define IARTD_H
#include <complex.h>

#include <vector>
class IARTD {
   public:
    IARTD(int MAX_N, int MAX_P);
    std::vector<double> feature_vector(int n, int p);
    double phase_at(int n, int p);
    double mag_at(int n, int p);
    std::complex<double> F_at(int n, int p);
    void phase_correction();
    void F_correction();
};

#endif /* IARTD_H */
