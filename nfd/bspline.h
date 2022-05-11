#pragma once
#include "nfd_instance.h"
#include "fitpackpp/BSplineCurve.h"

void bspline(std::vector<vec2d>& cntr_pts, int num, int degree, bool periodic);