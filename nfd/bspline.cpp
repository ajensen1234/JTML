#include "bspline.h"


void bspline(std::vector<vec2d> &cntr_pts, int num, int degree, bool periodic) {
	int count = cntr_pts.size();

	if (periodic) {
		int factor = (count + degree + 1) % count; // should always come out to (degree + 1)
		int fraction = (count + degree + 1) / count; // should always come out to 1
		for (int i = 0; i < fraction; i++) {
			cntr_pts.push_back(cntr_pts[i]); // this should append enough values to the end of the contour in order to create the necessary values
		}
	}

	// Calculate the knot vector
	std::vector<float> kv;

	
}