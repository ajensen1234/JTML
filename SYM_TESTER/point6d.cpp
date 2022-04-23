#include "point6d.h"


Point6D::Point6D(double xval, double yval, double zval, double xaval, double yaval, double zaval) {
	x = xval; y = yval; z = zval; xa = xaval; ya = yaval; za = zaval;
}

Point6D::Point6D() {
	x = 0; y = 0; z = 0; xa = 0; ya = 0; za = 0;
}