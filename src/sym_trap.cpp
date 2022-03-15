#include "sym_trap.h"

sym_trap::sym_trap(QWidget* parent, Qt::WindowFlags flags) : QDialog(parent, flags)

{
	ui.setupUi(this);

	QObject::connect(this, SIGNAL(Done()), this, SLOT(close()));

	QFontMetrics font_metrics(this->font());

	this->setStyleSheet(this->styleSheet() += "QGroupBox { margin-top: " + QString::number(font_metrics.height() / 2) + "px; }");
	int group_box_to_top_button_y = font_metrics.height() / 2;
}

sym_trap::~sym_trap()
{

}

Point6D sym_trap::compute_mirror_pose(Point6D pose) {
	// The main function of this equation is to find the mirror pose of a specific projection geometry
	// This is important when it comes to determining the different symmetry traps that might be present

	// To start, we create the transformation matrix based on the current 6D pose (3-1-2 rotation order)
	float rad2deg = 180.0 / 3.1415928;
	//float transform[4][4]; // blank matrix that will get populated
	float Rot[3][3]; // blank matrix that contains the rotation matrix of the above transformation matrix

	// create_312_transform(transform, pose);
	rotation_matrix(Rot, pose);

	// determine the location of the viewing vector, this is also the location of the center of mass

	float viewing[3];
	viewing[0] = pose.x; viewing[1] = pose.y; viewing[2] = pose.z;


	// pull out the z-axis of the pose, determined from the transformation matrix
	float z_ax[3];
	z_ax[0] = Rot[0][2]; z_ax[1] = Rot[1][2]; z_ax[2] = Rot[2][2];
	std::cout << "[" << z_ax[0] << " " << z_ax[1] << " " << z_ax[2] << "]" << std::endl;
	// normalize viewing vector, take negative to point from object -> camera
	float view_normed[3];
	float view_mag = sqrt(pow(viewing[0], 2) + pow(viewing[1], 2) + pow(viewing[2], 2));
	std::cout << "View normed : \n [";
	for (int i = 0; i < 3; ++i) {
		view_normed[i] = -viewing[i] / view_mag;
		std::cout << view_normed[i] << " ";
	}
	std::cout << "]" << std::endl;
	// Next, take the cross product of the two vectors (z-ax and normed) to get the axis of rotation.
	// call this axis M (Crane and Duffy reference)
	float M_temp[3];
	sym_trap::cross_product(M_temp, z_ax, view_normed);

	//need to normalize M
	float M_mag = sqrt(pow(M_temp[0], 2) + pow(M_temp[1], 2) + pow(M_temp[2], 2));
	float M[3];
	std::cout << "View M : \n [";
	for (int i = 0; i < 3; i++) {
		M[i] = M_temp[i] / M_mag;
		std::cout << M[i] << " ";
	}
	std::cout << "]" << std::endl;

	// Take the dot product between the two to determine the angle between them
	// we will keep this in radians for now 

	float temp_dot = 0;
	sym_trap::dot_product(temp_dot, z_ax, view_normed);

	std::cout << "temp_dot" << temp_dot << std::endl;
	float angle_between = acos(temp_dot);
	std::cout << "Angle Between: " << angle_between << std::endl;

	float desired_rotation = 2 * angle_between;

	std::cout << desired_rotation << std::endl;

	// Now we need to compute the equivalent axis-angle rotation matrix

	float pose2dual[3][3];
	equivalent_axis_angle_rotation(pose2dual, M, desired_rotation);

	// declare some varialbes that will be the final pose



	// calculate the rotation matrix of the dual pose in space using properties of transformation matrices
	float Rot_dual[3][3];
	matmult3(Rot_dual, pose2dual, Rot);
	std::cout << "X-Axis: \n";
	for (int i = 0; i < 3; i++) {
		std::cout << Rot[i][0] << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Y-Axis: \n";
	for (int i = 0; i < 3; i++) {
		std::cout << Rot[i][1] << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Z-Axis: \n";
	for (int i = 0; i < 3; i++) {
		std::cout << Rot[i][2] << std::endl;
	}
	std::cout << std::endl;

	// extract out each of the relevent values from the final rotation matrix
	float xr_dual;
	float zr_dual;
	float yr_dual;
	getRotations312(xr_dual, yr_dual, zr_dual, Rot_dual);

	// convert to degrees

	float xr_dual_deg = xr_dual * rad2deg;
	float yr_dual_deg = yr_dual * rad2deg;
	float zr_dual_deg = zr_dual * rad2deg;

	Point6D final_pose(pose.x, pose.y, pose.z, xr_dual_deg, yr_dual_deg, zr_dual_deg);
	return final_pose;

}

void sym_trap::create_312_transform(float transform[4][4],
	Point6D pose)
{
	float degtorad = 3.1415928 / 180.0;
	float zr_rad = pose.za * degtorad;
	float xr_rad = pose.xa * degtorad;
	float yr_rad = pose.ya * degtorad;

	float cx = cos(xr_rad);
	float cy = cos(yr_rad);
	float cz = cos(zr_rad);
	float sx = sin(xr_rad);
	float sy = sin(yr_rad);
	float sz = sin(zr_rad);

	transform[0][0] = cy * cz - sx * sy * sz;
	transform[0][1] = -cx * sz;
	transform[0][2] = cy * sx * sz + cz * sy;
	transform[0][3] = pose.x;

	transform[1][0] = cy * sz + cz * sx * sy;
	transform[1][1] = cx * cz;
	transform[1][2] = -cy * cz * sx + sy * sz;
	transform[1][3] = pose.y;

	transform[2][0] = -cx * sy;
	transform[2][1] = sx;
	transform[2][2] = cx * cy;
	transform[2][3] = pose.z;

	transform[3][0] = transform[3][1] = transform[3][2] = 0.0f;
	transform[3][3] = 1.0f;
}

void sym_trap::invert_transform(float result[4][4], const float tran[4][4])
{
	int     i, j;
	/* Upper left 3x3 of result is transpose of upper left 3x3 of tran. */
	for (i = 0; i < 3; ++i)
		for (j = 0; j < 3; ++j)
			result[i][j] = tran[j][i];
	/* Set the values for the last column of the result */
	result[3][0] = result[3][1] = result[3][2] = 0.0;
	result[3][3] = 1.0;
	/* Initialize the values of the last column of the result. */
	result[0][3] = result[1][3] = result[2][3] = 0.0;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			result[i][3] -= result[i][j] * tran[j][3];
		}
	}
}

void sym_trap::matmult4(float ans[4][4], float matrix1[4][4], float matrix2[4][4])
{
	int   i, j, k;
	for (i = 0; i < 4; i++)
		for (j = 0; j < 4; j++)
			ans[i][j] = 0.0;
	for (i = 0; i < 4; i++)
		for (j = 0; j < 4; j++)
			for (k = 0; k < 4; k++)
				ans[i][j] += matrix1[i][k] * matrix2[k][j];
}

void sym_trap::matmult3(float ans[3][3], const float matrix1[3][3], const float matrix2[3][3]) {
	int   i, j, k;
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
			ans[i][j] = 0.0;
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
			for (k = 0; k < 3; k++)
				ans[i][j] += matrix1[i][k] * matrix2[k][j];
}

void sym_trap::dot_product(float& result, const float vector1[3], const float vector2[3]) {

	for (int i = 0; i < 3; i++) {
		result += (vector1[i] * vector2[i]);
	}
}

void sym_trap::cross_product(float CP[3], const float v1[3], const float v2[3]) {
	CP[0] = v1[1] * v2[2] - v1[2] * v2[1];
	CP[1] = -(v1[0] * v2[2] - v1[2] * v2[0]);
	CP[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

void sym_trap::equivalent_axis_angle_rotation(float rot[3][3], const float m[3], const float angle) {
	float c = cos(angle);
	float s = sin(angle);
	float v = 1.0 - c;

	rot[0][0] = m[0] * m[0] * v + c;
	rot[0][1] = m[0] * m[1] * v - m[2] * s;
	rot[0][2] = m[0] * m[2] * v + m[1] * s;

	rot[1][0] = m[0] * m[1] * v + m[2] * s;
	rot[1][1] = m[1] * m[1] * v + c;
	rot[1][2] = m[1] * m[2] * v - m[0] * s;

	rot[2][0] = m[0] * m[2] * v - m[1] * s;
	rot[2][1] = m[1] * m[2] * v + m[0] * s;
	rot[2][2] = m[2] * m[2] * v + c;

}



void sym_trap::rotation_matrix(float R[3][3], Point6D pose) {

	float degtorad = 3.1415928 / 180.0;
	float zr_rad = pose.za * degtorad;
	float xr_rad = pose.xa * degtorad;
	float yr_rad = pose.ya * degtorad;

	float cx = cos(xr_rad);
	float cy = cos(yr_rad);
	float cz = cos(zr_rad);
	float sx = sin(xr_rad);
	float sy = sin(yr_rad);
	float sz = sin(zr_rad);

	R[0][0] = cy * cz - sx * sy * sz;
	R[0][1] = -cx * sz;
	R[0][2] = cy * sx * sz + cz * sy;

	R[1][0] = cy * sz + cz * sx * sy;
	R[1][1] = cx * cz;
	R[1][2] = -cy * cz * sx + sy * sz;

	R[2][0] = -cx * sy;
	R[2][1] = sx;
	R[2][2] = cx * cy;

}


void sym_trap::getRotations312(float& xr, float& yr, float& zr, const float Rot[3][3]) {

	float sx = Rot[2][1];
	xr = asin(sx);
	std::cout << xr << std::endl;;

	if (sx == 0 || sx == 1) {
		zr == 0;
		yr = atan2(Rot[0][2], Rot[0][0]);
	}
	else {
		zr = atan2(-Rot[0][1], Rot[1][1]);
		yr = atan2(-Rot[2][0], Rot[2][2]);
	}

}