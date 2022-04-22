#pragma once
#include "point6d.h"

//#include "../src/data_structures_6D.h"
class sym_trap 
{

public:
	sym_trap();
	~sym_trap();

	Point6D compute_mirror_pose(Point6D point);
	void matmult4(float ans[4][4], float matrix1[4][4], float matrix2[4][4]);
	void matmult3(float ans[3][3], const float matrix1[3][3], const float matrix2[3][3]);
	void invert_transform(float result[4][4], const float tran[4][4]);
	void equivalent_axis_angle_rotation(float rot[3][3], const float m[3], const float angle);
	void cross_product(float CP[3], const float v1[3], const float v2[3]);
	void dot_product(float &result, const float vector1[3], const float vector2[3]);
	void rotation_matrix(float R[3][3], Point6D pose);
	void create_312_transform(float transform[4][4], Point6D pose);
	void getRotations312(float &xr, float &yr, float &zr, const float Rot[3][3]);


	void change_value(float &num);


private:

	

};