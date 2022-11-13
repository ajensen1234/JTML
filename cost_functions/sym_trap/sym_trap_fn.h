#pragma once
#include <iostream>
#include "CostFunction.h"
#include <windows.h>
#include "Parameter.h"
#include "cuda_launch_parameters.h"
#include "gpu_imports.h"
#include "cuda_runtime_api.h"

using namespace jta_cost_function;
class sym_trap_fn final :public CostFunction
{
public:
	sym_trap_fn();
	double Run() override;
	bool Initialize(std::string& error_message) override;
	bool Destruct(std::string& error_message) override;
	int DIRECT_DILATION_current_dilation_parameter;
	int DIRECT_DILATION_current_white_pix_sum_dilated_comparison_image_A_;
	int DIRECT_DILATION_current_white_pix_sum_dilated_comparison_image_B_;

};



extern "C" __declspec(dllexport) CostFunction* create_fn()
{
	return new sym_trap_fn();
}


void invert_transform(float result[4][4], float tran[4][4])
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

void matmult(float ans[4][4], float matrix1[4][4], float matrix2[4][4])
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

void create_312_transform(float transform[4][4], float xt, float yt, float zt, float zr, float xr, float yr)
{
	float degtopi = 3.1415928 / 180.0;
	float zr_rad = zr * degtopi;
	float xr_rad = xr * degtopi;
	float yr_rad = yr * degtopi;

	float cx = cos(xr_rad);
	float cy = cos(yr_rad);
	float cz = cos(zr_rad);
	float sx = sin(xr_rad);
	float sy = sin(yr_rad);
	float sz = sin(zr_rad);

	transform[0][0] = cy * sx * sz - cz * sy;
	transform[0][1] = -cx * sz;
	transform[0][2] = cy * cz + sx * sy * sz;
	transform[0][3] = xt;

	transform[1][0] = -cy * cz * sx - sy * sz;
	transform[1][1] = cx * cz;
	transform[1][2] = cy * sz - cz * sx * sy;
	transform[1][3] = yt;

	transform[2][0] = cx * cy;
	transform[2][1] = sx;
	transform[2][2] = cx * sy;
	transform[2][3] = zt;

	transform[3][0] = transform[3][1] = transform[3][2] = 0.0f;
	transform[3][3] = 1.0f;
}
