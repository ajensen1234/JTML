#pragma once
/****************Headers*************/
/*Cost Function Tools Library*/
#include "gpu/gpu_dilated_frame.cuh"
#include "gpu/gpu_edge_frame.cuh"
#include "gpu/gpu_frame.cuh"
#include "gpu/gpu_image.cuh"
#include "gpu/gpu_intensity_frame.cuh"
#include "gpu/gpu_metrics.cuh"
#include "gpu/gpu_model.cuh"
#include "gpu/render_engine.cuh"
/*Stage Enum*/
#include "Stage.h"
/*Parameter Class*/
#include "Parameter.h"
#include "core/preprocessor-defs.h"

/****************Begin Custom Variables*************/
void invert_transformation(float result[4][4], float tran[4][4]) {
    int i, j;
    /* Upper left 3x3 of result is transpose of upper left 3x3 of tran. */
    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j) result[i][j] = tran[j][i];
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

void matmult(float ans[4][4], float matrix1[4][4], float matrix2[4][4]) {
    int i, j, k;
    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++) ans[i][j] = 0.0;
    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++)
            for (k = 0; k < 4; k++) ans[i][j] += matrix1[i][k] * matrix2[k][j];
}

void create_312_transform(float transform[4][4], float xt, float yt, float zt,
                          float zr, float xr, float yr) {
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
