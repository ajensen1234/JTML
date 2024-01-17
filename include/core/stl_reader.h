/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#pragma once
/*QT Headers*/
#include <QtCore/qfile.h>
#include <QtCore/qfileinfo.h>
#include <QtCore/qstring.h>
#include <QtCore/qtextstream.h>

/*Standard Library*/
#include <vector>

namespace stl_reader {

/*ENUM for STL file status*/
enum STL_STATUS { STL_INVALID, STL_ASCII, STL_BINARY };

/*Function to determine if file is a valid STL file and, if so, whether it is
 * binary or ascii*/
STL_STATUS getStlFileFormat(const QString &path);

/*STL reader function (binary or ascii)
Populates two vector<floats>, one contains the triangle vertices, the other
contains the triangle normals  */
STL_STATUS readAnySTL(const QString &path, std::vector<float> &triangleVertices,
                      std::vector<float> &triangleNormals);
}  // namespace stl_reader
