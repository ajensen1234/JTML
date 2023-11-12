#pragma once
/*QT Headers*/
#include <QtCore/qfile.h>
#include <QtCore/qfileinfo.h>
#include <QtCore/qstring.h>
#include <QtCore/qtextstream.h>

/*Standard Library*/
#include <vector>

namespace stl_reader_BIG {

/*ENUM for STL file status*/
enum STL_STATUS { STL_INVALID, STL_ASCII, STL_BINARY };

/*Function to determine if file is a valid STL file and, if so, whether it is
 * binary or ascii*/
STL_STATUS getStlFileFormat(const QString &path);

/*STL reader function (binary or ascii)
Populates a vector of two vector<floats>, one contains the traingle vertices,
the other contains the triangle normals  */
STL_STATUS readAnySTL(const QString &path,
                      std::vector<std::vector<float>> &stl_storage);
}  // namespace stl_reader_BIG
