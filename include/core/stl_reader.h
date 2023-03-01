#pragma once
/*QT Headers*/
#include <qstring.h>
#include <qfile.h>
#include <qfileinfo.h>
#include <qtextstream.h>

/*Standard Library*/
#include <vector>

namespace stl_reader {

	/*ENUM for STL file status*/
	enum STL_STATUS { STL_INVALID, STL_ASCII, STL_BINARY };

	/*Function to determine if file is a valid STL file and, if so, whether it is binary or ascii*/
	STL_STATUS getStlFileFormat(const QString &path);

	/*STL reader function (binary or ascii)
	Populates two vector<floats>, one contains the triangle vertices, the other contains the triangle normals  */
	STL_STATUS readAnySTL(const QString &path, std::vector<float> &triangleVertices, std::vector<float> &triangleNormals);
}