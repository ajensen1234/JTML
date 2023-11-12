#pragma once
/*QT Headers*/
#include <QtCore/qfile.h>
#include <QtCore/qfileinfo.h>
#include <QtCore/qstring.h>
#include <QtCore/qtextstream.h>

/*Basic Linear Algebra Structures*/
#include "BasicLinearAlgebraStructures.h"

/*Standard Library*/
#include <string>
#include <vector>

/*KP reader function
Populates a vector of vector<XYZPoint> containing the key points in the file,
returns if error or not*/
bool readKP(const QString &path, std::vector<basic_la::XYZPoint> &kp_storage);