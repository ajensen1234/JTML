// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*Model Header*/
#include "core/model.h"

/*Standard*/
#include <sstream>

using namespace std;

Model::Model(std::string file_location, std::string model_name,
             std::string model_type) {
    /*Set Public File Location string*/
    file_location_ = file_location;
    model_name_ = model_name;
    model_type_ = model_type;

    /*Load STL File to CADReader*/
    cad_reader_ = vtkSmartPointer<vtkSTLReader>::New();
    cad_reader_->SetFileName(file_location.c_str());

    /*Load Vertices and Normals*/
    if (LoadVerticesAndNormals() == stl_reader::STL_INVALID) {
        initialized_correctly_ = false;
    } else {
        initialized_correctly_ = true;
    }
}

stl_reader::STL_STATUS Model::LoadVerticesAndNormals() {
    return stl_reader::readAnySTL(QString::fromStdString(file_location_),
                                  triangle_vertices_, triangle_normals_);
}
