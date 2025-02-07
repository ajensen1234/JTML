// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "core/model.h"
#include "core/io/STLReader.h"

#include <sstream>

using namespace std;

Model::Model(std::string file_location, std::string model_name, std::string model_type) {
    // Set Public File Location string
    file_location_ = file_location;
    model_name_ = model_name;
    model_type_ = model_type;

    // Load STL File to CADReader
    cad_reader_ = vtkSmartPointer<vtkSTLReader>::New();
    cad_reader_->SetFileName(file_location.c_str());

    // Load Vertices and Normals
    initialized_correctly_ = LoadVerticesAndNormals();
}

bool Model::LoadVerticesAndNormals() {
    std::vector<float> vertices, normals;
    bool success = jtml::io::STLReader::read(QString::fromStdString(file_location_), vertices, normals);
    if (success) {
        triangle_vertices_ = std::move(vertices);
        triangle_normals_ = std::move(normals);
    }
    return success;
}