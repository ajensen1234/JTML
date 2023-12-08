/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*Header for Model Class Includes:
 */

/*Standard*/
#include <string>
#include <vector>

/*VTK*/
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>

/*Custom STL Reader*/
#include "stl_reader.h"

#ifndef MODEL_H
#define MODEL_H

/*AS OF VERSION 3.3.1 SHOULD BE ABLE TO LOAD BOTH BINARY AND ASCII STL FILES*/
class Model {
   public:
    Model(std::string file_location, std::string model_name,
          std::string model_type);
    Model(){};
    std::string file_location_;                 // Store File Location for Model
    vtkSmartPointer<vtkSTLReader> cad_reader_;  // Stores CAD model
    std::vector<float> triangle_vertices_;      // Vector of Triangle Vertices
    std::vector<float> triangle_normals_;       // Vector of Triangle Normals
    /*Model Name: taken from prefix of file name. If duplicates a (x) is added*/
    std::string model_name_;
    /*Model Type: could be femur or implant or bone or type of bone, anything
     * really...*/
    std::string model_type_;
    /*Bool indicating initialized correctly*/
    bool initialized_correctly_;

   private:
    stl_reader::STL_STATUS LoadVerticesAndNormals();
};

#endif /* MODEL_H */