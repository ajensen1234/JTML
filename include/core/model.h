// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#pragma once

#include <string>
#include <vector>
#include <vtkSmartPointer.h>
#include <vtkSTLReader.h>

/**
 * @brief Class representing a 3D model loaded from an STL file
 * 
 * Handles loading and storing of 3D model data, including vertices and normals.
 * Provides access to both raw geometry data and VTK-based representations.
 */
class Model {
public:
    /**
     * @brief Construct a new Model object
     * 
     * @param file_location Path to the STL file
     * @param model_name Name to identify the model
     * @param model_type Type classification of the model
     */
    Model(std::string file_location = "", std::string model_name = "", std::string model_type = "");

    // Public member variables for access to model properties
    std::string file_location_;  ///< Path to the STL file
    std::string model_name_;     ///< Name identifier for the model
    std::string model_type_;     ///< Type classification of the model
    bool initialized_correctly_; ///< Whether the model loaded successfully

    // Getters for geometry data
    const std::vector<float>& getVertices() const { return triangle_vertices_; }
    const std::vector<float>& getNormals() const { return triangle_normals_; }
    float* getVerticesData() { return triangle_vertices_.data(); }
    float* getNormalsData() { return triangle_normals_.data(); }
    size_t getVertexCount() const { return triangle_vertices_.size(); }
    size_t getNormalCount() const { return triangle_normals_.size(); }
    
    // VTK access
    vtkSmartPointer<vtkSTLReader> getReader() const { return cad_reader_; }

private:
    /**
     * @brief Load vertices and normals from the STL file
     * 
     * @return true if loading was successful, false otherwise
     */
    bool LoadVerticesAndNormals();

    std::vector<float> triangle_vertices_;  ///< Raw vertex data (x,y,z triplets)
    std::vector<float> triangle_normals_;   ///< Raw normal data (x,y,z triplets)
    vtkSmartPointer<vtkSTLReader> cad_reader_;  ///< VTK STL reader instance
};