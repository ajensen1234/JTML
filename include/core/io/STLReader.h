// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#pragma once

#include <QString>
#include <vector>

namespace jtml {
namespace io {

/**
 * @brief A modern C++ class for reading STL (STereoLithography) files in both ASCII and binary formats.
 * 
 * This class provides functionality to read 3D model data from STL files, supporting both ASCII
 * and binary formats. It offers multiple interfaces for reading the data, either as separate
 * vertex and normal vectors or as a combined data structure.
 */
class STLReader {
public:
    /**
     * @brief Enumeration of possible STL file formats
     */
    enum class Status {
        Invalid,  ///< File is invalid or cannot be read
        ASCII,    ///< File is in ASCII format
        Binary    ///< File is in binary format
    };

    /**
     * @brief Read STL file into separate vertex and normal vectors
     * 
     * @param path Path to the STL file
     * @param vertices Output vector for vertex coordinates (x,y,z triplets)
     * @param normals Output vector for normal vectors (x,y,z triplets)
     * @return true if successful, false otherwise
     */
    static bool read(const QString& path, std::vector<float>& vertices, std::vector<float>& normals);

    /**
     * @brief Read STL file into a vector of vectors (legacy format support)
     * 
     * @param path Path to the STL file
     * @param data Output vector containing {vertices, normals}
     * @return true if successful, false otherwise
     */
    static bool read(const QString& path, std::vector<std::vector<float>>& data);

    /**
     * @brief Get file format without reading full content
     * 
     * @param path Path to the STL file
     * @return Status indicating the file format
     */
    static Status getFileFormat(const QString& path);

private:
    /**
     * @brief Read an ASCII format STL file
     */
    static bool readASCII(const QString& path, std::vector<float>& vertices, std::vector<float>& normals);

    /**
     * @brief Read a binary format STL file
     */
    static bool readBinary(const QString& path, std::vector<float>& vertices, std::vector<float>& normals);
};

} // namespace io
} // namespace jtml