// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*STLReader header*/
#include "core/STLReader.h"

/*Standard Library*/
#include <bitset>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace stl_reader_BIG {
STL_STATUS getStlFileFormat(const QString& path) {
    /* TAKEN FROM:
     * https://stackoverflow.com/questions/26171521/verifying-that-an-stl-file-is-ascii-or-binary
     */
    /* CREDIT FOR THIS FUNCTION GOES TO STACK OVERFLOW USERS: OnlineCop, Remy
     * Lebeau, and Powerswitch*/
    // Each facet contains:
    //  - Normals: 3 floats (4 bytes)
    //  - Vertices: 3x floats (4 byteach, 12 bytes total)
    //  - AttributeCount: 1 short (2 bytes)
    // Total: 50 bytes per facet
    const size_t facetSize =
        3 * sizeof(float) + 3 * 3 * sizeof(float) + sizeof(uint16_t);

    QFile file(path);
    bool canFileBeOpened = file.open(QIODevice::ReadOnly);
    if (!canFileBeOpened) {
        qDebug("\n\tUnable to open \"%s\"", qPrintable(path));
        return STL_INVALID;
    }

    QFileInfo fileInfo(path);
    size_t fileSize = fileInfo.size();

    // The minimum size of an empty ASCII file is 15 bytes.
    if (fileSize < 15) {
        // "solid " and "endsolid " markers for an ASCII file
        qDebug("\n\tThe STL file is not long enough (%u bytes).",
               static_cast<uint>(fileSize));
        file.close();
        return STL_INVALID;
    }

    // Binary files should never start with "solid ", but just in case, check
    // for ASCII, and if not valid then check for binary...

    // Look for text "solid " in first 6 bytes, indicating the possibility that
    // this is an ASCII STL format.
    QByteArray sixBytes = file.read(6);
    if (sixBytes.startsWith("solid ")) {
        QString line;
        QTextStream in(&file);
        while (!in.atEnd()) {
            line = in.readLine();
            if (line.contains("endsolid")) {
                file.close();
                return STL_ASCII;
            }
        }
    }

    // Wasn't an ASCII file. Reset and check for binary.
    if (!file.reset()) {
        qDebug("\n\tCannot seek to the 0th byte (before the header)");
        file.close();
        return STL_INVALID;
    }

    // 80-byte header + 4-byte "number of triangles" for a binary file
    if (fileSize < 84) {
        qDebug("\n\tThe STL file is not long enough (%u bytes).",
               static_cast<uint>(fileSize));
        file.close();
        return STL_INVALID;
    }

    // Header is from bytes 0-79; numTriangleBytes starts at byte offset 80.
    if (!file.seek(80)) {
        qDebug("\n\tCannot seek to the 80th byte (after the header)");
        file.close();
        return STL_INVALID;
    }

    // Read the number of triangles, uint32_t (4 bytes), little-endian
    QByteArray nTrianglesBytes = file.read(4);
    if (nTrianglesBytes.size() != 4) {
        qDebug("\n\tCannot read the number of triangles (after the header)");
        file.close();
        return STL_INVALID;
    }

    uint32_t nTriangles = *((uint32_t*)nTrianglesBytes.data());

    // Verify that file size equals the sum of header + nTriangles value + all
    // triangles
    if (fileSize == (84 + (nTriangles * facetSize))) {
        file.close();
        return STL_BINARY;
    }

    return STL_INVALID;
}

STL_STATUS readAnySTL(const QString& path,
                      std::vector<std::vector<float>>& stl_storage) {
    /*STL reader function (binary or ascii)
    Returns a vector of two vector<floats>, one contains the traingle vertices,
    the other contains the triangle normals*/

    // Read in file and get status
    switch (getStlFileFormat(path)) {
        case STL_INVALID:
            std::cout << "ERROR: Invalid STL file.\n";
            return STL_INVALID;
            break;
        case STL_ASCII: {
            // OLD CRUDE ASCII READER
            std::ifstream file(path.toStdString());
            std::vector<float> triangleVertices;
            std::vector<float> triangleNormals;
            std::string str;
            while (std::getline(file, str)) {
                std::string buf;
                std::stringstream ss(str);

                std::vector<std::string> tokens;
                while (ss >> buf) {
                    tokens.push_back(buf);
                }

                if (tokens.size() == 4) {
                    if (tokens[0] == "vertex") {
                        triangleVertices.push_back(std::stof(tokens[1]));
                        triangleVertices.push_back(std::stof(tokens[2]));
                        triangleVertices.push_back(std::stof(tokens[3]));
                    }
                }
                if (tokens.size() == 5) {
                    if (tokens[0] == "facet" && tokens[1] == "normal") {
                        triangleNormals.push_back(std::stof(tokens[2]));
                        triangleNormals.push_back(std::stof(tokens[3]));
                        triangleNormals.push_back(std::stof(tokens[4]));
                    }
                }
            }
            stl_storage.clear();
            stl_storage.push_back(triangleVertices);
            stl_storage.push_back(triangleNormals);
            return STL_ASCII;
            break;
        }
        case STL_BINARY: {
            std::ifstream stl_file(path.toStdString().c_str(),
                                   std::ios::in | std::ios::binary);
            if (!stl_file) {
                std::cout << "ERROR: Could not read STL file.\n";
                return STL_INVALID;
            }

            /*Read in 80 byte header and 4 byte unsigned int representing the
             * number of triangles*/
            char header_info[80];
            char n_triangles[4];
            stl_file.read(header_info,
                          80);  // We do nothing with this information
            stl_file.read(n_triangles, 4);

            /*Storage (should someday update this)*/
            std::vector<float> triangleVertices;
            std::vector<float> triangleNormals;

            for (unsigned int i = 0; i < *(unsigned int*)(&n_triangles[0]);
                 i++) {
                /*Read in each triangle (little-endian assumed):
                1 normal (x,y,z)	(1 x 3 of 4 byte float)
                3 vertices (x,y,z)  (3 x 3 of 4 byte float)
                1 attribute			(1 x 1 of 2 byte short unsigned
                int) for a total of 50 bytes per triangle. */
                char f_buf[sizeof(float)];
                for (unsigned int j = 0; j < 3; j++) {
                    stl_file.read(f_buf, 4);
                    float value = *(float*)(&f_buf[0]);
                    triangleNormals.push_back(value);
                }
                for (unsigned int j = 0; j < 9; j++) {
                    stl_file.read(f_buf, 4);
                    float value = *(float*)(&f_buf[0]);
                    triangleVertices.push_back(value);
                }
                char temp[2];
                stl_file.read(temp, 2);  // Do nothing with this
            }
            stl_storage.clear();
            stl_storage.push_back(triangleVertices);
            stl_storage.push_back(triangleNormals);
            return STL_BINARY;
            break;
        }
    }
}
}  // namespace stl_reader_BIG
