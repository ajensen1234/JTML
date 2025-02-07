// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "core/io/STLReader.h"

#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <fstream>
#include <sstream>
#include <string>

namespace jtml {
namespace io {

STLReader::Status STLReader::getFileFormat(const QString& path) {
    const size_t facetSize = 3 * sizeof(float) + 3 * 3 * sizeof(float) + sizeof(uint16_t);

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug("\n\tUnable to open \"%s\"", qPrintable(path));
        return Status::Invalid;
    }

    QFileInfo fileInfo(path);
    size_t fileSize = fileInfo.size();

    // The minimum size of an empty ASCII file is 15 bytes
    if (fileSize < 15) {
        qDebug("\n\tThe STL file is not long enough (%u bytes).", static_cast<uint>(fileSize));
        file.close();
        return Status::Invalid;
    }

    // Check for ASCII format
    QByteArray sixBytes = file.read(6);
    if (sixBytes.startsWith("solid ")) {
        QString line;
        QTextStream in(&file);
        while (!in.atEnd()) {
            line = in.readLine();
            if (line.contains("endsolid")) {
                file.close();
                return Status::ASCII;
            }
        }
    }

    // Check for binary format
    if (!file.reset()) {
        qDebug("\n\tCannot seek to the 0th byte");
        file.close();
        return Status::Invalid;
    }

    if (fileSize < 84) {
        qDebug("\n\tThe STL file is not long enough for binary format (%u bytes).", static_cast<uint>(fileSize));
        file.close();
        return Status::Invalid;
    }

    if (!file.seek(80)) {
        qDebug("\n\tCannot seek to the 80th byte");
        file.close();
        return Status::Invalid;
    }

    QByteArray nTrianglesBytes = file.read(4);
    if (nTrianglesBytes.size() != 4) {
        qDebug("\n\tCannot read the number of triangles");
        file.close();
        return Status::Invalid;
    }

    uint32_t nTriangles = *reinterpret_cast<const uint32_t*>(nTrianglesBytes.data());

    if (fileSize == (84 + (nTriangles * facetSize))) {
        file.close();
        return Status::Binary;
    }

    file.close();
    return Status::Invalid;
}

bool STLReader::read(const QString& path, std::vector<float>& vertices, std::vector<float>& normals) {
    vertices.clear();
    normals.clear();

    Status format = getFileFormat(path);
    switch (format) {
        case Status::Invalid:
            qDebug("Error: Invalid STL file");
            return false;
            
        case Status::ASCII:
            return readASCII(path, vertices, normals);
            
        case Status::Binary:
            return readBinary(path, vertices, normals);
            
        default:
            return false;
    }
}

bool STLReader::read(const QString& path, std::vector<std::vector<float>>& data) {
    std::vector<float> vertices, normals;
    if (!read(path, vertices, normals)) {
        return false;
    }
    
    data.clear();
    data.push_back(std::move(vertices));
    data.push_back(std::move(normals));
    return true;
}

bool STLReader::readASCII(const QString& path, std::vector<float>& vertices, std::vector<float>& normals) {
    std::ifstream file(path.toStdString());
    if (!file) {
        qDebug("Error: Could not open ASCII STL file");
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::string buf;
        std::stringstream ss(line);
        std::vector<std::string> tokens;
        
        while (ss >> buf) {
            tokens.push_back(buf);
        }

        if (tokens.size() == 4 && tokens[0] == "vertex") {
            vertices.push_back(std::stof(tokens[1]));
            vertices.push_back(std::stof(tokens[2]));
            vertices.push_back(std::stof(tokens[3]));
        }
        else if (tokens.size() == 5 && tokens[0] == "facet" && tokens[1] == "normal") {
            normals.push_back(std::stof(tokens[2]));
            normals.push_back(std::stof(tokens[3]));
            normals.push_back(std::stof(tokens[4]));
        }
    }

    return !vertices.empty() && !normals.empty();
}

bool STLReader::readBinary(const QString& path, std::vector<float>& vertices, std::vector<float>& normals) {
    std::ifstream file(path.toStdString(), std::ios::in | std::ios::binary);
    if (!file) {
        qDebug("Error: Could not open binary STL file");
        return false;
    }

    // Skip header
    file.seekg(80);
    
    // Read number of triangles
    uint32_t triangleCount;
    file.read(reinterpret_cast<char*>(&triangleCount), 4);

    // Pre-allocate vectors
    vertices.reserve(triangleCount * 9);  // 3 vertices per triangle, 3 coordinates each
    normals.reserve(triangleCount * 3);   // 1 normal per triangle, 3 coordinates

    for (uint32_t i = 0; i < triangleCount; ++i) {
        float normal[3];
        float vertex[9];
        
        // Read normal
        file.read(reinterpret_cast<char*>(normal), sizeof(float) * 3);
        // Read vertices
        file.read(reinterpret_cast<char*>(vertex), sizeof(float) * 9);
        // Skip attribute
        file.seekg(2, std::ios::cur);

        // Store normal
        normals.insert(normals.end(), normal, normal + 3);
        // Store vertices
        vertices.insert(vertices.end(), vertex, vertex + 9);
    }

    return !vertices.empty() && !normals.empty();
}

} // namespace io
} // namespace jtml