// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "domain/pose_file_io.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace jta {
namespace pose_file {
namespace {

constexpr int kFieldCount = 6;

void WriteRow(
    std::ostream& out,
    double x,
    double y,
    double z,
    double z_rot,
    double x_rot,
    double y_rot,
    bool last_newline) {
    // Column order is X, Y, Z, Z_ROT, X_ROT, Y_ROT (matches the GUI and the
    // JTA/JT headers). Values are comma-separated; the value/length-dependent
    // tab padding of the original inspector output is cosmetic and the reader
    // is whitespace-tolerant, so a single tab separator keeps output clean and
    // losslessly round-trippable.
    auto emit = [&out](double v) {
        out << std::setprecision(10) << v << ",\t";
    };
    emit(x);
    emit(y);
    emit(z);
    emit(z_rot);
    emit(x_rot);
    out << std::setprecision(10) << y_rot << ",";
    if (last_newline) {
        out << "\n";
    } else {
        out << "\t";
    }
}

// Tokenize one data line into up to 6 numeric strings. Accepts comma- or
// whitespace-separated columns (comma-first for the GUI's native files, then
// whitespace for JT_EULER_312). Returns false if fewer than 6 are found.
bool TokenizeLine(
    const std::string& line,
    std::array<std::string, kFieldCount>& tokens) {
    std::vector<std::string> fields;
    // Split on commas first.
    std::string cur;
    bool any_comma = line.find(',') != std::string::npos;
    if (any_comma) {
        for (char c : line) {
            if (c == ',') {
                fields.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        fields.push_back(cur);
    } else {
        std::istringstream iss(line);
        std::string tok;
        while (iss >> tok) {
            fields.push_back(tok);
        }
    }
    // The first token may be padded with leading spaces (e.g. the first
    // column of a JT header line); trim each field.
    for (auto& f : fields) {
        while (!f.empty() &&
               std::isspace(static_cast<unsigned char>(f.front()))) {
            f.erase(f.begin());
        }
        while (!f.empty() &&
               std::isspace(static_cast<unsigned char>(f.back()))) {
            f.pop_back();
        }
    }
    if (fields.size() < kFieldCount) {
        return false;
    }
    for (int i = 0; i < kFieldCount; ++i) {
        tokens[i] = fields[i];
    }
    return true;
}

// Map the 6 stored columns (X,Y,Z,Z_ROT,X_ROT,Y_ROT) onto Point6D
// (x,y,z, x_rot, y_rot, z_rot).
Point6D ColumnsToPoint(const std::array<std::string, kFieldCount>& t) {
    double x = std::stod(t[0]);
    double y = std::stod(t[1]);
    double z = std::stod(t[2]);
    double z_rot = std::stod(t[3]);
    double x_rot = std::stod(t[4]);
    double y_rot = std::stod(t[5]);
    return Point6D(x, y, z, x_rot, y_rot, z_rot);
}

// Collect the non-empty, non-whitespace-only lines of a stream (a whitespace-
// only trailing/newline line is not a data frame).
std::vector<std::string> ReadLines(std::istream& in) {
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        bool whitespace_only = true;
        for (char c : line) {
            if (!std::isspace(static_cast<unsigned char>(c))) {
                whitespace_only = false;
                break;
            }
        }
        if (whitespace_only) {
            continue;
        }
        lines.push_back(line);
    }
    return lines;
}

}  // namespace

bool WritePose(std::ostream& out, const Point6D& pose) {
    if (!out) {
        return false;
    }
    out << "JTA_EULER_POSE\n"
        << "X_TRAN\t\tY_TRAN\t\tZ_TRAN\t\tZ_ROT\t\tX_ROT\t\tY_ROT\n";
    WriteRow(
        out,
        pose.x,
        pose.y,
        pose.z,
        pose.za,
        pose.xa,
        pose.ya,
        /*last_newline=*/true);
    return static_cast<bool>(out);
}

LoadResult ReadPose(std::istream& in, Point6D& out) {
    LoadResult res;
    auto lines = ReadLines(in);
    if (lines.empty()) {
        return res;
    }
    // The pose is either the third line of a JTA_EULER_POSE file (index 2) or,
    // for a raw single-row (.jtp-style) file, is the file's first line.
    const std::string* data_line = nullptr;
    if (!lines[0].empty() && lines[0] == "JTA_EULER_POSE") {
        res.kind = FileKind::Pose;
        if (lines.size() >= 3) {
            data_line = &lines[2];
        }
    } else {
        // Raw single-row file: the pose is line 0 itself (as the GUI's .jtp
        // loader reads InputList[0]). Only try line 0 so an unrelated header
        // is not mistaken for a pose.
        res.kind = FileKind::Pose;
        std::array<std::string, kFieldCount> probe;
        if (TokenizeLine(lines[0], probe)) {
            data_line = &lines[0];
        }
    }
    if (!data_line) {
        return res;
    }

    std::array<std::string, kFieldCount> tokens;
    if (!TokenizeLine(*data_line, tokens)) {
        return res;
    }
    if (tokens[0] == "NOT_OPTIMIZED") {
        res.not_optimized = true;
        return res;
    }
    try {
        out = ColumnsToPoint(tokens);
    } catch (const std::exception&) {
        return res;
    }
    res.ok = true;
    return res;
}

bool WriteKinematics(std::ostream& out, const std::vector<Point6D>& poses) {
    if (!out) {
        return false;
    }
    out << "JTA_EULER_KINEMATICS\n"
        << "X_TRAN\t\tY_TRAN\t\tZ_TRAN\t\tZ_ROT\t\tX_ROT\t\tY_ROT\n";
    for (const auto& p : poses) {
        WriteRow(out, p.x, p.y, p.z, p.za, p.xa, p.ya, /*last_newline=*/true);
    }
    return static_cast<bool>(out);
}

LoadResult ReadKinematics(
    std::istream& in,
    std::vector<std::optional<Point6D>>& out) {
    LoadResult res;
    auto lines = ReadLines(in);
    if (lines.empty()) {
        return res;
    }
    if (lines[0] != "JTA_EULER_KINEMATICS" && lines[0] != "JT_EULER_312") {
        return res;
    }
    res.kind = FileKind::Kinematics;
    // Data rows start after the header (line 0) + column title (line 1).
    // out is POSITION-PRESERVING: out[j] is the pose for frame j; a skipped
    // (NOT_OPTIMIZED or malformed) row leaves that frame as std::nullopt so
    // subsequent frames stay aligned (the original loader keyed frames by line
    // index, not by a compacted count).
    for (size_t i = 2; i < lines.size(); ++i) {
        std::array<std::string, kFieldCount> tokens;
        if (!TokenizeLine(lines[i], tokens)) {
            out.push_back(std::nullopt);
            continue;
        }
        if (tokens[0] == "NOT_OPTIMIZED") {
            res.not_optimized = true;
            out.push_back(std::nullopt);
            continue;
        }
        try {
            out.push_back(ColumnsToPoint(tokens));
        } catch (const std::exception&) {
            out.push_back(std::nullopt);  // malformed row -> that frame unset
        }
    }
    res.ok = std::any_of(
        out.begin(), out.end(), [](const std::optional<Point6D>& p) {
            return p.has_value();
        });
    return res;
}

bool WritePoseFile(const std::string& path, const Point6D& pose) {
    std::ofstream f(path);
    return WritePose(f, pose);
}

LoadResult ReadPoseFile(const std::string& path, Point6D& out) {
    std::ifstream f(path);
    if (!f) {
        return LoadResult{};
    }
    return ReadPose(f, out);
}

bool WriteKinematicsFile(
    const std::string& path,
    const std::vector<Point6D>& poses) {
    std::ofstream f(path);
    return WriteKinematics(f, poses);
}

LoadResult ReadKinematicsFile(
    const std::string& path,
    std::vector<std::optional<Point6D>>& out) {
    std::ifstream f(path);
    if (!f) {
        return LoadResult{};
    }
    return ReadKinematics(f, out);
}

}  // namespace pose_file
}  // namespace jta
