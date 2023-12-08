/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#pragma once
/*Standard Library*/
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
// #include <windows.h>
// #include <direct.h>
// #include <io.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <random>

/*File parsers (Reads a directory for specified file types)*/
void read_directory_for_directories(const std::string &name,
                                    std::vector<std::string> &v);
void read_directory_for_tif(const std::string &name,
                            std::vector<std::string> &v);
void read_directory_for_jts(const std::string &name,
                            std::vector<std::string> &v);
void read_directory_for_jtak(const std::string &name,
                             std::vector<std::string> &v);
void read_directory_for_stl(const std::string &name,
                            std::vector<std::string> &v);
void read_directory_for_txt(const std::string &name,
                            std::vector<std::string> &v);

bool alphabetic_compare(std::string a, std::string b);
