#pragma once
/*Standard Library*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>
//#include <windows.h>
//#include <direct.h>
//#include <io.h>
#include <random>
#include <chrono>
#include <filesystem>
#include <algorithm>

/*File parsers (Reads a directory for specified file types)*/
void read_directory_for_directories(const std::string& name, std::vector< std::string>& v);
void read_directory_for_tif(const std::string& name, std::vector< std::string>& v);
void read_directory_for_jts(const std::string& name, std::vector< std::string>& v);
void read_directory_for_jtak(const std::string& name, std::vector< std::string>& v);
void read_directory_for_stl(const std::string& name, std::vector< std::string>& v);
void read_directory_for_txt(const std::string& name, std::vector< std::string>& v);