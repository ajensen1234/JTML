/*File Parser Headers*/
#include "FileParsers.h"
#include <filesystem>

/*Standard Library Namespace*/
using namespace std;

/*File parsers*/
/*Read In Everything (Including other directories)*/
struct path_leaf_string_directories
{
	std::string operator()(const std::filesystem::directory_entry& entry) const
	{
		std::string extension = entry.path().extension().string();
		if (extension == "")
			return entry.path().string();
		else return "";
	}
};
void read_directory_for_directories(const std::string& name, vector<string>& v)
{
	std::filesystem::path p(name);
	std::filesystem::directory_iterator start(p);
	std::filesystem::directory_iterator end;
	std::transform(start, end, std::back_inserter(v), path_leaf_string_directories());
	v.erase(std::remove_if(v.begin(), v.end(), [](const string& x) {return x == ""; }), v.end());
};
/*Just Read Tif/Tiffs*/
struct path_leaf_string_tif
{
	std::string operator()(const std::filesystem::directory_entry& entry) const
	{
		std::string extension = entry.path().extension().string();
		if (extension == ".tif" || extension == ".tiff")
			return entry.path().string();
		else return "";
	}
};
void read_directory_for_tif(const std::string& name, vector<string>& v)
{
	std::filesystem::path p(name);
	std::filesystem::directory_iterator start(p);
	std::filesystem::directory_iterator end;
	std::transform(start, end, std::back_inserter(v), path_leaf_string_tif());
	v.erase(std::remove_if(v.begin(), v.end(), [](const string& x) {return x == ""; }), v.end());
};
/*Just Read .JTS Files*/
struct path_leaf_string_jts
{
	std::string operator()(const std::filesystem::directory_entry& entry) const
	{
		std::string extension = entry.path().extension().string();
		if (extension == ".jts")
			return entry.path().string();
		else return "";
	}
};
void read_directory_for_jts(const std::string& name, vector<string>& v)
{
	std::filesystem::path p(name);
	std::filesystem::directory_iterator start(p);
	std::filesystem::directory_iterator end;
	std::transform(start, end, std::back_inserter(v), path_leaf_string_jts());
	v.erase(std::remove_if(v.begin(), v.end(), [](const string& x) {return x == ""; }), v.end());
};
/*Just Read .JTAK Files*/
struct path_leaf_string_jtak
{
	std::string operator()(const std::filesystem::directory_entry& entry) const
	{
		std::string extension = entry.path().extension().string();
		if (extension == ".jtak")
			return entry.path().string();
		else return "";
	}
};
void read_directory_for_jtak(const std::string& name, vector<string>& v)
{
	std::filesystem::path p(name);
	std::filesystem::directory_iterator start(p);
	std::filesystem::directory_iterator end;
	std::transform(start, end, std::back_inserter(v), path_leaf_string_jtak());
	v.erase(std::remove_if(v.begin(), v.end(), [](const string& x) {return x == ""; }), v.end());
};
/*Just Read .STL Files*/
struct path_leaf_string_stl
{
	std::string operator()(const std::filesystem::directory_entry& entry) const
	{
		std::string extension = entry.path().extension().string();
		if (extension == ".stl")
			return entry.path().string();
		else return "";
	}
};
void read_directory_for_stl(const std::string& name, vector<string>& v)
{
	std::filesystem::path p(name);
	std::filesystem::directory_iterator start(p);
	std::filesystem::directory_iterator end;
	std::transform(start, end, std::back_inserter(v), path_leaf_string_stl());
	v.erase(std::remove_if(v.begin(), v.end(), [](const string& x) {return x == ""; }), v.end());
};
/*Just Read .TXT Files*/
struct path_leaf_string_txt
{
	std::string operator()(const std::filesystem::directory_entry& entry) const
	{
		std::string extension = entry.path().extension().string();
		if (extension == ".txt")
			return entry.path().string();
		else return "";
	}
};
void read_directory_for_txt(const std::string& name, vector<string>& v)
{
	std::filesystem::path p(name);
	std::filesystem::directory_iterator start(p);
	std::filesystem::directory_iterator end;
	std::transform(start, end, std::back_inserter(v), path_leaf_string_txt());
	v.erase(std::remove_if(v.begin(), v.end(), [](const string& x) {return x == ""; }), v.end());
};
