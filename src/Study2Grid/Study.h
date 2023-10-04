#pragma once
/*Standard Library Files*/
#include <string>
#include <vector>

/*Namespace*/
using namespace std;

/*Simple data structure that houses full file listings for each component of a study on bones or implants
as well as basic info about the study.
This includes:
FULL FILE LISTINGS:
1. Vector of stl files
2. Calibration text file
3. Vector of TIF/TIFF files
4. Vector of kinematics files (.JTAK or .JTS) - one kinematics file corresponding to exactly one stl file. 
	No kinematics file or stl file should not have a matching pair of the other type.
5. Study Directory
INFO:
1. Passed check for identifying all files and consistent image size?
2. Width for Images
3. Height for Images
4. Overall Study Name (ex: LIMA)
5. Patient Name
6. Session Number
7. Movement Name
8. Movement Number (for specific name)
9. STL file types
10. STL file basenames
FUNCTIONS:
1. Constructor (option to check if images are all the same size)
*/
struct Study {

	/*Functions*/
	Study(string fluoro_study_dir, bool key_points);
	Study();

	/*File Listings*/
	vector<string> stl_files_;
	string calibration_;
	vector<string> images_;
	vector<string> kin_files_;
	string study_dir_;

	/*Information*/
	bool passed_check_;
	bool key_points_;
	int width_;
	int height_;
	string study_name_;
	string patient_name_;
	int sess_num_;
	string mov_name_;
	int mov_num_;
	vector<string> stl_types_;
	vector<string> stl_basenames_;
	vector<bool> stl_basenames_have_kp_; // Same order as stl_types/basenames, states whether that basename has a kp file or not. If key_points_ == false, these are all automatically false.

};