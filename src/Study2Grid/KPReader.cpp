/*KP Reader*/
#include "KPReader.h"

/*Standard Library*/
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <fstream>

using namespace basic_la;

bool readKP(const QString &path, std::vector<XYZPoint> &kp_storage) {
	/*Clear Storage*/
	kp_storage.clear();
	/* New Method Uses IO Stream to Account for the different types of Keypoints that Shapeworks Generates*/

	/*Create Object to read file into*/
	fstream kp_file;

	/*Open keypoint file*/
	/*todo: add some checks to make sure that it is a valid keypoint file*/
	const std::string newpath = path.toStdString();
	kp_file.open(newpath, ios::in);

	if (kp_file.is_open())
	{
		std::string line; // creating string to read lines to
		double kp[3];
		//todo: figure out a good way to save memory here - i think it might be clogging things up
		/*Loop through each line*/
		while (std::getline(kp_file, line))
		{
			/*Create Double for each new loop*/
			

			/*Use string stream to load in each line*/
			std::stringstream kp_str(line);

			/*For each of the 3 values, populate the kpo double values*/
			kp_str >> kp[0] >> kp[1] >> kp[2];			

			/*Populate the Keypoint storage object*/
			kp_storage.push_back(XYZPoint(kp[0], kp[1], kp[2]));
		}
		kp_file.close(); // close the file
		return true;
	}
}




//	/* Old Way of Doing Things using Qt File Management*/
//	
//	 /*Read Each Line*/
//	QFile inputFile(path);
//	if (inputFile.open(QIODevice::ReadOnly))
//	{
//		QTextStream in(&inputFile);
//		unsigned int line_index = 0;
//		while (!in.atEnd())
//		{
//			QString line = in.readLine();
//			if (line_index == 0) {
//				if (line.simplified() != "BEGIN_KP") {
//					std::cout << "\nError: No proper header in .kp file: " + path.toStdString();
//					return false;
//				}
//			}
//			else {
//				if (line.simplified() == "END_KP")
//					return true;
//				QStringList name_and_location = line.simplified().split(":"); // split at colon
//				if (name_and_location.size() != 2) {
//					std::cout << "\nError: Incorrect size when split along colon character .kp file: " + path.toStdString();
//					return false;
//				}
//				QStringList location = name_and_location[1].split(",");
//				if (location.size() != 3) {
//					location = name_and_location[1].simplified().split(" ");
//					if (location.size() != 3) {
//						std::cout << "\nError: There are not three scalars representing a point in the .kp file: " + path.toStdString();
//						return false;
//					}
//				}
//				kp_storage.push_back(XYZPoint(location[0].simplified().toFloat(), location[1].simplified().toFloat(), location[2].simplified().toFloat()));
//			}
//			line_index++;
//		}
//		inputFile.close();
//	}
//	else {
//		std::cout << "\nError: Cannot open .kp file: " + path.toStdString();
//		return false;
//	}
//} 