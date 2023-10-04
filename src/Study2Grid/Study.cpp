/*Header for Study*/
#include "Study.h"

/*File Parsers*/
#include "FileParsers.h"

/*QT*/
#include <QtCore/qstring.h>
#include <QtCore/qstringlist.h>
#include <QtCore/qfileinfo.h>

/*OpenCV 3.1 Library*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*CV Namespace*/
using namespace cv;

Study::Study() {
	passed_check_ = false;
}

Study::Study(string fluoro_study_dir, bool key_points) {

	/*Save Study Directory*/
	study_dir_ = fluoro_study_dir;

	/*Save if Using Key Points or Not*/
	key_points_ = key_points;

	/*Begin with assuming passes check for correctly initialized*/
	passed_check_ = true;

	/*Get List of STL Files*/
	vector<string> stl_files;
	read_directory_for_stl(study_dir_, stl_files);
	/*Check There are STL Files, else return error*/
	if (stl_files.size() == 0) {
		cout << "\nError with " + study_dir_ << endl << "\t There are no STL files.";
		passed_check_ = false;
		return;
	}
	/*Create QFileInfos For All stl file names (stl file name should either be the type it is or end in _%type (for example MKTest_fem for a fem)*/
	for (int stl_file_ind = 0; stl_file_ind < stl_files.size(); stl_file_ind++) {
		QFileInfo info_stl(QString::fromStdString(stl_files[stl_file_ind]));
		QStringList stl_list = info_stl.baseName().split('_');
		string type_stl = stl_list[stl_list.size() - 1].toStdString();
		stl_files_.push_back(stl_files[stl_file_ind]);
		stl_types_.push_back(type_stl);
		stl_basenames_.push_back(info_stl.baseName().toStdString());
		QFileInfo info_kp(QString::fromStdString(fluoro_study_dir + "/" + type_stl + ".kp"));
		stl_basenames_have_kp_.push_back(info_kp.exists() && info_kp.isFile());
	}
	
	/*Check for duplicate stl model/implant types*/
	for (int stl_file_ind = 0; stl_file_ind < stl_types_.size(); stl_file_ind++) {
		for (int stl_file_ind2 = 0; stl_file_ind2 < stl_types_.size(); stl_file_ind2++) {
			if ((stl_file_ind != stl_file_ind2) && (stl_types_[stl_file_ind] == stl_types_[stl_file_ind2])) {
				cout << "\nError with " + study_dir_ << endl << "\t There are multiple stl files of type " + stl_types_[stl_file_ind] + ".";
				passed_check_ = false;
				return;
			}
		}
	}
	/*Get List of Text Files*/
	vector<string> txt_files;
	read_directory_for_txt(study_dir_, txt_files);
	/*Check There is Exactly One Text File, else return error*/
	if (txt_files.size() != 1) {
		cout << "\nError with " + study_dir_ << endl << "\t There is not exactly 1 text file (text files are reserved for calibration).";
		passed_check_ = false;
		return;
	}
	/*Create QFileInfos For Text file (should be called calibration)*/
	QFileInfo info_text_1(QString::fromStdString(txt_files[0]));
	if (info_text_1.baseName() == "calibration") {
		calibration_ = txt_files[0];
	}
	else {
		cout << "\nError with " + study_dir_ << endl << "\t There is no text file named \"calibration\".";
		passed_check_ = false;
		return;
	}

	/*Get list of JTAK Files*/
	vector<string> jtak_files;
	read_directory_for_jtak(study_dir_, jtak_files);
	/*Get list of JTS Files*/
	vector<string> jts_files;
	read_directory_for_jts(study_dir_, jts_files);
	/*Combine into one list*/
	vector<string> combined_jtak_jts_files_list;
	for (int jtak_ind = 0; jtak_ind < jtak_files.size(); jtak_ind++)
		combined_jtak_jts_files_list.push_back(jtak_files[jtak_ind]);
	for (int jts_ind = 0; jts_ind < jts_files.size(); jts_ind++)
		combined_jtak_jts_files_list.push_back(jts_files[jts_ind]);
	/*Ensure that each of the unique stl file types has a corresponding kinematics file and that there are no leftovers.
	The name of the kinematics file should be the model type.*/
	for (int stl_file_ind = 0; stl_file_ind<stl_types_.size(); stl_file_ind++) {
		bool found_corresponding_kin = false;
		for (int kin_indx = 0; kin_indx < combined_jtak_jts_files_list.size(); kin_indx++) {
			QFileInfo info_kin(QString::fromStdString(combined_jtak_jts_files_list[kin_indx]));
			string type_kin = info_kin.baseName().toStdString();
			if (stl_types_[stl_file_ind] == type_kin) {
				kin_files_.push_back(combined_jtak_jts_files_list[kin_indx]);
				combined_jtak_jts_files_list.erase(combined_jtak_jts_files_list.begin() + kin_indx);
				found_corresponding_kin = true;
				break;
			}
		}
		if (!found_corresponding_kin) {
			cout << "\nError with " + study_dir_ << endl << "\t Failed to find kinematics file for model type " + stl_types_[stl_file_ind] + ".";
			passed_check_ = false;
			return;
		}
	}
	if (combined_jtak_jts_files_list.size() > 0) {
		cout << "\nError with " + study_dir_ << endl << "\t There are kinematics files without matching stl files of the same model type.";
		passed_check_ = false;
		return;
	}

	/*Get list of TIF Files*/
	vector<string> tif_files;
	read_directory_for_tif(study_dir_, tif_files);
	/*Check There is Exactly At Least One Image File, else return error*/
	if (tif_files.size() == 0) {
		cout << "\nError with " + study_dir_ << endl << "\t There are no .tif or .tiff image files.";
		passed_check_ = false;
		return;
	}
	images_ = tif_files;
	/*Check that images_ files are all the same length. This can be problematic as Windows will list BOB1.tif, BOB2.tif, BOB3.tif, BOB10.tif in that order
	(and this is how it will be loaded into JointTrack or JointTrack Auto), but this will be read as BOB1.tif, BOB10.tif, BOB2.tif, BOB3.tif in this program
	and many other programs. To avoid this we keep the basename the same length aka BOB01.tif, BOB02.tif, BOB03.tif, BOB10.tif.
	Print out error if this happens.*/
	int image_basename_length = QFileInfo(QString::fromStdString(images_[0])).baseName().length();
	for (int imgs_indx = 0; imgs_indx < images_.size(); imgs_indx++) {
		if (QFileInfo(QString::fromStdString(images_[imgs_indx])).baseName().length() != image_basename_length) {
			cout << "\nError with " + study_dir_ << endl << "\t Not all of the image files have the same name length. They must all have the same length. " <<
				"\n\t If they came like this, please make them the same length.\n\t Then check that the kinematics are still correct for all models." <<
				"\n\t Here is a (or the) violating case:" <<
				"\n\t\t " << QFileInfo(QString::fromStdString(images_[0])).baseName().toStdString() <<" has character length "<< image_basename_length <<
				"\n\t\t " << QFileInfo(QString::fromStdString(images_[imgs_indx])).baseName().toStdString() << " has character length "
				<< QFileInfo(QString::fromStdString(images_[imgs_indx])).baseName().length();
			passed_check_ = false;
			return;
		}
	}
	/*Fill out Information About Study*/

	QStringList study_info_list = QString::fromStdString(study_dir_).split("\\");

	// Adding option to use '/' instead of '\\' for linux
	if (study_info_list.size() == 1) {
		study_info_list = QString::fromStdString(study_dir_).split("/");
	}

	QStringList movement_list = study_info_list[study_info_list.size() - 1].split("_"); // Movement
	mov_name_ = movement_list[0].toStdString();
	mov_num_ = movement_list[1].toInt();
	QStringList session_list = study_info_list[study_info_list.size() - 2].split("_"); // Session
	sess_num_ = session_list[1].toInt();
	patient_name_ = study_info_list[study_info_list.size() - 3].toStdString(); // Patient Name
	study_name_ = study_info_list[study_info_list.size() - 4].toStdString(); // Overall Study Name (File Usually Called Something a Bit Modified)


	/*Get Info about Images*/
	/*Initialize Image Sizes*/
	cv::Mat initial_img = cv::imread(tif_files[0], CV_8UC1);
	width_ = initial_img.cols;
	height_ = initial_img.rows;
	/*Check if images are the same size.*/
	for (int img_ind = 0; img_ind < tif_files.size(); img_ind++) {
		cv::Mat temp_img = cv::imread(tif_files[img_ind], CV_8UC1);
		if (width_ != initial_img.cols || height_ != initial_img.rows) {
			cout << "\nError with " + study_dir_ << endl << "\t Not all images are the same size.";
			passed_check_ = false;
			return;
		}
	}

	/*Delete all directories (and their contents) inside study.
	Create "Labels" folder and potentially "Key Points" folder and populate with subdirectories named for each %MODEL TYPE% (which are unique). */
	vector<string> directories_to_delete;
	std::error_code errorCode;
	read_directory_for_directories(study_dir_, directories_to_delete);
	for (int dir_to_del_ind = 0; dir_to_del_ind < directories_to_delete.size(); dir_to_del_ind++) {
		
		if (std::filesystem::remove_all(directories_to_delete[dir_to_del_ind], errorCode) == static_cast<std::uintmax_t>(-1)) {
			/*ERROR CODE is static_cast<std::uintmax_t>(-1)
			See: https://en.cppreference.com/w/cpp/experimental/fs/remove */
			cout << "\nError with " + study_dir_ << endl << "\t Could not delete existing subdirectories.\n\t " + errorCode.message();
			passed_check_ = false;
			return;
		}
	}
	
	/*Create Labels folder*/
	if (!std::filesystem::create_directory(study_dir_ + "/Labels", errorCode)) {
		cout << "\nError with " + study_dir_ << endl << "\t Could not create Labels directory.\n\t " <<errorCode.message();
		passed_check_ = false;
		return;
	}

	/*For each model type, create a subfolder in Labels*/
	for (int mod_type_ind = 0; mod_type_ind < stl_types_.size(); mod_type_ind++) {
		if (!std::filesystem::create_directory(study_dir_ + "/Labels/" + stl_types_[mod_type_ind], errorCode)) {
			cout << "\nError with " + study_dir_ << endl << "\t Could not create " + stl_types_[mod_type_ind] + " subdirectories in Labels directory.\n\t " << errorCode.message();
			passed_check_ = false;
			return;
		}
	}

	/*If Need Be Create Key_Points folder and for model types with */
	if (key_points) {
		if (!std::filesystem::create_directory(study_dir_ + "/Key_Points", errorCode)) {
			cout << "\nError with " + study_dir_ << endl << "\t Could not create Key_Points directory.\n\t " << errorCode.message();
			passed_check_ = false;
			return;
		}

		/*For each model type with a .kp file, create a subfolder in Labels*/
		for (int mod_type_ind = 0; mod_type_ind < stl_types_.size(); mod_type_ind++) {
			if (stl_basenames_have_kp_[mod_type_ind]) {
				if (!std::filesystem::create_directory(study_dir_ + "/Key_Points/" + stl_types_[mod_type_ind], errorCode)) {
					cout << "\nError with " + study_dir_ << endl << "\t Could not create " + stl_types_[mod_type_ind] + " subdirectories in Key Points directory.\n\t " << errorCode.message();
					passed_check_ = false;
					return;
				}
			}
		}
	}

	
	

};