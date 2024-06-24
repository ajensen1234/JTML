// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*Standard Library*/
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#ifdef defined(_WIN32) || defined(_WIN64)
#define OS_WINDOWS 1
#include <direct.h>
#include <io.h>
#include <windows.h>
#else
#define OS_WINDOWS 0
#include <unistd.h>
#endif
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <random>

/*File Parsers*/
#include "FileParsers.h"

/*Cost Function Tools Library*/
#include "gpu_dilated_frame.cuh"
#include "gpu_edge_frame.cuh"
#include "gpu_frame.cuh"
#include "gpu_image.cuh"
#include "gpu_image_functions.cuh"
#include "gpu_intensity_frame.cuh"
#include "gpu_metrics.cuh"
#include "gpu_model.cuh"
#include "render_engine.cuh"

/*Cuda Random*/
#include <curand.h>
#include <curand_kernel.h>

/*OpenCV 3.1 Library*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*STL Reader*/
#include "STLReader.h"

/*QT*/
#include <QtCore/qfileinfo.h>
#include <QtCore/qstring.h>
#include <QtCore/qstringlist.h>
#include <QtCore/qtextstream.h>

/*Study Class*/
#include "Study.h"

/*Image Info Class*/
#include "ImageInfo.h"

/*Basic Linear Algebra Structures*/
#include "BasicLinearAlgebraStructures.h"

/*KP Reader*/
#include "KPReader.h"

using namespace std;
using namespace gpu_cost_function;
using namespace cv;
using namespace stl_reader;

int main() {
    //////////////////////////////////////////*VARIABLES TO
    /// EDIT*////////////////////////////////////////////
    ///
    /// This is an old way of doing things, the new way takes in a batch file
    /// that has all these parameters defined

    /*HOME FILE DIRECTORY*/

    // string home_dir =
<<<<<<< HEAD
    //     "/media/ajensen123@ad.ufl.edu/Andrew's External "
    //     "SSD/Data/Datasets_FemCleaned/Lima/Lima_Organized_Updated";

    string home_dir =
        "/media/ajensen123@ad.ufl.edu/Andrew's External "
        "SSD/Data/Datasets_TSA/Nagoya_Organized";

    /*Write File Directory*/
    string write_grids_dir = "/home/ajensen123@ad.ufl.edu/repo/jtml-TSA/imgs/";

    // string write_grids_dir =
    //     "/home/ajensen123@ad.ufl.edu/Documents/Lima_Grids/";
    /*study Name*/
=======
    // "/media/ajensen123@ad.ufl.edu/Andrew's External "
    //     "SSD/Data/Datasets_FemCleaned/Lima/Lima_Organized_Updated";

    string home_dir = "/home/ajensen123@ad.ufl.edu/repo/jtml-TSA/Nagoya/";

    // Write File Directory
    string write_grids_dir = "/home/ajensen123@ad.ufl.edu/repo/jtml-TSA/imgs/";

    // string write_grids_dir =
    //     "/home/ajensen123@ad.ufl.edu/Documents/Lima_Grids/";
    /*Study Name*/
>>>>>>> d48d61ba09dbe44836e6dd294328032fd729abc0
    string study_name = "Nagoya_Updated";

    /*GPU Device Chosen*/
    unsigned int GPU_DEVICE = 0;
    cudaSetDevice(GPU_DEVICE);

    /*Grid Writer variables*/
    int GRID_WIDTH = 1;
    int GRID_HEIGHT = 1;
    int IMAGE_HEIGHT = 1024;
    int IMAGE_WIDTH = 1024;
    bool USE_PADDING =
        true;  // When resizing images, use padding to maintain aspect ratio
    bool INVERT_IMAGE = false;  // Invert grayscale

    /*Shuffle images?*/
    bool RAND_SHUFFLE_IMGS = false;

    /*Generate Key Points Output when .kp file available?*/
    bool GENERATE_KP_DATA = true;

    /*Randomly Invert*/
    bool RAND_INVERT = false;

    ////////////////////////////////////////*END VARIABLES TO
    /// EDIT*//////////////////////////////////////////

    /*Check that only running KP Mode if Using 1x1 grids*/
    if (GRID_HEIGHT * GRID_WIDTH != 1 && GENERATE_KP_DATA) {
        std::cout << "Error: Must use 1x1 grids if generating KP data!";
        std::getchar();
        return 0;
    }

    /*List of Directories of Each Fluoroscopy*/
    /*All directories have format:
    %PATIENT NAME% ---> SESSION_# ---> %MOVEMENT TYPE%_#
    %PATIENT NAME% can be any string (we ignore the % signs above and use them
    to demark one block of text) %MOVEMENT TYPE% is one word (ignore the %
    signs) but may change. Instead of underscores, there might be spaces - this
    is ok. Within this directory should be stl files, kinematics files for each
    stl (jts or jtak and named with the same name as the name after the last
    underscore in the corresponding stl), images in .tif or .tiff format, a
    calibration text file (named calibration). There may or may not be a
    directory called Labels (caps not sensitive) with subdirectories names for
    each of the stl files. If there is, delete it (in fact, delete all non
    .tif/.tiff/.txt/.jts/.jtak listings) and create a new one called Labels.
    These will be filled with silhouette images.
    */

    /*Populate fluoroscopic study directories*/
    vector<string> fluoro_study_dirs;
    {
        /*Get List of Patient File Directories*/
        vector<string> patient_dirs;
        read_directory_for_directories(home_dir, patient_dirs);

        /*For Each Patient Directory Get List of Sessions*/
        for (int patient_ind = 0; patient_ind < patient_dirs.size();
             patient_ind++) {
            vector<string> session_dirs;
            read_directory_for_directories(patient_dirs[patient_ind],
                                           session_dirs);

            /*For Each Patient Session Directory Get List of Movements*/
            for (int session_ind = 0; session_ind < session_dirs.size();
                 session_ind++) {
                vector<string> movement_dirs;
                read_directory_for_directories(session_dirs[session_ind],
                                               movement_dirs);

                /*Add these directories to fluoro_dirs*/
                for (int movement_ind = 0; movement_ind < movement_dirs.size();
                     movement_ind++)
                    fluoro_study_dirs.push_back(movement_dirs[movement_ind]);
            }
        }
    }

    /*Make List of Each Individual Study*/
    vector<Study> studies;

    for (int i = 0; i < fluoro_study_dirs.size(); i++) {
        std::cout << "Loading study " << i + 1 << " of "
                  << fluoro_study_dirs.size() << std::endl;
        Study temp_study = Study(fluoro_study_dirs[i], GENERATE_KP_DATA);
        if (temp_study.passed_check_)
            studies.push_back(temp_study);
        else {
            cout << "\nError with study formatting!";
            getchar();
            return 0;
        }
    }
    std::cout << "\nAll studies loaded.\n";

    /*Vector of ImageInfo Classes (Used later to shuffle and write to grid
     * whilst maintaing an in order data log*/
    vector<ImageInfo> info_for_imgs;

    /*Create Silhouette Images for Each Image (placed in same location as
     * original with same name + _label_%MODEL TYPE% at end*/
    for (int study_ind = 0; study_ind < studies.size(); study_ind++) {
        /*Progress*/
        std::cout << "\rCreating silhouette implant labels for study "
                  << study_ind + 1 << " of " << studies.size();

        /*Read in Calibration File*/
        CameraCalibration calibration_file;
        QFile inputFile(
            QString::fromStdString(studies[study_ind].calibration_));
        if (inputFile.open(QIODevice::ReadOnly)) {
            QTextStream in(&inputFile);
            QStringList InputList = in.readAll().split(QRegExp("[\r\n]|,|\t| "),
                                                       Qt::SkipEmptyParts);

            /*Valid Code for Monoplane*/
            if (InputList[0] == "JT_INTCALIB" ||
                InputList[0] == "JTA_INTCALIB") {
                /*Error Check*/
                if (InputList[4].toDouble() == 0) {
                    std::cout
                        << "\nPixel size (the last number in the calibration "
                           "file) "
                           "is specified as 0! This is impossible. Error in: "
                        << studies[study_ind].study_dir_;
                    getchar();
                    return 0;
                }

                /*Initialize Calibration*/
                calibration_file = CameraCalibration(
                    InputList[1].toDouble(),
                    -1 * InputList[2]
                             .toDouble(),  // Negative For Offsets to make
                                           // consistent with JointTrack
                    -1 * InputList[3].toDouble(), InputList[4].toDouble());
            }
            /*Invalid Code*/
            else {
                std::cout << "\nError with loading calibration file for: "
                          << studies[study_ind].study_dir_;
                getchar();
                return 0;
            }
            inputFile.close();
        }

        /*Vector of Vector of Poses for Each Model*/
        vector<vector<Pose>> model_poses_list;

        gpu_cost_function::GPUMetrics *gpumet =
            new GPUMetrics();  // Uncomment as part of edge detection (if you
                               // want)
        /*For each model type in study*/
        for (int model_types_indx = 0;
             model_types_indx < studies[study_ind].stl_types_.size();
             model_types_indx++) {
            /*STL Information*/
            vector<vector<float>> triangle_information;
            readAnySTL(QString::fromStdString(
                           studies[study_ind].stl_files_[model_types_indx]),
                       triangle_information);

            /*GPU Models for the current Model*/
            GPUModel *gpu_mod = new GPUModel(
                studies[study_ind].stl_basenames_[model_types_indx], true,
                studies[study_ind].width_, studies[study_ind].height_,
                GPU_DEVICE, false, &(triangle_information[0])[0],
                &(triangle_information[1])[0],
                triangle_information[0].size() / 9,
                calibration_file);  // BACKFACE CULLING APPEARS TO BE GIVING
                                    // ERRORS
            if (!gpu_mod) {
                std::cout << " problem here" << std::endl;
            }
            /*Read In Poses For Kinematics for current Model*/
            vector<Pose> poses_mod;
            QFile inputFile_kin_mod(QString::fromStdString(
                studies[study_ind].kin_files_[model_types_indx]));
            if (inputFile_kin_mod.open(QIODevice::ReadOnly)) {
                QTextStream in(&inputFile_kin_mod);
                QStringList InputList =
                    in.readAll().split(QRegExp("[\r\n]"), Qt::SkipEmptyParts);
                if (InputList.size() == 0) {
                    cout << "\nInvalid "
                         << studies[study_ind].stl_types_[model_types_indx]
                         << " kinematics file. Error in: "
                         << studies[study_ind].study_dir_;
                    getchar();
                    return 0;
                }
                if (InputList[0] == "JTA_EULER_KINEMATICS" ||
                    InputList[0] == "JT_EULER_312") {
                    for (int i = 2; i < InputList.length() &&
                                    (i - 2) < studies[study_ind].images_.size();
                         i++) {
                        QStringList LineList = InputList[i].split(
                            QRegExp("[,]"), Qt::SkipEmptyParts);
                        if (LineList.size() >= 6) {
                            LineList[0].replace(" ", "");
                            if (LineList[0] != "NOT_OPTIMIZED") {
                                Pose temp_pose = Pose(LineList[0].toDouble(),
                                                      LineList[1].toDouble(),
                                                      LineList[2].toDouble(),
                                                      LineList[4].toDouble(),
                                                      LineList[5].toDouble(),
                                                      LineList[3].toDouble());
                                poses_mod.push_back(temp_pose);
                            }
                        }
                    }
                } else {
                    cout << "\nInvalid "
                         << studies[study_ind].stl_types_[model_types_indx]
                         << " kinematics file format. Error in : "
                         << studies[study_ind].study_dir_;
                    getchar();
                    return 0;
                }
                inputFile_kin_mod.close();
            }
            model_poses_list.push_back(poses_mod);

            /*For Each Image In a Given Study Create Model Labels*/
            for (int img_ind = 0; img_ind < studies[study_ind].images_.size();
                 img_ind++) {
                /*Current Model*/
                if (!gpu_mod->RenderPrimaryCamera(poses_mod[img_ind])) {
                    std::cout
                        << "\nError. Error rendering "
                        << studies[study_ind].stl_types_[model_types_indx]
                        << " image for: " << studies[study_ind].study_dir_;
                    getchar();
                    return 0;
                } /* if (studies[study_ind].stl_types_[model_types_indx] ==
        "sca") {
                gpumet->EdgeDetectRenderedImplantModel(gpu_mod->GetPrimaryCameraRenderedImage());
        // Uncomment as part of edge detection (if you want)
                gpumet->DilateEdgeDetectedImage(gpu_mod->GetPrimaryCameraRenderedImage(),
        2);
        }*/
                string labels_dir;
                if (OS_WINDOWS) {
                    labels_dir =
                        studies[study_ind].study_dir_ + "\\Labels\\" +
                        studies[study_ind].stl_types_[model_types_indx] +
                        "\\";  // Windows
                } else {
                    labels_dir =
                        studies[study_ind].study_dir_ + "/Labels/" +
                        studies[study_ind].stl_types_[model_types_indx] +
                        "/";  // Linux
                }

                if (!gpu_mod->WritePrimaryCameraRenderedImage(
                        labels_dir +
                        QFileInfo(QString::fromStdString(
                                      studies[study_ind].images_[img_ind]))
                            .baseName()
                            .toStdString() +
                        "_label_" +
                        studies[study_ind].stl_types_[model_types_indx] +
                        ".tif")) {
                    std::cout
                        << "\nError writing rendered "
                        << studies[study_ind].stl_types_[model_types_indx]
                        << " image for: " << studies[study_ind].study_dir_;
                    getchar();
                    return 0;
                }
            }

            /*Delete GPU Models*/
            delete gpu_mod;
        }

        /*After Generating all labels for each image, scroll through models in
        the study, add the location of that model's label as well as the model
        pose, then create an ImageInfo Class*/
        for (int img_ind = 0; img_ind < studies[study_ind].images_.size();
             img_ind++) {
            vector<string> label_img_paths_list;
            vector<gpu_cost_function::Pose> pose_img_models_;
            vector<string> mod_types_in_img;

            /*Vector of Vector of Normalized KPs (empty vector stored if no .kp
             * file for model or not generating KP data) for each model*/
            vector<vector<basic_la::XYPoint>> norm_KP_points_list;

            /*Scroll Through the Models in The Image*/
            for (int model_types_indx = 0;
                 model_types_indx < studies[study_ind].stl_types_.size();
                 model_types_indx++) {
                string labels_dir;
                if (OS_WINDOWS) {
                    labels_dir =
                        studies[study_ind].study_dir_ + "\\Labels\\" +
                        studies[study_ind].stl_types_[model_types_indx] +
                        "\\";  // Windows
                } else {
                    labels_dir =
                        studies[study_ind].study_dir_ + "/Labels/" +
                        studies[study_ind].stl_types_[model_types_indx] +
                        "/";  // Linux
                }
                label_img_paths_list.push_back(
                    labels_dir +
                    QFileInfo(QString::fromStdString(
                                  studies[study_ind].images_[img_ind]))
                        .baseName()
                        .toStdString() +
                    "_label_" +
                    studies[study_ind].stl_types_[model_types_indx] + ".tif");
                pose_img_models_.push_back(
                    model_poses_list[model_types_indx][img_ind]);
                mod_types_in_img.push_back(
                    studies[study_ind].stl_types_[model_types_indx]);

                /*If Using KP for Model, Read In Key Points*/
                vector<basic_la::XYPoint>
                    projected_normalized_KP_;  // KP that have been projected to
                                               // the image plane and are
                                               // expressed in normalized pixel
                                               // coordinates where 0,0 is the
                                               // bottom left and 1,1 is the top
                                               // right
                if (studies[study_ind]
                        .stl_basenames_have_kp_[model_types_indx]) {
                    /*Read in 3-tuple(s) from .kp file*/
                    vector<basic_la::XYZPoint> orig_KP;
                    readKP(QString::fromStdString(
                               studies[study_ind].study_dir_ + "/" +
                               studies[study_ind].stl_types_[model_types_indx] +
                               ".kp"),
                           orig_KP);
                    /*Load Image to Draw KP(s) On*/
                    cv::Mat img_kp =
                        cv::imread(studies[study_ind].images_[img_ind]);
                    /* Rotate, translate, project, normalize and then store in
                     * projected_normalized_KP_ each point from orig_KP*/
                    for (unsigned int kp_idx = 0; kp_idx < orig_KP.size();
                         kp_idx++) {
                        gpu_cost_function::Pose kp_transform_pose =
                            model_poses_list[model_types_indx][img_ind];
                        basic_la::RotationMatrixZXY rot_mat(
                            kp_transform_pose.z_angle_,
                            kp_transform_pose.x_angle_,
                            kp_transform_pose.y_angle_);
                        basic_la::XYZPoint rotated_point =
                            rot_mat.RotatePoint(orig_KP[kp_idx]);
                        basic_la::XYZPoint translated_point =
                            basic_la::XYZPoint(
                                rotated_point.X_ +
                                    kp_transform_pose.x_location_,
                                rotated_point.Y_ +
                                    kp_transform_pose.y_location_,
                                rotated_point.Z_ +
                                    kp_transform_pose.z_location_);
                        double dist_over_pix_pitch =
                            -1.0 * calibration_file.principal_distance_ /
                            calibration_file.pixel_pitch_;
                        double pix_conversion_x =
                            (double)studies[study_ind].width_ / 2.0f -
                            calibration_file.principal_x_ /
                                calibration_file.pixel_pitch_;
                        double pix_conversion_y =
                            (double)studies[study_ind].height_ / 2.0f -
                            calibration_file.principal_y_ /
                                calibration_file.pixel_pitch_;
                        basic_la::XYPoint not_normalized(
                            (translated_point.X_ / translated_point.Z_) *
                                    dist_over_pix_pitch +
                                pix_conversion_x,
                            (translated_point.Y_ / translated_point.Z_) *
                                    dist_over_pix_pitch +
                                pix_conversion_y);
                        /*Draw Circle over  KP in Image and Label Circle*/
                        cv::circle(
                            img_kp,
                            cv::Point((int)floor(not_normalized.X_),
                                      (int)floor(studies[study_ind].height_ -
                                                 not_normalized.Y_)),
                            5, cv::Scalar(255, 255, 0));  // CIRCLE SCALE IS 5
                        cv::putText(
                            img_kp, QString::number(kp_idx).toStdString(),
                            cv::Point((int)floor(not_normalized.X_),
                                      (int)floor(studies[study_ind].height_ -
                                                 not_normalized.Y_)),
                            0, .5, cv::Scalar(0, 255, 0));  // FONT SCALE is 0.5
                        /*Save Normalized Points*/
                        projected_normalized_KP_.push_back(basic_la::XYPoint(
                            not_normalized.X_ /
                                (double)studies[study_ind].width_,
                            not_normalized.Y_ /
                                (double)studies[study_ind].height_));
                    }
                    /*Save KP Labeled Picture*/
                    cv::imwrite(
                        studies[study_ind].study_dir_ + "\\Key_Points\\" +
                            studies[study_ind].stl_types_[model_types_indx] +
                            "\\" +
                            QFileInfo(QString::fromStdString(
                                          studies[study_ind].images_[img_ind]))
                                .baseName()
                                .toStdString() +
                            "_KPlabel_" +
                            studies[study_ind].stl_types_[model_types_indx] +
                            ".tif",
                        img_kp);
                }
                norm_KP_points_list.push_back(projected_normalized_KP_);
            }
            info_for_imgs.push_back(ImageInfo(
                studies[study_ind], studies[study_ind].images_[img_ind],
                mod_types_in_img, label_img_paths_list, pose_img_models_,
                norm_KP_points_list));
        }
    }
    std::cout << "\nAll silhouette labels created.\n";

    /*Shuffle Vector of ImageInfo Classes*/
    if (RAND_SHUFFLE_IMGS) {
        // old way of randomizing
        // unsigned seed = std::srand(unsigned(std::time(0)));
        unsigned seed =
            std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(info_for_imgs.begin(), info_for_imgs.end(),
                     std::default_random_engine(seed));
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* AT THIS POINT COULD SCREEN IMAGE INFO CLASS FOR CRITERIA LIKE MOVEMENT,
    MODEL TYPE ETC AND DELETE IMAGEINFO INSTANCES THAT DO NOT MEED THIS CRITERIA
    FROM INFO_FOR_IMGS. THEN WILL ONLY PRINT GRIDS OF IMAGEINFO INSTANCES THAT
    PASS REQUIREMENTS AND WILL PRINT CORRESPONDING MODEL TYPE LABELS FOR EVERY
    MODEL THAT EXISTS (WILL BE BLANK BLACK IF A PARTICULAR MODEL IS NOT USED IN
    AN IMAGEINFO CLASS)*/
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /*Make Grids for Original Images*/
    for (int grids_ind = 0; grids_ind < ceil((float)info_for_imgs.size() /
                                             (float)(GRID_WIDTH * GRID_HEIGHT));
         grids_ind++) {
        if (RAND_INVERT) INVERT_IMAGE = rand() % 2;
        Mat grid = 255 * INVERT_IMAGE +
                   (1 - 2 * INVERT_IMAGE) *
                       Mat(GRID_HEIGHT * IMAGE_HEIGHT, GRID_WIDTH * IMAGE_WIDTH,
                           CV_8UC1);  // 255 - img if inverting
        grid.setTo(cv::Scalar::all(0));

        /*Progress*/
        cout << "\rCreating original image grid " << grids_ind + 1 << " of "
             << ceil((float)info_for_imgs.size() /
                     (float)(GRID_WIDTH * GRID_HEIGHT));
        string new_image_dir;
        /*Within this loop, load an image, pad and then resize it, and paste in
         * the appropriate spot (Direction is top left to bottom right)*/
        for (int image_ind = grids_ind * GRID_WIDTH * GRID_HEIGHT;
             image_ind < info_for_imgs.size() &&
             image_ind < (grids_ind + 1) * GRID_WIDTH * GRID_HEIGHT;
             image_ind++) {
            /*Read In and Pad*/
            Mat img = 255 * INVERT_IMAGE +
                      (1 - 2 * INVERT_IMAGE) *
                          imread(info_for_imgs[image_ind].image_path_,
                                 CV_8UC1);  // PUT 255 - this to invert
            Mat padded;
            // string image_path = info_for_imgs[image_ind].image_path_;
            // int image_path_length =
            // info_for_imgs[image_ind].image_path_.length(); string image_name
            // = image_path.substr(image_path.find_last_of("/"),
            // image_path_length-4- image_path.find_last_of("/")); string
            // study_dir = image_path.substr(0, image_path.find_last_of("/"));
            if (OS_WINDOWS) {
                new_image_dir =
                    info_for_imgs[image_ind].study_.study_dir_ + "\\1024\\";
            }

            else {
                new_image_dir =
                    info_for_imgs[image_ind].study_.study_dir_ + "/1024/";
            }

            if (!filesystem::is_directory(new_image_dir) ||
                !filesystem::exists(new_image_dir)) {
                filesystem::create_directory(new_image_dir);
            }

            int borderType = BORDER_CONSTANT;
            int top = 0;
            int bottom = 0;
            int right = 0;
            int left = 0;

            if (!USE_PADDING) {
                padded.create(img.rows, img.cols, img.type());
            } else if (img.cols > img.rows) {
                padded.create(img.cols, img.cols, img.type());
                top = ceil((img.cols - img.rows) / 2);
                bottom = floor((img.cols - img.rows) / 2);
            } else {
                padded.create(img.rows, img.rows, img.type());
                left = ceil((img.rows - img.cols) / 2);
                right = floor((img.rows - img.cols) / 2);
            }

            padded.setTo(cv::Scalar::all(0));
            img.copyTo(padded(Rect(0, 0, img.cols, img.rows)));

            copyMakeBorder(img, img, top, bottom, left, right, borderType, 0);
            resize(img, img, Size(IMAGE_WIDTH, IMAGE_HEIGHT));

            /*Resize*/
            resize(padded, padded, Size(IMAGE_WIDTH, IMAGE_HEIGHT));

            cv::imwrite(new_image_dir + to_string(image_ind) + ".tif", img);

            /*Paste Appropriately to Grid*/
            // padded.copyTo(grid(Rect((image_ind % GRID_WIDTH)*IMAGE_WIDTH,
            // ((image_ind - grids_ind*GRID_WIDTH*GRID_HEIGHT) /
            // GRID_WIDTH)*IMAGE_HEIGHT, padded.cols, padded.rows)));
            // padded.copyTo(grid(Rect(0, 0, padded.cols, padded.rows)));
            img.copyTo(grid(Rect(0, 0, padded.cols, padded.rows)));
        }

        /*Write grid*/
        ostringstream out;
        out << std::internal << std::setfill('0') << std::setw(12) << grids_ind;
        cv::imwrite(
            write_grids_dir + "/grid_" + study_name + "_" + out.str() + ".tif",
            grid);
    }
    cout << "\nAll original image grids created.\n";

    /*Get List of All the Unique Model Types that show up in the ImageInfo
     * instances*/
    vector<string> unique_mod_types_imginfo_list;
    for (int img_info_indx = 0; img_info_indx < info_for_imgs.size();
         img_info_indx++) {
        for (int ii_mt_indx = 0;
             ii_mt_indx < info_for_imgs[img_info_indx].model_types_.size();
             ii_mt_indx++) {
            bool unique_mt = true;
            for (int unq_ind = 0;
                 unq_ind < unique_mod_types_imginfo_list.size(); unq_ind++) {
                if (unique_mod_types_imginfo_list[unq_ind] ==
                    info_for_imgs[img_info_indx].model_types_[ii_mt_indx]) {
                    unique_mt = false;
                    break;
                }
            }
            if (unique_mt) {
                unique_mod_types_imginfo_list.push_back(
                    info_for_imgs[img_info_indx].model_types_[ii_mt_indx]);
            }
        }
    }
    cout << "Unique model types found:\n";
    for (int unq_indx_mt = 0;
         unq_indx_mt < unique_mod_types_imginfo_list.size(); unq_indx_mt++) {
        cout << "\t" << unq_indx_mt + 1 << ". \""
             << unique_mod_types_imginfo_list[unq_indx_mt] << "\"" << endl;
    }

    /*Get Number of KP  Per Model Type Over All Images And Check that If a Model
    Type has "n" > 0 KP then if it ever has > 0 KP , it has exactly "n".*/
    std::vector<int> total_kp_points_by_model;
    std::vector<int>
        number_of_kp_points_in_model_with_kp_file;  // Should be consistent
    for (int unq_indx_mt = 0;
         unq_indx_mt < unique_mod_types_imginfo_list.size(); unq_indx_mt++) {
        total_kp_points_by_model.push_back(0);
        number_of_kp_points_in_model_with_kp_file.push_back(0);
    }
    for (int img_info_indx = 0; img_info_indx < info_for_imgs.size();
         img_info_indx++) {
        for (int ii_mt_indx = 0;
             ii_mt_indx < info_for_imgs[img_info_indx].model_types_.size();
             ii_mt_indx++) {
            for (int unq_indx_mt = 0;
                 unq_indx_mt < unique_mod_types_imginfo_list.size();
                 unq_indx_mt++) {
                if (unique_mod_types_imginfo_list[unq_indx_mt] ==
                    info_for_imgs[img_info_indx].model_types_[ii_mt_indx]) {
                    int temp_kplist_size = info_for_imgs[img_info_indx]
                                               .norm_KP_points_list_[ii_mt_indx]
                                               .size();
                    total_kp_points_by_model[unq_indx_mt] += (temp_kplist_size);
                    if (temp_kplist_size > 0) {
                        if (number_of_kp_points_in_model_with_kp_file
                                [unq_indx_mt] == 0) {
                            number_of_kp_points_in_model_with_kp_file
                                [unq_indx_mt] = temp_kplist_size;
                        } else {
                            if (number_of_kp_points_in_model_with_kp_file
                                    [unq_indx_mt] != temp_kplist_size) {
                                std::cout << "\nError! Number of key points is "
                                             "not consistent "
                                             "throughout model type when KP "
                                             "points exist!\nError at: "
                                          << info_for_imgs[img_info_indx]
                                                 .study_.study_dir_;
                                std::getchar();
                                return 0;
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
    /*Print Out Total KP and KP Size When Non-Zero*/
    cout << "Key point totals for each unique model type:\n";
    for (int unq_indx_mt = 0;
         unq_indx_mt < unique_mod_types_imginfo_list.size(); unq_indx_mt++) {
        cout << "\t" << unq_indx_mt + 1 << ". \""
             << unique_mod_types_imginfo_list[unq_indx_mt]
             << "\": " << total_kp_points_by_model[unq_indx_mt]
             << " total key points with "
             << number_of_kp_points_in_model_with_kp_file[unq_indx_mt]
             << " points when a .kp file exists for the model." << endl;
    }

    /*Clear Output Logs*/
    vector<string> info_paths;  // Vector of info paths for each unique model
    for (int unique_mod_type = 0;
         unique_mod_type < unique_mod_types_imginfo_list.size();
         unique_mod_type++) {
        std::ofstream ofs;
        string info_path = write_grids_dir + "/info_" +
                           unique_mod_types_imginfo_list[unique_mod_type] +
                           ".txt";
        ofs.open(info_path, std::ofstream::out | std::ofstream::trunc);
        ofs.close();
        info_paths.push_back(info_path);
    }

    /*Clear KP Output Logs*/
    vector<string> kplabel_paths;  // Vector of normalized key point info paths
                                   // for each unique model with key points
    for (int unique_mod_type = 0;
         unique_mod_type < unique_mod_types_imginfo_list.size();
         unique_mod_type++) {
        if (total_kp_points_by_model[unique_mod_type] > 0) {
            std::ofstream ofs;
            string kp_label_path =
                write_grids_dir + "/" +
                unique_mod_types_imginfo_list[unique_mod_type] +
                "_KPlabels.txt";
            ofs.open(kp_label_path, std::ofstream::out | std::ofstream::trunc);
            ofs.close();
            kplabel_paths.push_back(kp_label_path);
        } else
            kplabel_paths.push_back("");
    }

    /*Keep Track of Min/Max for X and Y*/
    double min_x = 1, min_y = 1;
    double max_x = 0, max_y = 0;

    /*Write KP Output Logs*/
    for (int unique_mod_type = 0;
         unique_mod_type < unique_mod_types_imginfo_list.size();
         unique_mod_type++) {
        for (int image_ind = 0; image_ind < info_for_imgs.size(); image_ind++) {
            /*Check if Image Info Class Has Current Model Type*/
            bool image_contains_current_model_type = false;
            int mod_index_in_imageinfo = -1;
            for (int contain_indx = 0;
                 contain_indx < info_for_imgs[image_ind].model_types_.size();
                 contain_indx++) {
                if (info_for_imgs[image_ind].model_types_[contain_indx] ==
                    unique_mod_types_imginfo_list[unique_mod_type]) {
                    image_contains_current_model_type = true;
                    mod_index_in_imageinfo = contain_indx;
                    break;
                }
            }
            /*If image contains the model type and there exists at least one key
             * point for the model type, write it to file*/
            if (image_contains_current_model_type &&
                (info_for_imgs[image_ind]
                     .norm_KP_points_list_[mod_index_in_imageinfo]
                     .size() > 0)) {
                std::ofstream outfile;
                outfile.open(kplabel_paths[unique_mod_type],
                             std::ios_base::app);
                ostringstream out;
                out << std::internal << std::setfill('0') << std::setw(12)
                    << image_ind;
                outfile << "grid_" + study_name + "_" + out.str() + ".tif"
                        << endl;

                for (int num_kp_index = 0;
                     num_kp_index <
                     info_for_imgs[image_ind]
                         .norm_KP_points_list_[mod_index_in_imageinfo]
                         .size();
                     num_kp_index++) {
                    /*Update Min and Maxes*/
                    double x_val =
                        info_for_imgs[image_ind]
                            .norm_KP_points_list_[mod_index_in_imageinfo]
                                                 [num_kp_index]
                            .X_;
                    double y_val =
                        info_for_imgs[image_ind]
                            .norm_KP_points_list_[mod_index_in_imageinfo]
                                                 [num_kp_index]
                            .Y_;
                    if (x_val < min_x) min_x = x_val;
                    if (x_val > max_x) max_x = x_val;
                    if (y_val < min_y) min_y = y_val;
                    if (y_val > max_y) max_y = y_val;
                    /*Write*/
                    outfile << x_val << "," << y_val << endl;
                }
            }
        }
    }

    /*Display Max and Min For X and Y*/
    std::cout << "\n Min X: " << min_x;
    std::cout << "\n Max X: " << max_x;
    std::cout << "\n Min Y: " << min_y;
    std::cout << "\n Max Y: " << max_y << endl;

    /*Make Grids of Silhouette Labels for Each ImageInfo Class Instance/Unique
    Model Type Pair. If a Given Model Type is Not Associated with a Particular
    Instance, Use Blank Black Image for Label*/
    for (int unique_mod_type = 0;
         unique_mod_type < unique_mod_types_imginfo_list.size();
         unique_mod_type++) {
        /*Make Grids for %MODEL TYPE% Silhouette Label Images*/
        for (int grids_ind = 0;
             grids_ind < ceil((float)info_for_imgs.size() /
                              (float)(GRID_WIDTH * GRID_HEIGHT));
             grids_ind++) {
            Mat grid = Mat(GRID_HEIGHT * IMAGE_HEIGHT, GRID_WIDTH * IMAGE_WIDTH,
                           CV_8UC1);
            grid.setTo(cv::Scalar::all(0));

            /*Progress*/
            cout << "\rCreating " +
                        unique_mod_types_imginfo_list[unique_mod_type] +
                        " silhouette image label grid "
                 << grids_ind + 1 << " of "
                 << ceil((float)info_for_imgs.size() /
                         (float)(GRID_WIDTH * GRID_HEIGHT));

            /*Within this loop, load an image, pad and then resize it, and paste
             * in the appropriate spot (Direction is top left to bottom right)*/
            for (int image_ind = grids_ind * GRID_WIDTH * GRID_HEIGHT;
                 image_ind < info_for_imgs.size() &&
                 image_ind < (grids_ind + 1) * GRID_WIDTH * GRID_HEIGHT;
                 image_ind++) {
                /*Check if ImageInfo instance contains the current unique
                 * model*/
                bool image_contains_current_model_type = false;
                int mod_index_in_imageinfo = -1;
                for (int contain_indx = 0;
                     contain_indx <
                     info_for_imgs[image_ind].model_types_.size();
                     contain_indx++) {
                    if (info_for_imgs[image_ind].model_types_[contain_indx] ==
                        unique_mod_types_imginfo_list[unique_mod_type]) {
                        image_contains_current_model_type = true;
                        mod_index_in_imageinfo = contain_indx;
                        break;
                    }
                }
                /*Load if image contains the model type, else do nothing and
                 * will move to next grid*/
                if (image_contains_current_model_type) {
                    /*Read In and Pad*/
                    Mat img =
                        imread(info_for_imgs[image_ind]
                                   .label_img_paths_[mod_index_in_imageinfo],
                               CV_8UC1);
                    Mat padded;
                    int borderType = BORDER_CONSTANT;
                    int top = 0;
                    int bottom = 0;
                    int right = 0;
                    int left = 0;

                    if (!USE_PADDING) {
                        padded.create(img.rows, img.cols, img.type());

                    } else if (img.cols > img.rows) {
                        padded.create(img.cols, img.cols, img.type());
                        top = ceil((img.cols - img.rows) / 2);
                        bottom = floor((img.cols - img.rows) / 2);

                    }

                    else {
                        padded.create(img.rows, img.rows, img.type());
                        right = ceil((img.rows - img.cols) / 2);
                        left = floor((img.rows - img.cols) / 2);
                    }

                    padded.setTo(cv::Scalar::all(0));
                    img.copyTo(padded(Rect(0, 0, img.cols, img.rows)));
                    copyMakeBorder(img, img, top, bottom, left, right,
                                   borderType, 0);
                    resize(img, img, Size(IMAGE_WIDTH, IMAGE_HEIGHT));
                    /*Resize*/
                    resize(padded, padded, Size(IMAGE_WIDTH, IMAGE_HEIGHT));

                    /*Paste Appropriately to Grid*/
                    padded.copyTo(grid(Rect(
                        (image_ind % GRID_WIDTH) * IMAGE_WIDTH,
                        ((image_ind - grids_ind * GRID_WIDTH * GRID_HEIGHT) /
                         GRID_WIDTH) *
                            IMAGE_HEIGHT,
                        padded.cols, padded.rows)));

                    /*Write to Log*/
                    info_for_imgs[image_ind].AppendInformation(
                        info_paths[unique_mod_type], mod_index_in_imageinfo);

                    cv::imwrite(info_for_imgs[image_ind]
                                    .label_img_paths_[mod_index_in_imageinfo],
                                img);
                }
            }

            /*Write grid*/
            ostringstream out;
            out << std::internal << std::setfill('0') << std::setw(12)
                << grids_ind;
            cv::imwrite(write_grids_dir + "/" +
                            unique_mod_types_imginfo_list[unique_mod_type] +
                            "_label_grid_" + study_name + "_" + out.str() +
                            ".tif",
                        grid);
        }
        cout << "\nAll " + unique_mod_types_imginfo_list[unique_mod_type] +
                    " silhouette image label grids created.\n";
    }

    /*Finished*/
    std::cout << "\nFinished!";
    getchar();
    return 0;
}
