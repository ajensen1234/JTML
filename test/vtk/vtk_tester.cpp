#include <string.h>
#include <vtkAutoInit.h>
#include <vtkNew.h>

#include <QFile>
#include <QString>
#include <QTextStream>
#include <iostream>
#include <vector>

#include "core/calibration.h"
#include "core/frame.h"
#include "core/model.h"
#include "gpu/camera_calibration.h"
#include "gui/viewer.h"
// Calibration read_calibration(const QString& cal_path);

void print_hello() { std::cout << "hello" << std::endl; }
Calibration read_calibration(const QString& cal_path) {
    QFile inputfile(cal_path);
    if (inputfile.open(QIODevice::ReadOnly)) {
        QTextStream in(&inputfile);
        QStringList InputList = in.readAll().split(QRegExp("[\r\n]|,|\t| "),
                                                   QString::SkipEmptyParts);
        CameraCalibration cal(
            InputList[1].toDouble(), -1 * InputList[2].toDouble(),
            -1 * InputList[3].toDouble(), InputList[4].toDouble());
        return Calibration(cal);
    };
};

int main() {
    std::string home_dir = "C:/JTML/JTA-CMake/test/vtk/";
    vtkNew<vtkRenderer> my_renderer;
    auto actor = vtkSmartPointer<vtkActor>::New();
    print_hello();
    Viewer vw;
    vw.initialize_vtk_pointers();
    vw.initialize_vtk_mappers();
    vw.initialize_vtk_renderers();
    QString cal_path("./test_case/calibration.txt");
    std::string img_path = home_dir + "/test_case/HL_V1_K1_0001.tif";
    QStringList fem_path_extension("./test_case/KR_left_8_fem.stl");
    QStringList fem_loaded_name(
        QFileInfo(QString::fromStdString(fem_path_extension[0].toStdString()))
            .baseName());
    Calibration cal = read_calibration(cal_path);
    vw.setup_camera_calibration(cal);
    vw.place_image_actors_according_to_calibration(cal, 1024, 1024);
    std::vector<Frame> frame = {Frame(img_path, 3, 0, 150, 6)};
    vw.set_loaded_frames(frame);
    vw.update_display_background_to_original_image(0, true);
    vw.set_model_orientation_at_index(0, 0, 0, 0);
    vw.set_model_position_at_index(0, 0, 0, -800);
    vw.load_models(fem_path_extension, fem_loaded_name);

    vw.set_render_window_and_display();

    return EXIT_SUCCESS;
};
