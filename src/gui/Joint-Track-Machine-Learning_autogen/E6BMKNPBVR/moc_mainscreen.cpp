/****************************************************************************
** Meta object code from reading C++ file 'mainscreen.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../../include/gui/mainscreen.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainscreen.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_MainScreen_t {
    QByteArrayData data[61];
    char stringdata0[1909];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MainScreen_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MainScreen_t qt_meta_stringdata_MainScreen = {
    {
QT_MOC_LITERAL(0, 0, 10), // "MainScreen"
QT_MOC_LITERAL(1, 11, 17), // "UpdateDisplayText"
QT_MOC_LITERAL(2, 29, 0), // ""
QT_MOC_LITERAL(3, 30, 13), // "StopOptimizer"
QT_MOC_LITERAL(4, 44, 19), // "UpdateTimeRemaining"
QT_MOC_LITERAL(5, 64, 21), // "optimizer_launch_slot"
QT_MOC_LITERAL(6, 86, 34), // "on_load_calibration_button_cl..."
QT_MOC_LITERAL(7, 121, 28), // "on_load_image_button_clicked"
QT_MOC_LITERAL(8, 150, 28), // "on_load_model_button_clicked"
QT_MOC_LITERAL(9, 179, 32), // "on_camera_A_radio_button_clicked"
QT_MOC_LITERAL(10, 212, 32), // "on_camera_B_radio_button_clicked"
QT_MOC_LITERAL(11, 245, 41), // "on_image_list_widget_itemSele..."
QT_MOC_LITERAL(12, 287, 41), // "on_model_list_widget_itemSele..."
QT_MOC_LITERAL(13, 329, 35), // "on_edges_image_radio_button_c..."
QT_MOC_LITERAL(14, 365, 38), // "on_dilation_image_radio_butto..."
QT_MOC_LITERAL(15, 404, 38), // "on_original_model_radio_butto..."
QT_MOC_LITERAL(16, 443, 35), // "on_solid_model_radio_button_c..."
QT_MOC_LITERAL(17, 479, 41), // "on_transparent_model_radio_bu..."
QT_MOC_LITERAL(18, 521, 39), // "on_wireframe_model_radio_butt..."
QT_MOC_LITERAL(19, 561, 33), // "on_aperture_spin_box_valueCha..."
QT_MOC_LITERAL(20, 595, 36), // "on_low_threshold_slider_value..."
QT_MOC_LITERAL(21, 632, 37), // "on_high_threshold_slider_valu..."
QT_MOC_LITERAL(22, 670, 32), // "on_apply_all_edge_button_clicked"
QT_MOC_LITERAL(23, 703, 28), // "on_reset_edge_button_clicked"
QT_MOC_LITERAL(24, 732, 28), // "on_actionSave_Pose_triggered"
QT_MOC_LITERAL(25, 761, 34), // "on_actionSave_Kinematics_trig..."
QT_MOC_LITERAL(26, 796, 28), // "on_actionLoad_Pose_triggered"
QT_MOC_LITERAL(27, 825, 34), // "on_actionLoad_Kinematics_trig..."
QT_MOC_LITERAL(28, 860, 40), // "on_actionAbout_JointTrack_Aut..."
QT_MOC_LITERAL(29, 901, 27), // "on_actionControls_triggered"
QT_MOC_LITERAL(30, 929, 33), // "on_actionStop_Optimizer_trigg..."
QT_MOC_LITERAL(31, 963, 37), // "on_actionOptimizer_Settings_t..."
QT_MOC_LITERAL(32, 1001, 31), // "on_actionDRR_Settings_triggered"
QT_MOC_LITERAL(33, 1033, 29), // "on_actionReset_View_triggered"
QT_MOC_LITERAL(34, 1063, 34), // "on_actionReset_Normal_Up_trig..."
QT_MOC_LITERAL(35, 1098, 41), // "on_actionModel_Interaction_Mo..."
QT_MOC_LITERAL(36, 1140, 42), // "on_actionCamera_Interaction_M..."
QT_MOC_LITERAL(37, 1183, 32), // "on_actionSegment_FemHR_triggered"
QT_MOC_LITERAL(38, 1216, 32), // "on_actionSegment_TibHR_triggered"
QT_MOC_LITERAL(39, 1249, 48), // "on_actionReset_Remove_All_Seg..."
QT_MOC_LITERAL(40, 1298, 45), // "on_actionEstimate_Femoral_Imp..."
QT_MOC_LITERAL(41, 1344, 44), // "on_actionEstimate_Tibial_Impl..."
QT_MOC_LITERAL(42, 1389, 36), // "on_actionNFD_Pose_Estimate_tr..."
QT_MOC_LITERAL(43, 1426, 33), // "on_actionCopy_Next_Pose_trigg..."
QT_MOC_LITERAL(44, 1460, 37), // "on_actionCopy_Previous_Pose_t..."
QT_MOC_LITERAL(45, 1498, 44), // "on_actionAmbiguous_Pose_Proce..."
QT_MOC_LITERAL(46, 1543, 26), // "on_optimize_button_clicked"
QT_MOC_LITERAL(47, 1570, 30), // "on_optimize_all_button_clicked"
QT_MOC_LITERAL(48, 1601, 31), // "on_optimize_each_button_clicked"
QT_MOC_LITERAL(49, 1633, 31), // "on_optimize_from_button_clicked"
QT_MOC_LITERAL(50, 1665, 36), // "on_actionOptimize_Backward_tr..."
QT_MOC_LITERAL(51, 1702, 15), // "onUpdateOptimum"
QT_MOC_LITERAL(52, 1718, 16), // "onOptimizedFrame"
QT_MOC_LITERAL(53, 1735, 16), // "onOptimizerError"
QT_MOC_LITERAL(54, 1752, 13), // "error_message"
QT_MOC_LITERAL(55, 1766, 15), // "onUpdateDisplay"
QT_MOC_LITERAL(56, 1782, 26), // "onUpdateDilationBackground"
QT_MOC_LITERAL(57, 1809, 27), // "updateOrientationSymTrap_MS"
QT_MOC_LITERAL(58, 1837, 14), // "onSaveSettings"
QT_MOC_LITERAL(59, 1852, 17), // "OptimizerSettings"
QT_MOC_LITERAL(60, 1870, 38) // "jta_cost_function::CostFuncti..."

    },
    "MainScreen\0UpdateDisplayText\0\0"
    "StopOptimizer\0UpdateTimeRemaining\0"
    "optimizer_launch_slot\0"
    "on_load_calibration_button_clicked\0"
    "on_load_image_button_clicked\0"
    "on_load_model_button_clicked\0"
    "on_camera_A_radio_button_clicked\0"
    "on_camera_B_radio_button_clicked\0"
    "on_image_list_widget_itemSelectionChanged\0"
    "on_model_list_widget_itemSelectionChanged\0"
    "on_edges_image_radio_button_clicked\0"
    "on_dilation_image_radio_button_clicked\0"
    "on_original_model_radio_button_clicked\0"
    "on_solid_model_radio_button_clicked\0"
    "on_transparent_model_radio_button_clicked\0"
    "on_wireframe_model_radio_button_clicked\0"
    "on_aperture_spin_box_valueChanged\0"
    "on_low_threshold_slider_valueChanged\0"
    "on_high_threshold_slider_valueChanged\0"
    "on_apply_all_edge_button_clicked\0"
    "on_reset_edge_button_clicked\0"
    "on_actionSave_Pose_triggered\0"
    "on_actionSave_Kinematics_triggered\0"
    "on_actionLoad_Pose_triggered\0"
    "on_actionLoad_Kinematics_triggered\0"
    "on_actionAbout_JointTrack_Auto_triggered\0"
    "on_actionControls_triggered\0"
    "on_actionStop_Optimizer_triggered\0"
    "on_actionOptimizer_Settings_triggered\0"
    "on_actionDRR_Settings_triggered\0"
    "on_actionReset_View_triggered\0"
    "on_actionReset_Normal_Up_triggered\0"
    "on_actionModel_Interaction_Mode_triggered\0"
    "on_actionCamera_Interaction_Mode_triggered\0"
    "on_actionSegment_FemHR_triggered\0"
    "on_actionSegment_TibHR_triggered\0"
    "on_actionReset_Remove_All_Segmentation_triggered\0"
    "on_actionEstimate_Femoral_Implant_s_triggered\0"
    "on_actionEstimate_Tibial_Implant_s_triggered\0"
    "on_actionNFD_Pose_Estimate_triggered\0"
    "on_actionCopy_Next_Pose_triggered\0"
    "on_actionCopy_Previous_Pose_triggered\0"
    "on_actionAmbiguous_Pose_Processing_triggered\0"
    "on_optimize_button_clicked\0"
    "on_optimize_all_button_clicked\0"
    "on_optimize_each_button_clicked\0"
    "on_optimize_from_button_clicked\0"
    "on_actionOptimize_Backward_triggered\0"
    "onUpdateOptimum\0onOptimizedFrame\0"
    "onOptimizerError\0error_message\0"
    "onUpdateDisplay\0onUpdateDilationBackground\0"
    "updateOrientationSymTrap_MS\0onSaveSettings\0"
    "OptimizerSettings\0"
    "jta_cost_function::CostFunctionManager"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MainScreen[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      56,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,  294,    2, 0x06 /* Public */,
       3,    0,  297,    2, 0x06 /* Public */,
       4,    1,  298,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       5,    0,  301,    2, 0x0a /* Public */,
       6,    0,  302,    2, 0x0a /* Public */,
       7,    0,  303,    2, 0x0a /* Public */,
       8,    0,  304,    2, 0x0a /* Public */,
       9,    0,  305,    2, 0x0a /* Public */,
      10,    0,  306,    2, 0x0a /* Public */,
      11,    0,  307,    2, 0x0a /* Public */,
      12,    0,  308,    2, 0x0a /* Public */,
      13,    0,  309,    2, 0x0a /* Public */,
      14,    0,  310,    2, 0x0a /* Public */,
      15,    0,  311,    2, 0x0a /* Public */,
      16,    0,  312,    2, 0x0a /* Public */,
      17,    0,  313,    2, 0x0a /* Public */,
      18,    0,  314,    2, 0x0a /* Public */,
      19,    0,  315,    2, 0x0a /* Public */,
      20,    0,  316,    2, 0x0a /* Public */,
      21,    0,  317,    2, 0x0a /* Public */,
      22,    0,  318,    2, 0x0a /* Public */,
      23,    0,  319,    2, 0x0a /* Public */,
      24,    0,  320,    2, 0x0a /* Public */,
      25,    0,  321,    2, 0x0a /* Public */,
      26,    0,  322,    2, 0x0a /* Public */,
      27,    0,  323,    2, 0x0a /* Public */,
      28,    0,  324,    2, 0x0a /* Public */,
      29,    0,  325,    2, 0x0a /* Public */,
      30,    0,  326,    2, 0x0a /* Public */,
      31,    0,  327,    2, 0x0a /* Public */,
      32,    0,  328,    2, 0x0a /* Public */,
      33,    0,  329,    2, 0x0a /* Public */,
      34,    0,  330,    2, 0x0a /* Public */,
      35,    0,  331,    2, 0x0a /* Public */,
      36,    0,  332,    2, 0x0a /* Public */,
      37,    0,  333,    2, 0x0a /* Public */,
      38,    0,  334,    2, 0x0a /* Public */,
      39,    0,  335,    2, 0x0a /* Public */,
      40,    0,  336,    2, 0x0a /* Public */,
      41,    0,  337,    2, 0x0a /* Public */,
      42,    0,  338,    2, 0x0a /* Public */,
      43,    0,  339,    2, 0x0a /* Public */,
      44,    0,  340,    2, 0x0a /* Public */,
      45,    0,  341,    2, 0x0a /* Public */,
      46,    0,  342,    2, 0x0a /* Public */,
      47,    0,  343,    2, 0x0a /* Public */,
      48,    0,  344,    2, 0x0a /* Public */,
      49,    0,  345,    2, 0x0a /* Public */,
      50,    0,  346,    2, 0x0a /* Public */,
      51,    7,  347,    2, 0x0a /* Public */,
      52,   10,  362,    2, 0x0a /* Public */,
      53,    1,  383,    2, 0x0a /* Public */,
      55,    4,  386,    2, 0x0a /* Public */,
      56,    0,  395,    2, 0x0a /* Public */,
      57,    6,  396,    2, 0x0a /* Public */,
      58,    4,  409,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    2,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::UInt,    2,    2,    2,    2,    2,    2,    2,
    QMetaType::Void, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Bool, QMetaType::UInt, QMetaType::Bool, QMetaType::QString,    2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
    QMetaType::Void, QMetaType::QString,   54,
    QMetaType::Void, QMetaType::Double, QMetaType::Int, QMetaType::Double, QMetaType::UInt,    2,    2,    2,    2,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double,    2,    2,    2,    2,    2,    2,
    QMetaType::Void, 0x80000000 | 59, 0x80000000 | 60, 0x80000000 | 60, 0x80000000 | 60,    2,    2,    2,    2,

       0        // eod
};

void MainScreen::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MainScreen *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->UpdateDisplayText((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->StopOptimizer(); break;
        case 2: _t->UpdateTimeRemaining((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->optimizer_launch_slot(); break;
        case 4: _t->on_load_calibration_button_clicked(); break;
        case 5: _t->on_load_image_button_clicked(); break;
        case 6: _t->on_load_model_button_clicked(); break;
        case 7: _t->on_camera_A_radio_button_clicked(); break;
        case 8: _t->on_camera_B_radio_button_clicked(); break;
        case 9: _t->on_image_list_widget_itemSelectionChanged(); break;
        case 10: _t->on_model_list_widget_itemSelectionChanged(); break;
        case 11: _t->on_edges_image_radio_button_clicked(); break;
        case 12: _t->on_dilation_image_radio_button_clicked(); break;
        case 13: _t->on_original_model_radio_button_clicked(); break;
        case 14: _t->on_solid_model_radio_button_clicked(); break;
        case 15: _t->on_transparent_model_radio_button_clicked(); break;
        case 16: _t->on_wireframe_model_radio_button_clicked(); break;
        case 17: _t->on_aperture_spin_box_valueChanged(); break;
        case 18: _t->on_low_threshold_slider_valueChanged(); break;
        case 19: _t->on_high_threshold_slider_valueChanged(); break;
        case 20: _t->on_apply_all_edge_button_clicked(); break;
        case 21: _t->on_reset_edge_button_clicked(); break;
        case 22: _t->on_actionSave_Pose_triggered(); break;
        case 23: _t->on_actionSave_Kinematics_triggered(); break;
        case 24: _t->on_actionLoad_Pose_triggered(); break;
        case 25: _t->on_actionLoad_Kinematics_triggered(); break;
        case 26: _t->on_actionAbout_JointTrack_Auto_triggered(); break;
        case 27: _t->on_actionControls_triggered(); break;
        case 28: _t->on_actionStop_Optimizer_triggered(); break;
        case 29: _t->on_actionOptimizer_Settings_triggered(); break;
        case 30: _t->on_actionDRR_Settings_triggered(); break;
        case 31: _t->on_actionReset_View_triggered(); break;
        case 32: _t->on_actionReset_Normal_Up_triggered(); break;
        case 33: _t->on_actionModel_Interaction_Mode_triggered(); break;
        case 34: _t->on_actionCamera_Interaction_Mode_triggered(); break;
        case 35: _t->on_actionSegment_FemHR_triggered(); break;
        case 36: _t->on_actionSegment_TibHR_triggered(); break;
        case 37: _t->on_actionReset_Remove_All_Segmentation_triggered(); break;
        case 38: _t->on_actionEstimate_Femoral_Implant_s_triggered(); break;
        case 39: _t->on_actionEstimate_Tibial_Implant_s_triggered(); break;
        case 40: _t->on_actionNFD_Pose_Estimate_triggered(); break;
        case 41: _t->on_actionCopy_Next_Pose_triggered(); break;
        case 42: _t->on_actionCopy_Previous_Pose_triggered(); break;
        case 43: _t->on_actionAmbiguous_Pose_Processing_triggered(); break;
        case 44: _t->on_optimize_button_clicked(); break;
        case 45: _t->on_optimize_all_button_clicked(); break;
        case 46: _t->on_optimize_each_button_clicked(); break;
        case 47: _t->on_optimize_from_button_clicked(); break;
        case 48: _t->on_actionOptimize_Backward_triggered(); break;
        case 49: _t->onUpdateOptimum((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4])),(*reinterpret_cast< double(*)>(_a[5])),(*reinterpret_cast< double(*)>(_a[6])),(*reinterpret_cast< uint(*)>(_a[7]))); break;
        case 50: _t->onOptimizedFrame((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4])),(*reinterpret_cast< double(*)>(_a[5])),(*reinterpret_cast< double(*)>(_a[6])),(*reinterpret_cast< bool(*)>(_a[7])),(*reinterpret_cast< uint(*)>(_a[8])),(*reinterpret_cast< bool(*)>(_a[9])),(*reinterpret_cast< QString(*)>(_a[10]))); break;
        case 51: _t->onOptimizerError((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 52: _t->onUpdateDisplay((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3])),(*reinterpret_cast< uint(*)>(_a[4]))); break;
        case 53: _t->onUpdateDilationBackground(); break;
        case 54: _t->updateOrientationSymTrap_MS((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4])),(*reinterpret_cast< double(*)>(_a[5])),(*reinterpret_cast< double(*)>(_a[6]))); break;
        case 55: _t->onSaveSettings((*reinterpret_cast< OptimizerSettings(*)>(_a[1])),(*reinterpret_cast< jta_cost_function::CostFunctionManager(*)>(_a[2])),(*reinterpret_cast< jta_cost_function::CostFunctionManager(*)>(_a[3])),(*reinterpret_cast< jta_cost_function::CostFunctionManager(*)>(_a[4]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 55:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< OptimizerSettings >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (MainScreen::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&MainScreen::UpdateDisplayText)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (MainScreen::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&MainScreen::StopOptimizer)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (MainScreen::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&MainScreen::UpdateTimeRemaining)) {
                *result = 2;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject MainScreen::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_meta_stringdata_MainScreen.data,
    qt_meta_data_MainScreen,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *MainScreen::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainScreen::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MainScreen.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int MainScreen::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 56)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 56;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 56)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 56;
    }
    return _id;
}

// SIGNAL 0
void MainScreen::UpdateDisplayText(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void MainScreen::StopOptimizer()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void MainScreen::UpdateTimeRemaining(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
