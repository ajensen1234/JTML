/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#ifndef INTERACTOR_H
#define INTERACTOR_H

#include <qcursor.h>
#include <vtkActor2DCollection.h>
#include <vtkInteractorStyleTrackballActor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>
#include <vtkPicker.h>
#include <vtkProp.h>
#include <vtkPropPicker.h>
#include <vtkRendererCollection.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>

#include <utility>
#include <memory>

/*Forward Declarations*/
class MainScreen;
class Viewer;
struct Point6D;

// Calibration To Convert Pose
#include "core/calibration.h"
extern Calibration interactor_calibration;

// Speed of Movement
extern double speed;
extern bool information;
extern bool interactor_camera_B; // Are we in Camera B?
extern bool middleDown;          // Is CM button down?
extern bool leftDown;            // Is LM button down?
extern bool rightDown;           // Is RM button down
extern int rightDownY;               // Y Pixel when RM Clicked
extern double rightDownModelZ;       // Model's Z Translation when RM Clicked

class KeyPressInteractorStyle : public vtkInteractorStyleTrackballActor {
public:
    static KeyPressInteractorStyle* New();
    vtkTypeMacro(KeyPressInteractorStyle, vtkInteractorStyleTrackballActor);

    /*Pointer to Main Window*/
    MainScreen* ms_;
    std::shared_ptr<Viewer> viewer_ = nullptr;

    void initialize_MainScreen(MainScreen* ms);
    void initialize_viewer(std::shared_ptr<Viewer> viewer);

    // Picked Function
    bool ActivePick();

    // KeyPress Turns Off Other Char Hotkeys
    void OnChar() override;

    // Keypress Function
    void OnKeyPress() override;

    // Mouse Down Functions
    void OnLeftButtonDown() override;
    void OnRightButtonDown() override;
    void OnMiddleButtonDown() override;

    // Mouse Up Functions
    void OnLeftButtonUp() override;
    void OnRightButtonUp() override;
    void OnMiddleButtonUp() override;

    // Mouse Movement
    void OnMouseMove() override;
};

class CameraInteractorStyle : public vtkInteractorStyleTrackballCamera {
public:
    static CameraInteractorStyle* New();
    vtkTypeMacro(CameraInteractorStyle, vtkInteractorStyleTrackballCamera);

    // KeyPress Turns Off Other Char Hotkeys
    void OnChar() override {}
};

#endif /* INTERACTOR_H */
