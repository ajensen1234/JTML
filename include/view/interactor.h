/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#ifndef INTERACTOR_H
#define INTERACTOR_H

#include <qcursor.h>
#include <vtkActor2DCollection.h>
#include <vtkInteractorStyleTrackballActor.h>
#include <vtkObjectFactory.h>
#include <vtkPicker.h>
#include <vtkProp.h>
#include <vtkPropPicker.h>
#include <vtkRendererCollection.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>

#include <utility>

/*Ref to QMainWindow*/
#include "view/mainscreen.h"
#include "view/viewer.h"

// Calibration To Convert Pose
#include "services/calibration.h"
Calibration interactor_calibration;

// Speed of Movement
double speed = 1;
bool information = true;
bool interactor_camera_B = false; // Are we in Camera B?
bool middleDown = false;          // Is CM button down?
bool leftDown = false;            // Is LM button down?
bool rightDown = false;           // Is RM button down
int rightDownY = 0;               // Y Pixel when RM Clicked
double rightDownModelZ = 0;       // Model's Z Translation when RM Clicked

class KeyPressInteractorStyle : public vtkInteractorStyleTrackballActor {
public:
    static KeyPressInteractorStyle* New();
    vtkTypeMacro(KeyPressInteractorStyle, vtkInteractorStyleTrackballActor);

    /*Pointer to Main Window*/
    MainScreen* ms_;
    std::shared_ptr<Viewer> viewer_ = nullptr;

    void initialize_MainScreen(MainScreen* ms) {
        ms_ = ms;
    }

    void initialize_viewer(std::shared_ptr<Viewer> viewer) {
        viewer_ = viewer;
    }

    // Picked Function
    bool ActivePick() {
        if (this->InteractionProp == NULL) {
            return false;
        }
        return true;
    }

    // KeyPress Turns Off Other Char Hotkeys
    void OnChar() override {
        vtkRenderWindowInteractor* rwi = this->Interactor;
        std::string key = rwi->GetKeySym();
        if (key == "Escape" || key == "escape" || key == "ESC" ||
            key == "Esc" || key == "esc") {
            ms_->VTKEscapeSignal();
        }
    }
    void update_info_text(vtkActor* actor) {
        vtkTextActor* text = viewer_ ? viewer_->get_actor_text() : nullptr;
        if (!text || !actor) return;

        if (!information) {
            text->GetTextProperty()->SetOpacity(0.0);
            return;
        }

        Point6D pose(
            actor->GetPosition()[0],
            actor->GetPosition()[1],
            actor->GetPosition()[2],
            actor->GetOrientation()[0],
            actor->GetOrientation()[1],
            actor->GetOrientation()[2]);
        if (interactor_camera_B)
            pose = interactor_calibration.convert_Pose_B_to_Pose_A(pose);

        text->SetInput(format_pose(pose, speed).c_str());
        text->GetTextProperty()->SetOpacity(1.0);
        text->GetTextProperty()->SetColor(actor->GetProperty()->GetColor());
    }
    // Keypress Function
    void OnKeyPress() override {
        // Get the keypress
        vtkRenderWindowInteractor* rwi = this->Interactor;
        if (rwi == viewer_->get_interactor()) {
        }
        if (this->InteractionProp == NULL) {
            std::string key = rwi->GetKeySym();
            // Handle information toggle
            if (key == "i" || key == "I") {
                if (information == true) {
                    information = false;
                    viewer_->make_actor_text_invisible();
                } else {
                    information = true;
                    viewer_->make_actor_text_visible();
                }
            }

            this->Interactor->GetRenderWindow()->Render();
            return;
        }

        vtkActor* actor = vtkActor::SafeDownCast(this->InteractionProp);
        std::string key = rwi->GetKeySym();
        double* Position = actor->GetPosition();

        // Shift Class
        if (rwi->GetShiftKey()) {
            // Handle Increase Request
            if (key == "plus") {
                if (speed < 20 && speed >= 1) {
                    speed++;
                } else if (speed < 1) {
                    speed += 0.1;
                }
            }

            // Handle Decrease Request
            if (key == "underscore") {
                if (speed > 1) {
                    speed--;
                } else if (speed >= 0.2) {
                    speed -= 0.1;
                }
            }

            // Handle an arrow key
            if (key == "Up") {
                actor->RotateX(speed);
                this->Interactor->GetRenderWindow()->Render();
            }
            // Handle an arrow key
            if (key == "Down") {
                actor->RotateX(-1 * speed);
                this->Interactor->GetRenderWindow()->Render();
            }

            // Handle an arrow key
            if (key == "Left") {
                actor->RotateY(-1 * speed);
                this->Interactor->GetRenderWindow()->Render();
            }

            // Handle an arrow key
            if (key == "Right") {
                actor->RotateY(speed);
                this->Interactor->GetRenderWindow()->Render();
            }
        }
        // Control Class
        else if (rwi->GetControlKey()) {
            // Handle an arrow key
            if (key == "Up") {
                actor->SetPosition(
                    Position[0], Position[1], Position[2] + speed);
                this->Interactor->GetRenderWindow()->Render();
            }
            // Handle an arrow key
            if (key == "Down") {
                actor->SetPosition(
                    Position[0], Position[1], Position[2] - speed);
                this->Interactor->GetRenderWindow()->Render();
            }

            // Handle an arrow key
            if (key == "Left") {
                actor->RotateZ(-1 * speed);
                this->Interactor->GetRenderWindow()->Render();
            }

            // Handle an arrow key
            if (key == "Right") {
                actor->RotateZ(speed);
                this->Interactor->GetRenderWindow()->Render();
            }
        }
        // Naked Class
        else {
            // Handle Increase Request
            if (key == "equal") {
                if (speed < 20 && speed >= 1) {
                    speed++;
                } else if (speed < 1) {
                    speed += 0.1;
                }
            }

            // Handle Decrease Request
            if (key == "minus") {
                if (speed > 1) {
                    speed--;
                } else if (speed >= 0.2) {
                    speed -= 0.1;
                }
            }

            // Handle an arrow key
            if (key == "Up") {
                actor->SetPosition(
                    Position[0], Position[1] + speed, Position[2]);
                this->Interactor->GetRenderWindow()->Render();
            }
            // Handle an arrow key
            if (key == "Down") {
                actor->SetPosition(
                    Position[0], Position[1] - speed, Position[2]);
                this->Interactor->GetRenderWindow()->Render();
            }

            // Handle an arrow key
            if (key == "Left") {
                actor->SetPosition(
                    Position[0] - speed, Position[1], Position[2]);
                this->Interactor->GetRenderWindow()->Render();
            }

            // Handle an arrow key
            if (key == "Right") {
                actor->SetPosition(
                    Position[0] + speed, Position[1], Position[2]);
                this->Interactor->GetRenderWindow()->Render();
            }

            // Handle information toggle
            if (key == "i" || key == "I") {
                if (information == true) {
                    information = false;
                } else {
                    information = true;
                }
            }

            // Handle information toggle
            if (key == "p" || key == "P") {
                if (!ms_->currently_optimizing_) {
                    ms_->VTKMakePrincipalSignal(actor);
                    return;
                }
            }
        }

        // Information Toggle
        update_info_text(actor);
        this->Interactor->GetRenderWindow()->Render();

        // Forward events
        vtkInteractorStyleTrackballActor::OnKeyPress();
        this->Interactor->GetRenderWindow()->Render();

        // Forward events
        vtkInteractorStyleTrackballActor::OnKeyPress();
    }

    // Left Mouse Down Function
    void OnLeftButtonDown() override {
        leftDown = true;

        // Forward Events
        vtkInteractorStyleTrackballActor::OnLeftButtonDown();
    }

    // Right Mouse Down Function
    void OnRightButtonDown() override {
        rightDown = true;
        rightDownY = QCursor::pos().y();

        // Forward Events
        vtkInteractorStyleTrackballActor::OnRightButtonDown();

        if (this->InteractionProp == NULL) {
            return;
        }

        vtkActor* actor = vtkActor::SafeDownCast(this->InteractionProp);
        rightDownModelZ = actor->GetPosition()[2];
    }

    // Middle Mouse Down Funtion
    void OnMiddleButtonDown() override {
        middleDown = true;

        // Forward Events
        vtkInteractorStyleTrackballActor::OnMiddleButtonDown();
    }

    // Left Mouse Up Function
    void OnLeftButtonUp() override {
        leftDown = false;
        if (this->InteractionProp == nullptr) {
            vtkInteractorStyleTrackballActor::OnLeftButtonUp();
            return;
        }
        vtkActor* actor = vtkActor::SafeDownCast(this->InteractionProp);
        update_info_text(actor);
        this->Interactor->GetRenderWindow()->Render();
        vtkInteractorStyleTrackballActor::OnLeftButtonUp();
    }

    // Right Mouse Up Function
    void OnRightButtonUp() override {
        rightDown = false;

        // Forward Events
        vtkInteractorStyleTrackballActor::OnRightButtonUp();
    }

    // Middle Mouse Up Function
    void OnMiddleButtonUp() override {
        middleDown = false;

        // Forward Events
        vtkInteractorStyleTrackballActor::OnMiddleButtonUp();
    }

    // Mouse Movement
    void OnMouseMove() override {
        if (this->InteractionProp == NULL) {
            return;
        }
        if (leftDown || rightDown || middleDown) {
            vtkActor* actor = vtkActor::SafeDownCast(this->InteractionProp);
            if (!leftDown && !middleDown) {
                double* Position = actor->GetPosition();
                actor->SetPosition(
                    Position[0],
                    Position[1],
                    QCursor::pos().y() - rightDownY + rightDownModelZ);
            }
            update_info_text(actor);
            this->Interactor->GetRenderWindow()->Render();
        }

        // Forward Events
        if (!rightDown) {
            vtkActor* actor1 = vtkActor::SafeDownCast(this->InteractionProp);
            if (actor1 == viewer_->get_model_actor_at_index(0)) {
            } else if (actor1 == viewer_->get_model_actor_at_index(1)) {
            } else {
            }
            vtkInteractorStyleTrackballActor::OnMouseMove();
        }
    }

private:
    static std::string format_pose(const Point6D& p, double spd) {
        auto n = [](double v) { return std::to_string(v); };
        return "Location: <" + n(p.x) + "," + n(p.y) + "," + n(p.z) +
               ">\nOrientation: <" + n(p.xa) + "," + n(p.ya) + "," + n(p.za) +
               ">\nKeyboard Speed: " + n(spd);
    }
};

vtkStandardNewMacro(KeyPressInteractorStyle);

class CameraInteractorStyle : public vtkInteractorStyleTrackballCamera {
public:
    static CameraInteractorStyle* New();
    vtkTypeMacro(CameraInteractorStyle, vtkInteractorStyleTrackballCamera);

    // KeyPress Turns Off Other Char Hotkeys
    void OnChar() override {}
};

vtkStandardNewMacro(CameraInteractorStyle);

#endif /* INTERACTOR_H */
