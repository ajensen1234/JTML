#include "gui/interactor.h"
#include "gui/mainscreen.h"
#include "gui/viewer.h"
#include "core/calibration.h"
#include "core/data_structures_6D.h"

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
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkActor.h>

// Definitions of Global Interactor State
Calibration interactor_calibration;
double speed = 1;
bool information = true;
bool interactor_camera_B = false;
bool middleDown = false;
bool leftDown = false;
bool rightDown = false;
int rightDownY = 0;
double rightDownModelZ = 0;

vtkStandardNewMacro(KeyPressInteractorStyle);
vtkStandardNewMacro(CameraInteractorStyle);

void KeyPressInteractorStyle::initialize_MainScreen(MainScreen* ms) {
    ms_ = ms;
}

void KeyPressInteractorStyle::initialize_viewer(std::shared_ptr<Viewer> viewer) {
    viewer_ = viewer;
}

// Picked Function
bool KeyPressInteractorStyle::ActivePick() {
    if (this->InteractionProp == NULL) {
        return false;
    }
    return true;
}

// KeyPress Turns Off Other Char Hotkeys
void KeyPressInteractorStyle::OnChar() {
    vtkRenderWindowInteractor* rwi = this->Interactor;
    std::string key = rwi->GetKeySym();
    if (key == "Escape" || key == "escape" || key == "ESC" ||
        key == "Esc" || key == "esc") {
        ms_->VTKEscapeSignal();
    }
}

// Keypress Function
void KeyPressInteractorStyle::OnKeyPress() {
    // Get the keypress
    vtkRenderWindowInteractor* rwi = this->Interactor;
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

    // Information Update logic would go here if needed...
    this->Interactor->GetRenderWindow()->Render();

    // Forward events
    vtkInteractorStyleTrackballActor::OnKeyPress();
}

void KeyPressInteractorStyle::OnLeftButtonDown() {
    leftDown = true;
    vtkInteractorStyleTrackballActor::OnLeftButtonDown();
}

void KeyPressInteractorStyle::OnRightButtonDown() {
    rightDown = true;
    rightDownY = QCursor::pos().y();
    vtkInteractorStyleTrackballActor::OnRightButtonDown();
    if (this->InteractionProp == NULL) return;
    vtkActor* actor = vtkActor::SafeDownCast(this->InteractionProp);
    rightDownModelZ = actor->GetPosition()[2];
}

void KeyPressInteractorStyle::OnMiddleButtonDown() {
    middleDown = true;
    vtkInteractorStyleTrackballActor::OnMiddleButtonDown();
}

void KeyPressInteractorStyle::OnLeftButtonUp() {
    leftDown = false;
    vtkInteractorStyleTrackballActor::OnLeftButtonUp();
}

void KeyPressInteractorStyle::OnRightButtonUp() {
    rightDown = false;
    vtkInteractorStyleTrackballActor::OnRightButtonUp();
}

void KeyPressInteractorStyle::OnMiddleButtonUp() {
    middleDown = false;
    vtkInteractorStyleTrackballActor::OnMiddleButtonUp();
}

void KeyPressInteractorStyle::OnMouseMove() {
    if (this->InteractionProp == NULL) {
        vtkInteractorStyleTrackballActor::OnMouseMove();
        return;
    }
    if (leftDown || rightDown || middleDown) {
        vtkActor* actor = vtkActor::SafeDownCast(this->InteractionProp);
        if (rightDown && !leftDown && !middleDown) {
            double* Position = actor->GetPosition();
            actor->SetPosition(Position[0], Position[1], QCursor::pos().y() - rightDownY + rightDownModelZ);
        }
        this->Interactor->GetRenderWindow()->Render();
    }
    if (!rightDown) {
        vtkInteractorStyleTrackballActor::OnMouseMove();
    }
}
