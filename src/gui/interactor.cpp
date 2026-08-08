#include "gui/interactor.h"
#include "gui/mainscreen.h"
#include "gui/viewer.h"
#include "services/calibration.h"

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

// Calibration To Convert Pose
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


void KeyPressInteractorStyle::initialize_MainScreen(MainScreen* ms) {
    ms_ = ms;
}

void KeyPressInteractorStyle::initialize_viewer(std::shared_ptr<Viewer> viewer) {
    viewer_ = viewer;
}

void KeyPressInteractorStyle::OnChar() {
    vtkRenderWindowInteractor* rwi = this->Interactor;
    std::string key = rwi->GetKeySym();
    if (key == "Escape" || key == "escape" || key == "ESC" ||
        key == "Esc" || key == "esc") {
        ms_->VTKEscapeSignal();
    }
}

void KeyPressInteractorStyle::OnKeyPress() {
    // Get the keypress
    vtkRenderWindowInteractor* rwi = this->Interactor;
    if (rwi == viewer_->get_interactor()) {
    }
    if (this->InteractionProp == NULL) {
        std::string key = rwi->GetKeySym();
        vtkTextActor* text =
            vtkTextActor::SafeDownCast(this->Interactor->GetRenderWindow()
                                           ->GetRenderers()
                                           ->GetFirstRenderer()
                                           ->GetActors2D()
                                           ->GetLastActor2D());

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
    std::string infoText = "Location: <";
    vtkTextActor* text =
        vtkTextActor::SafeDownCast(this->Interactor->GetRenderWindow()
                                       ->GetRenderers()
                                       ->GetFirstRenderer()
                                       ->GetActors2D()
                                       ->GetLastActor2D());
    if (information == true) {
        if (interactor_camera_B == false) {
            infoText +=
                std::to_string(
                    static_cast<long double>(actor->GetPosition()[0])) +
                "," +
                std::to_string(
                    static_cast<long double>(actor->GetPosition()[1])) +
                "," +
                std::to_string(
                    static_cast<long double>(actor->GetPosition()[2])) +
                ">\nOrientation: <" +
                std::to_string(
                    static_cast<long double>(actor->GetOrientation()[0])) +
                "," +
                std::to_string(
                    static_cast<long double>(actor->GetOrientation()[1])) +
                "," +
                std::to_string(
                    static_cast<long double>(actor->GetOrientation()[2])) +
                ">\nKeyboard Speed: " + std::to_string(speed);

        } else {
            auto current_position_B = Point6D(
                actor->GetPosition()[0],
                actor->GetPosition()[1],
                actor->GetPosition()[2],
                actor->GetOrientation()[0],
                actor->GetOrientation()[1],
                actor->GetOrientation()[2]);
            Point6D current_position_A =
                interactor_calibration.convert_Pose_B_to_Pose_A(
                    current_position_B);
            infoText +=
                std::to_string(
                    static_cast<long double>(current_position_A.x)) +
                "," +
                std::to_string(
                    static_cast<long double>(current_position_A.y)) +
                "," +
                std::to_string(
                    static_cast<long double>(current_position_A.z)) +
                ">\nOrientation: <" +
                std::to_string(
                    static_cast<long double>(current_position_A.xa)) +
                "," +
                std::to_string(
                    static_cast<long double>(current_position_A.ya)) +
                "," +
                std::to_string(
                    static_cast<long double>(current_position_A.za)) +
                ">\nKeyboard Speed: " + std::to_string(speed);
        }
        text->GetTextProperty()->SetOpacity(1.0);
        text->GetTextProperty()->SetColor(actor->GetProperty()->GetColor());
    } else {
        text->GetTextProperty()->SetOpacity(0.0);
    }
    text->SetInput(infoText.c_str());
    this->Interactor->GetRenderWindow()->Render();

    // Forward events
    vtkInteractorStyleTrackballActor::OnKeyPress();
}

vtkStandardNewMacro(KeyPressInteractorStyle);


vtkStandardNewMacro(CameraInteractorStyle);