/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#pragma once
/*VTK*/

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
#include "core/calibration.h"

/*Drr Tool Header*/
#include "gui/drr_tool.h"

/*DRR Globals*/
bool middleDownDRR = false;     // Is CM button down?
bool leftDownDRR = false;       // Is LM button down?
bool rightDownDRR = false;      // Is RM button down
int rightDownDRRY = 0;          // Y Pixel when RM Clicked
double rightDownDRRModelZ = 0;  // Model's Z Translation when RM Clicked

class DRRInteractorStyle : public vtkInteractorStyleTrackballActor {
   public:
    static DRRInteractorStyle* New();
    vtkTypeMacro(DRRInteractorStyle, vtkInteractorStyleTrackballActor);

    /*Pointer to Main Window*/
    DRRTool* drrtool_;
    void initialize_DRRTool(DRRTool* drrtool) { drrtool_ = drrtool; }

    // Picked Function
    bool ActivePick() {
        if (this->InteractionProp == NULL)
            return false;
        else
            return true;
    }

    // KeyPress Turns Off Other Char Hotkeys
    virtual void OnChar() {}

    // Keypress Function
    virtual void OnKeyPress() {
        // Get the keypress
        vtkRenderWindowInteractor* rwi = this->Interactor;
        if (this->InteractionProp == NULL) {
            std::string key = rwi->GetKeySym();

            this->Interactor->GetRenderWindow()->Render();
            return;
        }

        vtkActor* actor = vtkActor::SafeDownCast(this->InteractionProp);
        std::string key = rwi->GetKeySym();
        double* Position = actor->GetPosition();

        // Shift Class
        if (rwi->GetShiftKey()) {
            // Handle an arrow key
            if (key == "Up") {
                actor->RotateX(1);
                this->Interactor->GetRenderWindow()->Render();
            }
            // Handle an arrow key
            if (key == "Down") {
                actor->RotateX(-1 * 1);
                this->Interactor->GetRenderWindow()->Render();
            }

            // Handle an arrow key
            if (key == "Left") {
                actor->RotateY(-1 * 1);
                this->Interactor->GetRenderWindow()->Render();
            }

            // Handle an arrow key
            if (key == "Right") {
                actor->RotateY(1);
                this->Interactor->GetRenderWindow()->Render();
            }
        }
        // Control Class
        else if (rwi->GetControlKey()) {
            // Handle an arrow key
            if (key == "Up") {
                actor->SetPosition(Position[0], Position[1], Position[2] + 1);
                this->Interactor->GetRenderWindow()->Render();
            }
            // Handle an arrow key
            if (key == "Down") {
                actor->SetPosition(Position[0], Position[1], Position[2] - 1);
                this->Interactor->GetRenderWindow()->Render();
            }

            // Handle an arrow key
            if (key == "Left") {
                actor->RotateZ(-1 * 1);
                this->Interactor->GetRenderWindow()->Render();
            }

            // Handle an arrow key
            if (key == "Right") {
                actor->RotateZ(1);
                this->Interactor->GetRenderWindow()->Render();
            }
        }
        // Naked Class
        else {
            // Handle an arrow key
            if (key == "Up") {
                actor->SetPosition(Position[0], Position[1] + 1, Position[2]);
                this->Interactor->GetRenderWindow()->Render();
            }
            // Handle an arrow key
            if (key == "Down") {
                actor->SetPosition(Position[0], Position[1] - 1, Position[2]);
                this->Interactor->GetRenderWindow()->Render();
            }

            // Handle an arrow key
            if (key == "Left") {
                actor->SetPosition(Position[0] - 1, Position[1], Position[2]);
                this->Interactor->GetRenderWindow()->Render();
            }

            // Handle an arrow key
            if (key == "Right") {
                actor->SetPosition(Position[0] + 1, Position[1], Position[2]);
                this->Interactor->GetRenderWindow()->Render();
            }
        }

        this->Interactor->GetRenderWindow()->Render();

        // Forward events
        vtkInteractorStyleTrackballActor::OnKeyPress();
    }

    // Left Mouse Down Function
    virtual void OnLeftButtonDown() {
        leftDownDRR = true;

        // Forward Events
        vtkInteractorStyleTrackballActor::OnLeftButtonDown();
    }

    // Right Mouse Down Function
    virtual void OnRightButtonDown() {
        rightDownDRR = true;
        rightDownDRRY = QCursor::pos().y();

        // Forward Events
        vtkInteractorStyleTrackballActor::OnRightButtonDown();

        if (this->InteractionProp == NULL) return;
        vtkActor* actor = vtkActor::SafeDownCast(this->InteractionProp);
        rightDownDRRModelZ = actor->GetPosition()[2];
    }

    // Middle Mouse Down Funtion
    virtual void OnMiddleButtonDown() {
        middleDownDRR = true;

        // Forward Events
        vtkInteractorStyleTrackballActor::OnMiddleButtonDown();
    }

    // Left Mouse Up Function
    virtual void OnLeftButtonUp() {
        if (this->InteractionProp == NULL) return;
        vtkActor* actor = vtkActor::SafeDownCast(this->InteractionProp);

        leftDownDRR = false;
        this->Interactor->GetRenderWindow()->Render();
        // Forward Events
        vtkInteractorStyleTrackballActor::OnLeftButtonUp();
    }

    // Right Mouse Up Function
    virtual void OnRightButtonUp() {
        rightDownDRR = false;

        // Forward Events
        vtkInteractorStyleTrackballActor::OnRightButtonUp();
    }

    // Middle Mouse Up Function
    virtual void OnMiddleButtonUp() {
        middleDownDRR = false;

        // Forward Events
        vtkInteractorStyleTrackballActor::OnMiddleButtonUp();
    }

    // Mouse Movement
    virtual void OnMouseMove() {
        if (this->InteractionProp == NULL) return;
        if (leftDownDRR == true || rightDownDRR == true ||
            middleDownDRR == true) {
            vtkActor* actor = vtkActor::SafeDownCast(this->InteractionProp);

            // If Right Down and Not Left or MiddleScale The Z
            if (!leftDownDRR && !middleDownDRR) {
                double* Position = actor->GetPosition();
                actor->SetPosition(
                    Position[0], Position[1],
                    QCursor::pos().y() - rightDownDRRY + rightDownDRRModelZ);
            }
            this->Interactor->GetRenderWindow()->Render();
        }

        // Forward Events
        if (!rightDownDRR) vtkInteractorStyleTrackballActor::OnMouseMove();

        /*Draw DRR*/
        drrtool_->DrawDRR();
    }
};
vtkStandardNewMacro(DRRInteractorStyle);
