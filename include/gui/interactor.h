#ifndef INTERACTOR_H
#define INTERACTOR_H

#include <vtkObjectFactory.h>
#include <vtkInteractorStyleTrackballActor.h>
#include <vtkRendererCollection.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkActor2DCollection.h>
#include <vtkPicker.h>
#include <vtkPropPicker.h>
#include <vtkProp.h>
#include <qcursor.h>

/*Ref to QMainWindow*/
#include "gui/mainscreen.h"

//Calibration To Convert Pose
#include "core/calibration.h"
Calibration interactor_calibration;

//Speed of Movement
int speed = 1;
bool information = true;
bool interactor_camera_B = false; //Are we in Camera B?
bool middleDown = false; // Is CM button down?
bool leftDown = false; //Is LM button down?
bool rightDown = false; //Is RM button down
int rightDownY = 0; //Y Pixel when RM Clicked
double rightDownModelZ = 0; //Model's Z Translation when RM Clicked

class KeyPressInteractorStyle : public vtkInteractorStyleTrackballActor {
public:
	static KeyPressInteractorStyle* New();
	vtkTypeMacro(KeyPressInteractorStyle, vtkInteractorStyleTrackballActor);

	/*Pointer to Main Window*/
	MainScreen* ms_;
	std::shared_ptr<Viewer> viewer_;

	void initialize_MainScreen(MainScreen* ms) {
		ms_ = ms;
	}

	void initialize_viewer(std::shared_ptr<Viewer> viewer) {
		viewer_ = viewer;
	}

	//Picked Function
	bool ActivePick() {
		if (this->InteractionProp == NULL) {
			return false;
		}
		return true;
	}

	//KeyPress Turns Off Other Char Hotkeys
	void OnChar() override {
		vtkRenderWindowInteractor* rwi = this->Interactor;
		std::string key = rwi->GetKeySym();
		if (key == "Escape" || key == "escape" || key == "ESC" || key == "Esc" || key == "esc") {
			ms_->VTKEscapeSignal();
		}
	}

	//Keypress Function
	void OnKeyPress() override {
		// Get the keypress
		vtkRenderWindowInteractor* rwi = this->Interactor;
		if (this->InteractionProp == NULL) {
			std::string key = rwi->GetKeySym();

			// Handle information toggle
			if (key == "i" || key == "I") {
				vtkTextActor* text = vtkTextActor::SafeDownCast(
					this->Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActors2D()->
					      GetLastActor2D());
				if (information == true) {
					information = false;
					text->GetTextProperty()->SetOpacity(0.0);
				}
				else {
					information = true;
					text->GetTextProperty()->SetOpacity(1.0);
				}
			}

			this->Interactor->GetRenderWindow()->Render();
			return;
		}

		vtkActor* actor = vtkActor::SafeDownCast(this->InteractionProp);
		std::string key = rwi->GetKeySym();
		double* Position = actor->GetPosition();

		//Shift Class
		if (rwi->GetShiftKey()) {
			//Handle Increase Request
			if (key == "plus") {
				if (speed < 20) {
					speed++;
				}
			}

			//Handle Decrease Request
			if (key == "underscore") {
				if (speed > 1) {
					speed--;
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
		//Control Class
		else if (rwi->GetControlKey()) {
			// Handle an arrow key
			if (key == "Up") {
				actor->SetPosition(Position[0], Position[1], Position[2] + speed);
				this->Interactor->GetRenderWindow()->Render();
			}
			// Handle an arrow key
			if (key == "Down") {
				actor->SetPosition(Position[0], Position[1], Position[2] - speed);
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
		//Naked Class
		else {
			//Handle Increase Request
			if (key == "equal") {
				if (speed < 20) {
					speed++;
				}
			}

			//Handle Decrease Request
			if (key == "minus") {
				if (speed > 1) {
					speed--;
				}
			}

			// Handle an arrow key
			if (key == "Up") {
				actor->SetPosition(Position[0], Position[1] + speed, Position[2]);
				this->Interactor->GetRenderWindow()->Render();
			}
			// Handle an arrow key
			if (key == "Down") {
				actor->SetPosition(Position[0], Position[1] - speed, Position[2]);
				this->Interactor->GetRenderWindow()->Render();
			}

			// Handle an arrow key
			if (key == "Left") {
				actor->SetPosition(Position[0] - speed, Position[1], Position[2]);
				this->Interactor->GetRenderWindow()->Render();
			}

			// Handle an arrow key
			if (key == "Right") {
				actor->SetPosition(Position[0] + speed, Position[1], Position[2]);
				this->Interactor->GetRenderWindow()->Render();
			}

			// Handle information toggle
			if (key == "i" || key == "I") {
				if (information == true) {
					information = false;
				}
				else {
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

		//Information Toggle
		std::string infoText = "Location: <";
		vtkTextActor* text = vtkTextActor::SafeDownCast(
			this->Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActors2D()->GetLastActor2D());
		if (information == true) {
			if (interactor_camera_B == false) {
				infoText += std::to_string(static_cast<long double>(actor->GetPosition()[0])) + ","
					+ std::to_string(static_cast<long double>(actor->GetPosition()[1])) + ","
					+ std::to_string(static_cast<long double>(actor->GetPosition()[2])) + ">\nOrientation: <"
					+ std::to_string(static_cast<long double>(actor->GetOrientation()[0])) + ","
					+ std::to_string(static_cast<long double>(actor->GetOrientation()[1])) + ","
					+ std::to_string(static_cast<long double>(actor->GetOrientation()[2])) + ">\nKeyboard Speed: " +
					std::to_string(speed);

			}
			else {
				auto current_position_B = Point6D(actor->GetPosition()[0], actor->GetPosition()[1],
				                                  actor->GetPosition()[2],
				                                  actor->GetOrientation()[0], actor->GetOrientation()[1],
				                                  actor->GetOrientation()[2]);
				Point6D current_position_A = interactor_calibration.convert_Pose_B_to_Pose_A(current_position_B);
				infoText += std::to_string(static_cast<long double>(current_position_A.x)) + ","
					+ std::to_string(static_cast<long double>(current_position_A.y)) + ","
					+ std::to_string(static_cast<long double>(current_position_A.z)) + ">\nOrientation: <"
					+ std::to_string(static_cast<long double>(current_position_A.xa)) + ","
					+ std::to_string(static_cast<long double>(current_position_A.ya)) + ","
					+ std::to_string(static_cast<long double>(current_position_A.za)) + ">\nKeyboard Speed: " +
					std::to_string(speed);
			}
			text->GetTextProperty()->SetOpacity(1.0);
			text->GetTextProperty()->SetColor(actor->GetProperty()->GetColor());
		}
		else {
			text->GetTextProperty()->SetOpacity(0.0);
		}
		text->SetInput(infoText.c_str());
		this->Interactor->GetRenderWindow()->Render();

		//Forward events
		vtkInteractorStyleTrackballActor::OnKeyPress();
	}

	//Left Mouse Down Function
	void OnLeftButtonDown() override {
		leftDown = true;

		// Forward Events
		vtkInteractorStyleTrackballActor::OnLeftButtonDown();
	}

	//Right Mouse Down Function
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

	//Middle Mouse Down Funtion
	void OnMiddleButtonDown() override {
		middleDown = true;

		// Forward Events
		vtkInteractorStyleTrackballActor::OnMiddleButtonDown();
	}

	//Left Mouse Up Function
	void OnLeftButtonUp() override {
		if (this->InteractionProp == NULL) {
			return;
		}
		vtkActor* actor = vtkActor::SafeDownCast(this->InteractionProp);

		leftDown = false;
		//Information Toggle
		std::string infoText = "Location: <";
		vtkTextActor* text = vtkTextActor::SafeDownCast(
			this->Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActors2D()->GetLastActor2D());
		if (information == true) {
			if (interactor_camera_B == false) {
				infoText += std::to_string(static_cast<long double>(actor->GetPosition()[0])) + ","
					+ std::to_string(static_cast<long double>(actor->GetPosition()[1])) + ","
					+ std::to_string(static_cast<long double>(actor->GetPosition()[2])) + ">\nOrientation: <"
					+ std::to_string(static_cast<long double>(actor->GetOrientation()[0])) + ","
					+ std::to_string(static_cast<long double>(actor->GetOrientation()[1])) + ","
					+ std::to_string(static_cast<long double>(actor->GetOrientation()[2])) + ">\nKeyboard Speed: " +
					std::to_string(speed);

			}
			else {
				auto current_position_B = Point6D(actor->GetPosition()[0], actor->GetPosition()[1],
				                                  actor->GetPosition()[2],
				                                  actor->GetOrientation()[0], actor->GetOrientation()[1],
				                                  actor->GetOrientation()[2]);
				Point6D current_position_A = interactor_calibration.convert_Pose_B_to_Pose_A(current_position_B);
				infoText += std::to_string(static_cast<long double>(current_position_A.x)) + ","
					+ std::to_string(static_cast<long double>(current_position_A.y)) + ","
					+ std::to_string(static_cast<long double>(current_position_A.z)) + ">\nOrientation: <"
					+ std::to_string(static_cast<long double>(current_position_A.xa)) + ","
					+ std::to_string(static_cast<long double>(current_position_A.ya)) + ","
					+ std::to_string(static_cast<long double>(current_position_A.za)) + ">\nKeyboard Speed: " +
					std::to_string(speed);
			}
			text->GetTextProperty()->SetOpacity(1.0);
			text->GetTextProperty()->SetColor(actor->GetProperty()->GetColor());
		}
		else {
			text->GetTextProperty()->SetOpacity(0.0);
		}
		text->SetInput(infoText.c_str());
		this->Interactor->GetRenderWindow()->Render();
		// Forward Events
		vtkInteractorStyleTrackballActor::OnLeftButtonUp();
	}

	//Right Mouse Up Function
	void OnRightButtonUp() override {
		rightDown = false;

		// Forward Events
		vtkInteractorStyleTrackballActor::OnRightButtonUp();
	}

	//Middle Mouse Up Function
	void OnMiddleButtonUp() override {
		middleDown = false;

		//Forward Events
		vtkInteractorStyleTrackballActor::OnMiddleButtonUp();
	}

	//Mouse Movement
	void OnMouseMove() override {
		if (this->InteractionProp == NULL) {
			return;
		}
		if (leftDown == true || rightDown == true || middleDown == true) {
			vtkActor* actor = vtkActor::SafeDownCast(this->InteractionProp);

			//If Right Down and Not Left or MiddleScale The Z
			if (!leftDown && !middleDown) {
				double* Position = actor->GetPosition();
				actor->SetPosition(Position[0], Position[1], QCursor::pos().y() - rightDownY + rightDownModelZ);
			}

			//Information Toggle
			std::string infoText = "Location: <";
			vtkTextActor* text = vtkTextActor::SafeDownCast(
				this->Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActors2D()->
				      GetLastActor2D());
			if (information == true) {
				if (interactor_camera_B == false) {
					infoText += std::to_string(static_cast<long double>(actor->GetPosition()[0])) + ","
						+ std::to_string(static_cast<long double>(actor->GetPosition()[1])) + ","
						+ std::to_string(static_cast<long double>(actor->GetPosition()[2])) + ">\nOrientation: <"
						+ std::to_string(static_cast<long double>(actor->GetOrientation()[0])) + ","
						+ std::to_string(static_cast<long double>(actor->GetOrientation()[1])) + ","
						+ std::to_string(static_cast<long double>(actor->GetOrientation()[2])) + ">\nKeyboard Speed: " +
						std::to_string(speed);

				}
				else {
					auto current_position_B = Point6D(actor->GetPosition()[0], actor->GetPosition()[1],
					                                  actor->GetPosition()[2],
					                                  actor->GetOrientation()[0], actor->GetOrientation()[1],
					                                  actor->GetOrientation()[2]);
					Point6D current_position_A = interactor_calibration.convert_Pose_B_to_Pose_A(current_position_B);
					infoText += std::to_string(static_cast<long double>(current_position_A.x)) + ","
						+ std::to_string(static_cast<long double>(current_position_A.y)) + ","
						+ std::to_string(static_cast<long double>(current_position_A.z)) + ">\nOrientation: <"
						+ std::to_string(static_cast<long double>(current_position_A.xa)) + ","
						+ std::to_string(static_cast<long double>(current_position_A.ya)) + ","
						+ std::to_string(static_cast<long double>(current_position_A.za)) + ">\nKeyboard Speed: " +
						std::to_string(speed);
				}
				text->GetTextProperty()->SetOpacity(1.0);
				text->GetTextProperty()->SetColor(actor->GetProperty()->GetColor());
			}
			else {
				text->GetTextProperty()->SetOpacity(0.0);
			}
			text->SetInput(infoText.c_str());
			this->Interactor->GetRenderWindow()->Render();
		}

		// Forward Events
		if (!rightDown) {
			vtkInteractorStyleTrackballActor::OnMouseMove();
		}
	}
};

vtkStandardNewMacro(KeyPressInteractorStyle);


class CameraInteractorStyle : public vtkInteractorStyleTrackballCamera {
public:
	static CameraInteractorStyle* New();
	vtkTypeMacro(CameraInteractorStyle, vtkInteractorStyleTrackballCamera);

	//KeyPress Turns Off Other Char Hotkeys
	void OnChar() override {
	}
};

vtkStandardNewMacro(CameraInteractorStyle);

#endif /* INTERACTOR_H */
