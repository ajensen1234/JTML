# Agent Operating Guide — GUI Module (src/gui)

## OVERVIEW
Interactive 2D/3D visualization and optimization control. Centralized in the monolithic `MainScreen` (5.8k lines) which manages UI, VTK, and orchestration.

## STRUCTURE
- `MainScreen`: Central controller; handles Qt signals/slots and high-level logic
- `Viewer`: VTK abstraction; manages renderers, actors, and cameras
- `KeyPressInteractorStyle`: Custom VTK interactor for manual model manipulation
- `SettingsControl`: UI for optimizer hyperparameter management
- `DRRTool`: UI for Digitally Reconstructed Radiograph generation

## WHERE TO LOOK
- **Qt Slots**: `include/gui/mainscreen.h` (lines 257-399)
- **VTK Rendering**: `src/gui/viewer.cpp` (renderer/actor setup)
- **Manual Interaction**: `include/gui/interactor.h` (keyboard/mouse logic)
- **UI Layout**: `MainScreen::ArrangeMainScreenLayout` in `mainscreen.cpp`

## ANTI-PATTERNS
- **Monolithic Slots**: Avoid adding complex logic directly to `on_button_clicked` slots. Delegate to `Viewer` or `OptimizerManager`.
- **Blocking UI**: Never run heavy computation (e.g., PyTorch inference) on the main thread. Use `QThread` or `QtConcurrent`.
- **Direct VTK Access**: Prefer using `Viewer` methods over direct manipulation of VTK pointers in `MainScreen`.
- **Global State**: Avoid using global variables in `interactor.h` (e.g., `speed`, `middleDown`). Move to class members.

## CONVENTIONS
- **Naming**: Qt slots follow `on_<objectName>_<signalName>` for auto-connection.
- **VTK**: Use `vtkSmartPointer` for all VTK objects to ensure proper reference counting.
- **Threading**: Use `Q_SIGNALS` and `Q_SLOTS` for communication between the UI and worker threads.
