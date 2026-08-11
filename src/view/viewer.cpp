// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "view/viewer.h"

#include <cmath>

#include <vtkRendererCollection.h>

#include "services/render_pipeline_builder.h"

namespace {

/*The widgets' scene view-angle derivation — the ex-Viewer::
 * calculate_and_set_viewing_angle_from_calibration body, relocated verbatim
 * into a file-local helper when the scene camera recipe moved to the shared
 * jta::render_pipeline builder (006 U4): the formula is PINNED behavior,
 * not to be "fixed". The builder's ApplySceneCameraProjection applies the
 * angle + clipping; this helper only derives the angle (view-side math).*/
double ViewingAngleFromCalibration(int h, int fy) {
    const double pi = 3.14159265358979323846;
    return (180.0 / pi) * 2 * atan2(h, 2 * fy);
}

}  // namespace

Viewer::Viewer() {
    initialize_vtk_pointers();
    initialize_vtk_mappers();
    initialize_vtk_renderers();
}

Viewer::~Viewer() {}

void Viewer::initialize_vtk_pointers() {
    // TODO: Throw a catch statement here to make sure there are no errors
    background_renderer_ = vtkSmartPointer<vtkRenderer>::New();
    scene_renderer_ = vtkSmartPointer<vtkRenderer>::New();

    background_camera_ = background_renderer_->GetActiveCamera();
    scene_camera_ = scene_renderer_->GetActiveCamera();
    actor_image_ = vtkSmartPointer<vtkActor>::New();
    current_background_ = vtkSmartPointer<vtkImageData>::New();
    stl_reader_ = vtkSmartPointer<vtkSTLReader>::New();
    image_mapper_ = vtkSmartPointer<vtkDataSetMapper>::New();
    actor_text_ = vtkSmartPointer<vtkTextActor>::New();
    importer_ = vtkSmartPointer<vtkImageImport>::New();
    actor_text_->GetTextProperty()->SetFontSize(16);
    actor_text_->GetTextProperty()->SetFontFamilyToCourier();
    actor_text_->SetPosition2(0, 0);
    actor_text_->GetTextProperty()->SetColor(
        214.0 / 255.0,
        108.0 / 255.0,
        35.0 / 255.0); // Earth Reda
    render_window_interactor_ =
        vtkSmartPointer<vtkRenderWindowInteractor>::New();
}

void Viewer::initialize_vtk_mappers() {
    // Shared pipeline recipe (006 U4): importer -> imageData -> mapper ->
    // actor -> background renderer, incl. importer->SetOutput(imageData)
    // (previously wired lazily on the first update_display_background —
    // identical, since no Update runs before that) + SetPickable(0). The
    // text actor stays view-side (the QML renderer has no text overlay).
    jta::render_pipeline::ConfigureBackgroundChain(
        importer_.Get(), current_background_.Get(), image_mapper_.Get(),
        actor_image_.Get(), background_renderer_.Get());
    actor_text_->SetPickable(0);
}

void Viewer::initialize_vtk_renderers() {
    background_renderer_->AddActor2D(actor_text_);
}

void Viewer::load_render_window(vtkSmartPointer<vtkRenderWindow> in) {
    qvtk_render_window_ = in;
    render_window_interactor_->SetRenderWindow(qvtk_render_window_);
}

vtkSmartPointer<vtkRenderer> Viewer::get_renderer() {
    return background_renderer_;
}

int Viewer::get_number_of_model_actors() {
    return this->model_actor_list_.size();
}

vtkSmartPointer<vtkActor> Viewer::get_actor_image() {
    return actor_image_;
}

vtkSmartPointer<vtkImageData> Viewer::get_current_background() {
    return current_background_;
}

vtkSmartPointer<vtkSTLReader> Viewer::get_stl_reader() {
    return stl_reader_;
}

vtkSmartPointer<vtkDataSetMapper> Viewer::get_image_mapper() {
    return image_mapper_;
}

vtkSmartPointer<vtkTextActor> Viewer::get_actor_text() {
    return actor_text_;
}

vtkSmartPointer<vtkImageImport> Viewer::get_importer() {
    return importer_;
}

void Viewer::update_display_background(cv::Mat desiredBackground) {
    // Shared pipeline recipe (006 U4): zero-copy import configuration
    // (spacing/origin/extent/scalar-type/channels/SetImportVoidPointer/
    // Modified/Update — the dead matToVTK semantics). The importer wraps
    // the Mat buffer WITHOUT copying; the caller (MainScreen) keeps the
    // owning Frame alive. Empty Mat -> no-op (keeps the current
    // background).
    jta::render_pipeline::RefreshBackgroundImport(importer_, desiredBackground);
}

void Viewer::make_image_invisible() {
    actor_image_->SetVisibility(false);
}

void Viewer::set_loaded_frames(std::vector<Frame>& frames) {
    loaded_frames_ = frames;
}

void Viewer::set_loaded_frames_b(std::vector<Frame>& frames) {
    loaded_frames_B_ = frames;
}

void Viewer::update_display_background_to_edge_image(
    int frame_number, bool CameraASelected) {
    // if 1, do 2. Else, do 3 (ternary operator)
    (CameraASelected)
        ? update_display_background(loaded_frames_[frame_number].GetEdgeImage())
        : update_display_background(
              loaded_frames_B_[frame_number].GetEdgeImage());
}

void Viewer::update_display_background_to_original_image(
    int frame_number, bool CameraASelected) {
    (CameraASelected) ? update_display_background(
                            loaded_frames_[frame_number].GetOriginalImage())
                      : update_display_background(
                            loaded_frames_B_[frame_number].GetOriginalImage());
}

void Viewer::update_display_background_to_dilation_image(
    int frame_number, bool CameraASelected) {
    (CameraASelected) ? update_display_background(
                            loaded_frames_[frame_number].GetDilationImage())
                      : update_display_background(
                            loaded_frames_B_[frame_number].GetDilationImage());
}

void Viewer::update_display_background_to_inverted_image(
    int frame_number, bool CameraASelected) {
    (CameraASelected) ? update_display_background(
                            loaded_frames_[frame_number].GetInvertedImage())
                      : update_display_background(
                            loaded_frames_B_[frame_number].GetInvertedImage());
}

void Viewer::setup_camera_calibration(Calibration cal) {
    // Shared pipeline recipe (006 U4): background camera focal (0,0,-1),
    // position (0,0,0), clipping (0.1, 2*fy). The old no-op cal.type_
    // ternary (both branches set (0,0,-1)) is gone with it.
    jta::render_pipeline::SetupBackgroundCamera(
        background_renderer_, cal.camera_A_principal_.fy());
}

void Viewer::setup_camera_coronal_plane() {
    background_camera_->SetPosition(-1, 1, 0);
}

void Viewer::place_image_actors_according_to_calibration(
    Calibration cal, int img_w, int img_h) {
    // Shared pipeline recipe (006 U4): actor at (-0.5w, -0.5h, z) with
    // parallel scale 0.5h; z = -fy*pixel_pitch is the widgets' value of
    // camera divergence (a) (the QML side passes -focalLengthPx). The old
    // no-op cal.type_ ternary is gone with it.
    const double z_pos = -cal.camera_A_principal_.fy() *
                         cal.camera_A_principal_.pixel_pitch_;
    jta::render_pipeline::PlaceBackgroundImage(
        actor_image_, background_renderer_, img_w, img_h, z_pos);
}

void Viewer::load_3d_models_into_actor_and_mapper_list() {
    for (int i = num_models_loaded_; i < loaded_models_->size(); i++) {
        vtkSmartPointer<vtkActor> new_actor = vtkSmartPointer<vtkActor>::New();
        vtkSmartPointer<vtkPolyDataMapper> new_mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
        // Shared pipeline recipe (006 U4): reader output -> mapper -> actor
        // -> scene renderer. The per-model bookkeeping stays view-side.
        jta::render_pipeline::BuildModelActor(
            new_mapper, new_actor,
            loaded_models_->at(i).cad_reader_->GetOutputPort(),
            scene_renderer_);
        model_actor_list_.push_back(new_actor);
        model_mapper_list_.push_back(new_mapper);
        num_models_loaded_++;
    }
}

void Viewer::load_model_actors_and_mappers_with_3d_data() {
    for (int i = 0; i < model_actor_list_.size(); i++) {
        model_actor_list_[i]->PickableOff();
        model_actor_list_[i]->VisibilityOff();
    }
}

std::vector<vtkSmartPointer<vtkPolyDataMapper>>
Viewer::get_model_mapper_list() {
    return model_mapper_list_;
}

std::vector<vtkSmartPointer<vtkActor>> Viewer::get_model_actor_list() {
    return model_actor_list_;
}

void Viewer::set_3d_model_color(int index, double RGB[3]) {
    model_actor_list_[index]->GetProperty()->SetColor(
        RGB[0] / 255.0, RGB[1] / 255.0, RGB[2] / 255.0);
}

void Viewer::load_models(QStringList cad_files, QStringList cad_models) {
    for (int i = 0; i < cad_files.size(); i++) {
        loaded_models_->push_back(Model(
            cad_files[i].toStdString(), cad_models[i].toStdString(), "BLANK"));
    }
}

bool Viewer::are_models_loaded_correctly(int index) {
    return loaded_models_->at(index).initialized_correctly_;
}

bool Viewer::are_models_loaded_incorrectly(int index) {
    return !loaded_models_->at(index).initialized_correctly_;
}

void Viewer::change_model_opacity_to_original(int index) {
    model_actor_list_[index]->PickableOn();
    model_actor_list_[index]->VisibilityOn();
    model_actor_list_[index]->GetProperty()->SetRepresentationToSurface();
    model_actor_list_[index]->GetProperty()->SetAmbient(0);
    model_actor_list_[index]->GetProperty()->SetDiffuse(1);
    model_actor_list_[index]->GetProperty()->SetOpacity(1);
}

void Viewer::change_model_opacity_to_wire_frame(int index) {
    model_actor_list_[index]->PickableOn();
    model_actor_list_[index]->VisibilityOn();
    model_actor_list_[index]->GetProperty()->SetAmbient(0);
    model_actor_list_[index]->GetProperty()->SetDiffuse(1);
    model_actor_list_[index]->GetProperty()->SetRepresentationToWireframe();
    model_actor_list_[index]->GetProperty()->SetOpacity(1);
}

void Viewer::change_model_opacity_to_transparent(int index) {
    model_actor_list_[index]->PickableOn();
    model_actor_list_[index]->VisibilityOn();
    model_actor_list_[index]->GetProperty()->SetRepresentationToSurface();
    model_actor_list_[index]->GetProperty()->SetAmbient(1);
    model_actor_list_[index]->GetProperty()->SetDiffuse(0);
    model_actor_list_[index]->GetProperty()->SetOpacity(0.2);
}

void Viewer::change_model_opacity_to_solid(int index) {
    model_actor_list_[index]->PickableOn();
    model_actor_list_[index]->VisibilityOn();
    model_actor_list_[index]->GetProperty()->SetRepresentationToSurface();
    model_actor_list_[index]->GetProperty()->SetAmbient(1);
    model_actor_list_[index]->GetProperty()->SetDiffuse(0);
    model_actor_list_[index]->GetProperty()->SetOpacity(1);
}

void Viewer::set_model_position_at_index(
    int index, double x, double y, double z) {
    model_actor_list_[index]->SetPosition(x, y, z);
}

void Viewer::set_model_orientation_at_index(
    int index, double xrot, double yrot, double zrot) {
    model_actor_list_[index]->SetOrientation(xrot, yrot, zrot);
}

std::string
Viewer::print_location_and_orientation_of_model_at_index(int index) {
    std::string infoText = "Location: <";
    infoText += std::to_string(static_cast<long double>(
                    model_actor_list_[index]->GetPosition()[0])) +
                "," +
                std::to_string(static_cast<long double>(
                    model_actor_list_[index]->GetPosition()[1])) +
                "," +
                std::to_string(static_cast<long double>(
                    model_actor_list_[index]->GetPosition()[2])) +
                ">\nOrientation: <" +
                std::to_string(static_cast<long double>(
                    model_actor_list_[index]->GetOrientation()[0])) +
                "," +
                std::to_string(static_cast<long double>(
                    model_actor_list_[index]->GetOrientation()[1])) +
                "," +
                std::to_string(static_cast<long double>(
                    model_actor_list_[index]->GetOrientation()[2])) +
                ">";

    return infoText;
}

void Viewer::set_actor_text(std::string desired_text) {
    actor_text_->SetInput(desired_text.c_str());
}

void Viewer::set_actor_text_color_to_model_color_at_index(int index) {
    actor_text_->GetTextProperty()->SetColor(
        model_actor_list_[index]->GetProperty()->GetColor());
}

void Viewer::render_scene() {
    background_renderer_->Render();
    scene_renderer_->Render();
}

void Viewer::display_actors_in_renderer() {
    background_renderer_->GetActors()->Print(std::cout);
}

void Viewer::set_render_window_and_display() {
    render_window_->AddRenderer(background_renderer_);
    render_window_->SetWindowName("My Window");
    render_window_->Render();

    vtkSmartPointer<vtkRenderWindowInteractor> interactor =
        vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(render_window_);
    render_window_->Render();
    interactor->Start();
}

double* Viewer::get_model_orientation_at_index(int index) {
    return model_actor_list_[index]->GetOrientation();
}

double* Viewer::get_model_position_at_index(int index) {
    return model_actor_list_[index]->GetPosition();
}

void Viewer::make_model_invisible_and_nonpickable_at_index(int index) {
    model_actor_list_[index]->PickableOff();
    model_actor_list_[index]->VisibilityOff();
}

void Viewer::make_model_visible_and_pickable_at_index(int index) {
    model_actor_list_[index]->PickableOn();
    model_actor_list_[index]->VisibilityOn();
}
void Viewer::make_all_models_invisible() {
    for (auto model : model_actor_list_) {
        model->PickableOff();
        model->VisibilityOff();
    }
}

std::shared_ptr<std::vector<Model>> Viewer::get_loaded_models() {
    return loaded_models_;
}

void Viewer::load_renderers_into_render_window(Calibration cal) {
    // Shared pipeline recipe (006 U4): layered renderers (layer 0 =
    // background, interactive off; layer 1 = scene, interactive on;
    // 2 layers + AddRenderer x2) + the scene-camera position (0,0,0) and
    // near-origin focal pivot (0,0,focal_dir) — the widgets' value of
    // camera divergence (b) (the QML side pins the focal at the primary
    // model's z).
    jta::render_pipeline::SetupLayeredRenderers(
        qvtk_render_window_, background_renderer_, scene_renderer_);
    float focal_dir;
    // This was setting focal_dir to 1 if cal.type_ was denver or any others -
    // decided to remove the dever else statement in response to that. Sets
    // focal_dir = -1 if "UF", else sets to 1
    focal_dir = (cal.type_ == "UF") ? -1 : 1;
    jta::render_pipeline::SetupSceneCameraFocal(scene_renderer_, focal_dir);
}

void Viewer::print_render_window() {
    qvtk_render_window_->Print(std::cout);
}

void Viewer::make_actor_text_invisible() {
    vtkTextActor::SafeDownCast(actor_text_)->GetTextProperty()->SetOpacity(0.0);
}

void Viewer::make_actor_text_visible() {
    vtkTextActor::SafeDownCast(actor_text_)->GetTextProperty()->SetOpacity(1.0);
}

void Viewer::load_in_interactor_style(
    vtkSmartPointer<vtkInteractorStyleTrackballActor> in) {
    render_window_interactor_->SetInteractorStyle(in);
}

int Viewer::model_actor_list_size() {
    return model_actor_list_.size();
}

vtkActor* Viewer::get_model_actor_at_index(int index) {
    return model_actor_list_[index].GetPointer();
}

void Viewer::print_interactor_information() {
    render_window_interactor_->Print(std::cout);
    int num_ren = render_window_interactor_->GetRenderWindow()
                      ->GetRenderers()
                      ->GetNumberOfItems();
}

vtkSmartPointer<vtkRenderWindowInteractor> Viewer::get_interactor() {
    return render_window_interactor_;
}

void Viewer::set_vtk_camera_from_calibration_and_image_size_if_jta(
    Calibration cal, int w, int h) {
    float cx = w / 2 - cal.camera_A_principal_.cx();
    float cy = h / 2 + cal.camera_A_principal_.cy();
    float fx = cal.camera_A_principal_.fx();
    float fy = cal.camera_A_principal_.fy();

    std::cout << cx << ", " << cy << std::endl;

    calculate_and_set_window_center_from_calibration(w, h, cx, cy);
    // Shared pipeline recipe (006 U4): scene view angle + clipping
    // (0.1*fx, 1.75*fx). The angle stays view-side math (the pinned atan2
    // formula); window-center + aspect calibration plumbing stays view-side
    // (camera divergence (c)).
    jta::render_pipeline::ApplySceneCameraProjection(
        scene_renderer_, ViewingAngleFromCalibration(h, fy), fx);
    calculate_and_set_camera_aspect_from_calibration(fx, fy);
}

void Viewer::set_vtk_camera_from_calibration_and_image_if_camera_matrix(
    Calibration cal, int w, int h) {
    float cx = cal.camera_A_principal_.cx();
    float cy = cal.camera_A_principal_.cy();
    float fx = cal.camera_A_principal_.fx();
    float fy = cal.camera_A_principal_.fy();

    calculate_and_set_window_center_from_calibration(w, h, cx, cy);
    jta::render_pipeline::ApplySceneCameraProjection(
        scene_renderer_, ViewingAngleFromCalibration(h, fy), fx);
    calculate_and_set_camera_aspect_from_calibration(fx, fy);
    scene_camera_->SetViewUp(0, -1, 0);
}

void Viewer::calculate_and_set_window_center_from_calibration(
    const int w, const int h, const float cx, const float cy) {
    this->wcx = -(2 * cx - w) / w;
    this->wcy = (2 * cy - h) / h;

    this->windowCenter = true;
    scene_camera_->SetWindowCenter(wcx, wcy);
}

bool Viewer::windowCenterSet() {
    return this->windowCenter;
}

void Viewer::update_window_center_on_resize() {
    // render window width and height
    float rww = qvtk_render_window_->GetSize()[0];
    float rwh = qvtk_render_window_->GetSize()[1];

    // ternary to calculate whether to expand in the x or y directions
    rww > rwh
        ? scene_camera_->SetWindowCenter(this->wcx * (rwh / rww), this->wcy)
        : scene_camera_->SetWindowCenter(this->wcx, this->wcy * (rww / rwh));
}

void Viewer::calculate_and_set_camera_aspect_from_calibration(
    const float fx, const float fy) {
    vtkSmartPointer<vtkMatrix4x4> m = vtkSmartPointer<vtkMatrix4x4>::New();
    m->Identity();
    double aspect = fx / fy;
    m->SetElement(0, 0, aspect);

    vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New();

    t->SetMatrix(m);
    scene_camera_->SetUserTransform(t);

    vtkMatrix4x4* mat =
        scene_camera_->GetProjectionTransformMatrix(scene_renderer_);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << mat->GetElement(i, j) << ", ";
        }
        std::cout << std::endl;
    }
}

void Viewer::print_scene_camera_directions() {
    double fp[3];
    double vu[3];
    double pn[3];
    double pn_img[3];
    scene_camera_->GetFocalPoint(fp);
    scene_camera_->GetViewUp(vu);
    scene_camera_->GetViewPlaneNormal(pn);
    background_camera_->GetViewPlaneNormal(pn_img);
}
