// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "gui/calibration_service.h"

#include <QFile>
#include <QFileDialog>
#include <QListWidget>
#include <QRegExp>
#include <QTextStream>

#include <cmath>
#include <numbers>

#include <vtkCamera.h>
#include <vtkRenderer.h>

#include "gui/interactor.h"
#include "gui/viewer.h"

namespace {

constexpr int kUfMonoplaneTokenCount = 5;
constexpr int kUfBiplaneTokenCount = 21;
constexpr int kDenverTokenCount = 12;

constexpr int kUfPrincipalDistanceIndex = 1;
constexpr int kUfPrincipalXIndex = 2;
constexpr int kUfPrincipalYIndex = 3;
constexpr int kUfPixelPitchIndex = 4;
constexpr int kBiplanePrincipalDistanceIndex = 5;
constexpr int kBiplanePrincipalXIndex = 6;
constexpr int kBiplanePrincipalYIndex = 7;
constexpr int kBiplanePixelPitchIndex = 8;
constexpr int kOriginBXIndex = 9;
constexpr int kOriginBYIndex = 10;
constexpr int kOriginBZIndex = 11;
constexpr int kAxesB11Index = 12;
constexpr int kAxesB12Index = 13;
constexpr int kAxesB13Index = 14;
constexpr int kAxesB21Index = 15;
constexpr int kAxesB22Index = 16;
constexpr int kAxesB23Index = 17;
constexpr int kAxesB31Index = 18;
constexpr int kAxesB32Index = 19;
constexpr int kAxesB33Index = 20;
constexpr int kDenverFxIndex = 6;
constexpr int kDenverScaleIndex = 7;
constexpr int kDenverCxIndex = 8;
constexpr int kDenverFyIndex = 10;
constexpr int kDenverCyIndex = 11;

constexpr float kMinimumClippingRange = 0.1F;
constexpr float kClippingRangeMultiplier = 2.0F;
constexpr double kHalf = 0.5;
constexpr double kViewAngleMultiplier = 2.0;
constexpr double kDegreesPerRadian = 180.0 / std::numbers::pi_v<double>;

void ResetCalibrationState(jta_core::SessionContext& context) {
    context.calibrated_for_monoplane_viewport_ = false;
    context.calibrated_for_biplane_viewport_ = false;
}

bool HasRequiredTokens(const QStringList& input_list, int minimum_count) {
    return input_list.size() >= minimum_count;
}

} // namespace

namespace jta_gui {

CalibrationService::CalibrationService(
    Viewer* primary_viewer,
    Viewer* coronal_viewer,
    vtkRenderer* renderer,
    QListWidget* image_list_widget,
    QObject* parent) :
    QObject(parent),
    primary_viewer_(primary_viewer),
    coronal_viewer_(coronal_viewer),
    renderer_(renderer),
    image_list_widget_(image_list_widget) {}

void CalibrationService::LoadCalibration(
    QWidget* parent, jta_core::SessionContext& context) {
    const QString calibration_file_extension = QFileDialog::getOpenFileName(
        parent,
        tr("Load Calibration"),
        ".",
        tr("Calibration File (*.txt)"));

    if (calibration_file_extension.isEmpty()) {
        return;
    }

    QFile input_file(calibration_file_extension);
    if (!input_file.open(QIODevice::ReadOnly)) {
        return;
    }

    QTextStream in(&input_file);
    const QStringList input_list =
        in.readAll().split(QRegExp("[\r\n]|,|\t| "), Qt::SkipEmptyParts);

    if (input_list.isEmpty()) {
        ResetCalibrationState(context);
        emit error("Invalid Configuration File!");
        return;
    }

    QString error_message;
    bool loaded = false;
    if (input_list[0] == "JT_INTCALIB" || input_list[0] == "JTA_INTCALIB") {
        loaded = TryLoadUfMonoplane(input_list, context, error_message);
    } else if (input_list[0] == "JTA_INTCALIB_BIPLANE") {
        loaded = TryLoadUfBiplane(input_list, context, error_message);
    } else if (input_list[0] == "image") {
        loaded = TryLoadDenver(input_list, context, error_message);
    } else {
        error_message = "Invalid Configuration File!";
    }

    if (!loaded) {
        ResetCalibrationState(context);
        emit error(error_message);
        return;
    }

    interactor_calibration = context.calibration_file_;
    interactor_camera_B = false;

    ApplyCalibrationToViewers(context);
    emit calibrationLoaded(context.calibration_file_);
}

void CalibrationService::ApplyCalibrationToViewers(
    jta_core::SessionContext& context) {
    primary_viewer_->load_renderers_into_render_window(context.calibration_file_);
    coronal_viewer_->load_renderers_into_render_window(context.calibration_file_);

    if (context.calibrated_for_monoplane_viewport_) {
        primary_viewer_->setup_camera_calibration(context.calibration_file_);
        coronal_viewer_->setup_camera_calibration(context.calibration_file_);
        coronal_viewer_->setup_camera_coronal_plane();

        const int current_row =
            image_list_widget_ == nullptr ? -1 : image_list_widget_->currentRow();
        if (current_row >= 0 &&
            current_row < static_cast<int>(context.loaded_frames.size())) {
            Frame& frame = context.loaded_frames.at(current_row);
            primary_viewer_->place_image_actors_according_to_calibration(
                context.calibration_file_,
                frame.GetOriginalImage().cols,
                frame.GetOriginalImage().rows);
            renderer_->GetActiveCamera()->SetViewAngle(CalculateViewingAngle(
                context,
                frame.GetOriginalImage().rows,
                true));
        }
        return;
    }

    if (context.calibrated_for_biplane_viewport_) {
        renderer_->GetActiveCamera()->SetFocalPoint(
            0,
            0,
            -1 * context.calibration_file_.camera_A_principal_
                      .principal_distance_ /
                context.calibration_file_.camera_A_principal_.pixel_pitch_);
        renderer_->GetActiveCamera()->SetPosition(0, 0, 0);
        renderer_->GetActiveCamera()->SetClippingRange(
            kMinimumClippingRange,
            kClippingRangeMultiplier *
                context.calibration_file_.camera_A_principal_
                      .principal_distance_ /
                context.calibration_file_.camera_A_principal_.pixel_pitch_);
    }
}

double CalibrationService::CalculateViewingAngle(
    const jta_core::SessionContext& context,
    int height,
    bool camera_a) {
    if (camera_a) {
        const double y =
            static_cast<double>(height) *
                context.calibration_file_.camera_A_principal_.pixel_pitch_ * kHalf +
            std::abs(static_cast<double>(
                context.calibration_file_.camera_A_principal_.principal_y_));
        return kDegreesPerRadian * kViewAngleMultiplier *
               std::atan2(
                   y,
                   context.calibration_file_.camera_A_principal_
                       .principal_distance_);
    }

    const double y =
        static_cast<double>(height) *
            context.calibration_file_.camera_B_principal_.pixel_pitch_ * kHalf +
        std::abs(static_cast<double>(
            context.calibration_file_.camera_B_principal_.principal_y_));
    return kDegreesPerRadian * kViewAngleMultiplier *
           std::atan2(
               y,
               context.calibration_file_.camera_B_principal_.principal_distance_);
}

bool CalibrationService::TryLoadUfMonoplane(
    const QStringList& input_list,
    jta_core::SessionContext& context,
    QString& error_message) {
    if (!HasRequiredTokens(input_list, kUfMonoplaneTokenCount)) {
        error_message = "Invalid Configuration File!";
        return false;
    }

    if (input_list[kUfPixelPitchIndex].toFloat() == 0.0F) {
        error_message =
            "Pixel size (the last number in the calibration file) is "
            "specified as 0! This is impossible.";
        return false;
    }

    context.calibrated_for_monoplane_viewport_ = true;
    context.calibrated_for_biplane_viewport_ = false;

    const CameraCalibration principal_calibration_file(
        input_list[kUfPrincipalDistanceIndex].toFloat(),
        -1.0F * input_list[kUfPrincipalXIndex].toFloat(),
        -1.0F * input_list[kUfPrincipalYIndex].toFloat(),
        input_list[kUfPixelPitchIndex].toFloat());

    context.calibration_file_ = Calibration(principal_calibration_file);
    return true;
}

bool CalibrationService::TryLoadUfBiplane(
    const QStringList& input_list,
    jta_core::SessionContext& context,
    QString& error_message) {
    if (!HasRequiredTokens(input_list, kUfBiplaneTokenCount)) {
        error_message = "Invalid Configuration File!";
        return false;
    }

    if (input_list[kUfPixelPitchIndex].toFloat() == 0.0F ||
        input_list[kBiplanePixelPitchIndex].toFloat() == 0.0F) {
        error_message =
            "Pixel size (the last number in the calibration file) is "
            "specified as 0! This is impossible.";
        return false;
    }

    context.calibrated_for_monoplane_viewport_ = false;
    context.calibrated_for_biplane_viewport_ = true;

    const CameraCalibration principal_calibration_file_A(
        input_list[kUfPrincipalDistanceIndex].toFloat(),
        -1.0F * input_list[kUfPrincipalXIndex].toFloat(),
        -1.0F * input_list[kUfPrincipalYIndex].toFloat(),
        input_list[kUfPixelPitchIndex].toFloat());
    const CameraCalibration principal_calibration_file_B(
        input_list[kBiplanePrincipalDistanceIndex].toFloat(),
        -1.0F * input_list[kBiplanePrincipalXIndex].toFloat(),
        -1.0F * input_list[kBiplanePrincipalYIndex].toFloat(),
        input_list[kBiplanePixelPitchIndex].toFloat());
    const Vect_3 origin_B(
        input_list[kOriginBXIndex].toFloat(),
        input_list[kOriginBYIndex].toFloat(),
        input_list[kOriginBZIndex].toFloat());
    const Matrix_3_3 orthogonal_axes_B(
        input_list[kAxesB11Index].toFloat(),
        input_list[kAxesB12Index].toFloat(),
        input_list[kAxesB13Index].toFloat(),
        input_list[kAxesB21Index].toFloat(),
        input_list[kAxesB22Index].toFloat(),
        input_list[kAxesB23Index].toFloat(),
        input_list[kAxesB31Index].toFloat(),
        input_list[kAxesB32Index].toFloat(),
        input_list[kAxesB33Index].toFloat());

    context.calibration_file_ = Calibration(
        principal_calibration_file_A,
        principal_calibration_file_B,
        origin_B,
        orthogonal_axes_B);
    return true;
}

bool CalibrationService::TryLoadDenver(
    const QStringList& input_list,
    jta_core::SessionContext& context,
    QString& error_message) {
    if (!HasRequiredTokens(input_list, kDenverTokenCount)) {
        error_message = "Invalid Configuration File!";
        return false;
    }

    const CameraCalibration denver_calibration_A(
        input_list[kDenverFxIndex].toFloat(),
        input_list[kDenverScaleIndex].toFloat(),
        input_list[kDenverCxIndex].toFloat(),
        input_list[kDenverFyIndex].toFloat(),
        input_list[kDenverCyIndex].toFloat());

    context.calibrated_for_monoplane_viewport_ = true;
    context.calibrated_for_biplane_viewport_ = false;
    context.calibration_file_ = Calibration(denver_calibration_A, "Denver");
    return true;
}

} // namespace jta_gui
