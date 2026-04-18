#include "gui/image_loading_service.h"

#include <QFileInfo>
#include <QSet>

namespace {

bool are_frame_sizes_compatible(
    const Frame& frame,
    const std::vector<Frame>& existing_frames,
    const std::vector<Frame>& pending_frames) {
    const auto frame_size = cv::Size(
        const_cast<Frame&>(frame).GetEdgeImage().cols,
        const_cast<Frame&>(frame).GetEdgeImage().rows);
    const auto matches = [&](const std::vector<Frame>& frames) {
        for (const auto& existing_frame : frames) {
            const auto existing_frame_size = cv::Size(
                const_cast<Frame&>(existing_frame).GetEdgeImage().cols,
                const_cast<Frame&>(existing_frame).GetEdgeImage().rows);
            if (frame_size != existing_frame_size) {
                return false;
            }
        }
        return true;
    };

    return matches(existing_frames) && matches(pending_frames);
}

QString make_unique_model_name(const QString& base_name, QSet<QString>& used_names) {
    if (!used_names.contains(base_name)) {
        used_names.insert(base_name);
        return base_name;
    }

    int duplicate_index = 2;
    QString candidate = base_name + "(" + QString::number(duplicate_index) + ")";
    while (used_names.contains(candidate)) {
        ++duplicate_index;
        candidate = base_name + "(" + QString::number(duplicate_index) + ")";
    }

    used_names.insert(candidate);
    return candidate;
}

} // namespace

namespace jta_gui {

ImageLoadingService::ImageLoadingService(QObject* parent) : QObject(parent) {}

ImageLoadResult ImageLoadingService::LoadImages(
    const std::vector<Frame>& existing_frames_a,
    const std::vector<Frame>& existing_frames_b,
    const QStringList& camera_a_files,
    const QStringList& camera_b_files,
    int aperture,
    int low_threshold,
    int high_threshold,
    int dilation) {
    ImageLoadResult result;

    const bool is_biplane = !camera_b_files.isEmpty();
    if (is_biplane && camera_a_files.size() != camera_b_files.size()) {
        emit error("Please Load the Same Number of Images for Each Camera!");
        return result;
    }

    for (int i = 0; i < camera_a_files.size(); ++i) {
        Frame frame_a(
            camera_a_files[i].toStdString(),
            aperture,
            low_threshold,
            high_threshold,
            dilation);

        if (!are_frame_sizes_compatible(frame_a, existing_frames_a, result.frames_a)) {
            emit error("Images Loaded Must Be The Same Size!");
            break;
        }

        if (is_biplane) {
            Frame frame_b(
                camera_b_files[i].toStdString(),
                aperture,
                low_threshold,
                high_threshold,
                dilation);

            if (!are_frame_sizes_compatible(frame_b, existing_frames_b, result.frames_b)) {
                emit error("Images Loaded Must Be The Same Size!");
                break;
            }

            result.frames_a.push_back(frame_a);
            result.frames_b.push_back(frame_b);
            result.display_names.push_back(
                "A: " + QFileInfo(camera_a_files[i]).baseName() + "\nB: " +
                QFileInfo(camera_b_files[i]).baseName());
            continue;
        }

        result.frames_a.push_back(frame_a);
        result.display_names.push_back(QFileInfo(camera_a_files[i]).baseName());
    }

    result.loaded_count = result.display_names.size();
    if (result.loaded_count > 0) {
        emit framesLoaded(result.loaded_count);
    }

    return result;
}

ModelLoadResult ImageLoadingService::LoadModels(
    const std::vector<Model>& existing_models, const QStringList& cad_files) {
    ModelLoadResult result;
    QSet<QString> used_names;

    for (const auto& existing_model : existing_models) {
        used_names.insert(QString::fromStdString(existing_model.model_name_));
    }

    for (const auto& cad_file : cad_files) {
        const QString unique_name =
            make_unique_model_name(QFileInfo(cad_file).baseName(), used_names);
        result.model_names.push_back(unique_name);
        result.models.emplace_back(
            cad_file.toStdString(), unique_name.toStdString(), "BLANK");
    }

    result.loaded_count = result.models.size();
    if (result.loaded_count > 0) {
        emit modelsLoaded(result.loaded_count);
    }

    return result;
}

} // namespace jta_gui
