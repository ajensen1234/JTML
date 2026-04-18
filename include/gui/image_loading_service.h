#pragma once

#include <QObject>
#include <QStringList>

#include <vector>

#include "core/frame.h"
#include "core/model.h"

namespace jta_gui {

struct ImageLoadResult {
    std::vector<Frame> frames_a;
    std::vector<Frame> frames_b;
    QStringList display_names;
    int loaded_count = 0;
};

struct ModelLoadResult {
    QStringList model_names;
    std::vector<Model> models;
    int loaded_count = 0;
};

class ImageLoadingService : public QObject {
    Q_OBJECT

public:
    explicit ImageLoadingService(QObject* parent = nullptr);

    ImageLoadResult LoadImages(
        const std::vector<Frame>& existing_frames_a,
        const std::vector<Frame>& existing_frames_b,
        const QStringList& camera_a_files,
        const QStringList& camera_b_files,
        int aperture,
        int low_threshold,
        int high_threshold,
        int dilation);

    ModelLoadResult
    LoadModels(
        const std::vector<Model>& existing_models,
        const QStringList& cad_files);

Q_SIGNALS:
    void framesLoaded(int count);
    void modelsLoaded(int count);
    void error(QString message);
};

} // namespace jta_gui
