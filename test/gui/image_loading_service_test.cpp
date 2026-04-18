#include "gui/image_loading_service.h"

#include <filesystem>
#include <iostream>

namespace {

QString test_file(const std::filesystem::path& relative_path) {
    return QString::fromStdString((std::filesystem::path(JTML_TEST_DATA_DIR) / relative_path).string());
}

int fail(const char* message) {
    std::cerr << message << std::endl;
    return 1;
}

} // namespace

int main() {
    jta_gui::ImageLoadingService service;

    {
        auto result = service.LoadImages(
            {},
            {},
            {test_file(std::filesystem::path{"test_case"} / "HL_V1_K1_0001.tif")},
            {},
            3,
            0,
            150,
            6);

        if (result.loaded_count != 1) {
            return fail("expected one loaded image");
        }
        if (result.display_names.size() != 1 || result.display_names.front() != "HL_V1_K1_0001") {
            return fail("expected loaded image display name");
        }
        if (result.frames_a.size() != 1 || !result.frames_b.empty()) {
            return fail("expected one monoplane frame result");
        }
    }

    {
        std::vector<Model> existing_models;
        existing_models.push_back(Model(
            test_file(std::filesystem::path{"test_case"} / "KR_left_8_fem.stl").toStdString(),
            "KR_left_8_fem",
            "BLANK"));

        auto result = service.LoadModels(
            existing_models,
            {
                test_file(std::filesystem::path{"test_case"} / "KR_left_8_fem.stl"),
                test_file(std::filesystem::path{"test_case"} / "KR_left_8_fem.stl"),
            });

        if (result.loaded_count != 2) {
            return fail("expected two loaded models");
        }
        if (result.model_names.size() != 2 || result.model_names[0] != "KR_left_8_fem(2)" ||
            result.model_names[1] != "KR_left_8_fem(3)") {
            return fail("expected deduplicated model names");
        }
        if (result.models.size() != 2) {
            return fail("expected created model instances");
        }
        if (!result.models[0].initialized_correctly_ || !result.models[1].initialized_correctly_) {
            return fail("expected STL models to initialize correctly");
        }
    }

    return 0;
}
