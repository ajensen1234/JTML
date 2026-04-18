#ifndef SETTINGS_SERVICE_H
#define SETTINGS_SERVICE_H

#pragma once

#include "core/session_context.h"

namespace jta_gui {

struct EdgeDetectionSettings {
    int aperture = 3;
    int low_threshold = 40;
    int high_threshold = 120;
};

class SettingsService {
public:
    void LoadSettings(jta_core::SessionContext& session);
    void SaveSettings(const jta_core::SessionContext& session);

    void SaveEdgeDetectionSettings(int aperture, int low_threshold, int high_threshold);
    [[nodiscard]] const EdgeDetectionSettings& GetEdgeDetectionSettings() const;
    [[nodiscard]] bool WasFirstTimeLoading() const;

private:
    EdgeDetectionSettings edge_detection_settings_{};
    bool first_time_loading_ = false;
};

} // namespace jta_gui

#endif /* SETTINGS_SERVICE_H */
