#include "gui/worker_orchestrator.h"

#include <type_traits>
#include <utility>

static_assert(std::is_base_of_v<QObject, jta_gui::WorkerOrchestrator>);
static_assert(std::is_same_v<
              decltype(std::declval<jta_gui::WorkerOrchestrator&>().ConfigureSegmentation(
                  std::declval<jta_gui::SegmentationConfig>())),
              void>);
static_assert(std::is_same_v<
              decltype(std::declval<jta_gui::WorkerOrchestrator&>().StartSegmentation(
                  std::declval<jta_core::SessionContext&>())),
              void>);

int main() {
    return 0;
}
