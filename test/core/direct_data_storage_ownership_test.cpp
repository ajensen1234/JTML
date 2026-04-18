#define private public
#include "core/direct_data_storage.h"
#undef private

#include <memory>
#include <type_traits>
#include <vector>

static_assert(
    std::is_same_v<
        decltype(DirectDataStorage::storage_matrix_),
        std::vector<std::vector<std::unique_ptr<HyperBox6D>>>>,
    "DirectDataStorage should own HyperBox6D instances with std::unique_ptr");

int main() {
    DirectDataStorage storage;
    storage.DeleteAllStoredHyperboxes();
    return storage.GetNumberColumns() == 0 && storage.storage_matrix_.empty() &&
                   storage.minimum_value_columns_.empty() &&
                   storage.size_columns_.empty()
               ? 0
               : 1;
}
