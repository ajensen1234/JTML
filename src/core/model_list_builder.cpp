// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "core/model_list_builder.h"

#include <string>

namespace jta {

std::vector<std::string> ModelListBuilder::UniquifyModelNames(
    const std::vector<std::string>& new_names,
    const std::vector<std::string>& existing_names) {
    // Mirror MainScreen::on_load_model_button_clicked exactly (R15: no
    // behavior change). The scan restarts (j = -1) whenever a match is found,
    // incrementing already_exists against the *mutated* temp name, so N
    // identical inputs yield ["A(2)","A(3)","A"] for N=3.
    std::vector<std::string> result = new_names;

    // Pass 1: de-dup within the newly-loaded set.
    for (size_t i = 0; i < result.size(); ++i) {
        std::string temp_name = result[i];
        int already_exists = 1;
        for (size_t j = 0; j < result.size(); ++j) {
            if (result[j] == temp_name && j != i) {
                j = -1;  // restart scan against the mutated name
                already_exists++;
                temp_name = new_names[i] + "(" + std::to_string(already_exists) +
                            ")";
            }
        }
        result[i] = temp_name;
    }

    // Pass 2: de-dup against the already-loaded model names.
    for (size_t i = 0; i < result.size(); ++i) {
        std::string temp_name = result[i];
        int already_exists = 1;
        for (size_t j = 0; j < existing_names.size(); ++j) {
            if (existing_names[j] == temp_name) {
                j = -1;  // restart scan against the mutated name
                already_exists++;
                temp_name = result[i] + "(" + std::to_string(already_exists) +
                            ")";
            }
        }
        result[i] = temp_name;
    }

    return result;
}

bool ModelListBuilder::AllSameSize(
    int w, int h, const std::vector<std::pair<int, int>>& sizes) {
    for (const auto& s : sizes) {
        if (s.first != w || s.second != h) {
            return false;
        }
    }
    return true;
}

}  // namespace jta
