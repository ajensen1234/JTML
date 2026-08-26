// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#pragma once

#include <string>
#include <vector>

// Pure load-path list builder (plan U7, U10 / R10).
//
// Owns the widget-free name-uniquification and size-consistency logic that
// used to live inline in MainScreen::on_load_model_button_clicked /
// on_load_image_button_clicked. It takes plain std::string values (no Qt, no
// widgets), so it is headless-testable, and returns the display names / a
// size predicate. The view keeps only addItem(...) and the VTK/color binding.
namespace jta {

class ModelListBuilder {
public:
    // Uniquify CAD model display names exactly as MainScreen did (R15: no
    // behavior change). mirrows the two passes in the original: first de-dup
    // within the newly-loaded set (each duplicate gets a "(N)" suffix, with
    // the scan restarting on the *mutated* names, so N identical inputs yield
    // ["A(2)","A(3)","A"] for N=3), then de-dup against the existing loaded
    // names. new_names are the base names of the files being added;
    // existing_names are the already-loaded model names. Returns the ordered
    // unique display names (one per new file).
    static std::vector<std::string> UniquifyModelNames(
        const std::vector<std::string>& new_names,
        const std::vector<std::string>& existing_names);

    // True iff every loaded edge-image size in sizes matches (w,h). Used by
    // the image-load size-consistency gate (monoplane + biplane).
    static bool
    AllSameSize(int w, int h, const std::vector<std::pair<int, int>>& sizes);
};

}  // namespace jta
