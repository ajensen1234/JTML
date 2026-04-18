// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*Implementation of Data Storage class for Direct*/
#include "core/direct_data_storage.h"

// Standard
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace std;

namespace {

constexpr double kInitialCenterCoordinate = 0.5;
constexpr double kInitialSideLength = 1.0;

Point6D InitialCenter() {
    return {
        kInitialCenterCoordinate,
        kInitialCenterCoordinate,
        kInitialCenterCoordinate,
        kInitialCenterCoordinate,
        kInitialCenterCoordinate,
        kInitialCenterCoordinate};
}

Point6D InitialSides() {
    return {
        kInitialSideLength,
        kInitialSideLength,
        kInitialSideLength,
        kInitialSideLength,
        kInitialSideLength,
        kInitialSideLength};
}

}  // namespace

DirectDataStorage::DirectDataStorage(double initial_value) {
    /*Create New Vector of HyperBoxes and New HyperBox @ (.5, .5, .5, .5, .5,
     * .5) with initial_value*/
    std::vector<std::unique_ptr<HyperBox6D>> initial_column;
    initial_column.push_back(std::make_unique<HyperBox6D>(
        initial_value, InitialCenter(), InitialSides()));
    const HyperBox6D* initial_hyperbox = initial_column.back().get();
    storage_matrix_.push_back(std::move(initial_column));

    /*Add to Minimum Containers*/
    minimum_value_columns_.push_back(initial_hyperbox->value_);
    size_columns_.push_back(initial_hyperbox->size_);
}

DirectDataStorage::DirectDataStorage() {
    /*Create New Vector of HyperBoxes and New HyperBox @ (.5, .5, .5, .5, .5,
     * .5) with initial value of -1*/
    std::vector<std::unique_ptr<HyperBox6D>> initial_column;
    initial_column.push_back(std::make_unique<HyperBox6D>(
        -1, InitialCenter(), InitialSides()));
    const HyperBox6D* initial_hyperbox = initial_column.back().get();
    storage_matrix_.push_back(std::move(initial_column));

    /*Add to Minimum Containers*/
    minimum_value_columns_.push_back(initial_hyperbox->value_);
    size_columns_.push_back(initial_hyperbox->size_);
}

// DirectDataStorage::~DirectDataStorage() {
//	/*Delete Contents of storage_matrix_ safely*/
//	DeleteStoredHyperboxes();
// }

void DirectDataStorage::DeleteAllStoredHyperboxes() {
    /*Clear Storage Matrix and Minimum Container Matrices*/
    storage_matrix_.clear();
    minimum_value_columns_.clear();
    size_columns_.clear();
}

struct HyperBoxGreaterThanSize {
    bool operator()(
        const std::vector<std::unique_ptr<HyperBox6D>>& old,
        double comparison) {
        return (comparison > old[0]->size_);
    }
};

struct HyperBoxLessThanValue {
    bool operator()(const std::unique_ptr<HyperBox6D>& old, double comparison) {
        return (comparison < old->value_);
    }
};

void DirectDataStorage::AddHyperBox(HyperBox6D new_box) {
    auto owned_box = std::make_unique<HyperBox6D>(new_box);
    const double new_box_value = owned_box->value_;
    const double new_box_size = owned_box->size_;

    /*Search for Correct Size, If Doesn't Exist Insert New*/
    auto iterator = std::lower_bound(  // NOLINT(modernize-use-ranges)
        storage_matrix_.begin(),
        storage_matrix_.end(),
        new_box_size,
        HyperBoxGreaterThanSize());
    const auto iterator_index =
        static_cast<std::size_t>(std::distance(storage_matrix_.begin(), iterator));
    const auto iterator_offset =
        static_cast<decltype(minimum_value_columns_)::difference_type>(iterator_index);

    /*IF in range*/
    if (iterator != storage_matrix_.end()) {
        /*If Already Exists, Insert in That Column*/
        if (iterator->at(0)->size_ == new_box_size) {
            auto column_iterator = std::lower_bound(  // NOLINT(modernize-use-ranges)
                iterator->begin(),
                iterator->end(),
                new_box_value,
                HyperBoxLessThanValue());

            /*IF in range, insert at column_iterator*/
            if (column_iterator != iterator->end()) {
                iterator->insert(column_iterator, std::move(owned_box));
            } else {
                /*Add New HyperBox At End*/
                iterator->push_back(std::move(owned_box));

                /*Replace Minimum Containers (NOT SAFE)*/
                minimum_value_columns_[iterator_index] = new_box_value;
                size_columns_[iterator_index] = new_box_size;
            }
        } else {
            /*The Index Instead Points where To insert a new column*/
            std::vector<std::unique_ptr<HyperBox6D>> new_column;
            new_column.push_back(std::move(owned_box));
            storage_matrix_.insert(iterator, std::move(new_column));

            /*Insert Minimum Containers (NOT SAFE)*/
            minimum_value_columns_.insert(
                minimum_value_columns_.begin() + iterator_offset, new_box_value);
            size_columns_.insert(
                size_columns_.begin() + iterator_offset, new_box_size);
        }
    } else {
        /*Add New Column At End*/
        std::vector<std::unique_ptr<HyperBox6D>> new_column;
        new_column.push_back(std::move(owned_box));
        storage_matrix_.push_back(std::move(new_column));

        /*Add to Minimum Containers*/
        minimum_value_columns_.push_back(new_box_value);
        size_columns_.push_back(new_box_size);
    }
}

void DirectDataStorage::DeleteHyperBoxes(const std::vector<int>& col_ids) {
    /*CAN ASSUME col_ids IS SORTED IN DECREASING ORDER*/

    /*Variable for Current Col ID*/
    int col_id = -1;

    /*Scroll Through All Ids*/
    for (const int current_col_id : col_ids) {
        col_id = current_col_id;
        /*Check if Valid Column ID*/
        if (col_id >= 0 && static_cast<std::size_t>(col_id) < storage_matrix_.size()) {
            const auto column_index = static_cast<std::size_t>(col_id);
            const auto column_offset = static_cast<decltype(storage_matrix_)::difference_type>(column_index);
            /*Delete HyperBox*/
            if (!storage_matrix_[column_index].empty()) {
                storage_matrix_[column_index]
                    .pop_back(); /*At End Because Min Value is at the End!*/
            }

            /*If Column Is Now Empty, Delete It*/
            if (storage_matrix_[column_index].empty()) {
                storage_matrix_.erase(storage_matrix_.begin() + column_offset);

                /*Delete Place in Minimum Containers (NOT SAFE)*/
                minimum_value_columns_.erase(
                    minimum_value_columns_.begin() + column_offset);
                size_columns_.erase(size_columns_.begin() + column_offset);
            } else {
                /*Reset Minimum Value Container (NOT SAFE)*/
                minimum_value_columns_[column_index] =
                    storage_matrix_[column_index].back()->value_;
            }
        }
    }
}

unsigned int DirectDataStorage::GetNumberColumns() {
    return storage_matrix_.size();
}

int DirectDataStorage::GetLowestFValColId() {
    double min_fval = DBL_MAX;
    int col_id = -1;
    for (int i = 0; i < minimum_value_columns_.size(); i++) {
        if (minimum_value_columns_[i] < min_fval) {
            min_fval = minimum_value_columns_[i];
            col_id = i;
        }
    }
    return col_id;
}

double DirectDataStorage::GetFValAtColId(int col_id) {
    if (col_id >= 0 && col_id < minimum_value_columns_.size()) {
        return minimum_value_columns_[col_id];
    }
    return DBL_MAX;
}

double DirectDataStorage::GetSizeAtColId(int col_id) {
    if (col_id >= 0 && col_id < size_columns_.size()) {
        return size_columns_[col_id];
    }
    return -1;
}

HyperBox6D DirectDataStorage::GetLowestFValHyperBoxAtColId(int col_id) {
    if (col_id >= 0 && col_id < storage_matrix_.size()) {
        return *storage_matrix_[col_id].back();
    }
    return {};
}

void DirectDataStorage::RemoveHyperBoxAtColId(int col_id, HyperBox6D box) {
    if (col_id >= 0 && col_id < storage_matrix_.size()) {
        storage_matrix_[col_id].pop_back();
        if (storage_matrix_[col_id].empty()) {
            storage_matrix_.erase(storage_matrix_.begin() + col_id);
            minimum_value_columns_.erase(minimum_value_columns_.begin() + col_id);
            size_columns_.erase(size_columns_.begin() + col_id);
        } else {
            minimum_value_columns_[col_id] = storage_matrix_[col_id].back()->value_;
        }
    }
}

HyperBox6D DirectDataStorage::GetMinimumHyperbox(int col_id) {
    /*Return if in bounds*/
    if (col_id >= 0 && static_cast<std::size_t>(col_id) < storage_matrix_.size()) {
        return (*storage_matrix_[static_cast<std::size_t>(col_id)].back());
    }
    return {}; /*Return dummy value if error*/
}

void DirectDataStorage::PrintContents() {

    /*Print Column Headers*/
    std::cout << "\nColumn #:";
    std::size_t column_number = 0;
    for (const auto& column : storage_matrix_) {
        static_cast<void>(column);
        std::cout << "\t" << column_number++;
    }
    std::size_t maximum = 0;
    std::cout << "\nColumn Length:";
    for (const auto& column : storage_matrix_) {
        std::cout << "\t" << column.size();
        maximum = std::max(maximum, column.size());
    }
    std::cout << "\nMinimum Value:";
    for (const double minimum_value : minimum_value_columns_) {
        std::cout << "\t" << minimum_value;
    }
    std::cout << "\nSize (Min):";
    for (const auto& column : storage_matrix_) {
        std::cout << "\t" << column[0]->size_;
    }
    std::cout << "\nSize (Matrix):";
    for (const auto& column : storage_matrix_) {
        std::cout << "\t" << column[0]->size_;
    }

    /*Print Matrix*/
    for (std::size_t j = 0; j < maximum; j++) {
        std::cout << "\n\t";
        for (const auto& column : storage_matrix_) {
            std::cout << "\t";
            if (j < column.size()) std::cout << column[j]->value_;
        }
    }
}
