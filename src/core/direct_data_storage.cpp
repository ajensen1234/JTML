// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*Implementation of Data Storage class for Direct*/
#include "core/direct_data_storage.h"

// Standard
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

DirectDataStorage::DirectDataStorage(double initial_value) {
    /*Create New Vector of HyperBoxes and New HyperBox @ (.5, .5, .5, .5, .5,
     * .5) with initial_value*/
    auto initial_column = new std::vector<HyperBox6D*>();
    auto initial_hyperbox =
        new HyperBox6D(initial_value, Point6D(.5, .5, .5, .5, .5, .5),
                       Point6D(1, 1, 1, 1, 1, 1));
    initial_column->push_back(initial_hyperbox);
    storage_matrix_.push_back(initial_column);

    /*Add to Minimum Containers*/
    minimum_value_columns_.push_back(initial_hyperbox->value_);
    size_columns_.push_back(initial_hyperbox->size_);
}

DirectDataStorage::DirectDataStorage() {
    /*Create New Vector of HyperBoxes and New HyperBox @ (.5, .5, .5, .5, .5,
     * .5) with initial value of -1*/
    auto initial_column = new std::vector<HyperBox6D*>();
    auto initial_hyperbox = new HyperBox6D(-1, Point6D(.5, .5, .5, .5, .5, .5),
                                           Point6D(1, 1, 1, 1, 1, 1));
    initial_column->push_back(initial_hyperbox);
    storage_matrix_.push_back(initial_column);

    /*Add to Minimum Containers*/
    minimum_value_columns_.push_back(initial_hyperbox->value_);
    size_columns_.push_back(initial_hyperbox->size_);
}

// DirectDataStorage::~DirectDataStorage() {
//	/*Delete Contents of storage_matrix_ safely*/
//	DeleteStoredHyperboxes();
// }

void DirectDataStorage::DeleteAllStoredHyperboxes() {
    /*Delete Memory Allocated by New*/
    for (int i = 0; i < storage_matrix_.size(); i++) {
        /*Delete Hyperboxes*/
        for (int j = 0; j < storage_matrix_[i]->size(); j++) {
            delete (*storage_matrix_[i])[j];
        }
        /*Delete Columns of Hyperboxes*/
        delete storage_matrix_[i];
    }

    /*Clear Storage Matrix and Minimum Container Matrices*/
    storage_matrix_.clear();
    minimum_value_columns_.clear();
    size_columns_.clear();
}

struct HyperBoxGreaterThanSize {
    bool operator()(const std::vector<HyperBox6D*>* old, double comparison) {
        return (comparison > (*old)[0]->size_);
    }
};

struct HyperBoxLessThanValue {
    bool operator()(const HyperBox6D* old, double comparison) {
        return (comparison < old->value_);
    }
};

void DirectDataStorage::AddHyperBox(HyperBox6D* new_box) {
    /*Search for Correct Size, If Doesn't Exist Insert New*/
    auto iterator =
        std::lower_bound(storage_matrix_.begin(), storage_matrix_.end(),
                         new_box->size_, HyperBoxGreaterThanSize());
    int iterator_index = std::distance(storage_matrix_.begin(), iterator);

    /*IF in range*/
    if (iterator != storage_matrix_.end()) {
        /*If Already Exists, Insert in That Column*/
        if ((*iterator)->at(0)->size_ == new_box->size_) {
            auto column_iterator =
                std::lower_bound((*iterator)->begin(), (*iterator)->end(),
                                 new_box->value_, HyperBoxLessThanValue());

            /*IF in range, insert at column_iterator*/
            if (column_iterator != (*iterator)->end()) {
                (*iterator)->insert(column_iterator, new_box);
            } else {
                /*Add New HyperBox At End*/
                (*iterator)->push_back(new_box);

                /*Replace Minimum Containers (NOT SAFE)*/
                minimum_value_columns_[iterator_index] = new_box->value_;
                size_columns_[iterator_index] = new_box->size_;
            }
        } else {
            /*The Index Instead Points where To insert a new column*/
            auto new_column = new std::vector<HyperBox6D*>();
            new_column->push_back(new_box);
            storage_matrix_.insert(iterator, new_column);

            /*Insert Minimum Containers (NOT SAFE)*/
            minimum_value_columns_.insert(
                minimum_value_columns_.begin() + iterator_index,
                new_box->value_);
            size_columns_.insert(size_columns_.begin() + iterator_index,
                                 new_box->size_);
        }
    } else {
        /*Add New Column At End*/
        auto new_column = new std::vector<HyperBox6D*>();
        new_column->push_back(new_box);
        storage_matrix_.push_back(new_column);

        /*Add to Minimum Containers*/
        minimum_value_columns_.push_back(new_box->value_);
        size_columns_.push_back(new_box->size_);
    }
}

void DirectDataStorage::DeleteHyperBoxes(std::vector<int> col_ids) {
    /*CAN ASSUME col_ids IS SORTED IN DECREASING ORDER*/

    /*Variable for Current Col ID*/
    int col_id = -1;

    /*Scroll Through All Ids*/
    for (int i = 0; i < col_ids.size(); i++) {
        col_id = col_ids[i];
        /*Check if Valid Column ID*/
        if (col_id < storage_matrix_.size() && col_id >= 0) {
            /*Delete HyperBox*/
            if (storage_matrix_[col_id]->size() > 0) {
                delete storage_matrix_[col_id]
                    ->back(); /*Delete Pointer To Last HyperBox*/
                storage_matrix_[col_id]
                    ->pop_back(); /*At End Because Min Value is at the End!*/
            }

            /*If Column Is Now Empty, Delete It*/
            if (storage_matrix_[col_id]->size() == 0) {
                delete storage_matrix_[col_id]; /*Delete Pointer*/
                storage_matrix_.erase(storage_matrix_.begin() + col_id);

                /*Delete Place in Minimum Containers (NOT SAFE)*/
                minimum_value_columns_.erase(minimum_value_columns_.begin() +
                                             col_id);
                size_columns_.erase(size_columns_.begin() + col_id);
            } else {
                /*Reset Minimum Value Container (NOT SAFE)*/
                minimum_value_columns_[col_id] =
                    storage_matrix_[col_id]->back()->value_;
            }
        }
    }
}

unsigned int DirectDataStorage::GetNumberColumns() {
    return storage_matrix_.size();
}

HyperBox6D DirectDataStorage::GetMinimumHyperbox(int col_id) {
    /*Return if in bounds*/
    if (col_id < storage_matrix_.size() && col_id >= 0) {
        return (*storage_matrix_[col_id]->back());
    }
    return HyperBox6D(); /*Return dummy value if error*/
}

double DirectDataStorage::GetMinimumHyperboxValue(int col_id) {
    /*Return if in bounds*/
    if (col_id < minimum_value_columns_.size() && col_id >= 0) {
        return minimum_value_columns_[col_id];
    }
    return -1; /*Return -1 value if error*/
}

double DirectDataStorage::GetSizeStoredInColumn(int col_id) {
    /*Return if in bounds*/
    if (col_id < size_columns_.size() && col_id >= 0) {
        return size_columns_[col_id];
    }
    return -1; /*Return -1 value if error*/
}

void DirectDataStorage::PrintSize() {
    /*Print Size*/
    std::cout << "Columns: " << storage_matrix_.size();

    /*Calculate Column Lengths Data*/
    int minimum = INT_MAX;
    int maximum = 0;
    int average = 0;
    for (int i = 0; i < storage_matrix_.size(); i++) {
        int current_size = (*storage_matrix_[i]).size();
        if (current_size <= minimum) minimum = current_size;
        if (current_size >= maximum) maximum = current_size;
        average += current_size;
    }

    /*Output Data*/
    std::cout << "\nColumn Length Minimum: " << minimum;
    std::cout << "\nColumn Length Maximum: " << maximum;
    std::cout << "\nColumn Length Average: "
              << static_cast<double>(average) /
                     static_cast<double>(storage_matrix_.size())
              << std::endl;
}

void DirectDataStorage::PrintContents() {
    /*Print Column Headers*/
    std::cout << "\nColumn #:";
    for (int i = 0; i < storage_matrix_.size(); i++) {
        std::cout << "\t" << i;
    }
    int maximum = 0;
    std::cout << "\nColumn Length:";
    for (int i = 0; i < storage_matrix_.size(); i++) {
        std::cout << "\t" << (*storage_matrix_[i]).size();
        if ((*storage_matrix_[i]).size() >= maximum)
            maximum = (*storage_matrix_[i]).size();
    }
    std::cout << "\nMinimum Value:";
    for (int i = 0; i < minimum_value_columns_.size(); i++) {
        std::cout << "\t" << minimum_value_columns_[i];
    }
    std::cout << "\nSize (Min):";
    for (int i = 0; i < storage_matrix_.size(); i++) {
        std::cout << "\t" << (*storage_matrix_[i])[0]->size_;
    }
    std::cout << "\nSize (Matrix):";
    for (int i = 0; i < storage_matrix_.size(); i++) {
        std::cout << "\t" << (*storage_matrix_[i])[0]->size_;
    }

    /*Print Matrix*/
    for (int j = 0; j < maximum; j++) {
        std::cout << "\n\t";
        for (int i = 0; i < storage_matrix_.size(); i++) {
            std::cout << "\t";
            if (j < (*storage_matrix_[i]).size())
                std::cout << (*storage_matrix_[i])[j]->value_;
        }
    }
}
