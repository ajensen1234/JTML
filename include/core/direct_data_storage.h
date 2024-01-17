/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#ifndef DIRECT_DATA_STORAGE_H
#define DIRECT_DATA_STORAGE_H

/*Header for Data Storage Class of DIRECT algorithm (different (faster) than one
 * in JTA but doesnt support Lipschitz search)*/

/*6D Data Structures*/
#include "data_structures_6D.h"

// Standard
#include <string>
#include <vector>

class DirectDataStorage {
   public:
    /*Initialize Direct Storage with a Unit Size HyperBox @ (.5, .5, .5, .5, .5,
     * .5) with initial_value*/
    DirectDataStorage(double initial_value);
    DirectDataStorage(); /*Use Default Value of -1 For initial value*/
    //~DirectDataStorage();

    /*Remove and Add HyperBoxes*/
    void AddHyperBox(HyperBox6D *new_box);
    void DeleteHyperBoxes(
        std::vector<int> col_ids); /*Deletes the Best Hyperbox at the List of
                                      Column IDs and Deletes Empty Columns*/

    /*Get Number of Columns*/
    unsigned int GetNumberColumns();

    /*Get smallest Fvalue (last one) in column*/
    HyperBox6D GetMinimumHyperbox(int col_id);

    /*Get Smallest Fvalue (Last one) in Column*/
    double GetMinimumHyperboxValue(int col_id);

    /*Get Hyperbox Size Stored Column*/
    double GetSizeStoredInColumn(int col_id);

    /*Delete Contents of storage_matrix_ safely (also called in destructor)*/
    void DeleteAllStoredHyperboxes();

    /*Print Columns, Min/Max/Avg Column Length*/
    void PrintSize();
    /*Print */
    void PrintContents();

   private:
    /*Vector of Vector of HyperBoxes:
    Low Level Vector of HyperBoxes Represents All Hyperboxes of a Given Size,
    Kept in Sorted Decreasing (Max Value @ 0) Order High Level Vector of Vectors
    Represents All Sizes of Current Vectors, Kept in Sorted Increasing (Min
    Value @ 0) Order*/
    std::vector<std::vector<HyperBox6D *> *> storage_matrix_;

    /*Vectors for Storing Minimum Hyperbox Size/Function Value Respectively for
    Each Column. This is done for speed as it is much faster to access. Might
    need to reserve space in constructor. */
    std::vector<double> minimum_value_columns_;
    std::vector<double> size_columns_;
};

#endif /* DIRECT_DATA_STORAGE_H */
