// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "view/frame_list_model.h"

int FrameListModel::rowCount(const QModelIndex& parent) const {
    return parent.isValid() ? 0 : static_cast<int>(names_.size());
}

QVariant FrameListModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.row() < 0 ||
        index.row() >= static_cast<int>(names_.size())) {
        return QVariant();
    }
    if (role != Qt::DisplayRole) {
        return QVariant();
    }
    return names_.at(index.row());
}

void FrameListModel::AppendFrame(const QString& name) {
    const int row = static_cast<int>(names_.size());
    beginInsertRows(QModelIndex(), row, row);
    names_.push_back(name);
    endInsertRows();
}
