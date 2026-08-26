// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "view/model_list_model.h"

#include <string>
#include <vector>

#include "domain/model_list_builder.h"

int ModelListModel::rowCount(const QModelIndex& parent) const {
    return parent.isValid() ? 0 : static_cast<int>(names_.size());
}

QVariant ModelListModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.row() < 0 ||
        index.row() >= static_cast<int>(names_.size())) {
        return QVariant();
    }
    if (role != Qt::DisplayRole) {
        return QVariant();
    }
    return names_.at(index.row());
}

QVector<QString> ModelListModel::AppendModels(
    const QVector<QString>& new_names) {
    std::vector<std::string> new_bases;
    new_bases.reserve(static_cast<size_t>(new_names.size()));
    for (const auto& n : new_names) {
        new_bases.push_back(n.toStdString());
    }
    std::vector<std::string> existing;
    existing.reserve(static_cast<size_t>(names_.size()));
    for (const auto& n : names_) {
        existing.push_back(n.toStdString());
    }
    // Two-pass dedup with the mutated-name rescan quirk, byte-identical to
    // the old inline MainScreen logic (R5 / R13).
    const std::vector<std::string> unique =
        jta::ModelListBuilder::UniquifyModelNames(new_bases, existing);

    QVector<QString> display;
    display.reserve(static_cast<int>(unique.size()));
    const int first = static_cast<int>(names_.size());
    if (!unique.empty()) {
        beginInsertRows(
            QModelIndex(), first, first + static_cast<int>(unique.size()) - 1);
        for (const auto& n : unique) {
            names_.push_back(QString::fromStdString(n));
            display.push_back(QString::fromStdString(n));
        }
        endInsertRows();
    }
    return display;
}
