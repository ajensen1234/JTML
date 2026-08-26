// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#pragma once

#include <QAbstractListModel>
#include <QString>
#include <QVector>

// Model-list view-model (plan 004 U2, R4/R5).
//
// Owns the CAD-model display names (write-once, display-only) and reuses
// jta::ModelListBuilder::UniquifyModelNames for the two-pass dedup (including
// the mutated-name-rescan quirk: N identical inputs yield
// ["A(2)","A(3)","A"] for N=3). QtCore-only, no widgets: headless-testable.
class ModelListModel : public QAbstractListModel {
    Q_OBJECT

public:
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole)
        const override;

    // Uniquify new_names against the already-loaded names and append them.
    // Returns the unique display names (one per input, same order) so the
    // view can bind the VTK models to the same names. Mirrors the old
    // addItem loop + the inline UniquifyModelNames call in the load slot.
    QVector<QString> AppendModels(const QVector<QString>& new_names);

private:
    QVector<QString> names_;
};
