// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#pragma once

#include <QAbstractListModel>
#include <QString>
#include <QVector>

// Image-list view-model (plan 004 U2, R4/R5).
//
// Owns the frame display names (write-once, display-only — never read back).
// The QListView renders them passively; selection lives in the view's
// QItemSelectionModel (model + selectionModel together are the
// headless-testable unit). QtCore-only, no widgets: testable under
// QCoreApplication with no display.
//
// Biplane multiline frame names ("A: <base>\nB: <base>") are preserved
// verbatim — the name passed to AppendFrame is stored as-is.
class FrameListModel : public QAbstractListModel {
    Q_OBJECT

public:
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole)
        const override;

    // Append one frame name (mirrors the old per-item addItem call; the load
    // slots insert inside their loop, so partial loads keep the rows appended
    // so far, exactly like the old goto stop / stop_biplane paths).
    void AppendFrame(const QString& name);

private:
    QVector<QString> names_;
};
