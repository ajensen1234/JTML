// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "FileDialogBridge.h"

#include <QFileDialog>
#include <QStandardPaths>

FileDialogBridge::FileDialogBridge(QObject* parent) : QObject(parent) {}

QStringList FileDialogBridge::getOpenFileNames(const QString& title,
                                               const QString& filter,
                                               const QString& startDir) {
    const QString dir = startDir.isEmpty()
                            ? QStandardPaths::writableLocation(
                                  QStandardPaths::HomeLocation)
                            : startDir;
    // DontUseNativeDialog: keep Qt's in-process dialog regardless of the
    // platform theme — the portal path is backend-dependent for multi-select
    // on this box (see the header comment).
    return QFileDialog::getOpenFileNames(
        nullptr, title, dir, filter, nullptr,
        QFileDialog::DontUseNativeDialog);
}
