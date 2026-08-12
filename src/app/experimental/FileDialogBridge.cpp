// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "FileDialogBridge.h"

#include <QFileDialog>
#include <QFileInfo>
#include <QStandardPaths>

namespace {
// MRU cap for the sidebar bookmarks.
constexpr int kMaxMruDirs = 5;
}  // namespace

FileDialogBridge::FileDialogBridge(QObject* parent)
    : QObject(parent),
      settings_(QStringLiteral("JointTrackAutoGPU"),
                QStringLiteral("jtml_experimental")) {}

QStringList FileDialogBridge::getOpenFileNames(const QString& title,
                                               const QString& filter,
                                               const QString& startDir,
                                               const QString& purpose) {
    QString dir = startDir;
    if (dir.isEmpty()) {
        dir = lastDir(purpose);
    }
    if (dir.isEmpty()) {
        dir = QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
    }

    // DontUseNativeDialog: keep Qt's in-process dialog regardless of the
    // platform theme — the portal path is backend-dependent for multi-select
    // on this box (see the header comment).
    QFileDialog dialog(nullptr, title, dir, filter);
    dialog.setFileMode(QFileDialog::ExistingFiles);
    dialog.setOption(QFileDialog::DontUseNativeDialog, true);
    dialog.setSidebarUrls(mruUrls(purpose));

    if (dialog.exec() != QDialog::Accepted) {
        return {};
    }
    const QStringList files = dialog.selectedFiles();
    if (!files.isEmpty()) {
        rememberDir(QFileInfo(files.first()).absolutePath(), purpose);
    }
    return files;
}

void FileDialogBridge::rememberDir(const QString& dir, const QString& purpose) {
    if (dir.isEmpty()) {
        return;
    }
    settings_.setValue(QStringLiteral("dialogs/lastDir/") + purpose, dir);

    QStringList mru =
        settings_.value(QStringLiteral("dialogs/mru/") + purpose)
            .toStringList();
    mru.removeAll(dir);
    mru.prepend(dir);
    while (mru.size() > kMaxMruDirs) {
        mru.removeLast();
    }
    settings_.setValue(QStringLiteral("dialogs/mru/") + purpose, mru);
}

QString FileDialogBridge::lastDir(const QString& purpose) const {
    return settings_.value(QStringLiteral("dialogs/lastDir/") + purpose)
        .toString();
}

QList<QUrl> FileDialogBridge::mruUrls(const QString& purpose) const {
    const QStringList mru =
        settings_.value(QStringLiteral("dialogs/mru/") + purpose)
            .toStringList();
    QList<QUrl> urls;
    urls.reserve(mru.size());
    for (const QString& dir : mru) {
        urls.append(QUrl::fromLocalFile(dir));
    }
    return urls;
}
