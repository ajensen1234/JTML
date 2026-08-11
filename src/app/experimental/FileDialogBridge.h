// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 feedback (multi-select): Qt's native file dialog bridge.
//
// The QML FileDialog's native path routes through the xdg-desktop-portal
// FileChooser when the xdgdesktopportal theme is active. On this box (niri
// session) the portal dispatches FileChooser to the GTK backend, which does
// not honor multiple=true in practice — while the D-Bus trace proves Qt sends
// multiple=true, the dialog still single-selects. GTK/Qt apps multi-select
// because they use their in-process dialogs, not the portal.
//
// This bridge calls QFileDialog::getOpenFileNames with DontUseNativeDialog —
// Qt's own in-process native dialog (the same one every other Qt app uses):
// multi-select unconditionally (ctrl/shift-click), immune to portal backend
// dispatch, no system configuration involved.

#pragma once

#include <QObject>
#include <QStringList>

class FileDialogBridge : public QObject {
    Q_OBJECT

public:
    explicit FileDialogBridge(QObject* parent = nullptr);

    // Native multi-select open dialog; returns local file paths (or an empty
    // list on cancel). `startDir` empty => the user's home directory.
    Q_INVOKABLE QStringList getOpenFileNames(const QString& title,
                                             const QString& filter,
                                             const QString& startDir);
};
