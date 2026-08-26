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
// This bridge calls QFileDialog with DontUseNativeDialog — Qt's own
// in-process native dialog (the same one every other Qt app uses):
// multi-select unconditionally (ctrl/shift-click), immune to portal backend
// dispatch, no system configuration involved.
//
// Plan 007 (owner request 2026-08-12): the dialog remembers its directory
// per purpose (QSettings, org "JointTrackAutoGPU" / app "jtml_experimental"
// — deliberately NOT the SettingsService registry scope so the
// oracle-pinned registry keys stay untouched) and reopens there instead of
// $HOME; a per-purpose MRU of the last 5 directories appears in the dialog's
// sidebar as bookmarks. Paste-to-jump is stock QFileDialog behavior: paste
// a full path into the "File name:" field and press Enter. A full QML
// path-bar picker (copyable Location field) is the scheduled follow-up.

#pragma once

#include <QObject>
#include <QSettings>
#include <QStringList>
#include <QUrl>

class FileDialogBridge : public QObject {
    Q_OBJECT

public:
    explicit FileDialogBridge(QObject* parent = nullptr);

    // Native multi-select open dialog; returns local file paths (or an empty
    // list on cancel). `startDir` non-empty overrides the remembered
    // directory; `purpose` keys the remembered directory + sidebar MRU
    // (e.g. "images", "models" — anything stable).
    Q_INVOKABLE QStringList getOpenFileNames(
        const QString& title,
        const QString& filter,
        const QString& startDir,
        const QString& purpose = QStringLiteral("general"));

    // Directory-memory surface (public so the headless suite can pin the
    // QSettings round-trip without exec'ing a dialog — review fix
    // ce-code-review 2026-08-12, AC1). Not Q_INVOKABLE: the dialog flow is
    // the only QML entry point.
    // Remember `dir` as the purpose's last-used directory + MRU front.
    void rememberDir(const QString& dir, const QString& purpose);
    // The purpose's remembered start directory (empty if never set).
    QString lastDir(const QString& purpose) const;
    // The purpose's MRU directories as sidebar bookmark URLs.
    QList<QUrl> mruUrls(const QString& purpose) const;

private:
    QSettings settings_;
};
