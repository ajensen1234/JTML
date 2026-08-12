// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 007 review round (ce-code-review 2026-08-12, AC1): FileDialogBridge
// directory-memory pins. Deterministic Catch2 headless tests, QtCore-only —
// no dialog is exec'd (getOpenFileNames is covered by the owner's
// manual-visual leg). XDG_CONFIG_HOME is redirected to a QTemporaryDir
// BEFORE constructing the bridge (the QSettings member resolves the config
// path at construction), so nothing touches the user's real config.
//
// NOTE (review fix): the pins live in ONE test case — QSettings/
// QStandardPaths cache the resolved config path per process, so per-case
// env redirects silently fall back to the first temp dir (or the real
// config) and leak state across cases.
//
// Pins:
//  - rememberDir -> lastDir round-trip per purpose;
//  - purpose-bucket isolation (images vs models vs general);
//  - MRU order + dedupe + the kMaxMruDirs=5 cap.

#include "FileDialogBridge.h"

#include <QByteArray>
#include <QTemporaryDir>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("file_dialog: directory memory (round-trip, isolation, MRU cap)",
          "[experimental][file_dialog]") {
    QTemporaryDir dir;
    qputenv("XDG_CONFIG_HOME", dir.path().toUtf8());
    FileDialogBridge bridge;

    // --- Fresh state: nothing remembered yet ----------------------------
    CHECK(bridge.lastDir("images").isEmpty());
    CHECK(bridge.lastDir("models").isEmpty());
    CHECK(bridge.lastDir("general").isEmpty());

    // --- Round-trip per purpose ------------------------------------------
    bridge.rememberDir("/data/studies/Kneel_1", "images");
    CHECK(bridge.lastDir("images") == QStringLiteral("/data/studies/Kneel_1"));
    bridge.rememberDir("/data/studies/Kneel_2", "images");
    CHECK(bridge.lastDir("images") == QStringLiteral("/data/studies/Kneel_2"));

    // --- Purpose buckets are isolated ------------------------------------
    bridge.rememberDir("/data/models", "models");
    CHECK(bridge.lastDir("models") == QStringLiteral("/data/models"));
    // The default bucket stays untouched.
    CHECK(bridge.lastDir("general").isEmpty());
    // The MRU lists do not bleed across purposes.
    CHECK(bridge.mruUrls("images").size() == 2);
    CHECK(bridge.mruUrls("models").size() == 1);
    CHECK(bridge.mruUrls("general").isEmpty());

    // --- MRU order, dedupe, and the 5-entry cap --------------------------
    bridge.rememberDir("/d3", "images");
    bridge.rememberDir("/d4", "images");
    bridge.rememberDir("/d5", "images");
    bridge.rememberDir("/d6", "images");
    // The 6th distinct entry evicts the oldest (Kneel_1).
    auto urls = bridge.mruUrls("images");
    REQUIRE(urls.size() == 5);
    CHECK(urls[0].toLocalFile() == QStringLiteral("/d6"));
    CHECK(urls[1].toLocalFile() == QStringLiteral("/d5"));
    CHECK(urls[4].toLocalFile() == QStringLiteral("/data/studies/Kneel_2"));

    // Dedupe: re-remembering an existing entry moves it to the front
    // without growing the list.
    bridge.rememberDir("/d3", "images");
    urls = bridge.mruUrls("images");
    REQUIRE(urls.size() == 5);
    CHECK(urls[0].toLocalFile() == QStringLiteral("/d3"));
    CHECK(urls[1].toLocalFile() == QStringLiteral("/d6"));
    for (const auto& url : urls) {
        CHECK(url.toLocalFile() != QStringLiteral("/data/studies/Kneel_1"));
    }
}
