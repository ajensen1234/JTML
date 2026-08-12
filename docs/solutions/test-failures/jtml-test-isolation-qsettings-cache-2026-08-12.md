---
title: "Test-harness isolation gotchas: QSettings path cache defeats per-case XDG redirects; tryCompare does not resolve dotted paths"
date: 2026-08-12
category: test-failures
module: jtml_view
problem_type: test_failure
component: testing_framework
symptoms:
  - "a FileDialogBridge unit test read the user's REAL config despite redirecting XDG_CONFIG_HOME to a temp dir per test case"
  - "test state leaked across Catch2 test cases (MRU lists grew across cases)"
  - "tryCompare(fake, 'loadModelsLog.length', 2) always read undefined"
root_cause: test_isolation
resolution_type: test_fix
severity: medium
tags: [qsettings, xdg-config, trycompare, qmltest, catch2, test-isolation]
related_components:
  - tooling
---

# Test-harness isolation gotchas: QSettings path cache + tryCompare dotted paths

## Problem

Two test-infrastructure behaviors silently invalidated new pins during plan
007's review round: QSettings/QStandardPaths cache the resolved config path
per process, so per-test-case `XDG_CONFIG_HOME` redirects leak state across
cases (and can fall back to the real user config); and QtTest's
`tryCompare` does not resolve dotted property paths, so
`tryCompare(obj, "a.b", v)` always reads `undefined`.

## Symptoms

- `CHECK(bridge.lastDir("images").isEmpty())` failed with the real user
  config value — the temp-dir redirect never took effect.
- MRU counts accumulated across Catch2 test cases (an isolation test saw 5
  entries from the cap test's writes).
- `tryCompare(fakeStudy, "loadModelsLog.length", 2)` failed with
  "Actual: undefined" while `fakeStudy.loadModelsLog.length` was a valid
  array length.

## What Didn't Work

- Per-test-case `qputenv("XDG_CONFIG_HOME", tempDir)` + fresh bridge — the
  path resolution is cached per process, so later cases silently reuse the
  first temp dir (or the real config) and share state.
- Using a dotted path inside `tryCompare` — it treats the whole string as a
  property name.

## Solution

- **One test case per isolated config**: all FileDialogBridge directory-
  memory pins live in a single `TEST_CASE` with one `QTemporaryDir` + one
  `qputenv` before the first `QSettings` construction in the process. No
  per-case redirects.
- **Compare the array itself**: `tryCompare(fakeStudy.loadModelsLog,
  "length", 2)` (the array object's real `length` property) instead of the
  dotted path. For property chains, use a `wait(50)` + `compare()` with a
  plain JS expression instead of `tryCompare`.

## Why This Works

QSettings resolves its file path once per process (the static path cache
ignores later env changes), so isolation must be established before the
first construction and kept for the process lifetime. `tryCompare` is
documented for single property names; dotted paths are not resolved, so the
comparison always sees `undefined` — comparing the terminal object's own
property avoids the ambiguity.

## Prevention

- When testing anything that reads `QSettings`/`QStandardPaths`: set the
  env redirect once at the top of the process (single test case or a
  fixture that runs before any QSettings construction); never rely on
  per-case redirects.
- Prefer `compare()`/`tryCompare()` on the terminal object's own property;
  use `wait()` + JS expressions for chains.
- Both gotchas are now encoded in `test/unit/experimental_file_dialog_test.cpp`
  (single-case structure) and `test/qml/tst_StudyFlows.qml`
  (`test_toolbarModelsAfterCalibration` compares the log array directly).

## Related Issues

- `docs/solutions/conventions/jtml-qml-view-testing-2026-08-12.md` — the
  harness recipe these pins belong to
- Plan review round R4: `docs/plans/2026-08-12-007-refactor-qml-experimental-improvement-pass-plan.md`
