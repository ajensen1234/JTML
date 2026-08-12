Everything is green. Writing the acceptance report:

## U6 Summary

**Part A — Property injection (D1):** All 8 components now declare `required property var <bridge>`; main.qml injects the real bridges at every use site via root-level aliases (same-name passing is self-referential in QML — the aliases avoid binding loops). Zero context-property coupling left in components; qmllint unqualified warnings dropped ~300 → 67 (of which **0 are bridge-coupling**; the rest are main.qml's rootContext refs + the qmllint Unbound-delegate limitation, both in the gate's documented accepted set).

**Product bugs found & fixed (the harness earned its keep):**
1. **App was unloadable since U5** — `required property double x/y/z` on the pose-table delegate collided with Item's FINAL geometry props (`Cannot override FINAL property`). Fixed with the qt-qml-sanctioned `model.roleName` access.
2. **PosesDialog dirty-close guard was broken** — `discardConfirmed`/`discardRequested()` were referenced but never declared (runtime error on open/close). Declared.
3. **Commit-on-pool contract was broken in Qt 6.7** — pooling doesn't drop focus (hidden items keep activeFocus), so focus-loss `editingFinished` never fires on recycle and typed values were **silently clobbered** by `resetDisplay`. Also, delegate-root `signal pooled()`/`onPooled:` never fire — the notifications are **`ListView.onPooled`/`ListView.onReused` attached signals**. Fixed with an `edited` flag + explicit `commitIfEditing()` flushed from the attached handler.

**Part B/C — Harness:** `test/qml/` with `main.cpp` (QUICK_TEST_MAIN), `tests.qrc` (aliases to the **real** component sources — no drift; ViewportPanel excluded — VTK territory), 6 QML fakes, 5 tst files (25 test functions), and a block-scoped CMake registration (headless label, offscreen + Material style env, sole ctest run mechanism).