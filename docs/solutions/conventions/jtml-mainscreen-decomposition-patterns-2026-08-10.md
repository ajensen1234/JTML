---
date: 2026-08-10
module: jtml_view
tags: [view-models, qlistview, qabstractlistmodel, settings, pattern]
problem_type: convention
severity: medium
---

# MainScreen decomposition playbook: QListView swap, settings interleave, per-frame controller API (plan 004)

Captured after plan 004 (U1–U9, all landed): the three patterns that made the
5.7k-line god-object decomposition work, with the traps that cost real debugging
time. The phase moved 5693 → 5115 lines and 832 → 821 `ui.`-references.

## 1. QListWidget → QListView + QAbstractListModel swap (U2)

- **The by-name auto-connect silently dies.** `on_<object>_<signal>` slots are
  connected by `connectSlotsByName`; `QListView` has no `itemSelectionChanged`
  signal, so the handlers stop connecting with zero compile error. Replace with
  explicit `connect(view->selectionModel(), &QItemSelectionModel::selectionChanged, ...)`
  — and connect **only** `selectionChanged`, never `currentChanged`
  (MultiSelection arrow keys move current without selecting; connecting both is a
  behavior change).
- **`setModel()` must precede the connects** — `setModel` replaces the selection
  model; a connect issued before it dies silently.
- **Selection command semantics are mode-dependent and version-dependent.**
  `QListWidget::setCurrentRow` passes `SelectCurrent` (additive in
  Extended/MultiSelection, collapsing in SingleSelection); a one-arg
  `QAbstractItemView::setCurrentIndex` routes through `selectionCommand()` →
  `ClearAndSelect` in SingleSelection. The Qt-5 folklore "one-arg setCurrentIndex
  defaults to NoUpdate" is false in Qt 6. Prescription: `SelectCurrent|Rows` at
  navigation sites, `ClearAndSelect|Rows` at the one site that today collapses
  (the single-model radio). The selection-model API is mode-blind — you must
  replicate the command the old widget computed.
- **The swap forces a file-wide API sweep**, not just the enumerated sites:
  `count()` → `model()->rowCount()`, `currentRow()` → `currentIndex().row()`,
  `addItem()` → model `beginInsertRows`/`endInsertRows` + push (order matters —
  pushing before `beginInsertRows` violates the model contract and breaks views),
  `item(i)->setSelected()` → `selectionModel()->select(index, Select|Rows)`.
  Sites in slots owned by later units (pose, optimizer, segmentation) must be
  converted in the swap unit — note it in those units so per-cut diffs stay clean.
- **Stylesheet selectors don't follow the swap**: `.ui` `QListWidget::item:selected`
  blocks stop matching; add `QListView::item:selected` alongside (only the active
  blocks — count commented-out occurrences separately).

## 2. SettingsService interleave split (U3/U5)

- The edge slots wrote QSettings **inline, per-site with different sourcing**:
  three slider slots stored their own widget key plus the other two keys computed
  from the *current frame*; the apply-all slot stored all three from the widgets.
  A uniform "read all widgets" (or "all frame-sourced") service changes registry
  contents across restarts. Extract per-site verbatim; pin with a round-trip PBT
  (fractional/negative draws — the silent-narrowing invariant shape).
- First-run detection `childGroups().size() == 0` is not widget-free in the first
  branch: the CUDA probe + dialogs stay in the view; the service returns values.
- Registry writes can ride a **signal cascade** (reset-edge = three `setValue`
  calls; the write happens in the `valueChanged` slots) — don't "fix" it with a
  direct write.
- `QSettings` test isolation: constructor takes a path/format override
  (`IniFormat` + temp file); tests never touch the real registry.

## 3. Per-frame controller API for GPU/interleave-heavy blocks (U8)

- The segment/estimate block interleaves ~44 progress/`processEvents`/render calls
  with torch work. A controller-owned loop forces the interleave to change;
  instead the **view owns the loop and the controller exposes per-frame ops**
  (`SegmentFrame(image, ...) -> image`). Byte-identical interleave is then
  verifiable by counting the calls before/after.
- Placement follows the *actual* dependency: the estimator landed in `jtml_services`
  (no `CostFunctionManager` call in the math — only header-only calibration math),
  making services torch-linked; headless tests are unaffected because tests
  compile `.cpp` files directly.
- torch's `ATen/core/ivalue_inl.h` does `#undef slots` — a torch-bearing include
  must sit after Qt-object headers (`optimizer_manager.h`/`settings_control.h`/`drr_tool.h`).
- GPU-gated units gate on compile + the `oracle` label only (`ctest -L gpu` matches
  nothing — there is no `gpu` label); the oracle test skips cleanly when
  user-provided `.pt` fixtures are absent.

## Related

- `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md`
- `docs/solutions/tooling-decisions/jtml-rendering-runtime-xcb-qvtk-2026-08-10.md`
