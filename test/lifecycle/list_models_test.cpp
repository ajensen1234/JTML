// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Headless view-model tests (plan 004 U2, R4/R5/R14/R15).
//
// FrameListModel + ModelListModel + their QItemSelectionModel, compiled
// directly against Qt6::Core/Qt6::Test — no widgets, no VTK, no display.
// Pins the selection semantics the MainScreen wiring relies on after the
// QListView swap:
//  - selection writes go through selectionModel()->select()/setCurrentIndex()
//    with explicit flags (the old QListWidget-only API is gone);
//  - selectionChanged is the only signal the view connects to (a current-only
//    change must NOT fire it — currentChanged is never connected);
//  - the empty-selection fallback (deselect-all re-selects current) and the
//    per-item unbatched principal loops keep today's intermediate states.

#include <QItemSelection>
#include <QItemSelectionModel>
#include <QSignalSpy>
#include <QtTest/QtTest>

#include "view/frame_list_model.h"
#include "view/model_list_model.h"

namespace {

// Row numbers of a selection, in the order QItemSelectionModel::selectedRows
// reports them (ascending).
QVector<int> SelectedRows(const QItemSelectionModel& sm) {
    const QModelIndexList rows = sm.selectedRows();
    QVector<int> out;
    out.reserve(rows.size());
    for (const auto& idx : rows) {
        out.push_back(idx.row());
    }
    return out;
}

// Row numbers carried by one of the spy payload arguments (0 = selected
// ranges, 1 = deselected ranges) of the selectionChanged emission.
QVector<int> SpyRanges(const QSignalSpy& spy, int index, int arg) {
    const QItemSelection selection =
        spy.at(index).at(arg).value<QItemSelection>();
    QVector<int> out;
    for (const auto& range : selection) {
        for (int r = range.top(); r <= range.bottom(); ++r) {
            out.push_back(r);
        }
    }
    return out;
}

}  // namespace

class ListModelsTest : public QObject {
    Q_OBJECT
private slots:
    void FrameNamesExposeDataAndRowCount();
    void ModelNamesExposeDataAndRowCount();
    void SelectionFiresSelectionChanged();
    void EmptyModelReportsNoRows();
    void DeselectAllThenFallbackReselectsCurrent();
    void PrimaryModelIsFirstSelectedRow();
    void BiplaneMultilineFrameNamePreserved();
    void ModelDedupRescanQuirk();
    void PrincipalLoopsEmitIntermediateStates();
    void CurrentOnlyChangeDoesNotFireSelectionChanged();
    void ArrowKeyStyleNavigationFiresSelectionChanged();
};

void ListModelsTest::FrameNamesExposeDataAndRowCount() {
    FrameListModel model;
    QCOMPARE(model.rowCount(), 0);
    model.AppendFrame(QStringLiteral("frameA"));
    model.AppendFrame(QStringLiteral("frameB"));
    model.AppendFrame(QStringLiteral("frameC"));
    QCOMPARE(model.rowCount(), 3);
    QCOMPARE(model.data(model.index(0, 0), Qt::DisplayRole).toString(),
             QStringLiteral("frameA"));
    QCOMPARE(model.data(model.index(2, 0), Qt::DisplayRole).toString(),
             QStringLiteral("frameC"));
    // DisplayRole is the only role the views read.
    QVERIFY(!model.data(model.index(0, 0), Qt::ToolTipRole).isValid());
    QVERIFY(!model.data(model.index(3, 0), Qt::DisplayRole).isValid());
    QVERIFY(!model.data(QModelIndex(), Qt::DisplayRole).isValid());
}

void ListModelsTest::ModelNamesExposeDataAndRowCount() {
    ModelListModel model;
    QCOMPARE(model.rowCount(), 0);
    const QVector<QString> display =
        model.AppendModels({QStringLiteral("femur"),
                            QStringLiteral("tibia")});
    QCOMPARE(display.size(), 2);
    QCOMPARE(display.at(0), QStringLiteral("femur"));
    QCOMPARE(model.rowCount(), 2);
    QCOMPARE(model.data(model.index(1, 0), Qt::DisplayRole).toString(),
             QStringLiteral("tibia"));
}

void ListModelsTest::SelectionFiresSelectionChanged() {
    FrameListModel model;
    model.AppendFrame(QStringLiteral("a"));
    model.AppendFrame(QStringLiteral("b"));
    model.AppendFrame(QStringLiteral("c"));
    QItemSelectionModel sm(&model);
    QSignalSpy spy(&sm, &QItemSelectionModel::selectionChanged);
    sm.select(model.index(1, 0),
              QItemSelectionModel::Select | QItemSelectionModel::Rows);
    QCOMPARE(spy.count(), 1);
    QCOMPARE(SelectedRows(sm), (QVector<int>{1}));
    // Selecting the same row again changes nothing -> no signal.
    sm.select(model.index(1, 0),
              QItemSelectionModel::Select | QItemSelectionModel::Rows);
    QCOMPARE(spy.count(), 1);
    // Deselect fires too.
    sm.select(model.index(1, 0),
              QItemSelectionModel::Deselect | QItemSelectionModel::Rows);
    QCOMPARE(spy.count(), 2);
    QVERIFY(sm.selectedRows().isEmpty());
}

void ListModelsTest::EmptyModelReportsNoRows() {
    FrameListModel model;
    QCOMPARE(model.rowCount(), 0);
    QVERIFY(!model.index(0, 0).isValid());
    QVERIFY(!model.data(model.index(0, 0), Qt::DisplayRole).isValid());

    ModelListModel model_model;
    QCOMPARE(model_model.rowCount(), 0);
    // Appending nothing stays empty.
    QVERIFY(model_model.AppendModels({}).isEmpty());
    QCOMPARE(model_model.rowCount(), 0);
}

void ListModelsTest::DeselectAllThenFallbackReselectsCurrent() {
    ModelListModel model;
    model.AppendModels({QStringLiteral("a"), QStringLiteral("b"),
                        QStringLiteral("c")});
    QItemSelectionModel sm(&model);
    // Multi-select {0,2}, current at 1.
    sm.select(model.index(0, 0),
              QItemSelectionModel::Select | QItemSelectionModel::Rows);
    sm.select(model.index(2, 0),
              QItemSelectionModel::Select | QItemSelectionModel::Rows);
    sm.setCurrentIndex(model.index(1, 0),
                       QItemSelectionModel::NoUpdate);
    // Deselect all (what a user Ctrl+clicking the last row away does).
    sm.clearSelection();
    QVERIFY(sm.selectedRows().isEmpty());
    // MainScreen's empty-selection fallback: re-select the current row,
    // synchronous-direct, exactly like today's item()->setSelected(true).
    if (sm.currentIndex().row() >= 0) {
        sm.select(model.index(sm.currentIndex().row(), 0),
                  QItemSelectionModel::Select | QItemSelectionModel::Rows);
    }
    QCOMPARE(SelectedRows(sm), (QVector<int>{1}));
    QCOMPARE(sm.currentIndex().row(), 1);
}

void ListModelsTest::PrimaryModelIsFirstSelectedRow() {
    ModelListModel model;
    model.AppendModels({QStringLiteral("a"), QStringLiteral("b"),
                        QStringLiteral("c"), QStringLiteral("d")});
    QItemSelectionModel sm(&model);
    // selectedRows() preserves selection order (first selected row first, as
    // the user clicked them): the primary-model rule is selectedRows()[0].
    sm.select(model.index(3, 0),
              QItemSelectionModel::Select | QItemSelectionModel::Rows);
    sm.select(model.index(1, 0),
              QItemSelectionModel::Select | QItemSelectionModel::Rows);
    const QModelIndexList rows = sm.selectedRows();
    QCOMPARE(rows.size(), 2);
    QCOMPARE(rows.at(0).row(), 3);  // primary = first-selected row
    QCOMPARE(rows.at(1).row(), 1);
    QCOMPARE(SelectedRows(sm), (QVector<int>{3, 1}));
}

void ListModelsTest::BiplaneMultilineFrameNamePreserved() {
    FrameListModel model;
    const QString biplane = QStringLiteral("A: baseA\nB: baseB");
    model.AppendFrame(biplane);
    QCOMPARE(model.rowCount(), 1);
    QCOMPARE(model.data(model.index(0, 0), Qt::DisplayRole).toString(),
             biplane);
}

void ListModelsTest::ModelDedupRescanQuirk() {
    ModelListModel model;
    // N identical inputs -> ["A(2)","A(3)","A"] (mutated-name rescan quirk,
    // byte-identical to the old inline logic via ModelListBuilder).
    const QVector<QString> display =
        model.AppendModels({QStringLiteral("A"), QStringLiteral("A"),
                            QStringLiteral("A")});
    QCOMPARE(display,
             (QVector<QString>{QStringLiteral("A(2)"),
                               QStringLiteral("A(3)"),
                               QStringLiteral("A")}));
    QCOMPARE(model.rowCount(), 3);
    QCOMPARE(model.data(model.index(0, 0), Qt::DisplayRole).toString(),
             QStringLiteral("A(2)"));
    QCOMPARE(model.data(model.index(2, 0), Qt::DisplayRole).toString(),
             QStringLiteral("A"));
    // A new load of the same base name dedups against the mutated names too.
    const QVector<QString> again =
        model.AppendModels({QStringLiteral("A")});
    QCOMPARE(again, (QVector<QString>{QStringLiteral("A(4)")}));
    QCOMPARE(model.rowCount(), 4);
}

void ListModelsTest::PrincipalLoopsEmitIntermediateStates() {
    ModelListModel model;
    model.AppendModels({QStringLiteral("a"), QStringLiteral("b"),
                        QStringLiteral("c")});
    QItemSelectionModel sm(&model);
    sm.select(model.index(0, 0),
              QItemSelectionModel::Select | QItemSelectionModel::Rows);
    sm.select(model.index(1, 0),
              QItemSelectionModel::Select | QItemSelectionModel::Rows);
    sm.select(model.index(2, 0),
              QItemSelectionModel::Select | QItemSelectionModel::Rows);
    QSignalSpy spy(&sm, &QItemSelectionModel::selectionChanged);

    // VTKMakePrincipalSignal's two loops stay per-item and unbatched: the
    // intermediate re-entrant states are R13-visible.
    const int index_new_principal = 1;
    const QModelIndexList selected = sm.selectedRows();
    for (int i = 0; i < selected.size(); i++) {
        if (selected[i].row() != index_new_principal) {
            sm.select(model.index(selected[i].row(), 0),
                      QItemSelectionModel::Deselect |
                          QItemSelectionModel::Rows);
        }
    }
    QCOMPARE(spy.count(), 2);
    QCOMPARE(SelectedRows(sm), (QVector<int>{1}));
    // Deselect loop: each emission carries the DESELECTED ranges (arg 1).
    QCOMPARE(SpyRanges(spy, 0, 1), (QVector<int>{0}));  // deselected: 0
    QCOMPARE(SpyRanges(spy, 1, 1), (QVector<int>{2}));  // deselected: 2

    for (int i = 0; i < selected.size(); i++) {
        if (selected[i].row() != index_new_principal) {
            sm.select(model.index(selected[i].row(), 0),
                      QItemSelectionModel::Select |
                          QItemSelectionModel::Rows);
        }
    }
    QCOMPARE(spy.count(), 4);
    // Select loop: each emission carries the SELECTED ranges (arg 0).
    QCOMPARE(SpyRanges(spy, 2, 0), (QVector<int>{0}));  // selected: 0
    QCOMPARE(SpyRanges(spy, 3, 0), (QVector<int>{2}));  // selected: 2
    // Final state preserves range order: the surviving principal row comes
    // FIRST in selectedRows(), exactly the primary-model rule the swap relies
    // on (today's item()->setSelected(true) re-adds ranges the same way).
    QCOMPARE(SelectedRows(sm), (QVector<int>{1, 0, 2}));
}

void ListModelsTest::CurrentOnlyChangeDoesNotFireSelectionChanged() {
    FrameListModel model;
    model.AppendFrame(QStringLiteral("a"));
    model.AppendFrame(QStringLiteral("b"));
    model.AppendFrame(QStringLiteral("c"));
    QItemSelectionModel sm(&model);
    sm.select(model.index(0, 0),
              QItemSelectionModel::Select | QItemSelectionModel::Rows);
    QSignalSpy spy(&sm, &QItemSelectionModel::selectionChanged);
    // A pure current move (NoUpdate) must not fire selectionChanged: the
    // MainScreen wiring connects selectionChanged ONLY (currentChanged is
    // deliberately never connected — MultiSelection arrow-key behavior).
    sm.setCurrentIndex(model.index(2, 0), QItemSelectionModel::NoUpdate);
    QCOMPARE(spy.count(), 0);
    QCOMPARE(sm.currentIndex().row(), 2);
    QCOMPARE(SelectedRows(sm), (QVector<int>{0}));  // selection untouched
}

void ListModelsTest::ArrowKeyStyleNavigationFiresSelectionChanged() {
    FrameListModel model;
    model.AppendFrame(QStringLiteral("a"));
    model.AppendFrame(QStringLiteral("b"));
    model.AppendFrame(QStringLiteral("c"));
    QItemSelectionModel sm(&model);
    sm.setCurrentIndex(model.index(0, 0),
                       QItemSelectionModel::SelectCurrent |
                           QItemSelectionModel::Rows);
    QSignalSpy spy(&sm, &QItemSelectionModel::selectionChanged);
    // Qt arrow keys change current AND select; in the view that lands here
    // (selectionCommand in SingleSelection is ClearAndSelect, but the wiring
    // contract is: the handler is driven by selectionChanged, and any
    // selection-affecting navigation fires it). SelectCurrent|Rows is the
    // command the seven programmatic current-write sites use post-swap.
    sm.setCurrentIndex(model.index(1, 0),
                       QItemSelectionModel::SelectCurrent |
                           QItemSelectionModel::Rows);
    QCOMPARE(spy.count(), 1);
    QCOMPARE(sm.currentIndex().row(), 1);
    QCOMPARE(SelectedRows(sm), (QVector<int>{1}));
}

QTEST_GUILESS_MAIN(ListModelsTest)
#include "list_models_test.moc"
