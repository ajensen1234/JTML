// Plan 007 U6 — Qt Quick Test entry for the experimental view layer.
//
// QUICK_TEST_SOURCE_DIR is compile-defined to test/qml (on disk); the
// tst_*.qml files live there. The components under test are imported by
// the tst files via "qrc:/components" (tests.qrc aliases the REAL
// src/app/experimental sources — no drift). The C++ bridges are NOT
// linked: the fakes (test/qml/Fake*.qml, same-directory types) replace
// them, so this target is headless and VTK-free by construction.
#include <QtQuickTest/quicktest.h>

QUICK_TEST_MAIN(qml_view)
