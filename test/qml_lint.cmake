# test/qml_lint.cmake — plan 007 U2: the jtml.qml_lint gate script.
#
# Runs the Qt 6.7.2 qmllint over the experimental app's QML files and
# fails on any warning outside the documented accepted set.
#
# Passed via -D from test/CMakeLists.txt:
#   QML_LINT — absolute path to the Qt6 qmllint binary
#              ($CONDA_PREFIX/lib/qt6/bin/qmllint). The bare `qmllint`
#              name on PATH is Qt 5.15.8 (from qt-main, pulled by opencv's
#              qt5 build) and lacks --json — never use it.
#   QML_DIR  — src/app/experimental
#   QML_OUT  — write path for the qmllint --json output
#
# Accepted warning ids (each has a documented removal path):
#   import                — env Qt5+Qt6 module ambiguity (QtQuick.Dialogs
#                           defined twice: qt-main 5.15.8 + qt6 6.7.2) +
#                           the C++-module import note. Environmental.
#   unqualified           — context-property bridge coupling (studyBridge/
#                           optimizerBridge/... are engine-level root
#                           properties qmllint cannot see). Removed by
#                           plan 007 U3's property injection.
#   unresolved-type /
#   missing-property      — QmlVtkRenderer is C++-registered without a
#                           qmltypes file; qmllint cannot resolve it
#                           (runtime registration via main.cpp is fine).
#   use-proper-function   — SettingsPanel's `property var commit`
#                           pass-through glue (deliberate; QML has no
#                           function-typed property).
#   unused-imports        — resolved by U2 (kept accepted so a leftover
#                           does not silently rot the gate).
#
# Anything else (e.g. Quick.layout-positioning) FAILS the gate.
cmake_minimum_required(VERSION 3.19)

if(NOT DEFINED QML_LINT)
    message(FATAL_ERROR "qml_lint.cmake: QML_LINT not passed")
endif()
if(NOT DEFINED QML_DIR OR NOT EXISTS "${QML_DIR}")
    message(FATAL_ERROR "qml_lint.cmake: QML_DIR missing: ${QML_DIR}")
endif()
if(NOT DEFINED QML_OUT)
    set(QML_OUT "${CMAKE_CURRENT_LIST_DIR}/qmllint-out.json")
endif()

file(GLOB QML_FILES "${QML_DIR}/*.qml")
if(NOT QML_FILES)
    message(FATAL_ERROR "qml_lint.cmake: no .qml files in ${QML_DIR}")
endif()

execute_process(
    COMMAND "${QML_LINT}" --json "${QML_OUT}" -I "${QML_DIR}" ${QML_FILES}
    RESULT_VARIABLE rc)
# qmllint exits nonzero whenever ANY warning is emitted — including the
# documented-accepted ones (unqualified/import), so the exit code is NOT
# the gate. The JSON parse below is the authority: it fails only on
# non-accepted warnings. A missing JSON means qmllint itself failed.
if(NOT EXISTS "${QML_OUT}")
    message(FATAL_ERROR
        "qmllint failed to produce ${QML_OUT} (rc ${rc}) — is QML_LINT "
        "the Qt6 binary?")
endif()

file(READ "${QML_OUT}" json)
string(JSON nfiles LENGTH "${json}" files)
if(nfiles LESS 1)
    message(FATAL_ERROR "qmllint produced no file entries")
endif()
math(EXPR nfiles "${nfiles} - 1")

set(bad "")
foreach(i RANGE 0 ${nfiles})
    string(JSON fname GET "${json}" files ${i} filename)
    string(JSON success GET "${json}" files ${i} success)
    string(JSON nwarn LENGTH "${json}" files ${i} warnings)
    # success=false means qmllint could not fully resolve the file — that is
    # the documented C++-registration gap (QmlVtkRenderer without a
    # qmltypes file) when the file still emits unresolved-type warnings.
    # A silent failure (no warnings at all) is a real problem though.
    if(NOT success AND nwarn EQUAL 0)
        string(APPEND bad "  ${fname}: qmllint success=false with zero warnings\n")
    endif()
    if(nwarn GREATER 0)
        math(EXPR nwarn "${nwarn} - 1")
        foreach(j RANGE 0 ${nwarn})
            string(JSON wid GET "${json}" files ${i} warnings ${j} id)
            string(JSON wtype GET "${json}" files ${i} warnings ${j} type)
            # `line` is absent on import-ambiguity warnings — fall back.
            string(JSON wline ERROR_VARIABLE line_err
                GET "${json}" files ${i} warnings ${j} line)
            if(line_err)
                set(wline "?")
            endif()
            string(JSON wmsg GET "${json}" files ${i} warnings ${j} message)
            if(wtype STREQUAL "error")
                string(APPEND bad
                    "  ${fname}:${wline}: ERROR [${wid}] ${wmsg}\n")
            elseif(wid STREQUAL "import"
                   OR wid STREQUAL "unqualified"
                   OR wid STREQUAL "unresolved-type"
                   OR wid STREQUAL "missing-property"
                   OR wid STREQUAL "use-proper-function"
                   OR wid STREQUAL "unused-imports")
                # Documented accepted — counted, not failed.
            else()
                string(APPEND bad
                    "  ${fname}:${wline}: [${wid}] ${wmsg}\n")
            endif()
        endforeach()
    endif()
endforeach()

if(bad)
    message(FATAL_ERROR
        "qmllint gate FAILED — non-accepted warnings:\n${bad}")
endif()
message(STATUS "qmllint gate PASSED (only documented-accepted warnings)")
