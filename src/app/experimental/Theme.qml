// Theme.qml — the experimental app's palette (plan 005 feedback #6: all
// color/text choices live here, not scattered through the shell).
// Singleton (pragma + qmldir) so every file shares one instance.
pragma Singleton
import QtQuick

QtObject {
    readonly property color bg: "#14161a"          // window background
    readonly property color panel: "#1b1e24"       // panels / list areas
    readonly property color surface: "#101216"     // viewport placeholder
    readonly property color border: "#2a2f38"
    readonly property color fg: "#e6e9ee"          // primary text (light)
    readonly property color fgMuted: "#8b929c"     // secondary text
    readonly property color fgDim: "#6b7280"       // placeholders / hints
    readonly property color selection: "#2a4a7a"   // list highlight
    readonly property color ok: "#5a8a5f"          // calibrated / good
    readonly property color accent: "#4a90d9"      // interactive accents
    readonly property color badge: "#c0392b"       // debug readout
}
