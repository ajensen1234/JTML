// Theme.qml — the experimental app's palette + type scale (plan 005
// feedback #6: all color/text choices live here, not scattered through
// the shell). Singleton (pragma + qmldir) so every file shares one
// instance. Plan 007 U2: badge-pill tokens + TypeScale roles added; the
// scale is pinned (major second 1.125 from base 16) and applied by U2,
// verified by U4.
pragma Singleton
import QtQuick

QtObject {
    id: root

    // ---- Palette --------------------------------------------------------
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
    readonly property color badge: "#c0392b"       // errors / debug readout
    // Dirty/unsaved pill (dirty = unsaved amber, clean = saved green).
    readonly property color badgeDirtyBg: "#e5b567"
    readonly property color badgeDirtyFg: "#2a2118"
    readonly property color badgeCleanBg: "#3a4a3d"
    readonly property color badgeCleanFg: "#8fbf96"
    readonly property color badgeText: "#ffffff"   // text on Theme.badge
    // Run-lock overlay (plan 007 U3 D5): translucent dim over the viewport
    // while an optimizer run is in flight.
    readonly property color overlayDim: "#66101416"

    // ---- Type scale (pinned: major second 1.125 from base 16) ----------
    readonly property real caption: 12             // metadata / labels
    readonly property real label: 14               // section headers
    readonly property real body: 16                // primary reading
    readonly property real h2: 18                  // panel titles
}
