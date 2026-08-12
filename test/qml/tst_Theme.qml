// Plan 007 U6 — Theme singleton pins: tokens exist, the type scale is
// monotonic at the pinned values (12/14/16/18), and every used
// text/background pair meets WCAG 4.5:1 (the same values the U4 audit
// measured; this test keeps them pinned).
import QtQuick
import QtTest
import "qrc:/components"

TestCase {
    id: testCase
    name: "Theme"
    when: windowShown

    // Standard WCAG 2.x relative-luminance + contrast helpers.
    function lin(c) {
        // c is a 0..1 channel value
        return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
    }
    function luminance(color) {
        return 0.2126 * lin(color.r) + 0.7152 * lin(color.g) + 0.0722 * lin(color.b)
    }
    function ratio(a, b) {
        const la = luminance(a)
        const lb = luminance(b)
        const hi = Math.max(la, lb)
        const lo = Math.min(la, lb)
        return (hi + 0.05) / (lo + 0.05)
    }

    function test_tokensExist() {
        verify(Theme.bg !== undefined)
        verify(Theme.panel !== undefined)
        verify(Theme.surface !== undefined)
        verify(Theme.fg !== undefined)
        verify(Theme.fgMuted !== undefined)
        verify(Theme.fgDim !== undefined)
        verify(Theme.ok !== undefined)
        verify(Theme.badge !== undefined)
        verify(Theme.selection !== undefined)
        verify(Theme.accent !== undefined)
        verify(Theme.spacingXs !== undefined)
        verify(Theme.spacingSm !== undefined)
    }

    function test_typeScalePinned() {
        compare(Theme.caption, 12)
        compare(Theme.label, 14)
        compare(Theme.body, 16)
        compare(Theme.h2, 18)
    }

    function test_typeScaleMonotonic() {
        verify(Theme.caption < Theme.label)
        verify(Theme.label < Theme.body)
        verify(Theme.body < Theme.h2)
    }

    function test_spacingTokens() {
        compare(Theme.spacingXs, 4)
        compare(Theme.spacingSm, 8)
    }

    function test_contrastPairs() {
        // Primary text on every surface.
        verify(ratio(Theme.fg, Theme.bg) >= 4.5)
        verify(ratio(Theme.fg, Theme.panel) >= 4.5)
        // Secondary + dim text (the U4 audit bumped fgDim to hit >= 4.5
        // on panel — keep it there).
        verify(ratio(Theme.fgMuted, Theme.panel) >= 4.5)
        verify(ratio(Theme.fgDim, Theme.panel) >= 4.5)
        verify(ratio(Theme.fgDim, Theme.surface) >= 4.5)
        // Semantic colors on panels.
        verify(ratio(Theme.ok, Theme.panel) >= 4.5)
        verify(ratio(Theme.badge, Theme.panel) >= 4.5)
        // Badge pill internal pairs.
        verify(ratio(Theme.badgeDirtyFg, Theme.badgeDirtyBg) >= 4.5)
        verify(ratio(Theme.badgeCleanFg, Theme.badgeCleanBg) >= 4.5)
    }
}
