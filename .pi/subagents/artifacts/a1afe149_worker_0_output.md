All mechanics verified. I have a complete picture of the binding/state domain. Compiling the report.

**Verified-clean:** no multi-cycle binding loops anywhere (all bindings are one-way or converge in a single re-evaluation — e.g., the `checked:`→`onToggled`→bridge→`checked` chain converges); no `Qt.binding()` closures anywhere (so no loop-var capture class); no property alias chains; Theme.qml is fully `readonly`. The `messageDialog.title/messageText` BND-2 flag is a semantic false positive (literal initializers, no binding exists to kill — noted for the record).

**Confirmed findings (C1/C2 mechanics fully traced through the bridge code):**