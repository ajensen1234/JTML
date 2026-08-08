# Hegel Property-Based Testing — JTML Authoring Guide

Property-based tests (PBT) auto-generate thousands of inputs and assert
**invariants** ("for all X, P holds") instead of single example inputs. They find
edge cases a hand-written smoke test never thinks to try, then **shrink** the
failing input to its smallest reproducer.

JTML uses [hegel](https://github.com/hegeldev/hegel-cpp) (`v0.11.1` via CMake
`FetchContent`) for pure, CUDA/Qt-free logic. This file is the authoring guide;
for the **build/link** mechanics (conda `dl`/rpath, the `libhegel_c.so` runtime
server, FetchContent-vs-pixi decision) see
`docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md`.

**House rule:** when a piece of extracted pure logic has invariants worth locking
down (length preservation, collision-freedom, monotonicity, determinism), add a
hegel PBT test alongside its deterministic Catch2 unit test. PBT complements —
never replaces — the deterministic golden/test cases.

---

## 1. Two reference tests to follow

- `test/unit/test_direct_optimizer_properties.cpp` — the richest example. Covers
  search-cube containment, seed-cost monotonicity, cumulative call-offset, and
  determinism over randomized ranges/starting points/budgets. Uses nested
  generators, `hegel::Settings{.test_cases = N}`, and `tc.assume(...)`.
- `test/unit/test_model_list_builder_properties.cpp` — the simplest modern
  example: vector-of-strings inputs, length preservation, collision-freedom, and
  determinism. **Start here for a new extraction.**

Both are pure (no GPU, no Qt, no event loop) and registered under the `headless`
ctest label.

---

## 2. The shape of a PBT test

A hegel test is a plain lambda inside a Catch2 `TEST_CASE`. It draws random
values from generators, runs the code-under-test, and asserts an invariant with
normal `REQUIRE`.

```cpp
#include <hegel/hegel.h>
namespace gs = hegel::generators;

TEST_CASE("MyThing[PBT]: invariant X holds", "[mymodule][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto a = tc.draw("a", gs::integers<int>({.min_value = 1,
                                                      .max_value = 100}));
            auto b = tc.draw("b", gs::text({.min_size = 1, .max_size = 20}));
            // ... run code-under-test with a, b ...
            REQUIRE(invariant);
        },
        hegel::Settings{.test_cases = 300});
}
```

Key points:

- **Prefer a named draw** `tc.draw("name", gen)` so a failing replay prints
  `auto name = <value>;` instead of numbered `draw_1` placeholders. Name it after
  the variable it feeds.
- **Feed draws into later draws** (e.g. a target drawn *inside* the search range
  you just drew) to generate correlated, reachable inputs.
- **`tc.assume(condition)`** discards a draw that doesn't meet a precondition
  (e.g. `assume(opt.Run())` — skip cases the optimizer reports as failed). The
  engine reports "N discarded" and keeps trying.
- **Assert with normal `REQUIRE`/`REQUIRE_THAT`.** A thrown exception or a failed
  assertion falsifies the case; hegel then shrinks the drawn inputs to the
  smallest failing example and prints the replay command
  (`HEGEL_REPRODUCE_FAILURE(my_test, "AAEA...")`).

---

## 3. Built-in generators (the discovery FAQ)

The full API lives in the hegel source under
`.build/_deps/hegel-src/include/hegel/generators/*.h`. **To discover what exists,
read those headers** — each generator and each `*Params` struct is documented
inline. Quick map:

### Numeric — `hegel/generators/numeric.h`
- `gs::integers<T>({.min_value, .max_value})` — integral, bounds inclusive.
- `gs::floats<T>({.min_value, .max_value, .exclude_min, .exclude_max,
  .allow_nan, .allow_infinity})`.

### Strings — `hegel/generators/strings.h`
- `gs::text({.min_size, .max_size, .alphabet, .include_characters,
  .exclude_characters, .min_codepoint, .max_codepoint, ...})` — free-form or
  alphabet-restricted strings.
- `gs::characters(...)`, `gs::from_regex(pattern, ...)` — regex-driven strings.
- `gs::binary({.min_size, .max_size})`.

### Collections — `hegel/generators/collections.h`
- `gs::vectors(elem_gen, {.min_size, .max_size, .unique})`.
- `gs::sets(elem_gen, {.min_size, .max_size})`, `gs::maps(key_gen, value_gen, ...)`.
- `gs::tuples(g1, g2, ...)`, `gs::arrays<T,N>(elem_gen)`.

### Combinators — `hegel/generators/combinators.h`
- `gs::sampled_from({e1, e2, ...})` / `gs::sampled_from(vec)` — pick from a
  fixed vocabulary (great for name/state enums, and for **forcing collisions**
  by choosing from a small set).
- `gs::one_of(gen1, gen2, ...)` — pick a generator. `gs::variant(...)`,
  `gs::optional(gen)`, plus slice/reference combinators.

### Defaults — `hegel/generators/default.h`
- `gs::booleans()`, `gs::default_generator<T>()` for user structs (define
  `template<> struct DefaultGenerator<T>`).

Read the headers whenever you need a generator you don't already know: the
`Params` struct fields are the contract, and they tell you the *discoverable*
bounds/options without guessing. The generic rule — **pick the smallest
vocabulary that still exercises your invariant's boundary** (e.g. `sampled_from`
with a handful of names to make dedup collisions likely) — keeps PBT cheap and
fast.

---

## 4. Choosing test cases / determinism

- Set the case count explicitly with `hegel::Settings{.test_cases = N}` (default
  is fine, but a bound keeps fast pure tests fast; the existing tests use
  ~300–400).
- **Determinism is a first-class PBT property** in JTML: for pure logic like the
  model-name dedup, assert `out1 == out2` across repeated runs of the same drawn
  input — it catches a regression that silently renames things.
- Collision-freedom-style properties that assert "output never equals an input in
  some forbidden set" are excellent PBT targets because a hand-written example
  can't cover all collision orderings.

---

## 5. Put an invariant PBT behind extraction

When you extract logic out of `MainScreen` (strangle) or add a pure service, add
BOTH:
1. deterministic Catch2 unit tests pinning the *specific* behavior (incl. any
   quirky edge cases, e.g. the dedup `["A","A","A"] -> ["A(2)","A(3)","A"]`),
2. a hegel PBT test locking the *invariants* (length, collision-freedom,
   determinism, monotonicity) so a future edit can't silently drift the
   guarantees the rest of the app depends on.

They are complementary: unit tests document intent case-by-case; PBT guards the
invariants under combinatorial pressure.

---

## 6. Registering a new PBT target

Mirror `jtml.model_list_builder_props` in `test/CMakeLists.txt`. Compile the test
**.cpp plus only the pure source(s) it needs** (CUDA/Qt-free), link
`Catch2::Catch2WithMain hegel dl`, and append the hegel rpath (conda toolchain
ignores `BUILD_RPATH`):

```cmake
add_executable(jtml_test_<mod>_props
    unit/test_<mod>_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/core/<mod>.cpp
)
target_include_directories(jtml_test_<mod>_props PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_<mod>_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_<mod>_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.<mod>_props COMMAND jtml_test_<mod>_props)
set_tests_properties(jtml.<mod>_props PROPERTIES LABELS "headless" TIMEOUT 300)
```

Hegel's `libhegel_c.so` needs a `uv`-launched `hegel-core` Python server at
runtime (cached under `.hegel/`, auto-gitignored). If that ever blocks an offline
configure, the hegel block in `test/CMakeLists.txt` is a swappable, option-gated
layer (see the tooling-recipe doc).

---

## 7. When NOT to use PBT

- **GPU / VTK / widget / real-event-loop behavior** — those run under the
  `oracle`/manual gates, never in a headless PBT (R9).
- **Pure config / scaffolding** with no invariant worth locking.
- **User-facing string formatting** where a deterministic golden is the clearer
  contract (PBT would just re-derive the same format logic — the circular-test
  anti-pattern R2).

When unsure, ask: *"does a competent engineer ever hit a wrong outcome if this
invariant breaks?"* If yes and the logic is pure, add a PBT.
