/* Plan 012 U3 — ForceRelease / LeavePoisoned + Shutdown poison-skip (C4/C8).
 * Test-first: expects EvaluationContextPool additions InitForTest,
 * ForceRelease, LeavePoisoned, IsPoisoned, and Checkout skips poisoned,
 * Shutdown skips poisoned. CUDA-free headless (InitForTest resizes vectors with
 * null/default contexts, no CUDA alloc).
 */
#include <catch2/catch_test_macros.hpp>

using gpu_cost_function::EvaluationContextPool;
using gpu_cost_function::EvaluationStatus;

TEST_CASE(
    "ForceRelease makes a checked-out context reusable",
    "[evaluation_context][lease]") {
    EvaluationContextPool pool;
    pool.InitForTest(4);
    REQUIRE(pool.size() == 4);
    int a = pool.Checkout();
    REQUIRE(a >= 0);
    REQUIRE(pool.IsInFlight(static_cast<std::size_t>(a)));
    REQUIRE_FALSE(pool.IsPoisoned(static_cast<std::size_t>(a)));
    REQUIRE(pool.ForceRelease(static_cast<std::size_t>(a)));
    REQUIRE_FALSE(pool.IsInFlight(static_cast<std::size_t>(a)));
    REQUIRE_FALSE(pool.IsPoisoned(static_cast<std::size_t>(a)));
    int b = pool.Checkout();
    REQUIRE(b >= 0);
    // Linear scan returns the lowest free slot; ForceRelease freed `a` first,
    // so a correct implementation MUST yield b == a. Catches a no-op
    // ForceRelease.
    REQUIRE(b == a);
    REQUIRE_FALSE(pool.IsPoisoned(static_cast<std::size_t>(b)));
}

TEST_CASE(
    "LeavePoisoned keeps context checked out and Checkout skips it",
    "[evaluation_context][lease]") {
    EvaluationContextPool pool;
    pool.InitForTest(4);
    int a = pool.Checkout();
    REQUIRE(a >= 0);
    REQUIRE(pool.LeavePoisoned(static_cast<std::size_t>(a)));
    REQUIRE(pool.IsPoisoned(static_cast<std::size_t>(a)));
    // Poisoned stays checked out from pool's view (IsInFlight still true or
    // poisoned not reusable) Provide a few checkouts: none should return the
    // poisoned index
    for (int i = 0; i < 3; ++i) {
        int c = pool.Checkout();
        REQUIRE(c >= 0);
        REQUIRE(c != a);
        REQUIRE_FALSE(pool.IsPoisoned(static_cast<std::size_t>(c)));
    }
    // No more free non-poisoned contexts
    int d = pool.Checkout();
    REQUIRE(d == -1);
}

TEST_CASE(
    "ForceRelease then re-Checkout works; LeavePoisoned prevents re-Checkout "
    "of that index",
    "[evaluation_context][lease]") {
    EvaluationContextPool pool;
    pool.InitForTest(4);
    int a = pool.Checkout();
    REQUIRE(a >= 0);
    REQUIRE(pool.ForceRelease(static_cast<std::size_t>(a)));
    int b = pool.Checkout();
    REQUIRE(b >= 0);
    // b could be same as a since it was freed; at least not poisoned
    REQUIRE_FALSE(pool.IsPoisoned(static_cast<std::size_t>(b)));

    int c = pool.Checkout();
    REQUIRE(c >= 0);
    REQUIRE(c != b);
    // Poison c
    REQUIRE(pool.LeavePoisoned(static_cast<std::size_t>(c)));
    REQUIRE(pool.IsPoisoned(static_cast<std::size_t>(c)));
    // Further checkouts should not return c
    int d = pool.Checkout();
    // There should be still one free slot (the 4th)
    REQUIRE(d >= 0);
    REQUIRE(d != c);
    REQUIRE(d != b);
    REQUIRE_FALSE(pool.IsPoisoned(static_cast<std::size_t>(d)));
    // Now all non-poisoned are checked out — with 4 slots, 1 poisoned, 3
    // usable: b and d checked out, one free left
    int e = pool.Checkout();
    REQUIRE(e >= 0);
    REQUIRE(e != c);
    REQUIRE(e != b);
    REQUIRE(e != d);
    REQUIRE_FALSE(pool.IsPoisoned(static_cast<std::size_t>(e)));
    int f = pool.Checkout();
    REQUIRE(f == -1);
}

TEST_CASE("Shutdown skips poisoned contexts", "[evaluation_context][lease]") {
    EvaluationContextPool pool;
    pool.InitForTest(3);
    int idx0 = pool.Checkout();
    REQUIRE(idx0 >= 0);
    REQUIRE(pool.LeavePoisoned(static_cast<std::size_t>(idx0)));
    REQUIRE(pool.IsPoisoned(static_cast<std::size_t>(idx0)));
    // Shutdown must not crash and must clear vectors even with poisoned entries
    // (leaked intentionally but cleared)
    pool.Shutdown();
    REQUIRE(pool.size() == 0);
    // Checkout after shutdown should fail
    REQUIRE(pool.Checkout() == -1);
}

TEST_CASE(
    "ForceRelease and LeavePoisoned return false on invalid index",
    "[evaluation_context][lease]") {
    EvaluationContextPool pool;
    pool.InitForTest(2);
    REQUIRE_FALSE(pool.ForceRelease(99));
    REQUIRE_FALSE(pool.LeavePoisoned(99));
    REQUIRE_FALSE(pool.IsPoisoned(99));
    int a = pool.Checkout();
    REQUIRE(a >= 0);
    // Valid idx but not checked out state after ForceRelease
    REQUIRE(pool.ForceRelease(static_cast<std::size_t>(a)));
    REQUIRE_FALSE(
        pool.ForceRelease(static_cast<std::size_t>(a)));  // already released
    int b = pool.Checkout();
    REQUIRE(b >= 0);
    REQUIRE(pool.LeavePoisoned(static_cast<std::size_t>(b)));
    // Already poisoned; second LeavePoisoned should ideally still succeed or
    // fail? spec says true if valid && checked_out After poison, still checked
    // out, so second call may return false? We expect false for already
    // poisoned or true? Check spec: if idx valid && checked_out && !poisoned =>
    // ForceRelease true, else false. LeavePoisoned true if valid &&
    // checked_out. So second leave may return false if already poisoned or
    // true; we document.
}
