# P13 Pipe Correctness — DIRECT batch seam to the concurrent design

**Lens:** cuda-correctness (logic errors, oracle/bit-exact comparison, error propagation)
**Scope (read-only):** `src/domain/direct_optimizer.cpp`, `src/coordinator/optimizer_manager.cpp`,
`src/compute/CostFunctionManager.cpp`, `src/compute/cost_capacity_service.cu`,
`src/domain/direct_data_storage.cpp`, `include/domain/direct_optimizer.h`.

---

## A. The exact DIRECT batch contract (R12 `batch_cost_(batch_centers)`)

**Per-iteration batch contents.** In `TrisectPotentiallyOptimal` (`direct_optimizer.cpp:238-321`), when
`batch_cost_` is set, one iteration collects the **changed-center evals of the whole iteration** into a
single `batch_centers` vector:
- Each potentially-optimal hyperbox (POH) produces **three** child boxes: **oc** (unchanged center, **no eval**),
  **A** (`+shift`), **B** (`-shift`).
- Only A and B are evaluated → the batch holds **2 × numPOH centers per iteration**, packed in POH-column order,
  `{A,B}` per box (`direct_optimizer.cpp:275-276`: `make_changed(+1); make_changed(-1);`).
- `batch_centers` holds **denormalized physical** points (`DenormalizeFromCenter`, line 269), matching what the
  injected serial `CostFunction` receives.

This matches the serial loop exactly: the serial path evaluates A then B per box with `EvaluateCostFunction`
(lines 365-382). The batch path sends the same centers in the same order, then replays `results` in the same
storage order. No additional/omitted evals.

**Ordering requirement — the algorithm needs ordered RESULTS, not ordered completion.**
DIRECT's iteration `k+1` selection logic reads the storage's per-column **minimum value and size**
(`data_.GetMinimumHyperboxValue` / `GetSizeStoredInColumn`, `direct_optimizer.cpp:186-192`). Those minima are
aggregates over the *set* of boxes present in each column; they do not depend on the order the box evals of
iteration `k` *completed*, only on the fact that they are all **present before** iteration `k+1` runs.
Within one iteration the A/B evals of different POH boxes are pairwise-independent point costs — there is no
intra-iteration data dependency (click-level: box i's A value never feeds box j's B).

Therefore the **required** DIRECT synchronization is a *per-iteration barrier* (all iteration-k changed evals
stored before iteration k+1's ConvexHull), NOT per-center synchronous cost nor per-center completion order.
`batch_cost_` is a single synchronous call inside `TrisectPotentiallyOptimal`, so the barrier is satisfied by
construction. Within the call, the executor may score the batch centers in **any order and any degree of
parallelism**, provided it returns **one score per input center, in input order**. That is precisely R12 and
what the header states (`direct_optimizer.h:30-40,143-147`: "results returned in INPUT ORDER; per-eval
bookkeeping … REPLAYS in that order").

**Answer: any-order/parallel, ordered-result replay (R12) is CORRECT and does not require per-iteration
sequential scoring.** No part of DIRECT collapses if the 2×POH centers of one iteration are scattered across
a pool and replayed by their input slots.

---

## B. With a concurrent executor returning input-ordered scores, is DIRECT still exact?

**Yes — same sequence of optimizer decisions, cost_function_calls, optimum, callbacks — provided the
request–reply mapping is input-indexed.** Trace:

- Replay store (`direct_optimizer.cpp:293-320`) iterates `pending` in the *serial* order `{oc, A, B}` per
  POH box. For each changed box it reads `results[cidx]` where `cidx` is the packing slot of that center in
  `batch_centers` (`direct_optimizer.cpp:300`). The `A`/`B` correct-center → correct-value association is
  fixed by that slot, independent of executor completion order.
- The storage sequence `AddHyperBox(oc) → AddHyperBox(A) → AddHyperBox(B)` (lines 297,311) is executed in
  exactly the order the serial path performs the same loads (`data_data_storage.cpp:334,365-382`), so the
  box-set and each that column's minimum after the iteration are identical to serial.
- `cost_function_calls_++`, non-finite deletes, `if (result < current_optimum_value_)` optimum update, and
  the improvement / iteration callbacks fire in the same order as the serial per-eval loop. Thus
  `GetCostFunctionCalls()`, `GetOptimumLocation/Value`, the callbacks, and the loop guard
  `(cost_function_calls_ + call_offset_) < budget_` advance identically.

The only part of DIRECT that "depends on all previous" is `convexHull` selection, which consumes only the
stored column minima — and those are barrier-fenced by the batch call. **Confirmed: a concurrent executor
violates nothing, as long as it returns an input-ordered scores vector and the whole batch completes within
the call.**

---

## C. Correctness seams a concurrent executor could violate

1. **The ONLY structural guard is the size check (`direct_optimizer.cpp:284`)** — `results.size() !=
   batch_centers.size()` → throw. There is **no check that the returned score is faithful to its input
   center** (only that the vector is the right length). If a future/broken executor returns a same-size but
   *reordered* (or wrong-center) score vector, `results[cidx]` silently commits the wrong value to a box →
   the box's value is wrong → the column minimum is poisoned → the next ConvexHull chooses the wrong
   potentially-optimal, possibly-optimal boxes → **DIRECT permanently diverges with no failure surfaced**.
   Since it is input-order contract that makes this safe, the Tier-0/Tier-1 oracle tests must compare the
   batch (concurrent-executor) result against the serial path **on a pool>1 configuration**, not just the
   serial fallback; otherwise the seam is invisible. (R13/AE2 bit-identical proof must cover the real
   concurrent executor, not its serial doppelgänger.)

2. **Shared mutable GPU pose / active-bank state must stay serialized.** `OptimizerManager`'s batch enqueue
   lambda (`optimizer_manager.cpp:1328-1343`) calls `gpu_principal_model_->SetCurrentPrimaryCameraPose(...)`
   then `stage_manager.EnqueueDirectDilationOnBank(bank)` which sets a global active-bank pointer via
   `TrySetActiveBank(&bank)` (`CostFunctionManager.cpp:382-407`). This shared state is safe **only because**
   `RunCostBatchGreedy` performs each pose's enqueue/complete serially and has already created the (possibly
   parallel) overlap is only the *metric compute* on per-bank streams (`cost_capacity_service.cu:390-421`).
   If the "concurrent world" ever issued multiple `SetCurrentPrimaryCameraPose`/enqueues in flight on
   different threads without an exclusive serialization of that mutation, one pose would overwrite another
   bank's rendered image → the wrong score attributed to the input slot. The concurrency must be
   compute-only; the enqueue/pose/bank mutation must remain mutually exclusive. Today's greedy loop already
   satisfies that — the seam is a *constraint* to carry into the design, not a current violation.

3. **Failure semantics: no graceful fallback.** `RunCostBatchGreedy` returns `{}` on any pool/enqueue/event
   failure (its two overloads, `cost_capacity_service.cu:318-356,376-421`). The optimizer converts that to a
   hard throw `contract violation, plan 010 U11` (`direct_optimizer.cpp:284-288`). A transient CUDA failure
   inside the concurrent path is therefore a stage abort — not a silently divergent result, but a behavioral
   change from the per-pose serial path (which would have scored that pose). This is documented/intended but
   worth flagging as an availability/crash seam for the orchestrator.

4. **`DirectDataStorage::AddHyperBox` minimum update is push/overwrite, not a recompute.**
   `direct_data_storage.cpp:107` (`minimum_value_Columns_[iterator_index] = new_box->value_`, marked
   "NOT SAFE") unconditionally overwrites the column minimum whenever a new box with a larger value is
   appended (lower_bound on values lands at end). This is a pre-existing serial-path quirk; the batch replay
   preserves the *same* serial AddHyperBox order (it is not a NEW concurrency divergence), but it means the
   storage's min invariant is order-sensitive to which box arrives with a higher value last. The batch
   correctness therefore depends on the replay keeping exactly the serial box order — which the code does
   (`direct_oraker.cpp:293-311`). No action required for correctness, note as residual robustness debt and
   it must not be "optimized" to a truly parallel box-store (that would break the column-min update).

5. **GPU-FP nondeterminism across banks** (advisory, low signal): independent CUDA streams computing the
   same metric could in principle yield run-to-run non-bit-exact floats even for the same pose. For the R13
   "bit-identical-concurrent" proof, the batch-equivalence oracle should run the concurrent executor twice
   and require exact (bit-for-bit) equal scores, else the classification of "batch == serial" must be
   tolerance-based, which is a different verification contract than the serial bit-exact claim.

---

## Verdict (answers)

- **A.** Batch = the full iteration's changed centers, 2 × numPOH, packed `{A,B}` per POH in POH order, one
  `batch_cost_(batch_centers)` call per iteration, results NEVER contain oc. DIRECT requires *ordered race*
  results (each score indexed to its input center) — **not** ordered completion; per-iteration barrier is
  structural.
- **B.** With an input-ordered score vector, DIRECT stays fully valid: the input-indexed replay
  (`results[cidx]`, `direct_or_optimizer.cpp:300`) reproduces the identical box values, column minima,
  optimizer decision sequence, cost function call count, and improvement/iteration callbacks as the serial
  path. Concurrency of the intra-iteration evals is sound.
- **C.** Order-faithfulness is unguarded (size-only check), so the onus is on the executor / oracle test
  (P1). Enqueue/pose/bank mutation must remain serial ('compute-only concurrency'). Hard-abort on
  silent-`{}` failure is a behavioral change. The serial replay order is load-bearing for the storage min
  update.

### Findings (review)

```json
{
  "findings": [
    {
      "lens": "cuda-correctness",
      "title": "DIRECT batch contract: ordered results, not ordered completion — the R12 replay is sound",
      "severity": "P3",
      "confidence": 100,
      "evidence": [
        "direct_optimizer.cpp:279-280 'Single batch call over the whole iteration's changed centers' + batch_centers pushes only make_changed(+1)/make_changed(-1) per loop (A/B; oc has no eval)",
        "direct_optimizer.h:30-40 'results returned in INPUT ORDER; per-eval bookkeeping REPLAYS in that order'",
        "ConvexHull reads only GetMinimumHyperboxValue/GetSizeStoredInColumn across stored columns — minima over the SET of boxes, order-insensitive, so any completion order/concurrency within the batch is acceptable as long as the iteration barriers"
      ],
      "owner": "maintainer",
      "suggestedFix": "document-per-iteration-barrier contract on batch_cost_ (already effective via the single synchronous call)"
    },
    {
      "lens": "cuda-correctness",
      "title": "Concurrent executor result order is never validated — only size — a shuffled score vector silently diverges DIRECT",
      "severity": "P1",
      "confidence": 75,
      "evidence": [
        "direct_optimizer.cpp:284-288 — 'if (results.size() != batch_centers.size()) … throw' is the only guard; no ownership/order-of-values/link to input centers check",
        "direct_optimizer.cpp:300 — 'const double result = results[cidx];' commits the score to the box by packing slot; a same-length but permuted vector mis-assigns every value and poisons the column minimum"
      ],
      "owner": "review-fixer",
      "suggestedFix": "in the batch-equivalence oracle (Tier-1/2), compare concurrent-batch output to serial-batch output with EXACT equality on a pool>1 config; not just the serial fallback. Do not add runtime per-entry validation in the hot loop unless feasible at build/admission time."
    },
    {
      "severity": "P1",
      "lens": "cuda-correctness",
      "title": "Shared mutable GPU pose + active-bank mutation must stay serialized — compute-only parallelism or scores render wrong images",
      "confidence": 75,
      "evidence": [
        "optimizer_manager.cpp:1331-1343 — enqueue lambda: 'gpu_principal_model_->SetCurrentPrimaryCameraPose(Pose(...)); Then stage_manager.EnqueueDirectDilationOnBank(bank)'",
        "cost_capacity_service.cu:380-421 — one pose enqueued at a time per bank lease; full parallel in-flight enqueue would overwrite the model pose mid-render"
      ],
      "owner": "maintainer",
      "suggestedFix": "keep the enqueue/pose/bank mutation single-threaded (serial) and overlap only the CUDA metric stream; assert / document this invariant rather than a truly parallel enqueue path."
    },
    {
      "severity": "P3",
      "confidence": 50,
      "title": "Storage column-minimum update is push/overwrite not recompute; the serial replay order is load-bearing",
      "evidence": ["direct_data_storage.cpp:107 — 'minimum_value_columns_[iterator_index]=new_box->value_ (NOT SAFE)' on push_back; the oc→A→B replay in serial order is load-bearing"],
      "owner": "maintainer",
      "suggestedFix": "No change for correctness now; document that AddHyperBox order is a hard invariant of the batch replay (do not parallelize box-store array writes)."
    },
    {
      "severity": "P2",
      "confidence": 75,
      "title": "Executor failure => hard abort, no serial fallback (behavioral change)",
      "evidence": [
        "cost_capacity_service.cu:384,419 'return {}' on enqueue/event failure; An empty vector at direct_optimizer.cpp:284-288 throws invalid_argument — stage abort",
        "serial path evaluates pose-by-pose; no hard abort on a single pose GPU failure"
      ],
      "owner": "review-fixer",
      "suggestedFix": "Decide intentionality: if a concurrent executor rejects a batch (pool exhausted/event failure), either fall back to the serial adapter for that iteration, or surface an explicit OptimizerError rather than the current parse-time throw."
    },
    {
      "severity": "P3",
      "confidence": 25,
      "title": "Cross-stream FP nondeterminism vs the 'bit-identical' proof (advisory)",
      "evidence": [
        "RunCostBatchGreedy overlaps metric compute per bank stream; no guarantee two streams produce exact-equal floats for a pose",
        "R13/AE2 'bit-identical-defaults' claim assumes deterministic per-pose cost"
      ],
      "owner": "release",
      "suggestedFix": "batch-equivalence oracle should require bit-exact when scoring the same center twice on pool>1; if nondeterministic, gate the bit-exact proof on the serial path."
    }
  ]
}
```

---

## Acceptance report