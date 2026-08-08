---
module: jta_cost_function
date: 2026-08-08
problem_type: logic_error
component: tooling
severity: high
symptoms:
  - "fractional double cost parameters were silently truncated on storage (set 15.5, read back 15.0)"
  - "the jtml.cost_function hegel PBT round-trip invariant failed ('got == v' with 0.0 == 0.5)"
  - "cost-tuning UI values lost fractional precision in the parameter registry"
root_cause: logic_error
resolution_type: code_fix
tags:
  - cost-function
  - parameter
  - truncation
  - hegel
  - pbt
related_components:
  - testing
---

# Cost-function parameter registry silently truncated doubles to int

## Problem

In `jta_cost_function::Parameter<double>` (now part of the merged `jtml_compute`
lib), the stored value field was declared `int`, so any double parameter was
truncated to an integer when set — a fractional cost-tuning value was silently
corrupted.

## Symptoms

- A hegel PBT round-trip invariant failed: `setDoubleParameterValue(name, 15.5)`
  then `getDoubleParameterValue(name, got)` returned `15.0`, not `15.5`.
- Deterministic round-trip passed for integer-valued doubles, hiding the bug from
  example-based tests (values like `20.0` happen to round-trip through an int).

## What Didn't Work

Truncating the test expectation ("assert within 1.0") would have baked the bug in.
The right move was to treat the failing invariant as a genuine defect — the double
setter/ctor make the intended type unambiguous.

## Solution

`include/compute/Parameter.h`, `Parameter<double>`:

```cpp
// before
int parameter_value_;
// after
double parameter_value_;   // setter/ctor already take double; int was a typo-class bug
```

Forced-recompile the stale `jtml_compute.so` (a reused pre-split `.build` tree hid
the dependency once — `ninja` reported "no work to do" until the object was rebuilt;
a clean build dir/CI does not hit this).

## Why This Works

The field is now the same type as the accessor (`getParameterValue` returns
`double`, `setParameterValue` takes `double`), so storage is lossless. The bug class
was a silent narrowing that example tests could not see because integer-valued
doubles round-trip identically through an `int`.

## Prevention

Add a hegel PBT that asserts **bit-exact** typed round-trip (not approximate), with
draws spanning negatives, `±0.0`, and fractional values — these are the inputs that
catch silent narrowing. The `jtml.cost_function_props` target now does exactly this
for all three registry types (double/int/bool), plus unknown-name and cross-type
rejection. When a property test whose invariant is clearly right fails against
existing code, treat the code as the bug — do not weaken the invariant to "pass".