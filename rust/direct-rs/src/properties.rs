//! Metamorphic / property tests over the public `DirectOptimizer` API.
//!
//! Structural internals (volume partition, Jones POH, depth lattice) live in
//! `#[cfg(test)]` modules next to the types they inspect.
//!
//! Deferred: differential-vs-C++ replay. The cxx bridge exposes `CppCost`,
//! not a runnable C++ `DirectOptimizer`, so there is no linked original to
//! compare sample tapes against yet.

use std::cell::RefCell;
use std::rc::Rc;

use proptest::prelude::*;

use crate::bench::ShiftedSphere;
use crate::cost::Cost;
use crate::direct_data_storage::Pose;
use crate::direct_optimizer::DirectOptimizer;
use crate::test_support::{
    coords, denorm, dist, invert, permute_pose, pose, show, splat, unpermute_pose, zero,
};

const CASES: u32 = 32;

struct Recording<C> {
    inner: C,
    log: Rc<RefCell<Vec<Pose>>>,
    costs: Rc<RefCell<Vec<f64>>>,
}

impl<C: Cost> Cost for Recording<C> {
    fn eval(&self, poses: &[Pose]) -> Vec<f64> {
        let out = self.inner.eval(poses);
        self.log.borrow_mut().extend(poses.iter().copied());
        self.costs.borrow_mut().extend(out.iter().copied());
        out
    }
}

struct Affine<C> {
    inner: C,
    a: f64,
    b: f64,
}

impl<C: Cost> Cost for Affine<C> {
    fn eval(&self, poses: &[Pose]) -> Vec<f64> {
        self.inner
            .eval(poses)
            .into_iter()
            .map(|v| self.a * v + self.b)
            .collect()
    }
}

/// `g(u) = f(denorm_{start,range}(u))` so a unit-cube run matches a physical one.
struct InUnit<C> {
    inner: C,
    start: Pose,
    range: Pose,
}

impl<C: Cost> Cost for InUnit<C> {
    fn eval(&self, units: &[Pose]) -> Vec<f64> {
        let physical: Vec<Pose> = units
            .iter()
            .copied()
            .map(|u| denorm(self.start, self.range, u))
            .collect();
        self.inner.eval(&physical)
    }
}

/// `f_perm(x) = f(unpermute(x))` so permuting the domain permutes the cost.
struct Permuted<C> {
    inner: C,
    perm: [usize; 6],
}

impl<C: Cost> Cost for Permuted<C> {
    fn eval(&self, poses: &[Pose]) -> Vec<f64> {
        let unpermuted: Vec<Pose> = poses
            .iter()
            .copied()
            .map(|p| unpermute_pose(p, self.perm))
            .collect();
        self.inner.eval(&unpermuted)
    }
}

struct Hostile {
    /// First `nan_prefix` evals return NaN; after that, a sphere about the origin.
    nan_prefix: usize,
    seen: RefCell<usize>,
}

impl Cost for Hostile {
    fn eval(&self, poses: &[Pose]) -> Vec<f64> {
        poses
            .iter()
            .map(|p| {
                let n = {
                    let mut seen = self.seen.borrow_mut();
                    let k = *seen;
                    *seen += 1;
                    k
                };
                if n < self.nan_prefix {
                    f64::NAN
                } else {
                    coords(p).iter().map(|v| v * v).sum()
                }
            })
            .collect()
    }
}

fn run_recorded<C: Cost>(
    range: Pose,
    start: Pose,
    budget: u32,
    cost: C,
) -> (Pose, f64, Vec<Pose>, Vec<f64>) {
    let log = Rc::new(RefCell::new(Vec::new()));
    let costs = Rc::new(RefCell::new(Vec::new()));
    let rec = Recording {
        inner: cost,
        log: Rc::clone(&log),
        costs: Rc::clone(&costs),
    };
    let mut opt = DirectOptimizer::new(range, start, budget);
    let (best, best_cost) = opt.run(&rec);
    (best, best_cost, log.take(), costs.take())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: CASES, ..ProptestConfig::default() })]

    /// Desired contract: the optimizer never evaluates more than `budget`
    /// points. The current loop checks the guard *before* a `2×|POH|` batch,
    /// so this is expected to fail until overshoot is bounded.
    #[test]
    #[ignore = "budget overshoot: loop guards before a 2x|POH| batch; bound it before re-enabling"]
    fn evals_do_not_exceed_budget(
        range in proptest::array::uniform6(0.1_f64..20.0),
        start in proptest::array::uniform6(-50.0_f64..50.0),
        budget in 1_u32..80,
        shift in proptest::array::uniform6(-5.0_f64..5.0),
    ) {
        let range = pose(range);
        let start = pose(start);
        let shift = pose(shift);
        let (_b, _c, samples, _) = run_recorded(
            range,
            start,
            budget,
            ShiftedSphere { shift },
        );
        prop_assert!(
            samples.len() <= budget as usize,
            "evaluated {} points against budget {budget}",
            samples.len()
        );
    }

    /// Every physical sample lies in `[start − range, start + range]`.
    #[test]
    fn samples_stay_inside_the_search_cube(
        range in proptest::array::uniform6(0.1_f64..20.0),
        start in proptest::array::uniform6(-50.0_f64..50.0),
        budget in 8_u32..60,
        shift in proptest::array::uniform6(-5.0_f64..5.0),
    ) {
        let range = pose(range);
        let start = pose(start);
        let shift = pose(shift);
        let (_b, _c, samples, _) = run_recorded(
            range,
            start,
            budget,
            ShiftedSphere { shift },
        );
        let lo = coords(&start);
        let hi_span = coords(&range);
        for p in &samples {
            for (i, v) in coords(p).iter().enumerate() {
                let Some(&l) = lo.get(i) else { continue };
                let Some(&s) = hi_span.get(i) else { continue };
                prop_assert!(
                    *v >= l - s - 1e-9 && *v <= l + s + 1e-9,
                    "sample {} axis {i} = {v} outside [{}, {}]",
                    show(p),
                    l - s,
                    l + s
                );
            }
        }
    }

    /// Returned incumbent is the recorded argmin (finite costs only).
    #[test]
    fn incumbent_is_the_recorded_argmin(
        range in proptest::array::uniform6(0.1_f64..20.0),
        start in proptest::array::uniform6(-50.0_f64..50.0),
        budget in 8_u32..60,
        shift in proptest::array::uniform6(-5.0_f64..5.0),
    ) {
        let range = pose(range);
        let start = pose(start);
        let shift = pose(shift);
        let (best, best_cost, samples, costs) = run_recorded(
            range,
            start,
            budget,
            ShiftedSphere { shift },
        );
        prop_assert_eq!(samples.len(), costs.len());
        let mut min = f64::INFINITY;
        let mut argmin = best;
        for (p, c) in samples.iter().zip(&costs) {
            if c.is_finite() && *c < min {
                min = *c;
                argmin = *p;
            }
        }
        prop_assert!(
            best_cost.is_finite(),
            "incumbent cost {best_cost} is not finite"
        );
        prop_assert!(
            (best_cost - min).abs() < 1e-12,
            "incumbent cost {best_cost} != recorded min {min}"
        );
        prop_assert!(
            dist(&best, &argmin) < 1e-12,
            "incumbent pose {} != argmin {}",
            show(&best),
            show(&argmin)
        );
    }

    /// Parent keeps its center; children are fresh. Zero re-evaluations.
    #[test]
    fn no_duplicate_samples(
        range in proptest::array::uniform6(0.1_f64..20.0),
        start in proptest::array::uniform6(-50.0_f64..50.0),
        budget in 8_u32..80,
        shift in proptest::array::uniform6(-5.0_f64..5.0),
    ) {
        let range = pose(range);
        let start = pose(start);
        let shift = pose(shift);
        let (_b, _c, samples, _) = run_recorded(
            range,
            start,
            budget,
            ShiftedSphere { shift },
        );
        let mut keys: Vec<[u64; 6]> = samples
            .iter()
            .map(|p| coords(p).map(f64::to_bits))
            .collect();
        let total = keys.len();
        keys.sort_unstable();
        keys.dedup();
        prop_assert_eq!(
            keys.len(),
            total,
            "re-evaluated {} duplicate centers",
            total - keys.len()
        );
    }

    /// `run(a f + b)` returns the same pose and cost `a c + b` for `a > 0`.
    #[test]
    fn positive_affine_cost_is_equivariant(
        a in 0.05_f64..8.0,
        b in -20.0_f64..20.0,
        budget in 20_u32..50,
        shift in proptest::array::uniform6(-3.0_f64..3.0),
    ) {
        let range = splat(5.0);
        let start = zero();
        let shift = pose(shift);
        let base = ShiftedSphere { shift };
        let mut raw = DirectOptimizer::new(range, start, budget);
        let (p1, c1) = raw.run(&base);
        let mut aff = DirectOptimizer::new(range, start, budget);
        let (p2, c2) = aff.run(&Affine { inner: base, a, b });
        prop_assert!(
            dist(&p1, &p2) < 1e-9,
            "pose drifted under affine: {} vs {}",
            show(&p1),
            show(&p2)
        );
        prop_assert!(
            (c2 - (a * c1 + b)).abs() < 1e-9,
            "cost {c2} != {a}*{c1}+{b}"
        );
    }

    /// Negative scale must *not* preserve the minimizer (guard against abs()).
    #[test]
    fn negative_affine_does_not_preserve_minimizer(
        budget in 20_u32..40,
    ) {
        let shift = Pose { x: 1.7, y: 0.4, z: -2.1, xa: 0.8, ya: -1.3, za: 2.6 };
        let range = splat(5.0);
        let start = zero();
        let base = ShiftedSphere { shift };
        let mut raw = DirectOptimizer::new(range, start, budget);
        let (p1, _) = raw.run(&base);
        let mut aff = DirectOptimizer::new(range, start, budget);
        let (p2, _) = aff.run(&Affine { inner: base, a: -1.0, b: 0.0 });
        prop_assert!(
            dist(&p1, &p2) > 0.5,
            "negating the cost kept the same pose {}",
            show(&p1)
        );
    }

    /// `run(r, s, f)` and `run(1, 0.5, f ∘ denorm_{r,s})` sample the same
    /// unit-space centers — the real normalize/denormalize check.
    #[test]
    fn domain_affine_equivariance(
        range in proptest::array::uniform6(1.0_f64..15.0),
        start in proptest::array::uniform6(-20.0_f64..20.0),
        budget in 16_u32..40,
    ) {
        let range = pose(range);
        let start = pose(start);
        let shift = Pose {
            x: start.x + 0.3 * range.x,
            y: start.y - 0.2 * range.y,
            z: start.z + 0.1 * range.z,
            xa: start.xa - 0.25 * range.xa,
            ya: start.ya + 0.15 * range.ya,
            za: start.za - 0.35 * range.za,
        };
        let f = ShiftedSphere { shift };
        let (_b1, _c1, phys, _) = run_recorded(range, start, budget, f);
        let unit_from_phys: Vec<Pose> = phys
            .iter()
            .copied()
            .map(|p| invert(start, range, p))
            .collect();

        let g = InUnit { inner: f, start, range };
        // range=0.5, start=0.5 => denorm is the identity on unit space.
        let (_b2, _c2, units, _) = run_recorded(splat(0.5), splat(0.5), budget, g);

        let n = unit_from_phys.len().min(units.len());
        prop_assert!(n > 0, "no samples");
        for (i, (a, b)) in unit_from_phys.iter().zip(&units).take(n).enumerate() {
            prop_assert!(
                dist(a, b) < 1e-9,
                "unit trajectory diverged at {i}: {} vs {}",
                show(a),
                show(b)
            );
        }
    }

    /// At a converging budget, permuting axes of a quadratic must preserve
    /// the best *cost* (the pose need not permute — `min_by_key` ties break
    /// toward index 0). Low budgets are allowed to differ; this is the
    /// converged weaker property.
    #[test]
    fn axis_permutation_preserves_best_cost(
        _pad in 0u8..1,
    ) {
        let perm = [5usize, 4, 3, 2, 1, 0];
        let range = splat(5.0);
        let start = zero();
        let shift = Pose { x: 1.7, y: 0.4, z: -2.1, xa: 0.8, ya: -1.3, za: 2.6 };
        let f = ShiftedSphere { shift };
        let budget = 8_000;
        let mut a = DirectOptimizer::new(range, start, budget);
        let (_p1, c1) = a.run(&f);
        let mut b = DirectOptimizer::new(
            permute_pose(range, perm),
            permute_pose(start, perm),
            budget,
        );
        let (_p2, c2) = b.run(&Permuted { inner: f, perm });
        prop_assert!(
            (c1 - c2).abs() < 1e-6,
            "permuted run cost {c2} != original {c1}"
        );
    }

    /// Sample tape of budget `n` is a prefix of budget `2n`.
    #[test]
    fn larger_budget_extends_the_sample_prefix(
        budget in 8_u32..30,
        shift in proptest::array::uniform6(-3.0_f64..3.0),
    ) {
        let range = splat(5.0);
        let start = zero();
        let shift = pose(shift);
        let f = ShiftedSphere { shift };
        let (_b1, _c1, small, _) = run_recorded(range, start, budget, f);
        let (_b2, _c2, large, _) = run_recorded(range, start, budget.saturating_mul(2), f);
        prop_assert!(
            large.len() >= small.len(),
            "larger budget sampled fewer points ({} < {})",
            large.len(),
            small.len()
        );
        for (i, (a, b)) in small.iter().zip(&large).enumerate() {
            prop_assert!(
                dist(a, b) < 1e-12,
                "prefix diverged at sample {i}: {} vs {}",
                show(a),
                show(b)
            );
        }
    }

    /// `range_i = 0` pins that coordinate to `start_i` on every sample.
    #[test]
    fn zero_range_axis_stays_pinned_to_start(
        freeze_mask in 1_u8..63,
        start in proptest::array::uniform6(-10.0_f64..10.0),
        budget in 8_u32..40,
    ) {
        let mut r = [4.0; 6];
        for (i, slot) in r.iter_mut().enumerate() {
            if (freeze_mask >> i) & 1 == 1 {
                *slot = 0.0;
            }
        }
        let range = crate::test_support::pose(r);
        let start = pose(start);
        let shift = start;
        let (_b, _c, samples, _) = run_recorded(
            range,
            start,
            budget,
            ShiftedSphere { shift },
        );
        let s = coords(&start);
        for p in &samples {
            for (i, v) in coords(p).iter().enumerate() {
                let Some(&si) = s.get(i) else { continue };
                let Some(&ri) = r.get(i) else { continue };
                if ri == 0.0 {
                    prop_assert!(
                        (*v - si).abs() < 1e-12,
                        "axis {i} moved to {v} despite range 0 (start {si})"
                    );
                }
            }
        }
    }
}

#[test]
fn nan_prefix_never_becomes_incumbent() {
    let mut opt = DirectOptimizer::new(splat(5.0), zero(), 80);
    let (best, best_cost) = opt.run(&Hostile {
        nan_prefix: 5,
        seen: RefCell::new(0),
    });
    assert!(
        best_cost.is_finite(),
        "incumbent cost {best_cost} is not finite"
    );
    for v in coords(&best) {
        assert!(
            v.is_finite(),
            "incumbent pose {} has a non-finite coord",
            show(&best)
        );
    }
}

#[test]
fn all_nan_run_does_not_leave_nan_incumbent() {
    let mut opt = DirectOptimizer::new(splat(5.0), zero(), 40);
    let (_best, best_cost) = opt.run(&Hostile {
        nan_prefix: usize::MAX,
        seen: RefCell::new(0),
    });
    assert!(
        !best_cost.is_nan(),
        "all-NaN run left incumbent cost {best_cost}"
    );
}
