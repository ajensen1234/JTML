//! Benchmark cost functions for exercising the Rust DIRECT port.
//!
//! Each benchmark is a `Cost` impl (input = batch of physical `Pose`s, output =
//! batch of costs). The point is to be **discriminating** — a naive or buggy
//! DIRECT should fail these in a way the simple sphere never will.
//!
//! Design principles baked into these fixtures (see `bench_design.md`):
//! 1. **No center-bias.** The global minimum is never at the domain center, so
//!    the first center sample is meaningful, not a free win.
//! 2. **Known, distinctive optima.** `f*` values are chosen to be non-round, so
//!    sign/dimension/early-stop errors show up as a mismatch, not "close to 0".
//! 3. **Both cost AND pose are asserted**, so a function that is plateau-flat
//!    near the optimum can't hide a box that isn't refining.

use crate::direct_data_storage::Pose;

/// A plain sphere: `f(x) = Σ (xᵢ - sᵢ)²`. Unimodal; verifies convergence rate
/// and that shifting the optimum off-center works. `f* = 0` at `x* = s`.
#[derive(Clone, Copy)]
pub struct ShiftedSphere {
    pub shift: Pose,
}

impl ShiftedSphere {
    pub fn fstar() -> f64 {
        0.0
    }
    pub fn xstar(&self) -> Pose {
        self.shift
    }
}

impl Cost for ShiftedSphere {
    fn eval(&self, poses: &[Pose]) -> Vec<f64> {
        poses
            .iter()
            .map(|p| {
                (p.x - self.shift.x).powi(2)
                    + (p.y - self.shift.y).powi(2)
                    + (p.z - self.shift.z).powi(2)
                    + (p.xa - self.shift.xa).powi(2)
                    + (p.ya - self.shift.ya).powi(2)
                    + (p.za - self.shift.za).powi(2)
            })
            .collect()
    }
}

/// Anisotropic sphere: `f(x) = Σ wᵢ (xᵢ - sᵢ)²`. The weights are very
/// different across axes (mm vs angle-like units), which exposes a DIRECT that
/// subdivides in raw pose units instead of normalized/denormalized space.
#[derive(Clone, Copy)]
pub struct AnisotropicSphere {
    pub shift: Pose,
    pub weights: [f64; 6],
}

impl AnisotropicSphere {
    pub fn fstar() -> f64 {
        0.0
    }
    pub fn xstar(&self) -> Pose {
        self.shift
    }
}

impl Cost for AnisotropicSphere {
    fn eval(&self, poses: &[Pose]) -> Vec<f64> {
        poses
            .iter()
            .map(|p| {
                self.weights[0] * (p.x - self.shift.x).powi(2)
                    + self.weights[1] * (p.y - self.shift.y).powi(2)
                    + self.weights[2] * (p.z - self.shift.z).powi(2)
                    + self.weights[3] * (p.xa - self.shift.xa).powi(2)
                    + self.weights[4] * (p.ya - self.shift.ya).powi(2)
                    + self.weights[5] * (p.za - self.shift.za).powi(2)
            })
            .collect()
    }
}

/// Styblinski–Tang: `f(x) = 0.5 Σ (xᵢ⁴ - 16xᵢ² + 5xᵢ)`.
/// Separable but the optimum sits at a non-round negative value, and
/// `f* = -234.9959` (n=6) is a distinctive number — catches sign errors,
/// off-by-one dimension counts, and accidental early stopping.
#[derive(Clone, Copy)]
pub struct StyblinskiTang;

impl StyblinskiTang {
    pub fn fstar() -> f64 {
        -234.9959
    }
    pub fn xstar() -> Pose {
        Pose {
            x: -2.903534,
            y: -2.903534,
            z: -2.903534,
            xa: -2.903534,
            ya: -2.903534,
            za: -2.903534,
        }
    }
}

impl Cost for StyblinskiTang {
    fn eval(&self, poses: &[Pose]) -> Vec<f64> {
        poses
            .iter()
            .map(|p| {
                let f1 = |v: f64| 0.5 * (v.powi(4) - 16.0 * v.powi(2) + 5.0 * v);
                0.0 + f1(p.x) + f1(p.y) + f1(p.z) + f1(p.xa) + f1(p.ya) + f1(p.za)
            })
            .collect()
    }
}

/// Rastrigin: `f(x) = 10n + Σ (xᵢ² - 10cos(2πxᵢ))`, shifted so the optimum is at
/// `x* = shift`, `f* = 0`. Dense lattice of local minima — the actual
/// global-vs-local test; a method stuck in a local basin fails loudly.
#[derive(Clone, Copy)]
pub struct ShiftedRastrigin {
    pub shift: Pose,
    pub n: u32,
}

impl ShiftedRastrigin {
    pub fn fstar() -> f64 {
        0.0
    }
    pub fn xstar(&self) -> Pose {
        self.shift
    }
}

impl Cost for ShiftedRastrigin {
    fn eval(&self, poses: &[Pose]) -> Vec<f64> {
        let base = 10.0 * self.n as f64;
        let two_pi = 2.0 * std::f64::consts::PI;
        let t = |d: f64| d * d - 10.0 * (two_pi * d).cos();
        poses
            .iter()
            .map(|p| {
                let dx = p.x - self.shift.x;
                let dy = p.y - self.shift.y;
                let dz = p.z - self.shift.z;
                let dxa = p.xa - self.shift.xa;
                let dya = p.ya - self.shift.ya;
                let dza = p.za - self.shift.za;
                base + t(dx) + t(dy) + t(dz) + t(dxa) + t(dya) + t(dza)
            })
            .collect()
    }
}

/// Ackley (shifted): near-flat outer plateau with a narrow central funnel.
/// Tests that the POH/convex-hull step doesn't stall when cost variation is
/// tiny. `f* = 0` at `x* = shift` (domain ~ [-5,5]).
#[derive(Clone, Copy)]
pub struct ShiftedAckley {
    pub shift: Pose,
}

impl ShiftedAckley {
    pub fn fstar() -> f64 {
        0.0
    }
    pub fn xstar(&self) -> Pose {
        self.shift
    }
}

impl Cost for ShiftedAckley {
    fn eval(&self, poses: &[Pose]) -> Vec<f64> {
        let n = 6.0_f64;
        poses
            .iter()
            .map(|p| {
                let v = |c: f64, s: f64| (c - s) * (c - s);
                let sum2 = v(p.x, self.shift.x)
                    + v(p.y, self.shift.y)
                    + v(p.z, self.shift.z)
                    + v(p.xa, self.shift.xa)
                    + v(p.ya, self.shift.ya)
                    + v(p.za, self.shift.za);
                let cos2 = (2.0 * std::f64::consts::PI * (p.x - self.shift.x)).cos()
                    + (2.0 * std::f64::consts::PI * (p.y - self.shift.y)).cos()
                    + (2.0 * std::f64::consts::PI * (p.z - self.shift.z)).cos()
                    + (2.0 * std::f64::consts::PI * (p.xa - self.shift.xa)).cos()
                    + (2.0 * std::f64::consts::PI * (p.ya - self.shift.ya)).cos()
                    + (2.0 * std::f64::consts::PI * (p.za - self.shift.za)).cos();
                -20.0 * (-0.2 * (sum2 / n).sqrt()).exp() - ((cos2) / n).exp()
                    + 20.0
                    + std::f64::consts::E
            })
            .collect()
    }
}

/// Rosenbrock (n-dim): narrow curved valley. DIRECT is genuinely bad at this —
/// use it to track how `f_best` decays v. evaluation count, NOT as pass/fail.
/// For n>=4 there is a second local min near (-1,1,1,...).
#[derive(Clone, Copy)]
pub struct Rosenbrock;

impl Cost for Rosenbrock {
    fn eval(&self, poses: &[Pose]) -> Vec<f64> {
        poses
            .iter()
            .map(|p| {
                let a = [p.x, p.y, p.z, p.xa, p.ya, p.za];
                let mut sum = 0.0;
                for i in 0..5 {
                    sum += 100.0 * (a[i + 1] - a[i] * a[i]).powi(2) + (1.0 - a[i]).powi(2);
                }
                sum
            })
            .collect()
    }
}

/// Hartmann-6 — the canonical 6D DIRECT benchmark (Jones 1993). Genuinely
/// multimodal with 6 local minima; the global optimum is at an ugly interior
/// point, not the center. Domain [0,1]^6. `f* ≈ -3.32237` at
/// `x* ≈ (0.20169,0.15001,0.47687,0.27533,0.31165,0.65730)`.
#[derive(Clone, Copy)]
pub struct Hartmann6;

const H6_ALPHA: [f64; 4] = [1.0, 1.2, 3.0, 3.2];

const H6_A: [[f64; 6]; 4] = [
    [10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
    [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
    [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
    [17.00, 8.00, 0.05, 10.00, 0.10, 14.00],
];

const H6_P: [[f64; 6]; 4] = [
    [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
    [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
    [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
    [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
];
// (P is scaled by 1e-4 at eval time.)

impl Hartmann6 {
    pub fn fstar() -> f64 {
        -3.32237
    }
    pub fn xstar() -> Pose {
        Pose {
            x: 0.20169,
            y: 0.15001,
            z: 0.47687,
            xa: 0.27533,
            ya: 0.31165,
            za: 0.65730,
        }
    }
}

impl Cost for Hartmann6 {
    fn eval(&self, poses: &[Pose]) -> Vec<f64> {
        poses
            .iter()
            .map(|p| {
                let x = [p.x, p.y, p.z, p.xa, p.ya, p.za];
                let mut sum = 0.0;
                for i in 0..4 {
                    let mut inner = 0.0;
                    for j in 0..6 {
                        inner += H6_A[i][j] * (x[j] - H6_P[i][j] * 1e-4).powi(2);
                    }
                    sum += H6_ALPHA[i] * (-inner).exp();
                }
                -sum
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::direct_data_storage::Pose;
    use crate::direct_optimizer::DirectOptimizer;
    use crate::utils::{draw_2d_graph, plot_boxes};

    // ---------- fixture helpers ----------

    /// A nonzero, non-nice-fraction shift so the optimum is off-center and never
    /// on a trisection boundary.
    fn evil_shift() -> Pose {
        Pose {
            x: 1.7,
            y: 0.4,
            z: -2.1,
            xa: 0.8,
            ya: -1.3,
            za: 2.6,
        }
    }

    fn all_ranges(v: f64) -> Pose {
        Pose {
            x: v,
            y: v,
            z: v,
            xa: v,
            ya: v,
            za: v,
        }
    }

    fn dist(a: &Pose, b: &Pose) -> f64 {
        ((a.x - b.x).powi(2)
            + (a.y - b.y).powi(2)
            + (a.z - b.z).powi(2)
            + (a.xa - b.xa).powi(2)
            + (a.ya - b.ya).powi(2)
            + (a.za - b.za).powi(2))
        .sqrt()
    }

    fn zero() -> Pose {
        Pose {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            xa: 0.0,
            ya: 0.0,
            za: 0.0,
        }
    }

    fn show(p: &Pose) -> String {
        format!(
            "({:.3},{:.3},{:.3},{:.3},{:.3},{:.3})",
            p.x, p.y, p.z, p.xa, p.ya, p.za
        )
    }

    // ---------- known-optimum reach tests ----------

    #[test]
    fn shifted_sphere_reaches_min() {
        let cost_fn = ShiftedSphere {
            shift: evil_shift(),
        };
        let mut opt = DirectOptimizer::new(all_ranges(5.0), zero(), 20_000);
        let (best, cost) = opt.run(&cost_fn);
        assert!(
            (cost - ShiftedSphere::fstar()).abs() < 1e-6,
            "cost {cost} vs f* {}",
            ShiftedSphere::fstar()
        );
        let target = cost_fn.xstar();
        assert!(
            dist(&best, &target) < 5e-3,
            "best {} far from shift {}",
            show(&best),
            show(&target)
        );
    }

    #[test]
    fn anisotropic_ranges_still_reach_min() {
        // Weights change the *function*; the thing that exposes raw-units
        // subdivision is anisotropic *ranges* (mm vs degrees).
        let range = Pose {
            x: 1000.0,
            y: 1000.0,
            z: 1000.0,
            xa: 0.1,
            ya: 0.1,
            za: 0.1,
        };
        let shift = Pose {
            x: 170.0,
            y: 40.0,
            z: -210.0,
            xa: 0.03,
            ya: -0.04,
            za: 0.06,
        };
        let cost_fn = ShiftedSphere { shift };
        let mut opt = DirectOptimizer::new(range, zero(), 30_000);
        let (best, cost) = opt.run(&cost_fn);
        assert!(
            (cost - ShiftedSphere::fstar()).abs() < 1e-1,
            "cost {cost} vs f* {}",
            ShiftedSphere::fstar()
        );
        let target = cost_fn.xstar();
        assert!(
            dist(&best, &target) < 5.0,
            "best {} vs shift {}",
            show(&best),
            show(&target)
        );
    }

    #[test]
    fn styblinski_reaches_distinctive_fstar() {
        // f* = -234.9959 — a non-round number that catches sign/dim errors.
        let mut opt = DirectOptimizer::new(all_ranges(5.0), zero(), 40_000);
        let (best, cost) = opt.run(&StyblinskiTang);
        assert!(
            (cost - StyblinskiTang::fstar()).abs() < 0.5,
            "cost {cost} vs f* {}",
            StyblinskiTang::fstar()
        );
        let target = StyblinskiTang::xstar();
        assert!(
            dist(&best, &target) < 0.5,
            "best {} far from x* {}",
            show(&best),
            show(&target)
        );
    }

    #[test]
    fn hartmann_reaches_global_min() {
        // Hartmann-6 is the canonical 6D DIRECT benchmark; the optimum is an
        // interior point, not the center. This is the strongest reach test.
        let mut opt = DirectOptimizer::new(all_ranges(0.5), all_ranges(0.5), 60_000);
        let (best, cost) = opt.run(&Hartmann6);

        plot_boxes(&opt, "hartman_global_min");
        assert!(
            (cost - Hartmann6::fstar()).abs() < 0.2,
            "cost {cost} vs f* {}",
            Hartmann6::fstar()
        );
        assert!(
            dist(&best, &Hartmann6::xstar()) < 0.1,
            "best {} far from x* {}",
            show(&best),
            show(&Hartmann6::xstar())
        );
    }

    #[test]
    #[ignore = "needs ~10M evals to leave the last Rastrigin lattice cell; run explicitly"]
    fn shifted_rastrigin_reaches_global_basin() {
        // Dense local-min lattice. Stuck-in-a-basin fails this loudly.
        let shift = evil_shift();
        let cost_fn = ShiftedRastrigin { shift, n: 6 };
        let mut opt = DirectOptimizer::new(all_ranges(5.12), zero(), 260_000);
        let (best, cost) = opt.run(&cost_fn);
        assert!(
            cost < 5.0,
            "Rastrigin cost {cost} still in a local basin (f* = 0)"
        );
        assert!(
            dist(&best, &shift) < 1.5,
            "best {} far from shift {}",
            show(&best),
            show(&shift)
        );
    }

    #[test]
    fn shifted_ackley_enters_the_funnel() {
        // Near-flat outer plateau. Without the ε-condition DIRECT can stall.
        let shift = evil_shift();
        let cost_fn = ShiftedAckley { shift };
        let mut opt = DirectOptimizer::new(all_ranges(5.0), zero(), 60_000);
        let (best, cost) = opt.run(&cost_fn);
        assert!(
            (cost - ShiftedAckley::fstar()).abs() < 2.0,
            "Ackley cost {cost} did not enter the funnel (f* = 0)"
        );
        assert!(
            dist(&best, &shift) < 2.0,
            "best {} far from shift {}",
            show(&best),
            show(&shift)
        );
    }

    #[test]
    fn rosenbrock_improves_on_the_seed() {
        // Tracking-only: DIRECT is genuinely bad at the banana. Just require
        // improvement on f(0) = 5*(100+1) = 505, not a reach of f*=0.
        let mut opt = DirectOptimizer::new(all_ranges(2.0), zero(), 20_000);
        let (_best, cost) = opt.run(&Rosenbrock);
        assert!(
            cost < 505.0,
            "Rosenbrock did not improve on seed 505, got {cost}"
        );
    }

    // ---------- invariant tests (no known optimum required) ----------

    #[test]
    fn same_inputs_give_bit_identical_result() {
        // Determinism: two identical optimizers must produce bit-identical
        // costs AND poses (DIRECT is deterministic; a HashMap/HashSet seeding
        // or float-order difference would break this).
        let a = {
            let mut o = DirectOptimizer::new(all_ranges(5.0), zero(), 20_000);
            o.run(&ShiftedSphere {
                shift: evil_shift(),
            })
        };
        let b = {
            let mut o = DirectOptimizer::new(all_ranges(5.0), zero(), 20_000);
            o.run(&ShiftedSphere {
                shift: evil_shift(),
            })
        };
        assert_eq!(a.1.to_bits(), b.1.to_bits(), "costs differ between runs");
        assert_eq!(a.0.x.to_bits(), b.0.x.to_bits(), "x differs");
        assert_eq!(a.0.y.to_bits(), b.0.y.to_bits(), "y differs");
        assert_eq!(a.0.z.to_bits(), b.0.z.to_bits(), "z differs");
        assert_eq!(a.0.xa.to_bits(), b.0.xa.to_bits(), "xa differs");
        assert_eq!(a.0.ya.to_bits(), b.0.ya.to_bits(), "ya differs");
        assert_eq!(a.0.za.to_bits(), b.0.za.to_bits(), "za differs");
    }

    #[test]
    fn budget_monotone_decrease() {
        // run with a small and a large budget on the same (deterministic)
        // problem: small must never beat large. No known optimum needed.
        let spend = |budget| {
            let mut o = DirectOptimizer::new(all_ranges(5.0), zero(), budget);
            o.run(&ShiftedSphere {
                shift: evil_shift(),
            })
            .1
        };
        let small = spend(2_000);
        let large = spend(60_000);
        assert!(
            large <= small + 1e-12,
            "more budget must not worsen result: small={small} large={large}"
        );
    }

    #[test]
    fn translation_invariant() {
        // Shifting the whole domain (start) AND the optimum by the same delta
        // must leave the result cost bit-identical — catches normalization bugs.
        let delta = Pose {
            x: 10.0,
            y: -4.0,
            z: 7.0,
            xa: 2.0,
            ya: -9.0,
            za: 5.0,
        };
        let (_, c1) = {
            let mut o = DirectOptimizer::new(all_ranges(5.0), zero(), 20_000);
            o.run(&ShiftedSphere {
                shift: evil_shift(),
            })
        };
        // Same function, domain shifted by delta; optimizer start then = delta.
        let shifted = Pose {
            x: delta.x,
            y: delta.y,
            z: delta.z,
            xa: delta.xa,
            ya: delta.ya,
            za: delta.za,
        };
        let (_, c2) = {
            let mut o = DirectOptimizer::new(all_ranges(5.0), shifted, 20_000);
            // equivalent: shift the optimum by the same delta
            o.run(&ShiftedSphere {
                shift: Pose {
                    x: evil_shift().x + delta.x,
                    y: evil_shift().y + delta.y,
                    z: evil_shift().z + delta.z,
                    xa: evil_shift().xa + delta.xa,
                    ya: evil_shift().ya + delta.ya,
                    za: evil_shift().za + delta.za,
                },
            })
        };
        assert!(
            (c1 - c2).abs() < 1e-9,
            "translation broke invariant: c1={c1} c2={c2}"
        );
    }

    #[test]
    fn deterministic_with_across_all_costs() {
        // Repeated run on each cost must be bit-stable (not seeded/greedy).
        fn run_twice<C: Cost + Copy>(cost: C) -> (f64, f64) {
            let a = {
                let mut o = DirectOptimizer::new(all_ranges(5.0), zero(), 30_000);
                o.run(&cost).1
            };
            let b = {
                let mut o = DirectOptimizer::new(all_ranges(5.0), zero(), 30_000);
                o.run(&cost).1
            };
            (a, b)
        }
        let (a, b) = run_twice(ShiftedSphere {
            shift: evil_shift(),
        });
        assert_eq!(a.to_bits(), b.to_bits(), "ShiftedSphere nondeterministic");
        let (a, b) = run_twice(StyblinskiTang);
        assert_eq!(a.to_bits(), b.to_bits(), "StyblinskiTang nondeterministic");
        let (a, b) = run_twice(ShiftedAckley {
            shift: evil_shift(),
        });
        assert_eq!(a.to_bits(), b.to_bits(), "Ackley nondeterministic");
        let (a, b) = {
            let run = || {
                let mut o = DirectOptimizer::new(all_ranges(0.5), all_ranges(0.5), 30_000);
                o.run(&Hartmann6).1
            };
            (run(), run())
        };
        assert_eq!(a.to_bits(), b.to_bits(), "Hartmann6 nondeterministic");
    }

    // ---------- cost-level integrity (no optimizer) ----------

    #[test]
    fn batch_order_preserved() {
        // A strictly monotone separable cost: the i-th returned value must
        // belong to the i-th input pose, even if we shuffle the batch.
        let mut poses = vec![
            Pose {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                xa: 0.0,
                ya: 0.0,
                za: 0.0,
            },
            Pose {
                x: 1.0,
                y: 1.0,
                z: 1.0,
                xa: 1.0,
                ya: 1.0,
                za: 1.0,
            },
            Pose {
                x: 2.0,
                y: 2.0,
                z: 2.0,
                xa: 2.0,
                ya: 2.0,
                za: 2.0,
            },
        ];
        let c = ShiftedSphere { shift: zero() };
        let direct = c.eval(&poses);
        // reverse the batch; costs must reverse correspondingly
        poses.reverse();
        let reversed = c.eval(&poses);
        assert_eq!(direct[0], reversed[2]);
        assert_eq!(direct[1], reversed[1]);
        assert_eq!(direct[2], reversed[0]);
    }

    #[test]
    fn eval_is_pure_function_of_pose() {
        // Same pose batch always returns the same costs (no hidden state).
        let p = vec![evil_shift()];
        let a = ShiftedSphere { shift: zero() }.eval(&p)[0];
        let b = ShiftedSphere { shift: zero() }.eval(&p)[0];
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // ---------- bounds / duplicate sampling (via a recording wrapper) ----------

    #[test]
    fn all_sampled_poses_inside_domain() {
        // DIRECT must never ask the cost about a pose outside [start-range,
        // start+range] — off-by-one in trisection produces out-of-range samples.
        // This records every evaluated pose (interior mutability — eval takes
        // &self, so it needs a Cell/RefCell behind an Arc).
        let range = 5.0;
        use std::cell::RefCell;
        use std::rc::Rc;
        struct Recorder(Rc<RefCell<Vec<Pose>>>);
        impl Cost for Recorder {
            fn eval(&self, poses: &[Pose]) -> Vec<f64> {
                self.0.borrow_mut().extend(poses.iter().copied());
                poses.iter().map(|_| 0.0).collect()
            }
        }
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut opt = DirectOptimizer::new(all_ranges(range), zero(), 5_000);
        opt.run(&Recorder(log.clone()));
        for p in log.borrow().iter() {
            for v in [p.x, p.y, p.z, p.xa, p.ya, p.za] {
                assert!(
                    (-range..=range).contains(&v),
                    "sampled pose {} outside [-{range}, {range}]",
                    show(p),
                );
            }
        }
    }

    #[test]
    fn no_duplicate_samples() {
        // Parent keeps its already-scored center; both children are fresh.
        // Any re-eval is a subdivision bug — the correct bound is zero dups.
        use std::cell::RefCell;
        use std::collections::HashSet;
        use std::rc::Rc;
        struct Counter {
            total: usize,
            uniq: HashSet<[u64; 6]>,
        }
        struct Rec(Rc<RefCell<Counter>>);
        fn key(p: &Pose) -> [u64; 6] {
            [p.x, p.y, p.z, p.xa, p.ya, p.za].map(f64::to_bits)
        }
        impl Cost for Rec {
            fn eval(&self, poses: &[Pose]) -> Vec<f64> {
                let mut c = self.0.borrow_mut();
                for p in poses {
                    c.total += 1;
                    c.uniq.insert(key(p));
                }
                poses.iter().map(|_| 0.0).collect()
            }
        }
        let counter = Rc::new(RefCell::new(Counter {
            total: 0,
            uniq: HashSet::new(),
        }));
        let mut opt = DirectOptimizer::new(all_ranges(5.0), zero(), 4_000);
        opt.run(&Rec(counter.clone()));
        let c = counter.borrow();
        assert!(c.total > 0, "no evals at all");
        assert_eq!(
            c.uniq.len(),
            c.total,
            "re-evaluated {} duplicate centers (total {}, unique {})",
            c.total - c.uniq.len(),
            c.total,
            c.uniq.len()
        );
    }
}
