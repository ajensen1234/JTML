use crate::cost::Cost;
use crate::direct_data_storage::{DirectTree, Hyperbox, MinBoxSize, UnscoredHyperbox};
use crate::direct_data_storage::{Pose, DIRECTIONS};
use crate::ffi::{CppCost, RunOutcome};
use ordered_float::OrderedFloat;
use std::collections::BTreeMap;
use std::iter::zip;
use std::time::{self, Duration};

pub struct DirectOptimizer {
    pub boxes: DirectTree,
    current_best: (Pose, f64),
    budget: u32,
    range: Pose,
    starting_point: Pose,
    call_offset: u32,
    calls: u32,
    next_box_id: u64,
    poh_selection_strategy: POHSettings,
    min_box_size: MinBoxSize,
}

#[derive(Default)]
pub enum POHSettings {
    #[default]
    ConvexHull = 0,
    Pareto = 1,
}

#[derive(Copy, Clone)]
pub struct POHPoint {
    pub size: f64,
    pub cost: f64,
}

impl DirectOptimizer {
    pub fn new(range: Pose, starting_point: Pose, budget: u32) -> Self {
        Self::new_with_strat(range, starting_point, budget, POHSettings::default())
    }
    pub fn new_with_strat(
        range: Pose,
        starting_point: Pose,
        budget: u32,
        poh_strat: POHSettings,
    ) -> Self {
        Self {
            boxes: BTreeMap::new(),
            current_best: (starting_point, f64::INFINITY),
            budget,
            range,
            starting_point,
            call_offset: 0,
            calls: 0,
            next_box_id: 0,
            poh_selection_strategy: poh_strat,
            min_box_size: MinBoxSize::default(),
        }
    }

    fn next_id(&mut self) -> u64 {
        let id = self.next_box_id;
        self.next_box_id += 1;
        id
    }

    fn sort_cost(cost: f64) -> f64 {
        if cost.is_finite() {
            cost
        } else {
            f64::INFINITY
        }
    }

    pub fn run<T: Cost>(&mut self, cost: &T) -> (Pose, f64) {
        // seed: box lives at the unit center; cost eval at its physical pose
        let start = time::Instant::now();

        let unit = Self::unit_center();
        let physical = self.denormalize(unit);
        let seed_cost = *cost
            .eval(&[physical])
            .first()
            .expect("Cost must be returned");
        self.calls += 1;
        let seed = Hyperbox {
            cost_at_center: seed_cost,
            center: unit,
            depths: [0; 6],
        };
        let id = self.next_id();
        self.boxes
            .entry(OrderedFloat(seed.size()))
            .or_default()
            .insert((OrderedFloat(Self::sort_cost(seed_cost)), id), seed);

        if seed_cost.is_finite() && seed_cost < self.current_best.1 {
            self.current_best = (physical, seed_cost);
        }

        loop {
            if self.calls + self.call_offset >= self.budget {
                break;
            }

            let candidates = self.get_potentially_optimal_candidates();
            let poh = self.select_potentially_optimal(&candidates, &self.poh_selection_strategy);

            if poh.is_empty() {
                break;
            }
            let unscored = self.trisect_and_return_unscored(&poh);
            if unscored.is_empty() {
                break;
            }
            self.score_and_reinsert(cost, &unscored);
        }
        let elapsed = start.elapsed();
        let avg_per_call: Duration = elapsed / self.calls;
        let it_per_sec = self.calls as f64 / elapsed.as_secs_f64();
        println!("{:?} per iteration", avg_per_call);
        println!(
            "{:?} iterations/second for {:?} iterations",
            it_per_sec, self.calls
        );
        self.print_resolution_summary();

        if (self.current_best.1.is_finite()) && (!self.current_best.1.is_nan()) {
            return self.best();
        } else {
            return (physical, f64::INFINITY);
        }
    }

    pub fn run_rust_opt(&mut self, cost: &CppCost) -> RunOutcome {
        self.run(cost);
        let (best_pose, best_cost) = self.best();
        RunOutcome {
            num_iter: self.calls,
            optimal_value: best_cost,
            optimal_location: best_pose.to_array(),
        }
    }

    pub fn best(&self) -> (Pose, f64) {
        self.current_best
    }

    fn physical_width(&self, hb: &Hyperbox, axis: usize) -> f64 {
        let ranges = self.range.to_array();

        2.0 * ranges[axis].abs() * 3f64.powi(-(hb.depths[axis] as i32))
    }
    fn print_resolution_summary(&self) {
        let names = ["X", "Y", "Z", "XA", "YA", "ZA"];

        println!("--- DIRECT resolution summary ---");

        for axis in 0..6 {
            let Some(min_size) = self.min_box_size.values[axis] else {
                continue;
            };

            let smallest = self
                .boxes
                .values()
                .flat_map(|row| row.values())
                .map(|hb| self.physical_width(hb, axis))
                .fold(f64::INFINITY, f64::min);

            println!(
                "{:>2}: smallest={:.6}, target={:.6}, ratio={:.2}x",
                names[axis],
                smallest,
                min_size,
                smallest / min_size,
            );
        }
    }
    fn split_axis(&self, hb: &Hyperbox) -> Option<usize> {
        hb.depths
            .iter()
            .enumerate()
            .filter(|(axis, _)| {
                self.min_box_size.values[*axis]
                    .is_none_or(|min_size| self.physical_width(hb, *axis) > min_size)
            })
            .min_by_key(|(_, depth)| **depth)
            .map(|(axis, _)| axis)
    }

    /// Map a unit-space pose (each axis in [0,1]) to physical space.
    /// physical[i] = start[i] + (unit[i] - 0.5) * 2 * range[i]
    fn denormalize(&self, unit: Pose) -> Pose {
        Pose {
            x: self.starting_point.x + (unit.x - 0.5) * 2.0 * self.range.x,
            y: self.starting_point.y + (unit.y - 0.5) * 2.0 * self.range.y,
            z: self.starting_point.z + (unit.z - 0.5) * 2.0 * self.range.z,
            xa: self.starting_point.xa + (unit.xa - 0.5) * 2.0 * self.range.xa,
            ya: self.starting_point.ya + (unit.ya - 0.5) * 2.0 * self.range.ya,
            za: self.starting_point.za + (unit.za - 0.5) * 2.0 * self.range.za,
        }
    }

    fn unit_center() -> Pose {
        Pose {
            x: 0.5,
            y: 0.5,
            z: 0.5,
            xa: 0.5,
            ya: 0.5,
            za: 0.5,
        }
    }
    fn trisect_and_return_unscored(&mut self, boxes: &[POHPoint]) -> Vec<UnscoredHyperbox> {
        let mut unscored: Vec<UnscoredHyperbox> = Vec::new();
        for poh in boxes {
            let size_key = OrderedFloat(poh.size);

            let parent_key = self.boxes.get(&size_key).and_then(|row| {
                row.iter()
                    .find(|(_, hb)| self.split_axis(hb).is_some())
                    .map(|(key, _)| *key)
            });

            let Some(parent_key) = parent_key else {
                continue;
            };

            let parent = self
                .boxes
                .get_mut(&size_key)
                .and_then(|row| row.remove(&parent_key))
                .expect("selected refinable box must still exist");
            let axis = self
                .split_axis(&parent)
                .expect("selected parent must be refinable");

            let (center, shifted) = {
                let mut this = parent;
                this.depths[axis] += 1;
                let shift = 3f64.powi(-(this.depths[axis] as i32));
                let mut posc = this.center;
                let mut negc = this.center;
                posc.shift(&DIRECTIONS[axis], shift);
                negc.shift(&DIRECTIONS[axis], -shift);
                let pos_shift = UnscoredHyperbox {
                    center: posc,
                    depths: this.depths,
                };
                let neg_shift = UnscoredHyperbox {
                    center: negc,
                    depths: this.depths,
                };
                (this, [pos_shift, neg_shift])
            };

            let id = self.next_id();
            self.boxes
                .entry(OrderedFloat(center.size()))
                .or_default()
                .insert(
                    (OrderedFloat(Self::sort_cost(center.cost_at_center)), id),
                    center,
                );
            unscored.extend(shifted);
        }
        return unscored;
    }

    fn score_and_reinsert<T: Cost>(&mut self, cost: &T, unscored: &[UnscoredHyperbox]) {
        let centers: Vec<Pose> = unscored
            .iter()
            .map(|p| self.denormalize(p.center))
            .collect();
        let evaluated_costs: Vec<f64> = cost.eval(&centers);
        self.calls += evaluated_costs.len() as u32;

        for scored_box in zip(unscored, evaluated_costs).map(|v| v.0.add_score(v.1)) {
            let id = self.next_id();
            if scored_box.cost_at_center.is_finite() {
                let physical = self.denormalize(scored_box.center);
                if scored_box.cost_at_center < self.current_best.1 {
                    self.current_best = (physical, scored_box.cost_at_center);
                }
                self.boxes
                    .entry(OrderedFloat(scored_box.size()))
                    .or_default()
                    .insert(
                        (OrderedFloat(Self::sort_cost(scored_box.cost_at_center)), id),
                        scored_box,
                    );
            }
        }
    }

    fn get_potentially_optimal_candidates(&self) -> Vec<POHPoint> {
        let mut candidates = Vec::new();

        for (size_key, row) in &self.boxes {
            if let Some((key, _box)) = row.iter().find(|(_, hb)| self.split_axis(hb).is_some()) {
                candidates.push(POHPoint {
                    size: size_key.into_inner(),
                    cost: key.0.into_inner(),
                });
            }
        }

        return candidates;
    }

    fn select_potentially_optimal(
        &self,
        candidates: &[POHPoint],
        settings: &POHSettings,
    ) -> Vec<POHPoint> {
        let poh = match settings {
            POHSettings::ConvexHull => DirectOptimizer::convex_hull(candidates),
            POHSettings::Pareto => DirectOptimizer::pareto_front(candidates),
        };
        return poh;
    }

    fn pareto_front(candidates: &[POHPoint]) -> Vec<POHPoint> {
        let mut pts: Vec<POHPoint> = candidates.to_vec();
        // largest size first; cost ascending breaks ties so equal-size points
        // (shouldn't occur post-dedup, but be defensive) prefer the cheaper one
        pts.sort_by(|a, b| b.size.total_cmp(&a.size).then(a.cost.total_cmp(&b.cost)));

        let mut front = Vec::with_capacity(pts.len());
        let mut best_cost = f64::INFINITY;
        for p in pts {
            if p.cost.is_finite() && p.cost < best_cost {
                front.push(p);
                best_cost = p.cost;
            }
        }
        front.reverse(); // back to ascending size, matching convex_hull's convention
        return front;
    }

    pub(crate) fn convex_hull(candidates: &[POHPoint]) -> Vec<POHPoint> {
        let mut hull: Vec<POHPoint> = Vec::new();

        if candidates.len() < 2 {
            hull.extend(candidates);
            return hull;
        }
        for p in candidates.iter() {
            while (hull.len() >= 2)
                && (DirectOptimizer::cross(
                    (hull[hull.len() - 2].size, hull[hull.len() - 2].cost),
                    (hull[hull.len() - 1].size, hull[hull.len() - 1].cost),
                    (p.size, p.cost),
                ) < 0.0)
            {
                hull.pop();
            }
            hull.push(*p);
        }
        let cut = hull
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.cost.total_cmp(&b.cost).then(b.size.total_cmp(&a.size)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        return hull.split_off(cut);
    }

    fn cross(o: (f64, f64), p1: (f64, f64), p2: (f64, f64)) -> f64 {
        return (p1.0 - o.0) * (p2.1 - o.1) - (p1.1 - o.1) * (p2.0 - o.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::ShiftedSphere;
    use crate::cost::Cost;
    use crate::test_support::{coords, on_lattice, show, splat, zero};
    use proptest::prelude::*;
    use std::collections::BTreeSet;

    fn iter_boxes(opt: &DirectOptimizer) -> impl Iterator<Item = &Hyperbox> {
        opt.boxes.values().flat_map(|row| row.values())
    }

    /// Jones potentially-optimal test on `(size, cost)` representatives.
    /// `j` is POH iff `K_lo <= K_hi` and `K_hi > 0`, with same-size worse
    /// points treated as dominated.
    fn jones_poh(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        for (j, &(dj, fj)) in points.iter().enumerate() {
            let dominated = points
                .iter()
                .enumerate()
                .any(|(i, &(di, fi))| i != j && (di - dj).abs() <= 1e-15 && fi < fj);
            if dominated {
                continue;
            }
            let mut k_lo = f64::NEG_INFINITY;
            let mut k_hi = f64::INFINITY;
            for (i, &(di, fi)) in points.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dd = dj - di;
                if dd > 1e-15 {
                    k_lo = k_lo.max((fj - fi) / dd);
                } else if dd < -1e-15 {
                    k_hi = k_hi.min((fi - fj) / (di - dj));
                }
            }
            if k_lo <= k_hi && k_hi > 0.0 {
                out.push((dj, fj));
            }
        }
        out
    }

    fn hull_pairs(pts: &[POHPoint]) -> Vec<(f64, f64)> {
        DirectOptimizer::convex_hull(pts)
            .into_iter()
            .map(|p| (p.size, p.cost))
            .collect()
    }

    fn volume_checksum(opt: &DirectOptimizer) -> Result<(u128, u128), &'static str> {
        let sums: Vec<u32> = iter_boxes(opt).map(|b| b.depths.iter().sum()).collect();
        let Some(&s_max) = sums.iter().max() else {
            return Err("no boxes");
        };
        let rhs = 3u128.checked_pow(s_max).ok_or("pow overflow")?;
        let mut lhs = 0u128;
        for s in sums {
            let term = 3u128.checked_pow(s_max - s).ok_or("pow overflow")?;
            lhs = lhs.checked_add(term).ok_or("add overflow")?;
        }
        Ok((lhs, rhs))
    }

    #[test]
    fn jones_rejects_the_descending_left_hull() {
        // Larger box is cheaper: the small expensive vertices are NOT POH.
        let pts = [
            POHPoint {
                size: 1.0,
                cost: 10.0,
            },
            POHPoint {
                size: 2.0,
                cost: 5.0,
            },
            POHPoint {
                size: 3.0,
                cost: 0.0,
            },
        ];
        let hull = hull_pairs(&pts);
        let jones = jones_poh(&[(1.0, 10.0), (2.0, 5.0), (3.0, 0.0)]);
        assert_eq!(jones, vec![(3.0, 0.0)], "oracle sanity");
        assert_eq!(
            hull, jones,
            "convex_hull returned {hull:?}, Jones POH is {jones:?} — \
             drop vertices left of the global-min-cost hull vertex"
        );
    }

    #[test]
    fn jones_keeps_the_increasing_right_hull() {
        let pts = [
            POHPoint {
                size: 1.0,
                cost: 0.0,
            },
            POHPoint {
                size: 2.0,
                cost: 1.0,
            },
            POHPoint {
                size: 3.0,
                cost: 4.0,
            },
        ];
        let mut hull = hull_pairs(&pts);
        let mut jones = jones_poh(&[(1.0, 0.0), (2.0, 1.0), (3.0, 4.0)]);
        hull.sort_by(|a, b| a.0.total_cmp(&b.0));
        jones.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert_eq!(hull, jones);
    }

    #[test]
    fn hull_of_one_and_two_points() {
        let one = [POHPoint {
            size: 1.5,
            cost: 2.0,
        }];
        assert_eq!(hull_pairs(&one).len(), 1);
        let two = [
            POHPoint {
                size: 1.0,
                cost: 1.0,
            },
            POHPoint {
                size: 2.0,
                cost: 2.5,
            },
        ];
        assert_eq!(hull_pairs(&two).len(), 2);
    }

    #[test]
    fn collinear_lower_hull_keeps_interior_vertices() {
        // Jones selects every collinear lower-hull point. `cross <= 0` pops them.
        let pts = [
            POHPoint {
                size: 1.0,
                cost: 1.0,
            },
            POHPoint {
                size: 2.0,
                cost: 2.0,
            },
            POHPoint {
                size: 3.0,
                cost: 3.0,
            },
        ];
        let hull = hull_pairs(&pts);
        assert_eq!(
            hull.len(),
            3,
            "collinear interior vertex dropped ({hull:?}); \
             Jones keeps all of them — change the comparison deliberately if this is DIRECT-l"
        );
    }

    #[test]
    fn boxes_tile_the_unit_cube_exactly() {
        let mut opt = DirectOptimizer::new(splat(5.0), zero(), 80);
        opt.run(&ShiftedSphere {
            shift: Pose {
                x: 1.7,
                y: 0.4,
                z: -2.1,
                xa: 0.8,
                ya: -1.3,
                za: 2.6,
            },
        });
        let (lhs, rhs) = volume_checksum(&opt).expect("volume checksum");
        assert_eq!(lhs, rhs, "Σ 3^{{Smax-Σd}} = {lhs} != 3^{{Smax}} = {rhs}");
    }

    #[test]
    fn every_box_center_is_on_the_trisection_lattice() {
        let mut opt = DirectOptimizer::new(splat(5.0), zero(), 80);
        opt.run(&ShiftedSphere {
            shift: Pose {
                x: 1.7,
                y: 0.4,
                z: -2.1,
                xa: 0.8,
                ya: -1.3,
                za: 2.6,
            },
        });
        for hb in iter_boxes(&opt) {
            for (c, d) in coords(&hb.center).iter().zip(hb.depths) {
                assert!(
                    on_lattice(*c, d),
                    "box center {} depth {d:?} off-lattice",
                    show(&hb.center)
                );
            }
        }
    }

    #[test]
    fn no_two_boxes_share_a_unit_center() {
        let mut opt = DirectOptimizer::new(splat(5.0), zero(), 4_000);
        opt.run(&ShiftedSphere {
            shift: Pose {
                x: 1.7,
                y: 0.4,
                z: -2.1,
                xa: 0.8,
                ya: -1.3,
                za: 2.6,
            },
        });
        let mut seen = BTreeSet::new();
        for hb in iter_boxes(&opt) {
            let key = coords(&hb.center).map(f64::to_bits);
            assert!(
                seen.insert(key),
                "duplicate unit center {}",
                show(&hb.center)
            );
        }
    }

    #[test]
    fn each_box_depths_differ_by_at_most_one() {
        let mut opt = DirectOptimizer::new(splat(5.0), zero(), 120);
        opt.run(&ShiftedSphere {
            shift: Pose {
                x: 1.7,
                y: 0.4,
                z: -2.1,
                xa: 0.8,
                ya: -1.3,
                za: 2.6,
            },
        });
        for hb in iter_boxes(&opt) {
            let min = hb.depths.iter().copied().min().unwrap_or(0);
            let max = hb.depths.iter().copied().max().unwrap_or(0);
            assert!(
                max - min <= 1,
                "depths {:?} violate min-depth split",
                hb.depths
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 4096, ..ProptestConfig::default() })]

        #[test]
        fn convex_hull_matches_jones_on_unique_sizes(
            raw in proptest::collection::vec((0.1_f64..12.0, -8.0_f64..8.0), 1..12)
        ) {
            // Dedup sizes so the input matches what determine_potentially_optimal feeds.
            let mut pts = raw;
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
            prop_assume!(!pts.is_empty());
            let poh: Vec<POHPoint> = pts
                .iter()
                .map(|&(size, cost)| POHPoint { size, cost })
                .collect();
            let mut hull = hull_pairs(&poh);
            let mut jones = jones_poh(&pts);
            hull.sort_by(|a, b| a.0.total_cmp(&b.0));
            jones.sort_by(|a, b| a.0.total_cmp(&b.0));
            prop_assert_eq!(hull, jones);
        }

        #[test]
        fn volume_partition_holds_for_random_runs(
            budget in 8_u32..60,
            shift in proptest::array::uniform6(-4.0_f64..4.0),
        ) {
            let [x, y, z, xa, ya, za] = shift;
            let mut opt = DirectOptimizer::new(splat(5.0), zero(), budget);
            opt.run(&ShiftedSphere { shift: Pose { x, y, z, xa, ya, za } });
            match volume_checksum(&opt) {
                Ok((lhs, rhs)) => prop_assert_eq!(lhs, rhs),
                Err("pow overflow") => {}
                Err(e) => prop_assert!(false, "volume checksum failed: {e}"),
            }
        }
    }

    struct Sphere;
    impl Cost for Sphere {
        fn eval(&self, poses: &[Pose]) -> Vec<f64> {
            poses
                .iter()
                .map(|p| coords(p).iter().map(|v| v * v).sum())
                .collect()
        }
    }

    #[test]
    fn seed_evaluation_counts_as_one_call_when_budget_is_zero() {
        let mut opt = DirectOptimizer::new(splat(5.0), zero(), 0);
        let _ = opt.run(&Sphere);
        assert_eq!(opt.calls, 1, "seed must still be evaluated at budget 0");
    }
}
