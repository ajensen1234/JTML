use crate::cost::Cost;
use crate::direct_data_storage::{CostKey, Hyperbox, Pose, SizeKey, UnscoredHyperbox};
use crate::ffi::CppCost;
use ordered_float::OrderedFloat;
use std::collections::BTreeMap;
use std::iter::zip;
pub struct DirectOptimizer {
    boxes: BTreeMap<SizeKey, BTreeMap<CostKey, Hyperbox>>,
    current_best: (Pose, f64),
}

pub enum POHSettings {
    ConvexHull = 0,
}

#[derive(Copy, Clone)]
struct POHPoint {
    size: f64,
    cost: f64,
}

impl Default for POHSettings {
    fn default() -> Self {
        Self::ConvexHull
    }
}

impl DirectOptimizer {
    pub fn new()
    pub fn run(&mut self, cost: CppCost) {}
    fn trisect_and_return_unscored(&mut self, boxes: &[POHPoint]) -> Vec<UnscoredHyperbox> {
        let mut unscored: Vec<UnscoredHyperbox> = Vec::new();
        for poh in boxes {
            let Some(parent) = self
                .boxes
                .get_mut(&OrderedFloat(poh.size))
                .and_then(|row| row.pop_first())
                .map(|(_key, hb)| hb)
            else {
                continue;
            };

            let (center, shifted) = parent.trisect();
            self.boxes
                .entry(OrderedFloat(center.size()))
                .or_default()
                .insert((OrderedFloat(center.cost_at_center), 1), center);
            unscored.extend(shifted);
        }
        return unscored;
    }

    fn score_and_reinsert<T: Cost>(&mut self, cost: T, unscored: &[UnscoredHyperbox]) {
        let centers: Vec<Pose> = unscored.iter().map(|p| p.center).collect();
        let evaluated_costs: Vec<f64> = cost.eval(&centers);

        for scored_box in zip(unscored, evaluated_costs).map(|v| v.0.add_score(v.1)) {
            self.boxes
                .entry(OrderedFloat(scored_box.size()))
                .or_default()
                .insert((OrderedFloat(scored_box.cost_at_center), 1), scored_box);
        }
    }

    fn determine_potentially_optimal(&self, settings: POHSettings) -> Vec<POHPoint> {
        let mut init_hyperboxes: Vec<POHPoint> = Vec::new();

        // Because we're using a BTree based on the size and cost, this
        // returns the value of the hyperbox per-size with the lowest cost
        for (size_key, cost_val) in self.boxes.iter() {
            if let Some((key, _box)) = cost_val.first_key_value() {
                init_hyperboxes.push(POHPoint {
                    size: (*size_key).into_inner(),
                    cost: (key.0).into_inner(),
                })
            }
        }
        let poh = match settings {
            POHSettings::ConvexHull => DirectOptimizer::convex_hull(&init_hyperboxes),
            // POHSettings::PARETO => DirectOptimizer::pareto_front(&init_hyperboxes),
            // POHSettings::AGGRESSIVE => Vec::new(),
        };
        return poh;
    }

    fn pareto_front(poh: &[POHPoint]) -> Vec<POHPoint> {
        todo!();
    }

    fn convex_hull(poh: &[POHPoint]) -> Vec<POHPoint> {
        let mut poh_true: Vec<POHPoint> = Vec::new();

        for p in poh.iter() {
            while (poh_true.len() >= 2)
                & (DirectOptimizer::cross(
                    (
                        poh_true[poh_true.len() - 2].size,
                        poh_true[poh_true.len() - 2].cost,
                    ),
                    (
                        poh_true[poh_true.len() - 1].size,
                        poh_true[poh_true.len() - 1].cost,
                    ),
                    (p.size, p.cost),
                ) <= 0.0)
            {
                poh_true.pop();
            }
            poh_true.push(*p);
        }

        return poh_true;
    }

    fn cross(o: (f64, f64), p1: (f64, f64), p2: (f64, f64)) -> f64 {
        return (p1.0 - o.0) * (p2.1 - o.1) - (p1.1 - o.1) * (p2.0 - o.0);
    }
}
