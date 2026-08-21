use crate::direct_data_storage::{CostKey, Hyperbox, Pose, SizeKey, UnscoredHyperbox};
use crate::ffi::CppCost;
use ordered_float::OrderedFloat;
use std::collections::BTreeMap;

pub struct DirectOptimizer {
    boxes: BTreeMap<SizeKey, BTreeMap<CostKey, Hyperbox>>,
    current_best: (Pose, f64),
}

impl DirectOptimizer {}

pub enum POHSettings {
    CONVEX_HULL = 0,
    PARETO = 1,
    AGGRESSIVE = 2,
}

#[derive(Copy, Clone)]
struct POHPoint {
    size: f64,
    cost: f64,
}

impl Default for POHSettings {
    fn default() -> Self {
        Self::CONVEX_HULL
    }
}

impl DirectOptimizer {
    pub fn run(&mut self, cost: CppCost) {}
    fn trisect_and_reinsert(&mut self, boxes: &[POHPoint]) {
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
            POHSettings::CONVEX_HULL => DirectOptimizer::convex_hull(&init_hyperboxes),
            POHSettings::PARETO => DirectOptimizer::pareto_front(&init_hyperboxes),
            POHSettings::AGGRESSIVE => Vec::new(),
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
