use std::collections::BTreeMap;

use ordered_float::OrderedFloat;

#[expect(non_camel_case_types, reason = "matching cpp style")]
#[derive(Clone, Copy)]
pub enum Direction {
    X_DIR = 0,
    Y_DIR = 1,
    Z_DIR = 2,
    XA_DIR = 3,
    YA_DIR = 4,
    ZA_DIR = 5,
}

pub const DIRECTIONS: [Direction; 6] = [
    Direction::X_DIR,
    Direction::Y_DIR,
    Direction::Z_DIR,
    Direction::XA_DIR,
    Direction::YA_DIR,
    Direction::ZA_DIR,
];

#[derive(Copy, Clone, Default)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub xa: f64,
    pub ya: f64,
    pub za: f64,
}
impl Pose {
    pub fn shift(&mut self, dir: &Direction, amount: f64) {
        match dir {
            Direction::X_DIR => self.x += amount,
            Direction::Y_DIR => self.y += amount,
            Direction::Z_DIR => self.z += amount,
            Direction::XA_DIR => self.xa += amount,
            Direction::YA_DIR => self.ya += amount,
            Direction::ZA_DIR => self.za += amount,
        }
    }
    pub fn to_array(self) -> [f64; 6] {
        return [self.x, self.y, self.z, self.xa, self.ya, self.za];
    }
}

pub type SizeKey = OrderedFloat<f64>;
pub type CostKey = (OrderedFloat<f64>, u64);
pub type DirectTree = BTreeMap<SizeKey, BTreeMap<CostKey, Hyperbox>>;

#[derive(Clone, Copy)]
pub struct Hyperbox {
    pub cost_at_center: f64,
    pub center: Pose,
    pub depths: [u32; 6],
}

#[derive(Clone, Copy)]
pub struct UnscoredHyperbox {
    pub center: Pose,
    pub depths: [u32; 6],
}

#[derive(Clone, Copy)]
pub struct MinBoxSize {
    pub values: [Option<f64>; 6],
}

impl Default for MinBoxSize {
    fn default() -> Self {
        Self {
            values: [
                Some(0.1),
                Some(0.1),
                Some(0.1),
                Some(0.1),
                Some(0.1),
                Some(0.1),
            ],
        }
    }
}

impl Hyperbox {
    pub fn longest_axis(&self) -> usize {
        self.depths
            .iter()
            .enumerate()
            .min_by_key(|(_, depth)| **depth)
            .map(|(axis, _)| axis)
            .expect("array is fixed-size and non-empty")
    }

    pub fn size(&self) -> f64 {
        return self
            .depths
            .iter()
            .map(|e| 3f64.powf(-2.0 * (*e as f64)))
            .sum::<f64>()
            .sqrt();
    }
    // TODO: how to force usize to be the right size at runtime?
    pub fn trisect(mut self, axis: usize) -> (Hyperbox, [UnscoredHyperbox; 2]) {
        if let Some(x) = self.depths.get_mut(axis) {
            *x += 1;
        }

        let shift = 3f64.powi(-(self.depths[axis] as i32));

        let mut posc = self.center;
        let mut negc = self.center;

        posc.shift(&DIRECTIONS[axis], shift);
        negc.shift(&DIRECTIONS[axis], -shift);

        let pos_shift = UnscoredHyperbox {
            center: posc,
            depths: self.depths,
        };

        let neg_shift = UnscoredHyperbox {
            center: negc,
            depths: self.depths,
        };

        (self, [pos_shift, neg_shift])
    }
}

impl UnscoredHyperbox {
    pub fn add_score(self, score: f64) -> Hyperbox {
        return Hyperbox {
            cost_at_center: score,
            depths: self.depths,
            center: self.center,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{canonical_size, on_lattice, splat};
    use proptest::prelude::*;

    fn box_at(center: Pose, depths: [u32; 6]) -> Hyperbox {
        Hyperbox {
            cost_at_center: 0.0,
            center,
            depths,
        }
    }

    #[test]
    fn unit_box_size_is_sqrt_6() {
        let hb = box_at(splat(0.5), [0; 6]);
        let got = hb.size();
        assert!(
            (got - 6.0_f64.sqrt()).abs() < 1e-12,
            "unit size {got} != sqrt(6)"
        );
    }

    #[test]
    fn size_is_bit_identical_under_depth_permutation() {
        let a = [1u32, 2, 0, 3, 0, 4];
        let mut b = a;
        b.swap(0, 1);
        b.swap(2, 5);
        let sa = box_at(splat(0.5), a).size();
        let sb = box_at(splat(0.5), b).size();
        assert_eq!(
            sa.to_bits(),
            sb.to_bits(),
            "size() depends on depth order: {sa} vs {sb} (a={a:?} b={b:?})"
        );
    }

    #[test]
    fn longest_axis_picks_the_min_depth_axis_and_trisect_shrinks() {
        let parent = box_at(splat(0.5), [2, 0, 1, 3, 1, 4]);
        let parent_size = parent.size();

        let axis = parent.longest_axis();
        assert_eq!(axis, 1);

        let (center, [pos, neg]) = parent.trisect(axis);

        let changed: Vec<usize> = center
            .depths
            .iter()
            .zip([2u32, 0, 1, 3, 1, 4])
            .enumerate()
            .filter(|(_, (now, was))| *now != was)
            .map(|(i, _)| i)
            .collect();

        assert_eq!(changed, vec![1]);
        assert_eq!(center.depths[1], 1);
        assert_eq!(pos.depths, center.depths);
        assert_eq!(neg.depths, center.depths);
        assert!(center.size() < parent_size);
        assert!(canonical_size(pos.depths) < parent_size);
        assert!(canonical_size(neg.depths) < parent_size);
    }

    #[test]
    fn trisect_children_sit_on_the_center_lattice() {
        let parent = box_at(splat(0.5), [0; 6]);
        let axis = parent.longest_axis();
        let (center, [pos, neg]) = parent.trisect(axis);
        for (p, depths) in [
            (center.center, center.depths),
            (pos.center, pos.depths),
            (neg.center, neg.depths),
        ] {
            for (c, d) in crate::test_support::coords(&p).iter().zip(depths) {
                assert!(
                    on_lattice(*c, d),
                    "center coord {c} at depth {d} is off-lattice"
                );
            }
        }
    }

    #[test]
    fn repeated_min_depth_split_keeps_depths_within_one() {
        let mut hb = box_at(splat(0.5), [0; 6]);
        for _ in 0..18 {
            let axis = hb.longest_axis();
            let (next, _) = hb.trisect(axis);
            hb = next;
            let min = hb.depths.iter().copied().min().unwrap_or(0);
            let max = hb.depths.iter().copied().max().unwrap_or(0);
            assert!(
                max - min <= 1,
                "depths {:?} drifted more than 1 apart",
                hb.depths
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        #[test]
        fn size_matches_canonical_sorted_sum(d0 in 0u32..8, d1 in 0u32..8, d2 in 0u32..8,
                                             d3 in 0u32..8, d4 in 0u32..8, d5 in 0u32..8) {
            let depths = [d0, d1, d2, d3, d4, d5];
            let got = box_at(splat(0.5), depths).size();
            let want = canonical_size(depths);
            prop_assert!(
                (got - want).abs() < 1e-12,
                "size {got} != canonical {want} for {depths:?}"
            );
        }
    }
}
