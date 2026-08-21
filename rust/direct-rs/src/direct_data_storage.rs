use std::iter::Sum;

use ordered_float::OrderedFloat;

use crate::ffi::CppCost;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub enum Direction {
    X_DIR = 0,
    Y_DIR = 1,
    Z_DIR = 2,
    XA_DIR = 3,
    YA_DIR = 4,
    ZA_DIR = 5,
}

const DIRECTIONS: [Direction; 6] = [
    Direction::X_DIR,
    Direction::Y_DIR,
    Direction::Z_DIR,
    Direction::XA_DIR,
    Direction::YA_DIR,
    Direction::ZA_DIR,
];

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub xa: f64,
    pub ya: f64,
    pub za: f64,
}

trait Cost {
    fn eval(&self, pose: &[Pose]) -> f64;
}

impl Pose {
    pub fn shift(&mut self, dir: Direction, amount: f64) {
        match dir {
            Direction::X_DIR => self.x += amount,
            Direction::Y_DIR => self.y += amount,
            Direction::Z_DIR => self.z += amount,
            Direction::XA_DIR => self.xa += amount,
            Direction::YA_DIR => self.ya += amount,
            Direction::ZA_DIR => self.za += amount,
        }
    }
}

pub type SizeKey = OrderedFloat<f64>;
pub type CostKey = (OrderedFloat<f64>, u64);

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

impl Hyperbox {
    pub fn size(&self) -> f64 {
        return self
            .depths
            .iter()
            .map(|e| 3f64.powf(-2.0 * (*e as f64)))
            .sum::<f64>()
            .sqrt();
    }
    pub fn trisect(mut self) -> (Hyperbox, [UnscoredHyperbox; 2]) {
        let (min_idx, _) = self
            .depths
            .iter()
            .enumerate()
            .min_by_key(|&(_, v)| *v)
            .expect("array is fixed-size and non-empty");

        self.depths[min_idx] += 1;

        let mut posc = self.center;
        let mut negc = self.center;
        posc.shift(
            DIRECTIONS[min_idx],
            3f64.powf(-(self.depths[min_idx] as f64)),
        );

        negc.shift(
            DIRECTIONS[min_idx],
            -3f64.powf(-(self.depths[min_idx] as f64)),
        );

        let pos_shift: UnscoredHyperbox = UnscoredHyperbox {
            center: posc,
            depths: self.depths.clone(),
        };

        let neg_shift: UnscoredHyperbox = UnscoredHyperbox {
            center: negc,
            depths: self.depths.clone(),
        };

        return (self, [pos_shift, neg_shift]);
    }
}

#[cfg(test)]
mod tests {
    use crate::direct_data_storage::{Hyperbox, Pose};

    #[test]
    fn test_hyperbox_size() {
        let hb: Hyperbox = Hyperbox {
            cost_at_center: 25.0,
            center: Pose {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                xa: 0.0,
                ya: 0.0,
                za: 0.0,
            },
            depths: [3, 3, 3, 3, 3, 9],
        };

        println!("{}", hb.size());
    }
}
