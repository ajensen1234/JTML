mod bench;
mod direct_data_storage;
mod direct_optimizer;
mod test;
mod utils;

#[cfg(test)]
mod properties;
#[cfg(test)]
mod test_support;

use crate::{
    cost::Cost,
    direct_optimizer::{DirectOptimizer, POHSettings},
    ffi::CppCost,
};

#[cxx::bridge]
pub mod ffi {
    pub struct RunOutcome {
        pub num_iter: u32,
        pub optimal_value: f64,
        pub optimal_location: [f64; 6],
    }

    #[namespace = "direct_rs"]
    extern "Rust" {
        type DirectOptimizer;

        pub fn new_rust_opt(
            range: [f64; 6],
            starting_point: [f64; 6],
            budget: u32,
        ) -> Box<DirectOptimizer>;

        pub fn run_rust_opt(self: &mut DirectOptimizer, cost: &CppCost) -> RunOutcome;

    }

    unsafe extern "C++" {
        include!("domain/cost.h");
        include!("domain/data_structures_6D.h");
        pub type CppCost;
        type Point6D;

        pub fn evaluate(self: &CppCost, point: &Point6D) -> f64;
        pub fn IsBound(self: &CppCost) -> bool;

        #[Self=Point6D]
        pub fn new_point(x: f64, y: f64, z: f64, xa: f64, ya: f64, za: f64) -> UniquePtr<Point6D>;

        pub fn to_array(self: &Point6D) -> [f64; 6];

    }
}

impl Cost for CppCost {
    fn eval(&self, poses: &[direct_data_storage::Pose]) -> Vec<f64> {
        return poses
            .iter()
            .map(|pose| {
                self.evaluate(&ffi::Point6D::new_point(
                    pose.x, pose.y, pose.z, pose.xa, pose.ya, pose.za,
                ))
            })
            .collect();
    }
}

pub fn new_rust_opt(
    range: [f64; 6],
    starting_point: [f64; 6],
    budget: u32,
) -> Box<DirectOptimizer> {
    return Box::new(DirectOptimizer::new(
        direct_data_storage::Pose {
            x: range[0],
            y: range[1],
            z: range[2],
            xa: range[3],
            ya: range[4],
            za: range[5],
        },
        direct_data_storage::Pose {
            x: starting_point[0],
            y: starting_point[1],
            z: starting_point[2],
            xa: starting_point[3],
            ya: starting_point[4],
            za: starting_point[5],
        },
        budget,
    ));
}
