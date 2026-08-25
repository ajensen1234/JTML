mod bench;
mod cost;
mod direct_data_storage;
mod direct_optimizer;
mod test;
mod utils;

#[cfg(test)]
mod properties;
#[cfg(test)]
mod test_support;

#[cfg(test)]
mod dup_diag;

use crate::{
    cost::Cost,
    direct_optimizer::{DirectOptimizer, POHSettings},
    ffi::CppCost,
};

#[cxx::bridge]
pub mod ffi {

    #[namespace = "direct_rs"]
    extern "Rust" {
        type DirectOptimizer;
        type POHSettings;
    }

    unsafe extern "C++" {
        include!("domain/cost.h");
        include!("domain/data_structures_6D.h");
        type CppCost;
        type Point6D;

        pub fn evaluate(self: &CppCost, point: &Point6D) -> f64;
        pub fn IsBound(self: &CppCost) -> bool;

        #[Self=Point6D]
        pub fn new_point(x: f64, y: f64, z: f64, xa: f64, ya: f64, za: f64) -> UniquePtr<Point6D>;

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
