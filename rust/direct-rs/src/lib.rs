mod direct_data_storage;
mod direct_optimizer;
use crate::direct_optimizer::{DirectOptimizer, POHSettings};

#[cxx::bridge]
pub mod ffi {

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

    }
}
