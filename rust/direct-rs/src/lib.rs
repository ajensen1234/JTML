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
