//! Temporary diagnostic for the duplicate-sample finding. NEVER SHIP.
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::cost::Cost;
use crate::direct_data_storage::Pose;
use crate::direct_optimizer::DirectOptimizer;
use crate::test_support::{splat, zero};

struct Rec(Rc<RefCell<HashMap<[u64; 6], usize>>>);
impl Cost for Rec {
    fn eval(&self, poses: &[Pose]) -> Vec<f64> {
        let mut m = self.0.borrow_mut();
        for p in poses {
            *m.entry([p.x, p.y, p.z, p.xa, p.ya, p.za].map(f64::to_bits)).or_insert(0) += 1;
        }
        poses.iter().map(|_| 0.0).collect()
    }
}

#[test]
fn diag_duplicates() {
    let m = Rc::new(RefCell::new(HashMap::new()));
    let mut opt = DirectOptimizer::new(splat(5.0), zero(), 4_000);
    opt.run(Rec(Rc::clone(&m)));
    let m = m.borrow();
    let mut dups: Vec<_> = m.iter().filter(|&(_, &c)| c > 1).collect();
    dups.sort_by_key(|(_, c)| **c);
    println!("total distinct points: {}", m.len());
    println!("num duplicated points: {}", dups.len());
    let maxct: usize = dups.iter().map(|&(_, c)| *c).max().unwrap_or(0);
    println!("max multiplicity: {maxct}");
    for &(k, c) in dups.iter().rev().take(15) {
        let p: Vec<f64> = k.iter().map(|&b| f64::from_bits(b)).collect();
        println!(
            "dup count={c}: ({:.6},{:.6},{:.6},{:.6},{:.6},{:.6})",
            p[0], p[1], p[2], p[3], p[4], p[5]
        );
    }
}