use crate::{cost::Cost, Pose};

#[cfg(test)]
struct Sphere;

#[cfg(test)]
impl Cost for Sphere {
    fn eval(&self, poses: &[Pose]) -> Vec<f64> {
        poses
            .iter()
            .map(|p| p.x * p.x + p.y * p.y + p.z * p.z + p.xa * p.xa + p.ya * p.ya + p.za * p.za)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::direct_data_storage::Hyperbox;
    use crate::direct_data_storage::Pose;

    #[test]
    fn hyperbox_size_matches_depth_formula() {
        let unit = Hyperbox {
            cost_at_center: 0.0,
            center: Pose {
                x: 0.5,
                y: 0.5,
                z: 0.5,
                xa: 0.5,
                ya: 0.5,
                za: 0.5,
            },
            depths: [0; 6],
        };
        let unit_size = unit.size();
        assert!(
            (unit_size - 6.0_f64.sqrt()).abs() < 1e-12,
            "unit box size {unit_size} != sqrt(6)"
        );

        let hb = Hyperbox {
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
        // size = sqrt(Σ 3^{-2 d_i}) = sqrt(5/729 + 1/387420489)
        let expected = (5.0_f64 / 729.0 + 1.0 / 387_420_489.0).sqrt();
        let got = hb.size();
        assert!(
            (got - expected).abs() < 1e-12,
            "size {got} != expected {expected}"
        );
    }

    #[test]
    fn sphere_run_decreases_cost() {
        use super::Sphere;
        use crate::direct_optimizer::DirectOptimizer;

        let start = Pose {
            x: 1.0,
            y: 1.0,
            z: 1.0,
            xa: 1.0,
            ya: 1.0,
            za: 1.0,
        };
        let range = Pose {
            x: 5.0,
            y: 5.0,
            z: 5.0,
            xa: 5.0,
            ya: 5.0,
            za: 5.0,
        };
        let mut opt = DirectOptimizer::new(range, start, 5_000);
        let (_best, best_cost) = opt.run(&Sphere);
        println!("cost={best_cost}");
        // the global minimum is 0; DIRECT on a sphere should improve on the
        // seed's sum-of-squares (start=(1,1,1,...) => seed cost 6).
        assert!(
            best_cost < 6.0,
            "expected to improve on seed cost 6, got {best_cost}"
        );
    }
}
