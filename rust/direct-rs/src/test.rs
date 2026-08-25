use crate::{cost::Cost, direct_data_storage::Pose};

struct Sphere;

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
