use crate::direct_data_storage::Pose;

pub trait Cost {
    fn eval(&self, poses: &[Pose]) -> Vec<f64>;
}
