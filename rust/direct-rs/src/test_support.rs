//! Shared helpers for the DIRECT test suite. Compiled only under `cargo test`.

use crate::direct_data_storage::Pose;

pub(crate) fn coords(p: &Pose) -> [f64; 6] {
    [p.x, p.y, p.z, p.xa, p.ya, p.za]
}

pub(crate) fn pose(c: [f64; 6]) -> Pose {
    let [x, y, z, xa, ya, za] = c;
    Pose { x, y, z, xa, ya, za }
}

pub(crate) fn splat(v: f64) -> Pose {
    pose([v; 6])
}

pub(crate) fn zero() -> Pose {
    splat(0.0)
}

pub(crate) fn show(p: &Pose) -> String {
    format!(
        "({:.6},{:.6},{:.6},{:.6},{:.6},{:.6})",
        p.x, p.y, p.z, p.xa, p.ya, p.za
    )
}

pub(crate) fn dist(a: &Pose, b: &Pose) -> f64 {
    coords(a)
        .iter()
        .zip(coords(b))
        .map(|(u, v)| (u - v) * (u - v))
        .sum::<f64>()
        .sqrt()
}

/// physical[i] = start[i] + (unit[i] - 0.5) * 2 * range[i]
pub(crate) fn denorm(start: Pose, range: Pose, unit: Pose) -> Pose {
    let s = coords(&start);
    let r = coords(&range);
    let u = coords(&unit);
    let mut out = [0.0; 6];
    for ((slot, si), (ui, ri)) in out.iter_mut().zip(s).zip(u.iter().zip(r)) {
        *slot = si + (ui - 0.5) * 2.0 * ri;
    }
    pose(out)
}

/// Inverse of `denorm`. Axes with `range_i == 0` map back to 0.5.
pub(crate) fn invert(start: Pose, range: Pose, physical: Pose) -> Pose {
    let s = coords(&start);
    let r = coords(&range);
    let p = coords(&physical);
    let mut out = [0.5; 6];
    for ((slot, si), (pi, ri)) in out.iter_mut().zip(s).zip(p.iter().zip(r)) {
        if ri != 0.0 {
            *slot = 0.5 + (pi - si) / (2.0 * ri);
        }
    }
    pose(out)
}

/// Unit-space center at depth `d` must sit at an odd multiple of `3^{-d}/2`.
pub(crate) fn on_lattice(center: f64, depth: u32) -> bool {
    let scale = 3f64.powi(depth as i32);
    let scaled = center * 2.0 * scale;
    let nearest = scaled.round();
    let n = nearest as i64;
    (scaled - nearest).abs() < 1e-9 && n.unsigned_abs() % 2 == 1
}

/// `size = sqrt(Σ 3^{-2 d_i})` with terms summed in sorted order so the
/// reference is permutation-invariant.
pub(crate) fn canonical_size(depths: [u32; 6]) -> f64 {
    let mut terms: [f64; 6] = depths.map(|d| {
        let v = 3f64.powi(d as i32);
        1.0 / (v * v)
    });
    terms.sort_by(|a, b| a.total_cmp(b));
    terms.iter().sum::<f64>().sqrt()
}

pub(crate) fn permute_coords(c: [f64; 6], perm: [usize; 6]) -> [f64; 6] {
    let mut out = [0.0; 6];
    for (slot, src) in out.iter_mut().zip(perm) {
        if let Some(&val) = c.get(src) {
            *slot = val;
        }
    }
    out
}

pub(crate) fn permute_pose(p: Pose, perm: [usize; 6]) -> Pose {
    pose(permute_coords(coords(&p), perm))
}

pub(crate) fn unpermute_pose(p: Pose, perm: [usize; 6]) -> Pose {
    let c = coords(&p);
    let mut out = [0.0; 6];
    for (src, dst) in c.iter().zip(perm) {
        if let Some(slot) = out.get_mut(dst) {
            *slot = *src;
        }
    }
    pose(out)
}
