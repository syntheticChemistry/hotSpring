// SPDX-License-Identifier: AGPL-3.0-or-later

use std::path::Path;

use crate::constants::WALL_ACTIVE_THRESHOLD;
use crate::parse::parse_binding_colvar;
use crate::types::BindingCheck;

/// Check binding distance from a COLVAR file.
///
/// `max_threshold_nm`: if max distance exceeds this, substrate dissociated.
/// Typical: 2.0 nm.
pub fn check_binding_distance(
    path: &Path,
    max_threshold_nm: f64,
) -> Result<BindingCheck, String> {
    let (distances, wall_biases) = parse_binding_colvar(path)?;

    let n = distances.len();
    let max_d = distances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_d = distances.iter().sum::<f64>() / n as f64;
    let final_d = *distances.last().unwrap_or(&0.0);
    let wall_active = wall_biases.iter().filter(|&&w| w > WALL_ACTIVE_THRESHOLD).count();
    let wall_frac = wall_active as f64 / n as f64;

    Ok(BindingCheck {
        max_distance_nm: max_d,
        mean_distance_nm: mean_d,
        final_distance_nm: final_d,
        n_frames: n,
        dissociated: max_d > max_threshold_nm,
        wall_active_fraction: wall_frac,
    })
}
