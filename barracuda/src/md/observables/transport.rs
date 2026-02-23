// SPDX-License-Identifier: AGPL-3.0-only

//! Stress and heat current transport autocorrelations.
//!
//! Green-Kubo viscosity and thermal conductivity from MD snapshots.

use crate::tolerances::DIVISION_GUARD;

/// Plateau detection window: stop integrating Green-Kubo ACF after this
/// many seconds of non-increasing running integral (in time units / `dt_snap`).
const PLATEAU_DETECTION_TIME: f64 = 20.0;

/// Stress tensor autocorrelation result for viscosity computation
#[derive(Clone, Debug)]
pub struct StressAcf {
    pub t_values: Vec<f64>,
    pub c_values: Vec<f64>,
    pub viscosity: f64,
}

/// Compute off-diagonal stress tensor from positions, velocities, and Yukawa forces.
///
/// `sigma_xy`(t) = `sum_i` m×`v_ix`×`v_iy` + sum_{i<j} `F_ij_x` × `r_ij_y`
///
/// For transport, we only need the off-diagonal (xy) component.
/// Returns one scalar per snapshot.
#[must_use]
pub fn compute_stress_xy(
    pos_snapshots: &[Vec<f64>],
    vel_snapshots: &[Vec<f64>],
    n: usize,
    box_side: f64,
    kappa: f64,
    mass: f64,
) -> Vec<f64> {
    let mut stress_series = Vec::with_capacity(pos_snapshots.len());

    for (snap_idx, (pos, vel)) in pos_snapshots.iter().zip(vel_snapshots.iter()).enumerate() {
        if pos.len() < n * 3 || vel.len() < n * 3 {
            eprintln!("  StressXY: snapshot {snap_idx} too short, skipping");
            continue;
        }

        // Kinetic part: sum_i m * v_ix * v_iy
        let mut sigma_kin = 0.0;
        for i in 0..n {
            sigma_kin += mass * vel[i * 3] * vel[i * 3 + 1];
        }

        // Virial part: sum_{i<j} F_ij_x * r_ij_y
        // F_ij = prefactor * exp(-kappa*r) * (1 + kappa*r) / r^2 * r_hat
        let mut sigma_vir = 0.0;
        for i in 0..n {
            let xi = pos[i * 3];
            let yi = pos[i * 3 + 1];
            let zi = pos[i * 3 + 2];

            for j in (i + 1)..n {
                let mut dx = pos[j * 3] - xi;
                let mut dy = pos[j * 3 + 1] - yi;
                let mut dz = pos[j * 3 + 2] - zi;

                dx -= box_side * (dx / box_side).round();
                dy -= box_side * (dy / box_side).round();
                dz -= box_side * (dz / box_side).round();

                let r2 = dz.mul_add(dz, dy.mul_add(dy, dx * dx));
                let r = r2.sqrt();
                if r < DIVISION_GUARD {
                    continue;
                }

                let exp_kr = (-kappa * r).exp();
                let f_mag = exp_kr * kappa.mul_add(r, 1.0) / r2;

                // F_ij_x * r_ij_y = f_mag * (dx/r) * dy = f_mag * dx * dy / r
                sigma_vir += f_mag * dx * dy / r;
            }
        }

        stress_series.push(sigma_kin + sigma_vir);
    }

    stress_series
}

/// Compute stress autocorrelation and Green-Kubo viscosity.
///
/// `η*` = (V / `kT`) × `integral_0`^∞ ⟨`sigma_xy`(0) × `sigma_xy`(t)⟩ dt
///
/// `dt_snap` is the time between consecutive snapshots in reduced units.
#[must_use]
pub fn compute_stress_acf(
    stress_series: &[f64],
    dt_snap: f64,
    volume: f64,
    temperature: f64,
    max_lag: usize,
) -> StressAcf {
    let n_frames = stress_series.len();
    let n_lag = max_lag.min(n_frames);
    let mut c_values = vec![0.0f64; n_lag];
    let mut counts = vec![0usize; n_lag];

    for t0 in 0..n_frames {
        for lag in 0..n_lag {
            let t1 = t0 + lag;
            if t1 >= n_frames {
                break;
            }
            c_values[lag] += stress_series[t0] * stress_series[t1];
            counts[lag] += 1;
        }
    }

    for i in 0..n_lag {
        if counts[i] > 0 {
            c_values[i] /= counts[i] as f64;
        }
    }

    // Green-Kubo: eta = (1 / (V kT)) * integral <P_xy(0) P_xy(t)> dt
    // (stress series is TOTAL P_xy, not per-volume, so prefactor is 1/(VkT))
    // Plateau detection: stop when running integral peaks (noise dominates beyond)
    let prefactor = 1.0 / (volume * temperature.max(DIVISION_GUARD));
    let mut integral = 0.0;
    let mut eta_max = 0.0;
    let mut plateau_count = 0;
    let plateau_window = (PLATEAU_DETECTION_TIME / dt_snap).ceil() as usize;

    for i in 1..n_lag {
        integral += (0.5 * dt_snap).mul_add(c_values[i - 1] + c_values[i], 0.0);
        let eta_running = prefactor * integral;
        if eta_running > eta_max {
            eta_max = eta_running;
            plateau_count = 0;
        } else {
            plateau_count += 1;
            if plateau_count > plateau_window {
                break;
            }
        }
    }
    let viscosity = eta_max;

    let t_values: Vec<f64> = (0..n_lag).map(|i| i as f64 * dt_snap).collect();

    StressAcf {
        t_values,
        c_values,
        viscosity,
    }
}

/// Heat current autocorrelation result for thermal conductivity computation
#[derive(Clone, Debug)]
pub struct HeatAcf {
    pub t_values: Vec<f64>,
    pub c_values: Vec<f64>,
    pub thermal_conductivity: f64,
}

/// Compute the microscopic heat current `J_q`(t) from positions, velocities,
/// and the Yukawa interaction.
///
/// `J_q` = `sum_i` (`e_i` × `v_i`) + (1/2) sum_{i<j} (`F_ij` · `v_i`) `r_ij`
///
/// where `e_i` = (m/2)|`v_i`|² + (1/2) sum_{j≠i} u(`r_ij`) is the per-particle
/// energy. Returns one 3-vector (as [f64; 3]) per snapshot frame.
#[must_use]
pub fn compute_heat_current(
    pos_snapshots: &[Vec<f64>],
    vel_snapshots: &[Vec<f64>],
    n: usize,
    box_side: f64,
    kappa: f64,
    mass: f64,
) -> Vec<[f64; 3]> {
    let mut jq_series = Vec::with_capacity(pos_snapshots.len());

    for (pos, vel) in pos_snapshots.iter().zip(vel_snapshots.iter()) {
        if pos.len() < n * 3 || vel.len() < n * 3 {
            continue;
        }

        let mut pe_i = vec![0.0f64; n];
        let mut jq = [0.0f64; 3];

        // Per-particle PE from Yukawa pairs and virial heat current
        for i in 0..n {
            let xi = pos[i * 3];
            let yi = pos[i * 3 + 1];
            let zi = pos[i * 3 + 2];
            let vix = vel[i * 3];
            let viy = vel[i * 3 + 1];
            let viz = vel[i * 3 + 2];

            for j in (i + 1)..n {
                let mut dx = pos[j * 3] - xi;
                let mut dy = pos[j * 3 + 1] - yi;
                let mut dz = pos[j * 3 + 2] - zi;

                dx -= box_side * (dx / box_side).round();
                dy -= box_side * (dy / box_side).round();
                dz -= box_side * (dz / box_side).round();

                let r2 = dz.mul_add(dz, dy.mul_add(dy, dx * dx));
                let r = r2.sqrt();
                if r < DIVISION_GUARD {
                    continue;
                }

                let exp_kr = (-kappa * r).exp();
                let u_pair = exp_kr / r;
                pe_i[i] += 0.5 * u_pair;
                pe_i[j] += 0.5 * u_pair;

                let f_mag = exp_kr * kappa.mul_add(r, 1.0) / r2;
                let inv_r = 1.0 / r;
                let fx = f_mag * dx * inv_r;
                let fy = f_mag * dy * inv_r;
                let fz = f_mag * dz * inv_r;

                let vjx = vel[j * 3];
                let vjy = vel[j * 3 + 1];
                let vjz = vel[j * 3 + 2];

                let f_dot_vi = fz.mul_add(viz, fy.mul_add(viy, fx * vix));
                let f_dot_vj = -fz.mul_add(vjz, fy.mul_add(vjy, fx * vjx));

                jq[0] += 0.5 * (f_dot_vi + f_dot_vj) * dx;
                jq[1] += 0.5 * (f_dot_vi + f_dot_vj) * dy;
                jq[2] += 0.5 * (f_dot_vi + f_dot_vj) * dz;
            }
        }

        // Convective part: sum_i e_i * v_i
        for i in 0..n {
            let ke_i = 0.5
                * mass
                * vel[i * 3 + 2].mul_add(
                    vel[i * 3 + 2],
                    vel[i * 3 + 1].mul_add(vel[i * 3 + 1], vel[i * 3] * vel[i * 3]),
                );
            let e_i = ke_i + pe_i[i];
            jq[0] += e_i * vel[i * 3];
            jq[1] += e_i * vel[i * 3 + 1];
            jq[2] += e_i * vel[i * 3 + 2];
        }

        jq_series.push(jq);
    }

    jq_series
}

/// Compute heat current autocorrelation and Green-Kubo thermal conductivity.
///
/// `λ*` = (V / (3 `kT`²)) × `integral_0`^∞ ⟨`J_q`(0) · `J_q`(t)⟩ dt
#[must_use]
pub fn compute_heat_acf(
    jq_series: &[[f64; 3]],
    dt_snap: f64,
    volume: f64,
    temperature: f64,
    max_lag: usize,
) -> HeatAcf {
    let n_frames = jq_series.len();
    let n_lag = max_lag.min(n_frames);
    let mut c_values = vec![0.0f64; n_lag];
    let mut counts = vec![0usize; n_lag];

    for t0 in 0..n_frames {
        for lag in 0..n_lag {
            let t1 = t0 + lag;
            if t1 >= n_frames {
                break;
            }
            let j0 = &jq_series[t0];
            let j1 = &jq_series[t1];
            c_values[lag] += j0[2].mul_add(j1[2], j0[1].mul_add(j1[1], j0[0] * j1[0]));
            counts[lag] += 1;
        }
    }

    for i in 0..n_lag {
        if counts[i] > 0 {
            c_values[i] /= counts[i] as f64;
        }
    }

    // Green-Kubo: lambda = (1 / (3 V kT^2)) * integral <J_q(0)·J_q(t)> dt
    // (heat current is TOTAL J_q, not per-volume, so prefactor is 1/(3VkT^2))
    // Plateau detection: stop when running integral peaks
    let t2 = temperature * temperature;
    let prefactor = 1.0 / (3.0 * volume * t2.max(DIVISION_GUARD));
    let mut integral = 0.0;
    let mut lambda_max = 0.0;
    let mut plateau_count = 0;
    let plateau_window = (PLATEAU_DETECTION_TIME / dt_snap).ceil() as usize;

    for i in 1..n_lag {
        integral += (0.5 * dt_snap).mul_add(c_values[i - 1] + c_values[i], 0.0);
        let lambda_running = prefactor * integral;
        if lambda_running > lambda_max {
            lambda_max = lambda_running;
            plateau_count = 0;
        } else {
            plateau_count += 1;
            if plateau_count > plateau_window {
                break;
            }
        }
    }
    let thermal_conductivity = lambda_max;

    let t_values: Vec<f64> = (0..n_lag).map(|i| i as f64 * dt_snap).collect();

    HeatAcf {
        t_values,
        c_values,
        thermal_conductivity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_stress_xy_kinetic_and_virial() {
        let pos = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2 particles
        let vel = vec![1.0, 1.0, 0.0, -1.0, -1.0, 0.0]; // non-zero vx*vy
        let stress = compute_stress_xy(&[pos], &[vel], 2, 5.0, 2.0, 3.0);
        assert_eq!(stress.len(), 1);
        assert!(
            stress[0].abs() > 0.0,
            "stress_xy should be non-zero for moving particles"
        );
    }

    #[test]
    fn compute_stress_xy_skips_invalid_snapshot() {
        let pos = vec![0.0, 0.0, 0.0]; // too short for n=2
        let vel = vec![0.0, 0.0, 0.0];
        let pos_ok = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let vel_ok = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let stress = compute_stress_xy(&[pos, pos_ok], &[vel, vel_ok], 2, 5.0, 2.0, 3.0);
        assert_eq!(stress.len(), 1, "first snapshot too short, second ok");
    }

    #[test]
    fn compute_stress_acf_basic() {
        let stress = vec![1.0, 0.5, 0.25, 0.1]; // decaying
        let acf = compute_stress_acf(&stress, 0.1, 100.0, 0.01, 4);
        assert_eq!(acf.t_values.len(), 4);
        assert_eq!(acf.c_values.len(), 4);
        assert!(acf.viscosity >= 0.0);
    }

    #[test]
    fn compute_heat_current_basic() {
        let pos = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let vel = vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0];
        let jq = compute_heat_current(&[pos], &[vel], 2, 5.0, 2.0, 3.0);
        assert_eq!(jq.len(), 1);
        assert_eq!(jq[0].len(), 3);
    }

    #[test]
    fn compute_heat_current_skips_short_snapshots() {
        let pos_short = vec![0.0, 0.0, 0.0];
        let vel_short = vec![0.0, 0.0, 0.0];
        let jq = compute_heat_current(&[pos_short], &[vel_short], 2, 5.0, 2.0, 3.0);
        assert!(jq.is_empty(), "too-short snapshot should be skipped");
    }

    #[test]
    fn compute_heat_acf_basic() {
        let jq = vec![[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.0, 0.0]];
        let acf = compute_heat_acf(&jq, 0.1, 100.0, 0.01, 3);
        assert_eq!(acf.t_values.len(), 3);
        assert_eq!(acf.c_values.len(), 3);
        assert!(acf.thermal_conductivity >= 0.0);
    }
}
