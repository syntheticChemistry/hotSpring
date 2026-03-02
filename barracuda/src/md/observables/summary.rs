// SPDX-License-Identifier: AGPL-3.0-only

//! Observable summary printing for MD validation runs.
//!
//! Aggregates RDF, VACF, SSF, and energy metrics into a human-readable report.

use std::sync::Arc;

use barracuda::device::WgpuDevice;

use crate::md::config::MdConfig;
use crate::md::simulation::MdSimulation;
use crate::tolerances::RDF_TAIL_TOLERANCE;

use super::energy::validate_energy;
use super::rdf::compute_rdf;
use super::ssf::{compute_ssf, compute_ssf_gpu};
use super::vacf::{compute_vacf, compute_vacf_upstream_gpu};

/// Print summary of all observables.
///
/// If `gpu_device` is provided, SSF is computed on GPU via `SsfGpu::compute_axes`.
/// Otherwise falls back to CPU `compute_ssf`.
pub fn print_observable_summary(sim: &MdSimulation, config: &MdConfig) {
    print_observable_summary_with_gpu(sim, config, None);
}

/// Print observable summary with optional GPU device for SSF.
pub fn print_observable_summary_with_gpu(
    sim: &MdSimulation,
    config: &MdConfig,
    gpu_device: Option<&Arc<WgpuDevice>>,
) {
    println!();
    println!("  ── Observable Summary: {} ──", config.label);

    // Energy validation
    let energy_val = validate_energy(&sim.energy_history, config);
    let icon = if energy_val.passed { "PASS" } else { "FAIL" };
    println!(
        "    Energy: drift={:.3}% [{}] (< 5% required)",
        energy_val.drift_pct, icon
    );
    println!(
        "    Temperature: {:.6} +/- {:.6} (target {:.6})",
        energy_val.mean_temperature,
        energy_val.std_temperature,
        config.temperature()
    );

    // RDF
    if !sim.positions_snapshots.is_empty() {
        let rdf = compute_rdf(
            &sim.positions_snapshots,
            config.n_particles,
            config.box_side(),
            config.rdf_bins,
        );
        // Find first peak
        let peak_idx = rdf
            .g_values
            .iter()
            .enumerate()
            .skip(1) // skip r=0
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i);
        let peak_r = rdf.r_values[peak_idx];
        let peak_g = rdf.g_values[peak_idx];

        // Check g(r) → 1 at large r
        let tail_start = (rdf.g_values.len() * 3) / 4;
        let tail_mean: f64 = rdf.g_values[tail_start..].iter().sum::<f64>()
            / (rdf.g_values.len() - tail_start) as f64;
        let tail_err = (tail_mean - 1.0).abs();
        let rdf_icon = if tail_err < RDF_TAIL_TOLERANCE {
            "PASS"
        } else {
            "FAIL"
        };

        println!("    RDF: peak at r={peak_r:.3} a_ws, g(peak)={peak_g:.3}");
        println!("    RDF: tail asymptote={tail_mean:.4} (err={tail_err:.4}) [{rdf_icon}]");
    }

    // VACF — GPU (upstream barracuda) or CPU path
    if sim.velocity_snapshots.len() > 2 {
        let dt_dump = config.dt * config.dump_step as f64 * config.vel_snapshot_interval as f64;
        let max_lag = (sim.velocity_snapshots.len() / 2).max(10);
        let (vacf, vacf_label) = gpu_device
            .and_then(|dev| {
                compute_vacf_upstream_gpu(
                    dev,
                    &sim.velocity_snapshots,
                    config.n_particles,
                    dt_dump,
                    max_lag,
                )
                .map(|v| (v, "GPU barracuda"))
            })
            .unwrap_or_else(|| {
                let v = compute_vacf(
                    &sim.velocity_snapshots,
                    config.n_particles,
                    dt_dump,
                    max_lag,
                );
                (v, "CPU")
            });
        println!(
            "    VACF [{vacf_label}]: D*={:.4e} (from {} snapshots, {} lags)",
            vacf.diffusion_coeff,
            sim.velocity_snapshots.len(),
            max_lag
        );
    }

    // SSF — GPU or CPU path
    if !sim.positions_snapshots.is_empty() {
        let (ssf, ssf_label): (Vec<(f64, f64)>, &str) = gpu_device.map_or_else(
            || {
                let cpu_ssf = compute_ssf(
                    &sim.positions_snapshots,
                    config.n_particles,
                    config.box_side(),
                    20,
                );
                (cpu_ssf, "CPU")
            },
            |dev| {
                let gpu_ssf = compute_ssf_gpu(
                    dev,
                    &sim.positions_snapshots,
                    config.n_particles,
                    config.box_side(),
                    20,
                );
                if gpu_ssf.is_empty() {
                    // GPU failed, fall back to CPU
                    let cpu_ssf = compute_ssf(
                        &sim.positions_snapshots,
                        config.n_particles,
                        config.box_side(),
                        20,
                    );
                    (cpu_ssf, "CPU fallback")
                } else {
                    (gpu_ssf, "GPU SsfGpu")
                }
            },
        );

        if let (Some((k0, s0)), Some((k_max, s_max))) = (
            ssf.first(),
            ssf.iter()
                .max_by(|a: &&(f64, f64), b: &&(f64, f64)| a.1.total_cmp(&b.1)),
        ) {
            println!("    SSF [{ssf_label}]: S(k->0)={s0:.4} at k={k0:.3}");
            println!("    SSF [{ssf_label}]: peak S(k)={s_max:.4} at k={k_max:.3} a_ws^-1");
        }
    }

    println!(
        "    Performance: {:.1} steps/s, total {:.2}s",
        sim.steps_per_sec, sim.wall_time_s
    );
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use std::sync::Arc;

    use crate::md::config::quick_test_case;
    use crate::md::simulation::{init_fcc_lattice, init_velocities, EnergyRecord, MdSimulation};

    use super::{print_observable_summary, print_observable_summary_with_gpu};

    fn make_minimal_sim() -> (MdSimulation, crate::md::config::MdConfig) {
        let config = quick_test_case(32);
        let t = config.temperature();
        let energy_history: Vec<EnergyRecord> = (0..20)
            .map(|i| EnergyRecord {
                step: i * 10,
                ke: 48.0 * t,
                pe: -96.0,
                total: 48.0 * t - 96.0,
                temperature: t,
            })
            .collect();
        let sim = MdSimulation {
            config: config.clone(),
            energy_history,
            positions_snapshots: vec![],
            velocity_snapshots: vec![],
            rdf_histogram: vec![],
            wall_time_s: 1.5,
            steps_per_sec: 5000.0,
        };
        (sim, config)
    }

    fn make_full_sim() -> (MdSimulation, crate::md::config::MdConfig) {
        let config = quick_test_case(32);
        let box_side = config.box_side();
        let t = config.temperature();
        let mass = config.reduced_mass();
        let n = 32;

        let (positions, _) = init_fcc_lattice(n, box_side);
        let velocities = init_velocities(n, t, mass, 42);

        let n_pos_snapshots = 3;
        let positions_snapshots = vec![positions; n_pos_snapshots];

        let n_vel_snapshots = 15;
        let velocity_snapshots = vec![velocities; n_vel_snapshots];

        let energy_history: Vec<EnergyRecord> = (0..50)
            .map(|i| EnergyRecord {
                step: i * 10,
                ke: 48.0 * t,
                pe: -96.0,
                total: 48.0 * t - 96.0,
                temperature: t,
            })
            .collect();

        let sim = MdSimulation {
            config: config.clone(),
            energy_history,
            positions_snapshots,
            velocity_snapshots,
            rdf_histogram: vec![],
            wall_time_s: 2.0,
            steps_per_sec: 4000.0,
        };
        (sim, config)
    }

    #[test]
    fn summary_minimal_no_panic() {
        let (sim, config) = make_minimal_sim();
        print_observable_summary(&sim, &config);
    }

    #[test]
    fn summary_full_all_branches_no_panic() {
        let (sim, config) = make_full_sim();
        print_observable_summary(&sim, &config);
    }

    #[test]
    fn summary_gpu_none_cpu_fallback_no_panic() {
        let (sim, config) = make_full_sim();
        print_observable_summary_with_gpu(
            &sim,
            &config,
            None::<Arc<barracuda::device::WgpuDevice>>.as_ref(),
        );
    }

    #[test]
    fn summary_skips_vacf_when_too_few_velocity_snapshots() {
        let (sim, config) = make_full_sim();
        let sim_with_two_vel = MdSimulation {
            velocity_snapshots: sim.velocity_snapshots[..2].to_vec(),
            ..sim
        };
        print_observable_summary(&sim_with_two_vel, &config);
    }

    #[test]
    fn summary_energy_fail_icon_when_high_drift() {
        let config = quick_test_case(32);
        let t = config.temperature();
        let e_mean = 48.0 * t - 96.0;
        let drift = 0.01; // 1% drift — above ENERGY_DRIFT_PCT 0.5%
        let e_initial = e_mean;
        let e_final = e_mean * (1.0 + drift);
        let energy_history: Vec<EnergyRecord> = (0..20)
            .map(|i| {
                let e = if i == 0 {
                    e_initial
                } else if i == 19 {
                    e_final
                } else {
                    e_mean
                };
                EnergyRecord {
                    step: i * 10,
                    ke: 48.0 * t,
                    pe: e - 48.0 * t,
                    total: e,
                    temperature: t,
                }
            })
            .collect();
        let sim = MdSimulation {
            config: config.clone(),
            energy_history,
            positions_snapshots: vec![],
            velocity_snapshots: vec![],
            rdf_histogram: vec![],
            wall_time_s: 1.0,
            steps_per_sec: 5000.0,
        };
        print_observable_summary(&sim, &config);
    }
}
