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
use super::vacf::compute_vacf;

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
    gpu_device: Option<Arc<WgpuDevice>>,
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

    // VACF
    if sim.velocity_snapshots.len() > 2 {
        let dt_dump = config.dt * config.dump_step as f64 * config.vel_snapshot_interval as f64;
        let max_lag = (sim.velocity_snapshots.len() / 2).max(10);
        let vacf = compute_vacf(
            &sim.velocity_snapshots,
            config.n_particles,
            dt_dump,
            max_lag,
        );
        println!(
            "    VACF: D*={:.4e} (from {} snapshots, {} lags)",
            vacf.diffusion_coeff,
            sim.velocity_snapshots.len(),
            max_lag
        );
    }

    // SSF — GPU or CPU path
    if !sim.positions_snapshots.is_empty() {
        let (ssf, ssf_label) = if let Some(ref dev) = gpu_device {
            let gpu_ssf = compute_ssf_gpu(
                Arc::clone(dev),
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
        } else {
            let cpu_ssf = compute_ssf(
                &sim.positions_snapshots,
                config.n_particles,
                config.box_side(),
                20,
            );
            (cpu_ssf, "CPU")
        };

        if let Some((k0, s0)) = ssf.first() {
            // SAFETY: first() returned Some, so ssf has ≥1 element; max_by always returns Some
            #[allow(clippy::expect_used)]
            let (k_max, s_max) = ssf
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("SSF non-empty: guarded by first() check");
            println!("    SSF [{ssf_label}]: S(k->0)={s0:.4} at k={k0:.3}");
            println!("    SSF [{ssf_label}]: peak S(k)={s_max:.4} at k={k_max:.3} a_ws^-1");
        }
    }

    println!(
        "    Performance: {:.1} steps/s, total {:.2}s",
        sim.steps_per_sec, sim.wall_time_s
    );
}
