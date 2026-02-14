//! Observable computation for MD validation
//!
//! Computes RDF, VACF, SSF, and energy metrics from simulation snapshots.
//! CPU path computes from GPU-generated snapshots.
//! GPU path uses toadstool's SsfGpu for O(N) GPU-accelerated S(k).

use std::f64::consts::PI;
use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::md::observables::SsfGpu;

use crate::md::config::MdConfig;
use crate::md::simulation::{EnergyRecord, MdSimulation};

/// RDF result: g(r) binned at discrete r values
#[derive(Clone, Debug)]
pub struct Rdf {
    pub r_values: Vec<f64>,   // bin centers in a_ws
    pub g_values: Vec<f64>,   // g(r)
    pub dr: f64,              // bin width
}

/// VACF result: C(t) at discrete lag times
#[derive(Clone, Debug)]
pub struct Vacf {
    pub t_values: Vec<f64>,   // lag times in omega_p^-1
    pub c_values: Vec<f64>,   // normalized C(t) / C(0)
    pub diffusion_coeff: f64, // D* = (1/3) integral C(t) dt
}

/// Energy validation result
#[derive(Clone, Debug)]
pub struct EnergyValidation {
    pub mean_total: f64,
    pub std_total: f64,
    pub drift_pct: f64,
    pub mean_temperature: f64,
    pub std_temperature: f64,
    pub passed: bool,
}

/// Compute RDF from position snapshots (CPU post-process)
pub fn compute_rdf(
    snapshots: &[Vec<f64>],
    n: usize,
    box_side: f64,
    n_bins: usize,
) -> Rdf {
    let r_max = box_side / 2.0; // max meaningful distance with PBC
    let dr = r_max / n_bins as f64;
    let mut histogram = vec![0u64; n_bins];
    let n_frames = snapshots.len();

    for snap in snapshots {
        for i in 0..n {
            let xi = snap[i * 3];
            let yi = snap[i * 3 + 1];
            let zi = snap[i * 3 + 2];

            for j in (i + 1)..n {
                let mut dx = snap[j * 3] - xi;
                let mut dy = snap[j * 3 + 1] - yi;
                let mut dz = snap[j * 3 + 2] - zi;

                // PBC minimum image
                dx -= box_side * (dx / box_side).round();
                dy -= box_side * (dy / box_side).round();
                dz -= box_side * (dz / box_side).round();

                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let bin = (r / dr) as usize;
                if bin < n_bins {
                    histogram[bin] += 1;
                }
            }
        }
    }

    // Normalize: g(r) = histogram / (n_frames * N * n_density * 4π r² dr)
    let n_density = 3.0 / (4.0 * PI); // reduced units
    let n_f = n as f64;
    let r_values: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * dr).collect();
    let g_values: Vec<f64> = r_values
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            let shell_vol = 4.0 * PI * r * r * dr;
            let expected = n_f * n_density * shell_vol;
            // Factor 2 because we count pairs i<j, but g(r) normalizes per particle
            2.0 * histogram[i] as f64 / (n_frames as f64 * n_f * n_density * shell_vol).max(1e-30)
        })
        .collect();

    Rdf { r_values, g_values, dr }
}

/// Compute VACF from velocity snapshots (CPU post-process)
pub fn compute_vacf(
    vel_snapshots: &[Vec<f64>],
    n: usize,
    dt_dump: f64,
    max_lag: usize,
) -> Vacf {
    let n_frames = vel_snapshots.len();
    let n_lag = max_lag.min(n_frames);
    let mut c_values = vec![0.0f64; n_lag];
    let mut counts = vec![0usize; n_lag];

    for t0 in 0..n_frames {
        for lag in 0..n_lag {
            let t1 = t0 + lag;
            if t1 >= n_frames {
                break;
            }
            let v0 = &vel_snapshots[t0];
            let v1 = &vel_snapshots[t1];

            let mut dot_sum = 0.0;
            for i in 0..n {
                dot_sum += v0[i * 3] * v1[i * 3]
                    + v0[i * 3 + 1] * v1[i * 3 + 1]
                    + v0[i * 3 + 2] * v1[i * 3 + 2];
            }
            c_values[lag] += dot_sum / n as f64;
            counts[lag] += 1;
        }
    }

    // Average over time origins
    for i in 0..n_lag {
        if counts[i] > 0 {
            c_values[i] /= counts[i] as f64;
        }
    }

    // Normalize by C(0)
    let c0 = c_values[0].max(1e-30);
    let c_normalized: Vec<f64> = c_values.iter().map(|&c| c / c0).collect();

    // Diffusion coefficient: D* = (1/3) integral_0^inf C(t) dt
    // Trapezoidal rule
    let mut integral = 0.0;
    for i in 1..n_lag {
        integral += 0.5 * (c_values[i - 1] + c_values[i]) * dt_dump;
    }
    let diffusion_coeff = integral / 3.0;

    let t_values: Vec<f64> = (0..n_lag).map(|i| i as f64 * dt_dump).collect();

    Vacf {
        t_values,
        c_values: c_normalized,
        diffusion_coeff,
    }
}

/// Compute static structure factor S(k) from position snapshots
pub fn compute_ssf(
    snapshots: &[Vec<f64>],
    n: usize,
    box_side: f64,
    max_k_harmonics: usize,
) -> Vec<(f64, f64)> {
    let dk = 2.0 * PI / box_side;
    let n_frames = snapshots.len();
    let mut sk_values: Vec<(f64, f64)> = Vec::new();

    // Compute S(k) for k-vectors along principal axes
    for kn in 1..=max_k_harmonics {
        let k_mag = kn as f64 * dk;
        let mut sk_sum = 0.0;
        let mut count = 0;

        for snap in snapshots {
            // S(k) = <|rho(k)|²> / N
            // rho(k) = sum_j exp(i k . r_j)
            // For k along x-axis: rho_x = sum_j exp(i kx * x_j)
            for axis in 0..3 {
                let mut re = 0.0;
                let mut im = 0.0;
                for j in 0..n {
                    let r_component = snap[j * 3 + axis];
                    let phase = k_mag * r_component;
                    re += phase.cos();
                    im += phase.sin();
                }
                sk_sum += (re * re + im * im) / n as f64;
                count += 1;
            }
        }

        sk_values.push((k_mag, sk_sum / count as f64));
    }

    sk_values
}

/// Compute S(k) using toadstool's SsfGpu, averaged over snapshots.
///
/// This mirrors `compute_ssf` but runs each snapshot on the GPU via
/// `SsfGpu::compute_axes`. Falls back to CPU if GPU dispatch fails.
pub fn compute_ssf_gpu(
    device: Arc<WgpuDevice>,
    snapshots: &[Vec<f64>],
    _n: usize,
    box_side: f64,
    max_k_harmonics: usize,
) -> Vec<(f64, f64)> {
    if snapshots.is_empty() {
        return Vec::new();
    }

    // Accumulator: (k, sum_of_sk, count)
    let mut accumulator: Vec<(f64, f64, usize)> = Vec::new();

    for snap in snapshots {
        match SsfGpu::compute_axes(device.clone(), snap, box_side, max_k_harmonics) {
            Ok(sk_pairs) => {
                // Grow accumulator on first snapshot
                if accumulator.is_empty() {
                    accumulator = sk_pairs.iter().map(|&(k, sk)| (k, sk, 1)).collect();
                } else {
                    for (i, &(_k, sk)) in sk_pairs.iter().enumerate() {
                        if i < accumulator.len() {
                            accumulator[i].1 += sk;
                            accumulator[i].2 += 1;
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("  SsfGpu::compute_axes failed: {} — skipping snapshot", e);
            }
        }
    }

    // Average over snapshots
    accumulator
        .into_iter()
        .map(|(k, sum_sk, count)| {
            (k, if count > 0 { sum_sk / count as f64 } else { 0.0 })
        })
        .collect()
}

/// Validate energy conservation
pub fn validate_energy(history: &[EnergyRecord], config: &MdConfig) -> EnergyValidation {
    if history.is_empty() {
        return EnergyValidation {
            mean_total: 0.0,
            std_total: 0.0,
            drift_pct: 0.0,
            mean_temperature: 0.0,
            std_temperature: 0.0,
            passed: false,
        };
    }

    // Skip first 10% of production for transient effects
    let skip = history.len() / 10;
    let stable = &history[skip..];

    let mean_e: f64 = stable.iter().map(|e| e.total).sum::<f64>() / stable.len() as f64;
    let var_e: f64 = stable
        .iter()
        .map(|e| (e.total - mean_e).powi(2))
        .sum::<f64>()
        / stable.len() as f64;
    let std_e = var_e.sqrt();

    // Drift: (E_final - E_initial) / |E_mean|
    let e_initial = stable.first().unwrap().total;
    let e_final = stable.last().unwrap().total;
    let drift_pct = if mean_e.abs() > 1e-30 {
        ((e_final - e_initial) / mean_e.abs()).abs() * 100.0
    } else {
        0.0
    };

    let mean_t: f64 = stable.iter().map(|e| e.temperature).sum::<f64>() / stable.len() as f64;
    let var_t: f64 = stable
        .iter()
        .map(|e| (e.temperature - mean_t).powi(2))
        .sum::<f64>()
        / stable.len() as f64;
    let std_t = var_t.sqrt();

    let passed = drift_pct < 5.0; // acceptance criterion

    EnergyValidation {
        mean_total: mean_e,
        std_total: std_e,
        drift_pct,
        mean_temperature: mean_t,
        std_temperature: std_t,
        passed,
    }
}

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
    println!("    Energy: drift={:.3}% [{}] (< 5% required)", energy_val.drift_pct, icon);
    println!(
        "    Temperature: {:.6} +/- {:.6} (target {:.6})",
        energy_val.mean_temperature, energy_val.std_temperature, config.temperature()
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
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let peak_r = rdf.r_values[peak_idx];
        let peak_g = rdf.g_values[peak_idx];

        // Check g(r) → 1 at large r
        let tail_start = (rdf.g_values.len() * 3) / 4;
        let tail_mean: f64 = rdf.g_values[tail_start..]
            .iter()
            .sum::<f64>()
            / (rdf.g_values.len() - tail_start) as f64;
        let tail_err = (tail_mean - 1.0).abs();
        let rdf_icon = if tail_err < 0.15 { "PASS" } else { "FAIL" };

        println!("    RDF: peak at r={:.3} a_ws, g(peak)={:.3}", peak_r, peak_g);
        println!("    RDF: tail asymptote={:.4} (err={:.4}) [{}]", tail_mean, tail_err, rdf_icon);
    }

    // VACF
    if sim.velocity_snapshots.len() > 2 {
        let dt_dump = config.dt * config.dump_step as f64 * 100.0; // stored every 100 dumps
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
                dev.clone(),
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
            let (k_max, s_max) = ssf
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            println!("    SSF [{}]: S(k->0)={:.4} at k={:.3}", ssf_label, s0, k0);
            println!("    SSF [{}]: peak S(k)={:.4} at k={:.3} a_ws^-1", ssf_label, s_max, k_max);
        }
    }

    println!("    Performance: {:.1} steps/s, total {:.2}s", sim.steps_per_sec, sim.wall_time_s);
}
