// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-GPU temperature scan dispatcher for lattice QCD.
//!
//! Distributes independent temperature points across available GPUs
//! in the basement HPC mesh. Each temperature point runs an independent
//! HMC simulation — embarrassingly parallel.
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────┐
//! │  Temperature Scan Dispatcher                          │
//! │  β_1, β_2, ..., β_N_T                                │
//! │                                                       │
//! │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
//! │  │ GPU 0   │ │ GPU 1   │ │ GPU 2   │ │ ...     │   │
//! │  │ β_1 HMC │ │ β_2 HMC │ │ β_3 HMC │ │         │   │
//! │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
//! │                                                       │
//! │  Results: plaquette(β), Polyakov(β), action(β)        │
//! └───────────────────────────────────────────────────────┘
//! ```
//!
//! # References
//!
//! - `HotQCD` Collaboration workflow for temperature scans
//! - ecoPrimals/whitePaper/gen3/about/HARDWARE.md — basement HPC inventory

use super::hmc::{self, HmcConfig};
use super::wilson::Lattice;

use std::thread;
use std::time::Instant;

/// Configuration for a temperature scan.
#[derive(Clone, Debug)]
pub struct TemperatureScanConfig {
    /// Lattice dimensions [Ns, Ns, Ns, Nt]
    pub dims: [usize; 4],
    /// β values to scan (β = 6/g²)
    pub beta_values: Vec<f64>,
    /// HMC trajectories per β point
    pub n_trajectories: usize,
    /// Thermalization trajectories (discarded)
    pub n_thermalization: usize,
    /// HMC step size
    pub hmc_dt: f64,
    /// HMC leapfrog steps
    pub hmc_n_md_steps: usize,
    /// Number of parallel workers (threads)
    pub n_workers: usize,
    /// Random seed base (each `β` gets `seed_base` + index)
    pub seed_base: u64,
}

impl Default for TemperatureScanConfig {
    fn default() -> Self {
        Self {
            dims: [4, 4, 4, 4],
            beta_values: vec![5.0, 5.2, 5.4, 5.6, 5.7, 5.8, 6.0, 6.2, 6.5],
            n_trajectories: 50,
            n_thermalization: 20,
            hmc_dt: 0.05,
            hmc_n_md_steps: 15,
            n_workers: 4,
            seed_base: 42,
        }
    }
}

/// Result for a single temperature point.
#[derive(Clone, Debug)]
pub struct TemperaturePoint {
    pub beta: f64,
    pub mean_plaquette: f64,
    pub std_plaquette: f64,
    pub polyakov_loop: f64,
    pub acceptance_rate: f64,
    pub wall_time_s: f64,
}

/// Result of a full temperature scan.
#[derive(Clone, Debug)]
pub struct TemperatureScanResult {
    pub points: Vec<TemperaturePoint>,
    pub total_wall_time_s: f64,
}

/// Run a temperature scan on CPU threads (multi-core parallel).
///
/// Each β value runs an independent HMC simulation on a separate thread.
/// For GPU acceleration, each thread would own a `GpuF64` device instead.
pub fn run_temperature_scan(config: &TemperatureScanConfig) -> TemperatureScanResult {
    let t_start = Instant::now();
    let n_beta = config.beta_values.len();
    let n_workers = config.n_workers.min(n_beta);

    println!("  Temperature scan: {n_beta} β points × {n_workers} workers");
    println!("  Lattice: {:?}", config.dims);
    println!(
        "  HMC: {} traj + {} therm, dt={}, n_md={}",
        config.n_trajectories, config.n_thermalization, config.hmc_dt, config.hmc_n_md_steps
    );
    println!();

    // Partition β values across workers
    let chunks: Vec<Vec<(usize, f64)>> = {
        let indexed: Vec<(usize, f64)> = config
            .beta_values
            .iter()
            .enumerate()
            .map(|(i, &b)| (i, b))
            .collect();
        let chunk_size = indexed.len().div_ceil(n_workers);
        indexed
            .chunks(chunk_size)
            .map(<[(usize, f64)]>::to_vec)
            .collect()
    };

    let dims = config.dims;
    let n_traj = config.n_trajectories;
    let n_therm = config.n_thermalization;
    let hmc_dt = config.hmc_dt;
    let hmc_n_md = config.hmc_n_md_steps;
    let seed_base = config.seed_base;

    let handles: Vec<_> = chunks
        .into_iter()
        .map(|chunk| {
            thread::spawn(move || {
                let mut results = Vec::new();
                for (idx, beta) in chunk {
                    let t_point = Instant::now();
                    let mut lat = Lattice::hot_start(dims, beta, seed_base + idx as u64);
                    let mut hmc_config = HmcConfig {
                        n_md_steps: hmc_n_md,
                        dt: hmc_dt,
                        seed: seed_base + idx as u64 * 1000,
                        ..Default::default()
                    };

                    let stats = hmc::run_hmc(&mut lat, n_traj, n_therm, &mut hmc_config);
                    let poly = lat.average_polyakov_loop();

                    results.push(TemperaturePoint {
                        beta,
                        mean_plaquette: stats.mean_plaquette,
                        std_plaquette: stats.std_plaquette,
                        polyakov_loop: poly,
                        acceptance_rate: stats.acceptance_rate,
                        wall_time_s: t_point.elapsed().as_secs_f64(),
                    });
                }
                results
            })
        })
        .collect();

    let mut all_points: Vec<TemperaturePoint> = Vec::new();
    for handle in handles {
        match handle.join() {
            Ok(points) => all_points.extend(points),
            Err(e) => eprintln!("  Worker failed: {e:?}"),
        }
    }

    all_points.sort_by(|a, b| a.beta.total_cmp(&b.beta));

    let total_time = t_start.elapsed().as_secs_f64();

    TemperatureScanResult {
        points: all_points,
        total_wall_time_s: total_time,
    }
}

/// Print a formatted temperature scan summary.
pub fn print_scan_summary(result: &TemperatureScanResult) {
    println!(
        "  {:>6} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "β", "<plaq>", "σ(plaq)", "|L|", "acc%", "time"
    );
    for p in &result.points {
        println!(
            "  {:>6.2} {:>10.6} {:>10.6} {:>10.6} {:>7.1}% {:>7.2}s",
            p.beta,
            p.mean_plaquette,
            p.std_plaquette,
            p.polyakov_loop,
            p.acceptance_rate * 100.0,
            p.wall_time_s,
        );
    }
    println!(
        "\n  Total wall time: {:.1}s ({:.1}× parallel speedup)",
        result.total_wall_time_s,
        result.points.iter().map(|p| p.wall_time_s).sum::<f64>() / result.total_wall_time_s
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_with_two_beta_points() {
        let config = TemperatureScanConfig {
            dims: [4, 4, 4, 4],
            beta_values: vec![5.0, 6.0],
            n_trajectories: 5,
            n_thermalization: 3,
            hmc_dt: 0.1,
            hmc_n_md_steps: 5,
            n_workers: 2,
            seed_base: 42,
        };

        let result = run_temperature_scan(&config);
        assert_eq!(result.points.len(), 2);
        assert!(result.points[0].beta < result.points[1].beta);

        // Plaquette should be larger at larger β (weaker coupling = more ordered)
        assert!(
            result.points[1].mean_plaquette > result.points[0].mean_plaquette,
            "plaquette should increase with β"
        );
    }

    #[test]
    fn default_config_is_valid() {
        let config = TemperatureScanConfig::default();
        assert!(!config.beta_values.is_empty());
        assert!(config.n_workers > 0);
        assert!(config.hmc_dt > 0.0);
    }

    #[test]
    fn scan_single_beta_single_worker() {
        let config = TemperatureScanConfig {
            dims: [4, 4, 4, 4],
            beta_values: vec![5.5],
            n_trajectories: 3,
            n_thermalization: 2,
            hmc_dt: 0.1,
            hmc_n_md_steps: 5,
            n_workers: 1,
            seed_base: 123,
        };

        let result = run_temperature_scan(&config);
        assert_eq!(result.points.len(), 1);
        assert!((result.points[0].beta - 5.5).abs() < 1e-10);
    }

    #[test]
    fn scan_n_workers_exceeds_n_beta_clamped() {
        let config = TemperatureScanConfig {
            dims: [4, 4, 4, 4],
            beta_values: vec![5.0, 6.0],
            n_trajectories: 2,
            n_thermalization: 1,
            hmc_dt: 0.1,
            hmc_n_md_steps: 3,
            n_workers: 10,
            seed_base: 0,
        };

        let result = run_temperature_scan(&config);
        assert_eq!(result.points.len(), 2);
    }

    #[test]
    fn print_scan_summary_no_panic() {
        let result = TemperatureScanResult {
            points: vec![
                TemperaturePoint {
                    beta: 5.0,
                    mean_plaquette: 0.42,
                    std_plaquette: 0.03,
                    polyakov_loop: 0.1,
                    acceptance_rate: 0.75,
                    wall_time_s: 1.5,
                },
                TemperaturePoint {
                    beta: 6.0,
                    mean_plaquette: 0.55,
                    std_plaquette: 0.02,
                    polyakov_loop: 0.05,
                    acceptance_rate: 0.82,
                    wall_time_s: 2.0,
                },
            ],
            total_wall_time_s: 2.0,
        };
        print_scan_summary(&result);
    }
}
