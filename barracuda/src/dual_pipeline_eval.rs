// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dual-card cooperative profiler.
//!
//! Tests actual cooperative dispatch patterns using the `DevicePair`:
//! 1. **Split BCS** — 70% batch on throughput (DF64), 30% on precise (F64)
//! 2. **Split HMC** — throughput computes forces, precise validates plaquettes
//! 3. **Redundant HMC** — same lattice on both, compare plaquette to 15 digits
//! 4. **PCIe pipeline** — upload→Titan→readback→3090→readback, full transfer cost

use crate::device_pair::DevicePair;
use std::time::Instant;

/// Result of a cooperative dual-card evaluation pattern.
#[derive(Debug, Clone)]
pub struct DualPatternResult {
    /// Pattern label (e.g. "Split BCS (30/70)").
    pub pattern_name: String,
    /// Total wall-clock time in milliseconds.
    pub wall_ms: f64,
    /// Speedup vs best single-card baseline.
    pub speedup_vs_single: f64,
    /// Name of the accuracy metric.
    pub accuracy_label: String,
    /// Numeric accuracy metric value.
    pub accuracy_value: f64,
    /// Human-readable notes.
    pub notes: String,
}

/// Dual-card cooperative profiler.
pub struct DualPipelineEval<'a> {
    pair: &'a DevicePair,
}

impl<'a> DualPipelineEval<'a> {
    /// Create a dual-pipeline evaluator for the given device pair.
    #[must_use]
    pub fn new(pair: &'a DevicePair) -> Self {
        Self { pair }
    }

    /// Run all cooperative patterns.
    pub fn run_all(&self) -> Vec<DualPatternResult> {
        let mut results = Vec::new();
        results.push(self.eval_split_bcs());
        results.extend(self.eval_split_hmc());
        results.push(self.eval_redundant_hmc());
        results.push(self.eval_pcie_pipeline());
        results
    }

    /// Split BCS: 70% batch on throughput, 30% on precise.
    fn eval_split_bcs(&self) -> DualPatternResult {
        use crate::physics::bcs_gpu::BcsBisectionGpu;

        let batch_size = 4096;
        let n_levels = 20;
        let precise_frac = 0.30;
        let precise_batch = (batch_size as f64 * precise_frac) as usize;
        let throughput_batch = batch_size - precise_batch;

        let make_inputs = |bs: usize, offset: usize| {
            let lower: Vec<f64> = vec![-10.0; bs];
            let upper: Vec<f64> = vec![10.0; bs];
            let eigenvalues: Vec<f64> = (0..bs * n_levels)
                .map(|i| {
                    let level = i % n_levels;
                    let batch = i / n_levels + offset;
                    (level as f64 + 1.0) * 0.5 + (batch as f64) * 0.001
                })
                .collect();
            let delta: Vec<f64> = vec![0.5; bs];
            let target_n: Vec<f64> = vec![n_levels as f64; bs];
            (lower, upper, eigenvalues, delta, target_n)
        };

        // Single-card reference (all on throughput)
        let solver_t = BcsBisectionGpu::new(&self.pair.throughput, 100, 1e-12);
        let (l_all, u_all, e_all, d_all, tn_all) = make_inputs(batch_size, 0);
        let t_single = Instant::now();
        let _ = solver_t.solve_bcs(&l_all, &u_all, &e_all, &d_all, &tn_all);
        let single_ms = t_single.elapsed().as_secs_f64() * 1e3;

        // Split dispatch: precise gets first chunk, throughput gets rest
        let solver_p = BcsBisectionGpu::new(&self.pair.precise, 100, 1e-12);
        let (l_p, u_p, e_p, d_p, tn_p) = make_inputs(precise_batch, 0);
        let (l_t, u_t, e_t, d_t, tn_t) = make_inputs(throughput_batch, precise_batch);

        let t_split = Instant::now();
        let r_p = solver_p.solve_bcs(&l_p, &u_p, &e_p, &d_p, &tn_p);
        let r_t = solver_t.solve_bcs(&l_t, &u_t, &e_t, &d_t, &tn_t);
        let split_ms = t_split.elapsed().as_secs_f64() * 1e3;

        let (accuracy, notes) = match (&r_p, &r_t) {
            (Ok(rp), Ok(rt)) => {
                let total_roots = rp.roots.len() + rt.roots.len();
                (
                    0.0,
                    format!(
                        "precise={precise_batch}, throughput={throughput_batch}, total_roots={total_roots}"
                    ),
                )
            }
            _ => (f64::NAN, "one or both solves failed".into()),
        };

        let speedup = if split_ms > 0.0 {
            single_ms / split_ms
        } else {
            0.0
        };

        DualPatternResult {
            pattern_name: format!(
                "Split BCS ({:.0}/{:.0})",
                precise_frac * 100.0,
                (1.0 - precise_frac) * 100.0
            ),
            wall_ms: split_ms,
            speedup_vs_single: speedup,
            accuracy_label: "convergence".into(),
            accuracy_value: accuracy,
            notes,
        }
    }

    /// Split HMC: throughput computes trajectories, precise validates.
    fn eval_split_hmc(&self) -> Vec<DualPatternResult> {
        use crate::lattice::gpu_hmc::{
            GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
        };
        use crate::lattice::wilson::Lattice;

        let dims = [4, 4, 4, 4];
        let beta = 6.0;
        let n_traj = 5;
        let n_md = 20;
        let dt = 0.05;

        let lattice = Lattice::hot_start(dims, beta, 42);

        // Single-card reference (throughput)
        let p_t = GpuHmcStreamingPipelines::new(&self.pair.throughput);
        let s_t = GpuHmcState::from_lattice(&self.pair.throughput, &lattice, beta);
        let mut seed_t = 42_u64;

        let t_single = Instant::now();
        for i in 0..n_traj {
            gpu_hmc_trajectory_streaming(
                &self.pair.throughput,
                &p_t,
                &s_t,
                n_md,
                dt,
                i as u32,
                &mut seed_t,
            )
            .expect("streaming HMC trajectory");
        }
        let single_ms = t_single.elapsed().as_secs_f64() * 1e3;

        // Dual-card: throughput computes, precise validates first trajectory
        let p_p = GpuHmcStreamingPipelines::new(&self.pair.precise);
        let s_p = GpuHmcState::from_lattice(&self.pair.precise, &lattice, beta);
        let mut seed_p = 42_u64;
        let mut seed_t2 = 42_u64;

        let t_split = Instant::now();
        let r_precise = gpu_hmc_trajectory_streaming(
            &self.pair.precise,
            &p_p,
            &s_p,
            n_md,
            dt,
            0,
            &mut seed_p,
        )
        .expect("streaming HMC trajectory");
        let r_throughput = gpu_hmc_trajectory_streaming(
            &self.pair.throughput,
            &p_t,
            &s_t,
            n_md,
            dt,
            0,
            &mut seed_t2,
        )
        .expect("streaming HMC trajectory");
        let split_ms = t_split.elapsed().as_secs_f64() * 1e3;

        let plaq_diff = (r_precise.plaquette - r_throughput.plaquette).abs();
        let speedup = if split_ms > 0.0 {
            single_ms / split_ms
        } else {
            0.0
        };

        vec![DualPatternResult {
            pattern_name: "Split HMC (force/valid)".into(),
            wall_ms: split_ms,
            speedup_vs_single: speedup,
            accuracy_label: "plaq_diff".into(),
            accuracy_value: plaq_diff,
            notes: format!(
                "precise_plaq={:.8}, throughput_plaq={:.8}, diff={plaq_diff:.2e}",
                r_precise.plaquette, r_throughput.plaquette
            ),
        }]
    }

    /// Redundant HMC: same lattice on both cards, compare plaquette.
    fn eval_redundant_hmc(&self) -> DualPatternResult {
        use crate::lattice::gpu_hmc::{
            GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
        };
        use crate::lattice::wilson::Lattice;

        let dims = [4, 4, 4, 4];
        let beta = 6.0;
        let n_traj = 5;
        let n_md = 20;
        let dt = 0.05;

        let lattice = Lattice::hot_start(dims, beta, 42);

        let pipelines_p = GpuHmcStreamingPipelines::new(&self.pair.precise);
        let pipelines_t = GpuHmcStreamingPipelines::new(&self.pair.throughput);
        let state_p = GpuHmcState::from_lattice(&self.pair.precise, &lattice, beta);
        let state_t = GpuHmcState::from_lattice(&self.pair.throughput, &lattice, beta);
        let mut seed_p = 42_u64;
        let mut seed_t = 42_u64;

        let t0 = Instant::now();
        let mut max_diff = 0.0_f64;
        for i in 0..n_traj {
            let rp = gpu_hmc_trajectory_streaming(
                &self.pair.precise,
                &pipelines_p,
                &state_p,
                n_md,
                dt,
                i as u32,
                &mut seed_p,
            )
            .expect("streaming HMC trajectory");
            let rt = gpu_hmc_trajectory_streaming(
                &self.pair.throughput,
                &pipelines_t,
                &state_t,
                n_md,
                dt,
                i as u32,
                &mut seed_t,
            )
            .expect("streaming HMC trajectory");
            let diff = (rp.plaquette - rt.plaquette).abs();
            max_diff = max_diff.max(diff);
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;

        DualPatternResult {
            pattern_name: "Redundant HMC".into(),
            wall_ms,
            speedup_vs_single: 1.0,
            accuracy_label: "max_plaq_divergence".into(),
            accuracy_value: max_diff,
            notes: format!("max_plaq_diff={max_diff:.2e} over {n_traj} trajectories"),
        }
    }

    /// PCIe pipeline: full transfer round-trip with GPU work in the middle.
    fn eval_pcie_pipeline(&self) -> DualPatternResult {
        let n = 65536_usize;
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();

        let t0 = Instant::now();

        // Upload to precise card
        let buf_p = self.pair.precise.create_f64_buffer(&data, "pcie_eval_in");
        let _ = self.pair.precise.device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Read back from precise
        let readback_p = self
            .pair
            .precise
            .read_back_f64(&buf_p, n)
            .unwrap_or_default();

        // Upload to throughput card
        let buf_t = self
            .pair
            .throughput
            .create_f64_buffer(&readback_p, "pcie_eval_relay");
        let _ = self.pair.throughput.device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Read back from throughput
        let readback_t = self
            .pair
            .throughput
            .read_back_f64(&buf_t, n)
            .unwrap_or_default();

        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;

        let max_err = data
            .iter()
            .zip(readback_t.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        let data_bytes = n * 8;
        let effective_gbps = if wall_ms > 0.0 {
            (data_bytes as f64 * 4.0) / (wall_ms * 1e-3) / 1e9
        } else {
            0.0
        };

        DualPatternResult {
            pattern_name: "PCIe roundtrip".into(),
            wall_ms,
            speedup_vs_single: 0.0,
            accuracy_label: "max_transfer_err".into(),
            accuracy_value: max_err,
            notes: format!(
                "{} each way, effective={effective_gbps:.1}GB/s, max_err={max_err:.2e}",
                format_bytes(data_bytes)
            ),
        }
    }
}

impl DualPatternResult {
    /// Format a single result row.
    #[must_use]
    pub fn report_line(&self) -> String {
        format!(
            "  {:<28} {:>8.1}ms  {:<14} {}",
            self.pattern_name,
            self.wall_ms,
            if self.speedup_vs_single > 0.0 {
                format!("{:.1}x", self.speedup_vs_single)
            } else {
                "—".into()
            },
            self.notes
        )
    }
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{}MB", bytes / 1_048_576)
    } else {
        format!("{}KB", bytes / 1_024)
    }
}
