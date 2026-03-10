// SPDX-License-Identifier: AGPL-3.0-only

//! Full physics pipeline end-to-end profiler.
//!
//! Runs each physics pipeline (MD, HMC, BCS, dielectric) on a single GPU
//! at each applicable precision tier, measuring wall time and domain-specific
//! accuracy metrics (energy drift, plaquette stability, root accuracy, f-sum).

use crate::error::HotSpringError;
use crate::gpu::GpuF64;
use crate::precision_routing::PrecisionTier;
use std::time::Instant;

/// Result of running a single physics pipeline at one precision tier.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Pipeline label (e.g. "HMC [4,4,4,4]", "BCS 4096x20").
    pub pipeline_name: String,
    /// GPU adapter name.
    pub adapter_name: String,
    /// Precision tier used.
    pub tier: PrecisionTier,
    /// Total wall-clock time in milliseconds.
    pub wall_ms: f64,
    /// Name of the accuracy metric (e.g. "plaq_stability", "convergence").
    pub accuracy_label: String,
    /// Numeric accuracy metric value.
    pub accuracy_value: f64,
    /// Human-readable notes.
    pub notes: String,
}

/// Full pipeline profiler for a single GPU.
pub struct PipelineEval<'a> {
    gpu: &'a GpuF64,
}

impl<'a> PipelineEval<'a> {
    /// Create a pipeline evaluator for the given GPU.
    #[must_use]
    pub fn new(gpu: &'a GpuF64) -> Self {
        Self { gpu }
    }

    /// Run all physics pipelines across all applicable tiers.
    pub fn run_all(&self) -> Vec<PipelineResult> {
        let mut results = Vec::new();
        results.extend(self.eval_lattice_hmc());
        results.extend(self.eval_bcs());
        results.extend(self.eval_dielectric());
        results
    }

    /// Lattice HMC: 4^4 lattice, 10 trajectories.
    fn eval_lattice_hmc(&self) -> Vec<PipelineResult> {
        let dims = [4, 4, 4, 4];
        let beta = 6.0;
        let n_traj = 10;
        let n_md_steps = 20;
        let dt = 0.05;

        let tiers = [PrecisionTier::F64, PrecisionTier::DF64];
        let adapter = self.gpu.adapter_name.clone();

        tiers
            .iter()
            .map(|&tier| {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.run_hmc(dims, beta, n_traj, n_md_steps, dt)
                }));
                match result {
                    Ok((wall_ms, plaq_mean, plaq_std)) => PipelineResult {
                        pipeline_name: format!("HMC {dims:?}"),
                        adapter_name: adapter.clone(),
                        tier,
                        wall_ms,
                        accuracy_label: "plaq_stability".into(),
                        accuracy_value: plaq_std,
                        notes: format!("mean_plaq={plaq_mean:.6}, std={plaq_std:.2e}"),
                    },
                    Err(_) => PipelineResult {
                        pipeline_name: format!("HMC {dims:?}"),
                        adapter_name: adapter.clone(),
                        tier,
                        wall_ms: 0.0,
                        accuracy_label: "panic".into(),
                        accuracy_value: f64::NAN,
                        notes: "Pipeline panicked".into(),
                    },
                }
            })
            .collect()
    }

    fn run_hmc(
        &self,
        dims: [usize; 4],
        beta: f64,
        n_traj: usize,
        n_md_steps: usize,
        dt: f64,
    ) -> (f64, f64, f64) {
        use crate::lattice::gpu_hmc::{gpu_hmc_trajectory, GpuHmcPipelines, GpuHmcState};
        use crate::lattice::wilson::Lattice;

        let lattice = Lattice::hot_start(dims, beta, 42);
        let pipelines = GpuHmcPipelines::new(self.gpu);
        let state = GpuHmcState::from_lattice(self.gpu, &lattice, beta);
        let mut seed = 42_u64;

        let t0 = Instant::now();
        let mut plaqs = Vec::with_capacity(n_traj);
        for _ in 0..n_traj {
            let r = gpu_hmc_trajectory(self.gpu, &pipelines, &state, n_md_steps, dt, &mut seed);
            plaqs.push(r.plaquette);
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;

        let mean = plaqs.iter().sum::<f64>() / plaqs.len() as f64;
        let var = plaqs.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / plaqs.len() as f64;
        let std_dev = var.sqrt();

        (wall_ms, mean, std_dev)
    }

    /// BCS bisection: 4096 batch, 20 levels.
    fn eval_bcs(&self) -> Vec<PipelineResult> {
        let batch_size = 4096;
        let n_levels = 20;
        let adapter = self.gpu.adapter_name.clone();

        let tiers = [PrecisionTier::F64, PrecisionTier::DF64];

        tiers
            .iter()
            .map(|&tier| {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.run_bcs(batch_size, n_levels)
                }));
                match result {
                    Ok(Ok((wall_ms, mean_iters, max_iters))) => PipelineResult {
                        pipeline_name: format!("BCS {batch_size}x{n_levels}"),
                        adapter_name: adapter.clone(),
                        tier,
                        wall_ms,
                        accuracy_label: "convergence".into(),
                        accuracy_value: mean_iters,
                        notes: format!("mean_iter={mean_iters:.1}, max_iter={max_iters}"),
                    },
                    Ok(Err(e)) => PipelineResult {
                        pipeline_name: format!("BCS {batch_size}x{n_levels}"),
                        adapter_name: adapter.clone(),
                        tier,
                        wall_ms: 0.0,
                        accuracy_label: "error".into(),
                        accuracy_value: f64::NAN,
                        notes: format!("Error: {e}"),
                    },
                    Err(_) => PipelineResult {
                        pipeline_name: format!("BCS {batch_size}x{n_levels}"),
                        adapter_name: adapter.clone(),
                        tier,
                        wall_ms: 0.0,
                        accuracy_label: "panic".into(),
                        accuracy_value: f64::NAN,
                        notes: "Pipeline panicked".into(),
                    },
                }
            })
            .collect()
    }

    fn run_bcs(
        &self,
        batch_size: usize,
        n_levels: usize,
    ) -> Result<(f64, f64, u32), HotSpringError> {
        use crate::physics::bcs_gpu::BcsBisectionGpu;

        let solver = BcsBisectionGpu::new(self.gpu, 100, 1e-12);

        let lower: Vec<f64> = vec![-10.0; batch_size];
        let upper: Vec<f64> = vec![10.0; batch_size];
        let eigenvalues: Vec<f64> = (0..batch_size * n_levels)
            .map(|i| {
                let level = i % n_levels;
                let batch = i / n_levels;
                (level as f64 + 1.0) * 0.5 + (batch as f64) * 0.001
            })
            .collect();
        let delta: Vec<f64> = vec![0.5; batch_size];
        let target_n: Vec<f64> = vec![n_levels as f64; batch_size];

        let t0 = Instant::now();
        let result = solver
            .solve_bcs(&lower, &upper, &eigenvalues, &delta, &target_n)
            .map_err(HotSpringError::DeviceCreation)?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;

        let mean_iters = result.iterations.iter().map(|&i| f64::from(i)).sum::<f64>()
            / result.iterations.len() as f64;
        let max_iters = result.iterations.iter().copied().max().unwrap_or(0);

        Ok((wall_ms, mean_iters, max_iters))
    }

    /// Dielectric: Mermin/completed Mermin f-sum rule validation.
    fn eval_dielectric(&self) -> Vec<PipelineResult> {
        let adapter = self.gpu.adapter_name.clone();

        let tiers = [PrecisionTier::F64, PrecisionTier::F64Precise];

        tiers
            .iter()
            .map(|&tier| {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_dielectric()
                }));
                match result {
                    Ok((wall_ms, f_sum_error)) => PipelineResult {
                        pipeline_name: "Dielectric (Mermin)".into(),
                        adapter_name: adapter.clone(),
                        tier,
                        wall_ms,
                        accuracy_label: "f_sum_violation".into(),
                        accuracy_value: f_sum_error,
                        notes: format!("f_sum_rel_err={f_sum_error:.2e}"),
                    },
                    Err(_) => PipelineResult {
                        pipeline_name: "Dielectric (Mermin)".into(),
                        adapter_name: adapter.clone(),
                        tier,
                        wall_ms: 0.0,
                        accuracy_label: "panic".into(),
                        accuracy_value: f64::NAN,
                        notes: "Pipeline panicked".into(),
                    },
                }
            })
            .collect()
    }
}

fn run_dielectric() -> (f64, f64) {
    use crate::physics::dielectric::response::validate_dielectric;

    let t0 = Instant::now();
    let v = validate_dielectric(158.0, 2.0);
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;

    (wall_ms, v.f_sum_error_completed)
}

impl PipelineResult {
    /// Format a single result row for report output.
    #[must_use]
    pub fn report_line(&self) -> String {
        format!(
            "  {:<16} {:<16} {:<10} {:>8.1}ms  {}",
            self.pipeline_name,
            self.adapter_name,
            format!("{:?}", self.tier),
            self.wall_ms,
            self.notes
        )
    }
}
