// SPDX-License-Identifier: AGPL-3.0-or-later

//! Memoized physics process registry with cost model.
//!
//! Every computation in the Chuna engine (generation, flow, Wilson loops,
//! chiral condensate, etc.) is a `PhysicsProcess` with known dependencies,
//! estimated cost, and cached results. This module defines the process DAG
//! and provides precomputed cost estimates calibrated from benchmark data.
//!
//! # Cost model calibration
//!
//! Estimates are derived from actual hotSpring benchmarks on consumer hardware:
//! - HMC trajectory: ~4 ms/site at 8^4 CPU, scaling ∝ V^1.3
//! - Gradient flow (W7, t_max=4, eps=0.01): ~0.14s at 8^4 GPU, ~5.6s CPU
//! - CG solve: ~V × n_iter × 18 flops, n_iter ∈ [200, 2000] depending on mass
//! - Wilson loops: O(V × R_max²) per config

use serde::{Deserialize, Serialize};

/// A physics computation that can be scheduled and memoized.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhysicsProcess {
    /// HMC/RHMC gauge configuration generation.
    Generate,
    /// Wilson gradient flow (t0, w0 scale setting).
    GradientFlow,
    /// Wilson loop grid W(R,T).
    WilsonLoops,
    /// Chiral condensate via stochastic estimator.
    ChiralCondensate,
    /// Topological charge Q at finite flow time.
    TopologicalCharge,
    /// Hadronic correlators (point-to-all propagators).
    Correlators,
    /// Scale setting: convert t0/w0 to physical units.
    ScaleSetting,
}

impl PhysicsProcess {
    /// Direct dependencies that must complete before this process.
    pub fn dependencies(&self) -> &'static [PhysicsProcess] {
        match self {
            PhysicsProcess::Generate => &[],
            PhysicsProcess::GradientFlow
            | PhysicsProcess::WilsonLoops
            | PhysicsProcess::ChiralCondensate
            | PhysicsProcess::Correlators => &[PhysicsProcess::Generate],
            PhysicsProcess::TopologicalCharge | PhysicsProcess::ScaleSetting => {
                &[PhysicsProcess::GradientFlow]
            }
        }
    }

    /// All known process types.
    pub fn all() -> &'static [PhysicsProcess] {
        &[
            PhysicsProcess::Generate,
            PhysicsProcess::GradientFlow,
            PhysicsProcess::WilsonLoops,
            PhysicsProcess::ChiralCondensate,
            PhysicsProcess::TopologicalCharge,
            PhysicsProcess::Correlators,
            PhysicsProcess::ScaleSetting,
        ]
    }

    /// Short display name.
    pub fn name(&self) -> &'static str {
        match self {
            PhysicsProcess::Generate => "Generate",
            PhysicsProcess::GradientFlow => "GradientFlow",
            PhysicsProcess::WilsonLoops => "WilsonLoops",
            PhysicsProcess::ChiralCondensate => "ChiralCondensate",
            PhysicsProcess::TopologicalCharge => "TopologicalCharge",
            PhysicsProcess::Correlators => "Correlators",
            PhysicsProcess::ScaleSetting => "ScaleSetting",
        }
    }
}

/// Hardware tier for cost estimation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HardwareTier {
    /// Single CPU core.
    Cpu,
    /// Consumer GPU (RTX 3090 / 4070 class).
    ConsumerGpu,
    /// HPC GPU (A100 / H100 class).
    HpcGpu,
    /// Multi-GPU cluster.
    Cluster,
}

/// Specification for a computation task.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessSpec {
    /// Which physics process.
    pub process: PhysicsProcess,
    /// Lattice dimensions [Nx, Ny, Nz, Nt].
    pub dims: [usize; 4],
    /// Inverse bare coupling.
    pub beta: f64,
    /// Quark mass (0.0 for quenched).
    pub mass: f64,
    /// Number of dynamical flavors.
    pub nf: usize,
    /// Number of configurations to process.
    pub n_configs: usize,
    /// Process-specific parameters.
    #[serde(default)]
    pub params: ProcessParams,
}

/// Process-specific parameters (optional extras beyond the common set).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProcessParams {
    /// For Generate: number of thermalization trajectories.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_therm: Option<usize>,
    /// For Generate: measurement interval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meas_interval: Option<usize>,
    /// For GradientFlow: max flow time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub t_max: Option<f64>,
    /// For GradientFlow: step size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epsilon: Option<f64>,
    /// For WilsonLoops: max spatial extent R.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r_max: Option<usize>,
    /// For ChiralCondensate: number of stochastic sources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_sources: Option<usize>,
    /// For ChiralCondensate/Correlators: CG tolerance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cg_tol: Option<f64>,
    /// Random seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Result of a completed computation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessResult {
    /// Execution status.
    pub status: ProcessStatus,
    /// Output file paths.
    #[serde(default)]
    pub output_paths: Vec<String>,
    /// CRC checksums of output files.
    #[serde(default)]
    pub checksums: Vec<String>,
    /// Wall-clock time in seconds.
    pub wall_seconds: f64,
    /// ISO 8601 completion timestamp.
    pub completed_at: String,
}

/// Status of a computation in the registry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

impl std::fmt::Display for ProcessStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessStatus::Pending => write!(f, "pending"),
            ProcessStatus::Running => write!(f, "running"),
            ProcessStatus::Completed => write!(f, "completed"),
            ProcessStatus::Failed => write!(f, "failed"),
            ProcessStatus::Skipped => write!(f, "skipped"),
        }
    }
}

/// Cost model for estimating computation wall time.
///
/// Calibrated from hotSpring benchmark data on consumer hardware (RTX 3090,
/// Ryzen 9 5950X). Scaling laws are empirical fits.
pub struct CostModel;

impl CostModel {
    /// Estimate wall-clock seconds for a process on given hardware.
    pub fn estimate_wall_seconds(spec: &ProcessSpec, hardware: &HardwareTier) -> f64 {
        let volume = spec.dims.iter().product::<usize>() as f64;
        let n = spec.n_configs as f64;
        let hw_factor = Self::hardware_factor(hardware);

        match spec.process {
            PhysicsProcess::Generate => {
                // HMC trajectory: ~4ms/site at 8^4 CPU, V^1.3 scaling
                let n_therm = spec.params.n_therm.unwrap_or(200) as f64;
                let meas_interval = spec.params.meas_interval.unwrap_or(10) as f64;
                let total_traj = n_therm + n * meas_interval;
                let per_traj = 0.004 * volume.powf(1.3) / 4096.0_f64.powf(1.3);
                per_traj * total_traj / hw_factor
            }
            PhysicsProcess::GradientFlow => {
                // Gradient flow: ~5.6s per config at 8^4 CPU, linear in V
                let t_max = spec.params.t_max.unwrap_or(4.0);
                let eps = spec.params.epsilon.unwrap_or(0.01);
                let n_steps = t_max / eps;
                let base_per_step = 5.6 / 400.0; // 400 steps for t_max=4, eps=0.01
                base_per_step * n_steps * (volume / 4096.0) * n / hw_factor
            }
            PhysicsProcess::WilsonLoops => {
                // O(V * R_max^2) per config
                let r_max = spec.params.r_max.unwrap_or(4) as f64;
                let per_config = 0.001 * volume * r_max * r_max / 4096.0;
                per_config * n / hw_factor
            }
            PhysicsProcess::ChiralCondensate => {
                // CG solve dominated: V * n_iter * 18 flops * n_sources
                let n_sources = spec.params.n_sources.unwrap_or(10) as f64;
                let n_iter = Self::estimate_cg_iters(spec.mass) as f64;
                let per_solve = volume * n_iter * 18.0 / 1e9; // seconds (rough)
                per_solve * n_sources * n / hw_factor
            }
            PhysicsProcess::TopologicalCharge => {
                // Cheap measurement on already-flowed config
                0.01 * (volume / 4096.0) * n / hw_factor
            }
            PhysicsProcess::Correlators => {
                let n_iter = Self::estimate_cg_iters(spec.mass) as f64;
                let per_config = volume * n_iter * 18.0 / 1e9 * 4.0; // 4 propagators
                per_config * n / hw_factor
            }
            PhysicsProcess::ScaleSetting => {
                // Negligible: just arithmetic on flow results
                0.01 * n
            }
        }
    }

    /// Format a duration in human-readable form.
    pub fn format_duration(seconds: f64) -> String {
        if seconds < 60.0 {
            format!("{seconds:.1}s")
        } else if seconds < 3600.0 {
            format!("{:.1}min", seconds / 60.0)
        } else if seconds < 86400.0 {
            format!("{:.1}h", seconds / 3600.0)
        } else {
            format!("{:.1}d", seconds / 86400.0)
        }
    }

    fn hardware_factor(hw: &HardwareTier) -> f64 {
        match hw {
            HardwareTier::Cpu => 1.0,
            HardwareTier::ConsumerGpu => 40.0,
            HardwareTier::HpcGpu => 200.0,
            HardwareTier::Cluster => 1000.0,
        }
    }

    fn estimate_cg_iters(mass: f64) -> usize {
        // CG iterations scale roughly as 1/mass for light quarks
        if mass <= 0.0 || mass > 1.0 {
            200
        } else if mass < 0.01 {
            2000
        } else if mass < 0.05 {
            1000
        } else if mass < 0.1 {
            500
        } else {
            200
        }
    }
}

/// The full process catalog: all known process types with their dependency graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessCatalog {
    /// Registered process specifications with their results.
    pub entries: Vec<CatalogEntry>,
}

/// A single entry in the process catalog.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// Unique key: (process, dims, beta, mass, nf).
    pub spec: ProcessSpec,
    /// Cached result (if completed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<ProcessResult>,
}

impl ProcessCatalog {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a new process spec. Returns the index.
    pub fn register(&mut self, spec: ProcessSpec) -> usize {
        let idx = self.entries.len();
        self.entries.push(CatalogEntry { spec, result: None });
        idx
    }

    /// Find an entry matching the given process type and lattice parameters.
    pub fn find(
        &self,
        process: &PhysicsProcess,
        dims: [usize; 4],
        beta: f64,
    ) -> Option<&CatalogEntry> {
        self.entries.iter().find(|e| {
            &e.spec.process == process && e.spec.dims == dims && (e.spec.beta - beta).abs() < 1e-10
        })
    }

    /// Check if all dependencies for a process spec are completed.
    pub fn dependencies_met(&self, spec: &ProcessSpec) -> bool {
        spec.process.dependencies().iter().all(|dep| {
            self.find(dep, spec.dims, spec.beta)
                .and_then(|e| e.result.as_ref())
                .is_some_and(|r| r.status == ProcessStatus::Completed)
        })
    }

    /// Record a result for the given index.
    pub fn complete(&mut self, idx: usize, result: ProcessResult) {
        if let Some(entry) = self.entries.get_mut(idx) {
            entry.result = Some(result);
        }
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for ProcessCatalog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn process_dependencies() {
        assert!(PhysicsProcess::Generate.dependencies().is_empty());
        assert_eq!(
            PhysicsProcess::GradientFlow.dependencies(),
            &[PhysicsProcess::Generate]
        );
        assert_eq!(
            PhysicsProcess::TopologicalCharge.dependencies(),
            &[PhysicsProcess::GradientFlow]
        );
    }

    #[test]
    fn cost_model_scaling() {
        let small = ProcessSpec {
            process: PhysicsProcess::Generate,
            dims: [8, 8, 8, 8],
            beta: 6.0,
            mass: 0.0,
            nf: 0,
            n_configs: 100,
            params: ProcessParams {
                n_therm: Some(200),
                meas_interval: Some(10),
                ..Default::default()
            },
        };
        let large = ProcessSpec {
            process: PhysicsProcess::Generate,
            dims: [16, 16, 16, 16],
            beta: 6.0,
            mass: 0.0,
            nf: 0,
            n_configs: 100,
            params: ProcessParams {
                n_therm: Some(200),
                meas_interval: Some(10),
                ..Default::default()
            },
        };

        let t_small = CostModel::estimate_wall_seconds(&small, &HardwareTier::Cpu);
        let t_large = CostModel::estimate_wall_seconds(&large, &HardwareTier::Cpu);

        assert!(t_large > t_small, "larger lattice should cost more");
        assert!(t_small > 0.0, "cost should be positive");
    }

    #[test]
    fn cost_model_gpu_faster() {
        let spec = ProcessSpec {
            process: PhysicsProcess::GradientFlow,
            dims: [8, 8, 8, 8],
            beta: 6.0,
            mass: 0.0,
            nf: 0,
            n_configs: 10,
            params: Default::default(),
        };

        let cpu = CostModel::estimate_wall_seconds(&spec, &HardwareTier::Cpu);
        let gpu = CostModel::estimate_wall_seconds(&spec, &HardwareTier::ConsumerGpu);

        assert!(gpu < cpu, "GPU should be faster: gpu={gpu}, cpu={cpu}");
    }

    #[test]
    fn catalog_dependency_check() {
        let mut catalog = ProcessCatalog::new();

        let gen_spec = ProcessSpec {
            process: PhysicsProcess::Generate,
            dims: [8, 8, 8, 8],
            beta: 6.0,
            mass: 0.0,
            nf: 0,
            n_configs: 10,
            params: Default::default(),
        };
        let flow_spec = ProcessSpec {
            process: PhysicsProcess::GradientFlow,
            dims: [8, 8, 8, 8],
            beta: 6.0,
            mass: 0.0,
            nf: 0,
            n_configs: 10,
            params: Default::default(),
        };

        let gen_idx = catalog.register(gen_spec);
        catalog.register(flow_spec.clone());

        // Flow depends on Generate; Generate not yet completed
        assert!(!catalog.dependencies_met(&flow_spec));

        // Complete generation
        catalog.complete(
            gen_idx,
            ProcessResult {
                status: ProcessStatus::Completed,
                output_paths: vec![],
                checksums: vec![],
                wall_seconds: 1.0,
                completed_at: "2026-03-31T00:00:00Z".to_string(),
            },
        );

        // Now flow dependencies are met
        assert!(catalog.dependencies_met(&flow_spec));
    }

    #[test]
    fn catalog_json_roundtrip() {
        let mut catalog = ProcessCatalog::new();
        catalog.register(ProcessSpec {
            process: PhysicsProcess::Generate,
            dims: [4, 4, 4, 4],
            beta: 5.8,
            mass: 0.0,
            nf: 0,
            n_configs: 5,
            params: Default::default(),
        });

        let json = catalog.to_json();
        let parsed = ProcessCatalog::from_json(&json).unwrap();
        assert_eq!(parsed.entries.len(), 1);
        assert_eq!(parsed.entries[0].spec.process, PhysicsProcess::Generate);
    }

    #[test]
    fn format_duration_ranges() {
        assert_eq!(CostModel::format_duration(30.0), "30.0s");
        assert_eq!(CostModel::format_duration(120.0), "2.0min");
        assert_eq!(CostModel::format_duration(7200.0), "2.0h");
        assert_eq!(CostModel::format_duration(172800.0), "2.0d");
    }
}
