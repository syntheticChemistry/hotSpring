// SPDX-License-Identifier: AGPL-3.0-or-later

//! Persistent task matrix engine for scheduled and on-demand computation.
//!
//! The `TaskMatrix` is a grid of `(ensemble_params, observable_set)` cells.
//! It supports two modes:
//!
//! 1. **Sweep mode**: generate a full grid from parameter ranges
//!    (beta values × lattice sizes × observable types)
//! 2. **On-demand mode**: add specific targeted requests
//!    (e.g., "measure chiral condensate at beta=6.0, 16^4, mass=0.1")
//!
//! Each cell tracks status, cost estimates, and output paths. The matrix
//! persists to JSON and can be resumed across sessions.

use super::measurement::format_dims_id;
use super::process_catalog::{
    CostModel, HardwareTier, PhysicsProcess, ProcessParams, ProcessSpec, ProcessStatus,
};
use serde::{Deserialize, Serialize};
use std::fmt::Write;

/// Priority level for task scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    /// Background sweep tasks.
    Sweep = 0,
    /// Normal priority.
    Normal = 1,
    /// On-demand targeted requests (processed first).
    OnDemand = 2,
    /// Urgent requests from collaborators.
    Urgent = 3,
}

/// A single cell in the task matrix.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskCell {
    /// Unique task identifier.
    pub id: String,
    /// The computation to perform.
    pub spec: ProcessSpec,
    /// Current status.
    pub status: ProcessStatus,
    /// Priority level.
    pub priority: TaskPriority,
    /// Estimated wall time in seconds.
    pub estimated_seconds: f64,
    /// Actual wall time (filled on completion).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_seconds: Option<f64>,
    /// Output directory or file path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_path: Option<String>,
    /// Error message if failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// ISO 8601 timestamp when task was created.
    pub created_at: String,
    /// ISO 8601 timestamp when task completed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
}

/// The persistent task matrix.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskMatrix {
    /// Schema version.
    pub schema_version: String,
    /// Matrix identifier.
    pub matrix_id: String,
    /// Hardware tier used for cost estimates.
    pub hardware: HardwareTier,
    /// All tasks in the matrix.
    pub tasks: Vec<TaskCell>,
    /// ISO 8601 timestamp of last modification.
    pub last_modified: String,
}

/// Parameters for generating a sweep.
pub struct SweepParams {
    /// Beta values to sweep.
    pub betas: Vec<f64>,
    /// Spatial lattice sizes (isotropic N_s, with N_t = N_s by default).
    pub lattice_sizes: Vec<usize>,
    /// Temporal extent override (if different from spatial).
    pub nt_override: Option<Vec<usize>>,
    /// Observables to compute at each point.
    pub observables: Vec<PhysicsProcess>,
    /// Number of configs per ensemble point.
    pub n_configs: usize,
    /// Quark mass (0.0 for quenched).
    pub mass: f64,
    /// Number of dynamical flavors.
    pub nf: usize,
    /// Process-specific defaults.
    pub default_params: ProcessParams,
}

impl TaskMatrix {
    /// Create an empty task matrix.
    pub fn new(matrix_id: &str, hardware: HardwareTier) -> Self {
        Self {
            schema_version: "1.0".to_string(),
            matrix_id: matrix_id.to_string(),
            hardware,
            tasks: Vec::new(),
            last_modified: super::measurement::iso8601_now(),
        }
    }

    /// Generate a full sweep grid.
    pub fn from_sweep(matrix_id: &str, hardware: HardwareTier, params: &SweepParams) -> Self {
        let mut matrix = Self::new(matrix_id, hardware);

        for &beta in &params.betas {
            for (i, &ns) in params.lattice_sizes.iter().enumerate() {
                let nt = params
                    .nt_override
                    .as_ref()
                    .and_then(|v| v.get(i).copied())
                    .unwrap_or(ns);
                let dims = [ns, ns, ns, nt];

                // Always include Generate first
                let dims_tag = format_dims_id(dims);
                let gen_id = format!("gen_b{beta:.2}_{dims_tag}");
                let gen_spec = ProcessSpec {
                    process: PhysicsProcess::Generate,
                    dims,
                    beta,
                    mass: params.mass,
                    nf: params.nf,
                    n_configs: params.n_configs,
                    params: params.default_params.clone(),
                };
                matrix.add_task_internal(gen_id, gen_spec, TaskPriority::Sweep);

                for obs in &params.observables {
                    if *obs == PhysicsProcess::Generate {
                        continue;
                    }
                    let task_id = format!("{}_b{beta:.2}_{dims_tag}", obs.name().to_lowercase());
                    let spec = ProcessSpec {
                        process: obs.clone(),
                        dims,
                        beta,
                        mass: params.mass,
                        nf: params.nf,
                        n_configs: params.n_configs,
                        params: params.default_params.clone(),
                    };
                    matrix.add_task_internal(task_id, spec, TaskPriority::Sweep);
                }
            }
        }

        matrix
    }

    /// Add a targeted on-demand task.
    pub fn add_target(&mut self, task_id: &str, spec: ProcessSpec) -> usize {
        self.add_task_internal(task_id.to_string(), spec, TaskPriority::OnDemand)
    }

    fn add_task_internal(
        &mut self,
        id: String,
        spec: ProcessSpec,
        priority: TaskPriority,
    ) -> usize {
        let estimated = CostModel::estimate_wall_seconds(&spec, &self.hardware);
        let idx = self.tasks.len();
        self.tasks.push(TaskCell {
            id,
            spec,
            status: ProcessStatus::Pending,
            priority,
            estimated_seconds: estimated,
            actual_seconds: None,
            output_path: None,
            error: None,
            created_at: super::measurement::iso8601_now(),
            completed_at: None,
        });
        self.last_modified = super::measurement::iso8601_now();
        idx
    }

    /// Pick the next ready task, respecting dependencies and priority.
    ///
    /// On-demand/urgent tasks take precedence over sweep tasks.
    /// Within the same priority, tasks are returned in insertion order.
    pub fn next_ready(&self) -> Option<usize> {
        let mut best: Option<(usize, TaskPriority)> = None;

        for (i, task) in self.tasks.iter().enumerate() {
            if task.status != ProcessStatus::Pending {
                continue;
            }
            if !self.deps_met_for(i) {
                continue;
            }
            match best {
                None => best = Some((i, task.priority)),
                Some((_, bp)) if task.priority > bp => best = Some((i, task.priority)),
                _ => {}
            }
        }

        best.map(|(i, _)| i)
    }

    /// Check if all dependencies are met for a task at the given index.
    fn deps_met_for(&self, idx: usize) -> bool {
        let task = &self.tasks[idx];
        let deps = task.spec.process.dependencies();

        deps.iter().all(|dep| {
            self.tasks.iter().any(|t| {
                t.spec.process == *dep
                    && t.spec.dims == task.spec.dims
                    && (t.spec.beta - task.spec.beta).abs() < 1e-10
                    && t.status == ProcessStatus::Completed
            })
        })
    }

    /// Mark a task as running.
    pub fn mark_running(&mut self, idx: usize) {
        if let Some(task) = self.tasks.get_mut(idx) {
            task.status = ProcessStatus::Running;
            self.last_modified = super::measurement::iso8601_now();
        }
    }

    /// Mark a task as completed.
    pub fn mark_completed(&mut self, idx: usize, wall_seconds: f64, output_path: Option<String>) {
        if let Some(task) = self.tasks.get_mut(idx) {
            task.status = ProcessStatus::Completed;
            task.actual_seconds = Some(wall_seconds);
            task.output_path = output_path;
            task.completed_at = Some(super::measurement::iso8601_now());
            self.last_modified = super::measurement::iso8601_now();
        }
    }

    /// Mark a task as failed.
    pub fn mark_failed(&mut self, idx: usize, error: &str) {
        if let Some(task) = self.tasks.get_mut(idx) {
            task.status = ProcessStatus::Failed;
            task.error = Some(error.to_string());
            self.last_modified = super::measurement::iso8601_now();
        }
    }

    /// Count tasks by status.
    pub fn status_counts(&self) -> StatusCounts {
        let mut counts = StatusCounts::default();
        for task in &self.tasks {
            match task.status {
                ProcessStatus::Pending => counts.pending += 1,
                ProcessStatus::Running => counts.running += 1,
                ProcessStatus::Completed => counts.completed += 1,
                ProcessStatus::Failed => counts.failed += 1,
                ProcessStatus::Skipped => counts.skipped += 1,
            }
        }
        counts
    }

    /// Total estimated time remaining for all pending tasks.
    pub fn estimated_remaining_seconds(&self) -> f64 {
        self.tasks
            .iter()
            .filter(|t| t.status == ProcessStatus::Pending || t.status == ProcessStatus::Running)
            .map(|t| t.estimated_seconds)
            .sum()
    }

    /// Generate a summary table as a formatted string.
    pub fn summary_table(&self) -> String {
        let counts = self.status_counts();
        let remaining = self.estimated_remaining_seconds();
        let total: usize = self.tasks.len();

        let mut out = String::with_capacity(2048);
        let _ = writeln!(&mut out, "Task Matrix: {}", self.matrix_id);
        let _ = writeln!(
            &mut out,
            "Hardware: {:?}  |  Total: {}  |  Pending: {}  Running: {}  Done: {}  Failed: {}",
            self.hardware, total, counts.pending, counts.running, counts.completed, counts.failed
        );
        let _ = writeln!(
            &mut out,
            "Estimated remaining: {}",
            CostModel::format_duration(remaining)
        );
        let _ = writeln!(&mut out);

        let _ = writeln!(
            &mut out,
            "{:<40} {:<14} {:<10} {:<12} {:<10}",
            "Task ID", "Process", "Status", "Est. Time", "Actual"
        );
        out.push_str(&"-".repeat(86));
        out.push('\n');

        for task in &self.tasks {
            let actual = task
                .actual_seconds
                .map_or_else(|| "-".to_string(), CostModel::format_duration);
            let _ = writeln!(
                &mut out,
                "{:<40} {:<14} {:<10} {:<12} {:<10}",
                task.id,
                task.spec.process.name(),
                task.status,
                CostModel::format_duration(task.estimated_seconds),
                actual
            );
        }

        out
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Save to a file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    /// Load from a file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// Task status counts.
#[derive(Default, Debug)]
pub struct StatusCounts {
    pub pending: usize,
    pub running: usize,
    pub completed: usize,
    pub failed: usize,
    pub skipped: usize,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn sweep_generates_correct_grid() {
        let params = SweepParams {
            betas: vec![5.8, 6.0],
            lattice_sizes: vec![8, 12],
            nt_override: None,
            observables: vec![PhysicsProcess::GradientFlow, PhysicsProcess::WilsonLoops],
            n_configs: 50,
            mass: 0.0,
            nf: 0,
            default_params: ProcessParams::default(),
        };

        let matrix = TaskMatrix::from_sweep("test_sweep", HardwareTier::Cpu, &params);

        // 2 betas × 2 sizes × (1 Generate + 2 observables) = 12 tasks
        assert_eq!(matrix.tasks.len(), 12);

        let gen_count = matrix
            .tasks
            .iter()
            .filter(|t| t.spec.process == PhysicsProcess::Generate)
            .count();
        assert_eq!(gen_count, 4); // 2 betas × 2 sizes
    }

    #[test]
    fn next_ready_respects_dependencies() {
        let params = SweepParams {
            betas: vec![6.0],
            lattice_sizes: vec![8],
            nt_override: None,
            observables: vec![PhysicsProcess::GradientFlow],
            n_configs: 10,
            mass: 0.0,
            nf: 0,
            default_params: ProcessParams::default(),
        };

        let matrix = TaskMatrix::from_sweep("dep_test", HardwareTier::Cpu, &params);

        // First ready task should be Generate (has no dependencies)
        let next = matrix.next_ready().unwrap();
        assert_eq!(matrix.tasks[next].spec.process, PhysicsProcess::Generate);
    }

    #[test]
    fn on_demand_takes_priority() {
        let params = SweepParams {
            betas: vec![6.0],
            lattice_sizes: vec![8],
            nt_override: None,
            observables: vec![],
            n_configs: 10,
            mass: 0.0,
            nf: 0,
            default_params: ProcessParams::default(),
        };

        let mut matrix = TaskMatrix::from_sweep("priority_test", HardwareTier::Cpu, &params);

        // Add an on-demand Generate for a different beta
        matrix.add_target(
            "urgent_gen",
            ProcessSpec {
                process: PhysicsProcess::Generate,
                dims: [8, 8, 8, 8],
                beta: 5.5,
                mass: 0.0,
                nf: 0,
                n_configs: 5,
                params: ProcessParams::default(),
            },
        );

        let next = matrix.next_ready().unwrap();
        assert_eq!(matrix.tasks[next].id, "urgent_gen");
        assert_eq!(matrix.tasks[next].priority, TaskPriority::OnDemand);
    }

    #[test]
    fn task_lifecycle() {
        let mut matrix = TaskMatrix::new("lifecycle_test", HardwareTier::Cpu);
        let idx = matrix.add_target(
            "test_task",
            ProcessSpec {
                process: PhysicsProcess::Generate,
                dims: [4, 4, 4, 4],
                beta: 6.0,
                mass: 0.0,
                nf: 0,
                n_configs: 1,
                params: ProcessParams::default(),
            },
        );

        assert_eq!(matrix.tasks[idx].status, ProcessStatus::Pending);

        matrix.mark_running(idx);
        assert_eq!(matrix.tasks[idx].status, ProcessStatus::Running);

        matrix.mark_completed(idx, 1.5, Some("/tmp/output".to_string()));
        assert_eq!(matrix.tasks[idx].status, ProcessStatus::Completed);
        assert_eq!(matrix.tasks[idx].actual_seconds, Some(1.5));

        let counts = matrix.status_counts();
        assert_eq!(counts.completed, 1);
        assert_eq!(counts.pending, 0);
    }

    #[test]
    fn json_roundtrip() {
        let mut matrix = TaskMatrix::new("roundtrip", HardwareTier::ConsumerGpu);
        matrix.add_target(
            "t1",
            ProcessSpec {
                process: PhysicsProcess::GradientFlow,
                dims: [8, 8, 8, 8],
                beta: 6.0,
                mass: 0.0,
                nf: 0,
                n_configs: 10,
                params: ProcessParams::default(),
            },
        );

        let json = matrix.to_json();
        let parsed = TaskMatrix::from_json(&json).unwrap();
        assert_eq!(parsed.tasks.len(), 1);
        assert_eq!(parsed.matrix_id, "roundtrip");
    }

    #[test]
    fn summary_table_format() {
        let params = SweepParams {
            betas: vec![6.0],
            lattice_sizes: vec![8],
            nt_override: None,
            observables: vec![PhysicsProcess::GradientFlow],
            n_configs: 10,
            mass: 0.0,
            nf: 0,
            default_params: ProcessParams::default(),
        };

        let matrix = TaskMatrix::from_sweep("fmt_test", HardwareTier::Cpu, &params);
        let table = matrix.summary_table();

        assert!(table.contains("Task Matrix: fmt_test"));
        assert!(table.contains("Pending:"));
        assert!(table.contains("Generate"));
        assert!(table.contains("GradientFlow"));
    }
}
