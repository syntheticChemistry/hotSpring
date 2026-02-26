// SPDX-License-Identifier: AGPL-3.0-only

//! Streaming pipeline — daisy-chain workloads across substrates.
//!
//! A pipeline is a directed graph of stages connected by typed edges.
//! Each stage runs on a substrate (GPU, NPU, CPU) and produces output
//! that feeds the next stage. The pipeline can be reordered at runtime
//! to empirically discover which topology maximizes silicon efficiency.
//!
//! # Design
//!
//! ```text
//!   Stage(GPU:3090)  ──PCIe──▶  Stage(NPU:AKD1000)  ──PCIe──▶  Stage(GPU:TitanV)
//!    [HMC compute]               [phase classify]                 [validation oracle]
//! ```
//!
//! Contiguous work units flow through stages. Each stage processes a
//! `WorkUnit` and emits zero or more output units. Edges carry typed
//! channels (PCIe, shared memory, or within-device).
//!
//! # Absorption target: `toadstool::pipeline`
//!
//! This module evolves locally in hotSpring/metalForge. When the topology
//! explorer proves which orderings maximize throughput/watt, toadstool
//! absorbs the pipeline abstraction for all biomes.

use crate::substrate::{Substrate, SubstrateKind};
use std::collections::HashMap;
use std::fmt;

/// A compute stage in the streaming pipeline.
#[derive(Debug)]
pub struct Stage {
    pub id: StageId,
    pub name: String,
    pub substrate_kind: SubstrateKind,
    pub substrate_name: Option<String>,
    pub role: StageRole,
}

/// Unique identifier for a pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StageId(pub u32);

/// What a stage does in the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageRole {
    /// Heavy f64 compute (HMC, CG solver, force evaluation).
    Compute,
    /// Inference or classification (NPU phase detection, ESN prediction).
    Inference,
    /// Validation oracle (cross-card verification, checksum).
    Validation,
    /// Steering / decision (adaptive beta selection, convergence check).
    Steering,
    /// Reduction or aggregation (observable averaging, statistics).
    Reduce,
}

/// A directed edge between two stages.
#[derive(Debug)]
pub struct Edge {
    pub from: StageId,
    pub to: StageId,
    pub channel: ChannelKind,
}

/// How data moves between stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelKind {
    /// PCIe DMA transfer (GPU↔CPU, GPU↔NPU via CPU bridge).
    Pcie,
    /// Shared CPU memory (zero-copy between CPU stages).
    SharedMemory,
    /// Within the same device (no transfer needed).
    Local,
}

/// A streaming pipeline topology.
///
/// Stages are connected by edges forming a DAG. The pipeline can be
/// executed in topological order, or reordered to test alternative
/// topologies for silicon efficiency discovery.
#[derive(Debug)]
pub struct Pipeline {
    pub name: String,
    stages: Vec<Stage>,
    edges: Vec<Edge>,
}

/// A unit of work flowing through the pipeline.
///
/// Carries the minimum data needed between stages. For QCD: beta value,
/// plaquette, Polyakov loop, acceptance rate. Lightweight enough for
/// PCIe transfer (~100 bytes, not megabytes).
#[derive(Debug, Clone)]
pub struct WorkUnit {
    pub id: u64,
    pub stage_id: StageId,
    pub payload: HashMap<String, f64>,
}

/// Result of executing a pipeline with a specific topology.
#[derive(Debug)]
pub struct PipelineResult {
    pub topology_name: String,
    pub total_wall_s: f64,
    pub stage_times: Vec<(StageId, f64)>,
    pub throughput_units_per_s: f64,
    pub energy_j: Option<f64>,
}

impl Pipeline {
    /// Create a new empty pipeline.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a stage and return its ID.
    pub fn add_stage(
        &mut self,
        name: impl Into<String>,
        kind: SubstrateKind,
        role: StageRole,
    ) -> StageId {
        #[allow(clippy::cast_possible_truncation)]
        let id = StageId(self.stages.len() as u32);
        self.stages.push(Stage {
            id,
            name: name.into(),
            substrate_kind: kind,
            substrate_name: None,
            role,
        });
        id
    }

    /// Bind a stage to a discovered substrate.
    pub fn bind_stage(&mut self, stage_id: StageId, substrate: &Substrate) {
        if let Some(stage) = self.stages.iter_mut().find(|s| s.id == stage_id) {
            stage.substrate_name = Some(substrate.identity.name.clone());
        }
    }

    /// Connect two stages with a typed channel.
    pub fn connect(&mut self, from: StageId, to: StageId, channel: ChannelKind) {
        self.edges.push(Edge { from, to, channel });
    }

    /// Return stages in topological order.
    #[must_use]
    pub fn ordered_stages(&self) -> Vec<&Stage> {
        let mut in_degree: HashMap<StageId, usize> = HashMap::new();
        for stage in &self.stages {
            in_degree.entry(stage.id).or_insert(0);
        }
        for edge in &self.edges {
            *in_degree.entry(edge.to).or_insert(0) += 1;
        }

        let mut queue: Vec<StageId> = in_degree
            .iter()
            .filter_map(|(&id, &deg)| if deg == 0 { Some(id) } else { None })
            .collect();
        queue.sort_by_key(|id| id.0);

        let mut result = Vec::new();
        while let Some(current) = queue.pop() {
            if let Some(stage) = self.stages.iter().find(|s| s.id == current) {
                result.push(stage);
            }
            for edge in &self.edges {
                if edge.from == current {
                    if let Some(deg) = in_degree.get_mut(&edge.to) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(edge.to);
                            queue.sort_by_key(|id| id.0);
                        }
                    }
                }
            }
        }
        result
    }

    /// Get all stages.
    #[must_use]
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Get all edges.
    #[must_use]
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Count stages by substrate kind.
    #[must_use]
    pub fn substrate_counts(&self) -> HashMap<SubstrateKind, usize> {
        let mut counts = HashMap::new();
        for stage in &self.stages {
            *counts.entry(stage.substrate_kind).or_insert(0) += 1;
        }
        counts
    }
}

impl fmt::Display for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pipeline: {}", self.name)?;
        for stage in self.ordered_stages() {
            let bound = stage.substrate_name.as_deref().unwrap_or("unbound");
            writeln!(
                f,
                "  [{:?}] {} on {} ({})",
                stage.role, stage.name, stage.substrate_kind, bound
            )?;
        }
        for edge in &self.edges {
            writeln!(
                f,
                "  {} -> {} via {:?}",
                edge.from.0, edge.to.0, edge.channel
            )?;
        }
        Ok(())
    }
}

impl WorkUnit {
    /// Create a new work unit with an ID and stage.
    #[must_use]
    pub fn new(id: u64, stage_id: StageId) -> Self {
        Self {
            id,
            stage_id,
            payload: HashMap::new(),
        }
    }

    /// Set a named f64 value in the payload.
    pub fn set(&mut self, key: impl Into<String>, value: f64) {
        self.payload.insert(key.into(), value);
    }

    /// Get a named f64 value from the payload.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<f64> {
        self.payload.get(key).copied()
    }
}

/// Predefined pipeline topologies for QCD workloads.
pub mod topologies {
    use super::{ChannelKind, Pipeline, StageRole, SubstrateKind};

    /// Topology A: GPU → NPU → Titan V → CPU steering.
    ///
    /// Primary compute on 3090 (DF64), NPU screens for phase transitions,
    /// Titan V validates critical points with native f64, CPU steers.
    /// Runtime feedback (steering → compute) is handled outside the DAG.
    #[must_use]
    pub fn qcd_gpu_npu_oracle() -> Pipeline {
        let mut p = Pipeline::new("QCD: GPU → NPU → Oracle → CPU");
        let compute = p.add_stage("HMC Compute (3090)", SubstrateKind::Gpu, StageRole::Compute);
        let screen = p.add_stage(
            "Phase Screen (NPU)",
            SubstrateKind::Npu,
            StageRole::Inference,
        );
        let oracle = p.add_stage(
            "Validation (Titan V)",
            SubstrateKind::Gpu,
            StageRole::Validation,
        );
        let steer = p.add_stage(
            "Adaptive Steering (CPU)",
            SubstrateKind::Cpu,
            StageRole::Steering,
        );

        p.connect(compute, screen, ChannelKind::Pcie);
        p.connect(screen, oracle, ChannelKind::Pcie);
        p.connect(oracle, steer, ChannelKind::Pcie);
        p
    }

    /// Topology B: Parallel GPUs → NPU → CPU merge.
    ///
    /// Both GPUs compute simultaneously (3090: DF64, Titan V: native f64),
    /// NPU screens combined results, CPU merges and steers.
    #[must_use]
    pub fn qcd_parallel_gpu() -> Pipeline {
        let mut p = Pipeline::new("QCD: Parallel GPUs → NPU → CPU");
        let gpu1 = p.add_stage("HMC (3090 DF64)", SubstrateKind::Gpu, StageRole::Compute);
        let gpu2 = p.add_stage("HMC (Titan V f64)", SubstrateKind::Gpu, StageRole::Compute);
        let screen = p.add_stage(
            "Phase Screen (NPU)",
            SubstrateKind::Npu,
            StageRole::Inference,
        );
        let merge = p.add_stage(
            "Cross-Card Validate (CPU)",
            SubstrateKind::Cpu,
            StageRole::Validation,
        );
        let steer = p.add_stage("Steering (CPU)", SubstrateKind::Cpu, StageRole::Steering);

        p.connect(gpu1, merge, ChannelKind::Pcie);
        p.connect(gpu2, merge, ChannelKind::Pcie);
        p.connect(merge, screen, ChannelKind::Pcie);
        p.connect(screen, steer, ChannelKind::SharedMemory);
        p
    }

    /// Topology C: NPU-first screening → GPU compute.
    ///
    /// NPU predicts which beta points are interesting BEFORE expensive
    /// HMC. GPU only computes points the NPU flags as high-uncertainty.
    #[must_use]
    pub fn qcd_npu_first() -> Pipeline {
        let mut p = Pipeline::new("QCD: NPU → GPU → CPU");
        let predict = p.add_stage(
            "ESN Predict (NPU)",
            SubstrateKind::Npu,
            StageRole::Inference,
        );
        let compute = p.add_stage("HMC Compute (GPU)", SubstrateKind::Gpu, StageRole::Compute);
        let reduce = p.add_stage(
            "Observable Reduce (CPU)",
            SubstrateKind::Cpu,
            StageRole::Reduce,
        );

        p.connect(predict, compute, ChannelKind::Pcie);
        p.connect(compute, reduce, ChannelKind::Pcie);
        p
    }

    /// Topology D: CPU-only (baseline).
    ///
    /// Pure Rust CPU baseline for parity comparison.
    #[must_use]
    pub fn qcd_cpu_baseline() -> Pipeline {
        let mut p = Pipeline::new("QCD: CPU-only baseline");
        let compute = p.add_stage("HMC (CPU f64)", SubstrateKind::Cpu, StageRole::Compute);
        let reduce = p.add_stage("Reduce (CPU)", SubstrateKind::Cpu, StageRole::Reduce);

        p.connect(compute, reduce, ChannelKind::Local);
        p
    }

    /// Topology E: GPU-only (current Exp 018 benchmark).
    ///
    /// Single GPU with DF64 extension strategy.
    #[must_use]
    pub fn qcd_gpu_only() -> Pipeline {
        let mut p = Pipeline::new("QCD: GPU-only (DF64)");
        let compute = p.add_stage("HMC (GPU DF64)", SubstrateKind::Gpu, StageRole::Compute);
        let reduce = p.add_stage("Reduce (GPU)", SubstrateKind::Gpu, StageRole::Reduce);

        p.connect(compute, reduce, ChannelKind::Local);
        p
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::{topologies, ChannelKind, StageId, StageRole, SubstrateKind, WorkUnit};

    #[test]
    fn pipeline_creation() {
        let p = topologies::qcd_gpu_npu_oracle();
        assert_eq!(p.stages().len(), 4);
        assert_eq!(p.edges().len(), 3);
    }

    #[test]
    fn topological_order() {
        let p = topologies::qcd_npu_first();
        let ordered = p.ordered_stages();
        assert_eq!(ordered.len(), 3);
        assert_eq!(ordered[0].role, StageRole::Inference);
        assert_eq!(ordered[1].role, StageRole::Compute);
        assert_eq!(ordered[2].role, StageRole::Reduce);
    }

    #[test]
    fn parallel_topology_has_two_gpus() {
        let p = topologies::qcd_parallel_gpu();
        let counts = p.substrate_counts();
        assert_eq!(counts.get(&SubstrateKind::Gpu), Some(&2));
    }

    #[test]
    fn work_unit_payload() {
        let mut wu = WorkUnit::new(1, StageId(0));
        wu.set("plaquette", 0.531);
        wu.set("beta", 5.69);
        assert!((wu.get("plaquette").expect("set") - 0.531).abs() < 1e-10);
        assert!((wu.get("beta").expect("set") - 5.69).abs() < 1e-10);
        assert!(wu.get("missing").is_none());
    }

    #[test]
    fn cpu_baseline_is_local_only() {
        let p = topologies::qcd_cpu_baseline();
        assert!(p.edges().iter().all(|e| e.channel == ChannelKind::Local));
    }

    #[test]
    fn display_pipeline() {
        let p = topologies::qcd_gpu_npu_oracle();
        let s = format!("{p}");
        assert!(s.contains("HMC Compute"));
        assert!(s.contains("Phase Screen"));
        assert!(s.contains("Validation"));
    }
}
