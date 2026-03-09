// SPDX-License-Identifier: AGPL-3.0-only

//! Mixed-hardware pipeline infrastructure for metalForge integration.
//!
//! Defines substrate abstractions for cross-device compute:
//!   - GPU (RTX 3090, Titan V via Vulkan/wgpu)
//!   - NPU (Akida AKD1000 via akida-driver)
//!   - CPU (Threadripper 3970X via rayon)
//!
//! The target architecture is:
//!   GPU buffer → PCIe DMA → NPU inference → PCIe DMA → GPU buffer
//!
//! Current flow (CPU-mediated):
//!   GPU buffer → CPU readback → NPU inference → CPU → GPU upload
//!
//! This module provides the abstraction layer that metalForge will absorb.
//! hotSpring builds the physics pipeline locally; metalForge generalizes
//! the cross-substrate routing.

/// Compute substrate identification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Substrate {
    /// CPU (all cores available)
    Cpu,
    /// GPU via Vulkan/wgpu
    Gpu {
        /// Device index (0 = primary)
        device_id: u32,
    },
    /// Neural Processing Unit (Akida AKD1000)
    Npu,
}

impl std::fmt::Display for Substrate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu { device_id } => write!(f, "GPU:{device_id}"),
            Self::Npu => write!(f, "NPU"),
        }
    }
}

/// Transfer path between substrates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferPath {
    /// Direct PCIe DMA (no CPU round-trip)
    PcieDirect,
    /// CPU-mediated: device → host memory → device
    CpuMediated,
    /// Same device (no transfer needed)
    Local,
}

/// Capability of a substrate for a given task.
#[derive(Clone, Debug)]
pub struct SubstrateCapability {
    /// Which substrate
    pub substrate: Substrate,
    /// Whether this substrate can handle the task
    pub available: bool,
    /// Estimated throughput (operations per second)
    pub throughput_ops: f64,
    /// Estimated latency for single operation (microseconds)
    pub latency_us: f64,
}

/// Pipeline stage in a mixed-hardware workflow.
#[derive(Clone, Debug)]
pub struct PipelineStage {
    /// Human-readable name
    pub name: String,
    /// Where this stage executes
    pub substrate: Substrate,
    /// How data arrives from previous stage
    pub input_transfer: TransferPath,
    /// Estimated wall time (seconds)
    pub estimated_seconds: f64,
}

/// A complete mixed-hardware pipeline (e.g., HMC → NPU → GPU).
#[derive(Clone, Debug)]
pub struct MixedPipeline {
    /// Ordered list of pipeline stages.
    pub stages: Vec<PipelineStage>,
}

impl MixedPipeline {
    /// Total estimated wall time.
    #[must_use]
    pub fn total_estimated_seconds(&self) -> f64 {
        self.stages.iter().map(|s| s.estimated_seconds).sum()
    }

    /// Count CPU-mediated transfers (optimization targets).
    #[must_use]
    pub fn cpu_mediated_transfers(&self) -> usize {
        self.stages
            .iter()
            .filter(|s| s.input_transfer == TransferPath::CpuMediated)
            .count()
    }

    /// Count PCIe direct transfers (goal).
    #[must_use]
    pub fn pcie_direct_transfers(&self) -> usize {
        self.stages
            .iter()
            .filter(|s| s.input_transfer == TransferPath::PcieDirect)
            .count()
    }
}

/// Build the current HMC pipeline (CPU-mediated NPU steering).
///
/// This represents the current flow where GPU HMC results are read back
/// to CPU, sent to NPU for steering advice, then the advice is applied
/// on CPU before the next GPU dispatch.
#[must_use]
pub fn current_hmc_pipeline() -> MixedPipeline {
    MixedPipeline {
        stages: vec![
            PipelineStage {
                name: "GPU HMC trajectory".to_string(),
                substrate: Substrate::Gpu { device_id: 0 },
                input_transfer: TransferPath::Local,
                estimated_seconds: 0.5,
            },
            PipelineStage {
                name: "CPU readback (plaquette, deltaH)".to_string(),
                substrate: Substrate::Cpu,
                input_transfer: TransferPath::CpuMediated,
                estimated_seconds: 0.001,
            },
            PipelineStage {
                name: "NPU steering (accept/reject, dt adapt)".to_string(),
                substrate: Substrate::Npu,
                input_transfer: TransferPath::CpuMediated,
                estimated_seconds: 0.0005,
            },
            PipelineStage {
                name: "CPU apply steering".to_string(),
                substrate: Substrate::Cpu,
                input_transfer: TransferPath::CpuMediated,
                estimated_seconds: 0.0001,
            },
        ],
    }
}

/// Build the target HMC pipeline (PCIe direct NPU steering).
///
/// In this architecture, GPU HMC writes observables to a shared buffer,
/// NPU reads directly via PCIe DMA, writes steering advice back, and
/// GPU reads the advice — all without CPU involvement.
#[must_use]
pub fn target_hmc_pipeline() -> MixedPipeline {
    MixedPipeline {
        stages: vec![
            PipelineStage {
                name: "GPU HMC trajectory".to_string(),
                substrate: Substrate::Gpu { device_id: 0 },
                input_transfer: TransferPath::Local,
                estimated_seconds: 0.5,
            },
            PipelineStage {
                name: "NPU steering (PCIe DMA read)".to_string(),
                substrate: Substrate::Npu,
                input_transfer: TransferPath::PcieDirect,
                estimated_seconds: 0.0003,
            },
            PipelineStage {
                name: "GPU apply steering (PCIe DMA write)".to_string(),
                substrate: Substrate::Gpu { device_id: 0 },
                input_transfer: TransferPath::PcieDirect,
                estimated_seconds: 0.0001,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_pipeline_has_cpu_mediation() {
        let pipe = current_hmc_pipeline();
        assert!(pipe.cpu_mediated_transfers() > 0);
        assert_eq!(pipe.pcie_direct_transfers(), 0);
    }

    #[test]
    fn target_pipeline_eliminates_cpu() {
        let pipe = target_hmc_pipeline();
        assert_eq!(pipe.cpu_mediated_transfers(), 0);
        assert!(pipe.pcie_direct_transfers() > 0);
    }

    #[test]
    fn target_faster_than_current() {
        let current = current_hmc_pipeline();
        let target = target_hmc_pipeline();
        assert!(target.total_estimated_seconds() <= current.total_estimated_seconds());
    }

    #[test]
    fn substrate_display() {
        assert_eq!(Substrate::Cpu.to_string(), "CPU");
        assert_eq!(Substrate::Gpu { device_id: 0 }.to_string(), "GPU:0");
        assert_eq!(Substrate::Npu.to_string(), "NPU");
    }
}
