// SPDX-License-Identifier: AGPL-3.0-only

//! Substrate abstraction — runtime-discovered compute devices.
//!
//! A substrate is a compute device found on this machine. GPUs come from
//! wgpu adapter enumeration (same path toadstool/barracuda uses). NPUs
//! come from local device node probing. CPU comes from procfs.
//!
//! Capabilities are what matters for dispatch — code asks "can you do f64?"
//! not "are you an RTX 4070?".

use std::fmt;

/// A compute substrate discovered at runtime.
#[derive(Debug, Clone)]
pub struct Substrate {
    pub kind: SubstrateKind,
    pub identity: Identity,
    pub properties: Properties,
    pub capabilities: Vec<Capability>,
}

/// How we found this device and what to call it.
#[derive(Debug, Clone)]
pub struct Identity {
    pub name: String,
    /// GPU driver string from wgpu, e.g. "NVIDIA (580.82.09)".
    pub driver: Option<String>,
    /// wgpu backend, e.g. "Vulkan".
    pub backend: Option<String>,
    /// wgpu adapter index for GPU selection.
    pub adapter_index: Option<usize>,
    /// Device node, e.g. "/dev/akida0".
    pub device_node: Option<String>,
    /// PCI vendor:device if available.
    pub pci_id: Option<String>,
}

/// Measured properties of a substrate.
#[derive(Debug, Clone, Default)]
pub struct Properties {
    /// Total memory in bytes (RAM for CPU, VRAM for GPU if known).
    pub memory_bytes: Option<u64>,
    /// Physical core count (CPU).
    pub core_count: Option<u32>,
    /// Logical thread count (CPU).
    pub thread_count: Option<u32>,
    /// Cache size in KB (CPU).
    pub cache_kb: Option<u32>,
    /// Supports IEEE 754 f64 in shaders (GPU).
    pub has_f64: bool,
    /// Supports timestamp queries (GPU).
    pub has_timestamps: bool,
    /// Hardware FP64:FP32 throughput ratio (e.g. 1:2 for Titan V, 1:64 for Ampere).
    pub fp64_rate: Option<Fp64Rate>,
    /// GPU supports DF64 (double-float f32-pair emulation on FP32 cores).
    pub has_df64: bool,
}

/// Hardware FP64 throughput relative to FP32.
///
/// Determines the optimal `Fp64Strategy`: cards with `Full` or `Half` benefit
/// from native f64 paths, while `Narrow` cards benefit from DF64 extension.
/// `Concurrent` saturates native units THEN overflows to DF64 on FP32 cores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp64Rate {
    /// 1:1 FP64:FP32 (datacenter: A100, H100).
    Full,
    /// 1:2 FP64:FP32 (Titan V / Volta, some Turing).
    Half,
    /// 1:32 or 1:64 FP64:FP32 (consumer Ampere, Ada, Turing).
    Narrow,
}

/// How to execute f64 work on a GPU substrate.
///
/// DF64 *extends* native FP64 — it doesn't replace it. We saturate the
/// native FP64 units first, then overflow into DF64 on FP32 cores to get
/// more aggregate f64 TFLOPS from the same silicon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp64Strategy {
    /// Use only hardware FP64 units (safe for Full/Half rate GPUs).
    Native,
    /// Use only DF64 on FP32 cores (fallback when native f64 unavailable).
    Hybrid,
    /// Saturate native FP64 units AND overflow to DF64 on FP32 cores.
    /// Maximizes aggregate f64 throughput on Narrow-rate GPUs.
    Concurrent,
}

impl Fp64Strategy {
    /// Select the optimal strategy for a substrate's properties.
    #[must_use]
    pub const fn for_properties(props: &Properties) -> Self {
        match props.fp64_rate {
            Some(Fp64Rate::Full | Fp64Rate::Half) => Self::Native,
            Some(Fp64Rate::Narrow) if props.has_df64 => Self::Concurrent,
            _ if props.has_f64 => Self::Native,
            _ if props.has_df64 => Self::Hybrid,
            _ => Self::Native,
        }
    }
}

/// The kind of compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubstrateKind {
    Gpu,
    Npu,
    Cpu,
}

/// A capability discovered at runtime on a substrate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Capability {
    /// IEEE 754 f64 compute (GPU `SHADER_F64` or CPU native).
    F64Compute,
    /// DF64: double-float (f32-pair) emulated f64 on FP32 shader cores.
    /// Provides ~14 digits of precision at FP32-core throughput.
    DF64Compute,
    /// f32 compute.
    F32Compute,
    /// Integer quantized inference at a given bit width.
    QuantizedInference { bits: u8 },
    /// Batch inference with amortized dispatch.
    BatchInference { max_batch: usize },
    /// Weight mutation without full reprogramming.
    WeightMutation,
    /// Scalar reduction (e.g. GPU reduce pipeline).
    ScalarReduce,
    /// Sparse matrix-vector product.
    SparseSpMV,
    /// Eigensolve (Lanczos, etc.).
    Eigensolve,
    /// Conjugate gradient solver.
    ConjugateGradient,
    /// WGSL shader dispatch via wgpu.
    ShaderDispatch,
    /// AVX2/SSE SIMD on CPU.
    SimdVector,
    /// GPU timestamp query support.
    TimestampQuery,
    /// PCIe peer-to-peer or DMA transfer capability.
    PcieTransfer,
    /// Streaming pipeline stage support (daisy-chain).
    StreamingStage,
}

impl fmt::Display for SubstrateKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu => write!(f, "GPU"),
            Self::Npu => write!(f, "NPU"),
            Self::Cpu => write!(f, "CPU"),
        }
    }
}

impl fmt::Display for Substrate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}]", self.identity.name, self.kind)?;
        if let Some(ref driver) = self.identity.driver {
            write!(f, " {driver}")?;
        }
        if let Some(mem) = self.properties.memory_bytes {
            let mb = mem / (1024 * 1024);
            write!(f, " {mb}MB")?;
        }
        Ok(())
    }
}

impl Substrate {
    /// Check if this substrate has a specific capability.
    #[must_use]
    pub fn has(&self, cap: &Capability) -> bool {
        self.capabilities.contains(cap)
    }

    /// Return capabilities as a summary string.
    #[must_use]
    pub fn capability_summary(&self) -> String {
        let labels: Vec<&str> = self.capabilities.iter().map(Capability::label).collect();
        labels.join(", ")
    }
}

impl Capability {
    /// Human-readable label for display.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::F64Compute => "f64",
            Self::DF64Compute => "df64",
            Self::F32Compute => "f32",
            Self::QuantizedInference { .. } => "quant",
            Self::BatchInference { .. } => "batch",
            Self::WeightMutation => "weight-mut",
            Self::ScalarReduce => "reduce",
            Self::SparseSpMV => "spmv",
            Self::Eigensolve => "eigen",
            Self::ConjugateGradient => "cg",
            Self::ShaderDispatch => "shader",
            Self::SimdVector => "simd",
            Self::TimestampQuery => "timestamps",
            Self::PcieTransfer => "pcie",
            Self::StreamingStage => "streaming",
        }
    }
}

impl Identity {
    /// Minimal identity with just a name.
    #[must_use]
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            driver: None,
            backend: None,
            adapter_index: None,
            device_node: None,
            pci_id: None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn test_gpu() -> Substrate {
        Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity {
                name: String::from("Test GPU"),
                adapter_index: Some(0),
                ..Identity::named("Test GPU")
            },
            properties: Properties {
                has_f64: true,
                ..Properties::default()
            },
            capabilities: vec![Capability::F64Compute, Capability::ShaderDispatch],
        }
    }

    #[test]
    fn has_capability() {
        let gpu = test_gpu();
        assert!(gpu.has(&Capability::F64Compute));
        assert!(gpu.has(&Capability::ShaderDispatch));
        assert!(!gpu.has(&Capability::QuantizedInference { bits: 8 }));
    }

    #[test]
    fn display_shows_kind_and_name() {
        let gpu = test_gpu();
        let s = format!("{gpu}");
        assert!(s.contains("Test GPU"));
        assert!(s.contains("GPU"));
    }

    #[test]
    fn capability_labels() {
        assert_eq!(Capability::F64Compute.label(), "f64");
        assert_eq!(Capability::ShaderDispatch.label(), "shader");
        assert_eq!(Capability::TimestampQuery.label(), "timestamps");
    }
}
