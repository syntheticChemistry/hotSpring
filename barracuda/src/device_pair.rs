// SPDX-License-Identifier: AGPL-3.0-only

//! Heterogeneous dual-GPU device pair: "precise brain" + "throughput brain."
//!
//! Profiles two GPUs and assigns roles based on hardware FP64 rate:
//! - **Precise**: native f64 at full rate (Titan V, V100, MI250)
//! - **Throughput**: DF64 on f32 cores for ~10x effective throughput (3090, 4070)
//!
//! The bridge models CPU-mediated PCIe transfer cost between the two cards.
//!
//! ## Upstream evolution (toadStool S145)
//!
//! toadStool S145 evolved `PcieTransport` into full switch-level topology:
//! `PciBridge`, `GpuPairTopology`, `PcieTopologyGraph` (sysfs probed),
//! and `WorkloadRouter::route_multi_gpu()` for topology-aware multi-GPU
//! placement with 8 new `WorkloadPatterns` (Pairwise, BatchFitness,
//! HmmBatch, SpatialPayoff, Stochastic, PopulationPk, DoseResponse,
//! DiversityIndex). S145 added `ProviderRegistry` for spring-as-provider
//! socket resolution and `NvkZeroGuard` for zero-output detection.
//! When hotSpring integrates toadStool's orchestration layer, `DevicePair`
//! should delegate topology discovery to `PcieTopologyGraph` and multi-GPU
//! routing to `WorkloadRouter` instead of manual PCIe bandwidth estimation.

use barracuda::device::driver_profile::{Fp64Rate, Fp64Strategy};
use barracuda::unified_hardware::{BandwidthTier, PcieBridge, TransferCost};

use crate::error::HotSpringError;
use crate::gpu::GpuF64;

/// Computed capability profile for a device pair.
#[derive(Debug, Clone)]
pub struct PairProfile {
    /// Effective FP64 TFLOPS on the precise card (native f64).
    pub precise_tflops_f64: f64,
    /// Peak FP32 TFLOPS on the throughput card.
    pub throughput_tflops_f32: f64,
    /// Estimated DF64 TFLOPS on the throughput card (~0.5x f32 peak).
    pub throughput_tflops_df64: f64,
    /// Effective bridge bandwidth in GB/s (limited by slowest PCIe link).
    pub bridge_bandwidth_gbps: f64,
    /// Bridge one-way latency in microseconds.
    pub bridge_latency_us: f64,
    /// Name of the precise card.
    pub precise_name: String,
    /// Name of the throughput card.
    pub throughput_name: String,
}

impl PairProfile {
    /// Estimate transfer time in microseconds for `bytes` across the bridge.
    #[must_use]
    pub fn transfer_us(&self, bytes: usize) -> f64 {
        self.bridge_latency_us + (bytes as f64) / (self.bridge_bandwidth_gbps * 1000.0)
    }

    /// Compute the optimal split fraction for the precise card.
    ///
    /// Returns the fraction of work that should go to the precise card
    /// when splitting a throughput-bound f64 workload across both GPUs.
    #[must_use]
    pub fn precise_split_fraction(&self) -> f64 {
        let total = self.precise_tflops_f64 + self.throughput_tflops_df64;
        if total <= 0.0 {
            return 0.5;
        }
        self.precise_tflops_f64 / total
    }
}

/// Heterogeneous dual-GPU device pair.
///
/// Owns two `GpuF64` instances assigned by hardware capability:
/// the card with `Fp64Rate::Full` becomes the "precise" brain,
/// the other becomes the "throughput" brain (DF64 on f32 cores).
pub struct DevicePair {
    /// The "precise brain" — native f64 at full rate.
    pub precise: GpuF64,
    /// The "throughput brain" — DF64 on f32 cores.
    pub throughput: GpuF64,
    /// PCIe bridge cost model between the two cards.
    pub bridge: PcieBridge,
    /// Computed capability profile.
    pub profile: PairProfile,
}

/// Estimate peak f32 TFLOPS from adapter name heuristics.
fn estimate_f32_tflops(name: &str) -> f64 {
    let lower = name.to_lowercase();
    if lower.contains("3090") {
        35.6
    } else if lower.contains("4090") {
        82.6
    } else if lower.contains("4080") {
        48.7
    } else if lower.contains("4070 ti") {
        40.1
    } else if lower.contains("4070") {
        29.1
    } else if lower.contains("3080") {
        29.8
    } else if lower.contains("3070") {
        20.3
    } else if lower.contains("titan v") {
        14.9
    } else if lower.contains("v100") {
        15.7
    } else if lower.contains("a100") {
        19.5
    } else if lower.contains("mi250") || lower.contains("mi300") {
        47.9
    } else {
        10.0 // conservative fallback
    }
}

/// Estimate peak f64 TFLOPS from adapter name heuristics.
fn estimate_f64_tflops(name: &str, fp64_rate: Fp64Rate) -> f64 {
    let f32_tflops = estimate_f32_tflops(name);
    match fp64_rate {
        Fp64Rate::Full => f32_tflops / 2.0,
        Fp64Rate::Throttled => f32_tflops / 32.0,
        Fp64Rate::Minimal => f32_tflops / 64.0,
        Fp64Rate::Software => 0.0,
    }
}

impl DevicePair {
    /// Construct a device pair from two pre-initialized GPUs.
    ///
    /// Automatically assigns roles: the card with `Fp64Rate::Full` becomes
    /// `precise`; the other becomes `throughput`. If both have the same rate,
    /// the card with more VRAM becomes `throughput` (larger batch capacity).
    pub fn from_gpus(gpu_a: GpuF64, gpu_b: GpuF64) -> Self {
        let is_full_a = gpu_a.fp64_strategy() == Fp64Strategy::Native;
        let is_full_b = gpu_b.fp64_strategy() == Fp64Strategy::Native;

        let (precise, throughput) = if is_full_a && !is_full_b {
            (gpu_a, gpu_b)
        } else if is_full_b && !is_full_a {
            (gpu_b, gpu_a)
        } else {
            // Same rate: card with less memory is precise (more memory → bigger batches → throughput)
            let mem_a = precise_memory_proxy(&gpu_a);
            let mem_b = precise_memory_proxy(&gpu_b);
            if mem_a <= mem_b {
                (gpu_a, gpu_b)
            } else {
                (gpu_b, gpu_a)
            }
        };

        let precise_tier = BandwidthTier::detect_from_adapter_name(&precise.adapter_name);
        let throughput_tier = BandwidthTier::detect_from_adapter_name(&throughput.adapter_name);
        let bridge_tier = slower_tier(precise_tier, throughput_tier);

        let bridge = PcieBridge {
            p2p_available: false,
            source_label: precise.adapter_name.clone(),
            target_label: throughput.adapter_name.clone(),
            tier: bridge_tier,
        };

        let precise_fp64_rate = if precise.fp64_strategy() == Fp64Strategy::Native {
            Fp64Rate::Full
        } else {
            Fp64Rate::Minimal
        };
        let precise_tflops_f64 =
            estimate_f64_tflops(&precise.adapter_name, precise_fp64_rate);
        let throughput_tflops_f32 = estimate_f32_tflops(&throughput.adapter_name);
        let throughput_tflops_df64 = throughput_tflops_f32 * 0.5;

        let profile = PairProfile {
            precise_tflops_f64,
            throughput_tflops_f32,
            throughput_tflops_df64,
            bridge_bandwidth_gbps: bridge_tier.bandwidth_gbps(),
            bridge_latency_us: bridge_tier.latency_us(),
            precise_name: precise.adapter_name.clone(),
            throughput_name: throughput.adapter_name.clone(),
        };

        Self {
            precise,
            throughput,
            bridge,
            profile,
        }
    }

    /// Discover and initialize both GPUs automatically.
    ///
    /// Uses `HOTSPRING_GPU_PRIMARY` / `HOTSPRING_GPU_SECONDARY` env vars,
    /// or auto-discovers by memory/capability.
    ///
    /// # Errors
    ///
    /// Returns error if fewer than two compatible GPUs are available.
    pub async fn discover() -> Result<Self, HotSpringError> {
        let (primary_hint, secondary_hint) = crate::gpu::discover_primary_and_secondary_adapters();

        let primary_hint = primary_hint.ok_or_else(|| {
            HotSpringError::DeviceCreation("No primary GPU with SHADER_F64 found".into())
        })?;
        let secondary_hint = secondary_hint.ok_or_else(|| {
            HotSpringError::DeviceCreation("No secondary GPU with SHADER_F64 found".into())
        })?;

        let gpu_a = GpuF64::from_adapter_name(&primary_hint).await?;
        let gpu_b = GpuF64::from_adapter_name(&secondary_hint).await?;

        Ok(Self::from_gpus(gpu_a, gpu_b))
    }

    /// Bridge transfer cost for moving `bytes` between the two cards.
    #[must_use]
    pub fn transfer_cost(&self) -> TransferCost {
        self.bridge.transfer_cost()
    }
}

impl std::fmt::Display for DevicePair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DevicePair:")?;
        writeln!(
            f,
            "  Precise:    {} ({:.1} TFLOPS f64)",
            self.profile.precise_name, self.profile.precise_tflops_f64
        )?;
        writeln!(
            f,
            "  Throughput: {} ({:.1} TFLOPS f32, {:.1} TFLOPS DF64)",
            self.profile.throughput_name,
            self.profile.throughput_tflops_f32,
            self.profile.throughput_tflops_df64
        )?;
        write!(
            f,
            "  Bridge:     {:.1} GB/s, {:.0} us latency",
            self.profile.bridge_bandwidth_gbps, self.profile.bridge_latency_us
        )
    }
}

fn precise_memory_proxy(gpu: &GpuF64) -> u64 {
    gpu.device().limits().max_buffer_size
}

fn slower_tier(a: BandwidthTier, b: BandwidthTier) -> BandwidthTier {
    if a.bandwidth_gbps() <= b.bandwidth_gbps() {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tflops_known_cards() {
        assert!((estimate_f32_tflops("NVIDIA GeForce RTX 3090") - 35.6).abs() < 0.1);
        assert!((estimate_f32_tflops("NVIDIA TITAN V") - 14.9).abs() < 0.1);
    }

    #[test]
    fn estimate_f64_full_rate() {
        let t = estimate_f64_tflops("NVIDIA TITAN V", Fp64Rate::Full);
        assert!((t - 7.45).abs() < 0.1);
    }

    #[test]
    fn estimate_f64_minimal_rate() {
        let t = estimate_f64_tflops("NVIDIA GeForce RTX 3090", Fp64Rate::Minimal);
        assert!(t < 1.0, "3090 f64 should be < 1 TFLOPS, got {t}");
    }

    #[test]
    fn pair_profile_transfer_us() {
        let profile = PairProfile {
            precise_tflops_f64: 7.5,
            throughput_tflops_f32: 35.6,
            throughput_tflops_df64: 17.8,
            bridge_bandwidth_gbps: 15.75,
            bridge_latency_us: 5.0,
            precise_name: "Titan V".into(),
            throughput_name: "RTX 3090".into(),
        };
        let one_mb = profile.transfer_us(1_048_576);
        assert!(one_mb > 5.0, "1MB transfer should exceed latency");
        assert!(one_mb < 200.0, "1MB over PCIe3 should be fast");
    }

    #[test]
    fn pair_profile_split_fraction() {
        let profile = PairProfile {
            precise_tflops_f64: 7.5,
            throughput_tflops_f32: 35.6,
            throughput_tflops_df64: 17.8,
            bridge_bandwidth_gbps: 15.75,
            bridge_latency_us: 5.0,
            precise_name: "Titan V".into(),
            throughput_name: "RTX 3090".into(),
        };
        let frac = profile.precise_split_fraction();
        assert!(
            frac > 0.2 && frac < 0.4,
            "Titan should get ~30%, got {frac}"
        );
    }

    #[test]
    fn slower_tier_picks_min() {
        assert_eq!(
            slower_tier(BandwidthTier::PciE3x16, BandwidthTier::PciE4x16),
            BandwidthTier::PciE3x16
        );
        assert_eq!(
            slower_tier(
                BandwidthTier::HighBandwidthInterconnect,
                BandwidthTier::PciE4x16
            ),
            BandwidthTier::PciE4x16
        );
    }
}
