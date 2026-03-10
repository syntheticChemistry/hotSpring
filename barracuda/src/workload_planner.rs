// SPDX-License-Identifier: AGPL-3.0-only

//! Workload planner for heterogeneous dual-GPU dispatch.
//!
//! Given a `DevicePair` and a physics domain + data size, decides:
//! 1. **Where** the kernel runs (precise, throughput, or both)
//! 2. **How** to split if both (ratio based on effective TFLOPS)
//! 3. **Whether** PCIe transfer cost justifies the split
//!
//! ## Upstream evolution (toadStool S145)
//!
//! toadStool S145 evolved topology into `PcieTopologyGraph` with
//! `PciBridge`, `GpuPairTopology`, and sysfs-probed PCIe gen/lanes/NUMA.
//! `WorkloadRouter::route_multi_gpu()` provides topology-aware placement
//! via `MultiGpuPlacement` with 8 new `WorkloadPatterns`. S145 added
//! `ProviderRegistry` for spring-as-provider socket resolution and 5 new
//! capability domains (biology, health, measurement, optimization,
//! visualization). Future versions should query toadStool's topology
//! graph for real PCIe measurements and use `WorkloadRouter` for
//! placement decisions in multi-spring environments.

use crate::device_pair::DevicePair;
use crate::precision_routing::PhysicsDomain;

/// How a workload should be assigned across the device pair.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkloadAssignment {
    /// Run entirely on the precise card (native f64).
    PreciseOnly,
    /// Run entirely on the throughput card (DF64 on f32 cores).
    ThroughputOnly,
    /// Split across both cards with the given fraction on the precise card.
    Split {
        /// Fraction of work for the precise card (0.0..1.0).
        precise_fraction: f64,
    },
    /// Run on both cards with identical data for cross-validation.
    Redundant,
}

impl std::fmt::Display for WorkloadAssignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PreciseOnly => write!(f, "PreciseOnly"),
            Self::ThroughputOnly => write!(f, "ThroughputOnly"),
            Self::Split { precise_fraction } => {
                write!(f, "Split(precise={:.0}%)", precise_fraction * 100.0)
            }
            Self::Redundant => write!(f, "Redundant"),
        }
    }
}

/// Transfer cost gate: if transfer overhead exceeds this fraction of
/// compute time, don't split — keep data local.
const TRANSFER_GATE_FRACTION: f64 = 0.10;

/// Minimum compute time (us) below which splitting is never worthwhile
/// due to dispatch overhead on both GPUs.
const MIN_SPLIT_COMPUTE_US: f64 = 3000.0;

/// Plan a workload assignment for the given physics domain.
///
/// # Arguments
///
/// * `pair` — the device pair with profiled capabilities
/// * `domain` — the physics domain classification
/// * `data_bytes` — total data size that would need to cross the bridge for a split
/// * `compute_us_estimate` — estimated single-GPU compute time in microseconds
#[must_use]
pub fn plan_workload(
    pair: &DevicePair,
    domain: PhysicsDomain,
    data_bytes: usize,
    compute_us_estimate: f64,
) -> WorkloadAssignment {
    match domain {
        PhysicsDomain::Dielectric | PhysicsDomain::Eigensolve => WorkloadAssignment::PreciseOnly,

        PhysicsDomain::MolecularDynamics
        | PhysicsDomain::KineticFluid
        | PhysicsDomain::LatticeQcd
        | PhysicsDomain::GradientFlow
        | PhysicsDomain::NuclearEos
        | PhysicsDomain::PopulationPk
        | PhysicsDomain::Hydrology
        | PhysicsDomain::Bioinformatics
        | PhysicsDomain::Statistics
        | PhysicsDomain::General => {
            plan_throughput_or_split(pair, data_bytes, compute_us_estimate)
        }
    }
}

/// For throughput-bound domains: split if beneficial, else throughput-only.
fn plan_throughput_or_split(
    pair: &DevicePair,
    data_bytes: usize,
    compute_us_estimate: f64,
) -> WorkloadAssignment {
    if compute_us_estimate < MIN_SPLIT_COMPUTE_US {
        return WorkloadAssignment::ThroughputOnly;
    }

    let transfer_us = pair.profile.transfer_us(data_bytes);
    if transfer_us > TRANSFER_GATE_FRACTION * compute_us_estimate {
        return WorkloadAssignment::ThroughputOnly;
    }

    let frac = pair.profile.precise_split_fraction();
    if !(0.05..=0.95).contains(&frac) {
        return if frac < 0.05 {
            WorkloadAssignment::ThroughputOnly
        } else {
            WorkloadAssignment::PreciseOnly
        };
    }

    WorkloadAssignment::Split {
        precise_fraction: frac,
    }
}

/// Plan a redundant validation workload (same data on both cards).
#[must_use]
pub fn plan_redundant() -> WorkloadAssignment {
    WorkloadAssignment::Redundant
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_pair::PairProfile;

    fn make_profile() -> PairProfile {
        PairProfile {
            precise_tflops_f64: 7.5,
            throughput_tflops_f32: 35.6,
            throughput_tflops_df64: 17.8,
            bridge_bandwidth_gbps: 15.75,
            bridge_latency_us: 5.0,
            precise_name: "NVIDIA TITAN V".into(),
            throughput_name: "NVIDIA GeForce RTX 3090".into(),
        }
    }

    #[test]
    fn dielectric_always_precise() {
        let profile = make_profile();
        let assignment =
            plan_workload_from_profile(&profile, PhysicsDomain::Dielectric, 1024, 100_000.0);
        assert_eq!(assignment, WorkloadAssignment::PreciseOnly);
    }

    #[test]
    fn eigensolve_always_precise() {
        let profile = make_profile();
        let assignment =
            plan_workload_from_profile(&profile, PhysicsDomain::Eigensolve, 4096, 50_000.0);
        assert_eq!(assignment, WorkloadAssignment::PreciseOnly);
    }

    #[test]
    fn md_small_compute_stays_throughput() {
        let profile = make_profile();
        let assignment = plan_workload_from_profile(
            &profile,
            PhysicsDomain::MolecularDynamics,
            1_048_576,
            1000.0, // below MIN_SPLIT_COMPUTE_US
        );
        assert_eq!(assignment, WorkloadAssignment::ThroughputOnly);
    }

    #[test]
    fn lattice_large_compute_splits() {
        let profile = make_profile();
        let assignment = plan_workload_from_profile(
            &profile,
            PhysicsDomain::LatticeQcd,
            1_048_576, // 1MB
            500_000.0, // 500ms — large enough to justify split
        );
        match assignment {
            WorkloadAssignment::Split { precise_fraction } => {
                assert!(
                    precise_fraction > 0.2 && precise_fraction < 0.4,
                    "Titan should get ~30%, got {precise_fraction}"
                );
            }
            other => panic!("Expected Split, got {other}"),
        }
    }

    #[test]
    fn md_huge_transfer_stays_throughput() {
        let profile = make_profile();
        let assignment = plan_workload_from_profile(
            &profile,
            PhysicsDomain::MolecularDynamics,
            1_073_741_824, // 1GB — massive transfer
            10_000.0,      // but short compute
        );
        assert_eq!(assignment, WorkloadAssignment::ThroughputOnly);
    }

    #[test]
    fn redundant_plan() {
        assert_eq!(plan_redundant(), WorkloadAssignment::Redundant);
    }

    #[test]
    fn display_variants() {
        assert_eq!(WorkloadAssignment::PreciseOnly.to_string(), "PreciseOnly");
        assert_eq!(
            WorkloadAssignment::ThroughputOnly.to_string(),
            "ThroughputOnly"
        );
        assert!(WorkloadAssignment::Split {
            precise_fraction: 0.3
        }
        .to_string()
        .contains("30%"));
        assert_eq!(WorkloadAssignment::Redundant.to_string(), "Redundant");
    }

    /// Test helper: bypass `DevicePair` construction (requires GPUs) and
    /// use `PairProfile` directly.
    fn plan_workload_from_profile(
        profile: &PairProfile,
        domain: PhysicsDomain,
        data_bytes: usize,
        compute_us_estimate: f64,
    ) -> WorkloadAssignment {
        match domain {
            PhysicsDomain::Dielectric | PhysicsDomain::Eigensolve => {
                WorkloadAssignment::PreciseOnly
            }
            _ => plan_throughput_or_split_from_profile(profile, data_bytes, compute_us_estimate),
        }
    }

    fn plan_throughput_or_split_from_profile(
        profile: &PairProfile,
        data_bytes: usize,
        compute_us_estimate: f64,
    ) -> WorkloadAssignment {
        if compute_us_estimate < MIN_SPLIT_COMPUTE_US {
            return WorkloadAssignment::ThroughputOnly;
        }
        let transfer_us = profile.transfer_us(data_bytes);
        if transfer_us > TRANSFER_GATE_FRACTION * compute_us_estimate {
            return WorkloadAssignment::ThroughputOnly;
        }
        let frac = profile.precise_split_fraction();
        if !(0.05..=0.95).contains(&frac) {
            return if frac < 0.05 {
                WorkloadAssignment::ThroughputOnly
            } else {
                WorkloadAssignment::PreciseOnly
            };
        }
        WorkloadAssignment::Split {
            precise_fraction: frac,
        }
    }
}
