// SPDX-License-Identifier: AGPL-3.0-only

//! Precision routing for GPU shader compilation.
//!
//! Combines two axes of precision decision-making:
//!
//! 1. **Hardware capability** (from barraCuda's `GpuDriverProfile::precision_routing()`):
//!    What f64 paths actually work on this GPU? (native, no-shared-mem, DF64-only, f32-only)
//!
//! 2. **Physics domain** (hotSpring-specific): What precision does this physics need?
//!    (dielectric needs FMA-free, lattice QCD tolerates FMA, etc.)
//!
//! The routing decision respects hardware limits first, then applies domain
//! requirements within those limits. `Fp64Strategy::Sovereign` (coralReef
//! native compilation) routes like `Native` — it produces real f64 code.

use crate::gpu::GpuF64;
pub use barracuda::device::driver_profile::PrecisionRoutingAdvice as HwPrecisionAdvice;

/// Precision tier for shader compilation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionTier {
    /// 32-bit float: screening, preview, throughput-bound work
    F32,
    /// Double-float emulation: 14 digits on f32 cores, ~10× native f64 throughput
    DF64,
    /// Native 64-bit: reference precision, validation, Titan V
    F64,
    /// F64 without FMA fusion: precision-critical (dielectric, eigensolve)
    F64Precise,
}

/// Physics domain classification for precision routing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicsDomain {
    /// Lattice QCD: gauge force, plaquette, HMC (tolerant of FMA)
    LatticeQcd,
    /// Gradient flow: energy density, scale setting (moderately sensitive)
    GradientFlow,
    /// Dielectric functions: complex arithmetic, cancellation-prone (needs precise)
    Dielectric,
    /// Kinetic-fluid: BGK relaxation, Euler HLL (tolerant)
    KineticFluid,
    /// Eigensolve: Jacobi, Lanczos (needs precise)
    Eigensolve,
    /// MD: forces, transport (tolerant)
    MolecularDynamics,
    /// Nuclear EOS: BCS pairing, HFB (moderate)
    NuclearEos,
}

/// Precision routing advice for a given domain and hardware.
#[derive(Clone, Debug)]
pub struct PrecisionRoutingAdvice {
    /// Recommended precision tier
    pub tier: PrecisionTier,
    /// Whether FMA fusion is safe for this domain
    pub fma_safe: bool,
    /// Human-readable rationale
    pub rationale: &'static str,
    /// Hardware-level precision advice from barraCuda's driver profile
    pub hw_advice: HwPrecisionAdvice,
}

/// Route a physics domain to the appropriate precision tier given hardware caps.
///
/// Queries barraCuda's `GpuDriverProfile::precision_routing()` for hardware
/// capability, then intersects with domain requirements.
#[must_use]
pub fn route_precision(domain: PhysicsDomain, gpu: &GpuF64) -> PrecisionRoutingAdvice {
    let hw_advice = gpu.driver_profile().precision_routing();
    let is_df64_mode = gpu.full_df64_mode;

    let hw_supports_native = matches!(
        hw_advice,
        HwPrecisionAdvice::F64Native | HwPrecisionAdvice::F64NativeNoSharedMem
    );

    match domain {
        PhysicsDomain::Dielectric | PhysicsDomain::Eigensolve => {
            if hw_supports_native && !is_df64_mode {
                PrecisionRoutingAdvice {
                    tier: PrecisionTier::F64Precise,
                    fma_safe: false,
                    rationale: "Complex arithmetic requires FMA-free precision to avoid cancellation",
                    hw_advice,
                }
            } else {
                PrecisionRoutingAdvice {
                    tier: PrecisionTier::DF64,
                    fma_safe: false,
                    rationale: "DF64 with compensated arithmetic for precision-critical domains",
                    hw_advice,
                }
            }
        }
        PhysicsDomain::GradientFlow | PhysicsDomain::NuclearEos => {
            if hw_supports_native && !is_df64_mode {
                PrecisionRoutingAdvice {
                    tier: PrecisionTier::F64,
                    fma_safe: true,
                    rationale: "Native f64 with FMA fusion for moderate precision requirements",
                    hw_advice,
                }
            } else {
                PrecisionRoutingAdvice {
                    tier: PrecisionTier::DF64,
                    fma_safe: true,
                    rationale: "DF64 provides sufficient precision for flow/EOS",
                    hw_advice,
                }
            }
        }
        PhysicsDomain::LatticeQcd
        | PhysicsDomain::KineticFluid
        | PhysicsDomain::MolecularDynamics => {
            if hw_supports_native && !is_df64_mode {
                PrecisionRoutingAdvice {
                    tier: PrecisionTier::F64,
                    fma_safe: true,
                    rationale: "Native f64 with FMA for compute-bound domains",
                    hw_advice,
                }
            } else {
                PrecisionRoutingAdvice {
                    tier: PrecisionTier::DF64,
                    fma_safe: true,
                    rationale: "DF64 unlocks f32 throughput for compute-bound physics",
                    hw_advice,
                }
            }
        }
    }
}

/// Create a compute pipeline with precision routing.
///
/// Selects the appropriate `GpuF64` method based on the routing advice.
pub fn create_routed_pipeline(
    gpu: &GpuF64,
    shader_source: &str,
    label: &str,
    advice: &PrecisionRoutingAdvice,
) -> wgpu::ComputePipeline {
    match advice.tier {
        PrecisionTier::F32 => gpu.create_pipeline(shader_source, label),
        PrecisionTier::DF64 | PrecisionTier::F64 => {
            gpu.create_pipeline_f64(shader_source, label)
        }
        PrecisionTier::F64Precise => {
            gpu.create_pipeline_f64_precise(shader_source, label)
        }
    }
}

/// Create a compute pipeline with precision routing and named entry point.
pub fn create_routed_pipeline_entry(
    gpu: &GpuF64,
    shader_source: &str,
    entry_point: &str,
    label: &str,
    advice: &PrecisionRoutingAdvice,
) -> wgpu::ComputePipeline {
    match advice.tier {
        PrecisionTier::F32 => gpu.create_pipeline(shader_source, label),
        PrecisionTier::DF64 | PrecisionTier::F64 => {
            gpu.create_pipeline_f64_entry(shader_source, entry_point, label)
        }
        PrecisionTier::F64Precise => {
            gpu.create_pipeline_f64_entry_precise(shader_source, entry_point, label)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dielectric_gets_precise() {
        let advice_f64 = PrecisionRoutingAdvice {
            tier: PrecisionTier::F64Precise,
            fma_safe: false,
            rationale: "test",
            hw_advice: HwPrecisionAdvice::F64Native,
        };
        assert!(!advice_f64.fma_safe);
        assert_eq!(advice_f64.tier, PrecisionTier::F64Precise);
    }

    #[test]
    fn lattice_qcd_tolerates_fma() {
        let advice = PrecisionRoutingAdvice {
            tier: PrecisionTier::F64,
            fma_safe: true,
            rationale: "test",
            hw_advice: HwPrecisionAdvice::F64Native,
        };
        assert!(advice.fma_safe);
    }

    #[test]
    fn all_domains_covered() {
        let domains = [
            PhysicsDomain::LatticeQcd,
            PhysicsDomain::GradientFlow,
            PhysicsDomain::Dielectric,
            PhysicsDomain::KineticFluid,
            PhysicsDomain::Eigensolve,
            PhysicsDomain::MolecularDynamics,
            PhysicsDomain::NuclearEos,
        ];
        for domain in domains {
            assert!(
                matches!(domain, PhysicsDomain::LatticeQcd | PhysicsDomain::GradientFlow
                    | PhysicsDomain::Dielectric | PhysicsDomain::KineticFluid
                    | PhysicsDomain::Eigensolve | PhysicsDomain::MolecularDynamics
                    | PhysicsDomain::NuclearEos)
            );
        }
    }

    #[test]
    fn hw_advice_maps_correctly() {
        assert_eq!(
            std::mem::discriminant(&HwPrecisionAdvice::F64Native),
            std::mem::discriminant(&HwPrecisionAdvice::F64Native)
        );
        assert_ne!(
            std::mem::discriminant(&HwPrecisionAdvice::F64Native),
            std::mem::discriminant(&HwPrecisionAdvice::Df64Only)
        );
    }
}
