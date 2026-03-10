// SPDX-License-Identifier: AGPL-3.0-only

//! Self-routing precision brain.
//!
//! Built from `HardwareCalibration` results, the brain routes physics
//! workloads to the best available precision tier — never touching a tier
//! that failed probing. Drop into any spring, call `PrecisionBrain::new()`,
//! and it learns the hardware.
//!
//! # Design principles
//!
//! 1. **Probe-first**: Never assume a tier works. The RTX 3090 NVVM failure
//!    demonstrated that a single bad DF64 compilation poisons the entire
//!    wgpu device. The brain probes first, routes second.
//!
//! 2. **Data-driven**: Routing decisions come from measured calibration data,
//!    not from static driver-name heuristics. The brain knows *what actually
//!    works* on this specific GPU, not what *should* work.
//!
//! 3. **Domain-aware**: Physics requirements (FMA sensitivity, precision
//!    floor) are combined with measured hardware capability to pick the
//!    best tier, not just the fastest.
//!
//! 4. **Portable**: Only depends on `GpuF64`, `PrecisionTier`, `PhysicsDomain`.
//!    Works in any spring that can construct a `GpuF64`.

use crate::gpu::GpuF64;
use crate::hardware_calibration::HardwareCalibration;
use crate::precision_routing::{PhysicsDomain, PrecisionRoutingAdvice, PrecisionTier};
pub use crate::precision_routing::HwPrecisionAdvice;

/// Self-routing precision brain for a single GPU.
///
/// Constructed once at startup via `PrecisionBrain::new()`, which runs a
/// safe hardware probe. All subsequent routing calls are O(1) lookups.
pub struct PrecisionBrain {
    /// The calibration data from probing.
    pub calibration: HardwareCalibration,
    /// Pre-computed routing table: domain → tier.
    route_table: [PrecisionTier; 7],
}

impl PrecisionBrain {
    /// Probe the GPU and build the routing table.
    ///
    /// Runs 4 compilation probes (F32, F64, DF64, F64Precise) and uses
    /// the results to build a safe routing table for all physics domains.
    pub fn new(gpu: &GpuF64) -> Self {
        let calibration = HardwareCalibration::probe(gpu);
        eprintln!("[Brain] {calibration}");

        let route_table = build_route_table(&calibration, gpu);

        let brain = Self {
            calibration,
            route_table,
        };

        for (i, domain) in ALL_DOMAINS.iter().enumerate() {
            eprintln!(
                "[Brain] {:?} → {:?}",
                domain, brain.route_table[i]
            );
        }

        brain
    }

    /// Route a physics domain to the best available precision tier.
    ///
    /// O(1) lookup — the routing table was pre-computed at construction.
    #[must_use]
    pub fn route(&self, domain: PhysicsDomain) -> PrecisionTier {
        self.route_table[domain_index(domain)]
    }

    /// Route and return full advice (tier + rationale + FMA safety).
    #[must_use]
    pub fn route_advice(&self, domain: PhysicsDomain, gpu: &GpuF64) -> PrecisionRoutingAdvice {
        let tier = self.route(domain);
        let hw_advice = gpu.driver_profile().precision_routing();
        let (fma_safe, rationale) = domain_requirements(domain, tier);

        PrecisionRoutingAdvice {
            tier,
            fma_safe,
            rationale,
            hw_advice,
        }
    }

    /// Compile a shader at the brain-selected tier for the given domain.
    #[must_use]
    pub fn compile(
        &self,
        gpu: &GpuF64,
        domain: PhysicsDomain,
        shader_source: &str,
        label: &str,
    ) -> wgpu::ComputePipeline {
        let tier = self.route(domain);
        match tier {
            PrecisionTier::F32 => gpu.create_pipeline(shader_source, label),
            PrecisionTier::F64 => gpu.create_pipeline_f64(shader_source, label),
            PrecisionTier::DF64 => gpu.compile_full_df64_pipeline(shader_source, label),
            PrecisionTier::F64Precise => gpu.create_pipeline_f64_precise(shader_source, label),
        }
    }

    /// Check if a specific tier is safe on this hardware.
    #[must_use]
    pub fn tier_safe(&self, tier: PrecisionTier) -> bool {
        self.calibration.tier_safe(tier)
    }

    /// Get the adapter name.
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        &self.calibration.adapter_name
    }
}

impl std::fmt::Display for PrecisionBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PrecisionBrain[{}]:", self.calibration.adapter_name)?;
        for (i, domain) in ALL_DOMAINS.iter().enumerate() {
            writeln!(f, "  {domain:?} → {:?}", self.route_table[i])?;
        }
        Ok(())
    }
}

// ── Routing table construction ──────────────────────────────────────

const ALL_DOMAINS: [PhysicsDomain; 7] = [
    PhysicsDomain::LatticeQcd,
    PhysicsDomain::GradientFlow,
    PhysicsDomain::Dielectric,
    PhysicsDomain::KineticFluid,
    PhysicsDomain::Eigensolve,
    PhysicsDomain::MolecularDynamics,
    PhysicsDomain::NuclearEos,
];

const fn domain_index(domain: PhysicsDomain) -> usize {
    match domain {
        PhysicsDomain::LatticeQcd => 0,
        PhysicsDomain::GradientFlow => 1,
        PhysicsDomain::Dielectric => 2,
        PhysicsDomain::KineticFluid => 3,
        PhysicsDomain::Eigensolve => 4,
        PhysicsDomain::MolecularDynamics => 5,
        PhysicsDomain::NuclearEos => 6,
    }
}

fn build_route_table(cal: &HardwareCalibration, gpu: &GpuF64) -> [PrecisionTier; 7] {
    let hw_advice = gpu.driver_profile().precision_routing();
    let hw_native = matches!(
        hw_advice,
        HwPrecisionAdvice::F64Native | HwPrecisionAdvice::F64NativeNoSharedMem
    );

    ALL_DOMAINS.map(|domain| route_domain(domain, cal, hw_native))
}

fn route_domain(
    domain: PhysicsDomain,
    cal: &HardwareCalibration,
    hw_native: bool,
) -> PrecisionTier {
    match domain {
        // Precision-critical: prefer F64Precise, fall back through tiers
        PhysicsDomain::Dielectric | PhysicsDomain::Eigensolve => {
            if hw_native && cal.tier_safe(PrecisionTier::F64Precise) {
                PrecisionTier::F64Precise
            } else if cal.tier_safe(PrecisionTier::F64) {
                PrecisionTier::F64
            } else if cal.tier_safe(PrecisionTier::DF64) {
                PrecisionTier::DF64
            } else {
                PrecisionTier::F32
            }
        }
        // Moderate precision: prefer F64, fall back to DF64
        PhysicsDomain::GradientFlow | PhysicsDomain::NuclearEos => {
            if cal.tier_safe(PrecisionTier::F64) {
                PrecisionTier::F64
            } else if cal.tier_safe(PrecisionTier::DF64) {
                PrecisionTier::DF64
            } else {
                PrecisionTier::F32
            }
        }
        // Throughput-bound: prefer F64 if fast, else DF64 for throughput
        PhysicsDomain::LatticeQcd
        | PhysicsDomain::KineticFluid
        | PhysicsDomain::MolecularDynamics => {
            if cal.tier_safe(PrecisionTier::F64) {
                // If DF64 is also available and F64 is throttled, prefer DF64
                if cal.tier_safe(PrecisionTier::DF64) && is_f64_throttled(cal) {
                    PrecisionTier::DF64
                } else {
                    PrecisionTier::F64
                }
            } else if cal.tier_safe(PrecisionTier::DF64) {
                PrecisionTier::DF64
            } else {
                PrecisionTier::F32
            }
        }
    }
}

/// Detect whether native F64 is throttled (consumer GPU) by comparing
/// F64 dispatch time to F32. If F64 is >8x slower, it's throttled.
fn is_f64_throttled(cal: &HardwareCalibration) -> bool {
    let f64_us = cal
        .tier_cap(PrecisionTier::F64)
        .map_or(f64::INFINITY, |t| t.dispatch_us);
    let f32_us = cal
        .tier_cap(PrecisionTier::F32)
        .map_or(f64::INFINITY, |t| t.dispatch_us);

    if f32_us <= 0.0 {
        return false;
    }
    f64_us / f32_us > 8.0
}

fn domain_requirements(domain: PhysicsDomain, tier: PrecisionTier) -> (bool, &'static str) {
    match domain {
        PhysicsDomain::Dielectric | PhysicsDomain::Eigensolve => match tier {
            PrecisionTier::F64Precise => (false, "FMA-free f64: cancellation-safe for complex arithmetic"),
            PrecisionTier::F64 => (true, "Native f64 (FMA-free unavailable, acceptable precision)"),
            PrecisionTier::DF64 => (false, "DF64 emulation: ~14 digits, sufficient for most physics"),
            PrecisionTier::F32 => (true, "F32 fallback: reduced precision, validation recommended"),
        },
        PhysicsDomain::GradientFlow | PhysicsDomain::NuclearEos => match tier {
            PrecisionTier::F64 | PrecisionTier::F64Precise => {
                (true, "Native f64 with FMA for moderate precision needs")
            }
            PrecisionTier::DF64 => (true, "DF64 provides sufficient precision for flow/EOS"),
            PrecisionTier::F32 => (true, "F32 fallback: validate energy conservation"),
        },
        _ => match tier {
            PrecisionTier::F64 | PrecisionTier::F64Precise => {
                (true, "Native f64 for compute-bound domains")
            }
            PrecisionTier::DF64 => (true, "DF64 throughput mode: f32 cores for max dispatch rate"),
            PrecisionTier::F32 => (true, "F32 screening/preview mode"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware_calibration::TierCapability;

    fn make_cal(f32_ok: bool, f64_ok: bool, df64_ok: bool, precise_ok: bool) -> HardwareCalibration {
        let mk = |tier, ok: bool| TierCapability {
            tier,
            compiles: ok,
            dispatches: ok,
            transcendentals_safe: ok,
            compile_us: if ok { 100.0 } else { 0.0 },
            dispatch_us: if ok { 50.0 } else { 0.0 },
            probe_ulp: if ok { 0.0 } else { f64::NAN },
        };
        HardwareCalibration {
            adapter_name: "Test GPU".into(),
            tiers: vec![
                mk(PrecisionTier::F32, f32_ok),
                mk(PrecisionTier::F64, f64_ok),
                mk(PrecisionTier::DF64, df64_ok),
                mk(PrecisionTier::F64Precise, precise_ok),
            ],
            has_any_f64: f64_ok || precise_ok,
            df64_safe: df64_ok,
            nvvm_transcendental_risk: false,
        }
    }

    #[test]
    fn full_hw_routes_dielectric_to_precise() {
        let cal = make_cal(true, true, true, true);
        let tier = route_domain(PhysicsDomain::Dielectric, &cal, true);
        assert_eq!(tier, PrecisionTier::F64Precise);
    }

    #[test]
    fn no_precise_routes_dielectric_to_f64() {
        let cal = make_cal(true, true, true, false);
        let tier = route_domain(PhysicsDomain::Dielectric, &cal, true);
        assert_eq!(tier, PrecisionTier::F64);
    }

    #[test]
    fn no_f64_routes_dielectric_to_df64() {
        let cal = make_cal(true, false, true, false);
        let tier = route_domain(PhysicsDomain::Dielectric, &cal, false);
        assert_eq!(tier, PrecisionTier::DF64);
    }

    #[test]
    fn nothing_works_falls_to_f32() {
        let cal = make_cal(true, false, false, false);
        let tier = route_domain(PhysicsDomain::LatticeQcd, &cal, false);
        assert_eq!(tier, PrecisionTier::F32);
    }

    #[test]
    fn throughput_domain_prefers_f64_when_not_throttled() {
        let cal = make_cal(true, true, true, true);
        let tier = route_domain(PhysicsDomain::MolecularDynamics, &cal, true);
        assert_eq!(tier, PrecisionTier::F64);
    }

    #[test]
    fn throttled_f64_routes_md_to_df64() {
        let mut cal = make_cal(true, true, true, true);
        // Make F64 appear throttled: 10x slower than F32
        if let Some(f64_cap) = cal.tiers.iter_mut().find(|t| t.tier == PrecisionTier::F64) {
            f64_cap.dispatch_us = 500.0;
        }
        if let Some(f32_cap) = cal.tiers.iter_mut().find(|t| t.tier == PrecisionTier::F32) {
            f32_cap.dispatch_us = 50.0;
        }
        let tier = route_domain(PhysicsDomain::MolecularDynamics, &cal, true);
        assert_eq!(tier, PrecisionTier::DF64);
    }

    #[test]
    fn domain_index_roundtrip() {
        for (i, &domain) in ALL_DOMAINS.iter().enumerate() {
            assert_eq!(domain_index(domain), i);
        }
    }
}
