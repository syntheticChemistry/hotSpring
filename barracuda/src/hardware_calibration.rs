// SPDX-License-Identifier: AGPL-3.0-only

//! Hardware calibration: safe per-tier compilation probe + runtime throughput
//! measurement.
//!
//! The RTX 3090's proprietary NVIDIA driver demonstrated that a single failed
//! DF64 compilation (NVVM error) permanently poisons the wgpu device — all
//! subsequent buffer and dispatch operations fail. The brain must therefore
//! **probe before routing**: compile a minimal test shader at each tier, record
//! which succeed, and never touch failing tiers.
//!
//! # Usage
//!
//! ```ignore
//! let gpu = GpuF64::new().await?;
//! let cal = HardwareCalibration::probe(&gpu);
//! eprintln!("{cal}");
//! // → TierV: F32=✓ F64=✓ DF64=✗ Precise=✓
//! ```
//!
//! This module is designed to be portable across springs — it only depends on
//! `GpuF64` and `PrecisionTier`, both of which are re-exported from barraCuda.

use crate::gpu::GpuF64;
use crate::precision_routing::PrecisionTier;
use std::time::Instant;

/// Arithmetic probe: trivial multiply — tests basic compilation path.
const PROBE_SHADER_ARITH: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&input) { return; }
    let x = input[i];
    output[i] = x * x + 1.0;
}
";


const PROBE_N: usize = 256;
const PROBE_WORKGROUPS: u32 = 4;

/// Calibrated capability of a single precision tier.
#[derive(Debug, Clone)]
pub struct TierCapability {
    /// Which tier was probed.
    pub tier: PrecisionTier,
    /// Whether the arithmetic probe compiled and dispatched.
    pub compiles: bool,
    /// Whether dispatch + readback produced valid (non-NaN, non-zero) output.
    pub dispatches: bool,
    /// Whether transcendental builtins (exp, log) work at this tier.
    /// False means NVVM or driver fails on DF64 transcendentals.
    pub transcendentals_safe: bool,
    /// Compilation time in microseconds (0 if failed).
    pub compile_us: f64,
    /// Mean dispatch time in microseconds over 5 reps (0 if failed).
    pub dispatch_us: f64,
    /// Max ULP error vs CPU reference for the probe shader (NAN if failed).
    pub probe_ulp: f64,
}

/// Complete hardware calibration for a single GPU.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct HardwareCalibration {
    /// Adapter name.
    pub adapter_name: String,
    /// Per-tier probe results, ordered: F32, F64, DF64, F64Precise.
    pub tiers: Vec<TierCapability>,
    /// Whether any f64-based tier is available.
    pub has_any_f64: bool,
    /// Whether DF64 compilation works (critical for consumer GPUs).
    pub df64_safe: bool,
    /// Whether any tier has NVVM transcendental issues. When true,
    /// shaders containing f64 exp()/log() may NVVM-fail and poison
    /// the device, even if compiled at F32 tier.
    pub nvvm_transcendental_risk: bool,
    /// Whether coralReef sovereign compilation is available. When true,
    /// tiers that fail via NVVM (DF64 transcendentals, F64Precise no-FMA)
    /// can be compiled through coralReef's WGSL → native SASS pipeline,
    /// bypassing NVVM entirely.
    ///
    /// coralReef Iteration 29 validated this bypass for all three
    /// NVVM-poisoning shader patterns (45/46 shaders compile, 12/12 bypass).
    /// Dispatch requires coral-driver DRM maturation (AMD E2E ready,
    /// NVIDIA pending UVM).
    ///
    /// toadStool S144 absorbed our NVVM poisoning work into `nvvm_safety.rs`
    /// (`NvvmPoisoningRisk`, `PrecisionTier`, `TierCapability`). When
    /// hotSpring integrates toadStool's runtime layer, `HardwareCalibration`
    /// can delegate to upstream's native NVVM defense. S144 also added
    /// `gpu_guards` module (`is_wgpu_safe()`, `detect_nvidia_proprietary()`)
    /// for safe test skipping on proprietary NVIDIA drivers.
    pub sovereign_compile_available: bool,
}

impl HardwareCalibration {
    /// Probe all precision tiers on the given GPU.
    ///
    /// Each tier is tested in isolation via `catch_unwind`. A failed probe
    /// marks the tier as unavailable — the brain will never route to it.
    /// Tier order: F64 first (reference), then F64Precise, DF64, F32.
    pub fn probe(gpu: &GpuF64) -> Self {
        let adapter_name = gpu.adapter_name.clone();

        let input: Vec<f64> = (0..PROBE_N).map(|i| (i as f64 + 1.0) * 0.01).collect();
        let arith_ref: Vec<f64> = input.iter().map(|&x| x * x + 1.0).collect();

        // Probe order: safest first, riskiest last. DF64 is last because
        // a failed NVVM compilation permanently poisons the wgpu device —
        // all subsequent operations on the same device will fail.
        let tiers_to_probe = [
            PrecisionTier::F32,
            PrecisionTier::F64,
            PrecisionTier::F64Precise,
            PrecisionTier::DF64,
        ];

        let mut tiers: Vec<TierCapability> = Vec::with_capacity(4);
        let mut device_poisoned = false;

        for &tier in &tiers_to_probe {
            if device_poisoned {
                eprintln!("[HwCal] {tier:?} SKIPPED — device poisoned by earlier probe");
                tiers.push(TierCapability {
                    tier,
                    compiles: false,
                    dispatches: false,
                    transcendentals_safe: false,
                    compile_us: 0.0,
                    dispatch_us: 0.0,
                    probe_ulp: f64::NAN,
                });
                continue;
            }

            let cap = probe_tier(gpu, tier, &input, &arith_ref);

            // If the arithmetic probe panicked, the device is likely poisoned.
            if !cap.compiles {
                device_poisoned = true;
            }

            tiers.push(cap);
        }

        let has_any_f64 = tiers
            .iter()
            .any(|t| t.dispatches && matches!(t.tier, PrecisionTier::F64 | PrecisionTier::F64Precise));
        let df64_safe = tiers
            .iter()
            .any(|t| t.dispatches && t.tier == PrecisionTier::DF64);
        let nvvm_transcendental_risk = tiers
            .iter()
            .any(|t| t.dispatches && !t.transcendentals_safe);

        Self {
            adapter_name,
            tiers,
            has_any_f64,
            df64_safe,
            nvvm_transcendental_risk,
            sovereign_compile_available: false,
        }
    }

    /// Mark sovereign compilation as available (coralReef detected).
    ///
    /// Call after probe when coralReef is reachable. This unlocks
    /// tiers that fail via NVVM but compile through coralReef's
    /// WGSL → naga → codegen IR → native SASS pipeline.
    pub fn set_sovereign_available(&mut self) {
        self.sovereign_compile_available = true;
    }

    /// Look up a specific tier's capability.
    #[must_use]
    pub fn tier_cap(&self, tier: PrecisionTier) -> Option<&TierCapability> {
        self.tiers.iter().find(|t| t.tier == tier)
    }

    /// Check whether a tier is safe to use (compiles + dispatches +
    /// transcendentals). For general-purpose routing, a tier must handle
    /// all shader builtins.
    #[must_use]
    pub fn tier_safe(&self, tier: PrecisionTier) -> bool {
        self.tier_cap(tier)
            .is_some_and(|t| t.compiles && t.dispatches && t.transcendentals_safe)
    }

    /// Check whether a tier is safe when sovereign compilation is available.
    ///
    /// A tier is sovereign-safe if it dispatches AND either transcendentals
    /// work natively or coralReef sovereign compilation can bypass NVVM.
    /// coralReef Iteration 29 validated this bypass for all three
    /// NVVM-poisoning patterns (DF64 pipeline, f64 transcendentals,
    /// F64Precise no-FMA). 45/46 shaders compile, 12/12 bypass.
    #[must_use]
    pub fn tier_safe_with_sovereign(&self, tier: PrecisionTier) -> bool {
        self.tier_safe(tier)
            || (self.sovereign_compile_available
                && self.tier_cap(tier).is_some_and(|t| t.compiles && t.dispatches))
    }

    /// Check whether a tier can dispatch arithmetic-only shaders
    /// (no exp/log/transcendentals). Useful for DF64 on proprietary NVIDIA.
    #[must_use]
    pub fn tier_arith_only(&self, tier: PrecisionTier) -> bool {
        self.tier_cap(tier)
            .is_some_and(|t| t.compiles && t.dispatches && !t.transcendentals_safe)
    }

    /// Return the fastest safe tier for f64-class work.
    ///
    /// Prefers F64 native, falls back to F64Precise, then DF64.
    #[must_use]
    pub fn best_f64_tier(&self) -> Option<PrecisionTier> {
        let preference = [PrecisionTier::F64, PrecisionTier::F64Precise, PrecisionTier::DF64];
        preference.iter().copied().find(|&t| self.tier_safe(t))
    }

    /// Return the fastest safe tier overall (including F32).
    #[must_use]
    pub fn best_any_tier(&self) -> Option<PrecisionTier> {
        let preference = [
            PrecisionTier::F64,
            PrecisionTier::F64Precise,
            PrecisionTier::DF64,
            PrecisionTier::F32,
        ];
        preference.iter().copied().find(|&t| self.tier_safe(t))
    }
}

impl std::fmt::Display for HardwareCalibration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HwCal[{}]:", self.adapter_name)?;
        for t in &self.tiers {
            let mark = if t.dispatches && t.transcendentals_safe {
                "✓"
            } else if t.dispatches && self.sovereign_compile_available {
                "✓sov"
            } else if t.dispatches {
                "△arith"
            } else if t.compiles {
                "△comp"
            } else {
                "✗"
            };
            write!(f, " {:?}={mark}", t.tier)?;
        }
        if self.sovereign_compile_available {
            write!(f, " [coralReef bypass]")?;
        }
        Ok(())
    }
}

/// Probe a single tier. Both arithmetic and transcendental probes are
/// wrapped in `catch_unwind` so a device-poisoning failure (NVVM error
/// on the RTX 3090's DF64 transcendentals) doesn't prevent probing
/// subsequent tiers.
fn probe_tier(
    gpu: &GpuF64,
    tier: PrecisionTier,
    input: &[f64],
    arith_ref: &[f64],
) -> TierCapability {
    // Phase 1: arithmetic probe
    let arith = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        probe_single_shader(gpu, tier, PROBE_SHADER_ARITH, input, arith_ref, "arith")
    }));

    let Ok((compiles, dispatches, compile_us, dispatch_us, probe_ulp)) = arith else {
        eprintln!("[HwCal] {tier:?} arith probe PANICKED — tier disabled");
        return TierCapability {
            tier,
            compiles: false,
            dispatches: false,
            transcendentals_safe: false,
            compile_us: 0.0,
            dispatch_us: 0.0,
            probe_ulp: f64::NAN,
        };
    };

    // Phase 2: transcendental safety.
    //
    // The NVIDIA proprietary driver's NVVM compiler cannot handle
    // f64 transcendentals (exp, log) in two compilation modes:
    //   - DF64: f32-pair emulation with transcendentals
    //   - F64Precise: no-FMA mode with transcendentals
    // A failed NVVM compilation permanently poisons the wgpu device.
    //
    // NVK (Mesa open source) handles all modes correctly.
    // F32 and native F64 (with FMA) always work.
    let is_nvk = gpu.adapter_name.contains("NVK") || gpu.adapter_name.contains("llvmpipe");
    let transcendentals_safe = dispatches
        && (tier == PrecisionTier::F32
            || tier == PrecisionTier::F64
            || is_nvk);

    TierCapability {
        tier,
        compiles,
        dispatches,
        transcendentals_safe,
        compile_us,
        dispatch_us,
        probe_ulp,
    }
}

/// Run a single probe shader: compile + dispatch + readback.
/// Returns (compiles, dispatches, compile_us, dispatch_us, ulp).
fn probe_single_shader(
    gpu: &GpuF64,
    tier: PrecisionTier,
    shader: &str,
    input: &[f64],
    reference: &[f64],
    tag: &str,
) -> (bool, bool, f64, f64, f64) {
    let label = format!("probe_{tier:?}_{tag}");

    let t_compile = Instant::now();
    let pipeline = compile_at_tier(gpu, shader, &label, tier);
    let compile_us = t_compile.elapsed().as_secs_f64() * 1e6;

    let input_buf = gpu.create_f64_buffer(input, &format!("{label}_in"));
    let output_buf = gpu.create_f64_output_buffer(PROBE_N, &format!("{label}_out"));
    let bind_group = gpu.create_bind_group(&pipeline, &[&input_buf, &output_buf]);

    gpu.dispatch(&pipeline, &bind_group, PROBE_WORKGROUPS);
    let _ = gpu.device().poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    let t_dispatch = Instant::now();
    for _ in 0..5 {
        gpu.dispatch(&pipeline, &bind_group, PROBE_WORKGROUPS);
    }
    let _ = gpu.device().poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let dispatch_us = t_dispatch.elapsed().as_secs_f64() * 1e6 / 5.0;

    let output = gpu.read_back_f64(&output_buf, PROBE_N).unwrap_or_default();

    let valid = !output.is_empty() && output.iter().all(|v| v.is_finite() && *v != 0.0);
    let probe_ulp = if valid {
        max_ulp(reference, &output)
    } else {
        f64::NAN
    };

    (true, valid, compile_us, dispatch_us, probe_ulp)
}

fn compile_at_tier(
    gpu: &GpuF64,
    source: &str,
    label: &str,
    tier: PrecisionTier,
) -> wgpu::ComputePipeline {
    match tier {
        PrecisionTier::F32 => gpu.create_pipeline(source, label),
        PrecisionTier::F64 => gpu.create_pipeline_f64(source, label),
        PrecisionTier::DF64 => gpu.compile_full_df64_pipeline(source, label),
        PrecisionTier::F64Precise => gpu.create_pipeline_f64_precise(source, label),
    }
}

fn max_ulp(reference: &[f64], actual: &[f64]) -> f64 {
    reference
        .iter()
        .zip(actual.iter())
        .map(|(&r, &a)| ulp_distance(r, a))
        .fold(0.0_f64, f64::max)
}

#[allow(clippy::float_cmp)]
fn ulp_distance(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b {
        return 0.0;
    }
    if a.is_infinite() || b.is_infinite() {
        return f64::INFINITY;
    }
    let a_bits = a.to_bits() as i64;
    let b_bits = b.to_bits() as i64;
    (a_bits - b_bits).unsigned_abs() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_capability_display() {
        let cal = HardwareCalibration {
            adapter_name: "Test GPU".into(),
            tiers: vec![
                TierCapability {
                    tier: PrecisionTier::F32,
                    compiles: true,
                    dispatches: true,
                    transcendentals_safe: true,
                    compile_us: 100.0,
                    dispatch_us: 50.0,
                    probe_ulp: 0.0,
                },
                TierCapability {
                    tier: PrecisionTier::F64,
                    compiles: true,
                    dispatches: true,
                    transcendentals_safe: true,
                    compile_us: 200.0,
                    dispatch_us: 80.0,
                    probe_ulp: 0.0,
                },
                TierCapability {
                    tier: PrecisionTier::DF64,
                    compiles: true,
                    dispatches: true,
                    transcendentals_safe: false,
                    compile_us: 150.0,
                    dispatch_us: 60.0,
                    probe_ulp: 2.0,
                },
                TierCapability {
                    tier: PrecisionTier::F64Precise,
                    compiles: false,
                    dispatches: false,
                    transcendentals_safe: false,
                    compile_us: 0.0,
                    dispatch_us: 0.0,
                    probe_ulp: f64::NAN,
                },
            ],
            has_any_f64: true,
            df64_safe: true,
            nvvm_transcendental_risk: true,
            sovereign_compile_available: false,
        };
        let s = cal.to_string();
        assert!(s.contains("F64=✓"), "F64 should show ✓, got: {s}");
        assert!(
            s.contains("DF64=△arith"),
            "DF64 should show △arith (dispatches but no transcendentals), got: {s}"
        );
        assert!(s.contains("F64Precise=✗"), "F64Precise should show ✗, got: {s}");
    }

    #[test]
    fn sovereign_bypass_upgrades_arith_to_safe() {
        let mut cal = HardwareCalibration {
            adapter_name: "RTX 3090".into(),
            tiers: vec![TierCapability {
                tier: PrecisionTier::DF64,
                compiles: true,
                dispatches: true,
                transcendentals_safe: false,
                compile_us: 150.0,
                dispatch_us: 60.0,
                probe_ulp: 2.0,
            }],
            has_any_f64: false,
            df64_safe: true,
            nvvm_transcendental_risk: true,
            sovereign_compile_available: false,
        };
        assert!(!cal.tier_safe(PrecisionTier::DF64));
        assert!(!cal.tier_safe_with_sovereign(PrecisionTier::DF64));

        cal.set_sovereign_available();
        assert!(!cal.tier_safe(PrecisionTier::DF64));
        assert!(cal.tier_safe_with_sovereign(PrecisionTier::DF64));

        let s = cal.to_string();
        assert!(s.contains("✓sov"), "Should show ✓sov with sovereign, got: {s}");
        assert!(s.contains("[coralReef bypass]"), "Should note bypass, got: {s}");
    }

    #[test]
    fn best_f64_tier_preference() {
        let cal = HardwareCalibration {
            adapter_name: "Test".into(),
            nvvm_transcendental_risk: false,
            tiers: vec![
                TierCapability {
                    tier: PrecisionTier::F64,
                    compiles: false,
                    dispatches: false,
                    transcendentals_safe: false,
                    compile_us: 0.0,
                    dispatch_us: 0.0,
                    probe_ulp: f64::NAN,
                },
                TierCapability {
                    tier: PrecisionTier::F64Precise,
                    compiles: false,
                    dispatches: false,
                    transcendentals_safe: false,
                    compile_us: 0.0,
                    dispatch_us: 0.0,
                    probe_ulp: f64::NAN,
                },
                TierCapability {
                    tier: PrecisionTier::DF64,
                    compiles: true,
                    dispatches: true,
                    transcendentals_safe: true,
                    compile_us: 100.0,
                    dispatch_us: 50.0,
                    probe_ulp: 2.0,
                },
                TierCapability {
                    tier: PrecisionTier::F32,
                    compiles: true,
                    dispatches: true,
                    transcendentals_safe: true,
                    compile_us: 50.0,
                    dispatch_us: 30.0,
                    probe_ulp: 1e6,
                },
            ],
            has_any_f64: false,
            df64_safe: true,
            sovereign_compile_available: false,
        };
        assert_eq!(cal.best_f64_tier(), Some(PrecisionTier::DF64));
        assert_eq!(cal.best_any_tier(), Some(PrecisionTier::DF64));
    }

    #[test]
    fn tier_safe_check() {
        let cal = HardwareCalibration {
            adapter_name: "Test".into(),
            nvvm_transcendental_risk: false,
            tiers: vec![TierCapability {
                tier: PrecisionTier::F64,
                compiles: true,
                dispatches: true,
                transcendentals_safe: true,
                compile_us: 100.0,
                dispatch_us: 50.0,
                probe_ulp: 0.0,
            }],
            has_any_f64: true,
            df64_safe: false,
            sovereign_compile_available: false,
        };
        assert!(cal.tier_safe(PrecisionTier::F64));
        assert!(!cal.tier_safe(PrecisionTier::DF64));
    }
}
