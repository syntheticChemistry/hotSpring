# hotSpring → toadStool/barraCuda: Precision Brain + NVVM Device Poisoning

**Date:** March 10, 2026
**From:** hotSpring v0.6.25
**To:** toadStool / barraCuda team
**License:** AGPL-3.0-only
**Covers:** v0.6.24 → v0.6.25

---

## Executive Summary

- **Self-routing precision brain** (`PrecisionBrain`) discovers hardware at startup and routes physics workloads to the best safe precision tier — O(1) lookups, no static heuristics
- **NVIDIA proprietary driver NVVM poisoning** discovered: a single failed DF64 or F64Precise shader compilation permanently invalidates the wgpu device for the rest of the process
- **3-tier precision evaluation harness** benchmarked both GPUs (Titan V NVK + RTX 3090 proprietary) across all precision tiers with full pipeline E2E and dual-card cooperative patterns
- **Hardware calibration** probes each tier safely (F32 → F64 → F64Precise → DF64, safest to riskiest) without poisoning the device; transcendental safety inferred from driver identity
- New modules: `hardware_calibration.rs`, `precision_brain.rs`, `precision_eval.rs`, `transfer_eval.rs`, `pipeline_eval.rs`, `dual_pipeline_eval.rs`, `bench_precision_eval` binary

---

## Part 1: NVVM Device Poisoning — Critical Finding

### The Problem

On the NVIDIA proprietary driver (not NVK), certain WGSL shader compilations trigger NVVM failures that **permanently invalidate the wgpu device**. Once poisoned, ALL subsequent buffer creation, dispatch, and readback operations on that device panic with `"Buffer is invalid"`.

### Affected Compilation Paths

| Tier | Arithmetic (x*x+1) | Transcendentals (exp/log) |
|------|:---:|:---:|
| F32 (create_pipeline) | ✓ | ✗ if shader uses f64 types |
| F64 native | ✓ | ✓ |
| F64Precise (no FMA) | ✓ | **✗ — NVVM fails** |
| DF64 (compile_full_df64_pipeline) | ✓ | **✗ — NVVM fails** |

### Root Cause

The NVIDIA proprietary driver sends WGSL through naga→SPIR-V→NVVM. The NVVM compiler cannot handle:
1. `exp()`/`log()` on f64 values in DF64 mode (the f32-pair rewrite confuses NVVM's builtin resolution)
2. `exp()`/`log()` in F64Precise mode (no-FMA compilation flags break NVVM's transcendental implementation)
3. Any shader containing `array<f64>` with f64 transcendentals, even when compiled via the "F32" path (the f64 types still go through NVVM)

### Impact

The NVVM failure produces `"NVVM compilation failed: 1"` and then the device enters an error state. Every subsequent wgpu operation (buffer creation, dispatch, readback) on the same device panics. The only recovery is process restart.

### Recommendation for toadStool/barraCuda

1. **Never attempt DF64 or F64Precise compilation of shaders containing f64 transcendentals on proprietary NVIDIA** — the device death is unrecoverable
2. **Consider a disposable-device probe**: create a throwaway `WgpuDevice` to test risky compilations, discard it, keep the main device clean
3. **Probe order matters**: always test safest tiers first (F32 → F64 → F64Precise → DF64)
4. **NVK handles everything correctly** — all 4 tiers with transcendentals pass on NVK (Mesa)
5. **The brain's heuristic**: infer transcendental safety from adapter name (`"NVK"` = safe, otherwise DF64/F64Precise = unsafe for transcendentals). This works but is not data-driven — a disposable-device probe would be better

---

## Part 2: Hardware Calibration (`hardware_calibration.rs`)

Safe per-tier probe that runs at startup. Probes F32 → F64 → F64Precise → DF64 (safest to riskiest). If a probe panics, the device may be poisoned — all subsequent probes are skipped.

### `TierCapability`

```rust
pub struct TierCapability {
    pub tier: PrecisionTier,
    pub compiles: bool,
    pub dispatches: bool,
    pub transcendentals_safe: bool,  // inferred from driver, not probed
    pub compile_us: f64,
    pub dispatch_us: f64,
    pub probe_ulp: f64,
}
```

### `HardwareCalibration`

```rust
pub struct HardwareCalibration {
    pub adapter_name: String,
    pub tiers: Vec<TierCapability>,
    pub has_any_f64: bool,
    pub df64_safe: bool,
    pub nvvm_transcendental_risk: bool,  // true if any tier has broken transcendentals
}
```

### Measured Results

| GPU | F32 | F64 | F64Precise | DF64 |
|-----|-----|-----|------------|------|
| Titan V (NVK) | ✓ | ✓ | ✓ | ✓ |
| RTX 3090 (proprietary) | ✓ | ✓ | △arith | △arith |

**toadStool action:** Consider absorbing `HardwareCalibration` into barraCuda's device initialization. Every spring needs this — a single failed compilation shouldn't kill the device for the entire process.

---

## Part 3: Precision Brain (`precision_brain.rs`)

Self-routing brain built from calibration results. Constructed once at startup, O(1) routing thereafter.

### Routing Logic

1. **Precision-critical** (Dielectric, Eigensolve): F64Precise → F64 → DF64 → F32
2. **Moderate precision** (GradientFlow, NuclearEos): F64 → DF64 → F32
3. **Throughput-bound** (LatticeQcd, MD, KineticFluid): F64 (or DF64 if F64 throttled) → F32

### Key Innovation: F64 Throttle Detection

```rust
fn is_f64_throttled(cal: &HardwareCalibration) -> bool {
    f64_us / f32_us > 8.0  // consumer GPUs have 1:64 FP64:FP32
}
```

When F64 is >8x slower than F32 and DF64 is available, throughput-bound workloads route to DF64 for higher throughput.

### Measured Routing

| Domain | Titan V | RTX 3090 |
|--------|---------|----------|
| LatticeQcd | F64 | F64 |
| Dielectric | F64Precise | F64 |
| Eigensolve | F64Precise | F64 |
| MolecularDynamics | F64 | F64 |

**toadStool action:** The brain is designed to be portable — it only needs `GpuF64` and `PrecisionTier`. Consider absorbing it as `barracuda::precision::PrecisionBrain` so all springs get safe self-routing.

---

## Part 4: Benchmark Results

### Transfer Profiles

| Metric | Titan V (NVK, PCIe 3.0) | RTX 3090 (proprietary, PCIe 4.0) |
|--------|--------------------------|-----------------------------------|
| Upload peak | 2.2 GB/s @ 1MB | 6.6 GB/s @ 1MB |
| Readback peak | 4.4 GB/s @ 16MB | 4.3 GB/s @ 4MB |
| Dispatch overhead | 6 us | 3 us |
| Reduce scalar | 1915 us | 470 us |

### Pipeline End-to-End

| Pipeline | Titan V F64 | Titan V DF64 | RTX 3090 F64 | RTX 3090 DF64 |
|----------|-------------|--------------|--------------|---------------|
| HMC 4^4 (10 traj) | 275 ms | 269 ms | **196 ms** | **195 ms** |
| BCS 4096×20 | 3.6 ms | 2.7 ms | **2.5 ms** | **1.5 ms** |
| Dielectric (Mermin) | 22 ms | — | 22 ms | — |

### Dual-Card Cooperative

| Pattern | Wall | vs Single | Detail |
|---------|------|-----------|--------|
| Split BCS (30/70) | 4.4 ms | 0.4x | 1228 precise + 2868 throughput |
| Split HMC (force/valid) | 45 ms | 2.2x | throughput computes, precise validates |
| Redundant HMC | 228 ms | 1.0x | max plaquette diff: 2.88e-9 |
| PCIe roundtrip | 1.7 ms | — | 512KB, 1.2 GB/s effective |

---

## Part 5: New Modules (Absorption Candidates)

| Module | Purpose | Tier |
|--------|---------|------|
| `hardware_calibration.rs` | Safe per-tier probe + capability mask | **P0** — every spring needs this |
| `precision_brain.rs` | Self-routing brain from calibration | **P0** — portable, O(1) routing |
| `precision_eval.rs` | Per-shader precision/throughput profiler | P1 — benchmarking utility |
| `transfer_eval.rs` | PCIe bandwidth profiler | P1 — performance characterization |
| `pipeline_eval.rs` | Full physics pipeline E2E profiler | P2 — hotSpring-specific |
| `dual_pipeline_eval.rs` | Cooperative dual-card patterns | P2 — needs DevicePair |
| `bench_precision_eval` | Master benchmark binary | P2 — orchestrator |

---

## Part 6: Remaining Debt for Other Teams

1. **NVVM f64 transcendental support** — the proprietary NVIDIA driver cannot compile exp/log at DF64 or F64Precise. This limits consumer GPU throughput for physics domains that use transcendentals. NVK handles it correctly. Unlocking this on the proprietary driver would give 3090-class cards full DF64 transcendental throughput.

2. **Disposable-device probing** — the current brain infers transcendental safety from the adapter name. A more robust approach would create a throwaway wgpu device for the risky DF64 transcendental probe, discard it, and keep the main device clean. This requires `GpuF64::from_adapter_name()` or equivalent.

3. **F64 throttle detection at scale** — the current probe uses 256-element test vectors. Larger workloads may show different F64/F32 ratios. A more thorough calibration would use physics-representative workload sizes.

4. **3090 HMC uses Hybrid (DF64) strategy** — the HMC pipeline on the 3090 auto-selects `Fp64Strategy::Hybrid` which routes gauge force through DF64. This works for arithmetic-only shaders but would fail for any shader with transcendentals. The brain should gate this.

---

## Action Items

- **toadStool action:** Review NVVM poisoning finding. Consider adding a `device_health_check()` to `WgpuDevice` that detects poisoned state before operations.
- **barraCuda action:** Consider absorbing `HardwareCalibration` and `PrecisionBrain` as core primitives. Every spring reinvents this wheel.
- **barraCuda action:** The `compile_full_df64_pipeline()` path should have a safety gate — if the adapter is NVIDIA proprietary, refuse to compile shaders containing f64 transcendentals (or at minimum, warn).
- **coralReef action:** The sovereign WGSL→native path may bypass the NVVM issue entirely (compiling directly to SASS). Worth testing.
