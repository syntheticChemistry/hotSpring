# ToadStool Handoff: Site-Indexing Evolution + NAK Solver in Rust

**Date:** February 25, 2026
**From:** hotSpring v0.6.11 (biomeGate compute campaign)
**To:** ToadStool / BarraCuda core team
**Priority:** P1 (site-indexing), P2 (NAK solver)
**License:** AGPL-3.0-only

---

## Executive Summary

hotSpring v0.6.11 adopted toadStool's t-major site-indexing convention,
eliminating the incompatibility that forced hotSpring to maintain local shader
copies. This handoff charges toadStool with two evolutionary tasks:

1. **Evolve lattice ops to accept pre-computed neighbor buffers** — so any
   consumer (current or future springs) can use upstream ops regardless of
   their internal memory layout.

2. **Build a Rust-native NAK compiler solver** inside toadStool — making the
   ecosystem portable and self-sufficient without depending on Mesa's release
   cadence for performance-critical compiler fixes.

---

## Part 1: Site-Indexing — What Happened and What's Next

### The Discovery (v0.6.10)

hotSpring attempted to use toadStool's upstream lattice ops (`Su3HmcForce`,
`WilsonPlaquette`, `GpuKineticEnergy`) directly. The HMC immediately blew up:
ΔH ~+4800, negative plaquette — catastrophic.

Root cause: **site-indexing incompatibility.**

| Property | hotSpring (was) | toadStool | hotSpring (now, v0.6.11) |
|----------|----------------|-----------|--------------------------|
| Index formula | `x + Nx*(y + Ny*(z + Nz*t))` | `t*NxNyNz + x*NyNz + y*Nz + z` | `t*NxNyNz + x*NyNz + y*Nz + z` |
| Fastest varying | x | z | z |
| Slowest varying | t | t | t |
| Coordinate tuple | `[x, y, z, t]` | `vec4(t, x, y, z)` | `[x, y, z, t]` |
| Dims tuple | `[Nx, Ny, Nz, Nt]` | `(nt, nx, ny, nz)` | `[Nx, Ny, Nz, Nt]` |

hotSpring v0.6.11 changed `Lattice::site_index()` and `site_coords()` to
match toadStool's z-fastest convention. All 119 unit tests, 3/3 GPU HMC,
6/6 beta scan, and 7/7 streaming validations pass.

### What hotSpring Adopted (Option 1: DONE)

- Changed two functions in `wilson.rs` (lines 44–60)
- Zero WGSL shader changes (all use pre-computed neighbor buffers)
- Zero binary changes (all go through `site_index`/`site_coords`)
- Existing serialized lattice snapshots are incompatible (acceptable — these
  are benchmark artifacts, not production data)

### What toadStool Should Build (Option 2: Neighbor-Buffer Support)

toadStool's lattice ops currently compute neighbors inline from coordinates:

```wgsl
// Current toadStool approach (su3_hmc_force_f64.wgsl)
fn site_to_coords(s: u32) -> vec4<u32> { /* inline decomposition */ }
fn neighbor_plus(site: u32, mu: u32) -> u32 { /* inline coordinate math */ }
```

This works when the consumer stores data in toadStool's exact convention.
But it breaks if a spring uses a different layout (as hotSpring did), or if
the lattice has non-trivial geometry (e.g., domain decomposition for multi-GPU).

**Proposal: Dual-mode lattice ops.**

```rust
// New API surface for upstream ops
pub struct Su3HmcForce {
    // ... existing fields ...
    neighbor_mode: NeighborMode,
}

pub enum NeighborMode {
    /// Compute neighbors from (nt, nx, ny, nz) inline — current behavior
    InlineCoords { nt: u32, nx: u32, ny: u32, nz: u32 },
    /// Use a pre-computed neighbor buffer (8 * vol entries: ±μ for μ=0..3)
    PrecomputedBuffer,
}
```

When `PrecomputedBuffer` is selected:
- The shader reads `nbr[site * 8 + mu * 2 + 0]` (forward) and
  `nbr[site * 8 + mu * 2 + 1]` (backward) from a binding
- The consumer supplies the neighbor buffer at dispatch time
- The lattice dims are no longer needed in the shader params (only volume)

This is how hotSpring already works internally — all 20 local WGSL shaders
use the `nbr` buffer pattern. The pattern is proven across 119 tests and
months of production runs.

**Benefits:**
- Any spring can use upstream ops with any site ordering
- Domain decomposition for multi-GPU becomes trivial (neighbor buffer encodes
  halo boundaries)
- No performance cost (one extra buffer binding, same number of memory reads)

**Implementation estimate:** ~2 sessions. Modify `Su3HmcForce`, `WilsonPlaquette`,
`GpuKineticEnergy`, and their shader variants (f64 + DF64). Add a
`build_neighbor_table(dims, ordering)` utility function.

---

## Part 2: NAK Solver in Rust — Sovereign Compiler Infrastructure

### The Problem

NVK/NAK (Mesa's open-source NVIDIA shader compiler) has five documented
deficiencies that reduce Titan V performance to 3.4% of hardware peak:

| Deficiency | Impact | Current Workaround |
|-----------|--------|-------------------|
| No loop unrolling for bounded loops | 2–3× stalls | `WgslOptimizer` `@unroll_hint` |
| Poor register allocation (high spilling) | 20-40% overhead | `@ilp_region` scheduling |
| No FMA fusion for `a*b+c` patterns | 1.5–2× throughput loss | Pre-fused WGSL |
| Shared memory bank conflicts not avoided | 30-50% bandwidth loss | Manual padding |
| f64 transcendental assertion crash | exp/log unusable | Polynomial polyfill |

toadStool already works around all five via the `WgslOptimizer` (user-space
WGSL→WGSL pass) and `ShaderTemplate` workarounds. But these are band-aids
over a moving target — every Mesa release can change NAK's behavior.

### The Charge: Build a Rust-Native NAK Solver

Rather than patching Mesa's C/Rust code and waiting for upstream releases,
toadStool should evolve its own compiler infrastructure:

#### Phase 1: WGSL→SPIR-V Direct Path (Near-term)

toadStool already uses `naga` for WGSL→SPIR-V compilation. Extend `naga`'s
SPIR-V backend with architecture-aware instruction scheduling:

```rust
// Conceptual API
pub struct SovereignCompiler {
    optimizer: WgslOptimizer,       // existing
    naga_module: naga::Module,      // existing
    scheduler: InstructionScheduler, // NEW
    target: GpuArch,                // Volta, Ampere, Ada, RDNA3...
}

impl SovereignCompiler {
    pub fn compile(&self, wgsl: &str) -> Vec<u32> /* SPIR-V */ {
        let optimized_wgsl = self.optimizer.optimize(wgsl);
        let module = naga::front::wgsl::parse_str(&optimized_wgsl)?;
        let scheduled = self.scheduler.schedule(&module, self.target);
        naga::back::spv::write_vec(&scheduled, &self.spv_options)?
    }
}
```

This keeps the SPIR-V path (compatible with any Vulkan driver) but gives
toadStool control over instruction ordering before NAK/ACO/PTXAS see it.

#### Phase 2: Architecture-Specific Peephole Optimizations (Medium-term)

For known GPU architectures, apply peephole optimizations at the SPIR-V level:

- **FMA fusion**: Detect `OpFMul` + `OpFAdd` chains, replace with `OpExtInst FMA`
- **Loop unrolling**: Detect bounded loops with small trip counts, unroll in SPIR-V
- **Register pressure management**: Limit live variables per basic block to match
  the target's register file size (SM70: 255 registers per thread)
- **f64 transcendental injection**: For NVK targets, replace `OpExtInst Exp/Log`
  with polynomial approximation SPIR-V sequences (bypassing the NAK crash)

#### Phase 3: Native Code Generation (Long-term goal)

The ultimate goal: `WGSL → SPIR-V → PTX → SASS` or `WGSL → SPIR-V → GCN/RDNA`
entirely within toadStool. This is the "sovereign compute" endgame — the
ecosystem generates native GPU machine code without depending on any vendor's
compiler toolchain.

This is ambitious but tractable:
- PTX is documented (NVIDIA's ISA specification is public)
- SASS encoding for SM70/SM80 is partially reversed (by NVIDIA's own `nvdisasm`)
- AMD's GCN/RDNA ISA is fully public
- Rust has excellent tooling for code generation (`cranelift` patterns apply)

### Why This Matters

| Scenario | Without NAK Solver | With NAK Solver |
|----------|-------------------|-----------------|
| Titan V fp64 throughput | 0.25 TFLOPS (3.4% of peak) | Target: 2+ TFLOPS (30%+) |
| Mesa release dependency | Every 3 months, may regress | Independent release cycle |
| New GPU architecture | Wait for Mesa support | Add latency model, ship |
| Cross-vendor parity | NAK vs ACO vs PTXAS differ | Same optimizer for all |
| Consumer GPU DF64 | Works (workaround-based) | Works (natively optimized) |

### Portability Advantage

By solving compiler issues in Rust within toadStool:
- The solutions ship with the crate, not with the OS/driver
- They work on any platform (Linux, macOS, Windows, embedded)
- They're testable in CI without GPU hardware
- They evolve with the `barracuda` crate versioning, not Mesa's

This is more portable than contributing to Mesa (which helps only one driver
at a time) and more sustainable than maintaining out-of-tree Mesa patches.

---

## Part 3: Immediate NVK Issues Still Open

### 3.1 PTE Fault on Large Allocations (Titan V, >1.2 GB combined)

**Status:** OPEN. toadStool has `Workaround::NvkLargeBufferLimit` with a
conservative 1.2 GB cap. The root cause is in nouveau's `drm_gpuvm`
buffer object mapping.

**Action items:**
1. Test with Mesa git HEAD (25.1.5 is 6+ months old)
2. If still failing, build a minimal Vulkan reproducer (single compute
   shader, two 800 MB buffers) and file upstream Mesa/nouveau bug
3. If the kernel module fix is too complex, investigate buffer sub-allocation
   (allocate one large buffer, use offsets internally)

### 3.2 NAK exp/log f64 Assertion

**Status:** WORKAROUND in place (`Workaround::NvkExpF64Crash`). The fix is
in Mesa's `src/nouveau/compiler/nak/from_nir.rs` — 128-bit f64 return values
from transcendental builtins not correctly split for register allocation.

**Action items:**
1. If Phase 1 of the NAK solver includes f64 transcendental injection at
   the SPIR-V level, this workaround becomes permanent and the upstream
   fix becomes optional
2. Consider contributing the minimal reproducer upstream anyway (good
   citizenship)

### 3.3 Titan V Performance Gap

**Status:** MITIGATED by `WgslOptimizer`. NAK on SM70 achieves 3.4% of
fp64 hardware peak vs ~50% for PTXAS on the same hardware.

**Action items:**
1. Phase 2 of the NAK solver (architecture-specific peephole) directly
   targets this
2. In the interim, the `@ilp_region` and `@unroll_hint` annotations in
   the WGSL optimizer provide the best available workaround
3. The NAK-optimized eigensolve shader (`jacobi_eigensolve_nak_opt.wgsl`)
   serves as a reference for what manual optimization achieves on SM70

---

## Part 4: Hardware Available for Testing

toadStool team has:

| Card | VRAM | fp64 | Driver | Role |
|------|------|------|--------|------|
| RTX 3090 (biomeGate) | 24 GB | 1:64 (DF64 → 3.24 TFLOPS) | NVIDIA proprietary | Production, DF64 validation |
| Titan V (biomeGate) | 12 GB | 1:2 (7.45 TFLOPS hw) | NVK/NAK (Mesa 25.1.5) | NAK optimization target |
| AMD consumer (toadStool) | varies | varies | RADV/ACO | Cross-vendor parity |
| NVIDIA consumer (toadStool) | varies | 1:64 | NVK or proprietary | DF64 validation |

The Titan V + RTX 3090 running side-by-side is the ideal test rig:
- Titan V validates NAK compiler improvements (fp64 throughput)
- RTX 3090 validates DF64 core streaming (fp32 throughput)
- Both validate cross-driver parity (NVK vs proprietary)

---

## Part 5: Cross-Spring Evolution Trail

```
hotSpring Exp 012 (Feb 24)
  └─ df64_core.wgsl → toadStool S58 absorption
      └─ toadStool S58-S62: full DF64 HMC pipeline + Fp64Strategy
          └─ hotSpring v0.6.10: imports upstream DF64 math,
             discovers site-indexing incompatibility
              └─ hotSpring v0.6.11: adopts toadStool's t-major indexing
                  └─ THIS HANDOFF: charges toadStool with:
                      (a) neighbor-buffer support in lattice ops
                      (b) Rust-native NAK solver
                      (c) NVK PTE fault investigation

wetSpring (Feb 24)
  └─ NvvmAdaF64Transcendentals workaround → toadStool driver_profile.rs

neuralSpring
  └─ ESN reservoir shaders → candidate for DF64 precision ops
```

---

## Acceptance Criteria

### Site-Indexing (Option 2)
- [ ] `Su3HmcForce` accepts `NeighborMode::PrecomputedBuffer`
- [ ] hotSpring can call upstream force op with its neighbor buffer
- [ ] Physics parity: identical ΔH and plaquette to inline-coords mode
- [ ] Tests cover both modes

### NAK Solver Phase 1
- [ ] `SovereignCompiler` struct with `WgslOptimizer` + `InstructionScheduler`
- [ ] SPIR-V output passes Vulkan validation layer
- [ ] f64 transcendental polyfill injected at SPIR-V level (removes NAK crash)
- [ ] Benchmark: scheduled SPIR-V vs unscheduled on Titan V, measure throughput
