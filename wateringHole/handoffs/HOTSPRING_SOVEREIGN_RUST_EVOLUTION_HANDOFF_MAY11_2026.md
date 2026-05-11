# Sovereign Rust Evolution — From Jelly Strings to Pure Primals

**Date:** May 11, 2026
**From:** hotSpring sovereign compute sprint
**To:** coralReef team, toadStool team, barracuda team, springs teams, primalPsing audit
**Scope:** Warm-catch pipeline evolution, ELF patching, era-agnostic sovereign compute

---

## Executive Summary

All three local GPUs — RTX 5060 (GB206/Blackwell), Titan V (GV100/Volta), and
Tesla K80 (GK210/Kepler) — are now sovereign. The warm-catch mechanism that
resolved the last two GPU sovereignty gaps (GAP-HS-073 and GAP-HS-076) was
initially proven via shell scripts and Python, then elevated to pure Rust inside
coralReef. This handoff documents what was built, the patterns that emerged, and
what upstream teams should absorb.

---

## 1. The Warm-Catch Technique

### Problem

Modern NVIDIA GPUs require complex initialization (memory training, firmware
loading, engine gating) that open-source drivers handle but sovereign VFIO
pipelines cannot replicate from scratch — yet. For Kepler (K80) and Volta
(Titan V), the GPU silicon sits behind initialization barriers that our
SovereignInit pipeline cannot cross without firmware cooperation.

### Solution: Binary-Patched nouveau

Temporarily load a modified `nouveau.ko` kernel module where 4 teardown
functions are NOP'd at the machine-code level:

```
gf100_gr_fini       → mov eax, 0; ret  (teardown-safe)
nvkm_pmu_fini       → mov eax, 0; ret
nvkm_mc_disable     → mov eax, 0; ret
nvkm_fifo_fini      → mov eax, 0; ret
```

nouveau initializes the GPU fully (memory training, FECS boot, GPC activation),
but when unbound, it cannot tear down what it built. The warm state persists
across the vfio-pci rebind.

### Results

| GPU | Gap | Memory | FECS | GPCs | Status |
|-----|-----|--------|------|------|--------|
| K80 (GK210) | GAP-HS-076 | 12 GiB GDDR5 trained | RUNNING (0x00060005) | 5 active | RESOLVED |
| Titan V (GV100) | GAP-HS-073 | 12 GB HBM2 (BIOS POST) | RUNNING (0x0c060006) | 1 active | RESOLVED |
| RTX 5060 (GB206) | — | GDDR6 (native VFIO) | N/A (GSP era) | Direct dispatch | Already sovereign |

---

## 2. Pure Rust Pipeline (coralReef)

### New Crate Components

| Component | Path | Purpose |
|-----------|------|---------|
| `KmodPatcher` | `coral-driver/src/tools/elf_patcher.rs` | Pure Rust ELF binary patcher using the `object` crate. Replaces `patch_nouveau_teardown.py`. |
| `WarmStateSnapshot` | `coral-driver/src/vfio/warm_probe.rs` | Standalone GPU warm-state probe (PMC, PRAMIN, FECS, GPC registers). |
| `warm_catch` handler | `coral-ember/src/ipc/handlers_warm_catch.rs` | `ember.warm_catch` JSON-RPC: patch → swap → settle → probe → swap back. |
| `warm_catch_pre_check` | `coral-driver/src/vfio/sovereign_init.rs` | Auto-detect cold GPU + available warm-catch infrastructure. |
| `coralctl warm-catch` | `coral-glowplug/.../coralctl_main_linux.rs` | CLI entry point routing through glowplug → ember RPC. |

### Key Design Decisions

1. **ELF patching in Rust (`object` crate)**: Zero subprocess calls. Vendor-
   agnostic via `PatchTarget` struct — add new targets for any kernel module,
   any driver, any architecture.

2. **Era-aware settle durations**: `MemoryType` enum determines how long to
   wait after nouveau loads before probing warm state. GDDR5=10s (K80),
   HBM2=12s (Titan V), GDDR6=8s (RTX 5060).

3. **RAII module restoration**: `ModuleCleanupGuard` ensures stock `nouveau.ko`
   is restored in `/lib/modules/...` even on panic. The patched binary never
   persists beyond the warm-catch operation.

4. **Separation of concerns**: `sovereign_init.rs` reports whether warm-catch
   is needed but does not execute it (it has no driver-swap authority).
   `coral-ember` owns driver swaps and orchestrates the full pipeline.

5. **`--dry-run` mode**: Validates ELF patching logic without loading any
   kernel modules. Safe for CI and pre-flight checks.

### API Surface

```rust
// ELF Patching
KmodPatcher::default_nouveau_targets()  // 4 NVIDIA teardown NOP targets
KmodPatcher::with_targets(custom)       // Any kernel module + custom targets
KmodPatcher::patch(&self, input, output) -> PatchResult

// Warm State
WarmStateSnapshot { pmc_enable, pramin_sentinel, fecs_mc, gpc_mask, ... }
probe_warm_state(bar0: &MappedBar) -> WarmStateSnapshot

// Pre-check
warm_catch_pre_check(bar0) -> (is_cold: bool, warm_catch_available: bool)

// CLI
coralctl warm-catch <BDF> [--settle <secs>] [--memory-type gddr5|hbm2|gddr6] [--dry-run]
```

---

## 3. For coralReef Team

### Absorption Targets

- **`elf_patcher.rs`** is self-contained. Consider extracting to a shared
  `coral-tools` crate if other primals need kernel module patching.
- **`warm_probe.rs`** exports `WarmStateSnapshot` which replaces ad-hoc
  register reads scattered across experiment binaries. All warm-state queries
  should route through this module.
- **`handlers_warm_catch.rs`** demonstrates the ember RPC handler pattern
  with `ModuleCleanupGuard`. This RAII pattern should be adopted for any
  operation that modifies system state (firmware uploads, PCI resets, etc.).

### Known Limitations

1. **nouveau must recognize the GPU**: K80 (GK210) requires an upstream
   one-line patch (`case 0x0f2: device->chip = &nvf1_chipset;`) or a
   locally-patched nouveau.
2. **Root/CAP_SYS_MODULE required**: Loading kernel modules requires
   privileges. The ember service already runs as root.
3. **Single GPU per warm-catch**: The pipeline patches one module copy per
   invocation. Concurrent warm-catch on multiple GPUs is safe (separate
   ember instances per BDF).

---

## 4. For toadStool Team

### What to Absorb

toadStool currently dispatches compute workloads via wgpu (vendor path) or
coralReef RPC (sovereign path). The warm-catch pipeline extends the sovereign
path to legacy GPUs:

```
[Before]  toadStool → coralReef → sovereign_init → FAIL (cold GPU)
[After]   toadStool → coralReef → warm_catch → sovereign_init → PASS
```

**Recommendation**: When toadStool detects a cold legacy GPU (via
`warm_catch_pre_check()`), it should orchestrate `coralctl warm-catch`
before attempting sovereign dispatch. This can be wired into the existing
`GpuLifecycle` trait or a new `WarmCatchLifecycle` trait.

### Compute Dispatch Readiness

| GPU | Compiler | Dispatch | Status |
|-----|----------|----------|--------|
| RTX 5060 | SM120 (Blackwell) | QMD v5.0 + UVM | ✅ PROVEN (8/8 QCD/HMC/MD) |
| Titan V | SM70 (Volta) | QMD v2.x + GPFIFO | FECS running — dispatch next |
| K80 | SM35 (Kepler) | PIO FECS | GPCs active — dispatch next |

---

## 5. For Springs Teams

### Pattern: Jelly Strings → Pure Rust → Sovereign Primals

The evolution arc observed in hotSpring:

1. **Jelly Strings**: Shell scripts and Python prove the concept
   (fast iteration, high brittleness)
2. **Pure Rust**: Elevate proven concepts into typed, tested crate code
   (medium iteration, high robustness)
3. **Sovereign Primals**: Compose Rust modules into IPC-discoverable services
   via NUCLEUS patterns (low iteration, maximum composability)

This arc applies to any spring discovering new hardware interaction patterns.
Write the ugly shell first, validate it works, then elevate.

### Composition Pattern for Hardware

```
Spring (physics/AI/crypto workload)
  └─ toadStool (GPU dispatch abstraction)
      └─ coralReef (sovereign driver stack)
          ├─ coral-glowplug (fleet orchestrator)
          ├─ coral-ember (per-device lifecycle + warm-catch)
          └─ coral-driver (BAR0 MMIO + ELF tools)
```

Springs should never touch PCIe/MMIO/sysfs directly. All hardware interaction
routes through coralReef's RPC surface. If a spring needs a new GPU operation,
it should be proposed as a new ember RPC method.

---

## 6. For primalPsing Audit

### Items to Verify

1. **`object` crate version**: coral-driver uses `object = "0.36"`. The
   `rustc`-bundled version may be `0.37`. If this causes resolution conflicts,
   bump coral-driver's dependency.
2. **`livepatch_nvkm_mc_reset.c`**: Superseded by binary patching. The source
   remains in `scripts/livepatch/` for fossil record. Build artifacts (`.ko`,
   `.mod.c`, `.cmd`, `Module.symvers`) should be gitignored.
3. **Experiment journal gap**: Experiments 182-184 are referenced in
   `experiments/README.md` but journal `.md` files may be stub entries
   (inline in the table only, no standalone file). Verify and document.
4. **`exp169_pmu_boot.rs`**: Default `--firmware` path is `/tmp/gv100_pmu_70k_B.bin`.
   This is brittle on shared systems — consider `std::env::temp_dir()`.

### Archived Scripts (fossil record in `scripts/archive/`)

| Script | Replaced By |
|--------|-------------|
| `patch_nouveau_teardown.py` | `coral-driver::tools::elf_patcher` |
| `k80_warm_catch.sh` | `coralctl warm-catch <BDF> --memory-type gddr5` |
| `titanv_warm_handoff.sh` | `coralctl warm-catch <BDF> --memory-type hbm2` |
| `bpf_warm_catch_guard.py` | Not functional (BPF annotations missing in kernel). Historical only. |
| `livepatch_nvkm_mc_reset.c` | Binary ELF patching (kernel 6.17 rejects out-of-tree livepatch). Historical only. |

---

## 7. Era-Agnostic Sovereign Compute Roadmap

The long-term sovereign compute vision:

```
WGSL shader
  → coralReef compiler (SM35/SM70/SM120/GFX10.3/...)
    → coralReef driver (sovereign init + warm-catch)
      → GPU hardware (any vendor, any era)
        → results (readback via VFIO BAR0)
```

**Current coverage**:
- NVIDIA: SM35 (Kepler), SM70 (Volta), SM120 (Blackwell) — compiler + dispatch
- AMD: GFX10.3 (RDNA2) — compiler proven, dispatch via sovereign driver

**Next frontiers**:
- Sovereign dispatch on Titan V (FECS is running — wire GPFIFO/QMD)
- Sovereign dispatch on K80 (GPCs active — wire Kepler PIO path)
- Cross-vendor same-session validation (AMD + NVIDIA)
- toadStool absorption (hardware orchestration layer available to all springs)
- barracuda trio pattern: coralReef (shader→compiler→driver) + toadStool
  (dispatch→results) + barracuda (physics workloads)

---

*This handoff was generated during the hotSpring sovereign Rust evolution sprint
(May 2026). All warm-catch pipeline code compiles cleanly. ELF patcher tests
pass. Original scripts archived as fossil record.*
