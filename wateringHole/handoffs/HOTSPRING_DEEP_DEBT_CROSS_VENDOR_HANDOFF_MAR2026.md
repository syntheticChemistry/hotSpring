# hotSpring → Compute Trio: Deep Debt Burndown + Cross-Vendor Dispatch + PMU Readiness

**Date:** March 22, 2026
**From:** hotSpring
**To:** coralReef, toadStool, barraCuda
**License:** AGPL-3.0-only
**Covers:** 13 deep-debt fixes across coralReef crates, cross-vendor CUDA dispatch via daemon RPC, RTX 5060 dual-use (display + compute), pkexec-free pipeline, PMU cracking readiness for Layer 6 MMU
**Experiment:** 075

---

## Executive Summary

- **13 deep-debt items resolved** across `coral-glowplug`, `coral-driver`, and `hotspring-barracuda` — 3 correctness bugs (P0), 4 robustness improvements (P1), 6 quality fixes (P2)
- **Cross-vendor dispatch validated**: CUDA-capable GPUs dispatch SAXPY kernels interchangeably through the glowplug daemon RPC pipeline — zero privileges, zero pkexec
- **RTX 5060 dual-use proven**: display GPU runs CUDA compute concurrently without driver displacement or DRM disruption — serves as a live page table oracle for PMU cracking
- **pkexec eliminated**: entire compute lifecycle (enumerate, swap, capture, dispatch, health) operates through Unix socket RPC to systemd services
- **PMU cracking tooling hardened**: `try_read_u32`/`try_write_u32` for safe BAR0 access, `OracleError` for clean error propagation, `BusyGuard` for safe concurrent dual-Titan oracle captures

---

## Part 1: Deep Debt Fixes (What Teams Should Absorb)

### For coralReef

**P0-1: TOCTOU BusyGuard Pattern** — `DeviceSlot` now has a `busy: Arc<AtomicBool>` field with RAII `BusyGuard`. Any operation that holds borrowed hardware state across an async boundary (`spawn_blocking`) must acquire the guard first. Mutating RPCs (`swap`, `reclaim`, `resurrect`) check `is_busy()` and refuse if set.

*Evolution guidance:* This pattern should extend to any future long-running operation that borrows device resources. The `BusyGuard` is `Send` — safe to pass into spawn_blocking tasks.

**P0-2: Buffer Handle Validation** — `CudaComputeDevice::dispatch_named` now returns `DriverError::BufferNotFound` for invalid handles instead of silently skipping them. Any compute backend should follow this pattern.

**P0-3: BDF-Specific Device Selection** — `from_bdf_hint` no longer falls back to device 0. Returns `DriverError::OpenFailed` when no device matches the requested BDF. Critical for multi-GPU correctness.

**P1-2: Async nvidia-smi** — `device.compute_info`, `device.quota`, and `device.set_quota` RPCs now use async handlers that release the device mutex before shelling out. Any future RPC that calls external processes must follow this pattern.

**P1-3: Bar0Rw::try_read_u32 / try_write_u32** — New `Result`-returning methods for BAR0 register access. Returns descriptive error for out-of-bounds access instead of sentinel values. Essential for PMU work where register values are diagnostic data.

**P1-4: DriverError::OracleError** — New error variant with `Cow<'static, str>` for oracle module errors. Use `DriverError::oracle(msg)` helper for `String -> DriverError` conversion.

**P2-4: BufReader Sizing** — Per-connection BufReader now starts at 64KB (was 4MB). Grows dynamically as needed. Reduces idle memory footprint for daemon connections.

### For toadStool

**Absorption targets (new coralReef primitives):**

| Primitive | Crate | What It Enables |
|-----------|-------|-----------------|
| `BusyGuard` | coral-glowplug | Safe concurrent device operations in hw-learn observer/applicator |
| `try_read_u32` / `try_write_u32` | coral-driver | Reliable BAR0 reads for hardware learning data collection |
| `OracleError` | coral-driver | Clean error chain from oracle → driver → glowplug → coralctl |
| `DriverError::BufferNotFound` | coral-driver | Explicit handle validation for dispatch operations |

**Dual-use pattern:** The RTX 5060 dual-use (display + compute) model is directly relevant to toadStool's gaming-PC scenario — single-GPU systems can run compute workloads without disrupting display. toadStool should consider a `DualUseCapability` flag in device discovery.

### For barraCuda

**P2-5: Optional Dependencies Pattern** — `cudarc` and `base64` are now optional behind a `cuda-validation` feature in `hotspring-barracuda`. `Cargo.toml` uses `required-features` on specific binaries. Adopt this pattern for any dependency that pulls in system-specific FFI (CUDA headers, ROCm, etc.).

**P2-6: PTX Compatibility** — Target `sm_70` as the universal NVIDIA baseline. CUDA JIT handles upward compatibility (sm_70 → sm_120 on Blackwell). Never target sm_90+ unless the binary is Hopper-only.

**Cross-vendor dispatch:** The `validate_cross_vendor_dispatch` binary demonstrates the full RPC dispatch path (user binary → glowplug → coral-driver CUDA → GPU). barraCuda's `WgpuDevice` and `CoralReefDevice` backends should expose the same `device.dispatch` contract for wgpu workloads.

---

## Part 2: Cross-Vendor Dispatch Architecture

```
User code (zero privileges)
  │ Unix socket JSON-RPC
  ▼
coral-glowplug (systemd, CAP_SYS_ADMIN)
  ├── device.list → enumerate managed devices + capabilities
  ├── device.dispatch → route kernel to specific device by BDF
  │     │
  │     ├── CUDA backend (cudarc → libcuda → NVIDIA GPU)
  │     ├── VFIO backend (sovereign BAR0 → Titan V) [Layer 6 blocked]
  │     └── wgpu backend (vulkan/metal → any vendor) [via barraCuda]
  │
  └── oracle.capture → BAR0 page table dump (via coral-driver)
```

**Key property:** The dispatch RPC is vendor-agnostic. The glowplug daemon routes to the correct backend based on device capabilities. User code never touches hardware directly.

**Validated path:** CUDA backend — SAXPY kernel dispatched to every managed CUDA device, results verified element-wise. The same PTX runs on Volta (sm_70), Turing, Ampere, Ada, and Blackwell (sm_120) via JIT compilation.

---

## Part 3: RTX 5060 Dual-Use — Display + Compute Oracle

The RTX 5060 runs the nvidia 580.x proprietary driver for display output (DRM/KMS) while simultaneously accepting CUDA compute workloads. This is not a new NVIDIA capability — CUDA has always coexisted with display — but it is newly leveraged in the sovereign pipeline as a **page table oracle**:

1. Launch a CUDA allocation on the 5060 → nvidia driver writes PDE/PTE entries to GPU memory
2. Capture BAR0 state via `try_read_u32` → observe the exact bit patterns the driver writes
3. Compare with our sovereign PTE encoding on the Titans → identify divergences
4. Replicate the working encoding on the VFIO path → resolve Layer 6 MMU blocker

**Dual-use does NOT require:**
- Driver swap (the 5060 stays on nvidia throughout)
- pkexec or root (CUDA context creation is unprivileged)
- Display disruption (compositor continues rendering)

**Dual-use DOES require:**
- The nvidia proprietary driver (not nouveau — nouveau lacks CUDA)
- Sufficient VRAM headroom (5060 has 8GB, display uses ~200MB)

---

## Part 4: PMU Cracking Strategy — Layer 6 MMU

### Current State (6/10 Layers Proven)

| Layer | Component | Status |
|-------|-----------|--------|
| 0-1 | PCIe/VFIO + PFB/MMU warm state | Proven (Exp 070-071) |
| 2-3 | PFIFO Engine + Scheduler | Proven — re-init sequence in ~25ms |
| 4-5 | Channel + PBDMA Context | Proven — 12 winning configs |
| **6** | **MMU Page Table Translation** | **BLOCKED** — `0xbad00200` PBUS timeout |
| 7-10 | GPFIFO → Commands → FECS → Shader | Blocked by Layer 6 |

### Attack Matrix

| Vector | Hardware | What It Tests | Deep-Debt Enabler |
|--------|----------|--------------|-------------------|
| 5060 Oracle Capture | RTX 5060 | Reference PTE bit patterns from working driver | `try_read_u32`, dual-use |
| Diff Analysis | 5060 vs Titan V | PTE encoding divergence | `OracleError`, `PageTableDump` |
| Dual Titan A/B | Titan V #1 + #2 | Parallel PTE encoding strategies | `BusyGuard` (concurrent captures) |
| BAR2-Resident Tables | Titan V | VRAM page tables via BAR2 aperture | `try_write_u32` |
| MMU Fault Buffer | Titan V | Specific VA/aperture/reason codes | `try_read_u32` |
| Tesla P80 (pending) | Tesla P80 | Third Volta data point, different PCIe topology | BDF-specific dispatch |

### Suspected Root Causes (Prioritized)

1. **IOMMU mapping gap** — page table DMA buffers at IOVAs 0x5000-0x9000 may not be fully GPU-reachable
2. **BAR2 requirement** — FBHUB MMU may route page table reads through BAR2 (VRAM window), not system memory bus
3. **PTE aperture** — SYS_MEM_COHERENT (aperture=2) may need SYS_MEM_NONCOH (aperture=3) or VRAM (aperture=0) for the page table chain
4. **Instance block flags** — page directory base pointer in RAMIN may need iommufd-specific flags

### What coralReef Should Focus On

1. **`mmu_oracle.rs` evolution** — extend `PageTableDump` to capture 5060 reference state alongside Titan V experimental state
2. **`PageTableDiffResult`** — add a diff mode that compares 5060 oracle PTE encoding with Titan V encoding field by field
3. **BAR2 page table path** — add a `bar2_write_page_tables()` that writes the page table chain to VRAM through BAR2 instead of system memory DMA
4. **MMU fault buffer decoder** — read and decode the MMU fault buffer registers after each experiment to get specific failure reason codes

---

## Part 5: pkexec-Free Operation Model

The complete elimination of `pkexec` from the compute pipeline is now validated end-to-end:

| Component | Runs As | Capabilities |
|-----------|---------|-------------|
| `coral-ember` | systemd service | CAP_SYS_ADMIN, CAP_SYS_MODULE, CAP_SYS_RAWIO |
| `coral-glowplug` | systemd service | CAP_SYS_ADMIN (inherited from ember fd-pass) |
| `coralctl` | regular user | None — communicates via Unix socket |
| Validation binaries | regular user | None — communicate via Unix socket |

**Security model:** Privileged operations (sysfs writes, VFIO fd acquisition, driver bind/unbind) are confined to systemd services with minimal capability sets. User-facing tools authenticate via Unix socket peer credentials. No SUID binaries, no polkit rules, no `pkexec` wrappers.

---

## Parallel-Safe

## Cleanup Notes

- **coralReef `oracle_diff.json`**: Debug output from oracle capture experiments sitting in the coralReef repo root (untracked). Should be added to `.gitignore` or moved to a `data/` directory — not committed.
- **hotSpring `scripts/data/`**: Oracle BAR0 capture binaries from March 15 — experimental data, retain as fossil record.
- **hotSpring `scripts/archive/`**: Superseded VFIO bind scripts — already archived, no action needed.

---

This handoff covers work in `coralReef/crates/` (coral-driver, coral-glowplug) and `hotSpring/barracuda/`. No toadStool or barraCuda source was modified. All teams can evolve in parallel.
