# Sovereign Validation Matrix

**Updated:** 2026-05-21 (Sprint G — TPC Wall Identified, Kernel Health Preflight)
**Purpose:** Single source of truth mapping every pipeline layer against dispatch paths, hardware substrates, and experiment evidence. **Sprint G (May 2026):** Exp 215 refines Tier 2 blocker from "GPC power domain" to **TPC PRI ring station wall** — GPC fabric survives warm handoff but TPC control registers return `0xBADF5040` (station missing). PMU software path **CLOSED** on Volta (Exp 211). Kernel build environment health check integrated (Exp 216). Sovereign driver rotation codified in diesel engine (Exp 211/S267). Exp 217 targets `sw_nonctx.bin` broadcast TPC wake. **Sprint F:** Tier model codified (Exp 210), CE runlist discovery, binary-patch warm handoff proven (Exp 211), sovereignty consolidation (Exp 212), live hardware handoff (Exp 213), D-state hardening (Exp 214). **Sprint E:** Binary-patched nouveau warm-catch resolves GAP-HS-073/076. ALL 3 GPUs sovereign infrastructure (Tier 1). **SovereignInit pipeline (Exp 165):** pure Rust `open_sovereign(bdf)` path.

## Dispatch Path Inventory

| Path | Driver | How dispatch works | Titan V | K80 | RTX 5060 |
|------|--------|-------------------|---------|-----|----------|
| **VFIO cold** | `vfio-pci` | Direct BAR0/GPFIFO, FECS must be booted from scratch | Bypassed by warm-catch | Bypassed by warm-catch | **PROVEN** (RTX 5060 full dispatch, Exp 175-177) |
| **VFIO warm** | binary-patched `nouveau` then `vfio-pci` | Patched nouveau boots GPU (4 teardown fns NOP'd at ELF level); swap to vfio preserves warm state | **SOVEREIGN** (FECS RUNNING, GAP-HS-073 RESOLVED, Exp 190) | **SOVEREIGN** (GDDR5 trained, 5 GPCs active, GAP-HS-076 RESOLVED, Exp 190) | N/A (native VFIO cold dispatch) |
| **nouveau DRM** | `nouveau` | GEM + VM_INIT/EXEC via DRM ioctls | **PROVEN** (NOP dispatch, Exp 163) | BLOCKED (not UEFI-POSTed) | N/A |
| **nvidia-drm + UVM** | `nvidia` proprietary | RM ioctls + `/dev/nvidia-uvm` GPFIFO | UNTESTED (code-complete) | UNTESTED | Available (RTX 5060) |
| **NVK/wgpu** | `nouveau` + Mesa NVK | Vulkan compute via wgpu abstraction | **PROVEN** (4-tier QCD) | N/A | N/A |
| **AMD DRM** | `amdgpu` | GEM + PM4 via DRM ioctls | N/A | N/A | N/A |

## Pipeline Layer Matrix

### Legend
- **PASS** — hardware-validated, experiment reference in parentheses
- **PROVEN** — validated end-to-end with physics workloads
- **PARTIAL** — some sublayers work, specific gap noted
- **BLOCKED** — known blocker identified, experiment reference
- **UNTESTED** — code exists but no hardware validation yet
- **N/A** — path not applicable to this hardware

### Titan V (GV100, SM70)

**Sovereignty Tier Status:**
- **Tier 1 (Warm Infrastructure):** VALIDATED (Exp 210) — VFIO, BAR0, DMA, PFIFO, channels, pushbuffers, FECS liveness, topology discovery. 183ms warm pipeline (Exp 208).
- **Tier 2 (Warm Compute):** CLASSIFIED (Exp 215) — GPC fabric alive (6/6 GPCs), CE4/CE5 alive. **Dispatch BLOCKED by TPC PRI ring wall** (`0x504000` = `0xBADF5040`). PMU software path CLOSED (Exp 211).
- **Tier 3 (Silicon Deistic):** Long-term target — VBIOS interpreter has 422 ops (Exp 204), ~100 unknown opcodes remain.

| Layer | VFIO Cold | VFIO Warm | nouveau DRM | nvidia+UVM | NVK/wgpu |
|-------|-----------|-----------|-------------|------------|----------|
| L1: Device binding | PASS | PASS | PASS | UNTESTED | PASS |
| L2: BAR0/BAR2 access | PASS | PASS | PASS | UNTESTED | PASS |
| L3: PMC enable + engines | PASS | PASS (23 engines preserved, Exp 211) | PASS | UNTESTED | PASS |
| L4: PFIFO init + PBDMA | PASS (Exp 058) | PASS (warm mode, PBDMA submit proven S263) | PASS | UNTESTED | PASS |
| L5: MMU fault buffer | PASS (Exp 076) | PASS | PASS | UNTESTED | PASS |
| L6: PBDMA context load | PARTIAL (Exp 058) | PASS (stale intr fix) | PASS | UNTESTED | PASS |
| L7: SEC2 + ACR DMA | PASS (Exp 110) | **PASS** (ACR DMA boot solved, Exp 206: FECS+GPCCS via iommufd) | N/A | N/A | N/A |
| L8: WPR/ACR + FECS boot | **BLOCKED** (cold WPR barrier) | **PASS** (FECS alive via ACR DMA; PC advancing in HS poll loop, Exp 206/215) | PASS (nouveau handles) | UNTESTED | PROVEN |
| L9: FECS alive + GR init | BLOCKED (depends on L8) | **BLOCKED** — TPC wall: GPC fabric alive, **TPC PRI stations missing** (`0xBADF5040`). FECS runs but cannot dispatch. PMU path CLOSED. (Exp 211/215) | PASS (nouveau handles) | UNTESTED | PROVEN |
| L10: GPFIFO submit | BLOCKED | **BLOCKED** — depends on L9 TPC ungating | **PASS** (Exp 163: NOP dispatch, pure Rust) | UNTESTED | PROVEN |
| L11: Fence + readback | BLOCKED | **BLOCKED** — depends on L10 | **PASS** (syncobj wait, Exp 163) | UNTESTED | PROVEN |
| Compile (WGSL->SASS) | PASS (SM70) | PASS (SM70) | PASS (SM70) | PASS (SM70) | PASS (via NVK) |

**TPC Wall Detail (Exp 215):**

| Register | Value | Meaning |
|----------|-------|---------|
| PMC_ENABLE (`0x200`) | `0x5FECDFF1` | 23 engines alive (warm) |
| GPC0 per-unit (`0x500000`) | `0x8780029F` | GPC fabric alive |
| GPC0 TPC0 control (`0x504000`) | `0xBADF5040` | **TPC PRI station missing** |
| GPC0 TPC0 SM0 (`0x504200`) | `0x000900F0` | SM accessible (different PRI sub-path) |
| CE4 (`0x108000`) | `0x01004005` | Alive |
| FECS PC | ~`0xEAC`, advancing | HS poll loop |

### Tesla K80 (GK210, Kepler) — Hardware Destroyed (Exp 199), RETIRED

**Status:** K80 #1 destroyed in Exp 199 (thermal event → `PowerSafetyProfile` created in Exp 200). K80 path retired — replaced by nvidia-470 nvsov dual-load injection (Exp 218) as primary Tier 2 strategy. Historical data below retained as fossil record.

| Layer | VFIO Cold | nouveau DRM | nvidia+UVM |
|-------|-----------|-------------|------------|
| L1: Device binding | PASS | PASS (nouveau DRM, kernel 6.17) | UNTESTED |
| L2: BAR0 access | PASS (Exp 123) | PASS | UNTESTED |
| L3: PMC enable | PASS (Exp 123, 128-A2) | PASS | UNTESTED |
| L4-L6: PFIFO/PBDMA | PASS (Exp 179: dynamic PBDMA→runlist, GPFIFO wired) | PASS (channel created, CTXSW_TIMEOUT) | UNTESTED |
| L7: SEC2/ACR | N/A (Kepler: no ACR) | N/A | N/A |
| L8: FECS boot | PASS (Exp 128-A2: PIO boot + Exp 179: internal firmware) | PASS (nouveau handles) | UNTESTED |
| L9-L11: Dispatch | **FRONTIER** (NOP dispatch wired, cold PLL fix, needs GPC HW validation) | BLOCKED (GR engine not init on kernel 6.17, Exp 181) | UNTESTED |
| Compile | PASS (SM37) | PASS (SM37) | PASS (SM37) |

**K80 Architecture Notes (2026-05-06, historical — hardware unavailable):**
- No FLR hardware — PMC soft-reset is the only recovery path
- PLX PEX 8747 PCIe switch: D3cold kills link, requires full AC power drain
- Unsigned falcons: no ACR barrier, `gk110_pmu_pgob()` available for TPC ungating
- nouveau on kernel 6.17.9: GR engine does NOT initialize (no `nvf2_chipset` entry)

### RTX 5060 (GB206, Blackwell)

| Layer | nvidia+UVM | NVK/wgpu |
|-------|------------|----------|
| L1-L3 | PASS (display GPU) | N/A (nvidia loaded) |
| L4-L11 | **PROVEN** (Exp 181: 8/8 dispatch, SM120 SASS) | N/A |
| Compile | PASS (SM120) | N/A |

### AMD RX 6950 XT (RDNA2, decommissioned)

| Layer | AMD DRM |
|-------|---------|
| L1-L11 | **PROVEN** (full pipeline, 6/6 dispatch tests PASS) |
| Compile | PASS (GFX1030) |

## Maze Strategy: Breaking the TPC Wall (Post-Exp 217)

```
            TPC PRI Wall — DEFINITIVELY FIRMWARE-MEDIATED
                         │
    ╔════════════════════╧════════════════════════╗
    ║ BAR0 writes CANNOT create TPC PRI stations  ║
    ║ sw_nonctx.bin: broadcast accepted (0x26F0)  ║
    ║ per-GPC TPC control: still 0xBADF5040       ║
    ║ PGRAPH reset: no effect on TPC stations     ║
    ╚════════════════════╤════════════════════════╝
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    nvidia-470       agentReagents
    nvsov dual-load  VM compute
    (Exp 218)        (available now)
         │               │
    DKMS 470.256.02  nvidia-470 in
    → patch NOPs     VM → full CUDA
    → strip ksymtab  No host handoff
    → rename nvsov   SBR on exit
    → insmod+bind        │
         │               ▼
         ▼          Direct compute
    TPC stations     inside VM
    preserved?       (parallel path)
         │
    warm swap to
    vfio-pci
         │
         ▼
    classify_tier()
    → tpc_alive?
         │
         ▼
    FECS dispatch + QMD
    = Tier 2 Sovereign
```

### Path A: nvidia-470 nvsov Dual-Load Injection (PRIMARY — Exp 218)
- **Status:** Co-load isolation SOLVED. Module loads alongside host nvidia-580. Reboot
  needed to clear zombie module from test oops, then full pipeline test.
- **Blockers Solved:**
  - `exports duplicate symbol` → `objcopy --remove-section` strips ksymtab sections
  - procfs/chardev conflicts → 5 co-load isolation NOPs (`nv_cap_init`, `nv_cap_drv_init`,
    `nv_procfs_init`, `nvidia_register_module`, `nv_cap_procfs_init`)
  - Relocation conflict at patch sites → ret0 at offset+5 (after ftrace preamble)
  - Kernel 6.17 nonzero relocation targets → PC32/PLT32 normalization added
- **Next:** Reboot, run `sovereign.warm_handoff` with `nvidia_patched_titanv`, verify
  TPC alive, classify tier.
- **RPC:** `sovereign.warm_handoff` with strategy `nvidia_patched_titanv`

### Path B: agentReagents VM Compute (parallel, available now)
- **Status:** `reagent-nvidia470-titanv.yaml` template exists. nvidia-470 + CUDA 11.4 inside
  VM. SBR on VM exit destroys state (no warm handoff back to host). Used when Titan V needs
  full compute and host sovereignty isn't required.
- **Value:** Compute immediately available. No TPC wall inside VM.

### ~~Path C: K80 Cross-Generation~~ (RETIRED)
- **Status:** K80 #1 destroyed (Exp 199). Hardware retired — no replacement.
  Historical data retained as fossil record. nvidia-470 nvsov path (Path A) supersedes.

### Path D: NVK/wgpu Fallback (proven, available now)
- **Status:** 4-tier compute validated, QCD production runs complete
- **Value:** Run real physics while cracking sovereignty; baseline for performance comparison

### Path E: BAR0 Register Writes (CLOSED — Exp 217)
- **Status:** DEFINITIVELY CLOSED. `sw_nonctx.bin` broadcast writes, full 5-phase ungating,
  PGRAPH reset, CG sweep, PRI enumerate — all tested with real firmware on both Titan Vs.
  Broadcast `0x419xxx` path configures TPC *settings* but cannot create TPC PRI ring
  *stations*. Twin study confirmed identical results on both cards.

## Upstream Integration Status

| System | Version | Integration | Impact |
|--------|---------|-------------|--------|
| toadStool | S268 | `sovereign.warm_handoff` RPC, `sovereign.kernel_health` preflight, `sovereign.experiment` stages, `toadstool kernel-health` CLI, diesel engine per-GPU lifecycle, 87 RPC methods, 700 cylinder tests | Full orchestration — warm handoff → tier classify → experiment stages pipeline |
| barraCuda | Sprint 23 | f64 precision pipeline fixed, `SovereignDevice` validates RPC contract | Correct physics results guaranteed; IPC contract validated |
| cylinder | S267 | `kmod` lifecycle, `module_patch` binary NOP, `sovereign_handoff` 8-step pipeline, `sovereign_tiers` classifier, `kernel_health` 3-layer preflight, `NvGspBridge` firmware interface | Hardware sovereignty stack — all stages implemented |

## Key Experiments Reference

| Exp | Title | Layer | Outcome |
|-----|-------|-------|---------|
| 058 | VFIO PBDMA Context Load | L6 | PBDMA loads context, GP_PUT DMA read issue |
| 076 | MMU Fault Buffer L6 Breakthrough | L5-L6 | DMA roundtrip verified |
| 087 | WPR Format Analysis | L8 | W1-W7 fixed, ACR processes WPR |
| 095 | Sysmem HS Mode Breakthrough | L7 | HS mode achieved (SCTL=0x3002) |
| 110 | Consolidation Matrix | L7 | PDE slot = sole HS determinant |
| 112 | Dual Phase Boot | L7-L8 | HS mode via dual-phase |
| 122 | WPR2 Resolution | L10 | WPR2 HW-locked — root cause definitive |
| 123 | K80 Sovereign Compute | L1-L3 | K80 BAR0 probed, no security barriers |
| 124 | VM Capture Cross-Analysis | L7-L10 | nvidia-470/535 VM captures analyzed |
| 125 | Warm Handoff Livepatch | L9 | Livepatch NOP strategy, FECS preservation |
| 128 | GPU Puzzle Box Matrix | L3-L8 | FECS PIO boot succeeds on K80 (A2), PFIFO PRI fault persists (A3) |
| 129 | Kepler VFIO Dispatch | L4-L10 | KeplerChannel created, dispatch blocked by PFIFO PRI faults |
| 130 | No-FLR Recovery + PRI Ring | L1-L3 | PRI fault false-positive fixes, PMC soft-reset validated, FLR routing through ember |
| 131 | Reset Architecture Evolution | L1-L11 | ember owns all resets, PRI-aware diagnostics, no sudo/pkexec, Titan V FLReset- confirmed |
| 159 | Titan V VM-POST HBM2 | L1 | HBM2 trained via VM, preserved through nouveau warm-cycle + reset_method clear |
| 163 | Firmware Boundary | L8-L11 | **Architectural pivot.** NOP dispatch via DRM (pure Rust). PMU mailbox mapped. PmuInterface created. Hot-handoff channel injection proven. |
| 164 | Sovereign Compute Dispatch | L11 | **5/5 E2E phases pass.** f32, f64, multi-workgroup, Lennard-Jones on Titan V via DRM. |
| 165 | SovereignInit Pipeline | L12 | **8-stage pure Rust nouveau replacement.** `open_sovereign(bdf)`. Firmware-as-ingredient. GR init extracted. FECS method probe. 429 tests. |
| 190 | Three-GPU Validation | L1-L11 | ALL 3 GPUs sovereign. RTX 5060 12/12 roundtrip. Titan V warm-catch. K80 5 GPCs. |
| 204 | VBIOS Interpreter | L1 | 422 ops executed, 231 BAR0 writes on cold Titan V. ~100 unknown opcodes. |
| 206 | Falcon ACR DMA Boot | L7-L8 | ACR DMA solved — FECS+GPCCS boot via iommufd. FECS cpuctl=0x10 on both Titans. |
| 208 | Warm Keepalive Pipeline | L1-L9 | **183ms warm pipeline** (76× faster than cold). fd store survives daemon restart. |
| 210 | GPC Boundary + Tier Model | L9 | Tier model codified. CE runlist=10. PTOP parser fix. **Tier 1 VALIDATED.** |
| 211 | PMU Mailbox + Driver Rotation | L9 | PMU HS-locked. Binary-patch warm handoff. `sovereign.warm_handoff` RPC. **PMU path CLOSED.** |
| 212 | Sovereignty Consolidation | L9 | Golden-state replay wire. Generation-aware `classify_tier_for_profile()`. |
| 213 | Live Hardware Warm Handoff | L1-L9 | IOMMU sibling unbind, anchor release, `/tmp` writable. Post-reboot validated. |
| 214 | D-State Hardening | L1 | Guarded sysfs, child-process isolation, timeout-guarded writes. |
| 215 | Sovereign Warm Compute Tier 2 | L9 | **TPC wall identified.** GPC fabric alive, TPC PRI stations missing (`0xBADF5040`). Tier 2 reclassified. |
| 216 | Kernel Autoconf Mismatch | L0 | 3-layer `autoconf.h` health check. `sovereign.kernel_health` RPC. All modules clean. |
| 217 | TPC PRI Station Creation | L9 | **TPC wall confirmed firmware-dependent.** `sw_nonctx.bin` (341 real writes, 94 TPC broadcast) applied via `NvGspBridge`. Broadcast `0x419xxx` writes accepted (returns `0x26F0`), but per-GPC TPC control (`0x504000`) remains `0xBADF5040`. Full 5-phase ungating + PGRAPH reset also failed. Twin study confirmed on both Titan Vs. Conclusion: TPC PRI stations require signed GPCCS firmware. |
