# Sovereign Validation Matrix

**Updated:** 2026-05-11 (Sprint E — ALL 3 GPUs Sovereign)
**Purpose:** Single source of truth mapping every pipeline layer against dispatch paths, hardware substrates, and experiment evidence. "Solve the maze from both sides." **Sprint E (May 2026):** Binary-patched nouveau warm-catch resolves GAP-HS-073 (Titan V FECS RUNNING) and GAP-HS-076 (K80 GDDR5 trained + GPCs active). Warm-catch pipeline elevated to pure Rust (`coralctl warm-catch`). ALL 3 GPUs sovereign. Dispatch validation and E2E physics wiring are the remaining gap. **Sprint D (May 2026):** Warm handoff DMATRF to FECS proven. Falcon v5 HS ROM security gate identified. **Sprint C:** HW validation run confirmed warm handoff as intermediate path. **Sprint B:** Titan V SEC2 FBIF, K80 warm NOP dispatch, SLM pool, unsafe audit. **SovereignInit pipeline (Exp 165):** pure Rust `open_sovereign(bdf)` path.

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

| Layer | VFIO Cold | VFIO Warm | nouveau DRM | nvidia+UVM | NVK/wgpu |
|-------|-----------|-----------|-------------|------------|----------|
| L1: Device binding | PASS | PASS | PASS | UNTESTED | PASS |
| L2: BAR0/BAR2 access | PASS | PASS | PASS | UNTESTED | PASS |
| L3: PMC enable + engines | PASS | PASS | PASS | UNTESTED | PASS |
| L4: PFIFO init + PBDMA | PASS (Exp 058) | PASS (warm mode) | PASS | UNTESTED | PASS |
| L5: MMU fault buffer | PASS (Exp 076) | PASS | PASS | UNTESTED | PASS |
| L6: PBDMA context load | PARTIAL (Exp 058) | PASS (stale intr fix) | PASS | UNTESTED | PASS |
| L7: SEC2 + ACR DMA | PASS (Exp 110) | **PARTIAL** (ACR BL starts mb0=1, stalls — PMU FW missing) | N/A | N/A | N/A |
| L8: WPR/ACR + FECS boot | **BLOCKED** (SEC2 ACR never completes: no PMU FW in linux-firmware) | **PARTIAL** (DMATRF FECS 101blk/192µs proven; ROM HS gate blocks unsigned code) | PASS (nouveau handles) | UNTESTED | PROVEN |
| L9: FECS alive + GR init | BLOCKED (depends on L8) | **FRONTIER** (depends on L8 ACR completion) | PASS (nouveau handles) | UNTESTED | PROVEN |
| L10: GPFIFO submit | BLOCKED (depends on L9) | **FRONTIER** (QMD+pushbuf ready) | **PASS** (Exp 163: NOP dispatch, pure Rust) | UNTESTED | PROVEN |
| L11: Fence + readback | BLOCKED (depends on L10) | **FRONTIER** | **PASS** (syncobj wait, Exp 163) | UNTESTED | PROVEN |
| Compile (WGSL->SASS) | PASS (SM70) | PASS (SM70) | PASS (SM70) | PASS (SM70) | PASS (via NVK) |

### Tesla K80 (GK210, Kepler)

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

**K80 Architecture Notes (2026-05-06):**
- No FLR hardware — PMC soft-reset is the only recovery path (`device.reset method=pmc`)
- PRI ring writes from userspace are destructive (corrupted die 0 ring fabric)
- PLX PEX 8747 PCIe switch: D3cold kills link, requires full AC power drain to recover
- Cold-boot sovereign: udev `drivers_probe` + `d3cold_allowed=0` fix applied (Exp 180)
- **SSEL PLL fix (May 2026):** `program_engine_plls` now uses per-engine `locked_mask` — only sets SSEL bits for PLLs that actually achieved lock. Prevents dead GPC clock from failed PLL.
- **Post-PMU PLL retry (May 2026):** After PMU boot, re-tests GPC PLL writability. If PMU ungated power domain, retries crystal clocks + engine PLLs.
- **Warm NOP dispatch wired:** GPFIFO push → doorbell → GP_GET poll infrastructure complete in `kepler_cold_pipeline.rs`.
- nouveau on kernel 6.17.9: GR engine does NOT initialize for K80 (chip 0xf2, no `nvf2_chipset` entry). CTXSW_TIMEOUT on all compute channels (Exp 181).

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

## Maze Strategy: Solving from Both Sides

```
                    Titan V (GV100)
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    VFIO Warm        nvidia+UVM      NVK/wgpu
    (sovereign)      (proprietary     (fallback
                      trace+learn)    compute)
         │               │               │
    livepatch NOP    code-complete    PROVEN for
    gk104_runl       in uvm_compute   physics
    ready to test    needs nvidia     today
         │           loaded for       │
         │           Titan V          │
         ▼               ▼               ▼
    FECS preserved   Learn FECS      Run science
    through swap?    init sequence    while cracking
         │           from RM ioctls   sovereignty
         │               │
         └───────┬───────┘
                 │
         Both inform sovereign
         FECS boot strategy
```

### Path A: VFIO Warm Handoff (our frontier)
- **Status:** Livepatch 4-function NOP deployed, dynamic enable/disable wired, reset_method fix proven
- **Next:** Test `coralctl warm-fecs` + `vfio_dispatch_warm_handoff`
- **If FECS stays alive:** Full E2E pipeline lights up via toadStool `shader.dispatch`

### Path B: nvidia+UVM Proprietary Tracing
- **Status:** `NvUvmComputeDevice` code-complete (RM+GPFIFO+QMD+sync)
- **Next:** Load nvidia proprietary for Titan V, run UVM dispatch, trace RM init sequence
- **Value:** Learn exactly how RM initializes FECS/GPCCS on Volta without WPR barriers
- **Dual-use:** RTX 5060 stays on nvidia for display; Titan V can temporarily use nvidia for learning

### Path C: NVK/wgpu Fallback (proven, available now)
- **Status:** 4-tier compute validated, QCD production runs complete
- **Value:** Run real physics while cracking sovereignty; baseline for performance comparison

## Upstream Integration Status

| System | Version | Integration | Impact |
|--------|---------|-------------|--------|
| toadStool | S168 | `shader.dispatch` wired, delegates to coralReef `compute.dispatch.execute` | Full orchestration ready — lights up when VFIO dispatch works |
| barraCuda | Sprint 23 | f64 precision pipeline fixed, `SovereignDevice` validates RPC contract | Correct physics results guaranteed; IPC contract validated |
| coral-ember | current | FLR for capable GPUs, PMC soft-reset for non-FLR (K80/Titan V), `device_has_flr()` PCIe cap check, auto-reset routing, `pri_fault` in fecs.state | Single reset authority; no sudo/pkexec needed |
| coral-glowplug | current | All resets route through ember (FD holder), PRI-aware warm_handoff polling, livepatch via ember RPC | No direct VFIO reset; clean diagnostic signals |
| coral-driver | current | PRI fault guards on all falcon state (fecs_is_alive, GrEngineStatus, FalconState, diagnostics), cold boot recipe | Zero false-positive warm handoff detection |

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
