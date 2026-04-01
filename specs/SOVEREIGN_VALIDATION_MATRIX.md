# Sovereign Validation Matrix

**Updated:** 2026-04-01
**Purpose:** Single source of truth mapping every pipeline layer against dispatch paths, hardware substrates, and experiment evidence. "Solve the maze from both sides."

## Dispatch Path Inventory

| Path | Driver | How dispatch works | Titan V | K80 | RTX 5060 |
|------|--------|-------------------|---------|-----|----------|
| **VFIO cold** | `vfio-pci` | Direct BAR0/GPFIFO, FECS must be booted from scratch | BLOCKED (WPR2 HW-locked) | BLOCKED (not UEFI-POSTed) | N/A (display GPU) |
| **VFIO warm** | `nouveau` then `vfio-pci` | nouveau boots FECS via ACR; livepatch freezes state; swap to vfio | **FRONTIER** (livepatch ready) | N/A (nouveau rejects) | N/A |
| **nouveau DRM** | `nouveau` | GEM + VM_INIT/EXEC via DRM ioctls | BLOCKED (missing PMU firmware) | BLOCKED (not UEFI-POSTed) | N/A |
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
| L7: SEC2 + ACR DMA | PASS (Exp 110) | N/A (nouveau handles) | N/A | N/A | N/A |
| L8: WPR/ACR + FECS boot | BLOCKED (Exp 122: WPR2 HW-locked) | N/A (nouveau handles) | BLOCKED (no PMU fw) | UNTESTED | PROVEN |
| L9: FECS alive + GR init | BLOCKED by L8 | **FRONTIER** (livepatch Exp 125) | BLOCKED by L8 | UNTESTED | PROVEN |
| L10: GPFIFO submit | BLOCKED by L9 | **FRONTIER** (QMD+pushbuf ready) | BLOCKED | UNTESTED | PROVEN |
| L11: Fence + readback | BLOCKED by L10 | **FRONTIER** | BLOCKED | UNTESTED | PROVEN |
| Compile (WGSL->SASS) | PASS (SM70) | PASS (SM70) | PASS (SM70) | PASS (SM70) | PASS (via NVK) |

### Tesla K80 (GK210, Kepler)

| Layer | VFIO Cold | nouveau DRM | nvidia+UVM |
|-------|-----------|-------------|------------|
| L1: Device binding | PASS | BLOCKED (not UEFI-POSTed) | UNTESTED |
| L2: BAR0 access | PASS (Exp 123) | BLOCKED | UNTESTED |
| L3: PMC enable | PASS (Exp 123, 128-A2) | BLOCKED | UNTESTED |
| L4-L6: PFIFO/PBDMA | BLOCKED (Exp 129: PRI fault — PFIFO clock domain not running) | BLOCKED | UNTESTED |
| L7: SEC2/ACR | N/A (Kepler: no ACR) | N/A | N/A |
| L8: FECS boot | PASS (Exp 128-A2: FECS PIO boot succeeds on die 0 with nvidia-470 recipe) | BLOCKED | UNTESTED |
| L9-L11: Dispatch | BLOCKED by L4 (need nvidia-470 recipe + devinit for PFIFO) | BLOCKED | UNTESTED |
| Compile | PASS (SM37) | PASS (SM37) | PASS (SM37) |

**K80 Architecture Notes (2026-04-01):**
- No FLR hardware — PMC soft-reset is the only recovery path (`device.reset method=pmc`)
- PRI ring writes from userspace are destructive (corrupted die 0 ring fabric)
- nvidia-470 recipe enables PGRAPH domain (FECS accessible) but NOT PFIFO domain
- PFIFO clock domain requires VBIOS devinit replay or full nvidia POST
- Cold K80 die 1 causes D-state on any PCI access (guarded open required)

### RTX 5060 (GB206, Blackwell)

| Layer | nvidia+UVM | NVK/wgpu |
|-------|------------|----------|
| L1-L3 | PASS (display GPU) | N/A (nvidia loaded) |
| L4-L11 | UNTESTED (code-complete) | N/A |
| Compile | PASS (SM89) | N/A |

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
