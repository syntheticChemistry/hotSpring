# Experiment 224: Sovereignty Audit + PostPrimordial Checkpoint

**Date**: 2026-05-26
**Status**: CHECKPOINT — Tier 1 sovereignty confirmed, Tier 2 NOT achieved, documentation corrected
**Depends on**: Exp 223 (ACR Sovereign Boot Catalyst), Exp 219 (Catalyst Driver Pattern), Exp 217 (TPC Wall)

## Purpose

Rigorous audit of what "sovereign" actually means for the current hotSpring fleet
on GV100 (Titan V) under VFIO. Separate proven infrastructure from aspirational
compute claims. Checkpoint real achievements honestly. Transition to postPrimordial
deployment model.

## Ground Truth: `sovereign.classify_tier` Results

Ran `sovereign.classify_tier` on both Titan Vs post-power-cycle (VBIOS POST → vfio-pci):

### Titan V #1 (`0000:02:00.0`)

```json
{
  "tier": "warm_infrastructure",
  "tier_level": 1,
  "tier_description": "Warm infrastructure — VFIO/DMA/PFIFO functional, engines gated",
  "evidence": {
    "tpc_alive": false,
    "tpc_status": "0xbadf5040",
    "gpc_enables": "0x00000000",
    "fecs_pc": "0xbadf5040",
    "ce_status": "0xbadf5040",
    "gr_status": "0x00000081",
    "pmc_enable": "0x5fecdff1",
    "pmc_popcount": 23,
    "pramin_accessible": true,
    "pbdma_intr": "0x10011111",
    "ce_runlist": 10
  }
}
```

### Titan V #2 (`0000:49:00.0`)

```json
{
  "tier": "warm_infrastructure",
  "tier_level": 1,
  "evidence": {
    "tpc_alive": false,
    "tpc_status": "0xbadf5040",
    "gpc_enables": "0x00000000",
    "fecs_pc": "0xbadf5040",
    "ce_status": "0xbadf5040",
    "gr_status": "0x00000081",
    "pmc_enable": "0x5fecdff1",
    "pmc_popcount": 23,
    "pramin_accessible": true,
    "pbdma_intr": "0x10011111",
    "ce_runlist": 10
  }
}
```

**Both GPUs: Tier 1 (WarmInfrastructure).** Identical evidence.

## The Three Misconceptions Corrected

### 1. VBIOS POST ≠ toadStool sovereign init

After a power cycle, the UEFI/VBIOS runs GPU POST on all cards: trains HBM2,
runs DEVINIT, loads PMU firmware, sets `PMC_ENABLE=0x5fecdff1`. Then `vfio-pci`
claims the Titan Vs at `t=10.3s`. The warm state that `sovereign.init` detects
is **VBIOS work**, not toadStool achievement. `sovereign.init` on a post-VBIOS
GPU correctly detects the warm state and skips all hard stages:

- `pgraph_reset`: "falcon warm — skipped to preserve FECS/GPCCS"
- `memory_training`: "warm detected — skipped"
- `gr_init`: "FECS warm-preserved/running: skipping re-bootstrap"

This is a valid and useful health check, but it is not sovereign initialization.

### 2. `compute_ready: true` ≠ dispatch readiness

From `sovereign_init.rs` line 761, `compute_ready` means all pipeline stages
passed. The `verify()` stage checks:

- PTIMER ticking
- PRAMIN sentinel write/readback
- PMC_ENABLE readback

It does **NOT** check:

- TPC PRI stations (`0x504000`) — the actual shader dispatch gate
- PBDMA device error bits
- Whether a shader actually runs

`compute_ready` should be understood as `init_pipeline_passed`.

### 3. RTX 5060 DRM dispatch ≠ sovereign compute

The RTX 5060 shader dispatch (8/8 roundtrips, Exp 218) runs through the nvidia
DRM driver on `0000:21:00.0`. This is **vendor-mediated compute**, not sovereign.
It validates the compile-then-dispatch pipeline (coralReef → toadStool), but the
GPU is controlled by nvidia's proprietary kernel module.

## The TPC Wall (Tier 2 Blocker)

From Exp 217 and confirmed by Exp 221/223 sovereignty audit:

- TPC PRI register at `0x504000` returns `0xBADF5040` (PRI fault)
- This means the TPC PRI ring stations were never created
- TPC stations are created by **GPCCS firmware** during GR initialization
- GPCCS is HS fuse-locked on GV100 — host cannot load or start code on it
- The only way TPC stations exist is if vendor firmware ran GPCCS (nvidia driver
  loaded, or VBIOS created them at POST — but VBIOS only does a subset)

**Tier 2 requires GPCCS firmware execution. On VFIO Titan V, this is blocked
by hardware fuses.** No software-only path exists without either:

1. Running the nvidia driver (catalyst pattern, Exp 219)
2. Reverse-engineering and signing GPCCS firmware (Tier 3 research)
3. Using a generation without HS fuse locks

## What IS Real and Proven (Tier 1 Achievements)

These are genuine, hardware-validated achievements:

| Achievement | Experiment | Evidence |
|-------------|-----------|----------|
| VFIO bind + BAR0 MMIO | Exp 191+ | Full register read/write on live GPU |
| DMA mapping | Exp 199 | IOMMU-backed DMA buffers via VFIO cdev |
| PRAMIN read/write | Exp 200 | VRAM access through BAR0 window |
| PFIFO channel creation | Exp 204 | PBDMA pushbuffer encoding |
| CE runlist discovery | Exp 210 | Copy Engine scheduling fabric |
| FECS liveness | Exp 207 | Falcon PC advancing, CPUCTL_ALIAS responsive |
| Warm handoff pipeline | Exp 208 | 183ms warm pipeline (76× faster than cold) |
| Catalyst driver pattern | Exp 219 | nvidia-470 as one-shot init, 83K regs captured |
| PRI ring recovery | Exp 221 | PGRAPH re-enable after kernel PCI unbind |
| Falcon register map | Exp 223 | v5 offsets, shared module, 16 unit tests |
| Bar0 hardening | Exp 223 | ENGCTL deny-list, alignment, dead-link sentinel |
| Sovereign tier taxonomy | Exp 210 | `SovereignTier` enum, evidence-based classification |
| RTX 5060 DRM dispatch | Exp 218 | 8/8 shader roundtrips (vendor-mediated, not sovereign) |
| AMD sovereign compiler | — | 24/24 QCD shaders |
| NVIDIA sovereign compiler | — | SM35 + SM70 + SM120 |

## Remaining Path to Tier 2

```
Current state (Tier 1)           Tier 2 requirement
─────────────────────            ──────────────────
VFIO + BAR0 ✅                   Same
DMA + PRAMIN ✅                  Same
PFIFO + PBDMA ✅                 Same
PRI ring ✅                      Same
Falcon register map ✅           Same
TPC stations ❌ (0xBADF5040)     TPC PRI ring stations created
GPCCS ❌ (HS fuse-locked)        GPCCS firmware running
GR init ❌ (gpc_enables=0)       GPC broadcast enabled
CE dispatch ❌ (0xBADF5040)      Copy Engine PRI alive
```

### Viable Tier 2 Paths

1. **Catalyst boot (Exp 219)**: Load nvidia-470, capture warm state including TPC
   stations, unbind nvidia, replay. Proven for register capture but TPC station
   persistence across unbind not yet validated.

2. **Runtime Services model**: Keep nvidia loaded as persistent compute service.
   toadStool manages infrastructure. Pragmatic but not sovereign.

3. **GPCCS firmware research (Tier 3)**: Reverse-engineer GPCCS microcode, find
   signing keys or exploit HS boundary. Long-term research, likely years.

4. **Generation pivot**: Test on GPUs without HS fuse locks (Kepler, Maxwell).
   Simpler hardware, but less relevant to modern compute.

## PostPrimordial Transition

This checkpoint marks hotSpring's transition to postPrimordial deployment:

- Local `wateringHole/` archived to `infra/wateringHole/handoffs/hotSpring/`
- NUCLEUS primals sourced from live `plasmidBin` deployments
- **toadStool remains a local build** — hotSpring team owns it, pushes upstream
- Other 8 primals (beardog, songbird, coralreef, barracuda, nestgate, rhizocrypt,
  loamspine, sweetgrass) target live plasmidBin
- Future handoffs go to `infra/wateringHole/` only

## Conclusion

hotSpring has built a genuine, hardware-validated Tier 1 sovereign control plane
for GV100 over VFIO. This is real infrastructure that no other open-source project
has achieved at this level of integration. But Tier 2 sovereign compute —
dispatching shaders without vendor firmware — remains blocked by the TPC wall and
HS fuse-locked GPCCS. The path forward is clearly mapped. The infrastructure is
ready for when the TPC wall falls.
