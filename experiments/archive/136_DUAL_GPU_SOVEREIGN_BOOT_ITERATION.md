# Experiment 136: Dual GPU Sovereign Boot Iteration

**Date**: 2026-04-02
**GPUs**: Titan V (GV100, 0000:03:00.0) + Tesla K80 (GK210, 0000:4c:00.0)
**Status**: RESULTS CAPTURED — both GPUs hit known barriers, clear next steps identified
**Predecessor**: Exp 132 (Ember-Frozen Warm Dispatch), Exp 133 (Kepler Sovereign Compute), Exp 134 (K80 Cold Boot Pipeline), Exp 135 (Dual GPU Sovereign Boot Attempt)

## Executive Summary

Both GPUs advanced understanding but did not achieve sovereign dispatch. The Titan V
revealed that **SEC2 is alive and responsive** post-nouveau (Strategy 8 breakthrough),
and the K80 confirmed that sovereign cold boot needs VBIOS POST for GDDR5 memory
training. Clear, specific next steps identified for both.

## System Configuration

```
RTX 5070 (GB206)  21:00.0  nvidia      display (PROTECTED)
Titan V  (GV100)  03:00.0  vfio-pci    IOMMU group 73 (isolated)
K80 die1 (GK210)  4c:00.0  vfio-pci    IOMMU group 37 (shares root complex with USB)
K80 die2 (GK210)  4d:00.0  vfio-pci    IOMMU group 38
```

Daemons: coral-ember (PID 1691), coral-glowplug (PID 1776)
Sockets: `/run/coralreef/{ember,glowplug}.sock`

## Titan V (GV100) — Diesel Engine Pipeline

### Phase 1: Nouveau Warm-FECS (baseline)

```
coralctl warm-fecs 0000:03:00.0
```

Result: FECS went from PRI fault to **halted in HS mode** (SCTL=0x3000).

| Register | Pre-warm | Post-warm |
|----------|----------|-----------|
| PMC_ENABLE (0x200) | 0x40000121 | **0x5fecdff1** (all engines) |
| FECS_CPUCTL (0x409100) | 0xbadf1201 (PRI fault) | **0x00000010** (halted) |
| FECS_SCTL (0x409240) | — | **0x00003000** (HS mode!) |
| GPCCS_CPUCTL (0x41a100) | 0xbadf1201 | **0x00000010** (halted) |
| GPCCS_SCTL (0x41a240) | — | **0x00003000** (HS mode!) |
| SEC2_SCTL (0x87240) | — | **0x00003000** (HS mode!) |
| PMU_SCTL (0x10a240) | — | **0x00003000** (HS mode!) |
| VRAM | dead | **ALIVE** (write-readback 0xdeadbeef OK) |
| WPR2_ADDR_LO (0x100CD4) | — | 0x00fffe02 (configured) |

**Key finding**: ACR chain completed during nouveau session. ALL four falcons
(FECS, GPCCS, SEC2, PMU) are in HS (secure) mode. WPR is configured. But
FECS never entered visible running state (cpuctl bit 4 always set).

### Phase 2: nvidia Warm-FECS (comparison)

```
coralctl warm-fecs-nvidia 0000:03:00.0 --keepalive --settle 8
```

Result: nvidia path was **more destructive**. Post-swap: PMC_ENABLE=0x40000020,
FECS PRI-faulting, VRAM dead. nvidia's teardown resets the GPU aggressively.

**Conclusion**: nouveau path preserves more state. nvidia path not viable for
diesel engine pattern.

### Phase 3: Nouveau with --poll-fecs --keepalive

```
coralctl warm-fecs 0000:03:00.0 --poll-fecs --keepalive
```

Result: 596 polls at 50ms intervals over ~30s. FECS **never** entered running
state. cpuctl=0x00000010 (halted) consistently throughout entire nouveau session.

**Root cause**: nouveau loads FECS firmware via ACR (establishing HS mode) but
the falcon never starts a visible execution loop. This is a nouveau/Volta
limitation — nouveau's GR init doesn't fully activate FECS scheduling on GV100.

### Phase 4: Host FECS restart attempts

Attempted to restart halted FECS from host:
- `CPUCTL STARTCPU (0x02)`: cpuctl → 0x12 (HRESET, halt not cleared)
- `CPUCTL_ALIAS (0x409130)`: write consumed but no effect
- Both paths: HS lockdown prevents host-initiated restart

### Phase 5: ACR Boot Solver (14 strategies)

```
coralctl acr-boot 0000:03:00.0
```

All 14 strategies failed, but **Strategy 8 is the breakthrough**:

**Strategy 8: "ACR mailbox command (live SEC2) + falcon start"**
- SEC2 was RUNNING (cpuctl=0x00000000) — alive and processing
- `BOOTSTRAP_FALCON mask=0x000c` → mb0=0x00000001, mb1=0x0000000c
- SEC2 acknowledged the command and attempted firmware DMA
- **FECS briefly started** (cpuctl=0x00000000 for 10ms)
- But: `GPCCS IMEM[0x3400] is empty after BOOTSTRAP_FALCON — ACR DMA failed`
- The DMA context (instance block / page tables) was destroyed by nouveau teardown

**Strategy 13: Direct IMEM upload**
- Firmware uploaded and verified (IMEM match=true)
- Both falcons start in HRESET (0x12) — HS lockdown blocks NS execution
- Confirms: correct firmware in IMEM insufficient without ACR authentication

### Titan V Next Steps

1. **SEC2 DMA context reconstruction** (HIGH PRIORITY): Strategy 8 proves SEC2
   is alive and responsive. If we set up proper DMA page tables before sending
   BOOTSTRAP_FALCON, ACR should be able to DMA firmware into FECS/GPCCS IMEM.
   The WPR is still configured — we just need to fix the DMA path.

2. **WPR register capture**: During the next nouveau session, capture the exact
   WPR register values and SEC2 DMEM state (command queues, instance block address)
   before teardown. This gives us the DMA configuration to reconstruct.

3. **Livepatch evolution**: Intercept nouveau's falcon halt path so FECS stays
   running through the entire nouveau session. This would make the diesel engine
   pattern work directly.

4. **PMU bootstrap**: PMU is in HS mode and halted. If we can restart PMU
   (which has authority over the WPR chain), it could re-bootstrap the entire
   ACR chain from scratch.

## K80 (GK210) — Sovereign Cold Boot

### Phase 1: Baseline state

```
BOOT0:      0x0f22d0a1 (hardware default — never POSTed)
PMC_ENABLE: 0xc0002020 (minimal engines)
PTIMER:     0xbad0da1d (PRI fault — no clocks)
FECS:       0xbadf1200 (PRI fault — PGRAPH not enabled)
VRAM:       dead
```

### Phase 2: Nouveau Cold-POST

```
coralctl cold-post 0000:4c:00.0 --settle 25
```

Result: nouveau cannot POST GK210. BOOT0 remains 0x0f22d0a1, all registers
unchanged. Nouveau detects the invalid BOOT0 and aborts init.

### Phase 3: Sovereign Cold Boot

```
coralctl cold-boot 0000:4c:00.0 \
  --recipe .../gk210_full_bios_recipe.json \
  --firmware-dir .../firmware/nvidia/gk110 \
  --pgraph --pccsr
```

Result: clock init (64 registers) + devinit (156 registers) applied successfully.
FECS firmware uploaded and reported "RUNNING". But post-boot registers all read
`0xffffffff` — the GPU dropped off the PCIe bus. The cold boot recipe put the
GPU into a state that caused PCIe completion timeouts.

**Post-cold-boot state**: K80 die #1 in D-state (PCI remove timed out after 10s).
Device requires reboot to recover. Rest of system unaffected (USB/Titan V/RTX 5070 OK).

### K80 Analysis

The sovereign cold boot recipe works for clocks and devinit but causes a PCIe
hang. The root cause is likely:

1. **Missing memory training**: GDDR5 never trained, so any register access that
   touches the memory controller (directly or indirectly) causes PCIe completion
   timeouts that cascade to a bus hang.

2. **Recipe incompleteness**: The nvidia-470 recipe was captured from a VM session
   where the GPU was already POSTed. Some register writes assume VRAM is alive
   and access memory-mapped regions that don't exist on a cold GPU.

### K80 Next Steps

1. **agentReagents VM POST** (QUICK WIN): Pass K80 to a micro-VM running nvidia-470
   (the documented path). nvidia-470 does full VBIOS POST including GDDR5 memory
   training. After POST, swap back to VFIO. The sovereign compute pipeline
   (Kepler QMD v1.7 + push buffer + dispatch) is 100% code-complete.

2. **Recipe surgery**: Separate the nvidia-470 recipe into phases:
   - Phase 0: Clock domains (safe — no memory access)
   - Phase 1: PMC + engine enable (safe)
   - Phase 2: Memory training (DANGEROUS — must be exactly right)
   - Phase 3: PFIFO + GR init (requires VRAM)
   Apply only phases 0-1 on cold hardware, skip 2-3 until memory is trained.

3. **GDDR5 memory training capture**: During a VM POST session, capture the exact
   register sequence for GDDR5 memory training (PFB domain: 0x100000-0x100FFF,
   PFBPA: 0x10F000, PMFB: 0x130000). Implement in devinit interpreter.

4. **K80 die #2 (4d:00.0)**: Use die #2 for next cold boot experiments to avoid
   blocking die #1 recovery (die #1 needs reboot).

## Architectural Observations

### Evolutionary Pattern Validated

The iteration-by-iteration approach works:

| GPU | Software | Hardware Barrier | Solution Path |
|-----|----------|------------------|---------------|
| **Titan V** | Diesel engine pipeline, ACR solver, 14 strategies | SEC2 DMA context destroyed after nouveau | Reconstruct DMA tables for Strategy 8 |
| **K80** | Full Kepler dispatch pipeline (QMD v1.7, pushbuf, PFIFO) | GDDR5 not trained (never POSTed) | VM POST via agentReagents, or sovereign memory training |

### Cross-Stack Leverage

- **coralReef** IR layer: Once FECS boots, coralReef compiles WGSL→SASS for
  both SM37 (Kepler) and SM70 (Volta). The sovereign compiler is architecture-aware.
- **barraCuda**: Tensor math + dispatch is ready. hotSpring validation binaries
  can target sovereign dispatch via `--features sovereign-dispatch`.
- **toadStool**: Hardware discovery via metalForge probe. Capability-based
  routing (F64 rate, precision strategy) is wired.

### Key Breakthrough: SEC2 is Alive

Strategy 8's proof that SEC2 accepts and processes BOOTSTRAP_FALCON commands
from the host changes the Titan V equation. We don't need to solve the full
PMU→WPR→SEC2 chain from scratch — SEC2 is already authenticated and running
in HS mode from nouveau's ACR boot. We just need to give it a valid DMA
context to fetch firmware from WPR.

## Register State Reference

### Titan V (post-nouveau warm-fecs, stable)

| Register | Value | Status |
|----------|-------|--------|
| PMC_ENABLE (0x200) | 0x5fecdff1 | All engines |
| FECS_CPUCTL (0x409100) | 0x00000010 | Halted |
| FECS_SCTL (0x409240) | 0x00003000 | **HS mode** |
| GPCCS_CPUCTL (0x41a100) | 0x00000010 | Halted |
| SEC2_CPUCTL (0x87100) | 0x00000010 | Halted (but restartable) |
| SEC2_SCTL (0x87240) | 0x00003000 | **HS mode** |
| PMU_CPUCTL (0x10a100) | 0x00000010 | Halted |
| PMU_SCTL (0x10a240) | 0x00003000 | **HS mode** |
| WPR2_ADDR_LO (0x100CD4) | 0x00fffe02 | Configured |
| WPR2_ADDR_HI (0x100CD8) | 0x08000005 | Configured |
| VRAM | ALIVE | Write-readback OK |

### K80 die #1 (post-cold-boot, PCIe hung)

| Register | Value | Status |
|----------|-------|--------|
| All BAR0 | 0xffffffff | **PCIe hung** — needs reboot |

### K80 die #2 (untouched, cold)

| Register | Value | Status |
|----------|-------|--------|
| BOOT0 (0x0) | 0x0f22d0a1 | Hardware default (never POSTed) |
| PMC_ENABLE (0x200) | 0xc0002020 | Minimal |
| VRAM | dead | Never trained |
