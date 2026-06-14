# Experiment 151: Revalidation — Fossil Record Review and Next Stages

**Date**: 2026-04-06
**GPUs**: Titan V (GV100, 0000:03:00.0), Tesla K80 (GK210, 0000:4c:00.0 + 0000:4d:00.0)
**Status**: REVALIDATION DOCUMENT — synthesizes Exp 058-150 into actionable foundation
**Predecessors**: All sovereign compute experiments (058-150)

---

## 1. What We've Built (Infrastructure)

### coralReef Sovereign Stack

| Layer | Component | Status |
|-------|-----------|--------|
| **Lifecycle** | coral-glowplug | Daemon manages ember lifecycle, heartbeat, resurrection |
| **GPU Interface** | coral-ember | Sacrificial canary — all GPU access routes through MMIO gateway RPCs |
| **Isolation** | Fork-isolated MMIO | Every BAR0 op runs in expendable child process; SBR + bus master kill on timeout |
| **Recovery** | Emergency quiesce | Disable bus master → drop BAR0 → mark faulted → voluntary death if all faulted |
| **Signal** | SIGTERM handler | Clean shutdown: bus master OFF on all devices before exit |
| **Driver** | coral-driver | VFIO device management, falcon upload, DMA safety, PRAMIN access |
| **Client** | ember_client | JSON-RPC client for experiments to call ember |
| **CLI** | coralctl | K80 cold boot pipeline, device management |

### Test Coverage

- coral-ember: 170 tests pass, 461 glowplug tests pass
- Integration: 17 ipc_dispatch tests pass
- Hardware: exp145 runs full ACR pipeline without system lockup

### Key Architectural Decisions

1. **Ember IS the GPU** — no bypass paths. `vfio_fds` deprecated. Test harness panics without ember.
2. **Fork isolation** — PRAMIN writes, falcon uploads, STARTCPU, falcon_poll all fork-isolated.
3. **Bus master kill** — after any fault, GPU DMA disabled via sysfs before anything else.
4. **Warm cycle** — `gpu_warm.sh` binds to nouveau briefly for VRAM initialization, then back to VFIO.

---

## 2. Titan V (GV100) — Investigation Summary

### Phase 1: Falcon Discovery (Exp 058-095)

Established the VFIO pipeline, GlowPlug daemon, discovered falcon boot sequences,
achieved GPCCS running state. Proved sovereign GPU compute is architecturally possible.

### Phase 2: WPR and ACR (Exp 110-126)

Explored WPR2 preservation across driver swaps, cold boot WPR2, sovereign DEVINIT.
Key finding: WPR2 region must be established by firmware before ACR can authenticate.

### Phase 3: Dual Boot + DMA (Exp 132-139)

Attempted dual GPU sovereign boot (K80 + Titan V). Discovered SEC2 DMA path requirements:
FBIF VIRT mode, sysmem page tables, falcon MMU routing. D-state root cause analyzed.

### Phase 4: Root Cause Narrowing (Exp 140-143)

| Discovery | Experiment | Impact |
|-----------|------------|--------|
| SEC2 is at PMC bit 5, not bit 22 | Exp 140 | All prior ACR attempts toggled wrong engine |
| VBIOS DEVINIT suspected | Exp 141 | Thought crypto engine uninitialized after SBR |
| DEVINIT hypothesis **contradicted** | Exp 142-143 | ACR fails even on BIOS-POSTed GPU (no SBR) |

### Phase 5: Ground Truth (Exp 144)

**The definitive diagnostic session.** Multiple sub-tests in one experiment:

| Finding | Detail | Impact |
|---------|--------|--------|
| SEC2 reset works with bit 5 | Scrub: 627µs (was 3s timeout) | PMC enable is now correct |
| IMEM PIO writes verified | Readback matches perfectly | LS mode does NOT block PIO IMEM |
| BOOTVEC is ignored | All 4 test values → same PC=0x0056 | Falcon resumes from ROM halt, not BOOTVEC |
| **VRAM is dead** | FBPA=0xbadf3000, PRAMIN=0xbad0ac0X | ALL prior VRAM writes silently failed |
| Instance block bind stalls | bind_stat=0x000e003f (never reaches state 5) | Can't walk page tables without VRAM |

### Phase 6: Sacrificial Ember + Warm GPU (Exp 145-150, today)

With warm GPU (nouveau cycle), PRAMIN is live. Exp 145 runs full pipeline:
- WPR write succeeds
- Instance block binding polls (bind_stat completes)
- BL code uploaded to IMEM, verified
- STARTCPU executes, BL runs (TRACEPC shows execution)
- SEC2 halts: `SCTL=0x3000, HS=false, PC=0x03fb, MB0=0xcafebeef`

**The BL executes but does not achieve HS mode.**

---

## 3. Tesla K80 (GK210) — Investigation Summary

### Architecture Advantage

Kepler has **no firmware security** — no FWSEC, no ACR, no WPR, no crypto chain.
If VRAM is live, sovereign compute should be achievable with:
FECS PIO upload → PGRAPH context switch → shader dispatch

### What Works (Exp 123, 133-135)

- Clock init: 258 nvidia470 registers applied, PTIMER ticks
- DEVINIT: 315 PMC/PBDMA registers applied
- FECS PIO upload: firmware starts, writes status to scratch0 (`0x802FEF0B`)
- FECS falcon core registers (0x409000-0x409500) accessible

### What Fails

- **VRAM dead** — never BIOS-POSTed (VFIO-bound from boot)
- PGRAPH CTXSW domain (0x409504+) PRI-faults
- GPCCS, GR hub, GPC all PRI-fault
- FECS_STATUS (0x409800) unreachable

### Root Cause

K80 was bound to vfio-pci at boot. GDDR5 memory training never ran. Without
VRAM, PGRAPH can't operate and FECS can't signal boot completion.

---

## 4. Validated State — What We Know For Certain

### Titan V

| Fact | Evidence |
|------|----------|
| SEC2 PMC bit is 5 | Exp 140: scrub 627µs vs 3s timeout with bit 22 |
| PIO IMEM writes work in LS mode | Exp 144: readback matches all 4 words |
| BOOTVEC is ignored on STARTCPU | Exp 144: 4 different values, same PC=0x0056 |
| VRAM requires warm cycle | Exp 144: FBPA=0xbadf3000 when cold; live after nouveau |
| BL code executes after warm | Exp 145 (Apr 6): TRACEPC shows instruction path |
| BL does not achieve HS mode | Exp 145: SCTL=0x3000, HS=false, MB0 unchanged |
| System survives full pipeline | Exp 145 (Apr 6): no lockup with sacrificial ember |
| Instance block binding works (warm) | Exp 145: bind_stat polls complete |
| BROM registers return badf5040 | Exp 145: ModSel, UcodeId, EngIdMask, ParaAddr0 |

### Tesla K80

| Fact | Evidence |
|------|----------|
| Clock init works from cold | Exp 135: 258 registers, PTIMER alive |
| FECS PIO upload works | Exp 135: firmware executes, scratch0=0x802FEF0B |
| VRAM dead without POST | Exp 135: 0xFFFFFFFF everywhere |
| PGRAPH domain faults above 0x409504 | Exp 135: boundary at exactly this address |
| No firmware security barrier | Architecture: Kepler has no FWSEC/ACR/WPR |

### Both GPUs

| Fact | Evidence |
|------|----------|
| VRAM initialization is the gate | Both fail the same way without VRAM |
| Nouveau warm cycle initializes VRAM | Titan V proven; K80 expected (needs test) |
| Ember sacrificial architecture works | Apr 6: exp145 full pipeline, no lockup |
| Fork-isolated MMIO prevents system crash | All handlers isolated, bus master killed on fault |

---

## 5. Open Questions

### Titan V — Why doesn't the BL achieve HS mode?

**Hypothesis A: Missing crypto context.** The BL's HS authentication requires
keys or signatures that depend on prior PMU initialization. PMU establishes
the WPR2 region and sets up the crypto keystore. Without PMU, the BL's
signature verification fails silently.

**Hypothesis B: DMA descriptor mismatch.** The BL descriptor points to virtual
addresses (ctx_dma=1) but the instance block's page table mapping may not
cover the ACR firmware region correctly. The BL tries to DMA the ACR payload
and gets garbage.

**Hypothesis C: BROM state not initialized.** BROM registers (ModSel, etc.)
return badf5040. Nouveau may write these before STARTCPU. Without them, the
BROM verification step can't find the firmware to authenticate.

**Hypothesis D: Missing DMEM initialization.** The BL expects specific DMEM
state beyond what we write (data section + descriptor). Nouveau may write
additional configuration that we're missing.

### Tesla K80 — Will nouveau warm cycle give us VRAM?

Expected yes (same pattern as Titan V), but not yet tested. If VRAM comes
alive, K80 should be solvable quickly since there's no firmware security.

### Both — What does nouveau's SEC2 boot sequence look like register-by-register?

An mmiotrace or warm-handoff livepatch capture of nouveau's exact sequence
would reveal any steps we're missing.

---

## 6. Next Stages

### Stage 1: K80 Warm Cycle + Sovereign Compute (fastest win)

**Goal**: First end-to-end sovereign shader dispatch on real hardware.

1. Modify boot config to bind K80 to nouveau at startup
2. Warm cycle: nouveau loads → VRAM alive → swap to VFIO
3. FECS PIO upload on warm K80 (PGRAPH domain should now work)
4. Attempt PGRAPH context switch + compute shader dispatch
5. If successful: first sovereign compute on consumer hardware without proprietary drivers

**Why K80 first**: No ACR barrier. FECS already executes firmware. Only VRAM was missing.

### Stage 2: Titan V — PMU-First Boot

**Goal**: Test whether PMU must precede SEC2 for ACR HS mode.

1. Load PMU firmware (from nouveau firmware files)
2. Boot PMU falcon before SEC2
3. Check if PMU establishes WPR2 region
4. Then attempt SEC2 ACR boot
5. Compare BROM register state before/after PMU boot

**Why**: Exp 113 identified PMU dependency. Exp 141-143 showed DEVINIT alone isn't enough.
PMU may be the missing link between "BL executes" and "HS achieved."

### Stage 3: Titan V — Post-BL Forensics

**Goal**: Understand exactly why the BL halts.

1. After BL halt (PC=0x03fb), read back ALL of DMEM
2. Check for error codes, status words, or diagnostic output
3. Map TRACEPC addresses against the BL binary to find divergence point
4. Read falcon EXCI register in detail (0x001f000f = which exception?)

### Stage 4: Exp 150 — Systematic Probe (now safe)

**Goal**: Map every register's behavior on warm vs cold GPU.

With sacrificial ember, we can safely execute all 14 probe phases.
If any probe kills ember, glowplug resurrects it and we log which operation triggered it.
This builds the complete register safety map for both GPUs.

### Stage 5: Nouveau SEC2 Boot Capture

**Goal**: Capture the exact register sequence nouveau uses for SEC2 boot.

Options:
- mmiotrace during nouveau load (captures all BAR0 writes)
- Warm-handoff livepatch (Exp 125 proved this works)
- Compare register-by-register against our sequence

### Stage 6: Cross-GPU Comparative Debugging

**Goal**: Use K80 success (no security) vs Titan V failure (secured) to isolate the ACR barrier.

Run identical probe sequences on both warm GPUs. Where K80 succeeds and Titan V fails
pinpoints exactly which step requires the ACR chain. This is the most data-efficient
way to narrow the remaining gap.

---

## 7. Hardware State Reference

```
Fleet:
  Titan V  (GV100, sm_70)  @ 0000:03:00.0  — VFIO-bound, needs warm cycle for VRAM
  K80 die1 (GK210, sm_37)  @ 0000:4c:00.0  — VFIO-bound, never POSTed, VRAM dead
  K80 die2 (GK210, sm_37)  @ 0000:4d:00.0  — VFIO-bound, never POSTed, VRAM dead
  RTX 5070 (GB206, sm_120) @ primary        — Host GPU (do not touch)

Services:
  coral-glowplug: active (manages ember lifecycle)
  coral-ember: active (holds 3 GPUs, MMIO gateway)

Key scripts:
  scripts/gpu_warm.sh — Titan V warm cycle (nouveau bind/unbind)
  scripts/boot/glowplug.toml — Device configuration
```

---

## 8. Absorption Opportunities for Primal Teams

### coralReef evolution needed

- **FBPA/LTC initialization** — sovereign VRAM init without driver dependency (Option B from Exp 144)
- **PMU falcon boot** — add PMU upload/start to ember's falcon handlers
- **Kepler compute dispatch** — PGRAPH context switch + QMD v1.7 + shader dispatch path
- **mmiotrace integration** — capture nouveau's boot sequence via ember interface

### toadStool / barraCuda impact

- Once sovereign dispatch works on K80, the GPU compute pipeline connects to barraCuda's shader runtime
- DF64 shaders can run on sovereign hardware without proprietary drivers

### hotSpring science impact

- Sovereign K80 dispatch → GPU MD / lattice QCD on fully sovereign pipeline
- Cross-architecture parity: same shaders on sovereign K80 vs wgpu RTX 5070
