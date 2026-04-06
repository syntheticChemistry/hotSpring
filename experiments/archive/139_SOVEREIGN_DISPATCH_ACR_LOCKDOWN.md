# Experiment 139: Sovereign Dispatch — ACR Lockdown Discovery

**Date:** 2026-04-02/03  
**GPUs:** Titan V (GV100, sm_70), K80 (GK210, sm_37)  
**Status:** Blocked (Titan V) / Blocked (K80 — cold, needs POST)  
**Follows:** Exp 132 (Ember Frozen Warm Dispatch), Exp 134 (K80 Cold Boot)

## Objective

Execute the first sovereign compute dispatch: WGSL → CoralIR → SASS → GPU execution,
entirely through the coralReef stack (coral-parse, coral-reef compiler, coral-driver VFIO).

## Pipeline Validated

```
WGSL source → coral-parse → CoralIR → coral-reef SASS codegen → GPFIFO submit → [FECS scheduling] → GPU
```

Compilation stages (WGSL→CoralIR→SASS) work correctly. The block is at FECS scheduling.

## Key Discovery: ACR Lockdown on Volta

### The Problem

After `warm-fecs` (nouveau boots FECS, swap back to vfio-pci), FECS is in **HS idle-halt**:

```
FECS_CPUCTL = 0x00000010  (halted=true, stopped=false)
FECS_SCTL   = 0x00003000  (bit12: AUTH_MODE, bit13: ACR_LOCKDOWN)
```

ACR lockdown (SCTL bits 12:13 ≥ 2) **prevents all host register writes** to falcon control
registers. Every wake strategy fails:

| Strategy | Result |
|---|---|
| SWGEN0 interrupt (IRQSSET bit 6) | Write silently ignored by ACR lockdown |
| STARTCPU via CPUCTL | Blocked — CPUCTL locked in HS mode |
| STARTCPU via CPUCTL_ALIAS | Sets bit 1 but falcon re-halts immediately |
| ENGCTL soft-reset (bit 0) | Write accepted but no effect |
| Runlist doorbell (RUNLIST_SUBMIT) | Hardware scheduler interrupt doesn't wake FECS |

### Root Cause of D-state

Our FECS wake attempts (IRQSSET, IRQMSET, IRQMODE, CPUCTL writes) hit ACR-locked registers,
generating **PRI ring faults** (0xbadf5040). These faults poison the PRI ring, causing
subsequent PFIFO register accesses (runlist submit, GPFIFO doorbell) to fail silently.
The GPFIFO never advances (gp_get=0, gp_put=1) → fence timeout.

When the VFIO device fd is closed on a GPU with a poisoned PRI ring, `vfio_pci_core_disable`
accesses PCI config space, triggering a PCIe completion timeout → kernel D-state.

### Fix Applied

**HS-mode safe path** in `restart_warm_falcons()` (init.rs):
- Detects HS mode (`(sctl >> 12) & 3 >= 2`) + halted
- Skips ALL falcon register writes (no IRQSSET, no STARTCPU, no IRQMODE)
- Only clears PRI ring faults and PFIFO interrupts
- Re-submits runlist (last resort — hardware scheduler may wake FECS)
- Result: no PRI ring pollution, clean fence timeout, safe device teardown

## PFIFO Config Fix

Switched `open_warm` from `warm_handoff()` to `warm_fecs()` PFIFO config:
- `warm_handoff`: no PBDMA clear, no runlist flush → stale nouveau entries on runlist
- `warm_fecs`: PBDMA force-clear, empty runlist flush → clean scheduling state

## BOOT0 Health Gate

Added early BAR0 BOOT0 check in both `open_warm()` and `open_from_fds()`:
- If BOOT0 = 0xFFFFFFFF → return error immediately before any channel/PFIFO operations
- Prevents D-state from operating on an unresponsive GPU
- K80 cold dispatch now fails cleanly: "BAR0 BOOT0=0xFFFFFFFF — GPU not responding"

## Cold Boot Fallback

Sovereign dispatch handler now tries warm→cold fallback:
1. Try `open_warm()` (preserves FECS from warm-fecs)
2. If warm fails (cold GPU, PRI faults) → request fresh ember fds → `open_from_fds()` (cold init)
3. If ember unavailable → `open()` (direct VFIO open)

Also fixed Kepler firmware loading: `FecsFirmware::load()` now handles missing `fecs_bl.bin`
(Kepler has no bootloader — direct IMEM load with BOOTVEC=0).

## coralctl dispatch --sovereign

Evolved the CLI to support sovereign dispatch natively:

```bash
coralctl dispatch 0000:03:00.0 --sovereign \
  --shader /path/to/shader.wgsl \
  --input /path/to/input.bin \
  --output-size 1024 \
  --workgroups 1,1,1
```

No more Python socket scripts. The handler prints compilation stats, output previews,
and writes output buffers to `--output-dir` if specified.

## K80 Status

- `nouveau` doesn't recognize GK210 chipset (BOOT0=0x0f22d0a1 → "unknown chipset")
- `nvidia` 580.x dropped Kepler support
- Created `/lib/firmware/nvidia/gk210` → `gk20a` symlink for firmware access
- K80 is un-POSTed (PMC_ENABLE=0xc0002020, most engines disabled)
- BOOT0 health gate prevents D-state on cold K80
- **Next step**: cold-POST via devinit or external mechanism

## Path Forward

### Titan V (Volta — HS mode)

The ONLY way to restart FECS under ACR lockdown is through the ACR chain:

```
Host → SEC2 CMDQ → SEC2 ACR → BOOTSTRAP_FALCON(FECS) → FECS restarts
```

Requirements:
1. Capture SEC2 DMEM layout during warm-fecs (when nouveau has SEC2 alive)
2. Parse CMDQ base address, read/write pointers, command format
3. After swap to vfio-pci, write BOOTSTRAP_FALCON command to SEC2's CMDQ in DMEM
4. SEC2 processes command → authenticates and restarts FECS
5. FECS enters scheduling loop → our channel gets scheduled → dispatch completes

### K80 (Kepler — no ACR)

Cold-POST path: enable GR engine via PMC_ENABLE, load gk20a FECS firmware directly
to IMEM/DMEM, STARTCPU (no ACR lockdown on Kepler). This is the simpler path to
prove sovereign compute end-to-end.

## Files Changed

- `coral-driver/src/nv/vfio_compute/init.rs` — HS-mode safe path, FECS warm detection
- `coral-driver/src/nv/vfio_compute/mod.rs` — BOOT0 health gate, warm_fecs PFIFO config
- `coral-driver/src/nv/vfio_compute/fecs_boot.rs` — Kepler BL-optional firmware loading
- `coral-driver/src/vfio/channel/mod.rs` — `submit_runlist` made pub
- `coral-glowplug/src/socket/handlers/compute.rs` — warm→cold fallback dispatch
- `coral-glowplug/src/bin/coralctl/main.rs` — `--sovereign` flag + `--sm` override
- `coral-glowplug/src/bin/coralctl/handlers_device/mod.rs` — `rpc_dispatch_sovereign`
- `scripts/deploy.sh` — also updates `~/.cargo/bin` to prevent PATH shadow
