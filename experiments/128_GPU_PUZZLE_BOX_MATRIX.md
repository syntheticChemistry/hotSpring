# Experiment 128: GPU Puzzle Box — Multi-Path Sovereign Compute Matrix

**Date**: 2026-03-28
**Status**: SUPERSEDED by Exp 141 (ACR HS Auth Root Cause). The puzzle box converged to VBIOS DEVINIT as the single remaining blocker. See Exp 142 for the resolution path.
**Depends on**: Exp 125 (livepatch breakthrough), Exp 127 (warm FECS dispatch), Exp 123 (K80 sovereign)

## Motivation

Exp 127 confirmed that on Volta HS+ (Titan V), FECS firmware survives the
`nouveau` → `vfio-pci` swap via livepatch, but enters an idle HALT state that
cannot be resumed by the host. The problem shifted from **preservation** to
**resumption**. This experiment implements a matrix of parallel solution paths
across both the Tesla K80 (Kepler, unsigned) and Titan V (Volta, HS+ signed).

## Architecture

Two GPUs, related generations, solving the same fundamental problem from
different angles. K80 validates dispatch infrastructure (no security barriers),
Titan V explores FECS lifecycle control.

## Track A: K80 Cold Boot (Kepler)

### A1: Full nvidia-470 Recipe Replay (DONE)

`k80_cold_boot.rs` now accepts a `ColdBootConfig` that controls which register
domains to include. The new `ColdBootConfig::full()` preset includes PGRAPH
(priority 30), PCCSR (priority 35), and PRAMIN (priority 40) — previously these
were excluded.

The `cold_boot()` function signature gained a `config: &ColdBootConfig` parameter,
and the result struct now includes `pgraph_replay: Option<ReplayResult>`.

**Files changed**:
- `coral-driver/src/vfio/channel/diagnostic/k80_cold_boot.rs`

### A2: FECS Boot After Full Replay (DONE)

Hardware test `exp128a2_full_recipe_fecs_boot` added to
`tests/exp123k_k80_sovereign.rs`. Orchestrates:
1. Engine enable (PMC_ENABLE full)
2. nvidia-470 cold→warm diff replay (ALL registers including PGRAPH)
3. GR reset toggle for clean falcon state
4. `kepler_falcon::boot_fecs_gpccs()` PIO upload + STARTCPU

**Awaiting hardware run** on K80 to validate.

### A3: Kepler GPFIFO Channel Dispatch (DONE)

Hardware test `exp128a3_kepler_gpfifo_dispatch` implements a skeletal channel
setup:
- PFIFO init + interrupt clear
- FECS method 0x10 (CTX_IMAGE_SIZE) to confirm FECS responsive
- Channel instance block setup in PRAMIN
- GPFIFO ring buffer + USERD allocation
- CCSR channel bind + GP_PUT NOP submission

This is a minimal validation of the PFIFO/PBDMA/channel pipeline. Full compute
dispatch requires MMU page table setup and compute class binding (0xA1C0).

## Track B: Titan V Warm FECS

### B1: Pre-Swap Keepalive (DONE)

`coralctl warm-fecs --keepalive` now spawns a subprocess that holds an open DRM
render node fd during the swap window. This prevents FECS from seeing "no
channels" in its scheduling loop and entering idle-halt.

The process is killed after `vfio-pci` binds successfully.

**Files changed**:
- `coral-glowplug/src/bin/coralctl/main.rs` (new `--keepalive` flag)
- `coral-glowplug/src/bin/coralctl/handlers_device/mod.rs` (spawn + cleanup logic)

### B2: nvidia Proprietary Warm Handoff (DONE)

`coralctl warm-fecs-nvidia <bdf>` implements the full flow:
1. Capture pre-swap FECS state via BAR0 sysfs
2. Swap to nvidia proprietary driver via Ember
3. Wait for RM initialization
4. Capture FECS state under nvidia RM (CPUCTL, SCTL, PC, SEC2)
5. Swap to vfio-pci
6. Capture post-swap FECS state
7. Diff analysis: survived? halted? running?

This path reveals how RM manages FECS lifecycle differently from nouveau. Even
if the warm handoff fails with the same HALT, the register traces teach us the
correct FECS initialization sequence.

**Files changed**:
- `coral-glowplug/src/bin/coralctl/main.rs` (new `WarmFecsNvidia` command)
- `coral-glowplug/src/bin/coralctl/handlers_device/mod.rs` (`rpc_warm_fecs_nvidia`)

### B5: Timing Attack (DONE)

`coralctl warm-fecs --poll-fecs` implements a 50ms-interval BAR0 register poll
that watches FECS CPUCTL for a running state (bit4=0, bit5=0). After a 2s
minimum init wait, it polls continuously and triggers the swap the instant FECS
is seen running, before it can enter its idle-halt loop.

Uses `read_bar0_u32()` — safe file I/O (seek + read) on sysfs `resource0`.

## Track C: Cross-Cutting

### C1: FECS Method Enumeration (DONE)

Implemented `fecs_stop_ctxsw()` (method 0x01) and `fecs_start_ctxsw()` (method
0x02) in `fecs_method.rs`. These control FECS's context scheduling:
- `STOP_CTXSW`: freezes scheduling without halting FECS — FECS stays running
  but won't switch contexts. Critical for the warm handoff window.
- `START_CTXSW`: resumes scheduling after a stop.

Both use the standard `fecs_ctrl_ctxsw` path (write MTHD_DATA + MTHD_CMD, poll
STATUS2 for completion).

### C2: Fix CPUCTL Bit Labeling (DONE)

The `CPUCTL_HRESET` (bit 4) and `CPUCTL_HALTED` (bit 5) constants were inverted
throughout the codebase. Exp 127 root cause analysis confirmed:
- Bit 4 = firmware HALT (executed HALT instruction, idle loop)
- Bit 5 = CPU STOPPED (idle)

**Rename applied**:
- `CPUCTL_HRESET` → `CPUCTL_HALTED` (bit 4)
- `CPUCTL_HALTED` → `CPUCTL_STOPPED` (bit 5)

Updated across 30+ files in coral-driver, coral-ember, and coral-glowplug.
Serde aliases added for backward JSON compatibility.

## Test Matrix

| Test | Command | GPU | Status |
|------|---------|-----|--------|
| K80 full recipe + FECS boot | `sudo cargo test exp128a2 ...` | K80 | Pending HW |
| K80 GPFIFO channel dispatch | `sudo cargo test exp128a3 ...` | K80 | Pending HW |
| Timing attack warm-fecs | `coralctl warm-fecs --poll-fecs <bdf>` | Titan V | Pending HW |
| Keepalive warm-fecs | `coralctl warm-fecs --keepalive <bdf>` | Titan V | Pending HW |
| nvidia warm handoff | `coralctl warm-fecs-nvidia <bdf>` | Titan V | Pending HW |
| STOP_CTXSW + warm-fecs | Requires integration | Titan V | Planned |

## Next Steps

1. **Run K80 tests** — validate that full nvidia-470 recipe replay unblocks
   FECS (PC != 0 after STARTCPU). If clocks are still wrong, diff the full
   recipe against our replay to find missing writes.

2. **Run Titan V timing attack** — if FECS is caught running and survives the
   swap, we've solved the warm handoff problem.

3. **STOP_CTXSW integration** — before enabling the livepatch, issue
   `fecs_stop_ctxsw()` to freeze FECS scheduling. This keeps FECS running
   (not halted) while preventing it from discovering the empty runlist.

4. **nvidia RM register trace** — compare FECS state under nvidia vs nouveau
   to learn what RM does differently during initialization.

5. **B3: ACR reboot** — if none of the above work, attempt to re-trigger the
   full ACR boot chain from vfio by manipulating SEC2 and FWSEC.
