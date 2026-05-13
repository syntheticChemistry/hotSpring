# Experiment 178: K80 PGOB nvidia-470 Binary Analysis

**Date:** April 29, 2026
**Status:** ⚠️ In Progress (PSW-only requires PMU firmware)
**Hardware:** Tesla K80 (GK210B)
**Primal:** coralReef (coral-driver)

## Objective

Reverse-engineer the proprietary nvidia-470 PGOB (Power Gating On Board) sequence
for GK210B via static binary analysis, to ungate GPCs and enable GR engine access.

## Method

1. Built nvidia-470.256.02 kernel module for kernel 6.17 (`build_nvidia470_kernel617.sh`)
2. QEMU VM with K80 VFIO passthrough — module probed successfully
3. mmiotrace empty (VFIO BAR mapping bypass) — pivoted to static disassembly
4. Identified `_nv029216rm` (ungate) and `_nv029114rm` (gate) in `nv-kernel.o_binary`

## Findings

- **PSW-only PGOB sequence at `0x10a78c`**: nvidia-470 uses only bits 0-1 of the
  PSW register. Skips `0x0205xx` power domain steps entirely.
- **PSW handshake requires running PMU firmware**: Without loaded PMU firmware,
  the PSW register writes are no-ops.
- **`0x0205xx` power steps succeed** on GK210B (no PRIVRING faults as previously
  thought) but GPC PRI routes remain broken.
- **Root cause narrowed**: PRI ring shows `pri_gpc_cnt=0` — zero GPC stations
  enrolled. GPCs aren't just power-gated, they're absent from PRI topology.
- **Two paths forward**: (1) PRI ring GPC enrollment, (2) PMU firmware load for
  PSW processing.

## Artifacts

- `agentReagents/tools/k80-sovereign/nvidia470_pgob_analysis.md` — full disassembly
- `agentReagents/tools/k80-sovereign/build_nvidia470_kernel617.sh` — build recipe
- `coral-driver/pgob.rs` — `nvidia470_pgob_disable()` / `nvidia470_pgob_enable()`

## Next

Pivot to warm-catch approach: let Nouveau perform full GR initialization (including
PGOB, PRI topology, FECS/GPCCS boot), then VFIO rebind for sovereign dispatch.
See Experiment 179.
