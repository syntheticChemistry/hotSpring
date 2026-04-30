# hotSpring → coralReef: K80 PGOB nvidia-470 Binary Analysis Handoff

**From:** hotSpring
**To:** coralReef, primalSpring
**Date:** April 29, 2026
**Experiment:** 178

## Summary

Static binary analysis of proprietary nvidia-470 kernel module revealed the PGOB
(Power Gating On Board) sequence for GK210B. Two functions identified:
`_nv029216rm` (ungate) and `_nv029114rm` (gate). PSW-only approach at `0x10a78c`
uses bits 0-1 only — skips `0x0205xx` power domain steps entirely.

## Key Finding

**PSW-only requires running PMU firmware.** The PSW handshake needs the PMU falcon
actively processing commands. Without loaded firmware, register writes are no-ops.

## Root Cause Narrowed

PRI ring shows `pri_gpc_cnt=0` — zero GPC stations enrolled. GPCs aren't just
power-gated, they're absent from PRI topology. The `0x0205xx` power steps succeed
(no PRIVRING faults) but GPC PRI routes remain broken.

## Artifacts Delivered

- `nvidia470_pgob_disable()` / `nvidia470_pgob_enable()` in `coral-driver/pgob.rs`
- Build recipe: `agentReagents/tools/k80-sovereign/build_nvidia470_kernel617.sh`
- Analysis doc: `agentReagents/tools/k80-sovereign/nvidia470_pgob_analysis.md`
- QEMU VM reagent template for K80 VFIO passthrough

## Path Forward

Cold sovereign boot (PGOB → PRI enrollment) remains blocked on PMU firmware load.
Pivoted to **warm-catch approach** (Exp 179): let Nouveau perform full GR init,
then VFIO rebind for sovereign dispatch. This approach has yielded FECS boot and
PFIFO runlist completion.
