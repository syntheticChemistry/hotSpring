# Experiment 217 — TPC PRI Station Creation

**Date**: 2026-05-21
**Status**: COMPLETE (Trials 1-2 executed — TPC wall confirmed as firmware-dependent)
**Hardware**: 2x NVIDIA Titan V (GV100), BDFs `0000:02:00.0`, `0000:49:00.0`
**Dependency**: Exp 215 (Tier 2 classified, TPC wall identified), Exp 216 (kernel health clean)

## Objective

Break the TPC PRI ring wall that blocks Tier 2 sovereign compute on Volta.
After patched-nouveau warm handoff, GPC fabric survives (6/6 GPCs alive) but
TPC PRI ring stations are missing (`0xBADF5040` at `0x504000`). This experiment
systematically tests three attack paths to create or wake TPC stations.

## Critical Discovery (Pre-Experiment)

**Exp 215 Stage 4 used `StubGspBridge` with no sequences loaded** — the
`sw_nonctx.bin` replay was a no-op. The real GV100 `sw_nonctx.bin` at
`/lib/firmware/nvidia/gv100/gr/sw_nonctx.bin` contains **341 register writes
including 94 TPC broadcast writes** in the `0x419xxx` range. This path was
**never tested with real firmware data.**

Additionally, `compute_device.rs` has a full 5-phase ungating sequence using
`NvGspBridge` that was designed for this scenario but was only tested with the
stub in prior experiments.

## Pre-Experiment State

After patched-nouveau warm handoff (same as Exp 215):

| Register | Value | Interpretation |
|----------|-------|----------------|
| PMC_ENABLE (0x200) | 0x5FECDFF1 | 23 engines alive |
| GPC0 per-unit (0x500000) | 0x8780029F | GPC fabric alive |
| GPC0 TPC0 control (0x504000) | 0xBADF5040 | **TPC PRI station MISSING** |
| GPC0 TPC0 SM0 (0x504200) | 0x000900F0 | SM accessible (different PRI sub-path) |
| CE4 (0x108000) | 0x01004005 | Alive |
| FECS PC | ~0xEAC | HS poll loop — running but cannot dispatch |

## Code Changes

### sovereign_stages.rs

**Stage 4 (modified):** Replaced `StubGspBridge` with `NvGspBridge::new("gv100")`.
Now loads real `sw_nonctx.bin` firmware and applies all 341 register writes
including 94 TPC broadcast writes (`0x419xxx`). Added post-replay TPC register
probe at `0x504000`, `0x504200`, and `0x419C04`.

**Stage 6 (new):** Full 5-phase ungating sequence:
1. CG sweep (21 clock-gate domains) + PRI recovery + PGOB broadcast ungate
2. PRI ringmaster forced enumerate
3. GPC MMU init (6 registers) + extended MMU writes
4. `sw_nonctx.bin` replay via `NvGspBridge` with real GV100 firmware
5. Second PRI recovery pass
6. TPC/CE/FECS probe — if TPC still faulted, destructive PGRAPH reset
   (PMC bit 12 toggle), PRI re-enumerate, second `sw_nonctx.bin` replay

### sovereign_tiers.rs

**`classify_tier()` updated:** Now probes TPC registers across GPC0-5 at
`0x504000 + gpc * 0x8000`. Tier 2 requires GPC + CE + **TPC** all alive.
If GPC/CE alive but TPC missing, GPU stays at Tier 1 — correctly reflecting
the TPC wall. New fields: `tpc_status`, `tpc_alive`.

### sovereign.rs (RPC handler)

**`sovereign.experiment` updated:** Stage range extended to 1-6. Stage 6
is now accessible via `{"bdf": "...", "stage": 6}`. Tier classification
response includes `tpc_status` and `tpc_alive` fields.

### Firmware Verification (nv_gsp_bridge.rs)

`apply_gr_bar0_init()` loads `sw_nonctx.bin` only. `sw_bundle_init.bin` (958
register pairs) and `sw_method_init.bin` (1537 pairs) use the FECS method
protocol (class methods via MTHD_CMD), not direct BAR0. They are not applied
by `apply_gr_bar0_init()` and would require a running FECS method interface.
Trial 3 will investigate if they can be applied post-TPC-wake.

## Trials

### Trial 1: `sw_nonctx.bin` Replay via `NvGspBridge` (Stage 4)

```bash
toadstool sovereign experiment --bdf 0000:02:00.0 --stage 4
```

**Hypothesis:** The 94 broadcast TPC writes (`0x419xxx`) in `sw_nonctx.bin`
may route through the PGRAPH hub to TPC units even though per-GPC addressing
(`0x504xxx`) PRI-faults. The broadcast path uses a different PRI ring routing.

**Success criterion:** `tpc_status` at `0x504000` returns non-`0xBADF` value.

### Trial 2: Full 5-Phase Ungating Sequence (Stage 6)

```bash
toadstool sovereign experiment --bdf 0000:02:00.0 --stage 6
```

**Hypothesis:** The combined CG sweep + PGOB + PRI enumerate + sw_nonctx +
PGRAPH reset sequence may succeed where individual steps failed — some
writes may have ordering dependencies.

**Success criteria:**
- TPC control non-fault
- PBDMA intr_0 bit 28 (DEVICE) clear
- `classify_tier()` returns `WarmCompute` with `tpc_alive = true`

### Trial 3: `sw_bundle_init.bin` + `sw_method_init.bin` (Post-TPC-Wake)

Only relevant if Trial 1 or 2 succeeds and TPCs wake up. These firmware blobs
use the FECS method protocol:

```
FECS MTHD_DATA_WR → MTHD_CMD → poll MTHD_CMD bit 31 clear
```

If FECS is responsive with TPCs alive, these method-init sequences may
complete the GR initialization that `sw_nonctx.bin` starts.

## Success Criteria

| Metric | Threshold | Register |
|--------|-----------|----------|
| TPC PRI alive | Non-`0xBADF` | `0x504000` |
| PBDMA DEVICE clear | bit 28 = 0 | PBDMA intr_0 |
| Tier 2 dispatch-ready | `WarmCompute` | `classify_tier()` |
| (stretch) FECS context switch | completes | MTHD_CMD bit 31 |
| (stretch) Compute readback | non-zero | GPFIFO fence semaphore |

## Results

### Trial 1: `sw_nonctx.bin` Replay (Stage 4) — FAIL

**Executed on:** Both Titan Vs (`0000:02:00.0`, `0000:49:00.0`)

| Metric | BDF 02:00 | BDF 49:00 |
|--------|-----------|-----------|
| Firmware present | true | true |
| sw_nonctx replay | completed (real data) | completed (real data) |
| TPC0 control (0x504000) | `0xBADF5040` | `0xBADF5040` |
| TPC0 SM0 (0x504200) | `0x000900F0` | `0x000900F0` |
| Broadcast TPC (0x419C04) | `0x000026F0` | `0x000026F0` |
| PRI recovery | 9 alive / 4 faulted | 9 alive / 4 faulted |
| TPC alive | **false** | **false** |
| Tier after | Tier 1 (WarmInfrastructure) | Tier 1 (WarmInfrastructure) |

**Key finding:** The broadcast TPC writes at `0x419xxx` went through — register
`0x419C04` returned `0x000026F0` (non-fault, accepted value). But per-GPC TPC
control registers at `0x504000` remain `0xBADF5040`. This means:
- The broadcast path works for **configuration registers** accessed via PGRAPH hub
- The per-GPC TPC **control/status registers** are on a different PRI ring sub-path
  that requires TPC PRI ring stations to exist
- `sw_nonctx.bin` writes configure what TPC units *should do* once alive, but
  cannot *create* the PRI stations that make them addressable

### Trial 2: Full 5-Phase Ungating + PGRAPH Reset (Stage 6) — FAIL

**Executed on:** Titan V `0000:02:00.0`

| Phase | Result |
|-------|--------|
| CG sweep | 0 changes, 12 faulted (expected — gated engines) |
| PRI recovery | 9 alive, 4 faulted (consistent) |
| PGOB ungate | PMC_CLKGATE_DISABLE at `0x260` → `0xBAD00200` (PRI fault) |
| PRI enumerate | Forced (value reset to 0) |
| GPC MMU init | 6 registers latched, extended write accepted |
| sw_nonctx replay | completed |
| Post-init PRI | 9 alive, 4 faulted (no change) |
| TPC probe | `0xBADF5040` — **still faulted** |
| PGRAPH reset | PMC bit 12 toggled; FECS PC moved `0x46A6` → `0x0300` |
| Post-reset sw_nonctx | Re-applied |
| Post-reset TPC | `0xBADF5040` — **still faulted** |

**Key findings:**
- `PMC_CLKGATE_DISABLE` at `0x260` returns `0xBAD00200` — even the clock-gate
  disable register PRI-faults, confirming these are unreachable domains
- PGRAPH engine reset (PMC bit 12) does NOT create TPC stations — it merely
  resets existing engine state (FECS PC changed but TPC unchanged)
- CE4 went `0x01004005` → `0x01004405` — minor state bit change from writes
- GPC fabric survived the entire sequence (`gpc0=0x8780029F` throughout)

### Trial 3: `sw_bundle_init.bin` + `sw_method_init.bin` — DEFERRED

As confirmed in firmware verification: these blobs use the FECS method protocol
(MTHD_CMD → MTHD_DATA → poll), not direct BAR0. Since FECS cannot dispatch to
TPCs (TPC stations missing), method-based initialization would fail. Trial 3 is
deferred until TPC stations can be created by another path.

### Twin Study Confirmation

Both Titan Vs produced identical results (same register values, same failure
mode, same PRI recovery counts). This confirms the TPC wall is a fundamental
property of the nouveau warm handoff state, not a per-card anomaly.

### Definitive Conclusion

**TPC PRI ring stations cannot be created by BAR0 register writes alone.** The
stations must be created by signed firmware running on GPCCS (or PMU on behalf
of GPCCS). This is the hardware security boundary — NVIDIA's TPC initialization
is firmware-mediated, not register-mediated.

The broadcast `0x419xxx` path successfully configures TPC *configuration*
registers through the PGRAPH hub, but the per-GPC TPC *control* registers at
`0x504xxx` require PRI ring stations that are created by the GPU's internal
ring infrastructure during GPCCS boot, which nouveau on Volta cannot perform
(no signed PMU/GPCCS firmware).

## Strategic Pivot: nvidia-470 via glowplug/agentReagents

All BAR0 paths exhausted. The TPC wall is a firmware-mediated hardware security
boundary. The remaining viable paths are:

### Path A: nvidia-470 Dual-Load Injection (Primary — Exp 218)

**Infrastructure status: 95% complete.** The diesel engine already has:
- `HandoffConfig::nvidia_patched_titanv()` — DKMS 470.256.02 → patch → rename `nvsov` → insmod
- `PatchSet::nvidia_warm_handoff()` — NOP `nv_pci_remove`, `gpuStateUnload_IMPL`, etc.
- `module_patch::rename_module_identity()` — `nvidia` → `nvsov` for dual-load
- `kmod::find_dkms_module("nvidia", "470.256.02")` — discovers DKMS-built module
- `sovereign.warm_handoff` RPC with strategy `nvidia_patched_titanv`
- `build_nvidia470_kernel617.sh` — isolated `/tmp` DKMS build recipe

**Blocker:** `nvsov` co-load conflicts with host nvidia-580:
- Symbol collisions (`nvidia_*` symbols exported by both)
- `/proc/driver/nvidia/` procfs entries conflict
- Character device `/dev/nvidia*` registration collision

**Resolution path (via glowplug):**
1. Expand `PatchSet::nvidia_warm_handoff()` to NOP procfs, chardev, and UVM registration
2. Add `nvsov` to `SysfsSwapExecutor::is_warm_preserving_swap()` warm-preserving list
3. Build `nvsov` with additional symbol renames for dual-load isolation
4. Or: use agentReagents to build nvidia-470 with `MODULE_BASE_NAME=nvsov` at compile time
   (like `build_nvidia_oracle.sh` did for 580-open), avoiding runtime rename entirely

### Path B: agentReagents VM Compute (Parallel — Available Now)

nvidia-470 inside VM computes directly — no TPC wall. But SBR on VM exit
destroys state (no warm handoff back to host). Used when Titan V needs full
compute and host sovereignty isn't required.

### ~~Path C: K80 Cross-Gen~~ (RETIRED)

K80 hardware destroyed (Exp 199), retired from fleet. Path superseded by
nvidia-470 nvsov dual-load (Path A, Exp 218).

## Related Experiments

| Exp | Relevance |
|-----|-----------|
| 210 | GPC boundary + tier model codified |
| 211 | PMU software path CLOSED (HS-locked) |
| 215 | TPC wall identified, experiment stages 1-5 created |
| 216 | Kernel health clean — autoconf.h restored |
| 206 | ACR DMA proven — falcon firmware upload works |
| 204 | VBIOS interpreter — 422 ops, silicon-deistic groundwork |
| 218 | (next) nvidia-470 nvsov dual-load co-loading resolution |

## Files Changed

**toadStool:**
- `crates/core/cylinder/src/vfio/sovereign_stages.rs` — Stage 4 NvGspBridge, Stage 6 full ungating
- `crates/core/cylinder/src/vfio/sovereign_tiers.rs` — TPC health check, module docs updated
- `crates/server/src/pure_jsonrpc/handler/sovereign.rs` — Stage 6 routing, TPC fields in tier response
- `crates/core/cylinder/src/vfio/sovereign_handoff.rs` — TierEvidence field additions

**hotSpring:**
- `experiments/217_TPC_PRI_STATION_CREATION.md` — This document
- `specs/SOVEREIGN_VALIDATION_MATRIX.md` — Updated with Exp 210-216 findings
- `docs/PRIMAL_GAPS.md` — GAP-HS-107 rewritten for TPC wall
- `EXPERIMENT_INDEX.md` — Exp 217 entry
