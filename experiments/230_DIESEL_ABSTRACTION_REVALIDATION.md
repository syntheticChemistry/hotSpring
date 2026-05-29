# Experiment 230: Diesel Engine Abstraction Revalidation

**Date:** 2026-05-28
**Status:** COMPLETE — Revalidation passed (2026-05-28 12:25 EDT)
**Hardware:** Dual Titan V (GV100, SM70) — 0000:02:00.0 + 0000:49:00.0
**Prerequisite:** Exp 229 (Catalyst Channel + Lockup Forensics — COMPLETE)

## Objective

Revalidate the Exp 229 lockup catalog through the newly abstracted diesel engine
infrastructure. All hardcoded Volta-specific register offsets, GPC counts, and
interrupt semantics have been replaced with generation-aware profiles. This
experiment confirms no regressions were introduced by the abstraction.

## What Changed (Silicon Deistic Abstraction)

### Phase 1: Generation-Aware Interrupt Defense
- `InterruptProfile` struct in `pmc.rs` — encodes per-generation interrupt
  register semantics (direct-write vs SET/CLEAR pair)
- All quench functions (`post_exit_quench`, `emergency_quench`,
  `quench_gpu_interrupts`) now dispatch via `InterruptProfile`
- `rm_trigger` binary accepts `--bdf` CLI arg (no more hardcoded `0000:49:00.0`)

### Phase 2: Catalyst Handoff Generation Context
- `HandoffConfig.sm_version` field drives profile selection
- `HandoffCapabilityProfile` captures GPC count, register topology, BAR0 domains
- Pipeline uses profile for TPC probes, PCCSR scan, FECS capture, firmware naming

### Phase 3: Patch Set Abstraction
- `PatchSet::from_recipe_toml()` loads patches from TOML recipe files
- `PatchSet::by_profile()` dispatches from `(ChipFamily, driver_version, strategy)`

### Phase 4: Watchdog Parameterization
- `activate()` accepts `InterruptProfile` + configurable timeout
- Heartbeat callbacks wired at 13 pipeline step boundaries

## Lockup Vectors to Revalidate

| # | Vector | Abstracted How | Expected Outcome |
|---|--------|----------------|------------------|
| 1 | PCIe bridge keepalive → `pci_lock` deadlock | Already generic (Phase 0) | PASS — `HandoffExclusionGuard` unchanged |
| 2 | INTR_EN quench after `free_irq`/`pci_disable_msi` | Via `InterruptProfile.disable_offset()` (was hardcoded 0x180) | PASS — Volta dispatches to CLEAR@0x180 |
| 3 | nvidia_close re-enables INTR_EN → post-exit quench | Via `pmc::quench_interrupts()` with profile | PASS — same register, cleaner dispatch |
| 4 | `nv_close_device` use-after-free | Patch set driven (RetAtEntry) | PASS — patch unchanged |
| 5 | `nv_pci_remove` hang | Patch set driven (RetAtEntry) | PASS — patch unchanged |

## Run Plan

1. Build: `cargo build --release --bin rm_trigger && cp target/release/rm_trigger /usr/local/bin/`
2. Restart ember: `systemctl restart toadstool-ember`
3. Execute catalyst handoff via RPC:
   ```bash
   socat - UNIX-CONNECT:/run/toadstool-ember.sock <<'EOF'
   {"jsonrpc":"2.0","method":"sovereign.warm_handoff","params":{"bdf":"0000:49:00.0","strategy":"nvidia_catalyst_titanv"},"id":1}
   EOF
   ```
4. Verify: Check journal for `InterruptProfile`-driven quench logs:
   - `interrupt quench complete` (not old `INTR_EN_CLEAR@0x180 written`)
   - `PCI INTx disabled` (unchanged PCI-level defense)
   - Heartbeat logs at step boundaries
   - GPC probe using profile topology (6 GPCs for GV100)
5. Confirm: System survives, no lockup, tier classification succeeds

## Success Criteria

- [x] Full catalyst handoff completes without lockup
- [x] Journal shows generation-aware quench dispatch (profile-driven logs)
- [x] Watchdog receives heartbeats (no false-positive trigger)
- [x] Tier classification matches Exp 229 Run #9 result
- [x] `rm_trigger` runs with `--bdf 0000:49:00.0` (not hardcoded)

## Run Results

### Run 1 — Preflight Halt (stuck module)

- **Outcome**: Preflight halt — `nvsov` stuck in `Unloading` (refcount -1) from prior crash
- **Not a regression**: Pipeline detected zombie module and halted gracefully
- **Positive signals**: Watchdog activated (450s), RAII deactivation on halt, keepalive exclude/restore
- **Resolution**: Full power cycle to clear zombie module

### Run 2 — FULL SUCCESS

- **Outcome**: `success: true`, Tier: `WarmCompute`, Total: `79,090ms`
- **PMC DEVINIT**: `0x40000121` (cold, popcount 4) → `0x5fecdff1` (warm, popcount 23)
- **Interrupt quench**: Volta path via `CLEAR@0x180` — `0x7fffffff → 0x00000000`
- **rm_trigger**: `--bdf 0000:49:00.0` passed correctly, exit 0
- **Module patching**: 20/20 targets applied, identity rename `nvidia→nvsov`
- **BAR0 capture**: 62,569 alive registers across 22 domains
- **FECS INIT_CTXSW**: Succeeded (`mb0=0x1`, `tpc0=0xcf`)
- **PRI recovery**: Complete — IMEM captured (24,314 nonzero bytes), 8/8 words alive
- **Firmware**: `fecs_imem_gv100.bin`, `gpccs_imem_gv100.bin` (chip_name-driven)
- **Watchdog**: Armed at 450s, deactivated cleanly on completion
- **Keepalive**: BDFs excluded during handoff, restored after

### Lockup Vector Verification

| # | Vector | Result | Evidence |
|---|--------|--------|----------|
| 1 | PCIe bridge keepalive deadlock | PASS | Exclusion guard active, no pci_lock contention |
| 2 | INTR_EN quench via InterruptProfile | PASS | `disable_offset: 0x180` (Volta CLEAR path) |
| 3 | nvidia_close re-enable → post-exit quench | PASS | `pmc::quench_interrupts()` with profile |
| 4 | nv_close_device use-after-free | PASS | RetAtEntry patch applied (offset 11189) |
| 5 | nv_pci_remove hang | PASS | RetAtEntry patch applied (offset 30741) |

### Known Non-Regression Issue

- `module_cleanup` step failed: `nvsov` rmmod returns -1 (refcount goes to -1)
- Pre-existing before abstraction — the patched module's cleanup_module path is neutered
- Module becomes zombie until reboot — tracked since Exp 225
