<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Experiment 208 — Reboot-Efficient Sovereign Evolution

**Date**: 2026-05-18
**Hardware**: Dual Titan V (GV100, 0000:02:00.0 + 0000:49:00.0), RTX 5060 (0000:21:00.0)
**Status**: ✅ Complete — 183ms warm pipeline, falcon warm preservation proven, fd store validated

## Objective

Maximize software evolution between costly reboots. Wire the systemd fd store
end-to-end so daemon restarts preserve warm GPUs, add a warm-state verification
RPC, and optimize the cold pipeline — then validate the full keepalive lifecycle
with a single meaningful power cycle.

## Context

After Exp 207 (sovereign boot abstraction + profiling), the fd store plumbing
existed (`store_anchors()`, `retrieve_anchors()`, systemd `FileDescriptorStoreMax=16`)
but had never been validated end-to-end with real hardware. Cold pipelines wasted
~10–14s on doomed `memory_training` calls. No RPC existed to inspect anchor/keepalive
state without running a full pipeline.

## Changes (toadStool)

### 1. `sovereign.warm_status` RPC

New lightweight JSON-RPC method that reports warm keepalive state for all known
GPUs without running any pipeline:

```json
{
  "anchor_count": 2,
  "fd_store_capable": true,
  "devices": {
    "0000:02:00.0": {
      "anchor_held": true,
      "boot_state": "warm",
      "pmc_enable": "0x5fecdff1",
      "pramin_ok": true,
      "fd_store_capable": true
    },
    "0000:49:00.0": { ... }
  }
}
```

Probes boot state via sysfs BAR0 mmap (no VFIO needed). Reports anchored and
cached devices. `fd_store_capable` reflects `NOTIFY_SOCKET` presence.

**Files**: `dispatch/mod.rs` (method + `probe_boot_state_sysfs` helper),
`handler/mod.rs` (router), `core/mod.rs` (method registry).

### 2. Cold Pipeline Early-Exit

New `skip_cold_memory_training` flag on `SovereignInitOptions`. When boot state
probe returns Cold, the pipeline skips `memory_training` and all subsequent
stages, returning immediately with `compute_ready: false` and
`halted_at: "memory_training (cold_early_exit)"`.

Both ember handlers (`sovereign_init_ember`, `sovereign_profile_ember`) set this
flag by default.

**Impact**: Cold pipeline reduced from ~14s to ~200ms (70× faster).

**Files**: `sovereign_types.rs` (new field), `sovereign_init.rs` (early-exit
logic after `boot_state_probe` stage), `dispatch/mod.rs` (flag set in both
handlers).

### 3. Anchor Population Confirmation

Added `anchor_held` to pipeline completion logs in both `sovereign_init_ember`
and `sovereign_profile_ember`. Confirmed that `get_or_create_device()` →
`dup_anchor_fds()` → anchor store population already works correctly for
both handlers.

## Validation Results

### Phase 1: Software (no reboot)

All three changes built and deployed without rebooting. `cargo check` clean,
release binary built in 1m46s.

### Phase 2: fd Store Chain (cold GPUs, no reboot)

| Step | Result |
|------|--------|
| `sovereign.init` card #1 | 204ms, cold early-exit, `anchor_held: true` |
| `sovereign.init` card #2 | 206ms, cold early-exit, `anchor_held: true` |
| `sovereign.warm_status` | 2 anchors, both cold, `fd_store_capable: true` |
| `systemctl restart` | SIGTERM → 4 fds stored → new process → 4 fds retrieved |
| `sovereign.warm_status` (after restart) | **2 anchors still held** — fds survived |

Log trail: `store_anchors(2 anchors, 4 fds)` → systemd holds →
`retrieve_anchors()` → `reconstructed VfioAnchor from systemd fd store` for
both BDFs.

### Phase 3: Power Cycle (warm GPUs)

Full power cycle performed. Boot ROM trained HBM2 on both Titan Vs.

| Step | Card #1 (0000:02:00.0) | Card #2 (0000:49:00.0) |
|------|------------------------|------------------------|
| `sovereign.init` | **warm**, 3,913ms, `compute_ready: true` | **warm**, 3,915ms, `compute_ready: true` |
| Boot state | `pmc=23 engines, pramin=true, falcon=Cold` | `pmc=23 engines, pramin=true, falcon=Cold` |
| memory_training | skipped (warm) | skipped (warm) |
| falcon_boot (ACR) | 3,698ms (94.5%) | 3,696ms (94.5%) |
| `sovereign.warm_status` | `anchor_held: true, boot_state: warm` | `anchor_held: true, boot_state: warm` |
| **After daemon restart** | **still warm, anchor held** | **still warm, anchor held** |

### First-Ever Warm Twin-Card Profiles

Post-restart warm profiles (both via `sovereign.profile`):

| Stage | Card #1 (µs) | Card #2 (µs) | Fraction |
|-------|-------------|-------------|----------|
| identity_probe | 12,000 | 12,000 | 0.3% |
| pmc_enable | 74,000 | 74,000 | 1.9% |
| pgraph_reset | 30,000 | 30,000 | 0.8% |
| cg_sweep | 0 | 0 | 0% |
| pri_recovery | 85,000 | 85,000 | 2.2% |
| pgob_ungating | 0 | 0 | 0% |
| boot_state_probe | 0 | 0 | 0% |
| memory_training | 0 (skipped) | 0 (skipped) | 0% |
| **falcon_boot** | **3,695,000** | **3,694,000** | **94.5%** |
| gr_init | 0 (skipped) | 0 (skipped) | 0% |
| verify | 12,000 | 12,000 | 0.3% |
| **TOTAL** | **3,910,000** | **3,908,000** | |

Profiling overhead: 73µs / 50µs respectively.

### Warm vs Cold Comparison

| Metric | Cold (Exp 207) | Cold (Early-Exit) | Warm pre-falcon | **Warm + falcon** |
|--------|---------------|-------------------|-----------------|-------------------|
| Total pipeline | 11.3–13.0s | **204ms** | **3.9s** | **183ms** |
| memory_training | 5.4–10.5s (fail) | 0ms (skipped) | 0ms (skipped) | 0ms (skipped) |
| falcon_boot | 3.7s | n/a | 3.7s (94.5%) | **0ms (skipped)** |
| pgraph_reset | 30ms | 30ms | 30ms | **0ms (skipped)** |
| compute_ready | false | false | **true** | **true** |
| Dominant stage | memory_training | identity_probe | falcon_boot | **pri_recovery** |
| Speedup vs cold | — | 70× | 3.6× | **76×** |

### Phase 4: Falcon Warm Preservation (validated)

After deploying the falcon warm detection fix and power cycling:

| Step | Card #1 (0000:02:00.0) | Card #2 (0000:49:00.0) |
|------|------------------------|------------------------|
| 1st `sovereign.init` | **185ms**, falcon=WarmRunning (PC=0xB0) | **188ms**, falcon=WarmRunning (PC=0xB9) |
| pgraph_reset | **skipped** (falcon warm) | **skipped** (falcon warm) |
| falcon_boot | **skipped** (warm-running) | **skipped** (warm-running) |
| 2nd `sovereign.init` | **184ms** | **183ms** |
| After `systemctl restart` | **183ms**, anchors survived, falcon still warm | **183ms** |

Final warm profiles (post-restart, via `sovereign.profile`):

| Stage | Card #1 (µs) | Card #2 (µs) | Fraction |
|-------|-------------|-------------|----------|
| identity_probe | 11,000 | 11,000 | 6.0% |
| pmc_enable | 73,000 | 74,000 | 40.3% |
| pgraph_reset | 0 (skipped) | 0 (skipped) | 0% |
| cg_sweep | 0 | 0 | 0% |
| pri_recovery | 85,000 | 85,000 | 46.3% |
| falcon_boot | 0 (skipped) | 0 (skipped) | 0% |
| gr_init | 0 (skipped) | 0 (skipped) | 0% |
| verify | 12,000 | 12,000 | 6.5% |
| **TOTAL** | **183,000** | **183,000** | |

The boot ROM leaves FECS running at PC=0xB0+ after POST. The early falcon
probe detects this and skips both pgraph_reset and falcon_boot on the **first**
sovereign.init call — no ACR boot needed at all when the boot ROM firmware
is still live.

## Key Findings

1. **183ms warm pipeline** — 76× faster than cold full (14s), sovereign compute
   initialization in under 200ms with `compute_ready: true`.

2. **fd store chain proven end-to-end**: VFIO file descriptors survive daemon
   restarts via systemd FileDescriptorStore. Anchors are stored on SIGTERM,
   retrieved on startup, and GPUs remain warm.

3. **Cold early-exit is 70× faster**: Skipping doomed memory_training reduces
   cold pipeline from ~14s to ~200ms with no loss of information.

4. **Falcon warm preservation eliminates ACR boot entirely**: Boot ROM FECS
   firmware survives VFIO device open (PC=0xB0+). Early probe before
   pgraph_reset detects this and skips the 3.7s ACR re-boot.

5. **Twin-card timing within 5ms**: Identical hardware produces near-identical
   profiles (183ms vs 183ms), confirming measurement reliability.

6. **Warm state survives daemon restarts**: `systemctl restart` stores 4 VFIO fds,
   new process retrieves them, anchors reconstructed, subsequent inits stay at 183ms.

## Falcon Warm Preservation (Experiment 208b)

### Root Cause

The 3.7s falcon_boot ACR dominance was caused by the pipeline's own
`pgraph_reset` (stage 2a) destroying FECS/GPCCS firmware state before the
boot state probe. The sequence:

1. `pgraph_reset` toggles PMC_ENABLE bit 12, resetting PGRAPH engine
2. FECS/GPCCS firmware killed (CPUCTL → 0, MAILBOX0 → 0)
3. `boot_state_probe` finds `falcon: Cold` (because we just killed it)
4. `falcon_boot` runs full ACR DMA boot (3.7s) to reload firmware

After a previous `sovereign.init` successfully boots FECS (CPUCTL=0x10),
subsequent calls would kill it in step 1 and re-boot it in step 4.

### Fix: Early Falcon Probe

Added an early falcon state probe **before** `pgraph_reset`. If the falcon
is `WarmPreserved` or `WarmRunning` from a previous init:

- `pgraph_reset` is **skipped** (preserving FECS/GPCCS firmware)
- `boot_state_probe` finds falcon warm (since PGRAPH wasn't reset)
- `falcon_boot` returns immediately ("warm-preserved" / "warm-running")
- CG sweep and PRI recovery still run (they don't affect falcons)

**Expected warm pipeline with this fix:** ~200ms (identity + pmc_enable +
cg_sweep + pri_recovery + verify) instead of ~3.9s.

The optimization activates on the **second** `sovereign.init` call after a
boot (first call boots FECS via ACR, second call finds it warm and skips).
After a power cycle + first init, all subsequent inits and daemon restarts
should benefit.

### Falcon Detection Details

After ACR HS boot, FECS state is:
- `CPUCTL` = 0x10 (HRESET bit — HS-secure falcon shows HRESET, not HALTED)
- `CPUCTL_ALIAS` = 0x00
- `PC` = 0xAE (firmware executing in command-wait loop)

The detection was initially fooled by:
1. `CPUCTL=0x10` being HRESET (bit 4), not HALTED (bit 5 = 0x20)
2. Post-FLR residual PCs (0x03) causing false positives

Fix: check PC register with threshold >= 0x40 when HRESET is set. ACR firmware
entry points are 0x80+ (code section). FLR residuals are 0x00-0x10 (boot ROM
artifacts). This correctly classifies:
- Post-ACR-boot: cpuctl=0x10, pc=0xAE → `WarmRunning` (skip pgraph_reset)
- Post-FLR: cpuctl=0x10, pc=0x03 → `Cold` (run pgraph_reset normally)

**Files**: `sovereign_init.rs` (early falcon probe + conditional pgraph_reset
skip), `sovereign_strategy.rs` (detection logic: HRESET+PC threshold).
