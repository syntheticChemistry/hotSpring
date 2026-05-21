# Experiment 213 — Live Hardware Warm Handoff & IOMMU Group Fix

**Date**: 2026-05-20
**Status**: COMPLETE (post-reboot validation passed — Exp 214/215 continue)
**Hardware**: 2x NVIDIA Titan V (GV100), BDFs `0000:02:00.0`, `0000:49:00.0`
**Dependency**: Exp 212 (Sovereignty Consolidation Sprint)

## Objective

Execute the first live hardware warm handoff pipeline (`sovereign.warm_handoff`)
on the Titan V GPUs using the consolidated Exp 212 infrastructure. Validate
generation-aware tier classification (`sovereign.classify_tier`) on live silicon.

## Phase 1: Generation-Aware Tier Classification — VALIDATED

### New RPC: `sovereign.classify_tier`

Added `sovereign.classify_tier` RPC endpoint that:
- Auto-detects SM version from BOOT0 register
- Looks up `GenerationProfile` (name, CE class, register offsets)
- Runs `classify_tier_for_profile()` with generation-specific offsets
- Returns full evidence including profile metadata

### Live Results — Both Titan Vs

```json
{
  "generation": "Volta",
  "sm_version": 70,
  "ce_class": "0xC3B5",
  "tier": "cold",
  "tier_level": 0,
  "evidence": {
    "pmc_enable": "0x5fecdff1",
    "pmc_popcount": 23,
    "pramin_accessible": false,
    "fecs_pc": "0xbadf5040",
    "gpc_enables": "0xbadf3000",
    "ce_status": "0xbadf3000"
  },
  "profile_offsets": {
    "fecs_pc": "0x00409624",
    "gpc_broadcast": "0x0041a004",
    "ce0_base": "0x00104000",
    "pgraph_status": "0x00400700"
  }
}
```

**Findings**:
- Generation auto-detection works: BOOT0 → chip_id 0x140 → SM 70 → "Volta"
- CE class correctly resolved: 0xC3B5 (VOLTA_DMA_COPY_A)
- Profile offsets match expected Volta register map
- Tier correctly classified as Cold (PMC alive but PRAMIN/CE/GPC all 0xBADF)
- Both cards show identical register fingerprints (same silicon revision)

### `sovereign.init` — Cold Start Performance

```
Card #1 (02:00.0): 203ms, cold start, compute_ready=false
Card #2 (49:00.0): 205ms, cold start, compute_ready=false
```

The `compute_ready=false` is expected — FECS isn't running after cold init
without a prior vendor driver bind. This confirms the need for the warm
handoff pipeline.

## Phase 2: Warm Handoff Pipeline Execution — BLOCKED

### Infrastructure Gaps Discovered

**Gap 1: IOMMU Group Siblings**

The `sovereign.warm_handoff` pipeline unbinds the target GPU from `vfio-pci`
and attempts to bind `nouveau`. On NVIDIA GPUs, function 1 (HD Audio) shares
the same IOMMU group. The kernel requires ALL devices in the group to be
released before any device can bind to a different driver.

The handoff was binding the GPU device (fn 0) but leaving the audio device
(fn 1) bound to `vfio-pci`, which blocked `nouveau` from probing.

**Fix**: Added `iommu_group_siblings()` discovery and sibling unbind to the
handoff pipeline (Step 2). After warm swap completes, siblings are rebound
to `vfio-pci` via `rebind_siblings_to_vfio()`.

**Gap 2: VFIO Anchor Release**

The toadstool daemon holds VFIO container/group file descriptors (anchors)
for warm keepalive. These FDs lock the IOMMU group at the kernel level.
Even after unbinding the device from `vfio-pci`, the IOMMU group remains
locked until the daemon closes its FDs.

**Fix**: The `sovereign_warm_handoff` RPC handler now releases the VFIO
anchor and cached device handle BEFORE spawning the handoff pipeline.
After the handoff completes and `vfio-pci` rebinds, `sovereign.init`
can be called to re-anchor.

**Gap 3: systemd ProtectSystem=strict**

The `toadstool-ember.service` unit uses `ProtectSystem=strict` which
makes all filesystems read-only except `ReadWritePaths`. The patched
`nouveau.ko` module was written to `/tmp`, which was not in the
allowed paths.

**Fix**: Added `/tmp` to `ReadWritePaths` in the service unit.

### Cascading Kernel Failure

During the first warm handoff attempt (before Gap 2 fix), the pipeline
successfully loaded the patched nouveau module but got stuck at the
`seeder_bind` step. A manual bind attempt to `nouveau` on the same device
also hung. This left `nouveau` in a stuck probe state.

When the daemon was restarted, the `rmmod nouveau` during cleanup entered
uninterruptible kernel sleep (refcount=-1, Unloading state). This left a
zombie process with a live tokio-rt-worker thread holding a VFIO device
reference, which blocked ALL subsequent VFIO operations on that IOMMU group.

**Root cause chain**:
1. `nouveau bind` hung waiting for exclusive IOMMU group access (daemon held FDs)
2. `rmmod nouveau` hung because nouveau's probe thread was stuck
3. Daemon exit hung on `vfio_unregister_group_dev` (blocked by step 2)
4. Zombie daemon's live worker thread holds VFIO reference indefinitely
5. New daemon can't acquire VFIO for that group → deadlock cascade

**Resolution**: Requires reboot to clear kernel state. Fixes in place
(anchor release + sibling unbind) should prevent recurrence.

## Code Changes

### `crates/server/src/pure_jsonrpc/handler/sovereign.rs`
- Added `sovereign_classify_tier()` RPC handler
- Uses `GenerationProfile` for generation-aware tier classification
- Auto-detects SM version from BOOT0, optional override

### `crates/server/src/pure_jsonrpc/handler/mod.rs`
- Routed `sovereign.classify_tier` to the new handler

### `crates/core/cylinder/src/vfio/sovereign_handoff.rs`
- Added `iommu_group_siblings()` — discovers sibling BDFs in IOMMU group
- Added `rebind_siblings_to_vfio()` — rebinds siblings after warm swap
- Updated Step 2 to unbind all IOMMU group siblings before seeder bind
- Added Step 6b to rebind siblings after successful warm swap

### `crates/server/src/pure_jsonrpc/handler/dispatch/mod.rs`
- `sovereign_warm_handoff()` now releases VFIO anchor + cached device
  before running the handoff pipeline

### `/etc/systemd/system/toadstool-ember.service`
- Added `/tmp` to `ReadWritePaths`

## Validation Checklist (Post-Reboot)

- [ ] `sovereign.init` on both Titan Vs — confirm compute_ready
- [ ] `sovereign.classify_tier` on both — confirm Tier 0 (cold) detection
- [ ] `sovereign.warm_handoff` on card #2 (nouveau_titanv) — full cycle
- [ ] Post-handoff `sovereign.classify_tier` — confirm Tier 1 (WarmInfra)
- [ ] Post-handoff `sovereign.init` — confirm warm detection + FECS alive
- [ ] Verify IOMMU sibling (audio device) properly rebound to vfio-pci
- [ ] Module cleanup — confirm patched nouveau.ko removed

## Next Steps

1. **Reboot** to clear stuck kernel state
2. Run full validation checklist
3. Investigate `nvidia-470` warm handoff path (GPCs may stay ungated → Tier 2)
4. Test silicon-deistic replay with `engine_init_path`
5. Prepare for K80 arrival (infrastructure confirmed ready in Exp 212)
