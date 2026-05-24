# Catalyst Driver Pattern — HW Validated — hotSpring Handoff

**Date:** May 24, 2026
**From:** hotSpring
**To:** primalSpring (audit), toadStool (upstream code landed)
**Status:** ✅ Hardware validated — catalyst pipeline completes in 26s, golden state captured
**Experiments:** 219 (HW execution phase — builds on May 22 infra handoff)
**Previous:** `HOTSPRING_CATALYST_DRIVER_PATTERN_EXP219_MAY22_2026.md` (infra)

## Summary

Exp 219 catalyst pipeline executed on live Titan V hardware. Three critical
optimizations resolved the 7-minute timeout that blocked the initial attempt:

1. **Domain-scoped BAR0 capture** — 515× faster (897ms vs 462s)
2. **Surgical `nv_pci_remove` NOP patches** — PCI resource cleanup preserved
3. **Fire-and-poll unbind** — toadstool-ember stays responsive during 7s RM teardown

## Key Results

| Metric | Value |
|--------|-------|
| Total pipeline time | **26,317ms** |
| BAR0 capture time | **897ms** (was 462,094ms) |
| Alive registers captured | 83,623 (across 22 Volta domains) |
| Tier achieved | Tier 1 — WarmInfrastructure |
| Frozen .ko archived | ✅ 41 MB |
| Replay sequence | ✅ 83,623 writes |
| Recipe JSON | ✅ persisted |
| Module cleanup | ✅ clean rmmod, 100ms |

## What Changed (toadStool)

### `warm_capture.rs`
- **`Bar0Snapshot::capture_domains()`**: Reads only specified domain ranges,
  skipping dead inter-domain MMIO gaps that cause PCIe completion timeouts
  (~110μs each). For Volta: 641K register reads vs 4.2M full scan.

### `module_patch.rs`
- **`nv_pci_remove` surgical NOPs**: Replaced blanket `RetAtEntry` with four
  targeted `NopCallAt` patches:
  - `0x374` → NOP `nv_shutdown_adapter`
  - `0x3a0` → NOP `rm_disable_gpu_state`
  - `0x1fe` → NOP `rm_cleanup_dynamic_power_mgmt`
  - `0x2a0` → NOP `rm_free_private_state`
  This allows `__release_region` and `pci_disable_device` to execute normally,
  preventing stale BAR0 claims in the kernel `iomem_resource` tree.

### `sovereign_handoff.rs`
- **Pipeline reordering**: BAR0 capture moved before sibling rebind (step 6b)
  to avoid PCI device lock contention with ongoing RM teardown.
- **Domain-scoped capture**: `capture_full` → `capture_domains` with
  `VOLTA_BAR0_DOMAINS`.
- **Per-step profiling**: `tracing::info!` with `pipeline_elapsed_s`, `open_ms`,
  `capture_ms` fields for every pipeline step.

### `guarded_sysfs.rs`
- **`sysfs_unbind_fire_and_poll`**: Non-blocking unbind + poll loop monitoring
  the driver symlink. Prevents D-state in toadstool-ember during 7s RM teardown.
- **`HANDOFF_DEADLINE`**: 400s (from 180s).

### `dispatch/mod.rs`
- **RPC timeout**: `sovereign.warm_handoff` timeout increased to 420s.

## Profiling Methodology

Full pipeline instrumented with `tracing::info!` at each step boundary.
Profiling revealed `Bar0Snapshot::capture_full()` as the sole bottleneck:

```
09:00:08.689032  warm_swap complete (vfio-pci bound)
09:00:08.689062  BAR0 open start (+30μs)
09:00:08.689248  BAR0 mmap success (+186μs)
  ... 462 seconds of MMIO reads ...
09:07:50.783588  capture_full complete (4,194,304 regs, 174,098 alive)
```

4.2M register reads × ~110μs/read (PCIe completion timeout on dead registers)
= 462 seconds. Domain-scoped capture eliminates 3.5M dead reads.

## Discovery: SBR Bridge Reset

PCI remove/rescan is insufficient to recover a GPU from a dirty catalyst state
(nouveau `preinit` fails with ETIMEDOUT). Bridge-level Secondary Bus Reset
via `setpci` recovers without a full power cycle:

```bash
# Assert SBR on upstream bridge
current=$(setpci -s 0000:00:01.3 BRIDGE_CONTROL.W)
setpci -s 0000:00:01.3 BRIDGE_CONTROL.W=$(printf "%04x" $((0x$current | 0x40)))
sleep 1
# Deassert
setpci -s 0000:00:01.3 BRIDGE_CONTROL.W=$current
sleep 3
# Remove and rescan
echo 1 > /sys/bus/pci/devices/0000:02:00.0/remove
echo 1 > /sys/bus/pci/rescan
```

This should be codified as a recovery path in the handoff rollback pipeline.

## Tier Evidence (pre-swap, while nvsov owns GPU)

```
tier = WarmInfrastructure
pmc_enable = 0x5fecdff1
fecs_cpuctl = 0x00000010  (halted, firmware loaded)
gpccs_cpuctl = 0x00000010 (halted, firmware loaded)
pmu_cpuctl = 0x00000010   (halted, firmware loaded)
pgraph_status = 0x00000000
tpc_alive = false
```

Falcon engines (FECS, GPCCS, PMU) all running. TPC stations not alive —
consistent with Exp 217/218 finding that TPC requires GPCCS firmware execution
which does not survive the warm swap.

## Upstream Gaps for primalSpring

### Resolved (from May 22 handoff)
- **Catalyst safety model**: D-state sacrifice pattern + fire-and-poll unbind
  + `HANDOFF_DEADLINE` keep ember alive through all failure modes.
- **`request_mem_region` stale claims**: Surgical `NopCallAt` patches allow
  PCI cleanup functions to run, preventing stale BAR0 claims.

### Still Open
- **Catalyst-free replay validation**: `sovereign.catalyst_boot` with
  `engine_init_path` pointing to golden state JSON — untested on hardware.
- **Cross-generation portability**: Golden state from GV100 cannot be replayed
  on other generations. Recipe system needs generation guards.
- **Catalyst version matrix**: Which nvidia driver versions work as catalysts
  for which GPU generations? Need systematic testing.
- **`objcopy` dependency**: Still shells out for ksymtab stripping.
- **SBR codification**: Bridge reset recovery should be integrated into
  `handoff_rollback` as an automated recovery path.
- **Domain map completeness**: Current VOLTA_BAR0_DOMAINS covers 22 domains.
  Captures show 83K alive regs from 641K scanned — some domains may benefit
  from narrower sub-ranges for even faster capture.

## Next Steps

1. Archive `/tmp/toadstool-catalyst-*` to `infra/golden_state/`
2. Run `sovereign.catalyst_boot` with captured replay sequence
3. Profile replay: do 83K register writes achieve Tier 2?
4. If not, investigate GPCCS firmware state preservation strategies
5. Twin-card differential (`sovereign.catalyst_diff`) with card B
