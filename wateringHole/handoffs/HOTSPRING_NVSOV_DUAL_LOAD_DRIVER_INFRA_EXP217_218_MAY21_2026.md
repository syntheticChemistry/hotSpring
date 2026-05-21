# nvidia-470 nvsov Dual-Load + Driver Infra Evolution — hotSpring Handoff

**Date:** May 21, 2026
**From:** hotSpring
**To:** primalSpring (audit), toadStool (upstream code landed)
**Status:** Exp 217 complete (BAR0 path closed), Exp 218 in progress (nvsov loads, reboot needed)
**Experiments:** 217, 218

## Summary

Two experiments advanced sovereignty toward Tier 2 (warm compute):

1. **Exp 217 — TPC PRI Station Creation**: Definitively closed the BAR0-only path.
   Full ungating sequences, `sw_nonctx.bin` register replay, and PGRAPH reset all
   fail to create TPC PRI ring stations. Conclusion: TPC stations are firmware-mediated
   (GPCCS required), not programmable via BAR0 writes alone.

2. **Exp 218 — nvidia-470 nvsov Dual-Load Injection**: Primary attack path. Load
   nvidia-470 as a renamed module (`nvsov`) alongside the host nvidia-580 display
   driver, use it as the warm handoff seeder for Titan V. Four distinct blockers
   diagnosed and solved in a single session.

## Driver Infrastructure Evolution (landed in toadStool)

### New Abstractions
- **`SymbolResolver` trait + `NmResolver`**: Symbol resolution decoupled from
  patch application — enables future pure-Rust ELF parsing.
- **`PatchSet::from_json()`**: Runtime patch set definition for experiment iteration.
- **`HandoffConfig` extensions**: `patch_set_override`, `skip_preflight` for
  flexible RPC control.
- **`WarmInitPlan::from_handoff_config()`**: Unified config derivation prevents
  cylinder/glowplug config divergence.

### New RPCs
- **`sovereign.snapshot`**: Read-only GPU state + tier evidence capture.
- **`sovereign.compare`**: Twin-card structured diff via `SnapshotDelta`.
- **`sovereign.warm_handoff` extensions**: `settle_secs`, `patch_set_json`,
  `skip_preflight` parameters.

### Generation Dispatch
- Experiment stages 4-6 accept optional `chip` parameter (e.g., "gv100", "gk210")
  with auto-detection from BOOT0 register.

## Exp 218 Blockers Solved

### 1. "exports duplicate symbol" (ENOEXEC)
nvidia-470 exports 111 symbols via `__ksymtab` that collide with nvidia-580.
**Fix**: `objcopy --remove-section` strips `__ksymtab`, `__kcrctab`,
`__ksymtab_strings`, `.rela__ksymtab` from the patched module. Safe because
`nvsov` is a leaf module (nothing depends on its exports).

### 2. "NVRM: failed to initialize capabilities"
`nv_cap_init` creates `/proc/driver/nvidia/` entries already owned by nvidia-580.
**Fix**: 5 co-load isolation NOPs added to `PatchSet::nvidia_warm_handoff()`:
`nv_cap_init`, `nv_cap_drv_init`, `nv_procfs_init`, `nvidia_register_module`,
`nv_cap_procfs_init`.

### 3. Relocation conflict at patch sites
`RetAtEntry` wrote `xor eax,eax; ret` at byte 0, but bytes 1-4 have a PLT32
relocation displacement. Kernel/objcopy overwrites NOP bytes.
**Fix**: Patch at **offset+5** (after 5-byte ftrace `call __fentry__` preamble).

### 4. "Invalid relocation target" (kernel 6.17)
`normalize_relocations` only handled `R_X86_64_64` (type 1). Kernel 6.17 also
checks `R_X86_64_PC32` (type 2) and `R_X86_64_PLT32` (type 4) with 32-bit targets.
**Fix**: Expanded normalizer to handle all three relocation types.

## Files Changed (toadStool)

| File | Changes |
|------|---------|
| `cylinder/src/vfio/module_patch.rs` | `SymbolResolver` trait, `PatchSet::from_json()`, `strip_ksymtab()`, expanded NOP set (11 targets), `RetAtEntry` patches at +5 with ret0, PC32/PLT32 normalization |
| `cylinder/src/vfio/sovereign_handoff.rs` | `HandoffConfig` extensions, `objcopy`-based ksymtab stripping in DKMS path |
| `cylinder/src/vfio/sovereign_stages.rs` | `SnapshotDelta`, `diff_structured()`, `sovereign_snapshot_only()`, `detect_chip()`, generation-dispatched stages |
| `glowplug/src/warm_init.rs` | `WarmInitPlan::from_handoff_config()` |
| `server/handler/sovereign.rs` | `sovereign.snapshot`, `sovereign.compare` RPCs, chip-aware experiment dispatch |
| `server/handler/dispatch/mod.rs` | Extended `sovereign.warm_handoff` params |
| `server/handler/mod.rs` | Routing for new RPCs |

## Validation Results

- `sovereign.snapshot` on both Titan Vs: consistent Tier 1 results
- `sovereign.compare` on twin Titan Vs: 1 register delta (`THERM_GATE`)
- `sovereign.warm_handoff` on both: 6/8 patches applied, Tier 1 confirmed
- nvidia-470 DKMS `nv_pci_remove` patchable at offset 0x7770
- `nvsov` module loads alongside nvidia-580 with all co-load fixes applied
- 700 cylinder + 109 glowplug unit tests pass

## Next Steps (post-reboot)

1. Reboot to clear zombie `nvsov` module from kernel oops during testing
2. Run full RPC pipeline: `sovereign.warm_handoff` with `nvidia_patched_titanv`
3. Verify nvidia-470 completes GR init (dmesg: GPCCS/FECS boot)
4. Warm swap to vfio-pci, classify tier — expect `WarmCompute` with `tpc_alive = true`
5. If TPC wall broken: sovereign compute Tier 2 achieved

## Upstream Gaps for primalSpring

- `nvsov` module naming convention needs ecosystem-level policy
- Driver dual-load safety model needs review (what if nvidia-580 crashes?)
- `objcopy` dependency in handoff pipeline — consider pure-Rust ELF section removal
