# Experiment 229: Catalyst Channel — RM Compute Channel Before Warm Swap

**Date:** 2026-05-27
**Status:** IN PROGRESS
**Hardware:** Dual Titan V (GV100, SM70) — 0000:02:00.0 + 0000:49:00.0
**Prerequisite:** Exp 228 (Sovereign Dispatch Sprint — pipeline proven, FECS ACR blocks)

## Objective

Overcome FECS ACR blocker (Exp 228: `pccsr=0x11000001` PENDING) by establishing a
full RM compute channel **before** warm swap. Let nvidia RM's signed FECS firmware
process at least one ctx-switch lifecycle, then test whether new sovereign channels
work post-swap.

## Strategy

Two-phase approach, tested in order:

- **Phase B** (primary): Create RM channel via `rm_trigger --channel`, warm swap,
  then create a new sovereign `VfioChannel`. Hypothesis: RM-primed FECS ctx-switch
  state machine accepts new channel registrations post-swap.

- **Phase A** (fallback): If Phase B fails, adopt RM's channel hardware layout
  post-swap — submit work through captured RM channel addresses instead of creating
  a new sovereign channel.

## Implementation

### rm_trigger --channel (16-step RM channel recipe)

Extended `rm_trigger` binary to create the complete Volta RM compute channel:

1. Root client (`NV01_ROOT`, 0x0000)
2. Device (`NV01_DEVICE_0`, 0x0080) with `Nv0080AllocParams`
3. Subdevice (`NV20_SUBDEVICE_0`, 0x2080) with `Nv2080AllocParams`
4. GR_GET_INFO control (triggers full GR init)
5. VA space (`FERMI_VASPACE_A`, 0x90F1) with `NvVaspaceAllocParams`
6. USERD memory (`NV01_MEMORY_SYSTEM`, 0x003E) — 4 KiB sysmem
7. GPFIFO ring memory (`NV01_MEMORY_SYSTEM`, 0x003E) — 4 KiB sysmem
8. Error notifier memory (`NV01_MEMORY_SYSTEM`, 0x003E) — type=13
9. TSG (`KEPLER_CHANNEL_GROUP_A`, 0xA06C) — GR0 engine
10. Context share (`FERMI_CONTEXT_SHARE_A`, 0x9067) — under TSG
11. GPFIFO channel (`VOLTA_CHANNEL_GPFIFO_A`, 0xC36F) — 64 entries
12. Compute engine (`VOLTA_COMPUTE_A`, 0xC3C0) — under channel
13. BIND channel to GR (`NV906F_CTRL_CMD_BIND`)
14. SCHEDULE TSG (`NVA06C_CTRL_CMD_GPFIFO_SCHEDULE`) — on TSG handle
15. GET_WORK_SUBMIT_TOKEN (`NVA06F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN`)

### PCCSR Channel Verification

Added PCCSR scan to Step 4b (catalyst_capture) — scans first 64 channel slots
for ACTIVE (status >= 5) vs PENDING (status < 5) while catalyst driver owns GPU.
Stored in `BootServiceEvidence::preserved_state`.

### Pipeline Integration

- `RmChannelEvidence` struct captures channel_id, work_submit_token, steps_completed
- `trigger_rm_init()` now accepts `create_channel: bool` flag
- `HandoffResult` carries `rm_channel_evidence: Option<RmChannelEvidence>`

## Success Criteria

- Phase B: post-swap sovereign channel PCCSR status transitions PENDING → ON_PBDMA (5/6/7)
- Phase A: dispatch through adopted RM channel yields non-zero readback
- Either: `shader.dispatch` on Titan V produces correct GPU readback

## Key Risks

- **470.x struct layout**: `Nvos64Parameters` is 32 bytes (not 48-byte 580.x `NvRmAllocParams`)
- **USERD in sysmem**: May fail DMA range check; VRAM USERD is preferred but more complex
- **GPU_PROMOTE_CTX**: Returns `INSUFFICIENT_PERMISSIONS` from userspace — skipped
- **Phase A adoption**: RM channel's instance block may not be readable after rmmod

## Files Changed

- `cylinder/src/bin/rm_trigger.rs` — extended to full 16-step RM channel creation
- `cylinder/src/nv/rm_abi.rs` — canonical RM ABI types (created in Deep Debt Sprint)
- `cylinder/src/vfio/sovereign_handoff/types.rs` — `RmChannelEvidence` struct
- `cylinder/src/vfio/sovereign_handoff/rm_trigger.rs` — `--channel` flag, evidence parsing
- `cylinder/src/vfio/sovereign_handoff/pipeline.rs` — PCCSR scan, evidence propagation
- `cylinder/src/vfio/sovereign_handoff/rollback.rs` — `rm_channel_evidence: None`

## Results

### Attempt 1 (2026-05-27)

Both Titan Vs stuck in D-state on vfio-pci unbind during catalyst pipeline.
Root cause: previous VFIO anchor session left kernel vfio-pci references that
prevent clean unbind even after anchor fd release. The sysfs write to
`/sys/bus/pci/drivers/vfio-pci/unbind` enters uninterruptible sleep (D-state).
This is a known PCIe subsystem issue requiring reboot to clear.

- Code: fully implemented, compiles clean, binary installed
- Hardware: requires reboot cycle to clear stuck PCIe state
- rm_trigger binary: struct sizes validated (Nvos64Parameters=32, NvChannelAllocParams=368)
- Pipeline: PCCSR scan, RmChannelEvidence, --channel flag all wired

### Post-Reboot Status (S279)

System fully cycled. Both Titan Vs clean on vfio-pci:
- 0000:02:00.0: Kernel driver in use: vfio-pci
- 0000:49:00.0: Kernel driver in use: vfio-pci

Workspace validation: 705 cylinder + 864 server = 1,569 lib tests pass. Full workspace `cargo check` clean.

### Attempt 2 (2026-05-27, S279): RM Channel Creation Fails

**Fix applied**: Removed anchor_release_guard cold-start halt (Exp 229 fix —
catalyst pipeline's purpose is to warm a cold GPU, not guard warm state).

**Pipeline ran to completion** (78.7s, `success=true`):
- 19/19 patches applied, nvsov loaded+bound, GPU opened (major=507)
- rm_trigger --channel: exit=0 but **root_alloc failed** — `status=0xdeadbeef`
  (sentinel never overwritten by RM). 0 of 15 steps completed.
- settle_health: "RM failed DEVINIT: PMC_ENABLE=0x40001121 (popcount=5)"
- 18,334 alive registers captured post-swap
- FECS INIT_CTXSW: status=0 (responded)
- Final tier: Cold (Tier 0)

**Root cause**: `nvidia_catalyst_handoff` patches NOP `init_module` too
aggressively — bytes at +0x7b/+0x8a overwrite the RM initialization call
that sets up the channel management state machine. Opening `/dev/nvidia0`
triggers `rm_init_adapter` (GPU device open), but `NV_ESC_RM_ALLOC` ioctls
need the full RM client/object infrastructure that only initializes during
the unpatched `init_module` flow.

**Key observations**:
- FECS_PC=0x18b37058 (RM firmware range) — FECS IS running
- FECS INIT_CTXSW returned status=0 (responsive)
- PMC_ENABLE=0x40001121 (only 5 engines vs ~23 normal)
- TPC probes: 0xbadf5040 FAULT (all 6 GPCs)
- GPCCS_CPUCTL=0x10 (halted), PMU_CPUCTL=0x20

**Paths forward**:
1. Reduce `init_module` patching — allow RM init to complete while still
   NOPping `nv_pci_remove` cleanup. The init_module bytes (+0x7b/+0x8a)
   suppress a critical RM init call.
2. Use the existing `nvidia_warm_handoff` strategy (unpatched nvidia-470)
   which does full RM init — but this destroys GPU state on removal.
3. Hybrid: unpatched load for channel creation, then patched reload for
   warm state preservation.

**Next**: investigate init_module patch scope — which call at +0x7b is being
NOPped and whether it can be preserved while still suppressing side effects
