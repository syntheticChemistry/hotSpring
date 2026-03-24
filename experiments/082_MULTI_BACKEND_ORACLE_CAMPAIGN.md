# Experiment 082: Multi-Backend Oracle Campaign

**Date:** 2026-03-23
**Goal:** Capture mmiotrace + falcon state from every driver backend on both Titan V GPUs. The captured traces become permanent "recipes" for the ACR boot solver.
**Status:** BLOCKED — awaiting coralReef team's trace integration (handoff delivered)

## Concept: Drivers as Software Tools

The RTX 5060 owns the system nvidia.ko (580.126.18 open, refcount=208, display DRM pinned). The Titans live on vfio-pci under GlowPlug/Ember. When we bind a Titan to a driver, we treat that driver as a **tool invocation** -- capture every MMIO write, then return to VFIO. The captured trace is a permanent asset.

Future evolution: `nvidia_oracle.ko` (renamed open module) enables driver version coexistence without touching the 5060. GlowPlug deploys oracle modules on demand.

## Hardware

- Titan V #1: `0000:03:00.0` (IOMMU group 69) -- oracle station
- Titan V #2: `0000:4a:00.0` (IOMMU group 34) -- second data point
- RTX 5060: `0000:21:00.0` -- nvidia 580.126.18 open, UNTOUCHED

## Campaign Sequence (Per Titan)

| Step | Backend | Captures | Purpose |
|------|---------|----------|---------|
| 1 | vfio-pci (cold) | BAR0 dump | Baseline register state |
| 2 | nouveau | mmiotrace, warm BAR0, residual BAR0 | Open driver init recipe |
| 3 | vfio-pci (post-nouveau) | BAR0 dump | Residual state after nouveau |
| 4 | nvidia (580 open) | mmiotrace, warm BAR0, residual BAR0 | **Proprietary ACR boot recipe** |
| 5 | vfio-pci (post-nvidia) | BAR0 dump | "Sub-cold" state (Exp 070 confirmed) |

## Key Captures for ACR Boot Solver

The nvidia mmiotrace during bind will show:
- SEC2 wakeup sequence (CPUCTL, BOOTVEC, ITFEN writes)
- Instance block setup and `bind_inst` (0x668) write
- `bind_stat` (0x0dc) polling pattern
- DMA context configuration
- ACR firmware load DMA setup
- MAILBOX0 polling sequence
- FECS/GPCCS HRESET release

## Execution Method

### Previous: External Script (Obsolete)

The original `capture_multi_backend.sh` required `pkexec` and had a control problem —
the unbind step hung on VFIO-bound devices because it operated outside GlowPlug's lifecycle.

### Current: Native GlowPlug Integration (Pending coralReef Implementation)

A detailed handoff has been delivered to the coralReef team:
`wateringHole/handoffs/CORALREEF_TRACE_INTEGRATION_HANDOFF.md`

Once implemented, the workflow is:

```bash
# Capture with trace during swap (Ember handles mmiotrace natively):
coralctl swap 0000:03:00.0 nouveau --trace
coralctl swap 0000:03:00.0 nvidia  --trace
coralctl swap 0000:4a:00.0 nouveau --trace
coralctl swap 0000:4a:00.0 nvidia  --trace

# Later, with nvidia_oracle built:
sudo hotSpring/scripts/build_nvidia_oracle.sh 580.126.18
coralctl swap 0000:03:00.0 nvidia_oracle --trace

# Check captures:
coralctl trace-list
```

### nvidia_oracle Build Script

Lives in `hotSpring/scripts/build_nvidia_oracle.sh`. Patches `MODULE_BASE_NAME` and
`NV_MAJOR_DEVICE_NUMBER` in the open kernel source, builds a coexisting `.ko`.

## Results

### Titan #1 (0000:03:00.0)

#### nouveau

*(pending execution)*

#### nvidia (580.126.18 open)

*(pending execution)*

### Titan #2 (0000:4a:00.0)

#### nouveau

*(pending execution)*

#### nvidia (580.126.18 open)

*(pending execution)*

## Diffs

### nouveau vs nvidia (falcon init)

*(pending -- diff mmiotrace_falcon_init.txt between backends)*

### Titan #1 vs Titan #2 (same driver)

*(pending -- diff between cards reveals hardware-specific vs universal patterns)*

## nvidia_oracle.ko Design (Phase 3)

Build a renamed nvidia module that coexists with the system nvidia.ko:

1. `MODULE_BASE_NAME "nvidia"` → `"nvidia_oracle"` in `nv-linux.h`
2. `NV_MAJOR_DEVICE_NUMBER 195` → `0` (dynamic) in `nv-chardev-numbers.h`
3. Build as separate `.ko`, `insmod` alongside system module
4. GlowPlug binds Titans to `nvidia_oracle` via `driver_override`
5. 5060 stays on `nvidia`, completely unaffected

Enables testing ANY nvidia version: `nvidia_oracle_535.ko`, `nvidia_oracle_525.ko`, etc.
Each version's mmiotrace becomes a permanent recipe in the driver library.

## Findings from Header Cross-Reference

Analysis of `/usr/src/nvidia-580.126.18/nvidia-uvm/hwref/volta/gv100/`:

| Finding | Detail | Impact |
|---------|--------|--------|
| PTE `>> 4` encoding | Confirmed correct for 4K-aligned addresses (equivalent to `>> 12` with implicit shift) | ACR boot PTE logic is sound |
| `SYS_MEM_COH_TARGET` inconsistency | `attempt_sysmem_acr_boot` uses `2`, `attempt_acr_chain` uses `3` for sysmem coherent DMA target at `0x668` bind_inst | **Needs reconciliation** — verify against nouveau source |
| Fault buffer register names | `GET/PUT/SIZE` offsets in `registers.rs` (`0xE2C/0xE30/0xE34`) mismatch `dev_fb.h` naming | cosmetic — addresses match, names differ |
| SEC engine ID 14 | `NV_PFAULT_MMU_ENG_ID_SEC` = 14 in `dev_fault.h` | Maps SEC2 engine in fault diagnostics |

## Dependencies

- **coralReef team**: Implement trace integration per handoff spec (Phase 1-6)
- **hotSpring**: `build_nvidia_oracle.sh` ready, experiment journal ready
- **Unblocks**: Exp 082 campaign execution, future oracle campaigns for other driver versions
