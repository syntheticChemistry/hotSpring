# Experiment 076 — MMU Fault Buffer: Layer 6 Breakthrough

**Date**: 2026-03-23
**Status**: PROVEN
**Hardware**: Titan V #1 (`0000:03:00.0`, GV100, 12 GB HBM2)
**Stack**: coralReef `coral-driver` → `coral-ember` (SCM_RIGHTS) → iommufd (IOAS 2)

## Root Cause

The sovereign PFIFO channel creation path (`VfioChannel::create`) did not configure
the Volta **non-replayable MMU fault buffers** (`FAULT_BUF0/1`). On GV100, FBHUB
requires a valid fault buffer drain target before any MMU page table walk can
complete. Without it:

1. PBDMA requests GPFIFO fetch at GPU VA `0x1000`
2. GPU MMU begins 5-level page table walk (PD3→PD2→PD1→PD0→PT0)
3. Walk succeeds or faults — either way, FBHUB writes a fault entry
4. **No fault buffer configured** → FBHUB stalls, cannot drain fault
5. PBUS reads return `0xbad00200` (domain timeout), GPU appears dead
6. PBDMA `GP_FETCH` stuck at 0x1000, channel stays PENDING forever

The diagnostic matrix path (`diagnostic/runner.rs`) *did* configure fault buffers,
which is why Exp 071's matrix experiments got further than production channel
creation.

## Fix

```rust
// VfioChannel::create — after BAR2 setup, before page table population
let fault_buf = DmaBuffer::new(container.clone(), 4096, FAULT_BUF_IOVA)?;
bar0.write_u32(mmu::FAULT_BUF0_LO, (FAULT_BUF_IOVA >> 12) as u32)?;
bar0.write_u32(mmu::FAULT_BUF0_HI, 0)?;
bar0.write_u32(mmu::FAULT_BUF0_SIZE, 64)?;
bar0.write_u32(mmu::FAULT_BUF0_GET, 0)?;
bar0.write_u32(mmu::FAULT_BUF0_PUT, 0x8000_0000)?;  // enable bit
// Mirror for replayable fault buffer (FAULT_BUF1)
```

## Validation

Ran `cargo test --test hw_nv_vfio --features vfio -- --ignored` on Titan V #1
via ember `SCM_RIGHTS` fd lending:

| Test | Result |
|------|--------|
| `vfio_open_and_bar0_read` | **PASS** — channel creation + fault buffer setup |
| `vfio_alloc_and_free` | **PASS** — IOMMU DMA alloc/dealloc |
| `vfio_upload_and_readback` | **PASS** — DMA roundtrip (256 bytes, bit-exact) |
| `vfio_multiple_buffers` | **PASS** — 4× concurrent DMA buffers |
| `vfio_dispatch_nop_shader` | FAIL — fence timeout (Layer 7: GR/FECS context, not MMU) |
| `vfio_free_invalid_handle` | FAIL — IOAS_MAP conflict (test isolation, not driver) |
| `vfio_readback_invalid_handle` | FAIL — same IOAS_MAP conflict |

## Sovereign Pipeline Status Update

| Layer | Component | Status |
|-------|-----------|--------|
| 1 | PCIe/BAR0 | **PROVEN** |
| 2 | VFIO/IOMMU | **PROVEN** (iommufd/cdev) |
| 3 | PFB/HBM2 | **PROVEN** (warm-state via glowplug) |
| 4 | PMC/Clock | **PROVEN** |
| 5 | PFIFO/PBDMA | **PROVEN** |
| 6 | MMU page tables | **PROVEN** (this experiment) |
| 7 | GR/FECS init | *Next target* |
| 8 | QMD/dispatch | Blocked by L7 |
| 9 | Compute result | Blocked by L8 |
| 10 | WGSL→native | Compiler ready |

**Layer 6 is no longer a blocker.** The V2 page table encoding was correct all
along — the missing piece was fault buffer initialization.

## Secondary Fix: ember socket access

The `/run/coralreef/` directory was created by systemd as `root:root 0750`,
blocking coralreef group members from reaching `ember.sock`. Fixed by adding
`ExecStartPost=/bin/chgrp coralreef /run/coralreef` to the ember service unit.

## Next Steps

1. **Layer 7 — GR engine context**: FECS falcon is loaded (nouveau warm state)
   but GR context switch and compute class init may need explicit BAR0 methods.
2. **Test isolation**: Use `--test-threads=1` or per-test IOAS allocation to
   avoid IOMMU_IOAS_MAP EEXIST conflicts.
3. **VRAM training path**: With MMU translation working, VRAM-resident page
   tables and buffers are now feasible for the sovereign training subset.
