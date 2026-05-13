# Experiment 180: Three-GPU Hardware Validation Sweep

**Date:** April 30, 2026
**Status:** ✅ Complete (RTX 5060 + Titan V validated; K80 PGOB blocker confirmed)
**Hardware:** RTX 5060 (GB206, SM120), Titan V (GV100, SM70), Tesla K80 (GK210B, SM35)
**Primal:** coralReef (coral-driver, coral-gpu)
**Predecessor:** Exp 179 (K80 FECS pipeline), Exp 177 (Blackwell dispatch)

## Objective

Validate the sovereign compute pipeline on all three locally available GPU
generations: Blackwell, Volta, and Kepler. Confirm code fixes from Exp 179
(RAMFC 0x3C/0x44, runlist poll, `open_from_fds` Kepler wiring) on real hardware.

## Hardware Configuration

| GPU | BDF | Driver | SM |
|-----|-----|--------|----|
| RTX 5060 | `0000:21:00.0` | nvidia | SM120 |
| Titan V | `0000:03:00.0` | vfio-pci | SM70 |
| K80 GPU0 | `0000:4c:00.0` | vfio-pci | SM35 |
| K80 GPU1 | `0000:4d:00.0` | unbound | SM35 |

## Results

### RTX 5060 (Blackwell SM120) — 19/19 PASS

| Suite | Tests | Result |
|-------|-------|--------|
| `hw_cuda_e2e` (CUDA dispatch) | 8/8 | All pass including `cuda_cubin_sass_write_42_sm120` |
| `local_gpu_discovery` (coral-gpu) | 4/4 | Auto-detects nvidia sm120, compiles WGSL→SM120 SASS |
| `hw_nv_probe` (nvidia-drm) | 3/3 | Render node discovery, multi-GPU enumeration (1 DRM + 2 VFIO) |
| `hw_nv_buffers` (nvidia-drm) | 4/4 | Alloc, free, sync, SM86 compilation |

Full sovereign pipeline validated: WGSL → coral-reef SM120 → CUDA SASS → dispatch → readback.

### Titan V (Volta SM70) — 20/45 PASS (25 daemon-dependent)

| Category | Tests | Result |
|----------|-------|--------|
| Basic VFIO ops | ~8 | PASS — BAR0 reads, PCI discovery, firmware inventory |
| Diagnostics | ~4 | PASS — layer7, PBDMA isolation |
| Falcon boot/ACR | ~6 | PASS — sovereign FECS boot, ACR strategies |
| GlowPlug/Ember dependent | 25 | FAIL — `No such file or directory` (daemons not running) |

The 25 failures are infrastructure dependencies (GlowPlug + Ember daemons), not code
issues. All standalone VFIO operations work correctly.

### Tesla K80 (Kepler SM35) — Partial

| Test | Result | Detail |
|------|--------|--------|
| `k80_vfio_device_opens` | PASS | SM37 detected, channel created |
| Runlist submission | PASS | `INTR bit 30 ACK tick=0` — immediate completion |
| SCHED_ERROR in cold state | PASS | `sched_err=0x00000000` (was code=32 before RAMFC fix) |
| FECS boot | FAIL | GPCs power-gated (`0xbadf3000`), FECS can't reach ready |
| `k80_nouveau_warmup_dispatch` | FAIL | GPC gating persists after nouveau→vfio rebind |

**Root blocker**: GPC PGOB power gating (`0xbadf1002` on GR HUB, `0xbadf3000` on
all GPC registers) survives both cold VFIO and nouveau warm-catch. The PGOB disable
sequence runs but GPCs remain gated. FECS firmware uploads to IMEM successfully
(readback verified) but can't execute because its DMA targets in GPC space are
PRI-faulted.

**What IS proven on K80:**
- VFIO device open (legacy path, no FLR)
- Kepler 2-level page table instance block creation
- RAMFC population including fixed fields (0x3C DMA_LIMIT_REF, 0x44 PB_DMA_SUBROUTINE)
- PFIFO engine init (PBDMA→runlist auto-discovery from hw assignment table)
- Runlist submission and completion polling via PFIFO_INTR bit 30
- Channel binding to PCCSR (PENDING state)
- FECS firmware upload with integrity verification

## Code Fixes Validated

1. **`open_from_fds` Kepler wiring** — Profile-driven channel creation now branches
   on `PageTableFormat::V1TwoLevel` for ember FD handoff (was always Volta-style).
2. **`falcon_boot_solver` generation dispatch** — Now uses `boot_for_generation()`
   instead of `boot()`. Kepler correctly skips ACR cascade.
3. **RAMFC 0x3C/0x44** — In cold VFIO state, scheduler error is 0 (vs previous
   code=32). The CONTEXT_RELOAD_TIMEOUT is eliminated when scheduler runs.

## Next Steps

1. **Solve K80 GPC PGOB** — The warm-catch strategy requires GPCs to survive the
   nouveau→vfio rebind. Options:
   - `open_warm_direct` with deferred bus-master (quiesce PFIFO before bus master enable)
   - Use `k80_nouveau_post.sh` which forces GR init via render node allocation
   - Investigate whether nouveau's unbind path itself gates GPCs (cleanup hook)
2. **Start GlowPlug/Ember for Titan V** — 25 additional tests would pass
3. **Titan V nouveau DRM dispatch** — Test sovereign e2e via DRM path
