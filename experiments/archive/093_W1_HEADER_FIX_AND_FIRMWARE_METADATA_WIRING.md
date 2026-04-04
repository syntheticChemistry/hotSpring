# Exp 093: W1 Header Fix + Firmware Metadata Wiring

**Date:** 2026-03-25
**Status:** HARDWARE VALIDATED — Path B blocked by LS mode authentication
**Depends:** Exp 091 (BOOTVEC discovery), Exp 092 (experiment loop), coralReef Iter 67
**Goal:** Fix the W1 header bug in the direct PIO boot path and wire BOOTVEC from firmware metadata across all boot strategies

## Summary

Three interrelated bugs were discovered and fixed in `coralReef` `coral-driver`:

1. **W1 Header Bug** — `fecs_boot.rs` loaded raw `*_bl.bin` files (including `nvfw_bin_hdr` + `nvfw_hs_bl_desc` headers) and uploaded them directly to IMEM. The falcon tried to execute header bytes as code, causing the `0x0307` exception seen in Exp 091.

2. **Wrong IMEM Layout** — The direct PIO path uploaded BL at IMEM[0] and set BOOTVEC=0. But the BL code is position-dependent — it expects to run at `bl_imem_off` (0x3400 for GPCCS, 0x7E00 for FECS). The correct layout is: inst at IMEM[0], BL at IMEM[bl_imem_off], BOOTVEC=bl_imem_off.

3. **Hardcoded BOOTVEC** — `strategy_mailbox.rs` used local `const GPCCS_BL_IMEM_OFF: u32 = 0x3400` / `const FECS_BL_IMEM_OFF: u32 = 0x7E00`. These values are chip-specific and should come from firmware metadata (`GrBlFirmware::bl_imem_off()`), not hardcoded constants.

## Code Changes

### 1. `fecs_boot.rs` — Parse BL through `GrBlFirmware`

`FecsFirmware::load()` and `GpccsFirmware::load()` now parse `*_bl.bin` files through `GrBlFirmware::parse()`, which:
- Strips the `nvfw_bin_hdr` + `nvfw_hs_bl_desc` headers
- Extracts only the code section (via `data_offset` + `data_size`)
- Computes `bl_imem_off` from `start_tag << 8`

New struct fields: `FecsFirmware::bl_imem_off`, `GpccsFirmware::bl_imem_off`.

### 2. `fecs_boot.rs` — `falcon_boot()` IMEM layout fix

`falcon_boot()` now takes a `bl_imem_off: u32` parameter:
- Inst code → IMEM[0] (application code base)
- BL code → IMEM[bl_imem_off] (position-dependent entry point)
- BOOTVEC → bl_imem_off (not 0)

Same change applied to `falcon_boot_probed()` for the capability-based path.

### 3. `strategy_mailbox.rs` — Firmware-derived BOOTVEC

New `FalconBootvecOffsets` struct carries firmware-derived BL IMEM offsets. `attempt_acr_mailbox_command()` now takes `&FalconBootvecOffsets` instead of using local hardcoded constants. The solver constructs it from `fw.gpccs_bl.bl_imem_off()` / `fw.fecs_bl.bl_imem_off()`.

### 4. `registers.rs` — Named FBIF constants

Added to the `falcon` module in `registers.rs`:
- `FBIF_TRANSCFG` (0x624) — falcon bus interface configuration
- `FBIF_TARGET_VIRT` (0x00), `FBIF_TARGET_PHYS_VID` (0x01), `FBIF_PHYSICAL_OVERRIDE` (0x80)

All raw `0x624` literals in `sec2_hal.rs` replaced with named constants.

### 5. `acr_boot/mod.rs` — Export `GrBlFirmware`

`GrBlFirmware` added to the public re-exports so `fecs_boot.rs` can import it.

### 6. Test: `vfio_sovereign_gr_boot` — Complete fix

The test now:
- Uses `GrBlFirmware`-parsed firmware (headers stripped)
- Uploads inst to IMEM[0], BL to IMEM[bl_imem_off]
- Sets BOOTVEC to bl_imem_off per falcon
- Logs `bl_imem_off` for each falcon in diagnostics

## Files Changed (coralReef)

| File | Change |
|------|--------|
| `crates/coral-driver/src/nv/vfio_compute/fecs_boot.rs` | BL header parsing, `bl_imem_off` fields, `falcon_boot()` + `falcon_boot_probed()` signature evolution |
| `crates/coral-driver/src/nv/vfio_compute/acr_boot/strategy_mailbox.rs` | `FalconBootvecOffsets`, firmware-derived BOOTVEC |
| `crates/coral-driver/src/nv/vfio_compute/acr_boot/solver.rs` | Pass `FalconBootvecOffsets` to Strategy 5 |
| `crates/coral-driver/src/nv/vfio_compute/acr_boot/mod.rs` | Export `GrBlFirmware`, `FalconBootvecOffsets` |
| `crates/coral-driver/src/nv/vfio_compute/acr_boot/sec2_hal.rs` | Named FBIF constants |
| `crates/coral-driver/src/vfio/channel/registers.rs` | `FBIF_TRANSCFG`, `FBIF_TARGET_*`, `FBIF_PHYSICAL_OVERRIDE` |
| `crates/coral-driver/tests/hw_nv_vfio/falcon.rs` | Fixed `vfio_sovereign_gr_boot` + `vfio_fecs_acr_boot_and_probe` |

## Hardware Validation Plan

### Phase 1: Path B — BOOTVEC + Header Fix (highest priority)

```bash
# Warm up engines via nouveau, then swap to VFIO
coralctl swap 0000:03:00.0 nouveau
sleep 10
coralctl swap 0000:03:00.0 vfio

# Run the fixed sovereign GR boot test
cargo test vfio_sovereign_gr_boot --features vfio -p coral-driver -- --nocapture --ignored
```

**Success criterion:** `gpccs_exci == 0 && gpccs_pc != 0 && fecs_pc != 0`. Exp 090 proved cpuctl==0x00 alone is misleading — check PC and EXCI.

**If FECS signals ready** (CTXSW_MAILBOX bit 0): L10+L11 cracked. Try FECS method probe for ctx_size.

### Phase 2: Boot solver with corrected Strategy 5

```bash
cargo test vfio_falcon_boot_solver --features vfio -p coral-driver -- --nocapture --ignored
```

All 10 strategies now use firmware-derived BOOTVEC. Strategy 5 (ACR mailbox) should show the same fix.

### Phase 3: Path A — SEC2 DMA (FBIF cold-boot)

If Path B succeeds, Path A becomes lower priority. If not:
- Run `sec2_prepare_physical_first` with the physical DMA mode (FBIF bit 7)
- Compare cold-VFIO vs warm-swap FBIF register state
- Instrument FBIF_TRANSCFG reads across both Titan Vs for characterization

## Hardware Validation Results

### Run 1: Direct PIO (secure=false)

```
Pre-boot:  FECS cpuctl=0x10 sctl=0x3000 | GPCCS cpuctl=0x10 sctl=0x3000
Firmware:  GPCCS bl=512B off=0x3400 | FECS bl=512B off=0x7e00
Post-start: GPCCS cpuctl=0x10 exci=0x02070000 pc=0x0000
            FECS  cpuctl=0x10 exci=0x02070000 pc=0x0a86
```

### Run 2: Direct PIO (secure=true, PMC_UNK260, IMEM verify)

```
PMC_UNK260: 0xbad00200 → 0xbad00200 (PRI error — register inaccessible)
Pre-exci:   FECS=0x00070000 GPCCS=0x00070000 (residual from nouveau)
IMEM verify: GPCCS [0x0000]=0x001400d0 [0x3400]=0x000400d0 — firmware present
Post-start: GPCCS cpuctl=0x10 exci=0x02070000 pc=0x0000
            FECS  cpuctl=0x10 exci=0x02070000 pc=0x0ae1
GPCCS PC trace (100ms): all zeros — never executed
```

### Run 3: Full Boot Solver (12 strategies)

Key observations from Strategy 12 (direct PIO with PMC GR reset):
```
PMC GR reset applied
IMEM verify: [0x0000]=0x001400d0 [0x3400]=0x000400d0
GPCCS after start: cpuctl=0x00000012 pc=0x0000 exci=0x00070000
```

**cpuctl=0x12** (STARTCPU|HALTED) — falcon ACKed the start command but immediately halted.

SEC2 strategies (1, 3, 11) successfully started SEC2 (mb0 changed), but SEC2 cannot DMA-load FECS/GPCCS firmware due to FBIF issues.

### Analysis

**Exception 0x02 at PC=0x0000** is an instruction authentication failure. In LS mode (SCTL=0x3000, fuse-enforced on GV100), FECS/GPCCS validate that code was loaded through the ACR DMA path. Host PIO uploads produce valid IMEM content (verified via readback) but are NOT authenticated — the falcon's secure boot hardware rejects execution.

**FECS PC drift** (0x0ae0→0x0ae1 across runs) is the ROM exception handler, not our firmware. The startup triggers exception 0x02 at PC=0, and the ROM handler at ~0x0ae0 catches it.

**Path B (direct PIO) is hardware-blocked on GV100 FECS/GPCCS.** The only route is Path A (SEC2 ACR → DMA-authenticated load → FECS/GPCCS).

## Strategic Pivot: Path A (SEC2 DMA) is the Only Route

The solver confirms SEC2 CAN start and execute from PIO upload (mb0=0xcafebeef). SEC2 runs the ACR BL which then needs DMA to:
1. Read WPR image from VRAM
2. Verify signatures
3. DMA-load FECS/GPCCS firmware into their IMEM (authenticated path)
4. Release FECS/GPCCS

The DMA step is blocked by Gap 14 (FBIF circular dependency / page table setup). **Cracking SEC2 DMA is now the critical path.**

## Root Cause Chain

```
Exp 091: GPCCS faults at PC=0x0000
  → Exp 091 analysis: BOOTVEC=0
    → Exp 093 code fix: BL headers stripped, BOOTVEC=bl_imem_off
      → Exp 093 hw validation: still faults at PC=0x0000 (exci=0x02070000)
        → Root cause: LS mode authentication blocks PIO-loaded code
          → Path B dead for FECS/GPCCS on GV100
            → Path A (SEC2 DMA) is the only route
```
