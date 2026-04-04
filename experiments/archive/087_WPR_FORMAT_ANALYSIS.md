# Experiment 087: WPR Format Deep Analysis — Root Cause Found

**Date:** 2026-03-24
**Layer:** L8 (WPR/ACR payload + FECS/GPCCS boot)
**Status:** ANALYSIS COMPLETE — 7 WPR construction bugs identified
**Depends on:** Exp 083, 085, 086

## Context

Exp 086 confirmed Layer 8 is an INTERFACE problem (we're mis-programming), not a
key+lock (hardware security gate). B8-B11 candidates were proposed based on
register snapshot diffs. This experiment investigates those candidates AND performs
a byte-level comparison of our `build_wpr()` against nouveau's `gp102_acr_wpr_build`.

## B8-B11 Resolution

All four candidates from Exp 086 were investigated against upstream nouveau source
and firmware files. **None are actual bugs** — coralReef already handles them:

| # | Candidate | Finding |
|---|-----------|---------|
| B8 | BOOTVEC=0xFD00 | **NOT A BUG.** All ACR strategies already set `BOOTVEC = bl_start_tag << 8`. Value 0xFD00 comes from `acr/bl.bin` header (start_tag=0xFD). Confirmed by firmware parse. |
| B9 | UNK090 bits[18:17] | **NOT A BUG.** Nouveau only writes bit 16 (same as us). Extra bits in readback are hardware side effects, not driver writes. Confirmed by `gm200_flcn_bind_inst` source. |
| B10 | DMAIDX full clear | **NOT A BUG.** Nouveau only clears low 3 bits (mask 0x07→0x00), same as us. Full-zero readback is engine state, not explicit write. |
| B11 | FBIF_624 bit 7 | **NOT A BUG for our path.** Bit 7 = IGNORE_ACTIVATION (envytools `falcon.xml`). Nouveau sets it when `!fw->inst` (no instance block). We use instance blocks, so we correctly leave it unset. |

## The REAL Layer 8 Problem: WPR Image Construction

Comparing `wpr.rs:build_wpr()` to nouveau's `gp102_acr_wpr_build` + `gm200_acr_wpr_build_lsb_tail`
revealed **7 bugs in WPR construction**, two of which are critical.

### Firmware File Structure (Key Discovery)

The `gr/fecs_bl.bin` and `gr/gpccs_bl.bin` files are NOT raw bootloader code.
They are packaged firmware files with headers:

```
Offset  Size  Content
------  ----  -------
0x000   24B   nvfw_bin_hdr (magic=0x10DE, version, offsets)
0x018    8B   padding
0x020   28B   nvfw_hs_bl_desc (start_tag, code_off/size, etc.)
0x03C    4B   padding
0x040  512B   ACTUAL BOOTLOADER CODE (data_section)
------
Total: 576 bytes
```

Parsed from actual firmware files:

| File | Total | data_off | data_size | start_tag | IMEM entry |
|------|-------|----------|-----------|-----------|------------|
| fecs_bl.bin | 576 | 64 | 512 | 0x7E | 0x7E00 |
| gpccs_bl.bin | 576 | 64 | 512 | 0x34 | 0x3400 |
| acr/bl.bin | 1280 | 512 | 768 | 0xFD | 0xFD00 |

### Bug W1 (CRITICAL): BL File Headers Included in WPR Image

**File:** `wpr.rs:build_wpr()` lines 156-167

```rust
let fecs_img = [
    fw.fecs_bl.as_slice(),    // ← FULL 576-byte file, including headers!
    fw.fecs_inst.as_slice(),
    fw.fecs_data.as_slice(),
].concat();
```

**Problem:** `fw.fecs_bl` contains the entire 576-byte file (loaded via
`read("gr/fecs_bl.bin")` in `firmware.rs:210`). The WPR image should contain
ONLY the 512-byte code section (bytes 64..576). The extra 64 bytes of BL
file headers at the start of the image shift ALL subsequent offsets.

**Effect:** ACR reads app code starting at `app_code_off` from the image base,
but our image starts with 64 bytes of garbage headers, so ACR reads the wrong
data for FECS instruction code.

**Fix:** Extract only the code section: `fw.fecs_bl[data_off..data_off + data_size]`.

### Bug W2 (CRITICAL): bl_imem_off Hardcoded to 0

**File:** `wpr.rs:write_lsb()` line 271

```rust
buf[t + 16..t + 20].copy_from_slice(&0u32.to_le_bytes()); // bl_imem_off = 0
```

**Problem:** `bl_imem_off` tells the ACR firmware where to place the
bootloader in the target falcon's IMEM. Falcon bootloaders are compiled for
specific IMEM addresses (top of IMEM, defined by `start_tag << 8`):

- FECS: start_tag=0x7E → bl_imem_off should be **0x7E00** (not 0)
- GPCCS: start_tag=0x34 → bl_imem_off should be **0x3400** (not 0)

Falcon code is NOT position-independent. Loading at IMEM[0] when the code
expects IMEM[0x7E00] means jump targets, data references, and entry vectors
are all wrong. The BL crashes immediately.

**Fix:** Parse `start_tag` from each BL file header, set `bl_imem_off = start_tag << 8`.

### Bug W3 (MEDIUM): bl_code_size Includes BL Headers

**File:** `wpr.rs:write_lsb()` line 269 (via `bl_size` parameter)

```rust
buf[t + 12..t + 16].copy_from_slice(&(bl_size as u32).to_le_bytes()); // bl_code_size
```

Called with `fw.fecs_bl.len()` = 576. Should be the code section size = 512.

**Effect:** ACR tries to copy 576 bytes of "bootloader code" from the image,
but only 512 bytes are actual code (the first 64 are headers). This loads
garbage header bytes into the target falcon's IMEM.

**Fix:** Pass parsed `data_size` (512) instead of full file length.

### Bug W4 (MEDIUM): BLD DMA Addresses Use Wrong BL Size

**File:** `wpr.rs:build_wpr()` lines 303-305

```rust
let fecs_data_dma = wpr_vram_base + fecs_img_off as u64
    + fw.fecs_bl.len() as u64 + fw.fecs_inst.len() as u64;
```

Uses `fw.fecs_bl.len()` (576) instead of the code section size (512). The
`data_dma_base` is off by 64 bytes, so ACR reads FECS data from the wrong
location.

**Fix:** Use parsed BL data_size for all offset calculations.

### Bug W5 (MINOR): bl_data_size = 256 Instead of 84

**File:** `wpr.rs:write_lsb()` line 275

Should be `sizeof(flcn_bl_dmem_desc_v2) = 84`. The current 256 causes ACR
to copy 172 extra zero bytes to the target falcon's DMEM. Likely harmless
but incorrect.

### Bug W6 (MINOR): wpr_header_v1.bin_version = 0

**File:** `wpr.rs:build_wpr()` lines 213, 217

```rust
w32(&mut buf, 16, 0); // bin_version — should be sig.version (= 2)
```

Nouveau reads `bin_version` from `lsf_signature_v1.version` at sig+0x50.
FECS sig has version=2. May or may not be checked by ACR firmware.

### Bug W7 (MINOR): Depmap Area Corruption in Signature

**File:** `wpr.rs:write_lsb()` lines 254-257

```rust
buf[s + 88..s + 92] = bl_size;  // writes to depmap[0] area
```

Offsets 88-104 from LSB start fall in the `depmap[88]` array of
`lsf_signature_v1`. Since `depmap_count=0`, ACR likely ignores this, but
it corrupts the signed signature data.

## Impact Analysis

**W1 + W2 together are the root cause of WPR COPY stall.** The ACR firmware
receives a corrupted image (W1) with wrong offset metadata (W2-W4), tries to
load a bootloader to the wrong IMEM address (W2), and the bootloader code
itself is corrupted with header bytes (W1+W3). The copy starts (status=1) but
the loaded falcon immediately fails, leaving ACR stuck.

## Verification Data

Firmware file analysis confirms the structure:

```
fecs_sig.bin: 192 bytes = full lsf_signature_v1
  b_prd_present=1  b_dbg_present=1  falcon_id=2
  supports_versioning=1  version=2  depmap_count=0
  kdf=b4c231e903b277d70e32a0698f4e8062

acr/bl.bin: 1280 bytes
  nvfw_bin_hdr: magic=0x10DE ver=1 size=1280
  nvfw_hs_bl_desc: start_tag=0xFD → BOOTVEC=0xFD00
  code: 768 bytes at offset 512

fecs_bl.bin: 576 bytes
  nvfw_bin_hdr: magic=0x3B1D14F0 ver=1
  nvfw_hs_bl_desc: start_tag=0x7E → bl_imem_off=0x7E00
  code: 512 bytes at offset 64

gpccs_bl.bin: 576 bytes (→ gp107)
  start_tag=0x34 → bl_imem_off=0x3400
  code: 512 bytes at offset 64
```

FECS HWCFG=0x20204080 → code_limit = (HWCFG & 0x1FF) << 8 = 0x8000.
BL at top of IMEM: 0x8000 - 512 = 0x7E00 ✓ (matches start_tag).

GPCCS HWCFG=0x20102840 → code_limit = 0x4000.
BL at top: the start_tag placement is specific to each BL binary.

## Recommended Fix Plan (Exp 087 → coralReef)

### Phase 1: Parse BL file headers (firmware.rs)
- Add `GrBlFirmware` struct with parsed header fields
- Extract code section bytes, start_tag, code_size from each BL file
- Expose via `AcrFirmwareSet.fecs_bl_code()`, `.fecs_bl_start_tag()`, etc.

### Phase 2: Fix build_wpr (wpr.rs)
- Use extracted code sections instead of raw BL files in image concat
- Set bl_imem_off = start_tag << 8 for each falcon
- Use parsed code_size for bl_code_size and DMA offset calculations
- Fix bl_data_size to 84 (sizeof flcn_bl_dmem_desc_v2)
- Read bin_version from sig file
- Remove depmap area writes

### Phase 3: Validate
- Run ACR boot on Titan post-nouveau (optimal state from Exp 086)
- Check WPR header status field: should transition from COPY(1) to loaded
- Monitor FECS/GPCCS bind_stat after ACR completion

## Hardware Validation Results (2026-03-24)

W1-W7 fixes applied to `firmware.rs` + `wpr.rs` and validated on Titan #1
(0000:03:00.0) in post-nouveau state.

### W1-W7 Fix Confirmation (Logs)

```
Parsed GR BL firmware name="fecs_bl" file_size=576 data_off=64 data_size=512 start_tag=126 bl_imem_off=32256
Parsed GR BL firmware name="gpccs_bl" file_size=576 data_off=64 data_size=512 start_tag=52  bl_imem_off=13312

WPR layout wpr_size=52480
  fecs_lsb=768  fecs_img=4096  fecs_img_size=30932  fecs_bld=35072  fecs_bl_imem_off=32256
  gpccs_lsb=35328 gpccs_img=36864 gpccs_img_size=15283 gpccs_bld=52224 gpccs_bl_imem_off=13312
```

Old `fecs_img_size` was 30996 (576+25632+4788). New is 30932 (512+25632+4788).
The 64-byte reduction confirms W1 (BL header stripping) is active.

### BREAKTHROUGH: ACR Now Processes WPR Successfully

**Strategy 5 (ACR mailbox command via live SEC2):**
```
SEC2 pre-command:                cpuctl=0x00000000 mb0=0xcafebeef mb1=0x00000000
After BOOTSTRAP_FALCON(FECS):   mb0=0x00000001    mb1=0x00000002
After BOOTSTRAP_FALCON(GPCCS):  mb0=0x00000001    mb1=0x00000003
FECS cpuctl:  0x00000012  (HALTED + ALIAS_EN)
GPCCS cpuctl: 0x00000012  (HALTED + ALIAS_EN)
```

### Before vs After W1-W7

| Metric | Before (Exp 085) | After (Exp 087) |
|--------|-------------------|------------------|
| WPR COPY status | Stuck at 1 (never completes) | **Completes** — ACR processes both entries |
| FECS cpuctl | 0x10 (HRESET, untouched) | **0x12** (HALTED + ALIAS_EN set by ACR) |
| GPCCS cpuctl | 0x10 (HRESET, untouched) | **0x12** (HALTED + ALIAS_EN set by ACR) |
| BOOTSTRAP_FALCON | No response | **mb0=1, mb1=falcon_id** (command acknowledged) |
| SEC2 DMEM | WPR data absent | WPR descriptor present at 0x210-0x264 |

### Interpretation

The W1-W7 fixes **unblocked Layer 8 entirely**. ACR can now:
1. Parse the WPR header table
2. Find FECS and GPCCS LSB entries
3. Read the correct firmware images (no header corruption)
4. Load BL code to the correct IMEM address (0x7E00 for FECS, 0x3400 for GPCCS)
5. Set ALIAS_EN on target falcons (HS mode configuration)

The remaining issue is **Layer 9**: FECS/GPCCS end up HALTED (cpuctl=0x12)
instead of RUNNING (cpuctl=0x00). The `mb0=1` return may indicate an ACR
error code, or it may mean "command complete" (needs protocol research).

**Possible L9 causes:**
- ACR's BL loaded correctly but the app code entry point is wrong
- HS mode requires additional page table or WPR hardware configuration
- The falcon BL halted after loading (waiting for host signal)
- SEC2 ACR firmware version mismatch with WPR format expectations

## Notes

Source research method: upstream nouveau kernel sources (torvalds/linux),
envytools rnndb register database, and direct firmware binary analysis.
W1-W7 fixes applied to coralReef `firmware.rs` + `wpr.rs` — compiles clean.
