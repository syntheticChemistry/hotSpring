# GPU Cracking Gap Tracker

**Updated:** 2026-03-26 (Exp 098 — clean-boot DMA trap identified, WPR copy partial, blob_base/size decoded)
**Goal:** Sovereign compute on Titan V (GV100) — FECS/GPCCS running without proprietary ACR chain

## MYTH BUSTED: SCTL Does NOT Block PIO (Exp 091+)

**Discovery:** The IMEMC register on GM200+ falcons uses **BIT(24)** (`0x0100_0000`)
for write auto-increment, not BIT(6) (`0x40`). All previous manual `coralctl` PIO
tests used the wrong control word format, creating the false impression that
SCTL=0x3000 blocks PIO access. **PIO to IMEM/DMEM/EMEM works regardless of
security mode** when the correct format is used.

**Evidence chain:**
1. nouveau `nvkm_falcon_v1_load_imem()` in `nvkm/falcon/v1.c` uses `start_addr | BIT(24)`
2. All Rust upload functions (`falcon_upload_imem`, `falcon_imem_upload_nouveau`) already
   used the correct BIT(24) format — the bug was only in manual `coralctl` commands
3. Hardware-validated: SEC2 IMEM write of `0xDEADBEEF` + readback confirmed on both Titan V cards
4. SCTL=0x3000 is fuse-enforced LS mode on GV100 — informational, not a PIO gate

**Impact:** Many experiment decisions driven by "must clear SCTL" were unnecessary:
FLR attempts (Titan V has no FLR), SBR for SCTL clearing, warm handoff to preserve
firmware. The actual remaining blocker is **DMA configuration** (FBIF mode, page tables),
not security mode.

**Code changes:** `FalconCapabilityProbe` in `falcon_capability.rs` now discovers PIO
format at runtime via bit-solving, preventing future IMEMC-class bugs on any GPU generation.

## Layer Model

| Layer | System | Status | Blocker |
|-------|--------|--------|---------|
| L1 | VFIO device binding | SOLVED | — |
| L2 | BAR0/BAR2 access | SOLVED | — |
| L3 | PMC enable + engine enumeration | SOLVED | — |
| L4 | PFIFO init + PBDMA discovery | SOLVED | — |
| L5 | MMU/FBHUB fault buffer setup | PROVEN (Exp 076) | — |
| L6 | PBDMA context load + channel submit | PARTIAL (Exp 058) | GP_PUT DMA read fails |
| L7 | SEC2 falcon binding + DMA | **CONFIRMED** (Exp 096) | **HS mode (0x3002) reproducible** via sysmem DMA + blob_size=0. Both sysmem and physical-sysmem paths achieve HS. VRAM DMA still corrupts (FBHUB PRI-dead). See Gap 14. |
| L8 | WPR/ACR payload + FECS/GPCCS boot | **SOLVED** (Exp 087) | W1-W7 fixed. ACR processes WPR, bootstraps FECS+GPCCS. |
| L9 | FECS boot + GR register init | **PARTIAL** (Exp 089b) | FECS halted at PC~0x058f, wakes on GR init. GPCCS stuck. |
| L10 | GPCCS bootstrap via ACR | **DMA TRAP** (Exp 098) | Volta SEC2 = one-shot ACR loader. WPR copy initiated (FECS=1 GPCCS=1) but DMA traps (EXCI=0x201F). Init msg in EMEM[0x80]. Next: check post-nouveau FECS/GPCCS state (Path R) or fix DMA mapping (Path Q). |
| L11 | GR context init + shader dispatch | **BLOCKED** by L10 | Needs GPCCS alive via authenticated ACR DMA load. |

## Active Gaps

### Gap 1: ACR Boot Chain — `bind_stat` Never Reaches 5

**Layer:** L7
**Exp:** 081, **083 (source analysis)**, 084, **085 (SOLVED)**
**Status:** **SOLVED** (Exp 085) — B1-B7 all fixed, bind_stat=5 on both Titans

The ACR boot chain requires writing a SEC2 instance block, then polling
`bind_stat` at `0x0dc` bits [14:12] until they equal 5. Currently they never reach 5.

**Exp 083 Discovery — Four Bugs Found:**

| # | Bug | Current (coralReef) | Correct (nouveau) | Severity |
|---|-----|--------------------|--------------------|----------|
| B1 | **Wrong register offset** | `SEC2_FLCN_BIND_INST = 0x668` | `gm200_flcn_bind_inst` writes `0x054` | **CRITICAL** — writes to wrong address |
| B2 | **Missing bit 30** | `(target << 28) \| (addr >> 12)` | `(1 << 30) \| (target << 28) \| (addr >> 12)` | HIGH — may be "enable" flag |
| B3 | **Wrong target in strategy_chain** | `SYS_MEM_COH_TARGET = 3` | 2 = coherent (HOST), 3 = non-coherent (NCOH) | MEDIUM — only affects one strategy |
| B4 | **Missing DMAIDX clear** | No write to `0x604` | `nvkm_falcon_mask(falcon, 0x604, 0x07, 0x00)` | MEDIUM — may default to 0 |

**Evidence source:** Upstream nouveau `nvkm/falcon/gm200.c`:
- `gm200_flcn_bind_inst(falcon, target, addr)` → writes falcon+`0x054`
- `gm200_flcn_fw_load()` → HOST=2, NCOH=3, VRAM=0
- GV100 SEC2 uses `gp102_sec2_flcn` → delegates to `gm200_flcn_bind_inst`

**B1-B4 FIXES APPLIED AND TESTED (Exp 084, 2026-03-24):** All four bugs
fixed. Hardware test confirms bind_inst at 0x054 accepts writes (values read
back correctly: 0x40000010 for VRAM, 0x60000040 for SysMem). BUT bind_stat
at 0x0dc stays at `0x000e003f` (bits[14:12]=0) — binding mechanism does NOT
activate. SEC2 firmware starts (PC reaches 0x0072/0x007c) then traps on DMA.

**Current blocker:** The falcon's internal bind state machine isn't starting
despite correct register writes. The 0x0dc value appears static (capability
register?). There may be a missing prerequisite (HS mode exit, clock gate,
power state) or 0x0dc may not be the correct bind_stat for GV100 SEC2.

**Data collected:**
- [x] **Nouveau warm comparison** — Exp 086 cross-driver profiling (both Titans)
- [x] **Full falcon register dump** — Exp 086 captured 50+ registers per engine per state
- [x] **bind_stat observation** — Exp 085 confirmed bind_stat=5 after B5-B7 trigger writes
- [x] **Cross-driver source analysis** — nouveau, nvidia-open, Mesa patterns compared (Exp 083, 085)
- [x] **WPR format analysis** — Exp 087 byte-level comparison resolved Layer 8

**Resolution:** B1-B4 fixed register-level bugs, B5-B7 added missing trigger writes
from nouveau's `gm200_flcn_bind_inst`. `falcon_bind_context()` encapsulates the
complete 8-step bind sequence. Both Titans reach bind_stat=5.

### Gap 2: SYS_MEM_COH_TARGET Inconsistency — RESOLVED (Exp 083)

**Layer:** L7
**Exp:** 082, **083 (resolved)**
**Status:** RESOLVED — answer is **2** (coherent)

Nouveau source (`gm200_flcn_fw_load` in `nvkm/falcon/gm200.c`) and envytools
(`g80_mem_target` in `g80_defs.xml`) both confirm:

| Value | Meaning | Nouveau Enum |
|-------|---------|-------------|
| 0 | VRAM | `NVKM_MEM_TARGET_VRAM` |
| 2 | System memory, coherent (snooped) | `NVKM_MEM_TARGET_HOST` |
| 3 | System memory, non-coherent (no snoop) | `NVKM_MEM_TARGET_NCOH` |

- [x] `strategy_sysmem.rs` uses 2 — **CORRECT**
- [x] `strategy_chain.rs` fixed from 3→2 (B3 fix applied 2026-03-24)

For IOMMU-mapped DMA (our use case), coherent (2) is correct. Non-coherent (3)
would require explicit cache management.

### Gap 3: No Valid mmiotrace Corpus

**Layer:** Data infrastructure
**Exp:** 082
**Status:** DEPRIORITIZED — Exp 086 BAR0 sysfs profiling provides sufficient register data

The mmiotrace corpus remains empty (98 lines of PCI enumeration, zero MMIO writes).
However, Exp 086's BAR0 sysfs mmap profiler captured comprehensive register state
across all driver backends without requiring mmiotrace. The sysfs approach is
non-invasive and works with any bound driver.

**mmiotrace remains useful for:** Capturing the exact write sequence and timing during
driver init (not just final state). This would help with Layer 9 debugging if the
falcon start sequence involves time-sensitive register orchestration.

**Action (if needed):** Integrate trace capture into Ember's driver swap lifecycle
per `CORALREEF_TRACE_INTEGRATION_HANDOFF.md`. Low priority — Exp 086 data sufficient
for current progress.

### Gap 4: FECS Firmware Loading via ACR — MOSTLY SOLVED (Exp 087)

**Layer:** L7-L8
**Exp:** 080, 081, **087 (WPR analysis)**
**Status:** UNDERSTOOD — WPR layout fully mapped, 7 construction bugs identified

GV100 FECS is **ACR-managed**: the Host cannot directly upload IMEM/DMEM and release HRESET. Instead:
1. SEC2 must boot with ACR firmware
2. ACR walks the WPR region to find LS (Light Secure) firmware images
3. ACR loads FECS BL to IMEM at `bl_imem_off`, copies BLD to DMEM
4. BL loads app code at `code_entry_point`, app data from `data_dma_base`
5. ACR releases FECS from HRESET

**Exp 087 resolved all unknowns:**
- [x] WPR layout: `wpr_header_v1[11]` → pad(256) → sub_wpr(0x100) → per-falcon: LSB(240) → img(4K-aligned) → BLD(256)
- [x] LSB = `lsf_signature_v1`(192B) + `lsb_header_tail`(48B)
- [x] BLD = `flcn_bl_dmem_desc_v2`(84B, padded to 256)
- [x] Image = BL_CODE(data section only, NOT full file) + inst + data
- [x] BL file structure: `nvfw_bin_hdr` + `nvfw_hs_bl_desc` + code_section
- [x] start_tag → bl_imem_off → FECS=0x7E00, GPCCS=0x3400
- [x] Firmware versions verified from actual `/lib/firmware/nvidia/gv100/` files

### Gap 5: Cross-Card + Cross-Driver Comparison — COMPLETE (Exp 086)

**Layer:** Data completeness
**Exp:** 070 (partial), **086 (COMPLETE)**
**Status:** COMPLETE — critical insights delivered

**Key findings from Exp 086:**
1. **WPR is NEVER hardware-configured** — inactive in all 12 profiles. This is
   an interface/format problem, not a hardware lock.
2. **nvidia destroys state** — PMC_ENABLE drops to 0x40000020, FECS/GPCCS
   return 0xBADF (PRI timeout). Post-nvidia is the WORST starting state.
3. **nouveau is the Rosetta Stone** — reveals correct SEC2 configuration:
   BOOTVEC=0xFD00, UNK090=0x00070040, DMAIDX=0, DMACTL=0, SCTL=0x7021.
4. **Both Titans are functionally identical** — only 4 registers differ
   (TRACEPC timing + one FECS capability fuse bit).
5. **Post-nouveau is optimal** — all engines remain powered and accessible.

**New bug candidates discovered:** B8 (BOOTVEC), B9 (UNK090 bits), B10
(DMAIDX), B11 (FBIF_624). See Gap 9.

### Gap 9: Layer 8 WPR Construction — SOLVED (Exp 087)

**Layer:** L8
**Exp:** 086 (profiling), **087 (analysis + fix + validation)**
**Status:** **SOLVED** — 7 bugs found, all fixed, hardware validated

B8-B11 from Exp 086 were all false positives (coralReef already handles them
correctly or they are hardware side effects). The REAL problem is in
`wpr.rs:build_wpr()` — the WPR image construction has multiple byte-level errors.

| # | Bug | Severity | Description |
|---|-----|----------|-------------|
| W1 | BL file headers in WPR image | **CRITICAL** | `fw.fecs_bl` (576B) includes 64B of nvfw_bin_hdr+nvfw_hs_bl_desc. Only the 512B code section should be in the image. Shifts all offsets by 64 bytes. |
| W2 | bl_imem_off = 0 | **CRITICAL** | FECS BL start_tag=0x7E → IMEM entry at 0x7E00. GPCCS start_tag=0x34 → 0x3400. Our code hardcodes 0. BL code is position-dependent — wrong address = crash. |
| W3 | bl_code_size includes headers | MEDIUM | Uses 576 (full file) instead of 512 (code only). ACR copies 64 garbage bytes to target IMEM. |
| W4 | BLD DMA offset uses wrong bl_size | MEDIUM | data_dma_base off by 64 bytes. ACR reads FECS data from wrong location. |
| W5 | bl_data_size = 256 vs 84 | MINOR | Should be sizeof(flcn_bl_dmem_desc_v2) = 84. Extra zeros likely harmless. |
| W6 | bin_version = 0 vs 2 | MINOR | WPR header should read version from sig file (version=2). |
| W7 | Depmap corruption | MINOR | Writes to offsets 88-104 in LSB header land in depmap array. depmap_count=0 so likely ignored. |

**W1 + W2 are the root cause of WPR COPY stall.** The ACR firmware receives a
corrupted image with wrong offset metadata, loads BL to IMEM[0] when it expects
IMEM[0x7E00], and the BL code is interleaved with header bytes.

**Fix plan:** See `experiments/087_WPR_FORMAT_ANALYSIS.md` for detailed analysis.

**FIXES APPLIED (2026-03-24):**
- `firmware.rs`: Added `GrBlFirmware` struct — parses `nvfw_bin_hdr` + `nvfw_hs_bl_desc`,
  extracts code section and `start_tag` from `gr/{fecs,gpccs}_bl.bin`.
- `wpr.rs`: All 7 fixes applied — code section in image (W1), `bl_imem_off = start_tag << 8` (W2),
  correct `bl_code_size` (W3), correct BLD DMA offsets (W4), `bl_data_size=84` (W5),
  `bin_version` from sig (W6), clean signature copy (W7).
- `cargo check --package coral-driver` passes clean.

### Gap 6: nvidia_oracle Module — Never Built

**Layer:** Driver coexistence
**Exp:** 082
**Status:** READY TO BUILD

The build script exists (`scripts/build_nvidia_oracle.sh`) and the design is sound. Building and testing it would:
1. Prove the renamed module loads alongside system nvidia
2. Enable nvidia mmiotrace captures without affecting the RTX 5060
3. Open the path to version-indexed recipe collection

**Action:** `sudo ./scripts/build_nvidia_oracle.sh 580.126.18` — requires NVIDIA open kernel source installed.

### Gap 7: Recipe Distillation Pipeline — Untested

**Layer:** Tooling
**Status:** UNVALIDATED

The `distill_oracle_recipe.sh` script references `toadStool/target/release/hw_learn_distill`. This binary may not exist yet. The pipeline from raw mmiotrace → structured recipe JSON is defined but never executed.

**Action:** Check if `hw_learn_distill` exists in toadStool. If not, this is a gap for the toadStool team handoff.

### Gap 8: KNOWN_TARGETS Allowlist Blocks Experimentation

**Layer:** Infrastructure
**Status:** HANDOFF DELIVERED — awaiting coralReef implementation

The `KNOWN_TARGETS` allowlist in `swap.rs` hard-rejects unknown driver names. This
prevents testing custom research drivers, version-indexed nvidia_oracle variants, or
nonsensical combinations (e.g. `amdgpu` on NVIDIA) that are valuable for validation.

**Handoff:** `CORALREEF_OPEN_TARGET_REAGENT_HANDOFF.md` — replaces allowlist with open
acceptance + validation, adds `Personality::Custom`, inverts trace to default-on.

**Architecture:** `specs/UNIVERSAL_DRIVER_REAGENT_ARCHITECTURE.md`

### Gap 10: Layer 9 — FECS/GPCCS Halt Release — ROOT CAUSE FOUND (Exp 091)

**Layer:** L9
**Exp:** 087 (discovered), **088 (partial)**, **090 (cpuctl misleading)**, **091 (BOOTVEC root cause)**
**Status:** **ROOT CAUSE FOUND** — three missing prerequisites identified

After W1-W7 fixes, ACR successfully processes the WPR and bootstraps FECS/GPCCS.
Both falcons reached `cpuctl=0x12` (HRESET + STARTCPU sticky) — ACR loaded them
but left them in HRESET for the host to start.

**Exp 091 Discovery — BOOTVEC is zero:**

The EXCI register format is `[31:16]=cause, [15:0]=PC_at_fault`:
- `exci=0x00070000` → cause=0x0007, faultPC=0x0000
- `exci=0x08070000` → cause=0x0807, faultPC=0x0000

GPCCS faults at PC=0x0000 because **BOOTVEC=0x00000000**. The ACR loaded the
GPCCS bootloader to IMEM[0x3400] (start_tag=0x34), but the mailbox-based
BOOTSTRAP_FALCON command does NOT set BOOTVEC. GPCCS starts executing at
IMEM[0] where there is no valid bootloader → immediate exception.

**Three missing prerequisites (all now fixed in coralReef):**

| # | Missing | Value | Status |
|---|---------|-------|--------|
| 1 | **BOOTVEC** | GPCCS=0x3400, FECS=0x7E00 | Fixed (Exp 091) |
| 2 | **ITFEN** (interface enable) | 0x04 | Fixed (Iter 66) |
| 3 | **INTR_ENABLE** | 0xfc24 | Fixed (Iter 66) |

**Evidence chain:**
- 089b: `GPCCS bootvec=0x00000000` observed before any of our code
- 089b: GPCCS IMEM[0..32] has real code but may be at wrong offset
- 087 W2: GPCCS bl_imem_off=0x3400 from start_tag=0x34
- `falcon_start_cpu()` reads BOOTVEC but never writes it
- `strategy_mailbox.rs` now sets BOOTVEC before STARTCPU

**Diagnostic standard (post-Exp 090):** Always verify PC and EXCI alongside
cpuctl. `FalconProbe` and `AcrBootResult` now include these fields. Success
requires `gpccs_exci == 0 && gpccs_pc != 0`, not just `cpuctl & HRESET == 0`.

### Gap 11: Layer 10 — GPCCS BOOTVEC=0 + CPUCTL Locked (Exp 089b, 091)

**Layer:** L10
**Exp:** 088 (discovered), 089 (initial analysis), **089b (deep root cause)**, **091 (BOOTVEC fix)**
**Status:** ROOT CAUSE IDENTIFIED — BOOTVEC=0 (primary) + CPUCTL write-lock (secondary)

**NOTE (SCTL Myth Busted):** SCTL=0x3000 does NOT block PIO to IMEM/DMEM.
The IMEMC format discovery (Exp 091+) proved PIO works with correct BIT(24)
format regardless of security mode. CPUCTL write-lock is a separate ACR mechanism.

**Exp 089b Deep Findings (register forensics):**

1. **Raw VFIO state** (before ANY init code runs):
   - FECS: `cpuctl=0x10` (STOPPED), PC=0x058f — halted mid-execution, wakes when GR init writes arrive
   - GPCCS: `cpuctl=0x12` (STOPPED + STARTCPU), PC=0x000, `exci=0x00070000` — ACR issued STARTCPU but GPCCS faulted at PC=0
   - SEC2: `cpuctl=0x00` (RUNNING), PC=0x058f — ACR bootloader in idle loop

2. **GPCCS CPUCTL is ACR-LOCKED** — writes to 0x41A100 are silently dropped. This is an ACR-enforced lock, separate from SCTL. SCTL=0x3000 (LS mode) is fuse-enforced but does not cause CPUCTL lock — the lock comes from ACR's ownership state after BOOTSTRAP_FALCON.

3. **GPCCS has valid firmware** — IMEM reads return real code (`0x001400d0 0x0004fe00...`), DMEM starts with firmware build date "Aug 8 2017 20:06:36". Hardware is fully present: hwcfg=0x20102840 (16KB IMEM, 5KB DMEM).

4. **SEC2 CMDQ is NOT initialized** — CMDQ head/tail registers (0x087a00/a04) are zero. MSGQ (0x087a30/a34) also zero. Full 64KB DMEM scan found NO init message. SEC2 never completed CMDQ setup after FLR/rebind.

5. **PMC GR reset breaks CPUCTL lock** — after resetting GR via PMC bit 12, GPCCS CPUCTL becomes writable (HRESET accepted). But LS mode (`sctl=0x3000`) persists, and STARTCPU still faults at PC=0.

6. **apply_gr_bar0_init wakes FECS** — FECS transitions from STOPPED to RUNNING after GR register writes (sw_nonctx.bin + dynamic init). GPCCS does NOT wake up because it faulted at PC=0 and can't process register changes.

7. **HWCFG is at offset +0x108** (not +0x008 as previously assumed). Previous "hwcfg=0x00000000" for GPCCS was reading the wrong register.

**Root Cause Analysis:**

The GPCCS execution failure is NOT a CMDQ problem — it's an **HS authentication problem**:
- SEC2 ACR loaded firmware into GPCCS IMEM and issued STARTCPU (cpuctl bit 1 set)
- GPCCS tried to execute but hit `exci=0x00070000` immediately at PC=0
- This exception likely means the HS authentication context is invalid or lost
- LS mode prevents the host from bypassing this authentication
- The CMDQ registers being zero is a secondary issue — SEC2 never completed full init

**Paths evaluated:**

**Path A: Preserve GPCCS across nouveau→VFIO transition — CLOSED (Exp 089c)**
- nouveau teardown halts all falcons. FLR was never the cause.

**Path B: Proper SEC2 full boot (fix bind_stat for strategies 1-4) — DEFERRED**
- Unnecessary if BOOTVEC fix resolves GPCCS start on mailbox path.

**Path C: Capture and replay nouveau state — CANCELLED**
- Exp 079/089c prove teardown destroys state.

**Path D: BOOTVEC + ITFEN + INTR_ENABLE fix — ACTIVE (Exp 091)**
- Root cause: mailbox BOOTSTRAP_FALCON loads firmware but doesn't set BOOTVEC
- Fix applied to `strategy_mailbox.rs`: set GPCCS BOOTVEC=0x3400 before STARTCPU
- Combined with Iter 66's ITFEN (0x04) and INTR_ENABLE (0xfc24)
- **Next:** Hardware validation on post-nouveau warm state

**Architectural alignment:** GlowPlug ring/mailbox system (Iter 66) provides
timestamped command tracking for the full boot chain. If BOOTVEC fix works,
L11 methods are already implemented in `fecs_method.rs`.

### Gap 14: Layer 7 — SEC2 DMA + HS Mode — BREAKTHROUGH (Exp 095)

**Layer:** L7 (SEC2 binding + DMA)
**Exp:** 091d-091e (root cause chain), **095 (nouveau cycle + VRAM ACR + sysmem ACR)**
**Status:** **MAJOR PROGRESS** — HS mode achieved via sysmem DMA; VRAM DMA path characterized

**Exp 095 Strategy:** Nouveau cycle (Phase 1) restores VRAM via DEVINIT, then sovereign
ACR boot in VFIO. Four-phase test: VRAM probe → nouveau cycle → SEC2 diagnostics → ACR boot.

**Key Discovery: FBHUB is PRI-dead but PRAMIN writes survive:**
- `0x100C2C = 0xbadf5040` (FBHUB PRI error — hub not accessible)
- PRAMIN sentinel writes verify: `wrote=0xcafedead read=0xcafedead ok=true`
- VRAM content (ACR payload, WPR, page tables) written via PRAMIN persists across SEC2 reset
- **FBHUB degradation corrupts DMA reads from VRAM**, preventing BL signature verification

**Three ACR boot paths tested (Exp 095, 8 iterations):**

| Path | DMA Source | SCTL | HS Mode | ACR State | FECS/GPCCS |
|------|-----------|------|---------|-----------|------------|
| VRAM (Phase 3) | VRAM via FBHUB | 0x3000 | NO | Runs in LS, deaf to commands | HRESET |
| Hybrid (Phase 3 v2) | Sysmem PTEs for code, VRAM for WPR | 0x3000 | NO | Different trace, still LS | HRESET |
| Sysmem (Phase 2.75) | System memory via IOMMU | **0x3002** | **YES** | HS mode, but **TRAPPED** (EXCI=0x201f0000) | HRESET |

**Root Cause Analysis — LS mode in VRAM path:**

The BL's job: DMA 256B of non-secure ACR code → verify HS signature → enter HS mode.
FBHUB corruption during VRAM DMA flips bits in the code image, causing signature
verification failure. The BL falls through to LS mode and runs the ACR in a degraded
state. Evidence:
- Identical BL upload + boot sequence, only DMA source differs
- Sysmem path enters HS mode (code is uncorrupted through IOMMU)
- VRAM path: ACR executes substantial code (TRACEPC reaches 0x4e5a) but never leaves LS mode
- blob_size=0 made NO difference (identical trace) → error is before blob DMA

**Root Cause Analysis — Trap in sysmem path:**

With blob_size=0 applied to `strategy_sysmem.rs`, the ACR should skip the internal
blob DMA that previously caused `EXCI=0x201f0000`. **Not yet tested** — this is the
immediate next step.

**Exp 095 TRACEPC analysis (31-entry circular buffer):**

| Phase | Last Trace Entries | Idle PC | Interpretation |
|-------|-------------------|---------|----------------|
| VRAM | `...0x2cf8 0x2d07 0x05ee 0x1239` | 0x1da6 | Polling loop → error handler → degraded idle |
| Hybrid | `...0x346d 0x35ee 0x4e5a 0x11c6 0x3d98 0x11c6` | 0x1e61 | Different execution path, different idle point |
| Sysmem | (from earlier runs) HS mode then trap at 0x1c21 | N/A | Entered HS but faulted during WPR operations |

**EMEM diagnostic (ACR internal state):**
```
EMEM[32..40]: 0x00042001 0x026c0200 0x01000000 0x00000080
              0x01000080 0x01000080 0x01000100 0xa5a51f00
```
`0x00042001` at EMEM[32] is likely an ACR status/error code. `0xa5a51f00` contains
partial sentinel (0xa5a5). This is the ACR's internal error reporting region.

**Code changes applied (Exp 095):**
- `strategy_sysmem.rs`: Added `blob_size=0` patch after `patch_acr_desc` — skips ACR's
  internal blob DMA transfer (WPR already pre-populated in DMA buffer)
- `sysmem_iova.rs`: Separated SHADOW (0x60000) from WPR (0x70000) for proper ACR verification
- `instance_block.rs`: Made `FALCON_PT0_VRAM` and `encode_sysmem_pte` public for hybrid page tables
- `dma.rs`: Made `DmaBuffer::new` public for test-level DMA allocation
- `mod.rs`: Added `dma_backend()` accessor to `NvVfioComputeDevice`
- `falcon.rs` test: Hybrid page table support (sysmem PTEs for ACR code pages, VRAM for rest)

**Exp 096 Results (unified diagnostics rerun):**

**Path J CONFIRMED — HS mode reproducible (no trap with blob_size=0):**

| Run | Strategy | SCTL | HS | TRACEPC | CMDQ | DMEM | EMEM |
|-----|----------|------|----|---------|------|------|------|
| 1 (cold) | All 12 strategies | 0x3000 | NO | BL traces (0xfd75...) | h=0 t=0x30 PENDING | readable | minimal |
| 2 (nouveau+sysmem) | sysmem ACR | **0x3002** | **YES** | all 0x0500 (HS loop) | h=0 t=0 EMPTY | **0xDEAD5EC2 (LOCKED)** | 0x00230406 + structure |
| 2 (nouveau+physical) | phys sysmem | **0x3002** | **YES** | all 0x0500 | h=0 t=0 EMPTY | LOCKED | same |
| 3 (conversation) | All 12 | 0x3000 | NO | BL traces | EMPTY | readable | minimal |

**Critical Exp 096 discoveries:**

1. **DMEM is completely locked in HS mode** — returns `0xDEAD5EC2` for every address. This is why
   `Sec2Queues::discover()` always fails: the init message (CMDQ/MSGQ layout) is in DMEM but inaccessible.
2. **EMEM remains readable in HS mode** — EMEM[0x000]=0x00230406, structured data at offsets 0x080-0x0A0.
   This is the only diagnostic window into ACR's internal state after HS. EMEM may contain queue addresses.
3. **SEC2 is STOPPED after HS boot** — cpuctl=0x00000010 (STOPPED flag set). Mailbox writes, IRQ signals,
   and SWGEN triggers all fail to move PC. The ACR completed initialization and halted, waiting for the
   host to initiate conversation through a mechanism we haven't discovered yet.
4. **Nouveau cycle is mandatory** — without it, SEC2 never enters HS (always 0x3000). Nouveau's DEVINIT
   loads the signed firmware that our BL chain uses.
5. **CMDQ state differs with nouveau** — cold boot shows CMDQ tail=0x30 PENDING (stale from ROM);
   after nouveau+HS boot, CMDQ is 0/0 EMPTY.
6. **VFIO IRQ capability**: INTX=1(flags=0x7), MSI=1(flags=0x9), MSI-X=0(flags=0x9).
   MSI available but not yet wired — may be needed to wake SEC2 from STOPPED state.

**EMEM diagnostic structure (ACR internal state after HS):**
```
EMEM[0x000]: 0x00230406          — ACR status/version?
EMEM[0x080]: 0x00042001          — ACR config word
EMEM[0x084]: 0x026c0200          — size/offset?
EMEM[0x088]: 0x01000000          — flag/count
EMEM[0x08c]: 0x00000080          — alignment/size
EMEM[0x090]: 0x01000080          — region descriptor?
EMEM[0x094]: 0x01000080          — region descriptor?
EMEM[0x098]: 0x01000100          — region descriptor?
EMEM[0x09c]: 0xa5a51f00          — partial sentinel (0xa5a5)
EMEM[0x0a0]: 0x00000406          — matches low bytes of EMEM[0]
```

**Remaining paths (updated post-Exp 096):**

| Path | Approach | Status |
|------|----------|--------|
| **J** | Sysmem ACR + blob_size=0 | **CONFIRMED** — HS mode reproducible. ACR runs, halts. |
| **L** | EMEM-based queue discovery (scan EMEM for CMDQ/MSGQ layout) | **NEW — IMMEDIATE NEXT** |
| **M** | MSI IRQ wiring (eventfd → SEC2 interrupt → wake from STOPPED) | **NEW — HIGH PRIORITY** |
| **N** | Pre-populate CMDQ/MSGQ in EMEM before boot (known layout from nouveau source) | NEW — fallback |
| **K** | Sysmem code + VRAM WPR | DEFERRED — HS works without VRAM |
| **F** | Warm handoff from nouveau | VIABLE but unnecessary |

### Gap 12: Layer 11 — GR Context Init + Shader Dispatch

**Layer:** L11
**Exp:** 088 (discovered)
**Status:** BLOCKED by Gap 11 — requires FECS+GPCCS cooperation

Once GPCCS is running, the GR context init path opens:
1. FECS method interface (`0x409500-0x409804`) responds to commands
2. `DISCOVER_IMAGE_SIZE` returns context buffer size
3. Golden context generation via `WFI_GOLDEN_SAVE`
4. Channel context binding via `BIND_POINTER`
5. GPFIFO dispatch becomes possible

**Research leads (for after Gap 11 is solved):**
1. nouveau `gf100_gr_init()` → golden context image generation
2. `GR_FECS_METHOD` commands → how nouveau sends init commands to FECS
3. nouveau `gf100_grctx_generate()` → how the golden context is built
4. Channel context binding → how a channel's GR context is loaded via FECS

### Gap 15: Layer 10 — SEC2 Conversation After HS (Exp 096→097)

**Layer:** L10
**Exp:** 095 (HS achieved), 096 (conversation characterization), **097 (EMEM discovery)**
**Status:** **ACTIVE** — Init message found, SEC2 HRESET is the blocker

#### Exp 097 BREAKTHROUGH: Init Message Found in EMEM

Full EMEM scan discovered the SEC2 init message at **EMEM offset 0x0080**:

```
w0=0x00042001  unit_id=0x01 size=32
w1=0x026c0200  msg_type=0x00 num_queues=2 os_debug=0x026c
queue0 (CMDQ): offset=0x01000000 size=128  id=0
queue1 (MSGQ): offset=0x01000080 size=128  id=1
```

Queue offsets are **falcon virtual addresses** (DMEM mapped at VA 0x01000000):
- **CMDQ ring** → DMEM offset 0x0000, 128 bytes
- **MSGQ ring** → DMEM offset 0x0080, 128 bytes

#### What Exp 097 Ruled Out

| Test | Result | Implication |
|------|--------|-------------|
| STARTCPU on HRESET SEC2 | cpuctl stays 0x10 | HS security blocks host CPUCTL writes |
| CPUCTL_ALIAS | Same — no effect | Both paths locked |
| MSI arm + 100ms wait | 0 fires | No spontaneous IRQs from HS SEC2 |
| IRQSSET=0x40 poke | IRQSTAT unchanged (0x10) | Host can't set falcon IRQ bits in HS |
| IRQMSET=0x40 | IRQMASK unchanged (0x00) | Host can't modify IRQ mask in HS |
| CMDQ head write + poke | Head writable (readback=0x10) but no response | Ring registers writable but SEC2 CPU dead |

EMEM lockdown sentinel: **0xDEAD5ED0** (distinct from DMEM's 0xDEAD5EC2). Starts at ~0x2000.

#### Root Cause: `blob_size=0` Causes Early Exit

The `blob_size=0` optimization (Exp 095) tells the ACR firmware to skip blob DMA.
Without a blob to process, the firmware completes its WPR setup, writes the init
message, and then **exits/halts** instead of entering the CMDQ idle loop. Normal flow:

```
1. Enter HS   2. Write init message to DMEM   3. DMA ACR blob
4. Bootstrap FECS/GPCCS   5. Initialize CMDQ/MSGQ rings   6. Enter idle loop (RUNNING)
```

With blob_size=0, firmware exits after step 2, skipping 3-6. SEC2 enters HRESET.

#### Data Resolved (from Exp 097)
- [x] EMEM contains init message at offset 0x80 — queue layout known
- [x] CMDQ at DMEM[0x0000], MSGQ at DMEM[0x0080], both 128 bytes
- [x] CPUCTL STARTCPU does NOT resume HRESET falcon in HS mode — blocked by security
- [x] EMEM[0x080-0x0A0] IS the nv_sec2_init_msg format
- [x] MSI/IRQ subsystem locked in HS mode — host has no IRQ control

#### Exp 098: Full Init (Path O) — DMA Trap During WPR→Falcon Copy

Clean boot (nouveau cycle → fresh LS → full-init) achieved:
- **bind_stat→5 OK** (66µs) — binding works from clean LS state
- **HS mode** — SCTL=0x3002
- **WPR copy started** — FECS=1, GPCCS=1 (1=initiated, not 0xFF=done)
- **DMA TRAP** — EXCI=0x201F0000 during falcon image copy

Volta SEC2 is a **one-shot ACR loader** (not a CMDQ daemon). It processes WPR
and exits. The CMDQ idle loop exists in Turing+ but NOT on GV100.

`patch_acr_desc` sets blob_base=0x70000 (WPR IOVA), blob_size=0xCD00. The blob
DMA itself succeeds (firmware reads WPR). The failure is during the subsequent
write of authenticated images to FECS/GPCCS.

**Investigation paths:**

| Path | Approach | Rationale |
|------|----------|-----------|
| **Q** | Investigate SEC2→FECS/GPCCS DMA fault | Check IOMMU fault log, extend page tables, verify target falcon IMEM accessibility |
| **R** | Use nouveau's post-ACR state directly | Nouveau runs full ACR during bind. After swap back, FECS/GPCCS IMEM may already contain authenticated firmware — skip our ACR |
| **S** | Map larger VA space in falcon page tables | PT covers 0..2MiB. ACR may need higher addresses for scratch/FECS/GPCCS writes |

## Priority Order (Post-Exp 104)

**Critical breakthrough: PDE slot fix (Exp 104) unlocked full ACR firmware execution. Firmware is alive and running in its idle loop. HS authentication is the final gate.**

**Path A (Fix HS Authentication — PRIMARY):**
1. **Gap 15 Path T** — Investigate why HS authentication fails with correct page tables.
   VRAM PTEs may cause the BL to load code from VRAM mirror instead of sysmem original.
   Test with all SYS_MEM PTEs (no VRAM aperture). If HS succeeds, the issue is
   VRAM mirror data fidelity.
2. **Gap 15 Path U** — WPR2 indexed register mismatch. Firmware reads WPR boundaries
   from 0x100CD4 (indexed), but our writes to 0x100CEC/CF0 are invisible there.
   May need to set WPR2 via the indexed register or accept that WPR2 HW check
   is unavoidable without PMU.
3. **If HS achieved:** ACR should complete blob DMA → bootstrap FECS/GPCCS → L10-L11.

**Key findings (Exp 099-104):**
- **FLR wipes all falcon memory** — Path R dead, ACR boot is necessary (Exp 099)
- **IOMMU coverage complete** — LOW/HIGH/MID catch-all buffers eliminate all IO_PAGE_FAULT (Exp 100)
- **PDE slot position is the root cause** — GV100 MMU v2 16-byte PDEs use UPPER 8 bytes for directory pointer. Sysmem strategy was writing to LOWER 8 bytes (Exp 104)
- **With correct PDEs: firmware alive** — 31 unique PCs, EMEM queues initialized, DMEM intact, CPU in idle loop (Exp 104)
- **0xDEAD5EC2 is HS read protection** — not a DMEM wipe. BAR0 returns this sentinel for DMEM in HS mode
- **FBIF/DMA config invariant** — HS transition does NOT modify FBIF, ITFEN, or DMACTL (Exp 102)
- **No GPU MMU faults** — FBHUB clean throughout all experiments (Exp 103)

**Resolved:**
- **Gap 15 Path R** (post-nouveau state) — **DEAD** (Exp 099). FLR wipes all falcon memory.
- **Gap 15 Path Q** (DMA fault) — **ROOT CAUSE FOUND** (Exp 104). PDE slot position.
- **Gap 14 Path J** (HS mode via sysmem) — **CONFIRMED** (Exp 096). Reproducible HS mode.
- **Gap 9** (L8 WPR W1-W7) — **SOLVED** (Exp 087).
- **Gap 1** (bind_stat B1-B7) — **SOLVED** (Exp 085).
- **Gap 5** (cross-driver profile) — **COMPLETE** (Exp 086).
- **Gap 2** (SYS_MEM_COH_TARGET) — **RESOLVED** (Exp 083).
- **Gap 4** (ACR/WPR format) — **SOLVED** (Exp 087).
- **Gap 8** (open targets) — handoff delivered.
- **SCTL blocks PIO** — **MYTH BUSTED** (Exp 091+). FalconCapabilityProbe added.
- **PDE format** — **FIXED** (Exp 104). Upper 8 bytes of 16-byte entry for all directory levels.

**Lower priority:**
- **Gap 3** (mmiotrace corpus) — useful for DMA timing, not critical
- **Gap 6** (nvidia_oracle) — nvidia is destructive (Exp 086)
- **Gap 7** (distillation) — depends on toadStool

## What hotSpring Can Do Now (No coralReef Dependency)

1. ~~Research SYS_MEM_COH_TARGET from nouveau source~~ — **DONE** (Exp 083)
2. ~~Apply B1-B4 fixes to coralReef~~ — **DONE** (2026-03-24)
3. ~~Hardware test B1-B4~~ — **DONE** (Exp 084): register writes validated, bind_stat still blocked
4. ~~Cross-driver source analysis~~ — **DONE** (Exp 085): B5-B7 trigger writes discovered from nouveau
5. ~~Run Exp 086 profiling campaign~~ — **DONE** (2026-03-24)
6. ~~Analyze Exp 086 results~~ — **DONE**: B8-B11 all false positives
7. ~~Exp 087 WPR format analysis~~ — **DONE**: root cause = W1 (headers in image) + W2 (bl_imem_off=0)
8. ~~Apply W1-W7 fixes to coralReef~~ — **DONE**: `firmware.rs` + `wpr.rs` (2026-03-24)
9. ~~Validate ACR boot~~ — **DONE** (Exp 087): FECS/GPCCS reach 0x12 (HALTED+ALIAS_EN). L8 SOLVED.
10. ~~Investigate L9~~ — **DONE** (Exp 088): post-ACR STARTCPU transitions falcons to RUNNING.
11. ~~Investigate L10 queue protocol~~ — **DONE** (Exp 095+096): `Sec2Queues`, `probe_and_bootstrap` implemented. DMEM locked in HS.
12. ~~Unified diagnostics~~ — **DONE** (Exp 096): `sec2_exit_diagnostics()` + `sec2_tracepc_dump()` across all 13 exits.
13. ~~Gap 15 Path L~~ — **DONE** (Exp 097): Init message at EMEM[0x80]. CMDQ=DMEM[0x0000]/128B, MSGQ=DMEM[0x0080]/128B.
14. ~~Gap 15 Path M~~ — **DONE** (Exp 097): MSI zero fires. STARTCPU blocked by HS. IRQSSET/IRQMSET locked.
15. ~~Gap 15 Path O~~ — **DONE** (Exp 098): Full init works, DMA traps on WPR→falcon copy. Volta SEC2 = one-shot loader.
16. ~~Gap 15 Path R~~ — **DEAD** (Exp 099): FLR wipes all falcon memory. ACR boot required.
17. ~~Gap 15 Path Q~~ — **ROOT CAUSE FOUND** (Exp 104): PDE slot position (upper 8 bytes of 16-byte entry). Fixed in `strategy_sysmem.rs`.
18. ~~Exp 100-103~~ — **DONE**: IOMMU coverage, VRAM page tables, DMEM data loading, memory controller diagnostics. All ruled out as root cause.
19. ~~Exp 104 PDE fix~~ — **BREAKTHROUGH** (2026-03-25): Firmware alive. 31 trace PCs, EMEM queues initialized, CPU at idle loop. HS authentication pending.
20. **Gap 15 Path T** — Test HS authentication with all SYS_MEM PTEs (no VRAM aperture). Determine if VRAM mirror causes signature mismatch.
21. **Gap 15 Path U** — WPR2 indexed register (0x100CD4) mismatch. Firmware may check different register than we write.
22. **Implement Ember Ring** — `EmberRing` trait + SEC2 CMDQ implementation (queue layout known, but Volta has no CMDQ daemon).
23. Build nvidia_oracle.ko (Gap 6) — low priority

### Gap 13: Swap Safety — D-State Hang on Incomplete Bind (Exp 089c)

**Layer:** Infrastructure / GlowPlug safety
**Exp:** 089c
**Status:** CRITICAL DEBT — caused forced power-off

During Exp 089c, a manual swap script unbound Titan #2 from vfio-pci, attempted
to bind to nouveau (which may have failed), and left the device unbound with
`driver_override` sysfs in a hung state. Subsequent BAR0 access or kernel
shutdown attempts caused a PCI D-state hang requiring forced power-off.

**Root cause:** No atomic rollback on failed bind. Device left in limbo.

**Required fix (GlowPlug/Ember):**
1. **Atomic swap:** If target driver bind fails, immediately rebind to previous driver
2. **Timeout watchdog:** If bind doesn't complete within N seconds, rollback
3. **Pre-swap validation:** Check if target module is loaded before attempting bind
4. **sysfs lock guard:** Never leave `driver_override` set without completing the bind
5. **D-state detection:** Monitor `power/runtime_status` during swap, abort if transitioning to D3

**Handoff:** Add to coralReef GlowPlug evolution — this is a safety-critical path.

## Handoffs Delivered to coralReef

| Handoff | Gaps Addressed | Status |
|---------|---------------|--------|
| `CORALREEF_TRACE_INTEGRATION_HANDOFF.md` | Gap 3 (native trace), Gap 6 (nvidia_oracle target) | Delivered |
| `CORALREEF_OPEN_TARGET_REAGENT_HANDOFF.md` | Gap 8 (open targets), trace-as-default | Delivered |

## Architecture Specs (hotSpring-owned)

| Spec | Scope |
|------|-------|
| `DRIVER_AS_SOFTWARE.md` | Swap-capture-return cycle, recipe distillation |
| `UNIVERSAL_DRIVER_REAGENT_ARCHITECTURE.md` | Open targets, reagent safety, trace foundation, **Ember Ring Architecture** |
| `NATIVE_COMPUTE_ROADMAP.md` | Late-stage: borrow compute from gaming GPUs |
