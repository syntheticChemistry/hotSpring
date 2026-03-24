# GPU Cracking Gap Tracker

**Updated:** 2026-03-24 (Exp 087 — WPR root cause found)
**Goal:** Sovereign compute on Titan V (GV100) — FECS/GPCCS running without proprietary ACR chain

## Layer Model

| Layer | System | Status | Blocker |
|-------|--------|--------|---------|
| L1 | VFIO device binding | SOLVED | — |
| L2 | BAR0/BAR2 access | SOLVED | — |
| L3 | PMC enable + engine enumeration | SOLVED | — |
| L4 | PFIFO init + PBDMA discovery | SOLVED | — |
| L5 | MMU/FBHUB fault buffer setup | PROVEN (Exp 076) | — |
| L6 | PBDMA context load + channel submit | PARTIAL (Exp 058) | GP_PUT DMA read fails |
| L7 | SEC2 falcon binding + DMA | **SOLVED** (Exp 085) | — (B1-B7 all fixed) |
| L8 | WPR/ACR payload + FECS/GPCCS boot | **SOLVED** (Exp 087) | W1-W7 fixed. ACR processes WPR, bootstraps FECS+GPCCS. |
| L9 | FECS/GPCCS full boot (HS mode release) | **BLOCKED** | Falcons reach HALTED+ALIAS_EN (0x12), not RUNNING. mb0=1 return code. |

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

### Gap 10: Layer 9 — FECS/GPCCS Halt Release (NEW)

**Layer:** L9
**Exp:** 087 (discovered)
**Status:** NEW — active frontier

After W1-W7 fixes, ACR successfully processes the WPR and bootstraps FECS/GPCCS.
Both falcons reach `cpuctl=0x12` (HALTED + ALIAS_EN) but do not transition to
RUNNING (0x00). SEC2 mailbox `mb0=1` after BOOTSTRAP_FALCON command.

**Observations:**
- `ALIAS_EN` (bit 1) confirms ACR set HS mode alias — the falcon is in secure mode
- `HALTED` (bit 4) means the falcon stopped execution — BL may have completed
  and is waiting for a host signal, or the BL hit an error
- `mb0=1` could mean "command complete" or "error code 1" — protocol research needed

**Research leads:**
1. nouveau `gm200_gr_init_400()` — post-ACR falcon start sequence
2. `FECS_CTXSW_MAILBOX` registers — how nouveau signals FECS to begin
3. `GR_INIT` boot commands — GR engine initialization after falcon boot
4. nouveau `gv100_gr_init()` → `nvkm_falcon_start()` — the HRESET release sequence
5. Check if GPCCS needs to start before FECS (dependency ordering)

**Solve strategies:**
1. **Nouveau source trace of post-ACR boot** — find what nouveau does between
   ACR bootstrap completion and GR engine ready
2. **Register profiling during nouveau GR init** — extend Exp 086 profiler to
   capture FECS/GPCCS state during nouveau's init window
3. **FECS mailbox protocol** — determine the BL→host handshake from firmware analysis
4. **HRESET release experiment** — try clearing HALTED bit after ACR bootstrap

## Priority Order

1. **Gap 9** (L8 WPR construction W1-W7) — **SOLVED** (Exp 087). ACR processes WPR, FECS/GPCCS reach cpuctl=0x12 (HALTED+ALIAS_EN). New L9 blocker: falcon boot completion.
2. **Gap 1** (bind_stat) — **SOLVED** (Exp 085). B1-B7 all fixed. bind_stat=5 on both Titans.
3. **Gap 5** (cross-driver profile) — **COMPLETE** (Exp 086). Interface problem confirmed, 4 new bug candidates.
4. **Gap 3** (mmiotrace corpus) — still useful for WPR layout, but Exp 086 data may be sufficient
5. **Gap 2** ~~(SYS_MEM_COH_TARGET)~~ — **RESOLVED** (Exp 083). Answer: 2 = coherent.
6. **Gap 4** (ACR/WPR format) — now tractable: interface problem, not key problem (Exp 086)
7. **Gap 8** (open targets) — handoff delivered, unblocks novel driver experiments
8. **Gap 6** (nvidia_oracle) — lower priority now that Exp 086 showed nvidia is destructive
9. **Gap 7** (distillation) — depends on toadStool, lowest priority

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
10. **Investigate L9** — why do FECS/GPCCS halt instead of running? mb0=1 return code research.
10. Build nvidia_oracle.ko (Gap 6)
11. Document findings in experiment journals

## Handoffs Delivered to coralReef

| Handoff | Gaps Addressed | Status |
|---------|---------------|--------|
| `CORALREEF_TRACE_INTEGRATION_HANDOFF.md` | Gap 3 (native trace), Gap 6 (nvidia_oracle target) | Delivered |
| `CORALREEF_OPEN_TARGET_REAGENT_HANDOFF.md` | Gap 8 (open targets), trace-as-default | Delivered |

## Architecture Specs (hotSpring-owned)

| Spec | Scope |
|------|-------|
| `DRIVER_AS_SOFTWARE.md` | Swap-capture-return cycle, recipe distillation |
| `UNIVERSAL_DRIVER_REAGENT_ARCHITECTURE.md` | Open targets, reagent safety, trace foundation |
| `NATIVE_COMPUTE_ROADMAP.md` | Late-stage: borrow compute from gaming GPUs |
