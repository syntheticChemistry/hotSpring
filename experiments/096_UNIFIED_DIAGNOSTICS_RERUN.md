# Exp 096: Unified Diagnostics + Full Conversation Rerun

**Date:** 2026-03-26
**Status:** COMPLETE
**Note:** Shares experiment number 096 with `096_SILICON_SCIENCE_TMU_QCD_MAPPING.md` (strandgate team, different track). Both created independently. No renumber — fossil record preserved.
**Depends on:** Exp 095 (HS mode breakthrough)
**Unlocks:** Gap 15 (SEC2 conversation after HS)

## Objective

Normalize SEC2 exit diagnostics across all 13 strategy exits via shared helpers,
then run three ordered hardware experiments to validate Exp 095's HS mode and
characterize the SEC2 conversation state across all boot paths.

## Code Changes (Part 1)

### 1a. Shared TRACEPC helper — `sec2_tracepc_dump()`

Created in `sec2_hal.rs`. Reads the TRACEPC circular buffer via indexed EXCI/TRACEPC
registers. Returns `(entry_count, entries)`. Replaces ad-hoc TRACEPC reads scattered
across strategy files.

### 1b. Shared exit diagnostics — `sec2_exit_diagnostics()`

Created in `sec2_hal.rs`. Captures in one call:
- SCTL + HS mode decode
- EXCI + PC
- Full TRACEPC dump
- First 64 words of EMEM (non-zero entries only)

### 1c. Wire into `probe_and_bootstrap()`

One-line call at the top of `sec2_queue::probe_and_bootstrap()`. All 13 strategy
exits now get uniform diagnostics through the existing conversation probe.

### 1d. Fix hybrid TRACEPC mislabel

`strategy_hybrid.rs` previously read `[PC, WATCHDOG, 0x038, 0x03C]` and labeled it
"TRACEPC". Replaced with actual `sec2_tracepc_dump()` call.

### Build verification

- `cargo check --features vfio -p coral-driver`: clean (3 pre-existing warnings)
- `cargo test --features vfio`: 4,140 passed, 0 failed, 0 regressions

## Hardware Runs (Part 2)

### Run 1: Full Boot Solver (cold, no nouveau cycle)

**Command:** `vfio_falcon_boot_solver` (all 12 strategies)

| Strategy | SCTL | HS | TRACEPC | CMDQ | Queue Discovery |
|----------|------|----|---------|------|-----------------|
| 1. nouveau-style IMEM+EMEM | 0x3000 | NO | BL traces | h=0 t=0x30 PENDING | failed (no init msg) |
| 2. Physical-first | 0x3000 | NO | BL traces | h=0 t=0x30 PENDING | failed |
| 3. VRAM ACR (PRAMIN) | 0x3000 | NO | BL traces | h=0 t=0x30 PENDING | failed |
| 4. Sysmem ACR (IOMMU) | 0x3000 | NO | BL traces | h=0 t=0x30 PENDING | failed |
| 5. Hybrid ACR | 0x3000 | NO | BL traces | h=0 t=0x30 PENDING | failed |
| 6. Direct FECS boot | 0x3000 | NO | BL traces | h=0 t=0x30 PENDING | failed |
| 7. ACR mailbox | 0x3000 | NO | BL traces | h=0 t=0x30 PENDING | failed |
| 8. HRESET experiments | 0x3000 | NO | BL traces | h=0 t=0x30 PENDING | failed |
| 9. Direct ACR IMEM | 0x3000 | NO | BL traces | h=0 t=0x30 PENDING | failed |
| 10. ACR chain DMA | 0x3000 | NO | BL traces | h=0 t=0x30 PENDING | failed |
| 11. EMEM-based boot | 0x3000 | NO | BL traces | h=0 t=0x30 PENDING | failed |
| 12. Direct IMEM/DMEM | 0x3000 | NO | BL traces | h=0 t=0x30 PENDING | failed |

**Key finding:** Without nouveau cycle, no strategy achieves HS mode. CMDQ shows
stale state (tail=0x30 PENDING) from GPU ROM init.

### Run 2: Exp 095 Path J Validation (nouveau cycle + sysmem ACR)

**Command:** `vfio_clean_vram_acr_boot`

| Phase | SCTL | HS | TRACEPC | DMEM | EMEM | CMDQ |
|-------|------|----|---------|------|------|------|
| Sysmem ACR | **0x3002** | **YES** | all 0x0500 | **0xDEAD5EC2 (LOCKED)** | 0x00230406 + structure | EMPTY |
| Physical sysmem | **0x3002** | **YES** | all 0x0500 | LOCKED | same | EMPTY |
| Firmware interaction | 0x3002 | YES | — | LOCKED | [0]=0x00230406 | EMPTY, not alive |

**Key findings:**
1. **HS mode confirmed reproducible** — both sysmem strategies achieve SCTL=0x3002
2. **DMEM is completely locked** — every address returns 0xDEAD5EC2
3. **EMEM remains readable** — structured data visible at offsets 0x000 and 0x080-0x0A0
4. **SEC2 is STOPPED** (cpuctl=0x00000010) — mailbox/IRQ/SWGEN signals do not wake it
5. **Physical sysmem also works** — SCTL survives PMC reset (0x3002 persists)

### Run 3: Dedicated Conversation + IRQ Probe

**Command:** `sec2_conversation_full_cycle` + `sec2_queue_probe_only`

| Test | Result |
|------|--------|
| Queue probe (2 tests) | All EMPTY (0/0), never initialized |
| Queue discovery (all 12) | All fail: "SEC2 init message not found in DMEM" |
| VFIO IRQ: INTX | count=1 flags=0x00000007 |
| VFIO IRQ: MSI | count=1 flags=0x00000009 (maskable) |
| VFIO IRQ: MSI-X | count=0 flags=0x00000009 |

**Key finding:** MSI is available (1 vector, maskable). This is a viable interrupt
path for SEC2 communication if the falcon can be made to service interrupts.

## Analysis (Part 3)

### Cross-Strategy Comparison

| Dimension | Cold (no nouveau) | Post-nouveau HS |
|-----------|-------------------|-----------------|
| SCTL | 0x3000 (LS) | 0x3002 (HS) |
| TRACEPC | BL execution paths (0xfd75...) | Uniform 0x0500 (HS idle loop) |
| DMEM | Readable (BL desc, ACR config) | LOCKED (0xDEAD5EC2) |
| EMEM | Minimal (write tests only) | ACR internal state visible |
| CMDQ | Stale (tail=0x30 PENDING) | Clean (EMPTY) |
| cpuctl | 0x00000000 (RUNNING) | 0x00000010 (STOPPED) |

### Root Cause: Why Conversation Fails

The SEC2 conversation protocol requires:
1. SEC2 firmware writes CMDQ/MSGQ layout to DMEM as an "init message"
2. Host scans DMEM for this message to discover queue base addresses
3. Host writes commands to CMDQ, SEC2 reads them and responds via MSGQ

After HS boot, step (2) fails because DMEM is HS-locked. The init message
exists but is inaccessible. SEC2 is STOPPED (not RUNNING), so it cannot process
commands even if we found the queue addresses.

### Next Steps → Gap 15

Three investigation paths identified (see gap tracker):
- **Path L:** EMEM-based queue discovery (EMEM is readable in HS)
- **Path M:** MSI IRQ wiring + STARTCPU on STOPPED falcon
- **Path N:** Pre-seed known queue layout before boot

## Files Changed

- `coralReef/crates/coral-driver/src/nv/vfio_compute/acr_boot/sec2_hal.rs` — new `sec2_tracepc_dump()` + `sec2_exit_diagnostics()`
- `coralReef/crates/coral-driver/src/nv/vfio_compute/acr_boot/sec2_queue.rs` — wired `sec2_exit_diagnostics()` into `probe_and_bootstrap()`
- `coralReef/crates/coral-driver/src/nv/vfio_compute/acr_boot/strategy_hybrid.rs` — fixed TRACEPC mislabel
- `coralReef/crates/coral-driver/src/nv/vfio_compute/acr_boot/mod.rs` — exported new functions
