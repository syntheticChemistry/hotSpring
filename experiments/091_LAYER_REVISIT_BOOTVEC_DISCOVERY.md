# Exp 091: Full Layer Revisit — BOOTVEC Discovery + SCTL Myth Busted

**Date:** 2026-03-23
**Status:** ACTIVE
**Depends:** Exp 085-090, coralReef Iter 66 (ring/mailbox/ITFEN)
**Goal:** Systematically revisit all layers, identify hidden timing/ordering bugs, test GPCCS BOOTVEC hypothesis

**CRITICAL UPDATE (SCTL Debt Sprint):** IMEMC format uses BIT(24) for auto-increment,
not BIT(6). SCTL=0x3000 does NOT block PIO. All previous "SCTL blocks PIO" conclusions
were caused by wrong IMEMC control word in manual `coralctl` commands. The Rust code
was always correct. `FalconCapabilityProbe` added to discover PIO format at runtime.

## Layer-by-Layer Audit

### L1-L3: VFIO Binding + BAR0 Access + PMC Enable — SOLID

No timing issues found. These are deterministic hardware setup steps:
- VFIO device open + BAR0 mmap
- PMC_ENABLE (0x200) write to power engines
- PMC_DEVICE_ENABLE (0x600) for engine enumeration

**No action needed.**

### L4: PFIFO Init + PBDMA Discovery — SOLID (with caveats)

Key timing dependencies discovered in audit:
- PMC bit 8 for PFIFO reset (not bit 1 — fixed in 077)
- Preempt ACK via BIT30 is the truth signal, not `PFIFO_ENABLE` readback
- Doorbell must follow GP_PUT (not precede it)
- Empty vs non-empty runlist BIT30 behavior differs (timing canary)

**Hidden pattern:** The runlist interrupt completion (BIT30) behaves differently
depending on whether the GR engine context is valid. When FECS/GPCCS are dead,
the scheduler can't process GR-bound channels — this connects L4 to L10.

### L5: MMU Fault Buffer Setup — SOLID

Fault buffer ordering is deterministic: configure AFTER BAR2, BEFORE page tables.
`FAULT_BUF0_PUT = 0x8000_0000` enable bit is required.

**Hidden pattern:** Fault buffer entries decode engine_id=0 as GR. When GPCCS is
faulted and GR submits fail, the fault buffer will show GR-related MMU faults
that look like page table errors but are actually GR engine stalls. Don't
confuse these with real MMU bugs.

### L6: PBDMA Context Load + GP_PUT DMA — PARTIAL

**Current state:** PBDMA loads context (SIGNATURE=0xFACE, GP_BASE set), but
GP_PUT DMA read fails. 8 configurations all fault (PBDMA_FAULTED|ENG_FAULTED).

**New insight (from L10 connection):**
The GP_PUT DMA failure may be a downstream effect of GR engine being dead.
The PBDMA→GR engine path requires:
1. Valid FECS/GPCCS (GR context switching firmware)
2. GR engine context bound via FECS method (0x409500/0x409504)
3. Golden context loaded

Without GPCCS alive, the GR engine can't accept work → PBDMA→GR submissions
fault with ENG_FAULTED. **L6 may self-resolve when L10 is fixed.**

**Verify after L10:** Run PBDMA submit with GPCCS alive. If GP_PUT advances
and engine accepts work, L6 is solved by L10.

### L7: SEC2 Binding (B1-B7) — SOLID

bind_stat=5 in ~1µs on both Titans. All trigger writes correct.
No timing issues.

### L8: WPR Construction (W1-W7) — SOLID

All 7 bugs fixed. ACR processes WPR, bootstraps FECS+GPCCS to cpuctl=0x12.
Iter 66 further improved: `GrBlFirmware` struct properly parses code section
+ start_tag. `bl_imem_off = start_tag << 8` confirmed.

**Critical data point from W2 fix:**
- FECS BL: start_tag=0x7E → `bl_imem_off=0x7E00`
- GPCCS BL: start_tag=0x34 → `bl_imem_off=0x3400`

These are the IMEM addresses where ACR places the bootloader code.
The falcon BOOTVEC register MUST match these values for correct execution.

### L9: FECS Start + GR Register Init — PARTIAL → DIAGNOSIS REFINED

**Current state:** FECS reaches idle loop (PC=0x023c), but can't complete init
because GPCCS is dead. All FECS methods (0x409800) timeout.

**Iter 66 additions that may help:**
1. INTR_ENABLE (0x00c) = 0xfc24 — FECS needs interrupts for init loop
2. ITFEN (0x048) = 0x04 — enables DMA/external interfaces
3. Clock-gating restore (0x260=1)
4. FECS exception config (0x409c24 = 0x000e0002)

**Key question:** Does FECS idle at PC=0x023c because GPCCS is dead, or because
FECS itself lacks proper initialization? Iter 66's ITFEN/INTR_ENABLE may
fix FECS independent of GPCCS. Test by checking if 0x409800 bit 0 fires.

### L10: GPCCS Bootstrap — CRITICAL BLOCKER → ROOT CAUSE FOUND

## BOOTVEC Discovery — The Smoking Gun

**EXCI register format:** `[31:16]=cause, [15:0]=PC_at_fault`

Decoding the observed exceptions:
```
exci=0x00070000  →  cause=0x0007, faultPC=0x0000
exci=0x08070000  →  cause=0x0807, faultPC=0x0000
```

Both fault at **PC=0x0000**. GPCCS tries to execute at IMEM[0], not IMEM[0x3400].

**Why?** From Exp 089b, the FalconProbe captured:
```
GPCCS: cpuctl=0x12  pc=0x0000  exci=0x00070000  bootvec=0x00000000
```

**BOOTVEC IS ZERO.** The ACR loaded GPCCS bootloader to IMEM[0x3400] (from WPR
start_tag=0x34), but BOOTVEC was never set to 0x3400. When STARTCPU fires,
GPCCS jumps to IMEM[0] — where there is no valid bootloader code — and
immediately faults with exception cause 0x0007.

**Three missing prerequisites identified (all fixable):**

| # | Missing | What it does | Status |
|---|---------|-------------|--------|
| 1 | **BOOTVEC = 0x3400** | Points GPCCS to correct IMEM entry | **NOT SET** (zero) |
| 2 | **ITFEN = 0x04** | Enables falcon DMA/external interface | Added in Iter 66 |
| 3 | **INTR_ENABLE = 0xfc24** | Enables falcon interrupt handling | Added in Iter 66 |

**Why ACR doesn't set BOOTVEC:**
Our mailbox-based BOOTSTRAP_FALCON command (`mb0=1, mb1=falcon_id`) is a
simplified protocol. The full nouveau path uses CMDQ ring with structured
command payloads that include BOOTVEC setup. The mailbox shortcut loads
firmware into IMEM but skips BOOTVEC configuration.

**The fix:** Before calling `falcon_start_cpu(GPCCS)`, write:
```
GPCCS_BASE + 0x104 = 0x3400  // BOOTVEC from gpccs_bl.start_tag << 8
```

Combined with Iter 66's ITFEN + INTR_ENABLE, this completes the three-piece
GPCCS start sequence.

**Evidence chain:**
1. 089b: BOOTVEC=0x00000000 observed on GPCCS
2. 089b: GPCCS IMEM[0..32] has real code ("0x001400d0 0x0004fe00")
3. 087 W2 fix: GPCCS bl_imem_off=0x3400, FECS bl_imem_off=0x7E00
4. Iter 66 firmware.rs: `gpccs_bl.bl_imem_off()` returns `start_tag << 8`
5. `falcon_start_cpu()` reads BOOTVEC but never writes it
6. `strategy_mailbox.rs` L9 sequence doesn't set BOOTVEC before STARTCPU

**Secondary: FECS BOOTVEC**
Same issue may apply to FECS (bl_imem_off=0x7E00). But FECS reaches PC=0x023c
(idle loop), so its BOOTVEC might have been set correctly by ACR, or FECS
bootloader at IMEM[0] happens to be valid. Need to verify.

### L11: GR Context Init — BLOCKED by L10

Once GPCCS starts executing (PC advances past 0), the path opens:
- `fecs_discover_image_size()` (method 0x10) — confirms FECS responsive
- `fecs_bind_pointer()` (method 0x03) — binds golden context
- `fecs_wfi_golden_save()` (method 0x09) — captures golden context
- Then L6 GP_PUT should resolve (GR engine can accept work)

**Iter 66 has `fecs_method.rs` ready** — full method interface implemented.

## Experiment Plan

### Phase 1: BOOTVEC Verification (diagnostic only)

Read GPCCS and FECS BOOTVEC at every stage of the boot:
1. Raw VFIO state (before any init)
2. After apply_gr_bar0_init
3. After ACR BOOTSTRAP_FALCON(GPCCS) via mailbox
4. Immediately before falcon_start_cpu

Read GPCCS IMEM at both 0x0000 and 0x3400 to verify firmware placement.

### Phase 2: BOOTVEC Fix + Three-Piece Start

Apply the fix in `strategy_mailbox.rs`:
```
// Before STARTCPU(GPCCS):
bar0.write_u32(GPCCS_BASE + BOOTVEC, 0x3400);  // from gpccs_bl.bl_imem_off()
bar0.write_u32(GPCCS_BASE + 0x048, 0x04);       // ITFEN (already in Iter 66)
bar0.write_u32(GPCCS_BASE + 0x00c, 0xfc24);     // INTR_ENABLE (already in Iter 66)
falcon_start_cpu(GPCCS_BASE);
```

Also verify FECS BOOTVEC:
```
// Before STARTCPU(FECS):
bar0.write_u32(FECS_BASE + BOOTVEC, 0x7E00);    // from fecs_bl.bl_imem_off()
bar0.write_u32(FECS_BASE + 0x048, 0x04);         // ITFEN
bar0.write_u32(FECS_BASE + 0x00c, 0xfc24);       // INTR_ENABLE
falcon_start_cpu(FECS_BASE);
```

### Phase 3: Verify Execution

After STARTCPU, sample:
- PC over 5 readings (10ms apart) — should advance beyond 0
- EXCI should be 0x00000000 (no exception)
- FECS method 0x10 (discover_image_size) — should respond

### Phase 4: L6 Revalidation

If GPCCS is alive, re-run PBDMA submit test. Check if GP_PUT DMA advances
and GR engine accepts work.

### Phase 5: L11 Gateway

If FECS methods respond:
- fecs_discover_image_size → get context buffer size
- fecs_bind_pointer → bind golden context
- fecs_wfi_golden_save → capture golden context
- First shader dispatch

## Ring/Mailbox Infrastructure Usage

Use GlowPlug mailbox to track firmware interaction:
```
coralctl mailbox.create --bdf 0000:65:00.0 --engine fecs --capacity 16
coralctl mailbox.create --bdf 0000:65:00.0 --engine gpccs --capacity 16
coralctl mailbox.create --bdf 0000:65:00.0 --engine sec2 --capacity 8
```

Use ring for command sequencing:
```
coralctl ring.create --bdf 0000:65:00.0 --name gpfifo --capacity 64
```

This provides timestamped command/response tracking across the full boot chain.

## Exp 091 Addendum: SCTL Debt Sprint Results (2026-03-23)

### Direct PIO Boot (Path B — bypass ACR)

**Test:** `vfio_sovereign_gr_boot` with BOOTVEC=0 + ITFEN=0x04 + INTR_ENABLE=0xfc24

**Results (GPU1 0000:03:00.0):**
- PIO upload: **SUCCESS** (SCTL=0x3000 does not block PIO)
- STARTCPU: **EXECUTES** (FECS reached PC=0x02b6 before fault)
- FECS exci=0x03070000 (cause=0x0307 at PC=0x02b6)
- GPCCS exci=0x03070000 (cause=0x0307 at PC=0x0000)

**Diagnosis:** Exception 0x0307 is likely caused by W1 header bytes in IMEM — raw
`fecs_bl.bin` includes 64B nvfw_bin_hdr + nvfw_hs_bl_desc before actual code. The
direct upload path uploads the entire file including headers. The ACR path (Exp 087)
strips these headers. **Next step: strip headers in direct upload path.**

### FBIF Characterization (Cold VFIO State)

**Both GPUs identical in cold VFIO state:**

| Register | GPU1 (03:00.0) | GPU2 (4a:00.0) | Meaning |
|----------|----------------|----------------|---------|
| SEC2 FBIF 0x624 | 0x110 | 0x110 | May need PHYS_VID mode for DMA |
| SEC2 DMACTL | 0x01 | 0x01 | DMEM scrub done |
| SEC2 ITFEN | 0x04 | 0x04 | ACCESS_EN set |
| SEC2 CPUCTL | 0x10 | 0x10 | HRESET/halted |
| SEC2 SCTL | 0x3000 | 0x3000 | LS mode (fuse-enforced) |
| PBUS BAR1_BLOCK | 0x002fff00 | 0x002fff00 | Has pointer |
| PBUS BAR2_BLOCK | 0x20000000 | 0x20000000 | Target=VRAM |
| PMC_ENABLE | 0x5fecdff1 | 0x5fecdff1 | Bit 22 (SEC2) locked on |
| NV_PFB 0x100c2c | 0xbadf5040 | 0xbadf5040 | PRI error — FBHUB not accessible |

**Key finding:** SEC2 FBIF=0x110 in cold state. Previous Exp 091e showed:
- GPU1 FBIF=0x190 → won't accept PHYS_VID writes, DMA fault 0x201f0007 (VIRT)
- GPU2 FBIF=0x91 → accepts PHYS_VID, DMA fault 0x051f0007 (PHYS)

**FBHUB PRI errors** (0xbadf5040) in PFB registers suggest the FBHUB MMU is
not initialized in cold VFIO state. This is the likely root cause of DMA failures:
SEC2 cannot DMA from VRAM because FBHUB doesn't have valid page tables.

**Next steps:**
1. Strip W1 headers from direct PIO upload (enables Path B)
2. Capture nouveau warm FBIF values for comparison (mmiotrace or warm handoff)
3. Try FBIF mode override to PHYS_VID before SEC2 DMA bind
