# Experiment 223 — ACR Sovereign Boot Catalyst (Frozen Reagent)

**Date**: 2026-05-25 (updated 2026-05-26)
**Status**: ✅ EVOLVED — toadStool `sovereign.init` achieves compute_ready on both GPUs; Rust exp224 hardened into shared `low_level/falcon.rs` module with unit tests
**Hardware**: 2x NVIDIA Titan V (GV100), BDFs `0000:02:00.0`, `0000:49:00.0`
**Dependency**: Exp 219 (catalyst pattern + Tier 2 PRI wall), Exp 222 (reagent capture pipeline)

## Objective

Extract the minimal, reusable register sequence ("frozen catalyst") from the
nvidia-535 mmiotrace that performs the PMU ACR secure boot chain. This is the
first step toward sovereign ACR→FECS→GPCCS→TPC boot — breaking through the
Tier 2 wall identified in Exp 219 by executing firmware rather than replaying
register state.

## Core Insight

Exp 219 proved that register replay (83K writes) achieves Tier 1 but not Tier 2.
The Tier 2 wall is firmware-mediated PRI ring station creation. To break through,
we must *execute* the GPCCS firmware, which requires the ACR secure boot chain:
PMU/SEC2 → ACR bootloader → FECS → GPCCS → TPC PRI stations.

This experiment isolates and freezes the ACR boot sequence as a reusable catalyst
reagent — taking just the parts needed for the nvidia initialization and leaving
the rest alone.

## Discoveries

### 1. HS Mode Architecture (Falcon v5 / Volta)

The PMU falcon has three HS (Hardware Secured) modes controlled by fuse-enforced
security logic:

| Mode | CPU Execution | Host IMEM/DMEM Access | Entry | Exit |
|------|--------------|----------------------|-------|------|
| HS2 | ✅ (running signed fw) | ❌ (blocked) | Power-on/VBIOS | HS unlock (0x3C0) |
| HS0 | ❌ (HWCFG2 bit 29 = 0) | ✅ (read/write) | HS unlock | Power cycle only |
| HS3 | ✅ (post-driver-load) | ❌ (partial, mailbox only) | ACR boot | — |

**Key finding**: HS mode 0 grants register access but *permanently disables CPU
execution* (HWCFG2 bit 29 clears). This is irreversible without hardware reset
(power cycle or SBR). The unlock is a security trade-off: inspect OR execute,
never both simultaneously.

### 2. HWCFG2 Bit 29 — CPU Operational Flag

Register PMU+0x148 (FALCON_HWCFG2) is read-only and reflects hardware state:

| GPU State | HWCFG2 | Bit 29 | Meaning |
|-----------|--------|--------|---------|
| VBIOS running (HS2) | 0x201F0000 | 1 | CPU operational |
| Post-HS-unlock (HS0) | 0x001F0000 | 0 | CPU disabled |
| Post-SBR + unlock | 0x061F0000 | 0 | CPU disabled, alt config |

Bit 29 cannot be set by host writes — it is controlled by the HS ROM hardware
during secure boot. PMC_ENABLE clock gating does not affect it.

### 3. CPUCTL Behavior in HS Mode 0

| Write | HS2 (pre-unlock) | HS0 (post-unlock, bit29=1 history) | HS0 (post-SBR, no bit29 history) |
|-------|------------------|-----------------------------------|----------------------------------|
| 0x02 (START) | n/a (already running) | Accepted (readback 0x12) but no exec | Rejected (readback 0x10) |
| 0x12 (HS ROM) | Triggers HS ROM | Accepted (readback 0x12) but no exec | Rejected (readback 0x10) |

On a GPU that was previously running (bit 29 was set before unlock), the START
bits are *accepted* into the register but CPU execution does not begin. On a
post-SBR GPU (bit 29 was never set), even the register write is rejected.

### 4. PMC_ENABLE Bit 12 — PMU Engine Clock

Post-SBR, the PMU engine clock (PMC_ENABLE bit 12) is disabled. Enabling it
resets falcon IMEM/DMEM contents but does not change HS mode or bit 29. The
correct initialization order is: enable clock → HS unlock → load firmware.

### 5. IMEM Loading — PIO Write Validation

PMU IMEM on Volta falcon v5 is accessible via PIO (Programmed I/O) in HS mode 0:

| Register | Offset | Purpose |
|----------|--------|---------|
| IMEMC | PMU+0x180 | Control: offset + auto-increment flags |
| IMEMD | PMU+0x184 | Data: 32-bit word read/write |
| IMEMT | PMU+0x188 | Tag: code authentication tag per 256-byte block |

**Auto-increment mode** (IMEMC bit 24): nvidia uses `0x0100FE00` to write
sequentially starting at offset 0xFE00. Tags are set at 64-word boundaries
(256 bytes): tag 0x100 for words 0-63, tag 0x101 for words 64-127.

**Verified**: 128 words (512 bytes) of nvidia's ACR bootloader successfully
loaded into PMU IMEM and verified via readback. All words match.

### 6. DMEM Loading — ACR Descriptor

The 21-word (84-byte) ACR descriptor loaded at DMEM offset 0 contains VRAM
addresses where firmware payloads must be pre-staged:

| Field | Offset | Value | Meaning |
|-------|--------|-------|---------|
| flags | 32 | 0x04 | ACR mode flags |
| ucode_vram_lo | 36 | 0xDD990000 | VRAM address of ACR ucode (low 32) |
| ucode_vram_hi | 40 | 0x00000001 | VRAM address high → 0x1DD990000 |
| ucode_size | 48 | 0x600 | ACR ucode: 1536 bytes |
| payload_size | 56 | 0x6900 | Total payload: 26880 bytes |
| loadtbl_vram_lo | 64 | 0xDD998000 | Load table VRAM addr (low 32) |
| loadtbl_vram_hi | 68 | 0x00000001 | → 0x1DD998000 |
| loadtbl_size | 72 | 0x42F0 | Load table: 17136 bytes |
| loadtbl_entries | 76 | 0x34 | 52 entries |
| falcon_count | 80 | 0x2F | 47 falcons |

### 7. VRAM Staging Gap

The mmiotrace recipe captures only MMIO register writes, not DMA data transfers.
nvidia stages the ACR firmware payload in VRAM via PCIe DMA before the PMU boot
sequence. Zero PRAMIN writes exist before step 32766 (CPUCTL trigger). The
firmware data must be staged via PRAMIN (BAR0+0x700000 window) or extracted
from the nvidia driver binary.

### 8. nvidia's Full PMU Boot Sequence

Extracted from nv535_recipe.json steps 32591-32833 (1203 PMU-domain writes):

```
Phase 1 (steps 32591-32595): HS Unlock
  INTR_MASK = 0xFFFF
  MB2 = 0x104
  UNK_0A4 = 0x02
  HS_CTRL = 1 → 0 (toggle)

Phase 2 (steps 32596-32611): Register Setup
  MB0 = 0 (clear)
  DMACTL = 0x20008
  FBIF config (0x10AA70/74, 0x10AE00-10/24/74)
  INTR_MASK, NTSTATUS, INTR_EN

Phase 3 (steps 32612-32633): DMEM Descriptor
  DMEMC = 0x01000000 (offset 0, auto-increment)
  21 x DMEMD writes (ACR descriptor)

Phase 4 (steps 32634-32764): IMEM Firmware
  IMEMC = 0x0100FE00 (offset 0xFE00, auto-increment)
  IMEMT = 0x100 (tag for block 0)
  64 x IMEMD (first 256 bytes)
  IMEMT = 0x101 (tag for block 1)
  64 x IMEMD (second 256 bytes)

Phase 5 (step 32765): Pre-trigger
  REG_104 = 0x10000

Phase 6 (step 32766): HS ROM Trigger
  CPUCTL = 0x12

Phase 7 (step 32767): Post-trigger acknowledge
  REG_004 = 0x10

Phase 8 (steps 32768-32769): HS ROM status check
  DMEMC reads at 0x1B20 and 0x1BA0

Phase 9 (steps 32770-32833): Full register cleanup + LS boot
  Zero all PMU regs 0x000-0x1F8
  BOOTVEC = 0x00
```

## Pre-Boot Initialization (Steps 0-32590)

The 32,591 steps before the PMU boot configure fundamental GPU infrastructure:

| Domain | Steps | Purpose |
|--------|-------|---------|
| LTC | 18,953 | L2 cache / memory controller init |
| CLK | 6,144 | Clock domain configuration |
| PTOP | 2,060 | Top-level device enumeration |
| PRI_MASTER | 1,543 | Primary Ring Interface setup |
| ROOT_PLL | 1,024 | Root PLL configuration |
| PCLOCK | 1,024 | Clock generation |
| PBUS | 787 | PCIe bus interface |
| PMC | 528 | Power management, engine enables |
| PFB | 256 | Framebuffer controller |
| PFB_NISO | 142 | Non-isochronous FB config |
| FBHUB | 128 | Framebuffer hub |
| FBPA | 2 | Framebuffer partition |

PMC_ENABLE sequence (steps 8196-8200) enables engine clocks:
`0x5FECCEF1 → 0x5FECDFF1 → 0xFFFFFFFF → 0x5FECDFF1 → 0x7FFCDFFD`

### 10. Register Map Corrections (Session 2)

Critical register misidentifications corrected using toadStool `falcon.rs`:

| Register | We Called It | Actual Purpose | Correct Offset |
|----------|-------------|----------------|----------------|
| PMU+0x104 | "pre-trigger reg" | **BOOTVEC** (boot vector) | 0x104 ✓ |
| PMU+0x108 | — | **HWCFG** (IMEM/DMEM sizes, security) | 0x108 |
| PMU+0x10C | "0x10C" | **DMACTL** (DMA control) | 0x10C ✓ |
| PMU+0x130 | (unused) | **CPUCTL_ALIAS** (host CPU control) | 0x130 |
| PMU+0x134 | "DMACTL" | Unknown PMU-specific register | ≠DMACTL |
| PMU+0x148 | "HWCFG2" | **EXCI** (exception info) | 0x148 |
| PMU+0x3C0 | "HS_CTRL" | **ENGCTL** (engine control/reset) | 0x3C0 |

### 11. ENGCTL vs HS Unlock

What we called "HS unlock" is actually a **falcon engine reset** via ENGCTL:
- Writing `1` then `0` to ENGCTL resets the falcon engine
- Side effect: SEC_MODE transitions from HS (2) to NS (0)
- CPU execution is permanently disabled after ENGCTL reset
- CPUCTL_ALIAS reads as 0 (dead) in NS mode
- IMEM/DMEM contents may be wiped
- **Only recoverable via full hardware reset (power cycle)**

### 12. CPUCTL_ALIAS — The Correct Boot Path

toadStool (Exp 206) uses CPUCTL_ALIAS (0x130) for ALL falcon boots:

```
CPUCTL_ALIAS ← 0x10 (HRESET — halt CPU)
... upload IMEM/DMEM via PIO ...
BOOTVEC ← 0x00
CPUCTL_ALIAS ← 0x01 (IINVAL — instruction cache invalidate)
CPUCTL_ALIAS ← 0x02 (STARTCPU — release from halt)
```

PIO (IMEMC/IMEMD/IMEMT) works in ALL security modes per toadStool comment:
> "DOES NOT block host PIO to IMEM/DMEM — PIO works with correct IMEMC
> format (BIT(24) write, BIT(25) read) regardless of security mode."

**No ENGCTL reset needed. No HS unlock. Stay in HS mode 2.**

### 13. DMACTL and CPUCTL Acceptance

On a fresh HS2 GPU after ENGCTL reset to NS mode:
- Writing DMACTL-at-0x134 = 0x20008 → CPUCTL writes REJECTED (readback 0x10)
- Restoring DMACTL-at-0x134 = 0x3FFFE → CPUCTL writes ACCEPTED (readback 0x12)
- The register at PMU+0x134 controls CPUCTL write acceptance

### 14. SEC_MODE Field Correction

SCTL register bits [1:0] = SEC_MODE (not bits [3:0]):
- SCTL=0x3002: SEC_MODE=2 (HS) — fresh VBIOS GPU
- SCTL=0x3000: SEC_MODE=0 (NS) — after ENGCTL reset
- We incorrectly used `sctl & 0xF` and called it "HS mode 0/2"

## Current State (2026-05-26)

**Post-reboot outcome (session 3):** Rust exp224 confirmed that direct host PIO
to PMU is blocked in HS mode 2 (VBIOS-initialized state). CPUCTL_ALIAS is
unresponsive. The PMU is a firmware fortress — host cannot directly load or
start code on it from a cold VBIOS state.

**Resolution:** toadStool's `sovereign.init` RPC (which uses the correct
multi-stage boot path: Boot Falcon → SEC2 → ACR → PMU) successfully brought
both GPUs to `compute_ready: true` with full DEVINIT (PMC_ENABLE=0x5fecdff1).
A WGSL compute shader was dispatched via `compute.dispatch` to confirm
functional Tier 2 compute.

**Infrastructure hardening (session 4):** Extracted shared falcon register map,
FalconSnapshot, PIO helpers, engine bases, and Bar0Domain presets from exp224
into `barracuda/src/low_level/falcon.rs`. Hardened `bar0.rs` with alignment
checks, `r32_checked` dead-link sentinel, BDF-based open, and ENGCTL deny-list
in SafeBar0. Exported `low_level` module from crate lib behind `low-level`
feature gate. 16 unit tests (7 falcon + 9 bar0). exp224 rewired to consume
shared modules.

### 9. Open-Source vs Proprietary Firmware

The open-source firmware (`/lib/firmware/nvidia/gv100/acr/`) is structurally
different from nvidia-535's proprietary firmware:

| File | Size | First Word | Notes |
|------|------|-----------|-------|
| `acr/bl.bin` (open) | 1280B | 0x000010DE | 256B header + 256B code + 768B data |
| mmiotrace ACR BL | 512B | 0x00A000D0 | Raw falcon instructions, no header |
| `acr/ucode_load.bin` (open) | 18688B | 0x000010DE | 256B header + 256B code + 18176B data |

The open-source `bl.bin` has a header (magic 0x10DE, code offset 0x100, data
offset 0x200) while the mmiotrace firmware is raw IMEM code. They are NOT
interchangeable — different signing keys, different ACR descriptor formats.

**Implication**: The catalyst must use the mmiotrace-extracted firmware (nvidia-535)
or implement a nouveau-style ACR boot with the open-source firmware. These are
separate paths requiring different ACR descriptors and VRAM staging.

## Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| **Rust catalyst (active)** | `barracuda/src/bin/exp224_pmu_acr_catalyst.rs` | Idiomatic Rust, CPUCTL_ALIAS path, corrected register map |
| ACR boot catalyst (SUPERSEDED) | `infra/catalysts/reagents/acr_sovereign_boot.py` | v1 Python, FLAWED HS unlock via ENGCTL |
| Post-reboot pipeline (SUPERSEDED) | `infra/catalysts/reagents/post_reboot_acr_boot.py` | v2 Python, correct approach but replaced by Rust |
| Catalyst recipe JSON | `infra/catalysts/reagents/gv100_acr_catalyst.json` | Segmented mmiotrace extraction |
| mmiotrace recipe | `/var/lib/toadstool/reagents/gv100_nvidia47025602_k6.17.9/mmiotrace/nv535_recipe.json` | 792,655 steps |
| Experiment doc | `experiments/223_ACR_SOVEREIGN_BOOT_CATALYST.md` | This file |

## Infrastructure (Hardened — Session 4)

### Shared Module: `barracuda/src/low_level/falcon.rs`

Canonical Falcon v5 register map, `FalconSnapshot`, PIO helpers, engine bases,
and `Bar0Domain` presets — extracted from exp224 into a tested, shared module.

### Hardened `bar0.rs`

- `Bar0Error` enum with `DeadLink`, `Unaligned`, `OutOfDomain`, `DenyListed`
- `r32_checked()` — dead-link sentinel detection (`0xFFFF_FFFF`)
- `Bar0Map::open_bdf()` — BDF-based open with `HOTSPRING_SYSFS_PCI` env support
- `SafeBar0::with_deny_list()` — ENGCTL deny-list prevents accidental destruction
- Alignment checks in `r32`/`w32`

### exp224 (Rewired)

Uses `hotspring_barracuda::low_level::{bar0::*, falcon::*}` instead of inline
constants. Opens target GPU via `SafeBar0::open_with_deny_list` with PMU+FECS
ENGCTL deny entries.

```
sudo cargo run --release --features low-level \
    --bin exp224_pmu_acr_catalyst -- \
    --target 0000:49:00.0 --control 0000:02:00.0
```

## Next Steps

1. **Investigate toadStool's `sovereign.init` internals** — map which Boot
   Falcon (NVDEC/SEC2) it uses, what DMA buffers are staged, and what sequence
   achieves PMU HS boot
2. **Extract the working boot sequence into a standalone Rust binary** (not
   dependent on toadStool daemon) for reboot-resilient sovereign boot
3. **Validate compute dispatch post-sovereign-init** on both GPUs with
   production physics workloads (not just toy shaders)

## Risk Analysis

| Risk | Mitigation |
|------|-----------|
| HS unlock is irreversible | Use only one GPU per attempt; keep other as control |
| VRAM firmware unknown | Try open-source `/lib/firmware/nvidia/gv100/acr/` first |
| ACR authentication fails | Firmware must match GPU fuse signing key |
| System lockup on GPU manipulation | Both GPUs expendable; SBR recovery available |
| PRAMIN window misconfigured | Verify BAR0_WINDOW register before staging |
