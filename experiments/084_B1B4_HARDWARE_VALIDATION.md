# Exp 084: B1-B4 bind_inst Hardware Validation

**Date:** 2026-03-24
**Type:** Hardware test (both Titan V GPUs)
**Goal:** Validate Exp 083's four bug fixes — does bind_stat reach 5?
**Result:** PARTIAL — B1-B4 fixes validated (register accepts writes), but bind_stat still times out

---

## Setup

- Both Titans on vfio-pci via GlowPlug (boot_personality="vfio")
- RTX 5060 on nvidia (PROTECTED)
- Fresh reboot — clean PCI state, no D-state processes
- coralReef built with all B1-B4 fixes applied

## Tests Run

| Test | GPU | Strategies |
|------|-----|-----------|
| `vfio_falcon_boot_solver` | Titan #2 (4a:00.0) | All 10 strategies |
| `vfio_sysmem_acr_boot` | Titan #2 (4a:00.0) | Sysmem + Hybrid |
| `vfio_falcon_boot_solver` | Titan #1 (03:00.0) | All 10 strategies |
| `vfio_sysmem_acr_boot` | Titan #1 (03:00.0) | Sysmem + Hybrid |

All tests passed (no panics), all strategies returned `success: false`.

## B1-B4 Fix Validation

| Fix | Validated? | Evidence |
|-----|-----------|----------|
| **B1: 0x054 (was 0x668)** | YES | bind_inst reads back correctly: 0x40000010 (VRAM), 0x60000040 (SysMem) |
| **B2: bit 30 enable** | YES | Values 0x40000010 and 0x60000040 both have bit 30 set |
| **B3: target=2 (was 3)** | YES | SysMem writes use 0x60000040 = (1<<30)\|(2<<28)\|(0x40>>0) |
| **B4: DMAIDX clear** | YES | 0x604 mask applied before bind in all strategies |

## bind_stat Analysis

**Register 0x0dc reads `0x000e003f` consistently across all strategies and both Titans.**

Decoding `0x000e003f`:
- bits [14:12] = 0 → bind_stat = 0 (idle/not started)
- bits [19:16] = 0xE (hardware flags)
- bits [5:0] = 0x3F (capability bits)

This value appears static — it does NOT change after bind_inst writes. The falcon's
internal bind state machine is not starting, despite the register accepting the value.

## Firmware Execution

Despite bind failure, SEC2 firmware does execute after STARTCPU:

| Metric | Value |
|--------|-------|
| PC progression | 0x0072 (VRAM strategy) or 0x007c (SysMem/Hybrid) |
| Execution time | ~56ms before timeout |
| EXCI register | 0x001f0000 (exception/trap) |
| mailbox0 | 0xcafebeef (our marker, unchanged — fw didn't write) |
| WPR FECS/GPCCS status | 1 (COPY) — ACR never completed |
| FECS/GPCCS | cpuctl=0x00000010 (still in HRESET) |

The firmware starts, runs briefly to PC 0x0072/0x007c, then traps — likely
because DMA fails without a completed instance block binding.

## Strategy 6 Notable Result (ACR Mailbox Command)

The "live SEC2" mailbox strategy showed SEC2 responding to commands:
- After BOOTSTRAP_FALCON(FECS): mb0=0x00000001, mb1=0x00000002
- After BOOTSTRAP_FALCON(GPCCS): mb0=0x00000001

mb1=0x00000002 might be an error code, but the SEC2 IS processing mailbox
commands — it's not completely dead.

## Hypotheses for bind_stat = 0

### H1: Falcon HS (High Security) mode blocks generic binding
After BIOS POST, the SEC2 falcon may be in HS-locked mode where the generic
`gm200_flcn_bind_inst` mechanism is disabled. In nouveau, the driver performs
a full engine reset cycle that transitions the falcon out of HS mode. Our VFIO
reset may not achieve the same effect.

**Evidence for:** Strategy 2 (VRAM) does an engine reset and bind_inst IS
accepted after reset, but bind_stat still doesn't activate.

**Evidence against:** cpuctl=0x00000010 (HALTED bit set) suggests the falcon IS
in a reset state after our reset sequence.

### H2: Missing power/clock prerequisite
The falcon's bind mechanism may require a specific clock domain to be active.
PRIV_RING errors (`priv_ring_intr=0xbad00100`) suggest there may be fabric
issues preventing the falcon's internal logic from executing the bind.

### H3: 0x0dc is not bind_stat for SEC2 on GV100
While nouveau uses `gm200_flcn_bind_stat` (offset 0x0dc) for all falcons
including gp102_sec2, the GV100 SEC2 might have a different bind_stat location.
The static value 0x000e003f looks more like a capability register than a status
register.

### H4: Instance block content is invalid
The falcon accepts the bind_inst pointer but the MMU can't parse the instance
block because the page table format or content is wrong. The bind state machine
starts but immediately fails back to state 0.

## Recommended Next Steps

1. **Verify 0x0dc is actually changing** — read 0x0dc at multiple points during
   the reset/bind sequence to see if it ever changes from 0x000e003f
2. **Check sctl (0x240) more carefully** — sctl=0x3000 might indicate a specific
   security state that prevents binding
3. **Compare against nouveau warm path** — swap a Titan to nouveau, let it init,
   then capture the bind_stat register and surrounding state
4. **mmiotrace during nouveau init** — would definitively show the exact bind
   sequence nouveau uses, including any prerequisites we're missing
5. **Read the full falcon register space** (0x000-0xFFF relative to SEC2 base)
   before and after bind_inst write to find any state change

## Raw Logs

Saved to `hotSpring/data/084/`:
- `titan1_boot_solver.txt` — all 10 strategies, Titan #1
- `titan1_sysmem_acr.txt` — sysmem + hybrid, Titan #1
- `titan2_boot_solver.txt` — all 10 strategies, Titan #2
- `titan2_sysmem_acr.txt` — sysmem + hybrid, Titan #2
