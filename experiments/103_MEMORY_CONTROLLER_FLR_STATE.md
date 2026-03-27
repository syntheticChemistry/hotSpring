# Experiment 103: Memory Controller and FLR State Investigation

**Date:** 2026-03-25
**Status:** COMPLETED — FLR not the cause, IMEM read bug found and fixed

## Objective

Determine if the GPU memory controller (FBHUB) is properly initialized after
FLR, and whether the DMA trap is caused by memory subsystem misconfiguration.

## Variants

### v1: VRAM + FBHUB + MC Diagnostics
- VRAM accessibility confirmed (write/readback sentinel OK)
- FBHUB reported FAULT_STATUS=0x00000000 (no GPU MMU fault)
- MEM_CTRL=0x00000000, MEM_ACK=0xbadf5040
- DMA trap still at TRACEPC=0x0500

### v2: No-FLR Test
- Created `exp103_no_flr.rs` — disable PCI reset before vfio-pci swap
- Preserved nouveau's full GPU initialization state
- **Same DMA trap at TRACEPC=0x0500** — FLR hypothesis disproved

### v3: Fixed IMEM Read + IRQ Diagnostics
- Found and fixed IMEM read addressing bug (block vs byte addressing for Falcon v5)
- IRQSTAT=0x00000010 (HALT only, no EXTERR) — crash is INTENTIONAL
- FW code at 0x500: `0x09980002` — real Falcon instruction, not garbage
- The ACR firmware deliberately halts at a precondition check

### v4: Nouveau Mailbox + VRAM PTEs for ACR Payload
- Set MAILBOX0 to 0xDEADA5A5 before boot (nouveau convention)
- Extended VRAM PTEs to cover ACR payload pages
- Post-boot: MAILBOX0 still 0xDEADA5A5 (firmware never wrote to it)
- ACR code halts before reaching mailbox write — very early failure

## Key Findings

1. **FLR is not the cause** — same crash with and without FLR
2. **No GPU MMU faults** — FBHUB is clean
3. **The halt is intentional** — IRQSTAT shows HALT, not EXTERR
4. **MAILBOX0 never written** — crash happens before error reporting
5. **Firmware instruction at crash PC is real** — not data corruption
