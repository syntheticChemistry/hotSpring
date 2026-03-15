# Experiment 060: BAR2 Self-Warm Glow Plug — Eliminating Nouveau Dependency

**Date**: March 15, 2026
**Hardware**: Titan V (GV100, SM70) at `0000:4a:00.0` on `vfio-pci`
**Software**: coralReef `coral-driver` crate, diagnostic matrix
**Status**: SUCCESS — full nouveau parity achieved from cold GPU
**Chat**: [VFIO BAR2 glow plug](28732f32-750e-4053-a1ae-a8d39a738d7a)

---

## Objective

Eliminate the dependency on nouveau kernel driver for GPU warm-up. Previously,
the PFIFO scheduler required a nouveau bind→unbind cycle to configure BAR2_BLOCK
(0x1714), without which channel switches failed with `CHSW_ERROR=0x4`
(RDAT_TIMEOUT). This experiment builds a minimal BAR2 page table in VRAM from
pure Rust, achieving identical results to nouveau warming.

---

## Background

### The BAR2 Problem
On a cold GPU (post-FLR or fresh vfio-pci bind), `NV_PBUS_BAR2_BLOCK` reads
`0x40000000` (invalid — bit 30 set, no page table). The PFIFO hardware
scheduler uses BAR2 for internal memory access, and refuses to context-switch
channels without a valid BAR2 configuration, resulting in RDAT_TIMEOUT.

### Previous Workaround
A nouveau bind→warm→unbind→vfio-pci rebind cycle configured BAR2_BLOCK to
`0x002ffedf` (VIRTUAL mode page table near top of VRAM). This persisted across
driver rebind because nouveau's `bar2_fini()` only clears bit 31 (MODE), leaving
the page table intact.

---

## Register Specification (dev_bus.ref.txt, GV100)

```
NV_PBUS_BAR1_BLOCK  0x1704   PTR[27:0] | TARGET[29:28] | MODE[31]
NV_PBUS_BIND_STATUS  0x1710   [0] BAR1_PENDING [2] BAR2_PENDING
NV_PBUS_BAR2_BLOCK  0x1714   PTR[27:0] | TARGET[29:28] | MODE[31]

MODE: 0=PHYSICAL, 1=VIRTUAL
TARGET: 0=VID_MEM, 2=SYS_MEM_COHERENT, 3=SYS_MEM_NONCOHERENT
PTR: instance block VRAM address >> 12
```

Note: our code previously had BAR1/BAR2 register addresses swapped. 0x1714 is
BAR2, not BAR1. Fixed in this experiment.

---

## Implementation: Glow Plug BAR2 Setup

### VRAM Layout (6 × 4KB pages at offset 0x20000)
```
0x20000  Instance block    PDB + GV100 subcontext entries
0x21000  PD3 root          4 entries × 8 bytes (2-bit VA[48:47])
0x22000  PD2               512 entries × 8 bytes (9-bit VA[46:38])
0x23000  PD1               512 entries × 8 bytes (9-bit VA[37:29])
0x24000  PD0               256 dual entries × 16 bytes (8-bit VA[28:21])
0x25000  SPT               512 PTEs × 8 bytes (9-bit VA[20:12], 4KB pages)
```

### Page Table Format (GP100/GV100 V2 MMU)

**PDE** (8 bytes): `(child_vram_addr >> 4) | (aperture << 1)`
- VRAM aperture = 1 in bits[2:1]

**PD0 dual entry** (16 bytes): lo = small PT PDE, hi = large PT PDE

**PTE** (8 bytes): `(phys_addr >> 4) | VALID(bit0) | aper(bits[2:1]) | VOL(bit3)`
- SYS_MEM_COH: aper=2, VOL=1 → flags = 0xD

**Instance block PDB** (offset 0x200): `pd3_addr | VER2(bit10) | 64KiB(bit11)`

**GV100 subcontexts** (offsets 0x298-0x6A8): SC0 mirrors PDB, SCs 1-63 = invalid

### Sequence
1. Steer PRAMIN window to VRAM 0x20000 via BAR0_WINDOW
2. Zero-fill 24KB, then write PD3→PD2→PD1→PD0→SPT hierarchy
3. Identity-map first 2MB of IOVAs in SPT (covers all DMA buffers)
4. Write instance block with PDB + GV100 subcontext entries
5. Program BAR1_BLOCK + BAR2_BLOCK = `0x80000020` (VIRTUAL mode, VRAM, inst@0x20000)
6. Wait for BIND_STATUS
7. Flush GPU MMU TLB via 0x100CBC (PAGE_ALL + HUB_ONLY)

### Key Source (nouveau reference)
- `gf100_bar_bar2_init()` — programs BAR2_BLOCK
- `nvkm_vmm_boot()` — bootstraps page table for BAR2
- `gp100_vmm_join()` + `gv100_vmm_join()` — instance block PDB + subcontexts
- `gp100_vmm_pde()` — PDE encoding
- `gf100_vmm_invalidate()` — TLB flush

---

## Results

### Cold GPU (self-warm glow plug, no nouveau)
```
GLOW PLUG — SELF-WARMING GPU
  PMC_ENABLE=0x40000020 → 0x5fecdff1 (all engines clocked)
  PFIFO_ENABLE=0x00000000 → toggled 0→1
  BAR2_BLOCK=0x40000000 (invalid) → 0x80000020 (VRAM page table)
  BAR1_BLOCK=0x00000000 → 0x80000020

Matrix results: 54 experiments, 12 scheduled, 0 faulted, 3 CHSW_ERR (D, Z, Z2)
```

### Previous nouveau-warm reference (same code, same matrix)
```
Matrix results: 54 experiments, 12 scheduled, 0 faulted, 3 CHSW_ERR (D, Z, Z2)
```

### Comparison: IDENTICAL
The 12 winning experiments, 3 CHSW failures, and all PBDMA/PCCSR register
readbacks are identical between self-warm and nouveau-warm. The 3 CHSW failures
(D_coh, Z_full_reinit, Z2_reinit) are expected — they deliberately write
invalid sentinel values to PBDMA registers or perform PMC resets.

---

## Winning Experiments (identical in both warm modes)
```
I_activate_sched_coh       I_activate_sched_ncoh
T_sched_doorbell_coh       T_sched_doorbell_ncoh
R_ramfc_sched_coh          R_ramfc_sched_ncoh
S_both_sched_coh           S_both_sched_ncoh
U_cleanSched_coh           U_cleanSched_ncoh
U2_nopPushbuf_coh          U2_nopPushbuf_ncoh
```

---

## Findings

1. **BAR2 is critical**: PFIFO scheduler cannot load channels without BAR2
2. **BAR1 likely unnecessary**: Setting BAR1 to same config as BAR2 for safety
3. **TLB flush required**: Without `gf100_vmm_invalidate`, first context switch
   may fault due to stale TLB entries
4. **VRAM is writable on cold GPU**: PRAMIN writes land even without FB init
5. **Register naming was wrong**: 0x1714 = BAR2_BLOCK (not BAR1). Fixed.
6. **PDE/PTE encoding bug found**: Our `encode_pde` FLAGS=6 decodes as
   NCOH(aperture=3) not COH(aperture=2). Works because NCOH still maps system
   memory, but should be fixed for correctness.

---

## Files Modified
- `registers.rs`: Fixed BAR1/BAR2 register addresses, added BIND_STATUS,
  added BAR2 VRAM layout constants
- `pfifo.rs`: Added `setup_bar2_page_table()` — builds V2 page table in VRAM,
  programs BAR1/BAR2_BLOCK, flushes TLB
- `runner.rs`: Glow plug calls `setup_bar2_page_table()` when BAR2_BLOCK is
  invalid, updated oracle comparison output
