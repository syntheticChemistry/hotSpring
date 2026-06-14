# Exp 095: Sysmem HS Mode Breakthrough — SEC2 Enters Heavy Secure via System Memory DMA

**Date:** 2026-03-26
**Status:** FULLY CHARACTERIZED (see Exp 110 Consolidation Matrix)
**Depends:** Exp 093 (W1 fix), Exp 094 (Path B dead), coralReef Iter 67
**Goal:** Achieve SEC2 HS mode entry and ACR initialization via sovereign DMA configuration

> **Exp 110 Cross-Reference:** The HS mode achieved here was fully explained by
> Exp 110's consolidation matrix. The mechanism was **not** sysmem vs VRAM DMA per se,
> but rather the PDE slot position. This experiment's `strategy_sysmem.rs` used
> legacy lower-8-byte PDE entries, causing MMU walker fallback to physical VRAM
> addressing, which satisfied HS authentication. See `experiments/110_CONSOLIDATION_MATRIX.md`
> for the definitive truth table and `specs/archive/GPU_CRACKING_GAP_TRACKER.md` Gap 14.

## Summary

Three ACR boot strategies were tested after a nouveau cycle restored VRAM via DEVINIT. The critical discovery: **FBHUB is PRI-dead after VFIO takeover**, corrupting all DMA reads from VRAM. System memory DMA through the IOMMU is uncorrupted and successfully enters Heavy Secure (HS) mode.

| Path | DMA Source | SCTL | HS Mode | ACR Outcome |
|------|-----------|------|---------|-------------|
| VRAM | VRAM via FBHUB | 0x3000 | NO | Runs in LS mode, deaf to commands |
| Hybrid | Sysmem PTEs for code, VRAM PTEs for WPR | 0x3000 | NO | Different trace but still LS |
| **Sysmem** | **System memory via IOMMU** | **0x3002** | **YES** | **HS mode achieved**, then trapped (EXCI=0x201f0000) |

The sysmem path proves the sovereign ACR boot chain works. The trap was caused by the ACR's internal blob DMA transfer — patching `blob_size=0` to skip this DMA (WPR is pre-populated in the DMA buffer) should resolve the trap.

## Key Discoveries

### 1. FBHUB is PRI-Dead but PRAMIN Survives

```
FBHUB diagnostic: 0x100C2C = 0xbadf5040 (PRI error — hub not accessible)
FAULT_BUF0_SIZE: 0x00000000
PRAMIN sentinel: wrote=0xcafedead read=0xcafedead ok=true
```

FBHUB is the frame buffer hub — it handles all GPU-side memory access including DMA reads from VRAM. After VFIO takeover, FBHUB returns PRI errors for all register reads. However, PRAMIN (BAR0 MMIO window to VRAM) still works for host-side writes. This means:

- VRAM content written via PRAMIN persists (ACR payload, WPR, page tables)
- DMA reads from VRAM are corrupted because they route through the dead FBHUB
- The BL cannot verify the HS signature from corrupted VRAM data → falls through to LS mode

### 2. System Memory DMA Bypasses FBHUB

When the ACR payload is placed in system memory (IOMMU-mapped host memory), DMA reads go through the PCIe IOMMU path — completely bypassing FBHUB. The BL reads uncorrupted code, verifies the HS signature successfully, and transitions SEC2 to Heavy Secure mode (SCTL=0x3002).

### 3. HS Mode Trap Analysis

The sysmem path entered HS mode but trapped at PC=0x1c21 with EXCI=0x201f0000 during WPR operations. WPR headers showed `status=1` (copy started, never completed) — the ACR's internal blob DMA transfer was the trap source.

**Fix applied:** `blob_size=0` in the ACR descriptor tells the ACR that the WPR is already pre-populated in the DMA buffer. The ACR should skip its internal blob DMA and proceed directly to reading WPR headers and bootstrapping FECS/GPCCS.

### 4. TRACEPC Analysis (31-Entry Circular Buffer)

| Path | Last Trace Entries | Idle PC | Interpretation |
|------|-------------------|---------|----------------|
| VRAM | `...0x2cf8 0x2d07 0x05ee 0x1239` | 0x1da6 | Polling loop → error handler → degraded idle |
| Hybrid | `...0x346d 0x35ee 0x4e5a 0x11c6 0x3d98` | 0x1e61 | Different execution path, different idle point |
| Sysmem | (earlier runs) HS mode then trap at 0x1c21 | N/A | Entered HS, faulted during WPR blob DMA |

The VRAM path shows the ACR executing substantial code (reaching PC=0x4e5a in trace) but never leaving LS mode — consistent with signature verification failure. The sysmem path enters HS mode quickly (BL verifies signature) then advances into ACR initialization before trapping on the blob DMA.

### 5. EMEM Diagnostic (ACR Internal State)

```
EMEM[32..40]: 0x00042001 0x026c0200 0x01000000 0x00000080
              0x01000080 0x01000080 0x01000100 0xa5a51f00
```

`0x00042001` at EMEM[32] is likely an ACR status/error code. `0xa5a51f00` contains partial sentinel (0xa5a5). This region is the ACR's internal error reporting area.

## Code Changes (coralReef)

| File | Change |
|------|--------|
| `strategy_sysmem.rs` | Added `blob_size=0` patch after `patch_acr_desc` — zeroes `payload[data_off+0x258..0x268]` to skip ACR blob DMA |
| `sysmem_iova.rs` | Separated SHADOW (0x60000) from WPR (0x70000) for proper ACR descriptor layout |
| `instance_block.rs` | Made `FALCON_PT0_VRAM` and `encode_sysmem_pte` public for hybrid page table construction |
| `dma.rs` | Made `DmaBuffer::new` public for test-level DMA allocation |
| `mod.rs` (vfio_compute) | Added `dma_backend()` accessor to `NvVfioComputeDevice` |
| `falcon.rs` (test) | Hybrid page table support: sysmem PTEs for ACR code pages, VRAM PTEs for WPR/shadow |

## Architectural Insight

```
Host PRAMIN writes → VRAM content (persists)
       │
       ├─ VRAM DMA path: Falcon → FBHUB (dead) → corrupted data → LS mode
       │
       └─ Sysmem DMA path: Falcon → PCIe IOMMU → clean data → HS mode ✓
```

**Conclusion:** On GV100 after VFIO takeover, all DMA must route through system memory. VRAM can be pre-populated via PRAMIN for WPR hardware protection, but the ACR's DMA engine must read from system memory. This is a fundamental constraint of the post-VFIO FBHUB state.

## Experiment Protocol

### Phase 1: Nouveau Cycle (VRAM Recovery)

GlowPlug swaps Titan V to nouveau, waits for DEVINIT to complete (HBM2 training, VRAM initialization), then swaps back to VFIO. This restores VRAM content that was lost during previous resets.

### Phase 2: SEC2 Diagnostics

Probe SEC2 state after VFIO takeover:
- FBHUB register reads (PRI error check)
- PRAMIN sentinel write/readback
- SEC2 engine reset via PMC
- FBIF/DMACTL configuration for physical DMA

### Phase 3: ACR Boot (Three Strategies)

All three strategies use the same ACR firmware payload, same WPR construction, same boot sequence. Only the DMA source differs:

1. **VRAM:** Identity-mapped page tables, all addresses in VRAM
2. **Hybrid:** VRAM page tables with sysmem PTEs for ACR code pages, VRAM PTEs for WPR/shadow
3. **Sysmem:** All DMA buffers in system memory via IOMMU

### Phase 4: Post-Boot Diagnostics

TRACEPC dump (31-entry circular buffer), EMEM read (ACR internal state), SCTL/CPUCTL/PC/EXCI for SEC2/FECS/GPCCS.

## Next Step → Exp 111 (VRAM-Native Page Tables)

~~Path J was confirmed in Exp 096 and the full variable space mapped in Exp 110.~~

The HS+MMU paradox identified by Exp 110 requires VRAM-native page tables:
build the entire PDE/PTE chain in VRAM with correct upper-8-byte slots so the MMU
walks correctly AND code DMA resolves to VRAM (satisfying HS auth). See
`experiments/110_CONSOLIDATION_MATRIX.md` → Next Steps.
