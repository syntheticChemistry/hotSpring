# hotSpring → coralReef Handoff: Sovereign Pipeline Hardening (May 6 2026, Sprint B)

> **Date:** May 6, 2026
> **coralReef iteration:** 90 (pending push)
> **Status:** All code changes compile clean, 465/465 lib tests pass, 0 warnings on modified files.

---

## Summary

Three-GPU sovereign pipeline hardening sprint targeting RTX 5060 (production),
Titan V GV100 (HBM2 reclamation), and Tesla K80 GK210B (legacy reclamation).
NVIDIA dropped driver support for both legacy GPUs — sovereign compute reclaims
their silicon through pure Rust + VFIO.

### Deliverables

| # | Deliverable | GPU | Files |
|---|-------------|-----|-------|
| 1 | SLM pool allocation (fixes shared/local memory dispatch) | All | `layout.rs`, `mod.rs`, `device_open.rs`, `dispatch.rs` |
| 2 | SSEL per-engine PLL masking (fixes dead GPC clock) | K80 | `kepler_nouveau_clk.rs` |
| 3 | Post-PMU GPC PLL retry | K80 | `kepler_cold.rs` |
| 4 | Titan V SEC2 FBIF instance-block DMA config | Titan V | `nouveau.rs`, `legacy_acr.rs`, `boot_prepare.rs` (prior sprint) |
| 5 | Volta sovereign pipeline diagnostic | Titan V | `examples/volta_sovereign_pipeline.rs` (NEW) |
| 6 | unsafe audit (all NECESSARY — no reduction needed) | All | N/A (analysis only) |

---

## 1. SLM Pool Allocation

**Problem:** `PushBuf::compute_init` was called with `slm_base=0, slm_per_tpc=0`,
meaning `SET_SHADER_LOCAL_MEMORY_A/B` and `SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A/B`
were zero. Any shader using local or shared memory would fault.

**Fix:**
- `layout.rs`: Added `SLM_IOVA` (0xA_0000), `SLM_SIZE` (2 MiB), `SLM_PER_TPC` (32 KiB).
  Moved `USER_IOVA_BASE` from 0x10_0000 to 0x30_0000. Updated const IOVA overlap assertions.
- `mod.rs`: Added `slm_buf: Option<DmaBuffer>` field + `slm_params()` helper.
- `device_open.rs`: Allocate SLM DMA buffer in all 3 open paths (cold, ember FD, warm).
  Non-fatal on allocation failure (falls back to slm_base=0).
- `dispatch.rs`: Both `dispatch_inner` and `dispatch_inner_traced` now use `slm_params()`.

**Matches UVM path:** 2 MiB total, 0x8000 per TPC (supports up to 64 TPCs).

---

## 2. SSEL Per-Engine PLL Masking (K80)

**Problem:** `program_engine_plls` in `kepler_nouveau_clk.rs` set all 3 SSEL bits
(`0x137100 |= 0x7`) when *any* PLL locked. If GPC's PLL failed to lock but ROP's
succeeded, GPC was forced into PLL mode with a broken PLL — producing a dead
GPC clock domain.

**Fix:** Track per-engine lock status in a `locked_mask: u32`. Only set SSEL bits
for engines whose PLLs actually achieved lock (bit 17 in PLL CTRL):

```rust
// Before: w(0x13_7100, ssel | 0x0000_0007);
// After:
w(0x13_7100, (ssel & !0x0000_0007) | locked_mask);
```

---

## 3. Post-PMU GPC PLL Retry (K80)

**Problem:** Cold-boot programs clocks (Phase 3) before PMU firmware boot (Phase 3.75).
PMU manages power domains — it may ungate the clock domain that blocked GPC PLL writes.

**Fix:** After PMU boot, if GPCs are still dead, re-test `0x137000` writability.
If PMU ungated the power domain, retry crystal clocks + engine PLLs.

---

## 4. Volta Sovereign Pipeline Diagnostic

**New file:** `examples/volta_sovereign_pipeline.rs`

8-stage diagnostic for Titan V cold sovereign boot:
1. VFIO open + BOOT0 identity
2. PCI hot reset (clear stale LS-mode falcons)
3. Falcon state probe (SEC2, FECS, GPCCS)
4. SEC2/ACR boot solver (all strategies)
5. FECS alive check
6. GR context discovery (FECS method interface)
7. Channel creation (5-level page tables)
8. NOP dispatch (GPFIFO push → doorbell → GP_GET poll)

Run with:
```
RUST_LOG=info cargo run --example volta_sovereign_pipeline --features vfio -- <BDF>
```

---

## 5. unsafe Audit Results

All `unsafe` blocks in the VFIO BAR access paths are **NECESSARY**:

| Module | Blocks | Classification |
|--------|--------|---------------|
| `mmio_region.rs` | 4 | volatile MMIO, mmap/munmap |
| `dma.rs` | 7 | alloc_zeroed, mlock, from_raw_parts |
| `mmio.rs` (VolatilePtr) | 2 | volatile read/write |
| `mapped_bar.rs` | 5 | fork-isolation, Send/Sync |

No reduction opportunities — abstractions are already well-encapsulated
behind `MmioRegion`, `VolatilePtr`, `DmaBufferBytes`.

---

## Hardware Validation Status

| GPU | BDF | Status | Next Step |
|-----|-----|--------|-----------|
| RTX 5060 (SM120) | display GPU | ✅ Dispatch live | Production |
| Titan V (GV100) | TBD | Code ready | Run `volta_sovereign_pipeline` |
| Tesla K80 (GK210B) | 0000:4b:00.0 | Code ready | Run `kepler_cold_pipeline` |

---

## For Next Sprint

- Run both diagnostic pipelines on hardware, capture traces
- If Titan V SEC2 boots: wire full compute dispatch (shader → QMD → GPFIFO)
- If K80 GPCs come alive: test NOP dispatch on cold path
- Begin era-agnostic abstraction: factor GPU-specific boot sequences behind trait
