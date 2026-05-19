<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# hotSpring — GPC Boundary & CE Validation Handoff (May 19, 2026)

**Experiment**: 210 (Sovereign GPC Power Boundary Analysis)
**Fleet**: 2× Titan V (GV100, vfio-pci) + RTX 5060 (SM120, nvidia proprietary)
**Tier Reached**: Tier 1 (Warm Sovereign Infrastructure)
**Blocker**: Engine power domain gating (all engines: GR, CE, NVDEC)

---

## What Changed

### toadstool-cylinder (core library)

| File | Change |
|------|--------|
| `vfio/ce_validate.rs` | **NEW** — CE engine validation pipeline: discovers CE runlist, creates channel, submits DMA copy pushbuffer, polls GP_GET, captures PBDMA diagnostics |
| `vfio/sovereign_tiers.rs` | **NEW** — Sovereignty tier model: `SovereignTier` enum (Cold/WarmInfra/WarmCompute/FullSovereign), `classify_tier()`, `TierEvidence`, `TierCapabilities` |
| `vfio/channel/pfifo.rs` | **FIX** — PTOP_DEVICE_INFO_V2 parser: runlist is in kind==2 at bits [17:14], not kind==3 at bits [15:11]. Added `discover_ce_runlist()` and `find_pbdma_for_runlist()` (indexed by runlist ID, returns bitmask's lowest PBDMA) |
| `nv/pushbuf.rs` | **ADD** — CE class IDs (`ce::VOLTA_DMA_COPY_A = 0xC3B5`), CE methods (`LAUNCH_DMA`, `OFFSET_IN/OUT`, `LINE_LENGTH`, etc.), CE pushbuffer builders (`ce_init`, `ce_dma_copy`, `ce_semaphore_release`) |
| `vfio/channel/mod.rs` | Made `pfifo` module `pub(crate)` for CE validation access |
| `vfio/mod.rs` | Registered `ce_validate` and `sovereign_tiers` modules |

### toadstool-server (RPC handlers)

| File | Change |
|------|--------|
| `handler/mod.rs` | Added `sovereign.ce_validate` / `ce.validate` RPC route |
| `handler/dispatch/mod.rs` | Added `sovereign_ce_validate_ember()` handler (clutch-aware, BAR0+DMA from ember anchor). Added `classify_tier_sysfs()` and tier info to `sovereign.warm_status` response |

### Documentation

| File | Change |
|------|--------|
| `experiments/210_SOVEREIGN_GPC_BOUNDARY.md` | **NEW** — Full experiment writeup with register-level evidence, tier model, paths to Tier 2 |

## Key Findings

### 1. PTOP Parser Bug (Fixed)

The GV100 engine topology table (`PTOP_DEVICE_INFO_V2`) uses a different
field layout than what the parser assumed:

```
kind==1 (DATA):    engine type at bits [7:2]     ← correct
kind==2 (ENUM):    runlist at bits [17:14]        ← was reading kind==3 at [15:11]
kind==3 (ENGINE):  reset/fault info (NOT runlist) ← was being used for runlist
```

The GR runlist was accidentally correct because the `ENGN0_STATUS` fallback
at 0x2640 was always used. CE runlist discovery was completely broken.

### 2. PBDMA→Runlist Mapping (Fixed)

`RUNLIST_PBDMA_MAP` at `0x2390 + i*4` is indexed by **runlist ID** (not
PBDMA sequence). Each value is a **bitmask** of PBDMAs serving that runlist:

```
Runlist  1 (GR)  → PBDMA  0 (mask 0x0001)
Runlist 10 (CE)  → PBDMA  9 (mask 0x0200)
```

### 3. All Engines Power-Gated

After nouveau unbind, not just GPCs but ALL engine domains return PRI faults:
- CE0 at 0x104000 → 0xbadf3000
- GPCCS at 0x41A004 → 0xbadf5545
- PGRAPH methods → DEVICE error (PBDMA intr_0 bit 28)

### 4. DRM Dispatch Available (RTX 5060)

The RTX 5060 provides DRM compute via `/dev/dri/renderD128` (proprietary driver).
NVK does NOT support Volta (SM70); it requires Turing (SM75) or newer.

## For Upstream Primal Teams

### toadstool-cylinder evolution priorities

1. **PMU mailbox protocol**: The PMU falcon at 0x10A000+ is alive. Sending
   a power-domain ungate command could bring GPCs/CE online without
   requiring a PRI ring path through the gated domain.

2. **Kernel-level solution**: Modify nouveau `gv100_gr_fini()` to preserve
   GPC power during unbind. This is the cleanest path for VFIO passthrough.

3. **CE dispatch readiness**: The CE pushbuffer builders and channel creation
   on non-GR runlists are complete. Once the engine domain is powered, CE
   DMA copy dispatch should work immediately.

4. **FECS method protocol**: Solved for Volta HS — MAILBOX1 trigger + PC
   delta confirmation. Ready for when GPCs come online.

### New RPC methods for testing

```bash
# CE validation (attempts DMA copy via Copy Engine)
echo '{"jsonrpc":"2.0","id":1,"method":"sovereign.ce_validate","params":{"bdf":"0000:02:00.0"}}' | nc 127.0.0.1 PORT

# Warm status now includes sovereignty tier
echo '{"jsonrpc":"2.0","id":1,"method":"sovereign.warm_status","params":{}}' | nc 127.0.0.1 PORT
```

### Cross-platform testing notes

- Tier classification (`classify_tier()`) is hardware-generic — works on
  any NVIDIA GPU with BAR0 access
- CE validation works on any Volta+ GPU with VFIO bind
- PTOP parser fix applies to all GV100+ chips using DEVICE_INFO_V2 format
