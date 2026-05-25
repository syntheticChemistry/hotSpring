# Experiment 222: Reagent Capture Pipeline — Sovereign Tier 2

**Date**: 2026-05-25 (updated 2026-05-25 session 2)
**Status**: EVOLVED — dual GPU compute validated, PMU firmware determinism proven, SBR cold reset achieved, sovereign boot path mapped.
**Upstream plan**: `reagent_capture_pipeline_af157b38.plan.md`
**Depends on**: Exp 221 (UEFI Model GPU Sovereignty)

## Hypothesis

Exp 221 discovered that GV100's FECS/GPCCS falcons enforce fuse-locked High
Security mode, making direct host IMEM PIO and independent firmware replay
non-viable for Tier 2. Instead of fighting the HS boundary, we can:

1. **Track A**: Keep nvidia loaded as a runtime compute service while toadStool
   manages infrastructure — immediate Tier 2 access
2. **Track B**: While nvidia is alive and everything is warm, systematically
   capture every firmware "chemical agent" that enables compute — store as
   versioned reagents for late-stage sovereign Tier 2 replay

This is the "diesel engine captures the chemical agents" strategy: use the
vendor driver as a chemistry engine, extract and catalog what it produces,
build a reagent library for future independent replay.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Track A: Runtime Services (nvidia loaded, Tier 2 now)      │
│   nvidia stays bound → FECS/GPCCS running → TPC alive     │
│   toadStool manages: PFIFO, DMA, VRAM, PRI ring           │
├─────────────────────────────────────────────────────────────┤
│ Track B: Reagent Capture (extract while alive)             │
│   BAR0 snapshot → PRAMIN VRAM read → mmiotrace distill    │
│   linux-firmware catalog → catalyst artifact copy          │
├─────────────────────────────────────────────────────────────┤
│ Track C: Late-stage replay (sovereign Tier 2 from reagents)│
│   Cold boot → WPR → SEC2 → ACR → FECS/GPCCS → TPC        │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

### Track A: Runtime Services

Added `RuntimeServicesConfig` to `sovereign_handoff.rs` — a handoff strategy
where nvidia stays bound as the `final_driver`. No unbind/swap occurs.

| Component | File | Description |
|-----------|------|-------------|
| `RuntimeServicesConfig` | `cylinder/src/vfio/reagent.rs` | Config struct for runtime services mode |
| `nvidia_runtime_services()` | `sovereign_handoff.rs` | Handoff config factory — nvidia stays bound |
| `is_runtime_services()` | `sovereign_handoff.rs` | Predicate for no-swap mode detection |
| `probe_runtime_services()` | `sovereign_handoff.rs` | Probes nvidia's live FECS/TPC/channel state |
| `RuntimeServicesProbe` | `sovereign_handoff.rs` | Structured probe result |
| `sovereign.runtime_services_probe` | RPC handler | JSON-RPC endpoint for runtime services probing |

**Key design decision**: nvidia remains loaded as a persistent service. toadStool
does NOT attempt to unbind or swap. This avoids the PRI ring destruction discovered
in Exp 221 while providing immediate Tier 2 compute access.

### Track B: Reagent Capture Pipeline

#### B1: ReagentManifest

New module `cylinder/src/vfio/reagent.rs` containing unified data structures
for all captured chemical agents:

| Struct | Purpose |
|--------|---------|
| `ReagentManifest` | Top-level manifest — chip, driver, kernel, artifact paths, completeness |
| `ReagentFirmware` | Per-falcon firmware artifacts (IMEM, DMEM, ACR sequence, linux-firmware) |
| `FirmwareBlob` | Individual cataloged firmware file with metadata |
| `ReagentCompleteness` | Bitfield tracking which capture strategies succeeded |
| `MmiotraceReagentSummary` | Summary of mmiotrace distillation results |
| `ReagentCaptureResult` | Full pipeline output with per-strategy results |
| `StrategyResult` | Individual strategy success/failure with artifacts |

#### B2: Firmware Capture Strategies

**B2a: linux-firmware catalog** (`catalog_linux_firmware()`)
- Scans `/lib/firmware/nvidia/{chip}/` for 19 known firmware files
- Cataloged blobs with subsystem, size, ACR-required flag
- GV100 result: 15 blobs found (10 ACR-required present)
  - gr/: fecs_bl.bin (576B), fecs_inst.bin (25632B), fecs_data.bin (4788B),
    gpccs_bl.bin (→gp107 symlink), gpccs_inst.bin (12643B), gpccs_data.bin (2128B),
    sw_ctx.bin, sw_nonctx.bin, sw_bundle_init.bin, sw_method_init.bin
  - acr/: bl.bin (→gp102 symlink), ucode_unload.bin (6400B)
  - sec2/: desc.bin (656B), image.bin (91136B), sig.bin (192B)
  - pmu/: MISSING (no directory — blocks nouveau compute, Exp 186)

**B2b: mmiotrace distillation** (`distill_mmiotrace_to_reagent()`)
- Parses existing `nvidia535_init_full.log` (1.16M ops, 401K BAR0 writes)
- Uses `BootTrace::from_mmiotrace()` to parse, `to_recipe()` for domain ordering
- Produces full recipe JSON + ACR-domain subset (PMC, PRI_MASTER, PMU, PFIFO, PBDMA, PRAMIN)
- ACR subset captures the exact register sequence for falcon boot chain

**B2c: VRAM firmware capture** (`read_vram_via_pramin()`, `capture_vram_firmware()`)
- PRAMIN window (BAR0 0x700000, 1 MiB) reads arbitrary VRAM when properly configured
- FECS firmware staged at VRAM 0x802FD458 (Exp 160 finding, nvidia-535)
- GPCCS firmware at hinted offset (0x10000 after FECS)
- Validation: rejects captures with <10% nonzero bytes (indicates wrong address)
- This bypasses HS IMEM PIO block entirely — reads from VRAM, not falcon registers

**B2d: Catalyst artifact copy** (`copy_catalyst_artifacts()`)
- Copies existing `/var/lib/toadstool/catalysts/` recipe/firmware files to reagent store
- Bridges prior catalyst captures into the unified reagent library

#### B3: Reagent Storage

Two-tier storage:
- **Runtime**: `/var/lib/toadstool/reagents/{chip}_{driver}_{kernel}/` — all artifacts
- **Repo**: `infra/catalysts/reagents/` — JSON manifests and small recipes only

```
reagents/
  gv100_nvidia47025602_k6.17.9/
    manifest.json
    firmware/
      fecs_inst.bin
      gpccs_inst.bin
      ...
    mmiotrace/
      nvidia535_recipe.json
      nvidia535_recipe_acr_subset.json
```

#### B4: RPC Orchestration

| RPC Method | Handler | Description |
|------------|---------|-------------|
| `sovereign.reagent_capture` | `sovereign.rs` | Full reagent capture pipeline |
| `sovereign.runtime_services_probe` | `sovereign.rs` | Probe nvidia live state |

`sovereign.reagent_capture` orchestrates:
1. Catalog linux-firmware blobs
2. Probe nvidia state via sysfs
3. Copy catalyst artifacts
4. Optionally distill mmiotrace (when `mmiotrace_path` param provided)
5. Persist ReagentManifest to storage

## Files Changed

### toadStool (upstream)

| File | Change |
|------|--------|
| `crates/core/cylinder/src/vfio/reagent.rs` | NEW — ReagentManifest, firmware catalog, mmiotrace distill, PRAMIN capture |
| `crates/core/cylinder/src/vfio/mod.rs` | Added `pub mod reagent` |
| `crates/core/cylinder/src/vfio/sovereign_handoff.rs` | Added RuntimeServicesConfig, probe, strategy |
| `crates/server/src/pure_jsonrpc/handler/sovereign.rs` | Added reagent_capture + runtime_services_probe RPCs |
| `crates/server/src/pure_jsonrpc/handler/mod.rs` | Wired two new RPC routes |

### hotSpring (this repo)

| File | Change |
|------|--------|
| `infra/catalysts/reagents/README.md` | NEW — reagent library documentation |
| `infra/catalysts/reagents/gv100_nvidia47025602_k6.17.9/manifest.json` | NEW — seed manifest |
| `experiments/222_REAGENT_CAPTURE_PIPELINE.md` | NEW — this document |

## Key Findings

### HS Boundary is a Firmware Interface, Not a Wall

Exp 221 showed HS mode is fuse-enforced — it cannot be bypassed by register
tricks. But it's a firmware boundary, not a silicon wall. The ACR boot chain
(SEC2 → ACR → FECS/GPCCS) is the interface. The reagent capture pipeline
extracts everything needed to eventually replay this interface independently.

### PRAMIN Window is the Bypass for HS IMEM Block

HS mode prevents direct IMEM PIO reads/writes from the host. But firmware is
staged in VRAM before the BootROM DMA loads it to IMEM. The PRAMIN window
(BAR0 0x700000) can read VRAM content while nvidia is loaded, bypassing the
IMEM block entirely. This is the capture path for Track B.

### linux-firmware Blobs ARE the Reagents

The signed firmware blobs in `/lib/firmware/nvidia/gv100/` are the exact
chemical agents that ACR authenticates and loads. Exp 206 already proved
ACR DMA boot works with these blobs on warm Volta. The gap is cold boot
(proper WPR + HS authentication), which is Track C.

## Runtime Validation Results (May 25, 2026)

### System Topology

| GPU | BDF | Driver | Role |
|-----|-----|--------|------|
| RTX 5060 | 0000:21:00.0 | nvidia-580 | Display GPU, runtime services host |
| Titan V #1 | 0000:02:00.0 | vfio-pci | Sovereign target, reagent catalog |
| Titan V #2 | 0000:49:00.0 | vfio-pci | Sovereign target |

### 1. Runtime Services Probe

```json
{
    "bdf": "0000:21:00.0",
    "driver": "nvidia",
    "fecs_state": "running (nvidia owns FECS context)",
    "nvidia_channels": 1,
    "nvidia_loaded": true,
    "runtime_services_ready": true,
    "tpc_alive": true
}
```

**Result**: nvidia-580 on RTX 5060 confirmed `runtime_services_ready: true`. FECS context alive.

### 2. Reagent Capture Pipeline

Full pipeline executed via `sovereign.reagent_capture` RPC:
- **15 linux-firmware blobs** cataloged (10 ACR-required)
- **Catalyst artifacts** copied (patch set recipe)
- **nvidia state** probed (driver=vfio-pci for Titan V)
- **Completeness**: 42.9% (3 of 7 strategies succeeded)

### 3. mmiotrace Distillation

**Bug fix**: MAP line parser was reading `parts[2]` (entry number) instead of `parts[3]` (physical address). Fixed in `boot_follower.rs`.

nvidia535_init_full.log (47.5 MB, 1.16M lines) distilled:
- **792,655 BAR0 writes** → recipe steps
- **762,135 ACR-relevant** (96% of total)
- Domain breakdown:
  - UNKNOWN (GPC/TPC/PGRAPH): 655,204
  - PBDMA: 68,905
  - PRAMIN: 32,768 (VRAM staging)
  - LTC: 18,953
  - PRI_RING: 20,779 (topology setup)
  - PMU: 1,203 (falcon programming)
  - FECS falcon: 4 (mailbox only — HS blocks IMEM)
  - GPCCS falcon: 4 (mailbox only)

**Critical finding**: Ranges 0x200000-0x390000 each have 8,192 writes — these are the **TPC PRI station registers** being created. This is the exact programming sequence needed for Track C.

### 4. PRAMIN VRAM Capture

PRAMIN window is accessible (BAR0_WINDOW = 0x0002fff0) but VRAM is empty — Titan V is on vfio-pci, not nvidia. PRAMIN capture requires nvidia bound to the target GPU. Deferred to next nvidia-on-TitanV session.

### 5. Linux-Firmware Blob Validation

| Blob | Size | ACR Chain |
|------|------|-----------|
| sec2/image.bin | 91,136 B | SEC2 → ACR |
| gr/fecs_inst.bin | 25,632 B | FECS IMEM |
| gr/gpccs_inst.bin | 12,643 B | GPCCS IMEM |
| gr/sw_method_init.bin | 12,296 B | GR init |
| gr/sw_ctx.bin | 9,756 B | Context switch |
| gr/sw_bundle_init.bin | 7,664 B | Bundle init |
| acr/ucode_unload.bin | 6,400 B | ACR unload |
| gr/fecs_data.bin | 4,788 B | FECS DMEM |
| gr/sw_nonctx.bin | 2,728 B | Non-ctx init |
| gr/gpccs_data.bin | 2,128 B | GPCCS DMEM |
| sec2/desc.bin | 656 B | SEC2 descriptor |
| sec2/sig.bin | 192 B | SEC2 signature |

PMU firmware: **MISSING** (no `/lib/firmware/nvidia/gv100/pmu/` directory)

## Artifacts

| File | Size | Description |
|------|------|-------------|
| `/var/lib/toadstool/reagents/gv100_nvidia47025602_k6.17.9/manifest.json` | 4 KB | Reagent manifest |
| `.../mmiotrace/nv535_recipe.json` | 73.6 MB | Full 792K-step recipe |
| `.../mmiotrace/nv535_recipe_acr_subset.json` | 70.7 MB | ACR-domain subset |
| `.../gv100_nvidia470_patchset.json` | 1.2 KB | Catalyst patch set |

## Hardware Safety Findings (May 25, 2026)

### Lockup Root Cause: No FLR on Titan V

Two system-wide hard lockups occurred during `sovereign.init` + recipe replay
attempts. Root cause analysis:

- **Titan V has no Function Level Reset**: `FLReset-` in PCIe DevCap
- **All device reset methods disabled**: `reset_method` is empty on both Titan Vs
- **`disable_idle_d3=1`** prevents D3 power state recovery
- Once `sovereign.init` writes registers that put the GPU in a conflicting state,
  there is **zero recovery** except full system power cycle
- Second init attempt writes into dirty state → uncorrectable PCIe errors → MCE/hang
- DevStatus 0x0009 (Correctable Error + Unsupported Request) visible even at cold boot

**Rule**: Do NOT run `sovereign.init` or recipe replay on Titan V via sysfs BAR0.
Only read-only operations are safe on vfio-pci-bound Titan Vs without FLR.

### Recipe Replay: `sovereign.recipe_replay` RPC

New RPC wired (`sovereign.recipe_replay`) that loads RecipeStep JSON and
replays via sysfs BAR0. Blocked on Titan V due to FLR absence.
Could be used on GPUs with proper reset support in future.

### nouveau Warm Handoff Attempt

Bound nouveau to Titan V #2 (49:00.0):
- nouveau initializes to **Tier 1** safely (no lockups) — PMC popcount 23
- **GR does not initialize**: `pmu: firmware unavailable`
- Desktop GV100 PMU firmware does not exist in linux-firmware (checked both
  installed and latest available 20260221 packages, also verified Fedora repos)
- FECS falcon responds (PC advances: 126 → 215 → 270) but remains halted
- TPC stations remain PRI-faulted across all 6 GPCs
- Channel alloc with GR classes (0xC397, 0xC3C0) succeeds but doesn't trigger GR init
- `k80_force_gr_init` tool fails: `nouveau_device_new: ENOSYS` on kernel 6.17 API change

**Blocker**: nouveau's ACR boot chain (SEC2 → FECS/GPCCS) requires PMU firmware
that NVIDIA never released for desktop Volta.

### RTX 5060 Runtime Services: Confirmed

`sovereign.runtime_services_probe` on RTX 5060 (0000:21:00.0):
- `runtime_services_ready: true`, FECS running, TPC alive, nvidia channels active
- sysfs BAR0 reads blocked by nvidia PRI security (all registers return PRI faults)
- Compute dispatch must go through nvidia APIs (CUDA, ioctl, UVM)

## Session 2 Results (2026-05-25)

### 1. Dual GPU Compute Validation

**RTX 5060 (host, nvidia-580)**:
- `runtime_services_ready: true` at BDF `0000:21:00.0`
- VectorAdd 1M elements: **PASS**
- SM 12.0, 30 SMs, 7702 MB VRAM
- CUDA 12.8, driver 580.126.18

**Titan V (VM, nvidia-470)**:
- VM `titanv-warmhandoff` started from existing qcow2
- nvidia-470.256.02 rebuilt for kernel 6.8.0-111
- CUDA Driver API context creation: **PASS**
- HBM2 memset+readback (0xCAFEBABE): **PASS**
- SM 7.0, 80 SMs, 12066 MB HBM2
- Temperature: 47°C

### 2. PMU Firmware Determinism — PROVEN

Live capture from running nvidia-470 VM confirmed firmware is identical
across reboots:

| Artifact | Size | MD5 (May 11 = May 25) |
|----------|------|-----------------------|
| pmu_imem.bin | 64 KB | `5a043df5073b5c7eb661258c4a77c7c9` |
| pmu_dmem.bin | 64 KB | `71d162800f93c9490130fa5b409c664f` |

IMEM: 16384/16384 words non-zero (ALL valid code)
DMEM: 16384/16384 words non-zero (ALL valid data)
First DMEM word: 0xdead5ec2 (nvidia RM signature)

### 3. Live Falcon State Capture (nvidia-470 RUNNING)

| Falcon | State | Detail |
|--------|-------|--------|
| PMU | RUNNING (0x20) | SCTL=0x3002 (HS3), MB0=0x300, PC advancing |
| FECS | HALTED (0x10) | HS-protected IMEM (reads zeros), ACR authenticated |
| GPCCS | HALTED (0x10) | HS-protected |
| SEC2 | NOT POWERED | 0xbadf1100 (Volta uses PMU for ACR, not SEC2) |
| GR | IDLE (0x00) | FECS OS=0x802fd441, PGRAPH ready |

**Key finding**: On Volta, PMU runs ACR (not SEC2). SEC2 is unused.

### 4. Sovereign PMU Boot Experiments

**Experiment A: PIO upload (HS mode blocking)**
- PIO IMEM writes: **BLOCKED** by HS mode 3 (readback shows HS ROM content)
- PIO DMEM writes: **WORK** (verified: 0xdead5ec2 matches)
- PMC_ENABLE writes: **BLOCKED** (in nouveau-loaded state)
- STARTCPU: **IGNORED** (HS mode prevents host-initiated CPU start)
- **System stable** — PIO writes are safe operations, no lockups

**Experiment B: DMATRF upload (alternative path)**
- PRAMIN staging: **WORKS** (firmware written to VRAM verified)
- DMATRF: 256/256 blocks completed, but DMATRFBASE readback mismatch
- IMEM readback: still HS ROM content — DMATRF didn't bypass HS
- Conclusion: both PIO and DMATRF blocked by HS mode 3

**Experiment C: SBR cold reset — BREAKTHROUGH**
- PCIe SBR via upstream bridge (0000:40:01.3): **WORKS**
- After SBR: all registers 0xFFFFFFFF (device in link-down)
- After PCI rescan + COMMAND enable: GPU responds
- **PMU auto-boots from HS ROM**: CPUCTL=0x20, MB0=0x300
- **PMC_ENABLE writes work in post-SBR state**: 0x40000020 → 0x42001120 accepted
- **FECS powers up after PMC_ENABLE**: CPUCTL=0x10, HALTED at HS ROM gate
- System remained 100% stable through entire SBR cycle

### 5. Volta Boot Chain — Complete Understanding

```
SBR/Power-On
    ↓
PMU HS ROM auto-boots (fuse-enforced)
    ↓ MB0=0x300 (ready for ACR commands)
Host writes PMC_ENABLE → powers FECS, GPCCS
    ↓
FECS/GPCCS HS ROMs auto-boot, halt waiting for ACR auth
    ↓
Host stages signed firmware + ACR ucode to VRAM (WPR region)
    ↓
Host signals PMU via mailbox with WPR address
    ↓
PMU (HS ROM code) reads firmware from VRAM via DMA
PMU runs ACR → authenticates FECS/GPCCS firmware
    ↓
ACR loads authenticated firmware into FECS/GPCCS falcons
    ↓
FECS initializes GR engine, creates TPC PRI stations
    ↓ GR_STATUS = 0x00 (idle), FECS_OS loaded
Sovereign Tier 2 — compute ready
```

### 6. nouveau GR Blocker — Root Cause Confirmed

nouveau does NOT request PMU firmware for GV100 (no `nvidia/gv100/pmu/`
in firmware manifest). Stock nouveau.ko tested — same result: modesetting
works, GR completely skipped. `pmu: firmware unavailable` is the only
PMU-related message; GR init is never attempted.

This is a **kernel design gap**: nouveau's GV100 support is display-only.
The GR initialization path requires PMU → ACR → FECS/GPCCS, which nouveau
cannot perform without PMU firmware.

## Current System State

| GPU | Driver | Path | Compute |
|-----|--------|------|---------|
| RTX 5060 (21:00.0) | nvidia-580 | Host | CUDA 12.8 VectorAdd PASS |
| Titan V #1 (02:00.0) | nvidia-470 | VM `titanv-warmhandoff` | CUDA 11.4 HBM2 PASS |
| Titan V #2 (49:00.0) | vfio-pci | Cold (post-SBR) | Reagent target |

## Next Steps — Sovereign Boot Path

### Remaining blocker: WPR + ACR mailbox protocol

The SBR → PMC_ENABLE → FECS power-up sequence works. The final piece is:

1. **Capture nvidia-470's WPR setup** from mmiotrace:
   - Where in VRAM does nvidia-470 stage ACR firmware?
   - What mailbox command triggers PMU's ACR execution?
   - What instance block / DMA context does PMU use?

2. **Replay WPR setup on cold GPU**:
   - After SBR, stage signed firmware to VRAM via PRAMIN
   - Configure WPR2 registers
   - Signal PMU via mailbox → PMU runs ACR → FECS/GPCCS boot

3. **Tool**: Create `sovereign_acr_boot` that orchestrates the full
   SBR → PMC → WPR → ACR → GR chain

### Resources available:
- PMU firmware: `data/firmware/gv100_nvidia470/{pmu_imem,pmu_dmem}.bin`
- ACR firmware: `/lib/firmware/nvidia/gv100/{acr,sec2,gr}/*`
- mmiotrace: 792K recipe steps (includes PMU mailbox writes)
- SBR tool: `sovereign_pmu_dmatrf.c` (working SBR + PMC_ENABLE sequence)
- VM: `titanv-warmhandoff` for live nvidia-470 reference state
