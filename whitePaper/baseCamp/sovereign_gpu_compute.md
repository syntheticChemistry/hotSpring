# baseCamp: Sovereign GPU Compute — GlowPlug & Falcon Boot Chain

**Date:** 2026-03-25 (updated 2026-04-27 — **RTX 5060 sovereign dispatch LIVE**, K80 init.rs split into 11 modules, IPC dedup, GPU solve tighten/refactor complete)  
**Domain:** Hardware — PCIe GPU lifecycle, falcon microcontrollers, HBM2 management, PFIFO command submission, ACR secure boot, WPR construction, cross-driver profiling, daemon RPC orchestration, adaptive experiment loop, sysmem DMA, GV100 MMU v2 page tables, WPR2 hardware protection, Kepler PIO falcon loading, VBIOS DEVINIT, fault containment architecture, **firmware-agnostic interfacing, PMU mailbox protocol, DRM ioctl sovereign pipeline, SM70 SASS compute dispatch, fork-isolated MMIO gateway, staged sovereign init, PCI remove/rescan with kernel override handling**  
**Experiments:** 060-176  
**Hardware:** NVIDIA Titan V (GV100, 12GB HBM2), 2× Tesla K80 (GK210, Kepler), RTX 5060 (GB206, Blackwell, display/validator)

---

## The Problem

Modern GPUs require vendor-signed firmware to activate their compute engines.
NVIDIA's Volta architecture (GV100) uses a chain of "falcon" microcontrollers:

```
BIOS POST → SEC2 (HS) → ACR → FECS/GPCCS (LS) → GR Engine → Compute
```

Without this chain completing, the GPU has 12GB of HBM2 memory and a working PCIe
interface, but no ability to execute shaders. The open-source `nouveau` driver
attempts this chain but **fails silently on GV100** due to a register map mismatch
(PRIVRING TIMEOUT at 0x600 — a GM200-era register absent on Volta).

## What We Proved

### Phase 1: D3hot → D0 VRAM Recovery (Exp 060-062)

The BIOS trains HBM2 at boot. The training survives D3hot power state.
A single sysfs write (`echo on > power/control`) restores full 12GB HBM2
read/write access via VFIO without any driver. 24/26 hardware tests pass.

### Phase 2: GlowPlug Persistent PCIe Broker (Exp 063-065)

`coral-glowplug` — a systemd daemon that holds GPU file descriptors open
across driver hot-swaps:

- Personality system: VFIO, nouveau, amdgpu, nvidia-proprietary, unbound
- Hot-swap: vfio→nouveau 4.1s, nouveau→vfio 1.5s
- Health monitor: 9-domain probe, auto-D0 recovery, PRAMIN sentinel
- State vault: register snapshots preserved across swaps
- Socket API: ListDevices, Health, Swap, Status, Shutdown

### Phase 3: SEC2/ACR Boot Chain Analysis (Exp 066)

Mapped the complete GV100 boot chain from nouveau source:

| Falcon | BAR0 Address | Role | State |
|--------|-------------|------|-------|
| SEC2 | 0x087000 | Runs ACR (authenticates firmware) | HS — EMEM always writable |
| FECS | 0x409000 | Front-End Context Switch | LS — writable when SCTL clean |
| GPCCS | ??? | GPC Context Switch | LS — address TBD on GV100 |
| PMU | 0x10A000 | Power Management | HS — fully locked |

Root cause of nouveau failure: PRIVRING TIMEOUT at register 0x600.

### Phase 4: SEC2 EMEM Breakthrough (Exp 067)

- SEC2 EMEM is **always host-writable** (even in full HS lockdown)
- D3hot→D0 produces a "clean" falcon state (SCTL=0x3000) where IMEM/DMEM/BOOTVEC are writable
- BIOS POST state (SCTL=0x7021) keeps everything locked

Two falcon states discovered:

| Property | BIOS POST (0x7021) | D3hot Clean (0x3000) |
|----------|-------------------|---------------------|
| IMEM | Protected | Writable |
| DMEM | Protected | Writable |
| BOOTVEC | Protected | Writable |
| EMEM | Writable | Writable |
| Instance (0x480) | Set by BIOS | Not writable |

### Phase 5: FECS Direct Execution (Exp 068) — THE BREAKTHROUGH

**FECS firmware executes from host-loaded IMEM** on the clean falcon:

- Loaded 25,632 bytes of `fecs_inst.bin` + 4,788 bytes of `fecs_data.bin`
- FECS executed to PC=0x63EE (offset 25,582 of 25,632 bytes — nearly complete)
- LS security protection NOT enforced when SCTL bits 0,5 are clear
- ACR bootloader also confirmed executing (PC=0xFD00 reached)
- ACR firmware runs (PC=0x12, mailbox written)

**This bypasses the entire SEC2→ACR→FECS chain.**

### Phase 6: Boot Persistence + Shutdown Safety (Exp 069)

coral-glowplug upgraded to production-grade system daemon:

- **Systemd service**: `coral-glowplug.service` starts at boot, binds GPUs before display manager
- **IOMMU group handling**: Auto-binds companion audio devices to vfio-pci
- **Graceful shutdown**: Disables PCI reset_method, pins D0, snapshots registers, then drops VFIO fds

**Critical lesson — DRM render node fencing:**

When the oracle card booted on nouveau, desktop apps (Cursor, Xorg) opened `/dev/dri/renderD129`.
Unbinding nouveau during shutdown yanked the DRM device from under Cursor's GPU thread,
causing a kernel oops (`do_task_dead` / `rcu_note_context_switch`). Three consecutive reboots
produced the same panic.

**Fix**: Boot ALL non-display GPUs on vfio-pci. No nouveau render node = no desktop apps
grabbing it = clean shutdown. The `resurrect_hbm2()` function handles temporary nouveau
binding only when no DRM consumers exist.

## Sovereign Pipeline Layer Status (March 25, 2026 — Exp 104)

| Layer | Component | Status |
|-------|-----------|--------|
| 0 | PCIe / VFIO | ✅ BAR0 MMIO, DMA buffers, IOMMU (iommufd/cdev) |
| 1 | PFB / MMU | ✅ Alive via warm-state transfer from nouveau |
| 2 | PFIFO Engine | ✅ Re-initialized (PMC bit 8 reset + soft enable + preempt) |
| 3 | Scheduler | ✅ Processes runlists, BIT30 acknowledged |
| 4 | Channel | ✅ Accepted by scheduler (STATUS=PENDING) |
| 5 | PBDMA Context | ✅ GP_BASE, USERD, SIG loaded correctly |
| 6 | MMU Translation | ✅ **RESOLVED** (Exp 076) — Volta FBHUB fault buffer config |
| 7 | SEC2 Falcon Binding + DMA | ✅ **SOLVED** (Exp 085) — B1-B7 all fixed, bind_stat=5 |
| 8 | WPR/ACR Payload + FECS/GPCCS Boot | ✅ **SOLVED** (Exp 087) — W1-W7 fixed, ACR bootstraps falcons |
| 9 | FECS Boot + GR Register Init | ✅ **FIX APPLIED** (Exp 091) — ITFEN + INTR_ENABLE + BOOTVEC |
| 9.5 | ACR DMA + Page Table Format | ✅ **PDE FIX** (Exp 104) — GV100 MMU v2 16-byte PDE upper slot. ACR firmware alive |
| 10 | HS Authentication + FECS/GPCCS Bootstrap | 🔶 **CLOSE** — firmware runs fully in LS mode, HS transition pending |
| 11 | GR Context Init + Shader Dispatch | 🔓 **UNBLOCKED** when L10 HS authentication achieved — `fecs_method.rs` ready |

**10.5 of 11 layers solved or fix-applied.** ACR firmware is alive and running
after the Exp 104 PDE slot fix — 31 unique PCs, EMEM queues initialized, DMEM
intact. The remaining gate is HS authentication: the firmware runs in LS mode
(SCTL=0x3000) and needs to transition to HS (SCTL=0x3002) before it can
authenticate and bootstrap FECS/GPCCS.

### Layer 10 Root Cause (Exp 091 — BOOTVEC Discovery)

**EXCI register format:** `[31:16]=cause, [15:0]=PC_at_fault`
- `exci=0x00070000` → cause=0x0007, faultPC=0x0000
- `exci=0x08070000` → cause=0x0807, faultPC=0x0000

GPCCS faults at PC=0 because BOOTVEC is zero. Three-piece fix applied:

| # | Register | Value | Purpose |
|---|----------|-------|---------|
| 1 | GPCCS BOOTVEC (0x104) | 0x3400 | Firmware entry point (start_tag << 8) |
| 2 | GPCCS ITFEN (0x048) | 0x04 | Enable DMA/external interfaces |
| 3 | GPCCS INTR_ENABLE (0x00c) | 0xfc24 | Enable interrupt handling |

**Evidence:** 089b observed `bootvec=0x00000000` on GPCCS before any host code ran.
Firmware is at IMEM[0x3400] (from WPR W2 fix). `falcon_start_cpu()` never wrote
BOOTVEC. Fix adds conditional write before STARTCPU in `strategy_mailbox.rs`.

**10 of 11 layers proven working.** Layer 7 resolved via nouveau source analysis
(7 binding bugs B1-B7). Layer 8 resolved via WPR format analysis (7 construction
bugs W1-W7). Layer 9-10 resolved by BOOTVEC discovery + ITFEN + INTR_ENABLE
(Exp 091). Layer 11 is the active frontier: `fecs_method.rs` ready for GR context.

## Phase 8: Pre-PMU Hardening (Exp 077, March 23, 2026)

The "Sovereign Pipeline Debt Burndown" sprint resolved five failure modes
discovered during the PFIFO dispatch debugging sessions (Exp 071-076):

### 1. SM Mismatch — BOOT0 Auto-Detection

**Problem:** Tests defaulted to SM 86, writing GA102 firmware to GV100 and
irreversibly corrupting the GPU until reboot. **Fix:** `boot0_to_sm()` decodes
BOOT0 register to SM version. `open()`/`open_from_fds()` auto-detect (sm=0)
or validate against hardware. Mismatch → `DriverError::OpenFailed` before any
writes touch the GPU.

### 2. PFIFO Init Unification

**Problem:** Two divergent init paths (production vs diagnostic) with different
scheduler enable, PMC gating, PBDMA clearing, and settle times. **Fix:**
`PfifoInitConfig` struct parameterizes the init sequence. Both paths call
`init_pfifo_engine_with()` with different configs. Code drift eliminated.

### 3. Architecture-Aware Diagnostic Matrix

**Problem:** Diagnostic matrix was GV100-hardcoded. **Fix:** `GpuCapabilities`
(from BOOT0) added to `ExperimentContext`. `ExperimentConfig::requires_sm`
gates experiments to specific architectures. `ExperimentResult::detected_sm`
records the SM for cross-run comparison.

### 4. PFIFO Liveness Probe

**Problem:** `PFIFO_ENABLE` (0x2200) reads 0 on GV100 even when functional,
flooding logs with 12+ false warnings per channel create. **Fix:** Single
post-init liveness probe (runlist preempt ACK). Legacy readback downgraded to
debug. One warning only if the probe itself fails.

### 5. PCIe Function Level Reset

**Problem:** Corrupted GPU requires reboot. **Fix:** `VfioDevice::reset()` →
`VFIO_DEVICE_RESET` ioctl. `DeviceSlot::reset_device()` in glowplug.
`coralctl reset <BDF>` CLI + `device.reset` RPC handler. Recovers from
firmware mismatch or init failure without reboot.

## Phase 7: D-State Resilient Ember + Swap Pipeline (Exp 074)

After MI50 was swapped for a second Titan V, the swap pipeline needed hardening.
Ember/glowplug had three failure modes:

1. **D-state sysfs writes**: `echo nouveau > driver_override` (or unbind/bind/remove/rescan)
   can enter uninterruptible kernel sleep. If the daemon's main thread does this, the
   entire IPC socket becomes unresponsive and the process cannot be killed.

2. **IOMMU group peers**: Audio devices (e.g. `03:00.1`) share the IOMMU group with
   the GPU. When swapping to nouveau, the audio peer must leave `vfio-pci` first, or
   nouveau cannot claim the group.

3. **Client socket errors**: EmberClient got `EAGAIN` from a single read and treated
   it as fatal, even though the swap might still be in progress.

**Solutions delivered:**

- **Process-isolated sysfs watchdog** (`guarded_sysfs_write`): Spawns `/bin/sh` child
  for each risky write. Parent polls with `try_wait()` + 10s timeout. D-state child
  is killed without blocking the daemon. Config-space writes (power/control,
  reset_method) use direct `std::fs::write` — no fork overhead.

- **Symmetric IOMMU peer handling**: `release_iommu_group_from_vfio()` unbinds peers
  and clears their `driver_override` before native swap. `bind_iommu_group_to_vfio()`
  re-binds peers when returning to vfio.

- **EmberClient retry**: 3 retries with 500ms×attempt backoff for `EAGAIN`/`EINTR`.
  `read_full_response()` loops until complete JSON line (newline delimiter).

- **DRM isolation auto-generation**: `drm_isolation.rs` generates udev rules and Xorg
  config from the device list at ember startup. Runs `udevadm control --reload-rules`
  when content changes.

- **MMU fault diagnostics**: `mmu_fault.rs` provides structured decoding of Volta+
  MMU fault registers. PRIV_RING 0xbad00100 identified as the actual sovereign
  pipeline blocker (not just MMU page tables).

**Validation**: nouveau ↔ vfio round-trip on Titan V #1 in 6s each direction. Both
Titans opened via iommufd/cdev at boot. Ember stayed in S-state (sleeping) throughout.

## Current Status (Exp 104 — March 25, 2026)

### Phase 12: ACR DMA Deep Dive (Exp 099-104)

After achieving HS mode in Exp 095, a persistent DMA trap at TRACEPC=0x0500
blocked ACR firmware from completing. Six experiments progressively eliminated
hypotheses:

| Exp | Hypothesis | Finding |
|-----|-----------|---------|
| 099 | Inherit nouveau firmware (skip ACR) | **Dead** — FLR wipes all falcon memory |
| 100 | IOMMU faults cause DMA trap | **Eliminated** — full IOVA coverage, trap persists |
| 101 | Page table location (VRAM vs sysmem) | VRAM PTs enable execution but prevent HS authentication |
| 102 | DMEM data loading, FBIF config, ctx_dma | DMA trap **invariant** to all five configurations |
| 103 | GPU memory controller, FLR state, IRQ | **Not FLR** — same trap with/without FLR. Halt is intentional. |
| **104** | **PDE slot position** | **ROOT CAUSE FOUND** — PDEs written to wrong 8-byte slot |

**Exp 104 Breakthrough:** GV100 MMU v2 uses 16-byte PDE entries. The directory
pointer goes in the UPPER 8 bytes (offset 8..16), lower 8 bytes zeroed. Our
`strategy_sysmem.rs` was writing all four PDE levels (PD3/PD2/PD1/PD0) to offset
0..8 — the wrong slot. After the fix:

- **31 unique trace PCs** (vs 2 before) — firmware runs through BL and ACR init
- **DMEM fully readable and correct** — ACR descriptor intact, BL descriptor valid
- **EMEM queues initialized** — firmware set up CMDQ/MSGQ
- **CPU alive at idle loop** (cpuctl=0x00000000, not halted)

### Remaining Blocker

**HS authentication not achieved** (SCTL=0x3000). The firmware runs extensively in
LS mode but never transitions to HS. With the old (wrong) PDEs, HS mode was
achieved but DMA trapped — the old PDEs created a "valid-looking" entry that
fooled the authenticator but broke the MMU walker. With correct PDEs, the MMU
walker works but authentication may be checking something different.

**Active investigation paths:**
1. WPR2 indexed register (0x100CD4) — firmware may validate WPR boundaries via this register, which doesn't reflect our direct writes
2. HS code authentication — the NS→HS transition may require specific WPR hardware state that only PMU or BIOS can configure

### Resolved (Exp 091-104 — March 25, 2026)

- ~~GPCCS PC=0 Fault (L10)~~ — Root cause: BOOTVEC=0. Triple fix: BOOTVEC=0x3400, ITFEN=0x04, INTR_ENABLE=0xfc24 (Exp 091)
- ~~Experiment infrastructure gaps~~ — Journal, observers, adaptive lifecycle, ring_meta (Exp 092)
- ~~DMA trap at TRACEPC=0x0500~~ — Root cause: PDE slot position in GV100 MMU v2 (Exp 104)
- ~~IOMMU fault coverage~~ — LOW_CATCH/HIGH_CATCH/mid-gap buffers (Exp 100)
- ~~FLR hypothesis~~ — Same crash with and without FLR (Exp 103)
- ~~FBHUB state~~ — No GPU MMU faults, FBHUB clean (Exp 103)
- ~~DMEM wipe mystery~~ — `0xDEAD5EC2` is HS read protection, not a wipe (Exp 102)

## Phase 10: Experiment Loop + Personality Sweep (Exp 092, March 25, 2026)

The "Wire Experiment Loop" sprint closed 7 infrastructure gaps:

### Self-Learning System

| Component | Purpose | Integration |
|-----------|---------|-------------|
| `SwapObservation` | Structured timing for every swap | Returned by ember RPC, stored in DeviceSlot |
| `ResetObservation` | Timing + success/failure for resets | Appended to journal from `ember.device_reset` |
| `Journal` | Persistent JSONL append/query/stats | Ember writes, coralctl reads, adaptive consumes |
| `AdaptiveLifecycle` | Wraps VendorLifecycle with learned data | Adjusts settle_secs from historical bind_ms, prunes reset methods by success rate |
| `DriverObserver` | Per-personality trace analysis | NouveauObserver (PRIV ring, PRAMIN, GSP), VfioObserver, NvidiaObserver, NvidiaOpenObserver |
| `RingMeta` | Mailbox/ring state serialization | Saved to ember before swap, restored after VFIO reacquire |
| `coralctl experiment sweep` | Automated personality characterization | Iterates targets, traces each, journals everything, prints comparison table |

### First Sweep Results (March 25, 2026)

Both Titan Vs swept through nouveau and nvidia-open:

| | Nouveau Bind | nvidia-open Bind | Trace Size | Cross-Card Variance |
|---|---|---|---|---|
| Titan V #1 | 21,084ms | 25,967ms | 1.4MB / 521B | — |
| Titan V #2 | 21,318ms | 25,959ms | 1.4MB / 521B | — |
| **Variance** | **234ms (1.1%)** | **8ms (0.03%)** | — | **Sub-1%** |

### Trace Analysis Discovery

nouveau GV100 mmiotrace (32,507 operations):
- **ZERO** writes to FECS (0x409xxx), GPCCS (0x41axxx), or SEC2 (0x840xxx)
- Entire GR init is firmware-driven through SEC2 DMA at 0x800000
- Visible phases: PRIV_RING topology → PMC_ENABLE → PRAMIN → PFB → GPC interrupts → SEC2 boot → PFIFO → Display → TLB flush
- nvidia-open trace is empty (521B header) — GSP handles everything
- **Warm-fecs second pass is 10× faster** (2.7s vs 21.9s)

### Resolved (Exp 082-088 — March 24, 2026)

- ~~GR/FECS context loading (Layer 7)~~ — SEC2 falcon binding: 7 bugs (B1-B7) found via nouveau source analysis, bind_stat=5 on both Titans (Exp 083-085)
- ~~GPCCS address~~ — confirmed at 0x41A000 via register profiling (Exp 086)
- ~~DMA requires instance block~~ — `falcon_bind_context()` implements full 8-step nouveau bind sequence (Exp 085)
- ~~WPR/ACR payload (Layer 8)~~ — 7 WPR construction bugs (W1-W7) found and fixed. Critical: BL headers in WPR image (W1) + bl_imem_off=0 (W2). ACR now processes WPR and bootstraps FECS/GPCCS (Exp 087)
- ~~Cross-driver profiling~~ — Exp 086: both Titans profiled across vfio/nouveau/nvidia. Verdict: WPR is interface problem, not key+lock. Post-nouveau is optimal starting state
- ~~FECS/GPCCS HALTED (Layer 9)~~ — Post-ACR STARTCPU sequence added (Exp 088). **PARTIAL:** cpuctl=0x00 was misleading (Exp 090) — GPCCS faults at PC=0x0000, FECS stuck at idle loop. PC/EXCI verification now required

### Resolved (Exp 060-081 — through March 23, 2026)

- ~~MMU page table translation (0xbad00200)~~ — Volta FBHUB fault buffer config (Exp 076)
- ~~SM mismatch corrupts GPU~~ — BOOT0 auto-detect + validation (Exp 077)
- ~~Divergent PFIFO init paths~~ — `PfifoInitConfig` unification (Exp 077)
- ~~PFIFO_ENABLE false warnings~~ — runlist preempt ACK liveness probe (Exp 077)
- ~~RAMFC GP_PUT=1 race~~ — empty ring init + post-submit doorbell (Exp 077)
- ~~False-positive MMU fault~~ — enable bit masking on fault buffer pointers (Exp 077)
- ~~No GPU reset without reboot~~ — `coralctl reset` PCIe FLR (Exp 077)
- ~~Ember D-state hangs~~ — process-isolated sysfs watchdog (Exp 074)
- ~~IOMMU group peer blocking~~ — symmetric bind/release for multi-device groups (Exp 074)
- ~~EmberClient EAGAIN fatal~~ — retry loop with backoff + full-response reader (Exp 074)
- ~~Manual DRM isolation rules~~ — auto-generated from device config at startup (Exp 074)
- ~~HBM2 not trained~~ — warm-state transfer from nouveau solves this
- ~~DRM consumer fencing~~ — VFIO-first boot + Xorg AutoAddGPU=false
- ~~AMD Vega metal stub~~ — AMD VendorLifecycle fully implemented (D3cold characterized)
- ~~SCM_RIGHTS fd passing~~ — Ember architecture delivers this
- ~~PFIFO disabled after nouveau unbind~~ — PMC reset + soft enable + preempt sequence

## Architecture Implications for coralReef

The GlowPlug daemon should orchestrate:

```
Boot → GlowPlug binds cards → D3hot→D0 cycle (clean falcons)
  → Set BIOS PMC_ENABLE (engines on, no GR toggle)
    → PIO-load FECS+GPCCS firmware into IMEM/DMEM
      → Start GPCCS then FECS
        → GR engine available for sovereign compute dispatch
```

For HBM2: use nouveau as oracle on one card to warm HBM2, then hot-swap to VFIO
with GlowPlug maintaining the state.

## Vendor-Agnostic Patterns

| Pattern | NVIDIA (GV100) | AMD (MI50, expected) |
|---------|---------------|---------------------|
| D3hot→D0 recovery | Proven | PCIe PM spec (should work) |
| HBM2 BIOS training | Survives D3hot | Likely similar |
| Falcon clean state | SCTL=0x3000 | Different micro-arch |
| Direct firmware load | FECS proven | Different falcon equiv |
| PRIVRING fault risk | GR bit 12 = fatal | Unknown |

The diagnostic matrix and GlowPlug should be vendor-agnostic. Per-vendor
knowledge lives in `coral-driver/src/nv/` and `coral-driver/src/amd/`.

## Reproducibility for Next GPU

| Step | Command / Action | Validates |
|------|-----------------|-----------|
| 1 | `cargo build --release -p coral-glowplug` | Binary compiles |
| 2 | Add BDF to `/etc/coralreef/glowplug.toml` with `boot_personality = "vfio"` | Config ready |
| 3 | `sudo systemctl restart coral-glowplug` | Device binds to vfio-pci, VRAM alive |
| 4 | `lsof /dev/dri/*` — no entries for new GPU | DRM consumer isolation |
| 5 | `lspci -ks {BDF}` — shows `vfio-pci` | Driver binding |
| 6 | Reboot → `systemctl status coral-glowplug` | Boot persistence |
| 7 | Shutdown → clean, no kernel oops | Graceful shutdown |

For AMD cards: `amd_metal.rs` stub must be implemented before VFIO BAR0
diagnostics work. The PCIe lifecycle (bind, health, shutdown) is vendor-agnostic.

## DRM Dispatch: The Parallel Track (Exp 072, March 21, 2026)

The sovereign VFIO path has 6/10 layers working but is blocked at MMU page table
translation (`0xbad00200`). In parallel, coralReef has **fully coded DRM dispatch
paths** for both vendors, with **AMD GCN5 now fully validated end-to-end**.

### GCN5 E2E Breakthrough + Complete Preswap Validation (March 2026)

**Full WGSL → coral-reef → coral-driver PM4 → MI50 → readback pipeline verified.**

Initial E2E: test shader writes `42.0f` to each thread's output slot, all 64 elements
correct. **Complete preswap validation: 6/6 phases PASS** — f64 write, f64 arithmetic,
multi-workgroup, multi-buffer read/write, HBM2 bandwidth streaming, **f64 Lennard-Jones
force calculation (Newton's 3rd law verified)**. 18 GCN5 encoding/compiler bugs found
and fixed total. 85 coral-reef unit tests pass.

Original 7 GCN5 bugs from initial bring-up:

1. PM4 wave size / VGPR granularity (GCN5=wave64, granularity 4)
2. VOP3 instruction prefix (110101→110100)
3. Missing `s_waitcnt vmcnt(0)` before `s_endpgm`
4. FLAT vs GLOBAL segment addressing (SEG=10 for compute)
5. Workgroup ID register file (SGPR, not VGPR)
6. Malformed ACQUIRE_MEM PM4 packet (missing POLL_INTERVAL dword)
7. VOP3-only opcode translation — GFX9 and RDNA2 use different values despite
   identical field layout. `vop3_only_opcode_for_gfx9()` provides LLVM-validated
   translation table

The key architectural discovery: VOP3a word-0 layout is identical on GFX9 and RDNA2
([31:26]=prefix, [25:16]=OP(10), [15]=CLAMP, [10:8]=ABS, [7:0]=VDST) but the opcode
VALUES differ. Group A (MAD/FMA/BFE/BFI, RDNA2 320-351) shifts by +128. Group B
(F64/MUL_HI, RDNA2 352+) has per-instruction mapping.

### Naga Bypass — Validated End-to-End

The DF64 Naga poisoning (Exp 055) breaks WGSL → SPIR-V → Vulkan dispatch for
double-precision transcendentals. DRM dispatch bypasses this entirely:

```
Broken:  WGSL → naga → SPIR-V → Vulkan driver → GPU  (all zeros)
Bypass:  WGSL → coral-reef → native ISA → coral-driver DRM → GPU  (64/64 correct ✓)
```

**GLOBAL_LOAD resolved and LJ force validated.** The DF64 Lennard-Jones force
kernel — the exact kernel that returns zeros through Vulkan — produces correct
forces via DRM. Next: K80 NVIDIA DRM, Titan V PMU investigation.

### Vendor Status

- **AMD** (`coral-driver::amd`): `AmdDevice` implements `ComputeDevice` with GEM
  buffer management, PM4 command construction, `DRM_AMDGPU_CS` submission, and
  fence sync. **Full preswap 6/6 PASS on MI50** — f64 write, f64 arithmetic,
  multi-workgroup, multi-buffer read/write, HBM2 bandwidth, f64 LJ force
  (Newton's 3rd law verified). 18 compiler/driver bugs fixed total.
- **NVIDIA** (`coral-driver::nv`): `NvDevice` implements new UAPI (`VM_INIT` →
  `VM_BIND` → `EXEC` + syncobj). Blocked on Titan V by missing PMU firmware for
  `CHANNEL_ALLOC`. K80 (Kepler, incoming) has no PMU requirement.

### Sovereign + DRM Pipeline Layer Status (updated March 25, 2026)

| # | Layer | Sovereign VFIO | DRM Dispatch |
|---|-------|----------------|--------------|
| 1 | PCIe / VFIO binding | Done | N/A (kernel) |
| 2 | PFB / MMU warm state | Done | N/A (kernel) |
| 3 | PFIFO engine enable | Done | N/A (kernel) |
| 4 | Scheduler: runlist load | Done | N/A (kernel) |
| 5 | Channel context binding | Done | N/A (kernel) |
| 6 | PBDMA context load | Done | N/A (kernel) |
| 7 | SEC2 falcon binding + DMA | **Done** (Exp 085) | Kernel handles this |
| 8 | WPR/ACR payload + FECS/GPCCS boot | **Done** (Exp 087) | N/A (kernel) |
| 9 | Page table format + ACR DMA | **Done** (Exp 104) — PDE slot fix, firmware alive | N/A (kernel) |
| 9.5 | HS authentication | **CLOSE** — firmware runs in LS mode, HS transition pending | N/A (kernel) |
| 10 | GR context init + shader dispatch | Awaiting HS authentication | **AMD: 6/6 preswap PASSED** (f64 LJ force verified). NVIDIA: PMU-blocked |

The DRM path offloads layers 1-9 to the kernel driver. AMD DRM dispatch
fully validated: 6/6 preswap phases pass including f64 Lennard-Jones force
(Newton's 3rd law verified). 18 bugs fixed. NVIDIA awaits K80 (no PMU needed).

Sovereign VFIO advanced to **10.5/11 layers** with the critical Exp 104 PDE
slot fix. ACR firmware is alive and running (31 trace PCs, EMEM queues
initialized). The sole remaining gate is HS authentication — the NS→HS
transition that unlocks FECS/GPCCS bootstrap.

### Phase 7: Layer 7 Assault — Falcon Diagnostics + ACR Boot Solver (Exp 078-081)

**Exp 078 (Diagnostic Matrix):** Comprehensive falcon state capture confirmed
FECS/GPCCS in HRESET — sole Layer 7 blocker. FECS `HWCFG.SECURITY_MODE = 0`.

**Exp 079 (Warm Handoff):** nouveau teardown halts falcons before unbind —
FECS IMEM does not survive swap. Infrastructure verified but approach failed.

**Exp 080 (Direct FECS Boot):** Direct IMEM upload succeeds but HS ROM shadows
IMEM and validates before releasing HRESET. ACR-managed boot required.

**Exp 081 (Falcon Boot Solver):** Multi-strategy SEC2→ACR→FECS boot chain.
SEC2 base corrected (`0x87000`), EMEM PIO verified, `nvfw_bin_hdr` decoded,
CPUCTL v4+ bits fixed, DMA context index corrected (PC advancing through HS ROM).
Full Nouveau-style PMC reset cycle implemented.

### Phase 9: Layer 7+8 Breakthrough — Falcon Binding + WPR Fix (Exp 082-087)

**Exp 082 (Multi-Backend Oracle Campaign):** Cross-card register profiling
infrastructure and oracle domain diff tooling built. Legacy NVIDIA closed-source
headers harvested for reverse engineering reference.

**Exp 083 (Nouveau Source Analysis):** Deep dive into upstream
`nvkm/falcon/gm200.c` revealed 4 bugs in coralReef's falcon binding:

| Bug | Description |
|-----|-------------|
| B1 | Wrong register offset: 0x668 → 0x054 |
| B2 | Missing bit 30 (enable flag) in bind_inst write |
| B3 | Wrong SYS_MEM_COH_TARGET: 3 (non-coherent) → 2 (coherent) |
| B4 | Missing DMAIDX clear before bind |

**Exp 084 (B1-B4 Hardware Validation):** All four bugs fixed. Register writes
accepted and read back correctly. But bind_stat at 0x0dc stays at 0 — binding
mechanism doesn't activate. Reveals missing trigger writes.

**Exp 085 (B5-B7 Bind Trigger Breakthrough — Layer 7 SOLVED):** Cross-driver
source analysis (nouveau, nvidia-open, Mesa) revealed three additional missing
steps in the falcon bind sequence:

| Bug | Register | Description |
|-----|----------|-------------|
| B5 | UNK090 (0x090) | Bit 16 — required post-bind trigger |
| B6 | ENG_CONTROL (0x0a4) | Bit 3 — engine activation |
| B7 | CHANNEL_TRIGGER (0x058) | Bit 1 (LOAD) — channel activation + INTR_ACK bit 3 clear |

New `falcon_bind_context()` helper encapsulates full 8-step nouveau bind
sequence. **bind_stat reaches 5 on both Titans.** SEC2 DMA active.

**Exp 086 (Cross-Driver Falcon Profile):** BAR0 sysfs mmap profiler captured
falcon register state across 12 configurations (vfio-cold, nouveau-warm,
nvidia-warm, post-warmup × 2 Titans). Key findings:

- **WPR is an INTERFACE problem**, not a key+lock hardware security gate
- **nvidia is destructive** — teardown powers down most engines
- **nouveau is the Rosetta Stone** — reveals correct SEC2 configuration
- **Both Titans are functionally identical** — only 4 registers differ
- **Post-nouveau is optimal starting state** for sovereign boot

**Exp 087 (WPR Format Analysis — Layer 8 SOLVED):** Byte-level comparison
of `build_wpr()` against nouveau's `gp102_acr_wpr_build` revealed 7 WPR
construction bugs:

| Bug | Severity | Description |
|-----|----------|-------------|
| W1 | **CRITICAL** | BL file headers (64B) included in WPR image — shifts all offsets |
| W2 | **CRITICAL** | bl_imem_off=0 — should be start_tag<<8 (FECS=0x7E00, GPCCS=0x3400) |
| W3 | MEDIUM | bl_code_size includes headers (576 vs 512) |
| W4 | MEDIUM | BLD DMA offset uses wrong BL size |
| W5 | MINOR | bl_data_size=256 instead of 84 (sizeof flcn_bl_dmem_desc_v2) |
| W6 | MINOR | bin_version=0 instead of reading from sig file (version=2) |
| W7 | MINOR | Depmap area corruption in signature |

All 7 fixes applied to `firmware.rs` + `wpr.rs`. Hardware validated on
Titan #1 post-nouveau: ACR now processes WPR, acknowledges BOOTSTRAP_FALCON
commands, and bootstraps FECS/GPCCS to cpuctl=0x12 (HALTED+ALIAS_EN).

**Exp 088 (Layer 9 Falcon Start — Layer 9 SOLVED):** Nouveau's
`gf100_gr_init_ctxctl_ext` shows that `BOOTSTRAP_FALCON` alone does not start
the falcons. After ACR returns, the host must:

1. Clear status registers: `0x409800`, `0x41a10c`, `0x40910c`
2. `nvkm_falcon_start(GPCCS)` — GPCCS first (FECS depends on it)
3. `nvkm_falcon_start(FECS)` — FECS second
4. Poll `0x409800` bit 0 for FECS ready

Added post-ACR falcon start to `strategy_mailbox.rs`. Hardware showed both
falcons transition cpuctl from 0x12 to 0x00. **However, Exp 090 revealed
cpuctl=0x00 was misleading** — GPCCS faults at PC=0x0000 (exci=0x08070000) and
FECS is stuck at an idle loop (PC=0x023c). All FECS methods timeout. Diagnostic
infrastructure now includes PC/EXCI/BOOTVEC in `FalconProbe` and `AcrBootResult`.

**Layer 9/10 remain the active frontier:** GPCCS PC=0 fault must be resolved.

### Phase 10: SCTL Myth Busted + FalconCapabilityProbe + Deep Code Evolution (Exp 091+)

**SCTL Myth Busted:** The IMEMC register on GM200+ falcons uses **BIT(24)**
(`0x0100_0000`) for write auto-increment, not BIT(6). All previous manual PIO
tests used the wrong format, creating a false impression that SCTL blocks PIO.
PIO works regardless of security mode. This invalidated FLR attempts, SBR for
SCTL clearing, and warm handoff as PIO-motivated strategies.

**FalconCapabilityProbe:** Runtime bit solver in `falcon_capability.rs` that
discovers register layouts on actual hardware. Probes `IMEMC`/`DMEMC`/`EMEMC`
format, falcon version, security mode, CPUCTL layout. Makes PIO interface
portable across GPU generations. Pattern: probe → `FalconCapabilities` struct
→ `FalconPio` safe API.

**Deep Code Quality Sprint:**
- 60+ hardcoded BAR0 hex offsets → named constants in `registers.rs`
- 4 `unsafe` blocks eliminated via safe `DmaBuffer::volatile_write_u32/u64`
- `*mut u8` → `NonNull<u8>` in DMA buffers
- Shared boot helpers extracted: `poll_falcon_boot`, `dmem_nonzero_summary`
- Mock/placeholder language cleaned across 4 production files
- 511 lib tests pass, zero new unsafe

**Updated Layer Model:**

| Layer | Status | Discovery |
|-------|--------|-----------|
| L7 | **REGRESSION** (Exp 091e) | DMA fault (not SCTL). FBIF circular dep. PIO works |
| L10 | **ROOT CAUSE FOUND** | BOOTVEC=0 — GPCCS starts at IMEM[0] instead of [0x3400] |
| L11 | BLOCKED by L10 | FECS at idle loop (PC=0x023c), methods timeout |

**Remaining blocker:** DMA configuration (FBIF mode, FBHUB MMU) for SEC2, plus
BOOTVEC fix for GPCCS. Not security mode.

### Phase 11: W1 Header Fix + Path B Dead + Sysmem HS Breakthrough (Exp 093-095)

**Exp 093 — W1 Header Fix + BOOTVEC Metadata Wiring:**

Three interrelated bugs fixed in `coral-driver`:

1. **W1 Header Bug** — `fecs_boot.rs` loaded raw `*_bl.bin` files including
   `nvfw_bin_hdr` + `nvfw_hs_bl_desc` headers. BL now parsed through
   `GrBlFirmware::parse()` — extracts code section only.
2. **Wrong IMEM Layout** — Fixed to: inst at IMEM[0], BL at IMEM[bl_imem_off],
   BOOTVEC=bl_imem_off.
3. **Hardcoded BOOTVEC** — Replaced local constants with firmware-derived
   `GrBlFirmware::bl_imem_off()` via `FalconBootvecOffsets`.

Hardware validation confirmed the fix is mechanically correct (IMEM readback
verified), but FECS/GPCCS still fault at PC=0x0000 with exception 0x02070000.

**Exp 094 — Path B Dead (LS Mode Authentication):**

GV100 FECS/GPCCS are fuse-enforced LS mode (SCTL=0x3000). In LS mode, the
falcon's secure boot hardware rejects PIO-uploaded code at execution time —
exception 0x02 (instruction authentication failure). PIO writes produce valid
IMEM content (verified via readback) but are NOT authenticated. The only route
to sovereign compute on Volta is Path A: SEC2 ACR boot via DMA.

**Exp 095 — Sysmem HS Mode Breakthrough:**

After a nouveau cycle (VRAM recovery via DEVINIT), three ACR boot strategies
were tested. The critical discovery: **FBHUB is PRI-dead after VFIO takeover**.

```
FBHUB diagnostic: 0x100C2C = 0xbadf5040 (PRI error)
PRAMIN sentinel:  wrote=0xcafedead read=0xcafedead ok=true
```

PRAMIN writes to VRAM survive, but DMA reads through FBHUB are corrupted.
The BL cannot verify the HS signature from corrupted data → falls through to
LS mode. System memory DMA through the IOMMU is clean → HS mode achieved.

| Path | DMA Source | SCTL | HS Mode | Outcome |
|------|-----------|------|---------|---------|
| VRAM | VRAM via FBHUB | 0x3000 | NO | LS mode, deaf to commands |
| Hybrid | Sysmem PTEs for code, VRAM for WPR | 0x3000 | NO | Different trace, still LS |
| **Sysmem** | **System memory via IOMMU** | **0x3002** | **YES** | **HS mode**, then trapped on blob DMA |

**Fix:** `blob_size=0` in ACR descriptor skips the internal blob DMA that caused
the trap (EXCI=0x201f0000). WPR is pre-populated in the sysmem DMA buffer.

**Code infrastructure built:**
- `strategy_sysmem.rs`: blob_size=0 patch after `patch_acr_desc`
- `sysmem_iova.rs`: separated SHADOW (0x60000) from WPR (0x70000)
- `instance_block.rs`: `FALCON_PT0_VRAM` + `encode_sysmem_pte` public
- `dma.rs`: `DmaBuffer::new` public for test-level DMA allocation
- `mod.rs`: `dma_backend()` accessor on `NvVfioComputeDevice`

**Updated Layer Model (Exp 095):**

| Layer | Status | Discovery |
|-------|--------|-----------|
| L7 | **BREAKTHROUGH** | HS mode via sysmem DMA. FBHUB PRI-dead corrupts VRAM DMA |
| L10 | **CLOSE** | Sysmem ACR + blob_size=0 should bootstrap FECS/GPCCS |
| L11 | BLOCKED by L10 | FECS methods already implemented in `fecs_method.rs` |

**Architectural conclusion:** On GV100 after VFIO takeover, all DMA must route
through system memory. VRAM can be pre-populated via PRAMIN for WPR hardware
protection, but the ACR's DMA engine must read from system memory.

---

## Phase 8: Consolidation Matrix Through Definitive Root Cause (Exp 110-122)

**Exp 110 (Consolidation Matrix):** 12-combination sweep of 6 ACR boot variables.
PDE slot position is the **SOLE determinant** of HS mode. All other variables zero effect.

**Exp 112 (Dual-Phase Boot):** HS mode achieved (SCTL=0x3002) via legacy PDE → HS auth
→ immediate PDE hot-swap. Zero MMU faults. Firmware TRAPs on DMA (cause=0x20).

**Exp 113 (TRAP Analysis):** All 5 variants trap identically. PMU dependency confirmed.
BL's authenticated code path requires PMU-initialized WPR2.

**Exp 114-121 (WPR Copy Stall Arc):** Seven experiments probing the persistent WPR copy
stall. LS mailbox (114), direct PIO (115, HW-blocked), blob_size (116), WPR2 tracking
(117: valid at 12GB during nouveau), WPR2 preservation (118: impossible), cold boot
(119: invalid), sovereign DEVINIT (120: not needed), minimal ACR (121: same stall).

**Exp 122 (WPR2 Resolution — Definitive Root Cause):**
- **122A:** ALL WPR2 registers hardware-locked. Host cannot write.
- **122B:** WPR2 at ~12GB VRAM. FBPA partitions offline. FECS/GPCCS HRESET even under nouveau.
- **122C:** FWSEC not in accessible VBIOS PROM. Loaded by GPU internal ROM.

Root cause chain: FWSEC (inaccessible) → WPR2 (12GB, HW-locked) → driver swap (destroys) → FBPA (offline) → ACR firmware cannot write to WPR2 → persistent copy stall.

---

## Phase 9: K80 Strategy — Kepler Bypasses Security Entirely (Exp 123)

Tesla K80 (GK210, Kepler, PCI 10de:102d) arriving 2026-03-26.
**Zero firmware security** — no FWSEC, no WPR2, no ACR, no signed firmware.

- FECS at 0x409000, GPCCS at 0x41A000 — same base addresses as Volta
- Direct PIO IMEM/DMEM upload with tag alignment (Falcon v1 protocol)
- Compute class: KEPLER_COMPUTE_B = 0xA1C0
- GF100-style 2-level page tables (40-bit DMA)
- Dual-GPU: two independent PCI devices per card

**Code built:**
- `nv::kepler_falcon` — PIO upload (IMEM/DMEM/start) with mock tests
- `nv::identity` — SM 35-37, BOOT0 0x0F0-0xFF, PCI 0x102D
- `exp123t_parasitic_probe` — sysfs BAR0 probe for any driver binding

**Parasitic probe results (2026-03-25):** Titan V #1 PMU in HS (0x3002, FWSEC-loaded)
but HRESET. FECS/GPCCS HALTED under vfio-pci. RTX 5070 confirmed as GB206 (BOOT0=0x1b6000a1).

**Updated Layer Model (Exp 122/123):**

| Layer | Titan V (GV100) | K80 (GK210) |
|-------|----------------|-------------|
| L1-L9 | ✅ SOLVED | ✅ Expected (no security barriers) |
| L10 | 🔴 ROOT CAUSE DEFINITIVE: WPR2 HW-locked, FWSEC inaccessible | ✅ N/A (no firmware security) |
| L11 | 🔓 Blocked by L10 | 🔓 **PRIORITY** — PIO boot → PFIFO → dispatch |

**If K80 sovereign compute works:** validates entire pipeline except security layer.
Titan V problem is precisely L10 only. Parasitic compute (sysfs BAR0 while nouveau
active) becomes the Titan V path after K80 validates the dispatch infrastructure.

---

## Phase 10: NVIDIA GPFIFO Pipeline Operational — RTX 3090 (2026-03-30)

**strandgate** (RTX 3090, SM86, driver 580.x GSP-RM) achieved full GPFIFO command
submission via coralReef's sovereign driver (`coral-driver`). This is the first time
the sovereign pipeline has successfully submitted and completed GPU work on a modern
NVIDIA GPU without any CUDA or nouveau involvement.

### Root Cause Analysis

The channel initialization was missing several steps that NVIDIA's 580.x GSP-RM
requires but are undocumented in open-source headers. Discovered via `LD_PRELOAD`
ioctl interception of CUDA's proprietary driver behavior:

| Issue | What Was Wrong | Fix |
|-------|---------------|-----|
| **Engine BIND** | Compute engine allocated but never bound to channel subchannel | `NV906F_CTRL_CMD_BIND` (0x906F0101) with `{h_engine, class, class, engine_type}` — 16-byte struct differs from 4-byte open-source header |
| **TSG Scheduling** | Schedule called on channel (0xA06F0103) — always INVALID_ARGUMENT | Call on TSG (0xA06C0101) instead — channel group scheduling |
| **Work Submit Token** | Using Kepler class (0xA06F0108) — NOT_SUPPORTED | Use Volta class (0xC36F0108) — returns correct hardware token |
| **VRAM USERD** | System memory USERD at 47-bit physical address — GPU DMA range exceeded | VRAM allocation with CUDA-matching flags (2 MiB, GPU_CACHEABLE, PAGE_SIZE_HUGE, KERNEL_PRIVILEGED) |
| **Error Notifier** | owner=client, type=0 | owner=device, type=NOTIFIER(13), CUDA-matching attrs |
| **RM_ALLOC struct** | 32-byte NVOS21_PARAMETERS | 48-byte NVOS64_PARAMETERS required on 580.x |
| **Context Share** | Missing entirely | FERMI_CONTEXT_SHARE_A (0x9067) under TSG, handle passed to channel |
| **GPFIFO Encoding** | push_buf_va >> 2 (incorrect bit shift) | Direct 4-byte-aligned VA in lower 42 bits |

### Correct NVIDIA Channel Initialization Sequence (580.x)

```
1. Root Client (0x0041)
2. Device (0x0080) under root
3. Subdevice (0x2080) under device
4. VOLTA_USERMODE_A (0xC461) under subdevice  ← doorbell register
5. FERMI_VASPACE_A (0x90F1) under device
6. VRAM allocation (0x0040) for USERD  ← 2 MiB, CUDA-matching flags
7. System memory (0x003E) for GPFIFO ring
8. Error notifier (0x003E, type=13, owner=device)
9. KEPLER_CHANNEL_GROUP_A (0xA06C) under device  ← TSG
10. FERMI_CONTEXT_SHARE_A (0x9067) under TSG
11. AMPERE_CHANNEL_GPFIFO_A (0xC56F) under TSG  ← with h_ctxshare, h_err, h_userd
12. AMPERE_COMPUTE_B (0xC7C0) under channel
13. NV906F_CTRL_CMD_BIND  ← bind compute engine to channel
14. NVA06C_CTRL_CMD_GPFIFO_SCHEDULE  ← enable TSG scheduling
15. GET_WORK_SUBMIT_TOKEN (0xC36F0108)  ← query doorbell token
16. Write GP_PUT to USERD, ring doorbell at USERMODE+0x90
```

### Impact on biomeGate Team (Titan V / Tesla K80)

This breakthrough directly benefits biomeGate's GPU cracking work:

- **Channel initialization sequence** is now correct for all Volta+ architectures — the BIND, TSG schedule, and context share patterns apply to Titan V and K80 equally
- **The 580.x GSP-RM quirks** (48-byte NVOS64, Volta-class WST, TSG scheduling) are driver-version issues, not architecture-specific — biomeGate's Titan V with the same driver will use identical code paths
- **VRAM USERD allocation** with proper flags eliminates the physical address range errors that blocked channel creation
- **Error notifier configuration** (type=13, owner=device) is a universal requirement discovered here
- **The remaining Titan V blocker is solely L10** (WPR2/FWSEC) — the dispatch infrastructure above L10 is now proven working

### AMD Sovereign Compute — Scratch/Local Memory Breakthrough (Exp 124, 2026-03-30)

Three-layer fix unlocks per-thread scratch memory on RDNA2 (RX 6950 XT, GFX10.3):

| Layer | Problem | Fix |
|-------|---------|-----|
| **Compiler** | `MemSpace::Local` emitted `SEG=GLOBAL` (2) | `encode_scratch_load/store` in `Rdna2Encoder` — `SEG=SCRATCH` (1) |
| **Driver** | No scratch buffer, no PM4 registers | GEM alloc + `COMPUTE_TMPRING_SIZE` + `COMPUTE_PGM_RSRC2.SCRATCH_EN` + `COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI` |
| **Hardware init** | amdgpu DRM CP does **not** auto-init `FLAT_SCRATCH` for compute IBs | Shader prolog: `S_MOV_B32 s11, va_lo` + `S_SETREG_B32 hwreg(FLAT_SCR_LO), s11` × 2 |

**Key discovery:** Unlike HSA/KFD (ROCm/HIP), the amdgpu DRM Command Processor
does not initialize `FLAT_SCRATCH_LO/HI` from `COMPUTE_DISPATCH_SCRATCH_BASE`.
The shader must set `HW_REG_FLAT_SCR_LO` (ID 20) and `HW_REG_FLAT_SCR_HI` (ID 21)
explicitly via a 24-byte prolog patched at dispatch time. Matches open ROCm #6030.

**Result:** `parity_hw_local_memory_f64` passes — `array<f64, 18>`, sum = 6.0.
**7/8 hardware parity tests pass.** 1672 unit tests pass (coral-driver 358 + coral-reef 1314).

This pattern is architecture-generic for GFX10+ and likely applies to GFX9 (MI50)
with adjusted HW_REG IDs — directly helps the biomeGate team's MI50/Vega work.

### AMD Sovereign Compiler Status

coralReef compiles all 24 QCD production shaders to native AMD GFX10.3 ISA:
- 19 standalone + 5 composite shaders → 59,992 bytes of native GPU machine code in 102ms
- 38/39 dispatch tests pass on AMD RDNA2
- 7/8 hardware parity tests pass (local memory now working)
- Remaining frontiers: EXEC masking for divergent wavefront control flow, Wilson plaquette QCD

### Test Results

- 358 coral-driver unit tests pass (0 failures)
- 1314 coral-reef unit tests pass (0 failures)
- 4/4 NVIDIA E2E tests pass (device open, alloc/free, sync, SM86 compilation)
- 7/8 coral-gpu hardware parity tests pass (Wilson plaquette: NVIDIA needs GR context, AMD needs EXEC mask)
- NOP smoke test confirms GPFIFO operational — GP_GET advances after doorbell ring


### Remaining Sovereign Pipeline Frontiers

| Frontier | Vendor | Blocker |
|----------|--------|---------|
| Wilson plaquette QCD dispatch | NVIDIA | GR context allocation via RM (SKEDCHECK05_LOCAL_MEMORY_TOTAL_SIZE) |
| Wilson plaquette QCD dispatch | AMD | EXEC masking for divergent wavefront control flow |
| Euler HLL f64 compilation | Both | SSARef LARGE_SIZE panic (allocator limit) |
| QMD v2.2 layout | NVIDIA | build_qmd_v21 uses Kepler-era positions — fix for Titan V/K80 |
| Full silicon saturation | Both | Tensor cores, RT cores, additional fixed-function units |

---

## Phase 11: VM-Based BAR0 Capture — K80 + Titan V (2026-03-29 → 03-30)

**Experiment 124 (biomeGate)** — VFIO passthrough VM captures yield complete VBIOS register recipes.

### Method

Pass GPUs into Ubuntu 24.04 VMs via VFIO, capture BAR0 at two points:
1. **Cold** (post-VBIOS, pre-driver) — reveals what BIOS POST programs
2. **Warm** (post-nvidia driver) — reveals what the driver adds

Using ember/glowplug IPC: `device.lend` → VM passthrough → capture → `device.reclaim`.

### Results

| Metric | K80 (GK210, nvidia-470) | Titan V (GV100, nvidia-535) |
|--------|------------------------|---------------------------|
| Cold BAR0 regs | 10,283 | 6,468 (15,363 raw, 8,895 BADF) |
| Driver delta | N/A | 255 regs |
| PCLOCK (PLL) | 64 | 99 |
| PGRAPH | 2,048 (full) | 0 (all falcon-gated) |
| SEC2/ACR visible | N/A | 5 ACR (rest BADF — falcon DMA) |
| Sovereign readiness | **HIGH** | **MEDIUM** |

### Key Finding: VBIOS Does 98%+ of GPU Init

The nvidia-535 driver changes only **255 registers** on top of the VBIOS recipe.
The VBIOS/BIOS POST sequence is the real initialization. For sovereign compute,
the VBIOS recipe IS the init sequence.

### Key Finding: Titan V Falcons Are Invisible Through BAR0

SEC2 and ACR registers return `0xBADF1100` (QEMU unmapped MMIO) in both cold
and warm snapshots. These falcon engines use **PRAMIN/falcon DMA windows**, not
direct BAR0 MMIO. To access them requires PRAMIN window mapping or host-level
IOMMU interception.

### Key Finding: PGRAPH Is the Architecture Divide

- **K80**: PGRAPH fully initialized by VBIOS (2,048 regs) — no falcon gates
- **Titan V**: PGRAPH entirely zero/BADF — requires FECS/GPCCS falcon authentication

### VM Configuration Lessons

- Titan V requires **UEFI without Secure Boot** (kernel lockdown blocks BAR mmap)
- Both IOMMU group functions (VGA + audio) must be passed through
- BIOS boot with VGA passthrough hangs (stuck in display init)
- In-guest `mmiotrace` is ineffective with VFIO (QEMU direct-maps BARs)

### Remaining Deep Debt

**Tier 1 — K80 Sovereign Cold Boot (ready now):**
- Build register replay in coral-driver
- Replay 10,283-reg GK210 recipe on cold K80
- FECS/GPCCS PIO upload → PFIFO → dispatch

**Tier 2 — Titan V Falcon Extraction:**
- Extract SEC2/ACR microcode from nvidia-535 package
- Map PRAMIN window for falcon DMEM/IMEM
- Resolve WPR copy stall (Exp 120 root cause)

**Tier 3 — Pipeline Integration:**
- Merge strandgate GPFIFO breakthrough with biomeGate's VFIO path
- AMD PM4 dispatch validation on 6950 XT (GFX10.3 codegen proven)
- Register replay as ember "personality"

---

## March 30, 2026 — Livepatch Strategy and Upstream Wiring

### Kernel Livepatch for Warm Handoff (Exp 125)

The warm handoff from nouveau to vfio-pci now uses a kernel livepatch module that NOPs four nouveau functions during teardown:

1. `nvkm_mc_reset` — prevents PMC engine-level reset (preserves IMEM/DMEM)
2. `gf100_gr_fini` — prevents GR engine teardown (FECS/GPCCS stay running)
3. `nvkm_falcon_fini` — prevents falcon CPU halt
4. `gk104_runl_commit` — prevents empty runlist submission (FECS self-reset)

The livepatch is dynamically controlled: disabled during nouveau init (all functions run normally), then enabled before teardown (all NOPs active). This prevents FECS from entering an unrecoverable HRESET state during channel teardown.

### Upstream Pipeline Complete

- **toadStool S168** landed `shader.dispatch` — the orchestration layer that delegates to coralReef's `compute.dispatch.execute`.
- **barraCuda Sprint 23** fixed the f64 precision pipeline — systematic correction across Bessel, Legendre, Hermite, Laguerre, PPPM.

### Validation Matrix

A new `specs/SOVEREIGN_VALIDATION_MATRIX.md` maps every pipeline layer (L1-L11) against dispatch paths (VFIO cold/warm, nouveau DRM, nvidia+UVM, NVK/wgpu, AMD DRM) and hardware substrates (Titan V, K80, RTX 5060, RX 6950 XT).

### DRM from Both Sides (Exp 126)

Four dispatch paths to Titan V compute:

- **VFIO warm handoff** — livepatch tested, FECS preserved but in idle-HALT (Exp 127)
- **nvidia-drm + UVM** — code-complete, pending on-site validation
- **nouveau DRM** — blocked by missing PMU firmware
- **NVK/wgpu** — proven for physics (4-tier QCD validated)

The nvidia+UVM path is valuable even if not sovereign: it reveals exactly how RM initializes FECS/GPCCS on Volta, informing our sovereign boot strategy.

### Warm FECS Dispatch Attack (Exp 127)

FECS firmware **survives** the nouveau→vfio-pci driver swap via livepatch — CPUCTL goes from `0xbadf1201` (dead) to `0x00000010` (halted), SCTL reads `0x00003000` (HS+), 23 engines powered. But FECS enters an idle HALT (`CPUCTL bit 4`) after nouveau's DRM close handler runs, and cannot be woken by host-side register writes (CPUCTL writes are ignored in HS+ mode). The problem shifted from **preservation** to **resumption**.

### Puzzle Box Matrix (Exp 128)

Two GPUs, related generations, solving the same fundamental problem from different angles:

**K80 (Kepler):** Full nvidia-470 recipe replay + FECS PIO boot + GPFIFO channel dispatch. No HS security barriers — validates dispatch infrastructure in isolation.

**Titan V (Volta):** Keepalive subprocess (hold DRM fd during swap), nvidia proprietary warm handoff (learn RM's FECS init), timing attack (catch FECS running before idle-halt), STOP_CTXSW method (freeze scheduling without halting).

**Cross-cutting:** FECS method enumeration (STOP/START_CTXSW), CPUCTL bit labeling fix (bit 4 = halted, bit 5 = stopped — was inverted), serde aliases for backward compatibility.

### Livepatch and GPU Toggles Wired Into Daemons (2026-03-30)

All GPU lifecycle control operations moved from shell scripts and `coralctl` into the `ember`/`glowplug` daemon layer as first-class JSON-RPC operations:

| RPC | Description |
|-----|-------------|
| `ember.livepatch.status` | Query loaded/enabled/transition state + patched functions |
| `ember.livepatch.enable` | `modprobe` + sysfs enable + transition polling (idempotent) |
| `ember.livepatch.disable` | Sysfs disable + transition polling (idempotent) |
| `ember.mmio.read` | BAR0 register read via mmap (replaces File::seek) |
| `ember.fecs.state` | Structured FECS register snapshot (CPUCTL, SCTL, PC, MB0/1, EXCI + derived booleans) |
| `device.warm_handoff` | Full orchestrated warm handoff (livepatch → swap → settle → poll FECS → swap back) |

Architectural wins: FECS register constants shared from `coral-driver::nv::bar0`, hex parsing consolidated into `coral-driver::parse_hex_u32`, `Bar0Access` DRY'd via `mmap_file` helper, `enrich_fecs_via_ember` uses typed booleans from the JSON response. 808 tests pass across all three crates.

---

## April 2, 2026 — Dual GPU Sovereign Boot + ACR Root Cause (Exp 132-141)

### Phase 12: Ember Diesel Engine + Kepler Compute (Exp 132-134)

**Exp 132 (Ember Frozen Warm Dispatch):** Evolved warm handoff to "diesel engine" pattern — glowplug orchestrates driver swap, ember holds VFIO fds and provides active intervention via `mmio.write`. STOP_CTXSW freezes FECS scheduling during swap. Accepts that PFIFO is destroyed by nouveau teardown and rebuilds it while FECS is frozen.

**Exp 133 (Kepler Sovereign Compute):** Correct Kepler compute dispatch path: QMD v1.7 (`cla1c0qmd.h`), Kepler-specific push buffer methods, `SEND_PCAS_A` at 0x02B4 (not Volta's 0x0D00), `SET_PROGRAM_REGION_A/B` for code address. Architecture-aware dispatch branching in `coral-driver`.

**Exp 134 (K80 Cold Boot Pipeline):** Single-command sovereign cold boot wired into `coralctl cold-boot <BDF> --recipe <path>` — D3cold→FECS-running without any vendor driver involvement.

### Phase 13: Dual GPU Sovereign Boot (Exp 135-136)

Both GPUs booted in parallel. Both hit fundamental barriers:

| GPU | What Worked | Blocker |
|-----|-------------|---------|
| **K80** (GK210) | Clock init (258 regs), DEVINIT (315 regs), FECS PIO upload, falcon running (CPUCTL=0x00) | PGRAPH CTXSW domain PRI-faults above 0x409504. VBIOS POST (memory training) required |
| **Titan V** (GV100) | SEC2 engine reset, ACR BL execution, DMA enabled | ACR HS authentication loop at PC 0x2d78 — never completes |

### Phase 14: SEC2 DMA Deep Dive (Exp 136-139)

Six progressive discoveries resolved all DMA issues:

| Discovery | Impact |
|-----------|--------|
| **FBIF locked in VIRT mode** | HS+ boot ROM sets FBIF_TRANSCFG=0x0100 for all DMA indices. Host writes blocked. Falcon always uses virtual addressing |
| **System memory page tables** | PT0 pages 384-389 rewritten as SYS_MEM_COH PTEs → ACR payload. New PT1 at VRAM 0x16000 → WPR buffer. All via PRAMIN |
| **ctx_dma = VIRT** | BL uses falcon MMU → our patched PTEs → IOMMU → system memory. PHYS_SYS_COH was wrong |
| **DMEM repair** | BL descriptor (84B) overwrites data_section[0..84] at DMEM@0. Repair after BL reads descriptor |
| **No warm-up STARTCPU** | `falcon_engine_reset` includes PMC reset → boot ROM = priming. Extra STARTCPU re-locks FBIF |
| **IOMMU clean** | No faults at ACR/WPR DMA addresses — DMA succeeds or is not attempted |

### Phase 15: Uncrashable GPU Safety Architecture (Exp 138, 140, Survivability Hardening Apr 7)

D-state root cause: sysfs writes to GPU power/driver files can enter uninterruptible kernel sleep if the GPU is in a bad state. Three-phase hardening:

**Phase 1 — Eliminate Critical Lockup Vectors:**
- **Fork-isolated MMIO everywhere** — preflight_check, low_level read/write/batch, post_swap_quiesce all run in expendable child processes
- **Zero-I/O recovery** — removed all parent-side sysfs writes (disable_bus_master) from fault/recovery/shutdown paths
- **`abort()` not `exit()`** — `std::process::exit` runs destructors that can stall; `abort()` terminates immediately
- **Removed `sync_all()` from trace** — disk flush before fork could stall on frozen NVMe

**Phase 2 — Harden Moderate Debt:**
- **Guarded sysfs everywhere** — `sysfs_write_direct` → `guarded_sysfs_write` with timeout, `guarded_sysfs_read` for power state queries
- **Non-blocking tracing** — critical fault paths use fire-and-forget background threads for tracing (prevents journald backpressure stalls)

**Phase 3 — Evolve Glowplug Resurrection:**
- **GPU warm cycle in resurrection** — glowplug performs nouveau bind/unbind before restarting ember, retraining GPU memory controller
- **`ember.warm_cycle` RPC** — live warm cycle without restarting ember (release → nouveau → unbind → reacquire)
- **FdVault integration** — periodic fd checkpoint keeps VFIO binding alive through ember death, enabling skip of warm cycle when vault has live fds

**Validated**: 8 consecutive exp145 runs — zero lockups. Cold VRAM (`0xbad0ac0X`) detected and reported as clean error. System stays responsive throughout.

### Phase 16: ACR HS Authentication Root Cause (Exp 141)

**ROOT CAUSE IDENTIFIED: Missing VBIOS DEVINIT**

The ACR Boot Loader executes, loads ACR code, and enters the HS authentication loop at PC 0x2d78 — but authentication never completes. After fixing all DMA/page-table issues, the persistent loop was traced to uninitialized security hardware:

1. **Recipe source**: Captured from nouveau's init (AFTER VBIOS POST)
2. **Missing**: The VBIOS DEVINIT scripts that run BEFORE nouveau
3. **VBIOS DEVINIT configures**: ROOT_PLL, NVPLL, MEMPLL, power sequencing, crypto engine, fuse access, memory controller calibration, clock domain routing
4. **Without it**: SEC2 crypto engine cannot verify HS signatures → auth loop retries and fails silently

**Evidence chain:**
- Strategy 12 (direct IMEM, DMA disabled): mb0=0x36 — ACR runs correctly but WPR read fails (no DMA)
- Strategy 7c (BL + DMA): mb0=0x00 — ACR passes WPR check (DMA works) but HS authentication fails
- IOMMU: no faults at DMA addresses — the DMA path is correct
- The loop is consistent across ALL DMA-enabled strategies → hardware-level issue, not software

### Updated Sovereign Pipeline Layer Status (April 7, 2026)

| Layer | Component | Status |
|-------|-----------|--------|
| 0 | PCIe / VFIO | ✅ BAR0 MMIO, DMA buffers, IOMMU (iommufd/cdev) |
| 1 | PFB / MMU | ✅ Alive via warm-state transfer from nouveau |
| 2 | PFIFO Engine | ✅ Re-initialized |
| 3 | Scheduler | ✅ Processes runlists |
| 4 | Channel | ✅ Accepted by scheduler |
| 5 | PBDMA Context | ✅ Loaded correctly |
| 6 | MMU Translation | ✅ Volta FBHUB fault buffer config |
| 7 | SEC2 Falcon Binding + DMA | ✅ bind_stat=5, DMA active, sysmem PTEs |
| 8 | WPR/ACR Payload | ✅ WPR format correct, ACR firmware alive |
| 9 | ACR DMA + Page Tables | ✅ PDE slot fix, FBIF VIRT mode, falcon MMU routing |
| 10 | HS Authentication | 🔴 **BLOCKED** — missing VBIOS DEVINIT (crypto engine uninitialized) |
| 11 | GR Context Init + Shader Dispatch | 🔓 Unblocked when L10 achieved |
| — | **Safety Infrastructure** | ✅ **Ember Survivability Hardening COMPLETE** — all MMIO fork-isolated, zero-I/O recovery, FdVault checkpoint, warm cycle resurrection. 8 consecutive fault runs survived. |

### Phase 17: Multi-Ember Fleet Architecture (2026-04-07)

**Architectural evolution**: Ember transformed from a monolithic process (one ember holds all GPUs) into a per-device fleet model:

- **Per-device ember**: Each GPU gets its own ember process (`coral-ember server --bdf 0000:03:00.0`), with per-device socket (`/run/coralreef/ember-{slug}.sock`)
- **Systemd template units**: `coral-ember@.service` spawns per-device instances; `coral-ember-standby@.service` spawns hot-standby pool
- **Glowplug fleet orchestrator**: `EmberFleet` manages N active + M standby embers with independent heartbeat/checkpoint/resurrection per device
- **Hot-standby pool**: Pre-spawned ember processes with no devices; receive `ember.adopt_device` RPC to take over a dead ember's device instantly
- **Fault-informed resurrection**: `FaultRecord` history drives strategy selection — `HotAdopt` (instant), `WarmThenRespawn`, `FullRecovery` (remove+rescan), `ColdRespawn`
- **Discovery file**: Glowplug writes `/tmp/biomeos/coral-ember-fleet.json` with BDF → socket mappings for external clients
- **Backward compatible**: Without `fleet_mode = true` in config, single-ember legacy mode preserved

**Key isolation gain**: K80 ember crash has zero impact on Titan V. Each GPU's fault domain is fully isolated.

**Config**: `[daemon] fleet_mode = true` and `standby_pool_size = 1` in `glowplug.toml`.

### Phase 18: Firmware Boundary — Architectural Pivot (Exp 159-163, 2026-04-07)

**The fundamental insight**: Falcon firmware (PMU, SEC2, FECS, GPCCS) is the GPU's internal operating system — to be **interfaced with**, not replaced. Previous phases attempted to replicate firmware functionality in the driver layer. This phase pivots to a firmware-agnostic interfacing approach, analogous to how toadStool interfaces with platform BIOS/UEFI.

**Three-layer delineation:**
- **Driver** (host, what we write): BAR0 MMIO, DMA buffers, channel structures, firmware mailbox communication
- **Firmware** (falcon processors): PMU (PRI gates, clocks, power), SEC2 (ACR trust root), FECS (GR scheduler), GPCCS (GPC context)
- **Hardware** (silicon): PBDMAs, PFIFO, copy engines, GR compute, GPU MMU, HBM2

**Key discoveries:**
1. **HBM2 training persists through nouveau warm-cycle** (Exp 159): `reset_method` clear on vfio-pci bind avoids FLR, preserving HBM2
2. **PFIFO scheduler is firmware-controlled** (Exp 163): Direct BAR0 writes to PFIFO_ENABLE/SCHED_EN are PRI-gated. Scheduler only responds to firmware commands.
3. **GV100 PMU uses register-based mailbox** (Exp 163): MBOX0/MBOX1 + IRQSSET, not queue-based RPC (Turing+). Queue offsets return 0xBADF5040.
4. **Hot-handoff channel injection works** (Exp 163): Channel 500 injected via PRAMIN alongside running nouveau — scheduler accepted it.
5. **NOP dispatch via DRM: SUCCEEDED** (Exp 163): Both raw C and pure Rust paths proven end-to-end.

**NOP dispatch pipeline (pure Rust):**
```
VM_INIT → CHANNEL_ALLOC(VOLTA_COMPUTE_A) → SYNCOBJ → GEM_NEW → VM_BIND → mmap → EXEC → SYNCOBJ_WAIT
```
New UAPI (kernel 6.6+) confirmed required and working on kernel 6.17 + GV100. Zero C, zero libc.

**`PmuInterface` struct created:** Encapsulates register-based mailbox protocol (MBOX0/MBOX1 + IRQSSET). Provides `attach()`, `mailbox_exchange()`, `poll_mbox0_bits()`, state probing. Firmware-agnostic — works with whatever PMU firmware is loaded.

**Scaling across GPU generations:**

| Era | Firmware Interface | Driver Role |
|-----|-------------------|-------------|
| Kepler (K80) | None — direct register writes | Full hardware control |
| Volta (GV100) | PMU mailbox + SEC2 ACR + FECS scheduling | Firmware interface + channel structures |
| Turing/Ampere | GSP RPC — host driver becomes thin RPC client | RPC message formatting |
| Hopper/Blackwell | GSP with extended offloaded functionality | Even thinner RPC layer |

Learning the Volta firmware interface is the foundation for ALL modern NVIDIA cards.

### Updated Sovereign Pipeline Layer Status (April 7, 2026)

| Layer | Component | Status |
|-------|-----------|--------|
| 0 | PCIe / VFIO | ✅ BAR0 MMIO, DMA buffers, IOMMU (iommufd/cdev) |
| 1 | PFB / MMU | ✅ Alive via warm-state transfer from nouveau |
| 2 | PFIFO Engine | ✅ Re-initialized (firmware-controlled) |
| 3 | Scheduler | ✅ Processes runlists (via firmware, not direct BAR0) |
| 4 | Channel | ✅ Accepted by scheduler (hot-handoff proven) |
| 5 | PBDMA Context | ✅ Loaded correctly |
| 6 | MMU Translation | ✅ Volta FBHUB fault buffer config |
| 7 | SEC2 Falcon Binding + DMA | ✅ bind_stat=5, DMA active, sysmem PTEs |
| 8 | WPR/ACR Payload | ✅ WPR format correct, ACR firmware alive |
| 9 | ACR DMA + Page Tables | ✅ PDE slot fix, FBIF VIRT mode, falcon MMU routing |
| 10 | HS Authentication | 🔴 **BLOCKED** (VFIO path) — firmware-agnostic DRM path bypasses this entirely |
| 11 | **NOP Dispatch via DRM** | ✅ **PROVEN** — pure Rust, new UAPI, SET_OBJECT(0xC3C0) on Titan V |
| — | **Firmware Interface** | ✅ `PmuInterface` struct, `FalconProbe`, register-based mailbox protocol |
| — | **Safety Infrastructure** | ✅ Ember Survivability Hardening COMPLETE |

### Phase 19: Full Compute Dispatch — PROVEN (Exp 164, 2026-04-08)

**5/5 E2E phases pass on Titan V via nouveau DRM.**

| Phase | Shader | Result |
|-------|--------|--------|
| A | f32 write (64 threads → 42.0) | PASS |
| B | f32 arithmetic (6×7=42) | PASS |
| C | Multi-workgroup (4×64=256 threads) | PASS |
| D | f64 write (64 threads → 42.0) | PASS |
| E | f64 Lennard-Jones (2-particle, Newton's 3rd law) | PASS |

**7 bugs found and fixed** between NOP dispatch (Exp 163) and full compute:

1. **NVIF object creation** — `CTXNOTVALID`: bare GPFIFO channel lacks GR context. Fixed with `DRM_NOUVEAU_NVIF` ioctl.
2. **Subchannel assignment** — `CLASS_SUBCH_MISMATCH`: compute (0xC3C0) must use subchannel 1, not 0 (reserved for GR 0xC397).
3. **NVC3C0 method offsets** — `ILLEGAL_MTHD`: all 5 compute methods had wrong offsets. Corrected from `clc3c0.h`.
4. **Local memory window** — `INVALID_BITFIELD`: 64-bit address exceeded 17-bit upper field. Corrected to 49-bit address.
5. **QMD v02_02 bitfields** — `SKED` errors: complete rewrite from `clc3c0qmd.h` (every field relocated from v2.1).
6. **SPH header offset** — `ILLEGAL_INSTR_ENCODING`: binary always has 128-byte header but dispatch used 80 for SM70.
7. **Phase E sign convention** — test comparison, not GPU: signs swapped in assertion.

**Dispatch pipeline:**
```
WGSL → coral-reef (SM70) → 128B SPH + SASS → NvDevice::dispatch()
  → GEM_NEW + VM_BIND (shader, CBUF desc, QMD)
  → Push: SET_OBJECT(sc1,0xC3C0) + INVALIDATE + LOCAL_MEM + SEND_PCAS
  → EXEC → SYNCOBJ_WAIT → readback
```

**Significance**: First end-to-end sovereign GPU compute — WGSL source to verified physics output on NVIDIA silicon, zero CUDA, zero C, pure Rust.

### Updated Sovereign Pipeline Layer Status (April 8, 2026)

| Layer | Component | Status |
|-------|-----------|--------|
| 0 | PCIe / VFIO | ✅ BAR0 MMIO, DMA buffers, IOMMU (iommufd/cdev) |
| 1 | PFB / MMU | ✅ Alive via warm-state transfer from nouveau |
| 2 | PFIFO Engine | ✅ Re-initialized (firmware-controlled) |
| 3 | Scheduler | ✅ Processes runlists (via firmware, not direct BAR0) |
| 4 | Channel | ✅ Accepted by scheduler (hot-handoff proven) |
| 5 | PBDMA Context | ✅ Loaded correctly |
| 6 | MMU Translation | ✅ Volta FBHUB fault buffer config |
| 7 | SEC2 Falcon Binding + DMA | ✅ bind_stat=5, DMA active, sysmem PTEs |
| 8 | WPR/ACR Payload | ✅ WPR format correct, ACR firmware alive |
| 9 | ACR DMA + Page Tables | ✅ PDE slot fix, FBIF VIRT mode, falcon MMU routing |
| 10 | HS Authentication | 🔴 **BLOCKED** (VFIO path) — firmware-agnostic DRM path bypasses this entirely |
| 11 | **Full Compute Dispatch via DRM** | ✅ **PROVEN** — f32, f64, multi-workgroup, Lennard-Jones. 5/5 phases pass. |
| — | **Firmware Interface** | ✅ `PmuInterface` struct, `FalconProbe`, register-based mailbox protocol |
| — | **Safety Infrastructure** | ✅ Ember Survivability Hardening COMPLETE |

### Phase 20: SovereignInit Pipeline — nouveau Replaced (Exp 165, 2026-04-08)

**Full pure Rust GPU initialization pipeline — zero nouveau, zero DRM.**

The `SovereignInit` orchestrator replaces nouveau's initialization subsystem by subsystem.
Firmware is treated as an **ingredient** (proprietary microcode loaded as blobs), not a subsystem to rebuild.

**8-Stage Pipeline:**

| Stage | Module | What It Does |
|-------|--------|-------------|
| 0 | HBM2 Training | Cold/warm detection. Cold boot: VBIOS DEVINIT interpreter trains HBM2/VRAM. Warm: skip. |
| 1 | PMC Engine Gating | Clock-gate engines via PMC registers. Matches nouveau `mc_init()`. |
| 2 | Topology Discovery | GPC/TPC/SM/FBP/PBDMA enumeration. Fuse reads + mask registers. |
| 3 | PFB Memory Controller | FBPA configuration, memory partitions, MMU setup. |
| 4 | Falcon Boot Chain | `FalconBootSolver` — 15 strategies to boot SEC2→ACR→FECS/GPCCS. `legacy-acr` feature flag gates older paths. |
| 5 | GR Engine Init | Standalone `apply_gr_bar0_init()` applies firmware register writes. FECS method probe validates responsiveness. |
| 6 | PFIFO Discovery | PBDMA/runlist enumeration, engine→runlist binding. |
| 7 | GR Context Setup | Optional. FECS exception config, context size discovery, DMA buffer allocation, bind + golden save. |

**Key Architectural Decisions:**

- **Firmware as ingredient**: Proprietary firmware blobs (PMU, SEC2, FECS, GPCCS) are loaded and executed by the hardware. We manage the loading, not the code inside.
- **`NvVfioComputeDevice::open_sovereign(bdf)`**: Single entry point. Takes a BDF string, runs all 8 stages, returns `(device, SovereignInitResult)`.
- **`SovereignInitResult`**: Structured reporting with `compute_ready()`, `diagnostic_summary()`, per-subsystem booleans.
- **GR init extraction**: `apply_gr_bar0_init`, `apply_nonctx_writes`, `apply_dynamic_gr_init` are standalone module-level functions, callable by both sovereign and DRM paths.
- **`TrainingBackend`**: Enum selecting HBM2 training method — VBIOS interpreter (current), differential replay (future), falcon upload (future).
- **`Hbm2Controller`**: Typestate pipeline (`Configured → Training → Trained`) enforcing correct sequencing.

**ember integration**: `ember.gpu.train_hbm2` RPC triggers HBM2 training via fork-isolated MMIO for crash safety.

**Test results**: 429 coral-driver tests pass. 171 coral-ember tests pass. `cargo check --features vfio` clean.

### Updated Sovereign Pipeline Layer Status (April 8, 2026)

| Layer | Component | Status |
|-------|-----------|--------|
| 0 | PCIe / VFIO | ✅ BAR0 MMIO, DMA buffers, IOMMU (iommufd/cdev) |
| 1 | PFB / MMU | ✅ Alive via warm-state transfer from nouveau |
| 2 | PFIFO Engine | ✅ Re-initialized (firmware-controlled) |
| 3 | Scheduler | ✅ Processes runlists (via firmware, not direct BAR0) |
| 4 | Channel | ✅ Accepted by scheduler (hot-handoff proven) |
| 5 | PBDMA Context | ✅ Loaded correctly |
| 6 | MMU Translation | ✅ Volta FBHUB fault buffer config |
| 7 | SEC2 Falcon Binding + DMA | ✅ bind_stat=5, DMA active, sysmem PTEs |
| 8 | WPR/ACR Payload | ✅ WPR format correct, ACR firmware alive |
| 9 | ACR DMA + Page Tables | ✅ PDE slot fix, FBIF VIRT mode, falcon MMU routing |
| 10 | HS Authentication | 🔴 **BLOCKED** (VFIO path) — bypassed by SovereignInit pipeline + DRM path |
| 11 | **Full Compute Dispatch via DRM** | ✅ **PROVEN** — f32, f64, multi-workgroup, Lennard-Jones. 5/5 phases pass. |
| 12 | **SovereignInit Pipeline** | ✅ **IMPLEMENTED** — 8-stage pure Rust init. `open_sovereign()`. GR context + golden save. |
| — | **Firmware Interface** | ✅ `PmuInterface`, `FalconProbe`, `FalconBootSolver` (15 strategies), `Hbm2Controller` |
| — | **Safety Infrastructure** | ✅ Ember Survivability Hardening COMPLETE. Fork-isolated HBM2 training. |

### Phase 21: Staged Sovereign Init — Ember Sacrificial Pattern (Exp 165+, 2026-04-12)

**Architectural principle: ember sacrifices rather than locks.**

Direct VFIO access from any process (including sovereign init examples) can cause PCIe bus hangs that freeze the entire system. Fork isolation kills the child process, but a hardware-level bus hang propagates through the AMD NBIO/data fabric and locks all CPU cores. The solution: all GPU interaction routes through ember, and ember runs each dangerous operation in a sacrificial child.

**Three hardening changes:**

1. **Zero-MMIO startup**: Ember startup opens VFIO fds but does NOT map BAR0 or run `post_swap_quiesce`. The device is held with zero hardware I/O. BAR0 mapping is deferred to explicit RPC calls. This eliminates the startup lockup vector entirely.

2. **Per-stage fork isolation**: The monolithic `SovereignInit::init_all()` previously ran inside a single fork child with a 60s timeout. Now each stage runs in its own fork-isolated child with a short timeout (3-15s). If a stage hangs, ember kills just that child and reports which stage failed:

| Stage | Timeout | What It Does |
|-------|---------|-------------|
| probe | 3s | Read-only: BOOT0, PMC_ENABLE, DEVINIT check |
| hbm2 | 10s | HBM2 training (cold only, auto-detected) |
| pmc | 5s | PMC engine ungating |
| topology | 5s | GPC/TPC/SM/FBP enumeration (read-only) |
| pfb | 5s | Memory controller configuration |
| pri_ring_reset | 5s | PRI ring fault drain + GR engine reset |
| falcons | 15s | SEC2 -> ACR -> FECS/GPCCS boot solver |
| gr | 10s | GR engine init |
| pfifo | 5s | PFIFO/PBDMA discovery |

3. **PRI ring reset stage**: After PMC ungating, stale engine state floods the PRI ring with faults, making falcon registers read as `0xbad00100`. A new `reset_pri_ring()` stage toggles the GR engine via PMC, enumerates the PRI ring master, and drains faults. This restored falcon register access.

**Hardware validation results (April 12, 2026):**

| Stage | Result | Detail |
|-------|--------|--------|
| probe | OK | PMC_ENABLE=0x40000020 (2 bits), warm |
| hbm2 | OK | DEVINIT done, VRAM alive |
| pmc | OK | 5 writes, engines ungated |
| topology | OK | 1 GPC, 16 SM, 12 FBP, 4 PBDMA |
| pfb | OK | Memory controller configured |
| pri_ring_reset | OK | 4 writes, PRI faults cleared |
| falcons | TIMEOUT | SEC2 started but hit trap (exci=0x041f0000) |

Stages 0-5 proven safe with system alive throughout. Falcon boot (stage 6) times out and ember sacrifices the child — no system lockup. The falcon trap is caused by FBP=0 after nouveau teardown (memory controller goes back to sleep when nouveau unbinds).

**Warm detection relaxed**: After nouveau teardown, PMC_ENABLE is minimal (2 bits) but HBM2 was trained. Changed from `count_ones() >= 4` to `pmc != 0` — any PMC bit means some initialization ran.

**coralctl updated**: `coralctl sovereign init` now displays per-stage results, including status (OK/FAIL/TIMEOUT/BLOCKED), write counts, topology detail, and halt location.

**Remaining gap**: Falcon boot needs active VRAM (FBP > 0) to upload firmware via PRAMIN. Nouveau teardown puts the memory controller to sleep. Two paths:
1. Capture the "hot" state before nouveau unbinds (FBP=12)
2. Wake the memory controller in the PFB stage via VBIOS DEVINIT interpreter

### Sovereign Pipeline Complete (April 16, 2026 — Exp 166–168)

**Warm handoff validated**: Full vfio→nouveau→vfio round-trip on Titan V. No D-state.
HBM2 training preserved across swap cycle. Three critical bugs fixed in ember:
- `AdaptiveLifecycle` delegation (forwarded `skip_sysfs_unbind` to inner lifecycle)
- `reset_method` permission error (best-effort write, not fatal)
- `vfio-pci.ids` kernel parameter (force-unbind wrong driver after PCI rescan)

**Fork-isolated MMIO gateway**: All BAR0 operations run in sacrificial child processes
via `rustix::runtime::kernel_fork`. Parent kills child on timeout. 11 new JSON-RPC
methods in ember (mmio.*, ember.sovereign.init, ember.devinit.*, ember.vbios.read).

**6-stage sovereign init**: bar0_probe → pmc_enable → hbm2_training → falcon_boot →
gr_init → verify. Uses existing HBM2 typestate controller + FECS boot infrastructure.
`SovereignInitResult` matches glowplug contract (all_ok, compute_ready, halted_at).

**908 tests** across coral-driver (680) + coral-ember (228), zero failures.

### Sovereign Compile Parity (April 18, 2026 — Exp 176)

**Full HMC pipeline compiles to native SASS on all 3 GPU generations**:
- SM35 (Kepler/K80): 10/10 shaders → native SASS
- SM70 (Volta/Titan V): 10/10 shaders → native SASS
- SM120 (Blackwell/RTX 5060): 10/10 shaders → native SASS

coralReef f64 transcendental lowering fixed — the `lower_f64_function` pass previously
skipped NVIDIA GPUs below SM70, leaving `f64rcp`/`f64exp2` ops in the IR which panicked
the SM32 encoder. Fix: removed the SM < 70 guard, added SM-aware `emit_iadd` (IAdd2
for Kepler, IAdd3 for Volta+) and `emit_shl_imm` (OpShl for Kepler, OpShf for Volta+).
All f64 transcendentals now lower via MUFU seed + Newton-Raphson sequences on all
NVIDIA generations.

Additionally, `as_imm_not_i20`/`as_imm_not_f20` in the IR source encoder now gracefully
fall back when source modifiers are attached to immediates (copy propagation artifact),
instead of asserting.

**QMD v5.0**: Blackwell (SM120+) uses a 384-byte QMD layout vs 256-byte for pre-Hopper.
Implemented in `coral-driver::nv::qmd::build_qmd_v50()`, dispatched via `build_qmd_for_sm()`.

**Vendor wgpu dispatch validated**: RTX 5060 via Vulkan — Wilson plaquette and sum
reduction run correctly through wgpu.

**validate_pure_gauge integration**: 16/16 checks pass including sovereign compile on
all three GPU generations.

**10 HMC pipeline shaders**: `wilson_plaquette_f64`, `sum_reduce_f64`,
`cg_compute_alpha_f64`, `su3_gauge_force_f64`, `metropolis_f64`,
`dirac_staggered_f64`, `staggered_fermion_force_f64`, `fermion_action_sum_f64`,
`hamiltonian_assembly_f64`, `cg_kernels_f64`.

### RTX 5060 Sovereign Dispatch LIVE (April 19, 2026 — Exp 176+)

**Full sovereign VFIO dispatch on RTX 5060 (SM120/Blackwell)**. Four hardware-level
bugs fixed in coralReef Iter 85:

| Bug | Fix |
|-----|-----|
| f64 division: `MUFU.RCP64H` returns 0 on SM120 | F2F(f64→f32) + MUFU.RCP + F2F(f32→f64) seed, 2 Newton-Raphson iterations |
| f64 sqrt: `MUFU.RSQ64H` returns 0 on SM120 | F2F(f64→f32) + MUFU.RSQ + F2F(f32→f64) seed, 2 Newton-Raphson iterations |
| `@builtin(num_workgroups)`: S2R NCTAID returns [0,0,0] on SM120 | LDC c[7][0/4/8] from driver constants CBUF |
| Semaphore fence ordering | SET_REPORT_SEMAPHORE on compute engine subchannel (not PBDMA) |

UVM write access: `gpu_mapping_type = 1` (ReadWriteAtomic). QMD v5.0 completeness:
GRID_RESUME fields, SM_CONFIG_SHARED_MEM_SIZE, QMD_GROUP_ID = 0x1f.

### GPU Solve Tighten and Refactor (April 27, 2026)

**coralReef coral-driver**: The monolithic `vfio_compute/init.rs` (5466 LOC) was split
into 11 focused modules, each under 1200 LOC:

| Module | LOC | Responsibility |
|--------|-----|---------------|
| `gr_bar0.rs` | 214 | Firmware-blob-driven BAR0 register writes |
| `warm_channel.rs` | 338 | Warm falcon restart and FECS channel init |
| `kepler_cold.rs` | 425 | Kepler cold-boot (PRI ring → clocks → FECS) |
| `kepler_warm.rs` | 1183 | Warm Kepler GR init |
| `kepler_recovery.rs` | 223 | Cold recovery after bus reset |
| `kepler_fecs_boot.rs` | 1687 | FECS/GPCCS firmware upload and boot |
| `pmu.rs` | 176 | PMU falcon firmware boot |
| `pgob.rs` | 174 | PGOB power gating control |
| `pri.rs` | 321 | PRI ring management |
| `quiesce.rs` | 66 | Engine quiesce before teardown |
| `vbios_devinit.rs` | 561 | VBIOS DEVINIT script interpreter |

Shared abstractions extracted:
- `write_kepler_hub_station_params()` deduplicated across 3 files
- `PGOB_POWER_STEPS` table deduplicated in `pgob.rs`
- Dead code removed: `kepler_pclock_pre_init`, `kepler_pri_station_probe`
- `kepler_csdata.rs`: visibility narrowed to `pub(crate)`, debug_assert for xfer==0
- `hardware_guard.rs`: named constants (`PMC_ENABLE`, `PGRAPH_BIT`, `DEAD_SENTINEL`)

**hotSpring barracuda**:
- IPC dedup: shared `primal_bridge::jsonrpc_request()` envelope builder
- GPU module DRY: `open_from_adapter_inner()`, `summarize_tiers()`, `finish()` extracted
- Experiment bin hygiene: 6 bins use `HOTSPRING_BDF` env var, exp154/158 cross-referenced
- `#[allow(dead_code)]` replaced with `#[expect(dead_code, reason="...")]`

Both repos: `cargo fmt` clean, `cargo clippy -- -W clippy::pedantic -W clippy::nursery` pass.

### Next Steps

1. **K80 FECS boot**: Complete internal firmware protocol — full 4096-byte FECS/GPCCS firmware capture, csdata loading verified, golden context save
2. **Titan V SEC2/ACR**: Resolve HS mode rejection on warm FECS (STARTCPU fails in secure mode). SBR hot reset via VFIO ioctl implemented.
3. **Three-generation sovereign dispatch**: RTX 5060 (DONE) → Titan V → K80 all running the same WGSL→SASS→dispatch pipeline
4. **Cross-vendor validation**: Run the same WGSL→compute pipeline on AMD and NVIDIA in the same session
5. **toadStool absorption**: Migrate ember's sovereign init into toadStool's hardware orchestration layer
