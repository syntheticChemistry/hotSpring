# baseCamp: Sovereign GPU Compute — GlowPlug & Falcon Boot Chain

**Date:** 2026-03-24  
**Domain:** Hardware — PCIe GPU lifecycle, falcon microcontrollers, HBM2 management, PFIFO command submission, ACR secure boot, WPR construction, cross-driver profiling, daemon resilience, BOOT0 auto-detection, PCIe FLR  
**Experiments:** 060-087  
**Hardware:** 2× NVIDIA Titan V (GV100, 12GB HBM2), RTX 5060 (display/validator)

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

## Sovereign Pipeline Layer Status (March 24, 2026 — Exp 087)

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
| 9 | FECS/GPCCS Full Boot (HS Mode Release) | ❌ **BLOCKED** — cpuctl=0x12 (HALTED+ALIAS_EN), not RUNNING |
| 10 | Shader Dispatch | Pending (depends on Layer 9) |

**8 of 10 layers proven working.** Layer 7 was resolved via nouveau source
analysis revealing 7 binding bugs (B1-B7), validated on both Titans. Layer 8
was resolved via byte-level WPR format analysis discovering 7 construction bugs
(W1-W7), validated on Titan #1 post-nouveau. Layer 9 is the active frontier.

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

## Remaining Blockers

1. **Layer 9: FECS/GPCCS HALTED instead of RUNNING** — ACR bootstraps both falcons to cpuctl=0x12 (HALTED+ALIAS_EN) but they don't reach RUNNING (0x00). mb0=1 return code needs protocol research. Possible causes: missing FECS start command, HS mode page tables, BL halt-and-wait protocol.

### Resolved (Exp 082-087 — March 24, 2026)

- ~~GR/FECS context loading (Layer 7)~~ — SEC2 falcon binding: 7 bugs (B1-B7) found via nouveau source analysis, bind_stat=5 on both Titans (Exp 083-085)
- ~~GPCCS address~~ — confirmed at 0x41A000 via register profiling (Exp 086)
- ~~DMA requires instance block~~ — `falcon_bind_context()` implements full 8-step nouveau bind sequence (Exp 085)
- ~~WPR/ACR payload (Layer 8)~~ — 7 WPR construction bugs (W1-W7) found and fixed. Critical: BL headers in WPR image (W1) + bl_imem_off=0 (W2). ACR now processes WPR and bootstraps FECS/GPCCS (Exp 087)
- ~~Cross-driver profiling~~ — Exp 086: both Titans profiled across vfio/nouveau/nvidia. Verdict: WPR is interface problem, not key+lock. Post-nouveau is optimal starting state.

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

### Sovereign + DRM Pipeline Layer Status (updated March 24, 2026)

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
| 9 | FECS/GPCCS full boot (HS mode release) | **BLOCKED** — HALTED (0x12) | N/A (kernel) |
| 10 | Shader dispatch | Pending | **AMD: 6/6 preswap PASSED** (f64 LJ force verified). NVIDIA: PMU-blocked |

The DRM path offloads layers 1-9 to the kernel driver. AMD DRM dispatch
fully validated: 6/6 preswap phases pass including f64 Lennard-Jones force
(Newton's 3rd law verified). 18 bugs fixed. NVIDIA awaits K80 (no PMU needed).

Sovereign VFIO advanced from 7/10 to **8/10 layers** with the Layer 7 falcon
binding breakthrough (Exp 085, B1-B7) and Layer 8 WPR construction fix (Exp 087,
W1-W7). Layer 9 (falcon halt release) is the active frontier.

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

**Layer 9 is the new frontier:** FECS/GPCCS reach HALTED but not RUNNING.
