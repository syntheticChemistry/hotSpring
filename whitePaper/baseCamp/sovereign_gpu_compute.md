# baseCamp: Sovereign GPU Compute — GlowPlug & Falcon Boot Chain

**Date:** 2026-03-21 (updated from 2026-03-16)  
**Domain:** Hardware — PCIe GPU lifecycle, falcon microcontrollers, HBM2 management, PFIFO command submission  
**Experiments:** 060-071  
**Hardware:** 2× NVIDIA Titan V (GV100, 12GB HBM2), RTX 5060 (display), Radeon VII (MI50/GFX906), BrainChip Akida AKD1000 NPU

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

## Sovereign Pipeline Layer Status (March 21, 2026 — Exp 071)

| Layer | Component | Status |
|-------|-----------|--------|
| 0 | PCIe / VFIO | ✅ BAR0 MMIO, DMA buffers, IOMMU |
| 1 | PFB / MMU | ✅ Alive via warm-state transfer from nouveau |
| 2 | PFIFO Engine | ✅ Re-initialized (PMC reset + soft enable + preempt) |
| 3 | Scheduler | ✅ Processes runlists, BIT30 acknowledged |
| 4 | Channel | ✅ Accepted by scheduler (STATUS=PENDING) |
| 5 | PBDMA Context | ✅ GP_BASE, USERD, SIG loaded correctly |
| 6 | MMU Translation | ❌ **BLOCKING** — 0xbad00200 PBUS timeout |
| 7-10 | GPFIFO→Commands→FECS→Shader | Blocked by Layer 6 |

**6 of 10 layers proven working** via 54-configuration diagnostic matrix.

## Remaining Blockers

1. **MMU page table translation** (PBDMA 0xbad00200 fetching GPU VA 0x1000) — the single remaining command submission blocker
2. **GPCCS address unknown** on GV100 (not at legacy 0x41A000)
3. **FECS halts at PC=0x2835** — likely waiting for GPCCS or a channel context
4. **DMA requires instance block** (SEC2+0x480) which is not host-writable in clean state
5. **PMC toggle of GR (bit 12) causes fatal unrecoverable PRIVRING fault** on GV100

### Resolved Since Last Update

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

### GCN5 E2E Breakthrough + Preswap Validation (March 2026)

**Full WGSL → coral-reef → coral-driver PM4 → MI50 → readback pipeline verified.**

Initial E2E: test shader writes `42.0f` to each thread's output slot, all 64 elements
correct. Extended preswap validation: **f64 write PASS, f64 arithmetic (6.0×7.0=42.0)
PASS, multi-workgroup dispatch PASS**. 12 GCN5 encoding bugs found and fixed total.

**GLOBAL_LOAD blocker:** Every `GLOBAL_LOAD` variant (GLC, FLAT, SADDR, load-only,
GTT/VRAM, cache invalidation) causes GPU hang. The existing E2E test uses only
`GLOBAL_STORE` — this is the first `GLOBAL_LOAD` attempt through coral-driver.
Root cause likely: missing PM4 register configuration (Mesa/RADV sets registers
that coral-driver doesn't). Blocks: multi-buffer, LJ force dispatch, HBM2 bandwidth.

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

Next milestone: fix `GLOBAL_LOAD` (reference Mesa radeonsi compute dispatch path),
then dispatch the DF64 Lennard-Jones force kernel — the exact kernel that returns
zeros through Vulkan — and verify correct forces via DRM.

### Vendor Status

- **AMD** (`coral-driver::amd`): `AmdDevice` implements `ComputeDevice` with GEM
  buffer management, PM4 command construction, `DRM_AMDGPU_CS` submission, and
  fence sync. **Store-only dispatch verified on MI50** — GLOBAL_STORE works (f64
  write, f64 arithmetic, multi-workgroup). **GLOBAL_LOAD blocked** — all variants
  hang. 12 compiler/driver bugs fixed total.
- **NVIDIA** (`coral-driver::nv`): `NvDevice` implements new UAPI (`VM_INIT` →
  `VM_BIND` → `EXEC` + syncobj). Blocked on Titan V by missing PMU firmware for
  `CHANNEL_ALLOC`. K80 (Kepler, incoming) has no PMU requirement.

### Sovereign + DRM Pipeline Layer Status (updated)

| # | Layer | Sovereign VFIO | DRM Dispatch |
|---|-------|----------------|--------------|
| 1 | PCIe / VFIO binding | Done | N/A (kernel) |
| 2 | PFB / MMU warm state | Done | N/A (kernel) |
| 3 | PFIFO engine enable | Done | N/A (kernel) |
| 4 | Scheduler: runlist load | Done | N/A (kernel) |
| 5 | Channel context binding | Done | N/A (kernel) |
| 6 | PBDMA context load | Done | N/A (kernel) |
| 7 | **MMU page table translation** | **BLOCKED** | Kernel handles this |
| 8 | NOP GPFIFO fetch | Pending | **AMD: PASSED** |
| 9 | FECS/GPCCS firmware load | Pending | N/A (kernel) |
| 10 | Shader dispatch | Pending | **AMD: store-only PASSED** (64/64), GLOBAL_LOAD blocked. NVIDIA: PMU-blocked |

The DRM path offloads layers 1-7 and 9 to the kernel driver. AMD DRM dispatch
validated for store-only shaders (f64 write, f64 arith, multi-workgroup). GLOBAL_LOAD
hangs — likely missing PM4 register config. NVIDIA awaits K80 (no PMU needed).
