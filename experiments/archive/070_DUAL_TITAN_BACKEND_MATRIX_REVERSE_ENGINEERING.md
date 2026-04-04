# Experiment 070: Dual-Titan Full Backend Matrix for Sovereign Reverse Engineering

> **Fossil Record**: "Config Deployment" section uses `pkexec cp` — superseded by
> `coralctl deploy-udev --config glowplug.toml` and `sudo systemctl restart coral-glowplug`.

**Date**: March 18, 2026 (executed March 19, 2026)
**Status**: EXECUTED — Phases 1–4 complete, pipeline gap identified
**Hardware**: 2x Titan V (GV100, SM70) — `0000:03:00.0` (oracle), `0000:4a:00.0` (target)
**Prerequisite**: GlowPlug config updated (both on vfio at boot), Experiment 069
**Desktop GPU**: RTX 5060 on nvidia — never managed by GlowPlug, always bound

---

## Objective

Use both Titans under GlowPlug to systematically reverse engineer the remaining
sovereign dispatch blocker (cold silicon — FECS/GPCCS firmware never loaded under
pure VFIO boot) by warming each card with **every compatible backend** and
capturing the resulting register/firmware state.

coralReef iter57 proved the VFIO dispatch stack is software-complete (BAR0 MMIO,
DMA, GPFIFO, PFIFO channel, V2 MMU, fault buffers, runlist encoding). The blocker
is that FECS/GPCCS firmware is never loaded on a cold VFIO-bound device. By
warming with each backend and diffing the resulting state, we can identify the
minimal firmware initialization sequence the sovereign pipeline must replicate.

---

## Backend Inventory (Titan V / GV100 / SM70)

| Backend | Kernel Module | Userspace | Vulkan | Compute | Notes |
|---------|---------------|-----------|--------|---------|-------|
| **nouveau** | `nouveau` | Mesa GL | NVK 1.3.311 | wgpu via NVK | Open source; trains HBM2, loads FECS/GPCCS |
| **nvidia** | `nvidia` (580.126.18 open) | libEGL/libGLX | NVIDIA 1.4.312 | wgpu via proprietary Vulkan | Proprietary; full firmware load, NVVM risk on some f64 |
| **vfio-pci** | `vfio-pci` | none | none | none | Cold passthrough — baseline reference |
| **coralReef** | `vfio-pci` | coral-driver | none (future) | coralReef DRM (blocked) | Sovereign — needs firmware init solved |

---

## Full Backend Matrix (2 Titans x 5 Configurations)

| Config | Oracle (03:00.0) | Target (4a:00.0) | Purpose |
|--------|------------------|-------------------|---------|
| **A** | vfio | vfio | Cold baseline — reference register dump for both |
| **B** | nouveau | vfio | nouveau warm oracle — capture FECS/GPCCS/MMU/PFIFO state |
| **C** | nvidia | vfio | nvidia warm oracle — capture proprietary firmware state |
| **D** | vfio | nouveau | nouveau warm target — NVK compute validation |
| **E** | vfio | nvidia | nvidia warm target — proprietary Vulkan compute validation |

**Additional cross-driver configs (phase 2):**

| Config | Oracle (03:00.0) | Target (4a:00.0) | Purpose |
|--------|------------------|-------------------|---------|
| **F** | nouveau | nvidia | Cross-driver — validate both warm paths simultaneously |
| **G** | nvidia | nouveau | Cross-driver — reversed roles for symmetry check |
| **H** | nouveau | nouveau | Dual nouveau — dual-GPU NVK compute (Mesa NVK built) |

**Stability invariant**: Boot always in Config A (both vfio). Swap to other
configs only after boot. Never leave desktop GPU (RTX 5060) unbound. Return
to Config A before shutdown.

---

## Phase 1: Cold Baseline (Config A)

**Goal**: Capture the cold VFIO register state of both cards as the zero reference.

**Procedure**:
1. Boot: both vfio (Config A)
2. For each Titan, read BAR0 register regions via VFIO:
   - PMC (0x000000–0x001000): engine enables, boot status
   - PFIFO (0x002000–0x004000): scheduler, runlist, PBDMA state
   - FECS (0x409000–0x40A000): falcon state, PC, SCTL, IMEM
   - SEC2 (0x087000–0x088000): ACR state, EMEM, SCTL
   - GPCCS scan (0x400000–0x520000): scan for falcon BOOT0 signature
   - MMU fault buffers (0x100C80–0x100D00): buffer config
   - GPU_TEMP (0x020460): thermal
3. Write: `data/070/cold_oracle.json`, `data/070/cold_target.json`

**Output**: Register snapshots for both cold cards. Every subsequent warm state
diffs against these.

---

## Phase 2: nouveau Warm Oracle (Config B)

**Goal**: Capture what nouveau initializes — FECS/GPCCS firmware, MMU, PFIFO.

**Procedure**:
1. Start from Config A
2. Swap oracle 03:00.0 → nouveau via GlowPlug
3. Wait ~5s for HBM2 training, FECS/GPCCS firmware load
4. Read all register regions (same as Phase 1) from oracle via sysfs resource0
5. Optionally: mmiotrace oracle during bind (captures write sequence)
6. Write: `data/070/nouveau_warm_oracle.json`
7. Diff: `nouveau_warm_oracle.json` vs `cold_oracle.json`
8. Swap oracle 03:00.0 → vfio

**Key registers to capture**:
- FECS SCTL (0x409240): HS-locked (0x7021) vs clean (0x3000)?
- FECS PC (0x409110): where does firmware halt?
- GPCCS: scan 0x400000–0x520000 for newly-alive falcons
- PBDMA SIGNATURE (0x41010 etc.): 0xFACE if initialized?
- MMU fault buffer PUT/GET/SIZE registers
- Runlist registers (0x002270/0x002274): populated?

**Experiments feeding this**: 061 (mmiotrace), 062 (D3hot VRAM), 065 (GlowPlug).

---

## Phase 3: nvidia Warm Oracle (Config C)

**Goal**: Capture what the proprietary driver initializes — compare with nouveau.

**Procedure**:
1. Start from Config A
2. Swap oracle 03:00.0 → nvidia via GlowPlug
3. Wait ~3s for driver init (nvidia typically faster than nouveau HBM2 training)
4. Read all register regions from oracle via sysfs resource0 (if nvidia exposes it)
5. If BAR0 not readable via sysfs under nvidia: use /dev/nvidia-uvm or debugfs
6. Write: `data/070/nvidia_warm_oracle.json`
7. Diff: `nvidia_warm_oracle.json` vs `cold_oracle.json`
8. Diff: `nvidia_warm_oracle.json` vs `nouveau_warm_oracle.json`
9. Swap oracle 03:00.0 → vfio

**Critical comparison**:
The diff between nouveau-warm and nvidia-warm reveals which firmware writes
are driver-specific vs hardware-required. Registers that both drivers set to
the same value are almost certainly hardware requirements. Registers that
differ are driver policy choices.

---

## Phase 4: Warm → Rebind → Dispatch (Critical Experiment)

**Goal**: Determine if firmware state survives nouveau → vfio rebind under GlowPlug.

**Procedure**:
1. Start from Config A
2. Swap target 4a:00.0 → nouveau (warms GPU, loads firmware)
3. Read warm registers from target: `data/070/nouveau_warm_target.json`
4. Swap target 4a:00.0 → vfio (GlowPlug holds fd — no PM reset)
5. Read registers again via VFIO: `data/070/rebind_vfio_target.json`
6. Diff: `nouveau_warm_target.json` vs `rebind_vfio_target.json`
7. If FECS/GPCCS state survives: attempt coralReef dispatch on warm-then-rebound card

**This is the key experiment**: if firmware state persists across the rebind,
then the sovereign dispatch path is `resurrect → rebind → dispatch`. If it
doesn't persist, we need to identify exactly which writes are lost and replay
them from VFIO.

Repeat with nvidia:
8. Swap target 4a:00.0 → nvidia (warms GPU)
9. Read warm registers: `data/070/nvidia_warm_target.json`
10. Swap target 4a:00.0 → vfio
11. Read again: `data/070/nvidia_rebind_vfio_target.json`
12. Diff: identify which nvidia-initialized state survives

---

## Phase 5: NVK Compute Validation (Config D)

**Goal**: Prove hotSpring physics works on Titan V via wgpu/NVK.

**Procedure**:
1. Start from Config A
2. Swap target 4a:00.0 → nouveau
3. Build Mesa with NVK (if not already): `meson -Dvulkan-drivers=nouveau`
4. Run validation suite through NVK:

```bash
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin validate_cpu_gpu_parity
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin validate_barracuda_cpu_gpu_parity
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin bench_md_parity
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin validate_spectral
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin validate_linalg
```

5. Record: adapter name, precision tiers, throughput
6. Swap target → vfio

---

## Phase 6: nvidia Vulkan Compute Validation (Config E)

**Goal**: Prove hotSpring physics works on Titan V via wgpu/nvidia Vulkan.

**Procedure**:
1. Start from Config A
2. Swap target 4a:00.0 → nvidia
3. Run same validation suite as Phase 5 through nvidia Vulkan:

```bash
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin validate_cpu_gpu_parity
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin validate_barracuda_cpu_gpu_parity
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin bench_md_parity
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin validate_spectral
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin validate_linalg
```

4. Record: adapter name, precision tiers, throughput, any NVVM poisoning
5. Compare NVK vs nvidia results (Phase 5 vs Phase 6)
6. Swap target → vfio

**The Phase 5 vs Phase 6 comparison** reveals whether NVVM poisoning affects
any hotSpring shaders on the Titan V. If NVK and nvidia produce identical
results, the proprietary path is clean. If they diverge on specific shaders,
those shaders need `F64Precise` FMA policy via coralReef sovereign compilation.

---

## Phase 7: Firmware Diff Analysis

**Goal**: Synthesize all register dumps into a minimal firmware initialization map.

**Deliverable**: `data/070/FIRMWARE_INIT_MAP.md` containing:

1. **Common writes** — registers both nouveau and nvidia set identically
   (hardware requirements — coralReef must replicate these)
2. **Driver-specific writes** — registers that differ between nouveau and nvidia
   (driver policy — coralReef chooses its own values)
3. **Rebind survivors** — registers that persist across nouveau→vfio rebind
   (state that survives — no sovereign replay needed)
4. **Rebind casualties** — registers that are lost during rebind
   (state that must be replayed from VFIO after warm cycle)
5. **GPCCS discovery** — GPCCS falcon base address (from BAR0 scan or from
   register state appearing after warm)
6. **Minimal sovereign init sequence** — the ordered register writes that
   coralReef needs to add for cold → warm transition without any kernel driver

---

## Experiment Dependency Graph

```
Exp 058 (PBDMA) ──┬─── Exp 060 (BAR2) ─── Exp 062 (D3hot VRAM)
                  |
Exp 061 (mmiotrace) ── Exp 065 (GlowPlug) ─── Exp 069 (boot persistence)
                  |
Exp 066 (SEC2) ── Exp 067 (EMEM) ─── Exp 068 (FECS direct)
                  |
                  └─── Exp 070 (this)
                        ├── Phase 1: Cold baseline
                        ├── Phase 2: nouveau warm oracle
                        ├── Phase 3: nvidia warm oracle
                        ├── Phase 4: Warm → rebind → dispatch (KEY)
                        ├── Phase 5: NVK compute validation
                        ├── Phase 6: nvidia Vulkan compute validation
                        └── Phase 7: Firmware diff analysis → FIRMWARE_INIT_MAP
                              |
                              └── Exp 071: Sovereign dispatch attempt
```

---

## GlowPlug Swap Commands

```bash
# List devices
echo 'ListDevices' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock

# ── nouveau swaps ──
echo '{"Swap":{"bdf":"0000:03:00.0","target":"nouveau"}}' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock
echo '{"Swap":{"bdf":"0000:4a:00.0","target":"nouveau"}}' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock

# ── nvidia swaps ──
echo '{"Swap":{"bdf":"0000:03:00.0","target":"nvidia"}}' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock
echo '{"Swap":{"bdf":"0000:4a:00.0","target":"nvidia"}}' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock

# ── return to vfio ──
echo '{"Swap":{"bdf":"0000:03:00.0","target":"vfio"}}' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock
echo '{"Swap":{"bdf":"0000:4a:00.0","target":"vfio"}}' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock
```

**Note**: JSON-RPC 2.0 format when coral-glowplug redeployed: `device.swap` method.

---

## Config Deployment

```bash
pkexec cp /home/biomegate/Development/ecoPrimals/hotSpring/scripts/boot/glowplug.toml /etc/coralreef/glowplug.toml
pkexec systemctl restart coral-glowplug
echo 'ListDevices' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock
```

---

## Success Criteria

| Phase | Criterion |
|-------|-----------|
| 1 | Cold register snapshots for both cards written |
| 2 | nouveau warm registers captured, diff against cold shows FECS/GPCCS alive |
| 3 | nvidia warm registers captured, diff against cold and against nouveau |
| 4 | Rebind survival map: which firmware state persists across nouveau→vfio |
| 5 | NVK compute produces correct physics (all validation binaries pass) |
| 6 | nvidia Vulkan compute produces correct physics, NVVM poisoning map |
| 7 | FIRMWARE_INIT_MAP.md delivered with minimal sovereign init sequence |

---

## Expected Outcomes

**Best case**: Firmware state survives rebind. Sovereign dispatch = warm via
nouveau, rebind vfio, dispatch. This is a pragmatic sovereign path that uses
GlowPlug lifecycle management.

**Middle case**: Some state survives, some doesn't. FIRMWARE_INIT_MAP identifies
the exact writes that must be replayed from VFIO BAR0 after rebind. coralReef
adds a `vfio_replay_firmware_init()` using the map.

**Worst case**: All firmware state is lost on rebind. The full cold → warm
sequence must be replicated in coralReef from scratch. The Phase 2/3 diffs
still provide the complete register write list for this.

In all cases, the experiment produces actionable data for closing the sovereign
dispatch path.

---

## Execution Results (March 19, 2026)

### Infrastructure Achieved

- **DRM Isolation**: Xorg `AutoAddGPU=false` + udev rules (61-priority) prevent
  compositor crashes during driver swaps. **Zero relogs** across all swaps.
- **Ember Architecture**: All swaps via `device.swap` JSON-RPC through
  `coral-glowplug` → `coral-ember` → sysfs. Atomic, deadlock-guarded.
- **nvidia Personality**: Added to glowplug registry during execution (was missing).
- **Ember fd Test Harness**: VFIO hardware tests updated to receive fds from ember
  via SCM_RIGHTS (`NvVfioComputeDevice::open_from_fds`), fixing test/ember
  coexistence. Stale IOVA recovery added to `DmaBuffer::new`.

### Phase 1: Cold Baseline (Config A) — COMPLETE

Both Titans cold on vfio after boot.

| Register | Oracle (03:00.0) | Target (4a:00.0) | Notes |
|----------|----------------:|----------------:|-------|
| PMC_ENABLE | 0x5fecdff1 | 0x40000121 | Oracle pre-warmed by isolation test |
| FECS_CPUCTL | 0x00000010 | 0xbadf1201 | Target truly cold: PRIV ring fault |
| GPCCS GPC0 | 0x8780029f | 0xbadf1201 | Target: GR engine off |
| PMU_CPUCTL | 0x00000010 | 0x00000020 | Different halt states |

### Phase 2: nouveau Warm (Config B) — COMPLETE

Cold → nouveau → vfio produces **33 register changes** (target):

| Group | Changes | Key Finding |
|-------|--------:|-------------|
| PMC | 1 | `PMC_ENABLE` 0x40000121 → 0x5fecdff1 (all engines on) |
| BAR | 2 | BAR1/BAR2 instance blocks programmed |
| PFIFO | 3 | Interrupts enabled, PBDMA intr enabled |
| PBDMA0 | 9 | GP_BASE, CHANNEL_STATE, SIGNATURE configured |
| FECS | 4 | 0xbadf1201 → 0x00000010 (halted but accessible) |
| GPCCS | 5 | GPC0 alive: BOOT0=0x8780029f, GR=0x32080a00 |
| PMU | 3 | Boot vector set to 0x00010000 |
| MMU | 4 | Fault buffers allocated, TLB programmed |

### Phase 3: nvidia Warm (Config C) — COMPLETE

**nvidia leaves GPU in SUB-COLD state on teardown**:

| Group | Changes (vs nouveau-warm) | Key Finding |
|-------|--------:|-------------|
| PMC | 1 | 0x5fecdff1 → 0x40000020 (fewer engines than cold boot!) |
| PFIFO | 23 | ALL registers → 0xbad0da00 (engine powered down) |
| PBDMA | 29 | ALL → 0xbad0da00 (completely inaccessible) |
| FECS | 4 | 0x00000010 → 0xbadf1201 (back to PRIV fault) |
| GPCCS | 5 | 0x8780029f → 0xbadf3000 (powered off, different fault code) |
| USERMODE | 5 | ALL → 0xbad00100 (engine off) |
| PCCSR | 4 | ALL → 0xbad0da00 |

**Conclusion**: nvidia proprietary is UNSUITABLE for warm-up. It aggressively
powers down all engines on unbind, leaving a worse state than cold boot.
**nouveau is the ONLY viable warm-up path.**

### Phase 4: Warm → Rebind → Dispatch — COMPLETE

After nouveau warm → vfio rebind, **FECS state survives** (0x00000010, accessible).
All engine enables survive. VFIO hardware tests on warmed silicon:

| Test | Result | Notes |
|------|--------|-------|
| vfio_open_and_bar0_read | ✅ PASS | Ember fd delivery works |
| vfio_alloc_and_free | ✅ PASS | DMA allocation + IOMMU mapping |
| vfio_upload_and_readback | ✅ PASS | Data round-trip through DMA |
| vfio_multiple_buffers | ✅ PASS | Multiple concurrent DMA allocs |
| vfio_free_invalid_handle | ✅ PASS | Error handling |
| vfio_readback_invalid_handle | ✅ PASS | Error handling |
| **vfio_dispatch_nop_shader** | ❌ **FENCE TIMEOUT** | FECS not consuming GPFIFO commands |

Pre-dispatch GR status shows `fecs_halted=false, gr_en=true` — the GR engine
is enabled and FECS is accessible, but the FECS falcon isn't running firmware
(no instruction fetch = won't consume channel methods). This is the **MIDDLE CASE**.

### Remaining Sovereign Pipeline Gap

The register diffs and dispatch test conclusively identify **one remaining blocker**:

**FECS/GPCCS Falcon Firmware**: nouveau loads signed firmware via SEC2/ACR
and starts the falcons. When nouveau unbinds, the falcon hardware is accessible
(registers read real values, not 0xbadf) but the microcode isn't executing
(halted at PC=0x00000000). The GPFIFO is configured correctly (PFIFO/PBDMA/
runlist/instance block all populated), but method packets require a running
FECS to dispatch.

**Options for Exp 071**:
1. **ACR replay from VFIO**: Load ACR payloads into SEC2 EMEM, boot SEC2 → FECS
   (builds on Exp 066/067/068)
2. **Direct FECS IMEM load**: Write falcon microcode to FECS IMEM/DMEM and start
   (requires HS-locked bypass or unsigned test firmware)
3. **Pragmatic hybrid**: Keep nouveau warm-up as permanent first step, accept
   that sovereign dispatch = `ember warm cycle → vfio rebind → dispatch`

### wgpu/Vulkan Benchmark (RTX 5060)

Yukawa OCP MD: 2000 particles, κ=2, Γ=160, 7000 total steps

| Backend | Wall (s) | Steps/s | Final KE | Final PE |
|---------|-------:|-------:|-------:|-------:|
| wgpu/Vulkan (RTX 5060) | 46.77 | 149.7 | 14.3381 | 230.2525 |

Physics validated. f64 shaders operational on RTX 5060 via wgpu.

---

## References

- `wateringHole/SOVEREIGN_COMPUTE_TRIO_CATCHUP_GUIDANCE_MAR17_2026.md` — gap ownership
- `wateringHole/GPU_SOVEREIGN_BRING_UP_GUIDE.md` — 7 gaps, bring-up sequence
- `specs/MULTI_BACKEND_DISPATCH.md` — Tier 1/2/3 architecture
- `experiments/058`–`069` — sovereign pipeline fossil record
- coralReef iter57: H1 cold silicon finding, GP_PUT root cause, GlowPlug lend/reclaim
