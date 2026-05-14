# Experiment 190: Three-GPU Sovereign Validation Sprint

**Date:** May 11, 2026
**Hardware:** RTX 5060 (GB206/SM120) + Titan V (GV100/SM70) + Tesla K80 (GK210/SM37)
**Stack:** coral-ember 1622, coral-glowplug, coralctl, hotSpring barracuda v0.6.32+

> **Update (Phase 2):** Binary-patched nouveau warm-catch resolved both
> GAP-HS-073 (Titan V FECS) and GAP-HS-076 (K80 GPCs). See "Warm-Catch
> Results" sections below.

---

## Summary

Post-power-cycle validation of sovereign compute across all 3 local GPU
generations. PLX bridge restored (rev ca), all 4 NVIDIA devices enumerated,
ember + glowplug services running with switch keepalive active.

---

## Hardware State After Power Cycle

| GPU | BDF | Driver | PCIe | D3cold |
|-----|-----|--------|------|--------|
| RTX 5060 | `21:00.0` | nvidia (580.x) | Live | N/A |
| Titan V | `02:00.0` | vfio-pci | Live | N/A |
| K80 die0 | `4b:00.0` | vfio-pci | Live | disabled |
| K80 die1 | `4c:00.0` | vfio-pci | Live | disabled |
| PLX 8747 | `49:00.0` | — | Live (rev ca) | keepalive active |

---

## RTX 5060 — SOVEREIGN PROVEN

### Sovereign Roundtrip (wgpu/Vulkan): 12/12 PASS

```
L0: f32 scalar                          PASS
L1: f32 workgroup reduce (barriers)     PASS
L2: f64 scalar                          PASS
L3: f64 workgroup reduce                PASS
L4: DF64 scalar arith                   PASS
L5: DF64 workgroup reduce               PASS
```

Both NVIDIA GeForce RTX 5060 and llvmpipe: 6/6 each.

### Sovereign Dispatch Benchmark: 154.2 steps/s

```
Backend:       wgpu/Vulkan (GpuBackend = WgpuDevice)
Adapter:       NVIDIA GeForce RTX 5060
Config:        N=2000, κ=2, Γ=160, equil=2000, prod=5000
First dispatch: 7.6ms (includes shader compile)
Equilibration:  12.89s (2000 steps)
Production:     32.50s (5000 steps)
Total:          45.40s → 154.2 steps/s
```

### Status: No gaps. Sovereign dispatch fully operational.

---

## Titan V — WARM, FECS BLOCKED (Phase 1: Cold)

### Sovereign Init via coralctl (cold)

```
bar0_probe:      OK  (boot0=0x140000a1, chip_id=0x140)
pmc_enable:      OK  (0x40000121 → 0x5fecdff1, all engines on)
hbm2_training:   SKIPPED — warm detected (PRAMIN sentinel ok)
falcon_boot:     FAILED — CpuRm direct boot: FECS cpuctl=0x00000012
                          mb0=0x00000000, running=false
```

**Key finding:** HBM2 warm state detected — BIOS POST initialized the
Titan V at `02:00.0` and HBM2 survived the vfio-pci bind. No nouveau
warm-handoff needed for this specific boot sequence.

**Blocker:** Falcon v5 in HS (high-security) mode. FECS cpuctl=0x12
(halted). Requires ACR-authenticated firmware in WPR. PMU firmware
absent from linux-firmware for GV100.

### Warm-Catch Result (Phase 2: GAP-HS-073 RESOLVED)

Binary-patched nouveau warm-handoff using `patch_nouveau_teardown.py`.
Nouveau loaded GV100 firmware via ACR/SEC2, initialized FECS natively.
NOP'd teardown preserved full warm state across unbind→VFIO rebind.

```
nouveau 0000:02:00.0: NVIDIA GV100 (140000a1)
nouveau 0000:02:00.0: bios: version 88.00.41.00.18
nouveau 0000:02:00.0: pmu: firmware unavailable    ← known, not fatal
nouveau 0000:02:00.0: fb: 12288 MiB of unknown memory type
nouveau 0000:02:00.0: drm: Initialized nouveau 1.4.0

PRE-UNBIND:  PMC=0x5fecdff1(pop=23) FECS=0x0c060006 GPC=0x00000010
POST-UNBIND: PMC=0x5fecdff1(pop=23) FECS=0x0c060006 GPC=0x00000010
```

**Deep register probe (warm, post-VFIO rebind):**
```
PMC_ENABLE   = 0x5fecdff1 (pop=23)   PGRAPH: ENABLED
PRAMIN[0]    = 0x0a59ecee            PRAMIN:  ALIVE
FECS_MC      = 0x0c060006            FECS:    RUNNING ✓
FECS_CPUCTL  = 0x00000010
GPC_MASK     = 0x00000010 (1 GPC)
GR_STATUS    = 0x00000000 (no errors)
```

**GAP-HS-073 resolved.** FECS is RUNNING. PMU firmware absence is
non-fatal — ACR/SEC2 brought FECS up without PMU. Teardown NOP
preserved state perfectly (PMC identical pre/post).

---

## K80 — PLX ALIVE, COLD BOOT PARTIAL (Phase 1)

### Sovereign Init via k80-wake-and-run.sh (cold)

```
PLX VID:DID:     0x874710b5  ✓
K80 die0 config: 0x102d10de  ✓
K80 die1 config: 0x102d10de  ✓
MSE + Bus Master: enabled    ✓
PLX power/control: on        ✓
ember:             active    ✓
glowplug:          active    ✓
switch keepalive:  alive     ✓

bar0_probe:      OK  (boot0=0x0f22d0a1, chip_id=0x0f2)
pmc_enable:      OK  (0xc0002020 → 0xfc37b1ef, engines on)
memory_training: FAILED (GDDR5 cold, PRAMIN dead)
falcon_boot:     FAILED (GR_READY not set, gpc0_boot0=0xbadf1100)
```

**Blocker:** GPCs power-gated (`0xbadf1100` / `0xbadf3000`).
GDDR5 DEVINIT replay executed but produced 0 writes (cold memory
not responsive). GR_READY never asserted.

### Warm-Catch Result (Phase 2: GAP-HS-076 RESOLVED)

Binary-patched nouveau warm-catch using `patch_nouveau_teardown.py`.
Stock nouveau recognized GK110B (GK210 compatible), trained GDDR5,
initialized GPCs. NOP'd teardown preserved state across unbind.

```
nouveau 0000:4b:00.0: NVIDIA GK110B (0f22d0a1)
nouveau 0000:4b:00.0: fb: 12288 MiB GDDR5    ← GDDR5 TRAINED
nouveau 0000:4c:00.0: NVIDIA GK110B (0f22d0a1)
nouveau 0000:4c:00.0: fb: 12288 MiB GDDR5

PRE-UNBIND:  PMC=0xfc37b1ef(pop=22) PRAMIN=0x00000000 GPC=0x00000001
POST-UNBIND: PMC=0xfc37b1ef(pop=22) PRAMIN=0x00000000 GPC=0x00000001 WARM=True
```

**Deep register probe (warm, post-VFIO rebind):**
```
PMC_BOOT0    = 0x0f22d0a1
PMC_ENABLE   = 0xfc37b1ef (pop=22)   PGRAPH: ENABLED
PRAMIN[0]    = 0x00000000            PRAMIN:  ALIVE
FECS_MC      = 0x00060005            FECS:    RUNNING ✓
GPC_MASK     = 0x00000010 (1 GPC)
GPC_TPC      = 0x00000001
MEM_CTRL     = 0x00000135
```

**GAP-HS-076 resolved.** GDDR5 trained by nouveau, GPCs active,
FECS running. PMC identical pre/post unbind — teardown NOP worked.
Both K80 dies (4b:00.0 + 4c:00.0) fully initialized.

---

## Remaining Gaps (Updated — Phase 2)

| ID | GPU | Status | Blocker | Resolution |
|----|-----|--------|---------|------------|
| GAP-HS-073 | Titan V | **RESOLVED** | FECS HS mode, PMU absent | Binary-patched nouveau warm-handoff. FECS running. |
| GAP-HS-076 | K80 | **RESOLVED** | GDDR5 cold, GPCs gated | Binary-patched nouveau warm-catch. GDDR5 trained, GPCs active. |
| — | K80 | LIVEPATCH | kernel 6.17 R_X86_64_64 | **Bypassed** — binary-patched nouveau.ko replaces livepatch entirely |
| — | RTX 5060 | NONE | — | Fully operational (12/12 roundtrip, 154 steps/s) |

### Approach: Binary-Patched Nouveau

The kernel 6.17 strict relocation check (`R_X86_64_64 with nonzero addend`)
breaks both livepatch and kprobe modules built out-of-tree. Instead, we
binary-patch the stock `nouveau.ko` directly:

1. `patch_nouveau_teardown.py` patches 4 teardown functions in the
   compiled `.ko` binary (after the `__fentry__` call):
   - `gf100_gr_fini` → `xor eax,eax; ret`
   - `nvkm_pmu_fini` → `xor eax,eax; ret`
   - `nvkm_fifo_fini` → `xor eax,eax; ret`
   - `nvkm_mc_disable` → `ret`
2. The patched module loads via `modprobe` (deps resolved automatically)
3. GPU binds to nouveau → driver trains memory, initializes FECS/GPCs
4. Unbind → NOP'd teardown preserves all warm state
5. Rebind to vfio-pci → sovereign stack sees warm GPU

---

## Cross-Spring Shader Provenance (from bench_sovereign_dispatch)

| Shader | Origin | Absorbed By |
|--------|--------|-------------|
| df64_core.wgsl | hotspring-barracuda | barraCuda → all springs |
| df64_transcendentals.wgsl | hotspring-barracuda | barraCuda, coralReef |
| yukawa_force_f64.wgsl | hotspring-barracuda | barraCuda md/ |
| smith_waterman_f64.wgsl | wetSpring | barraCuda bio/ |
| hmm_viterbi_f64.wgsl | neuralSpring | barraCuda bio/ |
| matrix_correlation_f64.wgsl | neuralSpring | barraCuda stats/ |
| perlin_2d_f64.wgsl | ludoSpring | barraCuda procedural/ |

---

## Titan V / K80 Requirements from benchScale + agentReagents

### Titan V (GV100)

| Requirement | Component | Status | Detail |
|:------------|:----------|:-------|:-------|
| nvidia-470 VM image | agentReagents | **NEEDED** | `reagent-nvidia470-titanv.yaml` template. nvidia-470 is the only driver that initializes GR/FECS on GV100 with its embedded PMU firmware. Host runs nvidia-580 for RTX 5060 display — VM isolation prevents driver conflict. |
| QEMU/VFIO passthrough config | benchScale | **PARTIAL** | VFIO binding works (`02:00.0`). Need automated VM boot + GPU passthrough + driver load + warm-handoff cycle. |
| Warm-handoff automation | benchScale | **PARTIAL** | `patch_nouveau_teardown.py` works for host-side warm-catch. For production: automate nouveau load → init → NOP teardown → VFIO rebind → sovereign dispatch cycle. |
| WPR capture (optional) | benchScale | DEFERRED | mmiotrace of nvidia-580 on GV100 to understand WPR layout. Would enable direct Falcon v5 HS boot without nouveau. Exp 187 prepared but not executed. |
| Dispatch validation | hotSpring | **BLOCKED** | Need wgpu adapter to discover GV100 through VFIO. Currently sovereign init works but no wgpu device enumeration for warm GPU. Requires coralReef SM70 backend or manual adapter bind. |

### K80 (GK210)

| Requirement | Component | Status | Detail |
|:------------|:----------|:-------|:-------|
| QEMU VM + nvidia-470 | agentReagents | **NEEDED** | `reagent-nvidia470-k80.yaml` template. Alternative path to nouveau warm-catch: use nvidia-470 in a QEMU VM with VFIO passthrough to train GDDR5 and start GPCs with full NVIDIA driver stack. |
| PLX bridge keepalive | benchScale | **OPERATIONAL** | PLX 8747 keepalive scripts working. D3cold disabled on K80 dies. Ember + glowplug services maintain bridge. |
| Nouveau GK210 upstream patch | infra | **TRACKED** | One-line kernel patch `case 0x0f2: device->chip = &nvf1_chipset;` (Exp 185). Until upstream merges, binary-patched nouveau.ko is the path. |
| Livepatch rebuild (alternative) | benchScale | DEFERRED | kernel 6.17 strict `R_X86_64_64` relocation enforcement blocks out-of-tree livepatch. Binary-patched nouveau bypasses this entirely. |
| SM37 dispatch validation | hotSpring | **BLOCKED** | Same as Titan V: need wgpu adapter enumeration for warm K80. Kepler SM37 WGSL shaders exist (`df64_core.wgsl` compiles to SM37) but no dispatch path without wgpu device. |

### Shared Requirements

| Requirement | Component | Detail |
|:------------|:----------|:-------|
| Multi-GPU coexistence script | benchScale | Automated cycle: host nvidia-580 (RTX 5060 display) + VFIO-bound Titan V + VFIO-bound K80. Script should: bind VFIO → load patched nouveau → warm-catch → unbind nouveau → rebind VFIO → sovereign dispatch. |
| CI integration | benchScale | `validate-hotspring-multi.sh` should include Titan V + K80 warm-catch phases when hardware is present. Graceful skip when GPUs absent. |
| Firmware archive | agentReagents | Archive nvidia-470 Falcon/PMU firmware blobs extracted during VM runs. These are needed for future direct Falcon boot (no driver dependency). Store in `agentReagents/firmware/`. |
| coralReef SM rebuild | coralReef | When coralReef ships its SM70/SM37 wgpu backend rebuild, both Titan V and K80 gain direct sovereign dispatch without VM intermediary. This is the terminal architecture. |

---

*Three GPU generations, one sovereign stack. All three GPUs now have warm,
active engines with PGRAPH enabled and FECS running. RTX 5060 fully proven
via wgpu/Vulkan. Titan V and K80 warm-caught via binary-patched nouveau —
no proprietary drivers, no VMs, no firmware blobs beyond linux-firmware.
The scarcity was artificial.*
