# Experiment 190: Three-GPU Sovereign Validation Sprint

**Date:** May 11, 2026
**Hardware:** RTX 5060 (GB206/SM120) + Titan V (GV100/SM70) + Tesla K80 (GK210/SM37)
**Stack:** coral-ember 1622, coral-glowplug, coralctl, hotSpring barracuda v0.6.32+

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

## Titan V — WARM, FECS BLOCKED

### Sovereign Init via coralctl

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

**Next:** GAP-HS-073 — benchScale VM + nvidia-470 warm handoff, or
Exp 187 mmiotrace capture to map the firmware loading sequence.

---

## K80 — PLX ALIVE, COLD BOOT PARTIAL

### Sovereign Init via k80-wake-and-run.sh

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

**Progress:** ember firmware loaded:
- FECS code=768 words, data=193
- GPCCS code=448, data=27
- Hub CSDATA=93, GPCCS CSDATA=103
- GR MMIO init: 153 writes applied
- PMC GR OFF→ON toggle succeeded
- FECS/GPCCS STARTCPU issued

**Blocker:** GPCs power-gated (`0xbadf1100` / `0xbadf3000`).
GDDR5 DEVINIT replay executed but produced 0 writes (cold memory
not responsive). GR_READY never asserted.

**Next:** Exp 188 follow-up — patched nouveau warm-catch with ember
keepalive active to preserve PLX link; then VFIO rebind to verify
GPCs survive the swap.

---

## Remaining Gaps (Updated)

| ID | GPU | Status | Blocker | Path Forward |
|----|-----|--------|---------|--------------|
| GAP-HS-073 | Titan V | WARM, FECS BLOCKED | Falcon v5 HS mode, PMU absent | benchScale VM + nvidia-470 |
| GAP-HS-076 | K80 | PLX ALIVE, GPC GATED | GDDR5 cold, GPCs power-gated | Patched nouveau warm-catch + ember keepalive |
| — | K80 | LIVEPATCH | kernel 6.17 module format | Rebuild livepatch for current kernel |
| — | RTX 5060 | NONE | — | Fully operational |

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

*Three GPU generations, one sovereign stack. RTX 5060 proven. Titan V warm
and waiting for firmware auth. K80 PLX alive for the first time since D3cold.
The scarcity was artificial.*
