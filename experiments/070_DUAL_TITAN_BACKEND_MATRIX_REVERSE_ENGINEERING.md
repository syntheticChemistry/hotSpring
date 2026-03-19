# Experiment 070: Dual-Titan Backend Matrix for Sovereign Reverse Engineering

**Date**: March 18, 2026
**Status**: PLANNED — Ready to execute with both Titans on GlowPlug
**Hardware**: 2× Titan V (GV100, SM70) — `0000:03:00.0` (oracle), `0000:4a:00.0` (target)
**Prerequisite**: GlowPlug config updated (both on vfio at boot), Experiment 069

---

## Objective

Use both Titans under GlowPlug to systematically reverse engineer the remaining
sovereign dispatch gaps (MMU fault buffers, runlist encoding, PBDMA, GPCCS) by
running **oracle vs target** comparisons. One card can be warm (nouveau) while
the other stays cold (VFIO) — we can compare register states, capture mmiotrace,
and validate fixes without blocking hardware.

---

## Backend Matrix (2 Titans × 4 Configurations)

| Config | Titan 03:00.0 (oracle) | Titan 4a:00.0 (target) | Use Case |
|--------|------------------------|------------------------|----------|
| **A** | vfio | vfio | Both sovereign — baseline cold state, parity checks |
| **B** | nouveau | vfio | Oracle warm — oracle dumps, mmiotrace capture, register diff |
| **C** | vfio | nouveau | Target warm — NVK validation, DF64 compute, wgpu adapter test |
| **D** | nouveau | nouveau | Both warm — dual-GPU compute, Kokkos parity (if Mesa built) |

**Stability invariant**: Boot always in Config A (both vfio). Swap to B/C/D only
after boot. Never leave desktop GPU (RTX 5060) unbound.

---

## Reverse Engineering Workflow

### Phase 1: Oracle Dumps (Config B)

**Goal**: Capture warm reference state from nouveau for comparison with cold VFIO.

```
1. Boot: both vfio (Config A)
2. Swap oracle 03:00.0 → nouveau
3. Wait ~5s for HBM2 training, domain init
4. Run: vfio_boot_follower_diff, vfio_oracle_root_pll_programming
5. Write oracle dump to data/oracle_nouveau_warm.txt
6. Swap oracle 03:00.0 → vfio
```

**Experiments feeding this**: 061 (mmiotrace), 062 (D3hot VRAM), 065 (GlowPlug swap).

### Phase 2: MMU Fault Buffer Comparison (Config B)

**Gap 2**: PBDMA needs fault buffer config. Oracle has it; target doesn't.

**Procedure**:
1. Config B — oracle on nouveau, target on vfio
2. Oracle: read MMU fault buffer registers via BAR0 (sysfs resource0 if root)
3. Target: read same registers via VFIO BAR0
4. Diff: identify which registers differ, what values nouveau sets
5. Implement in coralReef: allocate fault buffers, write to target registers

**Reference**: `GPU_SOVEREIGN_BRING_UP_GUIDE.md` Part 3 — MMU fault buffer spec.

### Phase 3: Runlist Encoding Validation (Config B)

**Gap 3**: Runlist register base and entry encoding wrong for GV100.

**Procedure**:
1. Config B
2. mmiotrace on oracle during nouveau bind → capture runlist submission
3. Parse: 0x2270 + rl_id*0x10 base, 12-byte entry format
4. Compare with coralReef's current encoding
5. Fix and re-test on target (Config A)

**Reference**: Exp 061 — 206,375 MMIO ops captured; Exp 058 — runlist ACK at 0x002A00.

### Phase 4: GPCCS Discovery (Config A)

**Gap 5**: FECS halts at PC=0x2835 waiting for GPCCS. GPCCS address unknown.

**Procedure**:
1. Config A — both vfio, both cold
2. Scan BAR0 0x400000–0x500000 for GPCCS falcon BOOT0 signature
3. If found: load GPCCS firmware, start before FECS
4. Exp 068 proved FECS executes from host IMEM — same pattern for GPCCS

**No oracle needed** — both cards identical; we're scanning for hardware layout.

### Phase 5: NVK Bridge Validation (Config C)

**Goal**: Prove math works on sovereign path via NVK while sovereign dispatch is blocked.

**Procedure**:
1. Config A at boot
2. Swap target 4a:00.0 → nouveau
3. Build Mesa with NVK: `-Dvulkan-drivers=nouveau`
4. Run: `HOTSPRING_GPU_ADAPTER=titan bench_md_parity`, `validate_cpu_gpu_parity`
5. Swap target → vfio
6. Compare: wgpu/NVK vs (future) coralReef/DRM on same hardware

**Reference**: `HOTSPRING_BACKEND_ANALYSIS_GLOWPLUG_SWAP_VALIDATION_MAR17_2026.md` 7-step plan.

---

## Experiment Dependency Graph

```
Exp 058 (PBDMA) ──┬─── Exp 060 (BAR2) ─── Exp 062 (D3hot VRAM)
                 │
Exp 061 (mmiotrace) ── Exp 065 (GlowPlug) ─── Exp 069 (boot persistence)
                 │
Exp 066 (SEC2) ── Exp 067 (EMEM) ─── Exp 068 (FECS direct)
                 │
                 └─── Exp 070 (this) ─── Gaps 2,3,4,5 closure
```

---

## GlowPlug Swap Commands

```bash
# List devices (serde format — current daemon)
echo 'ListDevices' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock

# Swap target to nouveau (for Config C)
echo '{"Swap":{"bdf":"0000:4a:00.0","target":"nouveau"}}' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock

# Swap target back to vfio
echo '{"Swap":{"bdf":"0000:4a:00.0","target":"vfio"}}' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock

# Swap oracle to nouveau (for Config B)
echo '{"Swap":{"bdf":"0000:03:00.0","target":"nouveau"}}' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock

# Swap oracle back to vfio
echo '{"Swap":{"bdf":"0000:03:00.0","target":"vfio"}}' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock
```

**Note**: JSON-RPC 2.0 format when coral-glowplug redeployed: `device.swap` method.

---

## Config Deployment

```bash
# Copy hotSpring config to coralReef deployment
pkexec cp /home/biomegate/Development/ecoPrimals/hotSpring/scripts/boot/glowplug.toml /etc/coralreef/glowplug.toml

# Restart daemon
pkexec systemctl restart coral-glowplug

# Verify both devices
echo 'ListDevices' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock
```

---

## Success Criteria

| Phase | Criterion |
|-------|-----------|
| 1 | Oracle dump written, diff run against cold target |
| 2 | MMU fault buffer registers identified, coralReef writes them |
| 3 | Runlist encoding fixed, PBDMA sees channel |
| 4 | GPCCS address found, or ruled out in range |
| 5 | NVK compute validated on Titan V after swap |

---

## References

- `wateringHole/SOVEREIGN_COMPUTE_TRIO_CATCHUP_GUIDANCE_MAR17_2026.md` — gap ownership
- `wateringHole/GPU_SOVEREIGN_BRING_UP_GUIDE.md` — 7 gaps, bring-up sequence
- `specs/MULTI_BACKEND_DISPATCH.md` — Tier 1/2/3 architecture
- `experiments/058`–`069` — sovereign pipeline fossil record
