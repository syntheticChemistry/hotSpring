# Reagent Capture Pipeline — Sovereign Boot Path Mapped — hotSpring Handoff

**Date:** May 25, 2026
**From:** hotSpring
**To:** primalSpring (audit), toadStool (sovereign boot tools)
**Status:** Experiment 222 evolved — dual GPU compute validated, PMU firmware determinism proven, SBR cold reset sovereign boot path mapped
**Experiments:** 222 (builds on Exp 221 falcon HS boundary mapping)
**Previous:** `HOTSPRING_UEFI_MODEL_PRI_RING_RECOVERY_EXP221_MAY25_2026.md`

## Summary

Exp 222 advanced from reagent capture validation to full sovereign boot path
mapping. Three paths executed in parallel — all feeding the sovereign boot chain
understanding. The reagent pipeline theory is now backed by reproducible
firmware captures and a proven SBR cold reset mechanism.

## Key Results

| Finding | Detail |
|---------|--------|
| RTX 5060 compute | VectorAdd 1M PASS, SM 12.0, 30 SMs, CUDA 12.8 |
| Titan V compute (VM) | CUDA context + HBM2 memset 0xCAFEBABE PASS, 80 SMs, 12066 MB |
| PMU firmware determinism | MD5 identical across boots (May 11 vs May 25) |
| PMU IMEM | 64 KB, 16384/16384 words non-zero, md5:`5a043df5` |
| PMU DMEM | 64 KB, 16384/16384 words non-zero, first word: 0xdead5ec2 |
| PIO safety | IMEM blocked by HS mode 3, DMEM writes work, zero lockups |
| SBR cold reset | Works via upstream bridge setpci, no lockups |
| PMC_ENABLE writes | Work in post-SBR state (not in post-nouveau state) |
| FECS power-up | Powers up after PMC_ENABLE in post-SBR cold state |
| nouveau GR blocker | No `nvidia/gv100/pmu/` in firmware manifest — design gap |
| Volta ACR model | PMU runs ACR (not SEC2). SEC2 is unused on Volta. |
| RPCs | 27 total (+1: `sovereign.recipe_replay`) |

## Volta Boot Chain — Complete Understanding

```
SBR/Power-On
    ↓
PMU HS ROM auto-boots (fuse-enforced, MB0=0x300)
    ↓
Host writes PMC_ENABLE → powers FECS, GPCCS
    ↓
FECS/GPCCS HS ROMs auto-boot, halt at security gate
    ↓
Host stages signed firmware to VRAM (WPR region) ← remaining blocker
    ↓
Host signals PMU via mailbox with WPR address
    ↓
PMU runs ACR → authenticates FECS/GPCCS firmware
    ↓
GR engine ready for compute
```

## What Changed

### New Tools (agentReagents)
- `tools/titanv-sovereign/capture_pmu_falcon.c` — live falcon state capture
  from running nvidia-470 (BAR0 resource0, PIO IMEM/DMEM read)
- `tools/titanv-sovereign/sovereign_pmu_boot.c` — SBR + PMC_ENABLE + PIO
  upload + DMATRF attempt + STARTCPU (validated safe — no lockups)

### Firmware Artifacts (hotSpring)
- `data/firmware/gv100_nvidia470/pmu_imem.bin` — already existed, confirmed deterministic
- `data/firmware/gv100_nvidia470/pmu_dmem.bin` — already existed, confirmed deterministic
- `infra/catalysts/reagents/.../manifest.json` — populated PMU firmware refs,
  corrected ACR model (PMU not SEC2 on Volta), fixed placeholder date

### Experiment 222 (hotSpring)
- Full session 2 results: SBR experiments, DMATRF, PIO safety, dual compute validation
- Boot chain diagram with remaining WPR+ACR blocker identified

### VM Infrastructure
- `titanv-warmhandoff` VM confirmed operational with nvidia-470.256.02
- nvidia-470 rebuilt for kernel 6.8.0-111 inside VM
- Both Titan Vs functional: #1 in VM (compute), #2 on host (experiments)

## Current System State

| GPU | Driver | Path | Compute |
|-----|--------|------|---------|
| RTX 5060 (21:00.0) | nvidia-580 | Host | CUDA 12.8 VectorAdd PASS |
| Titan V #1 (02:00.0) | nvidia-470 | VM | CUDA 11.4 HBM2 PASS |
| Titan V #2 (49:00.0) | vfio-pci | Cold | Reagent/experiment target |

## Open Gaps

### GAP-TS-222-A: WPR + ACR Mailbox Protocol
The final sovereign boot blocker. Need to determine:
1. Where nvidia-470 stages ACR firmware in VRAM
2. What mailbox command triggers PMU's ACR execution
3. What DMA context/instance block the PMU HS ROM uses

Source: 792K mmiotrace recipe steps contain this — PMU domain writes.

### GAP-TS-222-B: nouveau GV100 PMU Firmware
nouveau never requests `nvidia/gv100/pmu/` firmware. Creating a properly
formatted desc.bin + image.bin from our captured IMEM/DMEM could enable
nouveau GR init on desktop Volta. Requires understanding nouveau's PMU
firmware format (desc.bin header + image.bin layout).

### GAP-TS-222-C: Sovereign DMATRF Register Map
DMATRF register offsets used in exp169 (PMU_BASE+0x1C0/0x1C4/0x1C8) overlap
with DMEM PIO ports. Need correct falcon v5 DMATRF layout for Volta
(likely PMU_BASE+0x110/0x114/0x118/0x11C).

## Resolved from Exp 221

- GAP-TS-221-A (Runtime Services) — RTX 5060 validated, Titan V VM operational
- GAP-TS-221-D (Firmware extraction) — PMU firmware captured and verified

## References

- Experiment doc: `experiments/222_REAGENT_CAPTURE_PIPELINE.md`
- Previous: `experiments/221_UEFI_MODEL_GPU_SOVEREIGNTY.md`
- PMU firmware: `data/firmware/gv100_nvidia470/EXTRACTION_NOTES.md`
- VM template: `agentReagents/templates/reagent-nvidia470-titanv.yaml`
