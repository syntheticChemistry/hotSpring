# wateringHole — Cross-Project Handoffs

**Project:** hotSpring (ecoPrimals)
**Last Updated:** March 16, 2026

---

## What This Is

The wateringHole is where hotSpring communicates with other primals. Every
handoff is a unidirectional document: hotSpring writes it, the receiving
team reads it and acts. No primal imports another — they learn from each
other by reviewing code in `ecoPrimals/` and acting on handoffs.

```
hotSpring → wateringHole/handoffs/ → toadStool reads and absorbs
                                   → wetSpring reads for cross-spring context
                                   → metalForge reads for hardware context
```

---

## Conventions

### Naming

```
HOTSPRING_V{MAJOR}{MINOR}_{TOPIC}_HANDOFF_{MON}{DD}_{YYYY}.md
```

Examples:
- `HOTSPRING_V0614_TOADSTOOL_BARRACUDA_EVOLUTION_HANDOFF_FEB25_2026.md`
- `HOTSPRING_V0613_CROSS_SPRING_REWIRING_FEB25_2026.md`

Non-versioned handoffs for infrastructure topics:
- `BIOMEGATE_NVK_PIPELINE_ISSUES_FEB24_2026.md`
- `F64_TRANSCENDENTAL_INTERCONNECT_FEB25_2026.md`

### Structure

Every handoff follows this pattern:

1. **Header**: Date, From, To, License, Covers (version range)
2. **Executive Summary**: 3-5 bullet points with metrics
3. **Parts**: Numbered technical sections
4. **Tables**: Primitives, shaders, action items
5. **Action Items**: "toadStool action:" for upstream work

### Archive

Superseded handoffs move to `handoffs/archive/`. The archive is the
fossil record — never deleted, always available for provenance.

---

## Active Handoffs

| File | Date | Topic |
|------|------|-------|
| [`HOTSPRING_PIN_PRIMAL_EVOLUTION_HANDOFF_MAR16_2026.md`](handoffs/HOTSPRING_PIN_PRIMAL_EVOLUTION_HANDOFF_MAR16_2026.md) | Mar 16 | **PIN handoff**: hotSpring pausing for primal evolution sprint. Per-primal engineering backlog (7 coralReef tasks, 3 toadStool tasks), stability invariants, revalidation plan, open research questions |
| [`HOTSPRING_GLOWPLUG_BOOT_PERSISTENCE_SOVEREIGN_PIPELINE_HANDOFF_MAR16_2026.md`](handoffs/HOTSPRING_GLOWPLUG_BOOT_PERSISTENCE_SOVEREIGN_PIPELINE_HANDOFF_MAR16_2026.md) | Mar 16 | **GlowPlug boot persistence + sovereign pipeline status**: systemd daemon, VFIO-first boot, DRM render node fencing (kernel oops fix), graceful shutdown, unsolved blockers, per-primal action items, reproducibility checklist |
| [`HOTSPRING_SOVEREIGN_FALCON_DIRECT_LOAD_HANDOFF_MAR16_2026.md`](handoffs/HOTSPRING_SOVEREIGN_FALCON_DIRECT_LOAD_HANDOFF_MAR16_2026.md) | Mar 16 | **FECS direct execution**: LS security bypass on clean falcon. FECS runs from host-loaded IMEM (PC=0x63EE of 25KB). ACR BL/firmware execute. D3hot→D0 produces clean state. GlowPlug daemon ready. PRIVRING fault lesson. |
| [`HOTSPRING_VFIO_D3HOT_VRAM_BREAKTHROUGH_MAR16_2026.md`](handoffs/HOTSPRING_VFIO_D3HOT_VRAM_BREAKTHROUGH_MAR16_2026.md) | Mar 16 | **D3hot→D0 VRAM recovery**: BIOS trains HBM2, survives D3hot. 24/26 tests pass. Digital PMU. GlowPlug warm detection. |
| [`HOTSPRING_VFIO_PFIFO_PROGRESS_GP_PUT_HANDOFF_MAR09_2026.md`](handoffs/HOTSPRING_VFIO_PFIFO_PROGRESS_GP_PUT_HANDOFF_MAR09_2026.md) | Mar 09 | **VFIO PFIFO progress**: 6/7 PBDMA tests pass, USERD GP_PUT DMA read remaining. USERD_TARGET fix in runlist entry. |
| [`HOTSPRING_V0631_GAP_CLOSURE_REWIRE_HANDOFF_MAR12_2026.md`](handoffs/HOTSPRING_V0631_GAP_CLOSURE_REWIRE_HANDOFF_MAR12_2026.md) | Mar 12 | **Gap closure rewire**: hotSpring v0.6.31 absorbs toadStool S147, barraCuda 82ff983→7c1fd03a, coralReef Iter 37-39. Gap 1+4 CLOSED, Gap 3 WIRED. Sovereign routing with `sovereign_resolves_poisoning()`. |

---

## Archive

83+ superseded handoffs in `handoffs/archive/`. These document the full
evolution history from v0.4.x through v0.6.31:

- Early toadStool rewire documents (v1-v4)
- GPU primitive absorption records
- NPU integration handoffs
- Cross-spring evolution maps
- biomeGate hardware setup records

---

## Cross-Spring Context

### How hotSpring relates to other springs

```
hotSpring (physics/precision)  ──→ barracuda ←── wetSpring (bio/genomics)
                                       ↑
                                 neuralSpring (ML/eigen)
                                       ↑
                                  airSpring (weather/climate)
                                       ↑
                                  coralReef (sovereign shader compiler)
```

Each spring evolves independently. barraCuda absorbs shared math. toadStool
manages hardware dispatch. coralReef compiles WGSL→native (SASS/GFX).
Springs discover capabilities at runtime — no direct imports.

### Cross-spring documentation in other springs

- **wetSpring** (`../../../wetSpring/wateringHole/`): `CROSS_SPRING_SHADER_EVOLUTION.md` (612 WGSL shader provenance map)
- **toadStool** (`../../../phase1/toadstool/`): Shared compute library README, shader categories
- **neuralSpring**: ESN reservoir patterns shared with hotSpring MD pipeline

---

## License

AGPL-3.0-only. All handoff documents are part of the open science record.
