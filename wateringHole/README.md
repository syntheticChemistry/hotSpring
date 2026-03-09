# wateringHole — Cross-Project Handoffs

**Project:** hotSpring (ecoPrimals)
**Last Updated:** March 9, 2026

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
| [`../../../wateringHole/handoffs/HOTSPRING_V0624_CORALREEF_ITER26_SOVEREIGN_PIPELINE_HANDOFF_MAR09_2026.md`] | Mar 9 | **coralReef sovereign pipeline**: 44/46 shaders compile to native SASS, full `GpuBackend` impl, DRM dispatch status, lessons learned |
| [`../../../wateringHole/handoffs/HOTSPRING_V0624_MODERN_REWIRE_HANDOFF_MAR09_2026.md`] | Mar 9 | **Modern rewire**: v0.6.24, 769 tests, Chuna 44/44, coralReef Iter 26, barraCuda v0.3.3, toadStool S138 |
| [`../../../wateringHole/handoffs/HOTSPRING_CORALREEF_INTEGRATION_HANDOFF.md`] | Mar 9 | **coralReef integration**: local development setup, discovery manifest, feature gates |

---

## Archive

70+ superseded handoffs in `handoffs/archive/`. These document the full
evolution history from v0.4.x through v0.6.23:

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
