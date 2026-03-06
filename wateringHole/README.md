# wateringHole — Cross-Project Handoffs

**Project:** hotSpring (ecoPrimals)
**Last Updated:** March 6, 2026

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
| `HOTSPRING_V0619_PRECISION_STABILITY_HANDOFF_MAR06_2026.md` | Mar 6 | **Precision stability**: 9 cancellation families, stable BCS v² + W(z), FHE depth analysis, full debt resolution (zero clippy, all AGPL-3.0-only) |
| `HOTSPRING_V0619_CROSS_SPRING_EVOLUTION_HANDOFF_MAR06_2026.md` | Mar 6 | **Cross-spring evolution**: 724 tests, Chuna Papers 43-45, cross-spring shader provenance |
| `HOTSPRING_V0619_BARRACUDA_REWIRE_TOADSTOOL_S96_HANDOFF_MAR06_2026.md` | Mar 6 | **barraCuda + toadStool S96 rewire**: GPU promotion (gradient flow 38.5×, dielectric 12/12), universal shader compilation |
| `HOTSPRING_V0618_DEEP_DEBT_BARRACUDA_EVOLUTION_HANDOFF_MAR06_2026.md` | Mar 6 | **Deep debt**: Chuna paper GPU promotion, barracuda evolution priorities |
| `HOTSPRING_V0617_GRADIENT_FLOW_SCIENCE_LADDER_HANDOFF_MAR06_2026.md` | Mar 6 | **Science ladder**: gradient flow + integrators, N_f=4 infra, RHMC |

---

## Archive

63 superseded handoffs in `handoffs/archive/`. These document the full
evolution history from v0.4.x through v0.6.19:

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
```

Each spring evolves independently. ToadStool absorbs what works from each
spring into the shared barracuda library. Springs discover each other's
capabilities at runtime via `discovery.rs` probes — no direct imports.

### Cross-spring documentation in other springs

- **wetSpring** (`../../../wetSpring/wateringHole/`): `CROSS_SPRING_SHADER_EVOLUTION.md` (612 WGSL shader provenance map)
- **toadStool** (`../../../phase1/toadstool/`): Shared compute library README, shader categories
- **neuralSpring**: ESN reservoir patterns shared with hotSpring MD pipeline

---

## License

AGPL-3.0-only. All handoff documents are part of the open science record.
