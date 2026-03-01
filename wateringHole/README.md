# wateringHole — Cross-Project Handoffs

**Project:** hotSpring (ecoPrimals)
**Last Updated:** March 1, 2026

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
| `HOTSPRING_V0615_NPU_PARAM_CONTROL_TOADSTOOL_HANDOFF_MAR01_2026.md` | Mar 1 | NPU parameter controller + barracuda evolution review: dt/n_md control, mid-beta adaptation, Exp 029→031 lessons |
| `HOTSPRING_NAUTILUS_BRAIN_TOADSTOOL_HANDOFF_MAR01_2026.md` | Mar 1 | Nautilus Shell + Brain Architecture: evolutionary reservoir, concept edges, toadStool absorption guide |
| `HOTSPRING_TOADSTOOL_S68_BRAIN_SYNC_FEB28_2026.md` | Feb 28 | Brain architecture sync: 4-layer concurrent pipeline, NVK dual-GPU deadlock fix |
| `HOTSPRING_TOADSTOOL_S68_SYNC_STATUS_FEB27_2026.md` | Feb 27 | S68 sync status: rewiring plan, validation gates |
| `HOTSPRING_V0615_NPU_GPU_PREP_11HEAD_HANDOFF_FEB27_2026.md` | Feb 27 | v0.6.15: 11-head ESN, pipelined predictions, wgpu 22 fixes |
| `HOTSPRING_V0614_NPU_HARDWARE_CROSS_RUN_LEARNING_HANDOFF_FEB26_2026.md` | Feb 26 | NPU hardware integration: live AKD1000, cross-run ESN learning, barracuda evolution review |
| `HOTSPRING_V0614_CROSS_SUBSTRATE_ESN_TOADSTOOL_HANDOFF_FEB26_2026.md` | Feb 26 | Cross-substrate ESN: GPU dispatch, f32 buffers, NPU envelope, absorption guide |
| `METALFORGE_STREAMING_PIPELINE_EVOLUTION_FEB26_2026.md` | Feb 26 | metalForge streaming pipeline: daisy-chain topology, NPU placement, substrate routing |
| `AKIDA_BEHAVIOR_REPORT_FEB26_2026.md` | Feb 26 | Akida AKD1000 behavior report: capabilities beyond SDK, feedback for BrainChip |
| `HOTSPRING_V0614_TOADSTOOL_BARRACUDA_EVOLUTION_HANDOFF_FEB25_2026.md` | Feb 25 | v0.6.14 barracuda evolution + absorption roadmap + paper controls |
| `HOTSPRING_V0613_TOADSTOOL_ABSORPTION_HANDOFF_FEB25_2026.md` | Feb 25 | v0.6.10-14 comprehensive absorption manifest for toadStool |
| `F64_TRANSCENDENTAL_INTERCONNECT_FEB25_2026.md` | Feb 25 | f64 transcendental sharing across springs |
| `TOADSTOOL_SITE_INDEXING_NAK_SOLVER_HANDOFF_FEB25_2026.md` | Feb 25 | Site-indexing + NAK solver patterns |
| `BIOMEGATE_NVK_PIPELINE_ISSUES_FEB24_2026.md` | Feb 24 | NVK driver pipeline issues on Titan V |
| `BIOMEGATE_NVK_TITAN_V_SETUP_FEB23_2026.md` | Feb 23 | Titan V NVK setup guide |

---

## Archive

42 superseded handoffs in `handoffs/archive/`. These document the full
evolution history from v0.4.x through v0.6.15:

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
