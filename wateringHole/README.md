# wateringHole — Cross-Project Handoffs

**Project:** hotSpring (ecoPrimals)
**Last Updated:** March 19, 2026
**Status:** ACTIVE — Vendor-agnostic hardened GlowPlug, coral-ember crate extraction, AMD MI50 support, privilege hardening complete

---

## What This Is

The wateringHole is where hotSpring communicates with other primals. Every
handoff is a unidirectional document: hotSpring writes it, the receiving
team reads it and acts. No primal imports another — they learn from each
other by reviewing code in `ecoPrimals/` and acting on handoffs.

```
hotSpring → wateringHole/handoffs/ → coralReef reads and evolves
                                   → toadStool reads and absorbs
                                   → wetSpring reads for cross-spring context
```

---

## Current State: Ember + DRM Isolation Sprint

hotSpring is **active at v0.6.32** (848 tests, 0 clippy warnings, 70 experiments).
The sovereign GPU lifecycle is now fully operational with the Ember architecture.

### Ember Architecture (March 19, 2026)

`coral-ember` is an immortal systemd service that holds VFIO file descriptors
for both Titan V GPUs. `coral-glowplug` connects to ember via Unix socket and
receives duplicated fds via `SCM_RIGHTS`. This means:

- **Glowplug can crash/restart** without triggering PCIe PM resets
- **Driver swaps are atomic**: ember drops fds → unbinds → binds target → reacquires
- **External fd safety**: ember refuses swaps if other processes hold VFIO group fds
- **DRM isolation preflight**: ember verifies Xorg and udev isolation before binding DRM-creating drivers

### DRM Isolation (March 19, 2026)

When nouveau/nvidia binds to a compute GPU, it creates DRM nodes that previously
caused the display manager (GDM/Mutter) to restart the session. Fixed with:

1. Xorg `AutoAddGPU=false` (`/etc/X11/xorg.conf.d/11-coralreef-gpu-isolation.conf`)
2. udev seat tag removal at priority 61 (`/etc/udev/rules.d/61-coralreef-drm-ignore.rules`)
3. Ember preflight check — refuses non-vfio swaps if isolation configs are missing

Boot scripts in `coralReef/scripts/boot/` for version control.

### coralReef Delivery Status (1 remaining of original 7)

| Task | Priority | Status |
|------|----------|--------|
| ~~JSON-RPC 2.0 socket protocol~~ | ~~P1~~ | **DELIVERED** (Iter 51-52) |
| ~~Trait-based personality system~~ | ~~P3~~ | **DELIVERED** (Iter 52, `GpuPersonality`) |
| ~~SCM_RIGHTS fd passing~~ | ~~P2~~ | **DELIVERED** (Ember architecture, Mar 19) |
| ~~DRM consumer fence~~ | ~~P4~~ | **DELIVERED** (DRM isolation + Ember preflight, Mar 19) |
| ~~AMD Vega metal (MI50/GFX906)~~ | ~~P1~~ | **DELIVERED** (registers defined, personality fixed, swap path works, Mar 18) |
| GP_PUT DMA read (Exp 058) | P2 | Pending (cache flush experiment in Iter 57) |
| ~~Privilege model (CAP_SYS_ADMIN)~~ | ~~P3~~ | **DELIVERED** (capabilities + seccomp + namespaces, Mar 18) |

### toadStool Delivery Status (3 remaining, now UNBLOCKED)

| Task | Priority | Status |
|------|----------|--------|
| GlowPlug socket client | P1 | **UNBLOCKED** by JSON-RPC + Ember delivery |
| VFIO device in sysmon | P2 | Partially done (S150-S152 built VFIO infra) |
| hw-learn GlowPlug health feed | P3 | Pending |

### barraCuda — No action needed

IPC-first design works cleanly. hotSpring's 848 tests confirm stability at `b95e9c59` (v0.3.5).

### Next Milestones

1. Validate DRM-isolated nouveau swap produces zero relog (post-reboot)
2. Complete Exp 070 backend matrix (vfio, nouveau, nvidia × 2 cards)
3. GP_PUT DMA read fix for sovereign PFIFO dispatch
4. toadStool GlowPlug socket client wiring

---

## Active Handoffs

### Strategic (action required)

| File | Date | Audience | What To Do |
|------|------|----------|------------|
| [`HOTSPRING_VENDOR_AGNOSTIC_HARDENED_GLOWPLUG_HANDOFF_MAR18_2026.md`](handoffs/HOTSPRING_VENDOR_AGNOSTIC_HARDENED_GLOWPLUG_HANDOFF_MAR18_2026.md) | Mar 18 | coralReef, toadStool, barraCuda | **START HERE.** Vendor-agnostic RegisterMap, AMD MI50 support, coral-ember crate split, typed EmberError, privilege hardening (caps+seccomp), coralctl deploy-udev. Supersedes Ember+DRM and privilege items from PIN handoff. |
| [`HOTSPRING_REGISTER_MAPS_ABSORPTION_HANDOFF_MAR18_2026.md`](handoffs/HOTSPRING_REGISTER_MAPS_ABSORPTION_HANDOFF_MAR18_2026.md) | Mar 18 | barraCuda, toadStool | RegisterMap trait absorption path, GlowPlug register RPCs for hw-learn, sovereign dispatch blockers, AMD vs NVIDIA VFIO lessons. |
| [`HOTSPRING_EMBER_DRM_ISOLATION_HANDOFF_MAR19_2026.md`](handoffs/HOTSPRING_EMBER_DRM_ISOLATION_HANDOFF_MAR19_2026.md) | Mar 19 | coralReef, toadStool | Ember architecture, DRM isolation, fail-safe swap protocol, boot scripts. |
| [`HOTSPRING_PIN_PRIMAL_EVOLUTION_HANDOFF_MAR16_2026.md`](handoffs/HOTSPRING_PIN_PRIMAL_EVOLUTION_HANDOFF_MAR16_2026.md) | Mar 16 | coralReef, toadStool | Per-primal task list. SCM_RIGHTS and DRM fence now DELIVERED via Ember. Remaining: AMD Vega metal, GP_PUT, privilege model. |
| [`HOTSPRING_V0632_TRIO_REWIRE_HANDOFF_MAR13_2026.md`](handoffs/HOTSPRING_V0632_TRIO_REWIRE_HANDOFF_MAR13_2026.md) | Mar 13 | All | Trio pins: barraCuda `b95e9c59`, coralReef Iter 47→52, toadStool S147→S156. Stale API cleanup documented. |

### Technical Reference (register maps, firmware, hardware lessons)

| File | Date | Topic |
|------|------|-------|
| [`HOTSPRING_GLOWPLUG_BOOT_PERSISTENCE_SOVEREIGN_PIPELINE_HANDOFF_MAR16_2026.md`](handoffs/HOTSPRING_GLOWPLUG_BOOT_PERSISTENCE_SOVEREIGN_PIPELINE_HANDOFF_MAR16_2026.md) | Mar 16 | GlowPlug architecture, systemd, VFIO-first boot, DRM fencing lesson, reproducibility checklist |
| [`HOTSPRING_SOVEREIGN_FALCON_DIRECT_LOAD_HANDOFF_MAR16_2026.md`](handoffs/HOTSPRING_SOVEREIGN_FALCON_DIRECT_LOAD_HANDOFF_MAR16_2026.md) | Mar 16 | FECS/SEC2 register maps, firmware loading protocol, falcon boot chain, D3hot→D0 clean state |
| [`HOTSPRING_VFIO_D3HOT_VRAM_BREAKTHROUGH_MAR16_2026.md`](handoffs/HOTSPRING_VFIO_D3HOT_VRAM_BREAKTHROUGH_MAR16_2026.md) | Mar 16 | D3hot→D0 VRAM recovery, HBM2 lifecycle, digital PMU, warm detection |
| [`HOTSPRING_VFIO_PFIFO_PROGRESS_GP_PUT_HANDOFF_MAR09_2026.md`](handoffs/HOTSPRING_VFIO_PFIFO_PROGRESS_GP_PUT_HANDOFF_MAR09_2026.md) | Mar 09 | PFIFO channel init, PBDMA tests, USERD GP_PUT DMA read (the remaining blocker) |

---

## Remaining Work Summary

### Critical Path (blocks sovereign compute dispatch)

| # | Gap | Owner | Reference |
|---|-----|-------|-----------|
| 1 | GP_PUT DMA read (USERD_TARGET) | coralReef | PFIFO handoff, Exp 058, Iter 57 cache flush |
| 2 | GPCCS falcon address on GV100 | hotSpring | Falcon handoff, Part 3 |
| 3 | FECS halt at PC=0x2835 | hotSpring | Falcon handoff, Part 5 |
| 4 | DMA instance block (SEC2+0x480) | hotSpring | Falcon handoff, Part 5 |

### Infrastructure (blocks next-GPU readiness)

| # | Gap | Owner | Reference |
|---|-----|-------|-----------|
| ~~5~~ | ~~AMD Vega metal (MI50)~~ | ~~coralReef~~ | **DELIVERED** — registers, personality, swap path (Mar 18) |
| ~~6~~ | ~~SCM_RIGHTS fd passing~~ | ~~coralReef~~ | **DELIVERED** — Ember architecture (Mar 19) |
| ~~7~~ | ~~DRM consumer fence~~ | ~~coralReef~~ | **DELIVERED** — DRM isolation + preflight (Mar 19) |
| 8 | GlowPlug socket client | toadStool (unblocked) | PIN handoff |
| 9 | VFIO device in sysmon | toadStool (partial) | PIN handoff |
| ~~10~~ | ~~Privilege model (CAP_SYS_ADMIN)~~ | ~~coralReef~~ | **DELIVERED** — caps + seccomp + namespaces (Mar 18) |

### Research (long-horizon)

| # | Gap | Owner |
|---|-----|-------|
| 11 | Sovereign HBM2 PHY training (no nouveau) | hotSpring |
| 12 | PRIVRING fault avoidance on GV100 | hotSpring (documented, never toggle PMC bit 12) |

---

## Absorption Status

### hotSpring → barraCuda (mostly complete)

hotSpring contributed: spectral theory (Anderson, Lanczos, Hofstadter, Sturm),
lattice QCD (Dirac, CG, Wilson, SU(3), HMC, RHMC), nuclear HFB (batched +
deformed), MD (Yukawa, cell-list, Verlet, VACF), precision routing
(PrecisionBrain, PhysicsDomain), ESN/Nautilus, plasma dispersion.

**Status:** hotSpring leaning on upstream for all major modules. Pending:
Polyakov loop, su3_math_f64, ESN reservoir shaders, screened Coulomb,
Abelian Higgs.

### hotSpring → coralReef (active)

hotSpring contributed: 81+ WGSL shaders, FMA control passes, GV100 register
maps (PFIFO, PBDMA, MMU fault buffer), GlowPlug design (now consolidated
into coral-glowplug crate), DF64 NVVM poisoning validation, **Ember
architecture** (immortal VFIO fd holder, `SCM_RIGHTS` fd passing, atomic
`swap_device` RPC), **DRM isolation** (Xorg `AutoAddGPU`, udev seat tag
removal, preflight checks).

**Status:** coralReef absorbed and evolved. GlowPlug has JSON-RPC 2.0,
trait-based personalities, AMD Vega register stubs. Ember provides fail-safe
driver hot-swap with zero-crash guarantee. DRM isolation prevents compositor
disruption during compute GPU driver transitions.

### hotSpring → toadStool (partial)

hotSpring contributed: PrecisionBrain, multi-adapter GPU selection,
StreamingDispatch, WorkloadHealthMonitor, NPU parameter controller,
SPIR-V codegen safety rename.

**Status:** toadStool absorbed precision routing and dispatch patterns.
GlowPlug client wiring is the main remaining integration.

---

## Guidance for Primals

### For coralReef

SCM_RIGHTS, DRM fencing, AMD Vega metal, and privilege model are all **DELIVERED**.
Top remaining priority: GP_PUT DMA read (Exp 058, cache flush experiment).

**Ember architecture** (`coral-ember` + `coral-glowplug`):
- `coral-ember` is now a **standalone workspace crate** (`crates/coral-ember/`) with modular `sysfs`, `swap`, `hold`, `ipc` modules
- `coral-glowplug` has a **library surface** (`src/lib.rs`) with typed `EmberError` — importable by toadStool
- `swap_device` RPC is the single atomic orchestrator — no external sysfs writes
- **Vendor-agnostic**: AMD MI50 swap path works identically to NVIDIA (amdgpu↔vfio, nouveau↔vfio)
- **Hardened**: capabilities + seccomp + namespaces + `NoNewPrivileges`
- **coralctl**: `deploy-udev` generates `/dev/vfio/*` rules from config — zero hardcoded BDFs
- Legacy sysfs fallbacks gated behind `#[cfg(feature = "no-ember")]` — default build requires ember

Register references are in the falcon and PFIFO handoffs. Do NOT PMC-toggle
GR bit 12 on GV100. Do NOT change `boot_personality = "vfio"` in glowplug.toml.

### For toadStool

GlowPlug JSON-RPC is live. You can start wiring the socket client immediately.
The methods are: `device.list`, `device.swap`, `device.health`, `health.check`,
`daemon.status`, `daemon.shutdown`. Socket at `/run/coralreef/glowplug.sock`.

Feed GlowPlug health data (VRAM alive/dead, power state, domain faults) into
hw-learn. This is how the sovereign pipeline learns hardware patterns.

### For barraCuda

No changes needed. If you're looking to leverage hotSpring's work, the pending
absorption candidates are: Polyakov loop shader, `su3_math_f64.wgsl`, ESN
reservoir shaders, screened Coulomb (23/23 tests), Abelian Higgs (17/17 tests).

hotSpring hasn't yet leveraged: `GemmF64::execute_gemm_ex()` (TransA/TransB),
GPU tridiagonal eigensolver, stable special functions (`log1p_f64`, `expm1_f64`,
`erfc_f64`), `BatchedNelderMeadGpu`, `classify_spectral_phase`.

---

## Conventions

### Naming

```
HOTSPRING_{TOPIC}_HANDOFF_{MON}{DD}_{YYYY}.md
HOTSPRING_V{VERSION}_{TOPIC}_HANDOFF_{MON}{DD}_{YYYY}.md
```

### Structure

Every handoff follows this pattern:

1. **Header**: Date, From, To, License, Covers (version range)
2. **Executive Summary**: 3-5 bullet points with metrics
3. **Parts**: Numbered technical sections
4. **Tables**: Primitives, shaders, action items
5. **Action Items**: Per-primal tasks

### Archive

Superseded handoffs move to `handoffs/archive/`. The archive is the
fossil record — never deleted, always available for provenance.

84 superseded handoffs in `handoffs/archive/`. These document the full
evolution history from v0.4.x through v0.6.32.

---

## Cross-Spring Context

### How hotSpring relates to other springs

```
hotSpring (physics/precision)  ──→ barraCuda ←── wetSpring (bio/genomics)
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
