# wateringHole — Cross-Project Handoffs

**Project:** hotSpring (ecoPrimals)
**Last Updated:** March 21, 2026
**Status:** ACTIVE — Dual-track dispatch: sovereign VFIO (6/10 layers, MMU blocker) + DRM dispatch (**AMD GCN5 preswap: 6/6 PASS — f64 Lennard-Jones force verified, Newton's 3rd law confirmed**). 18 coral-reef bugs fixed. NVIDIA EXEC coded/PMU-blocked, K80 incoming. 72 experiments. Naga DF64 poisoning bypass **validated end-to-end on real physics (LJ force)**. AMD D3cold characterized (1/boot Vega 20 limit), BrainChip Akida NPU integrated, zero-sudo coralctl

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

## Current State: GCN5 Preswap Complete + Sovereign Sprint (March 2026)

hotSpring is **active at v0.6.32** (848 tests, 0 clippy warnings, 72 experiments).
The sovereign GPU lifecycle is production-grade across 3 vendors + 1 NPU.
**DRM dispatch achieved full GCN5 preswap validation: 6/6 phases PASS** — WGSL →
coral-reef compiler → coral-driver PM4 → MI50 GPU execution → readback verified.
The **Naga DF64 bypass is validated end-to-end on real physics** (f64 Lennard-Jones
force calculation with Newton's 3rd law verified). **18 GCN5 bugs found and fixed**
across the full bring-up: VOP1/VOP3/VOPC opcode translation tables, wave64 dispatch,
GLOBAL segment, SGPR mapping, flat_offset GFX9, OpF2F/OpI2F encoding, f64 literal
VGPR pair materialization, VOP3 fneg/fabs modifier encoding, integer negation in
IAdd3, is_f64_expr type resolution, S_WAITCNT, L1+L2 cache invalidation. 85 coral-reef
tests pass, 0 failures. **Sovereign command submission** is 6/10 layers deep (MMU
page table translation blocker).

### Hardware (biomeGate, March 20, 2026)

| Device | BDF | Role | Boot Driver | Round-trips |
|--------|-----|------|-------------|-------------|
| RTX 5060 (GB206) | varies | Display head | nvidia | — |
| Titan V (GV100) | 0000:03:00.0 | VFIO oracle | vfio-pci | Unlimited |
| Radeon VII (Vega 20) | 0000:4d:00.0 | AMD compute | amdgpu | 1/boot (HW limit) |
| BrainChip AKD1000 | 0000:45:00.0 | NPU inference | akida-pcie | Unlimited |

### Ember Architecture

`coral-ember` is an immortal systemd service holding VFIO fds. `coral-glowplug`
connects via Unix socket, receives duplicated fds via `SCM_RIGHTS`.

- **Zero-sudo**: Users join `coralreef` group → full `coralctl` access via socket (root:coralreef 0660)
- **Driver swaps are atomic**: ember drops fds → unbinds → binds target → reacquires
- **VendorLifecycle trait**: Vendor-specific hooks for each swap stage
- **stabilize_after_bind()**: Post-bind power pinning prevents AMD D3cold drift
- **DRM isolation**: Xorg `AutoAddGPU=false` + udev seat tag removal (61-prefix)

### AMD D3cold — Definitive Analysis (4 boot cycles)

The Vega 20 SMU firmware has a **one-shot reinitialization** property: it can
survive exactly one vfio→amdgpu transition per boot. The second transition
corrupts the SMU mailbox (`trn=2 ACK should not assert`). Four strategies were
tested (SimpleBind, PCI remove/rescan, PM power cycle, stabilize_after_bind);
all succeed on cycle 1, all fail on cycle 2. This is a hardware/firmware
limitation, not a software bug. Mitigations deployed: `amdgpu.runpm=0` kernel
cmdline, pre-boot `reset_method` clearing, bridge power pinning.

### coralReef Delivery Status (1 remaining of original 7)

| Task | Priority | Status |
|------|----------|--------|
| ~~JSON-RPC 2.0 socket protocol~~ | ~~P1~~ | **DELIVERED** (Iter 51-52) |
| ~~Trait-based personality system~~ | ~~P3~~ | **DELIVERED** (Iter 52, `GpuPersonality`) |
| ~~SCM_RIGHTS fd passing~~ | ~~P2~~ | **DELIVERED** (Ember architecture, Mar 19) |
| ~~DRM consumer fence~~ | ~~P4~~ | **DELIVERED** (DRM isolation + Ember preflight, Mar 19) |
| ~~AMD Vega metal (MI50/GFX906)~~ | ~~P1~~ | **DELIVERED** + **D3cold fully characterized** (Mar 20) |
| GP_PUT DMA read (Exp 058) | P2 | **Superseded** by Exp 071 — PFIFO re-init + diagnostic matrix. Root cause: MMU translation, not USERD DMA |
| ~~Privilege model (CAP_SYS_ADMIN)~~ | ~~P3~~ | **DELIVERED** — caps + seccomp + zero-sudo coralctl (Mar 20) |

### toadStool Delivery Status (3 remaining, now UNBLOCKED)

| Task | Priority | Status |
|------|----------|--------|
| GlowPlug socket client | P1 | **UNBLOCKED** — JSON-RPC + Ember + triangle handoff ready |
| VFIO device in sysmon | P2 | Partially done (S150-S152 built VFIO infra) |
| hw-learn GlowPlug health feed | P3 | Pending |

### barraCuda — RegisterMap convergence candidate

IPC-first design works cleanly. hotSpring's 848 tests confirm stability at `32554b0a` (v0.3.6).
`GpuDriverProfile` deprecated → `DeviceCapabilities`. `ShaderTemplate::for_driver_auto` replaces
`for_driver_profile`. `BandwidthTier::NvLink` → `HighBandwidthInterconnect`.
RegisterMap trait and VendorLifecycle trait both dispatch from PCI vendor IDs — candidate
for unified `VendorProfile` in the trio triangle architecture.

### Next Milestones

**DRM dispatch (parallel fast track — Exp 072):**
1. ~~**AMD NOP dispatch**~~ — **PASSED**
2. ~~**GCN5 backend in coral-reef**~~ — **COMPLETE** — `Gcn5` arch, `ShaderModelRdna2` parameterized by GFX version, VOP1/VOP3/VOPC opcode translation (LLVM-validated), wave64 dispatch, GLOBAL segment, ACQUIRE_MEM L2 flush
3. ~~**GCN5 E2E compute dispatch**~~ — **PASSED** — WGSL → coral-reef → coral-driver PM4 → MI50 → 64/64 readback verified. Naga bypass validated.
4. ~~**Preswap Phase A/B/C**~~ — **PASSED** — f64 write, f64 arithmetic, multi-workgroup. 5 compiler bugs fixed.
5. ~~**GLOBAL_LOAD + remaining phases**~~ — **RESOLVED** — 6 additional compiler bugs fixed (VOP1 opcode table, f64 materialization in transcendentals, OpI2F f64 dest, is_f64_expr type resolution, VOP3 fneg/fabs modifiers, integer negation IAdd3). Phases D/E/F all pass.
6. ~~**DF64 Lennard-Jones via DRM**~~ — **PASSED** — f64 LJ force matches CPU reference (tol=1e-8). Newton's 3rd law verified.
7. **K80 NVIDIA DRM** — legacy nouveau `CHANNEL_ALLOC` → `GEM_PUSHBUF` (no PMU needed)
8. **Titan V PMU investigation** — FECS-only channel? Compute-only type? K80 reference data

**Sovereign VFIO (ongoing — Exp 071):**
6. **MMU page table fix** — debug PDE/PTE encoding, verify IOMMU mapping, try BAR2-resident tables
7. **NOP GPFIFO fetch** — once MMU works, verify PBDMA fetches and executes NOP entries
8. **FECS/GPCCS firmware load** — after clean PBDMA dispatch, load compute engine firmware
9. **K80 sovereign** — no firmware signing, full 10-layer pipeline on unlocked hardware

**Infrastructure:**
10. **Ember per-device isolation**: One D3cold device must not freeze all operations
11. toadStool GlowPlug socket client wiring (triangle architecture)
12. RDNA validation (RX 5000/6000/7000 series)

---

## Active Handoffs

### Strategic (action required)

| File | Date | Audience | What To Do |
|------|------|----------|------------|
| [`HOTSPRING_PRESWAP_GLOBAL_LOAD_HANDOFF_MAR21_2026.md`](handoffs/HOTSPRING_PRESWAP_GLOBAL_LOAD_HANDOFF_MAR21_2026.md) | Mar 21 | coralReef, toadStool, barraCuda | **SUPERSEDED** — GLOBAL_LOAD resolved. See `HOTSPRING_GCN5_COMPLETE_PRESWAP_HANDOFF_MAR2026.md` for final 6/6 results. |
| [`HOTSPRING_GCN5_COMPLETE_PRESWAP_HANDOFF_MAR2026.md`](handoffs/HOTSPRING_GCN5_COMPLETE_PRESWAP_HANDOFF_MAR2026.md) | Mar 2026 | coralReef, toadStool, barraCuda | **START HERE.** GCN5 preswap 6/6 PASS — f64 write, f64 arith, multi-workgroup, multi-buffer, HBM2 bandwidth, **f64 LJ force (Newton's 3rd law verified)**. 18 bugs fixed. 85 coral-reef tests. Per-primal action items. |
| [`HOTSPRING_GCN5_E2E_BREAKTHROUGH_HANDOFF_MAR21_2026.md`](handoffs/HOTSPRING_GCN5_E2E_BREAKTHROUGH_HANDOFF_MAR21_2026.md) | Mar 21 | coralReef, toadStool, barraCuda | GCN5 E2E compute dispatch achieved — WGSL → coral-reef → MI50 → 64/64 verified. 7 bugs fixed. VOP3 opcode translation table. Naga bypass validated. Per-primal action items. DF64 Lennard-Jones next (blocked by GLOBAL_LOAD). |
| [`HOTSPRING_DRM_SOVEREIGN_DUAL_TRACK_HANDOFF_MAR21_2026.md`](handoffs/HOTSPRING_DRM_SOVEREIGN_DUAL_TRACK_HANDOFF_MAR21_2026.md) | Mar 21 | coralReef, toadStool, barraCuda | Dual-track strategy: DRM dispatch (AMD PM4 + NVIDIA EXEC) in parallel with sovereign VFIO. GCN5 backend **COMPLETE** (see GCN5 E2E handoff above). K80 incoming. Naga DF64 bypass validated. |
| [`HOTSPRING_PFIFO_MMU_SOVEREIGN_DISPATCH_HANDOFF_MAR21_2026.md`](handoffs/HOTSPRING_PFIFO_MMU_SOVEREIGN_DISPATCH_HANDOFF_MAR21_2026.md) | Mar 21 | coralReef, toadStool, barraCuda | 54-config PFIFO diagnostic matrix, PFIFO re-init sequence (PMC+preempt+clear), root cause analysis (MMU 0xbad00200), 6/10 sovereign pipeline layers proven. Register reference. Per-primal action items. |
| [`HOTSPRING_TRIO_EVOLUTION_AMD_AKIDA_HANDOFF_MAR20_2026.md`](handoffs/HOTSPRING_TRIO_EVOLUTION_AMD_AKIDA_HANDOFF_MAR20_2026.md) | Mar 20 | coralReef, toadStool, barraCuda | Triangle architecture, AMD D3cold definitive resolution (4 strategies, 1 round-trip/boot limit), BrainChip AKD1000 NPU integration, zero-sudo coralctl, per-primal evolution priorities. |
| [`HOTSPRING_VENDOR_LIFECYCLE_AMD_D3COLD_HANDOFF_MAR19_2026.md`](handoffs/HOTSPRING_VENDOR_LIFECYCLE_AMD_D3COLD_HANDOFF_MAR19_2026.md) | Mar 19 | coralReef, toadStool, barraCuda | VendorLifecycle trait, initial AMD D3cold analysis. **Superseded by Mar 20 trio handoff** for strategy conclusions. |
| [`HOTSPRING_VENDOR_AGNOSTIC_HARDENED_GLOWPLUG_HANDOFF_MAR18_2026.md`](handoffs/HOTSPRING_VENDOR_AGNOSTIC_HARDENED_GLOWPLUG_HANDOFF_MAR18_2026.md) | Mar 18 | coralReef, toadStool, barraCuda | Vendor-agnostic RegisterMap, AMD MI50 support, coral-ember crate split, typed EmberError, privilege hardening (caps+seccomp), coralctl deploy-udev. |
| [`HOTSPRING_REGISTER_MAPS_ABSORPTION_HANDOFF_MAR18_2026.md`](handoffs/HOTSPRING_REGISTER_MAPS_ABSORPTION_HANDOFF_MAR18_2026.md) | Mar 18 | barraCuda, toadStool | RegisterMap trait absorption path, GlowPlug register RPCs for hw-learn, sovereign dispatch blockers, AMD vs NVIDIA VFIO lessons. |
| [`HOTSPRING_EMBER_DRM_ISOLATION_HANDOFF_MAR19_2026.md`](handoffs/HOTSPRING_EMBER_DRM_ISOLATION_HANDOFF_MAR19_2026.md) | Mar 19 | coralReef, toadStool | Ember architecture, DRM isolation, fail-safe swap protocol, boot scripts |

### Technical Reference (register maps, firmware, hardware lessons)

| File | Date | Topic |
|------|------|-------|
| [`HOTSPRING_GLOWPLUG_BOOT_PERSISTENCE_SOVEREIGN_PIPELINE_HANDOFF_MAR16_2026.md`](handoffs/HOTSPRING_GLOWPLUG_BOOT_PERSISTENCE_SOVEREIGN_PIPELINE_HANDOFF_MAR16_2026.md) | Mar 16 | GlowPlug architecture, systemd, VFIO-first boot, DRM fencing lesson, reproducibility checklist |
| [`HOTSPRING_SOVEREIGN_FALCON_DIRECT_LOAD_HANDOFF_MAR16_2026.md`](handoffs/HOTSPRING_SOVEREIGN_FALCON_DIRECT_LOAD_HANDOFF_MAR16_2026.md) | Mar 16 | FECS/SEC2 register maps, firmware loading protocol, falcon boot chain, D3hot→D0 clean state |
| [`HOTSPRING_VFIO_D3HOT_VRAM_BREAKTHROUGH_MAR16_2026.md`](handoffs/HOTSPRING_VFIO_D3HOT_VRAM_BREAKTHROUGH_MAR16_2026.md) | Mar 16 | D3hot→D0 VRAM recovery, HBM2 lifecycle, digital PMU, warm detection |
| ~~`HOTSPRING_VFIO_PFIFO_PROGRESS_GP_PUT_HANDOFF_MAR09_2026.md`~~ | Mar 09 | **Archived** — superseded by Mar 21 PFIFO MMU handoff. See `handoffs/archive/`. |

---

## Remaining Work Summary

### Critical Path (blocks sovereign compute dispatch)

| # | Gap | Owner | Reference |
|---|-----|-------|-----------|
| 1 | **MMU page table translation** (PBDMA 0xbad00200) | coralReef/hotSpring | Exp 071, PFIFO MMU handoff Mar 21 |
| 2 | NOP GPFIFO entries (after MMU fix) | coralReef | Exp 071 next steps |
| 3 | GPCCS falcon address on GV100 | hotSpring | Falcon handoff, Part 3 |
| 4 | FECS+GPCCS firmware load | hotSpring | Falcon handoff, Part 5 |
| 5 | FECS halt at PC=0x2835 | hotSpring | Falcon handoff, Part 5 |

### Infrastructure (blocks next-GPU readiness)

| # | Gap | Owner | Reference |
|---|-----|-------|-----------|
| ~~5~~ | ~~AMD Vega metal (MI50)~~ | ~~coralReef~~ | **DELIVERED** + D3cold characterized (1/boot limit, Mar 20) |
| ~~6~~ | ~~SCM_RIGHTS fd passing~~ | ~~coralReef~~ | **DELIVERED** — Ember architecture (Mar 19) |
| ~~7~~ | ~~DRM consumer fence~~ | ~~coralReef~~ | **DELIVERED** — DRM isolation + preflight (Mar 19) |
| 8 | GlowPlug socket client | toadStool (unblocked) | Trio handoff Mar 20, triangle architecture |
| 9 | VFIO device in sysmon | toadStool (partial) | PIN handoff |
| ~~10~~ | ~~Privilege model (CAP_SYS_ADMIN)~~ | ~~coralReef~~ | **DELIVERED** — zero-sudo coralctl (Mar 20) |
| 11 | Ember per-device isolation | coralReef | Single-threaded ember blocks all devices on D3cold hang |
| 12 | RDNA validation | coralReef | AmdRdnaLifecycle untested on actual RDNA hardware |

### Research (long-horizon)

| # | Gap | Owner |
|---|-----|-------|
| 13 | Sovereign HBM2 PHY training (no nouveau) | hotSpring |
| 14 | PRIVRING fault avoidance on GV100 | hotSpring (documented, never toggle PMC bit 12) |
| 15 | VendorProfile convergence (RegisterMap + VendorLifecycle) | trio |

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
removal, preflight checks), **VendorLifecycle trait** (NVIDIA, AMD Vega 20,
AMD RDNA, Intel Xe, BrainChip, Generic), **AMD D3cold resolution** (4 strategies
tested, `PmResetAndBind`, `stabilize_after_bind()`), **BrainChip Akida NPU**
(personality, lifecycle, driver swap), **zero-sudo coralctl** (socket group
permissions).

**Status:** coralReef absorbed and evolved. GlowPlug has JSON-RPC 2.0,
trait-based personalities (8 including Akida), VendorLifecycle dispatch.
Ember provides fail-safe driver hot-swap for GPUs and non-GPU accelerators.
157 tests pass. Next: per-device thread isolation in ember, RDNA validation.

### hotSpring → toadStool (partial)

hotSpring contributed: PrecisionBrain, multi-adapter GPU selection,
StreamingDispatch, WorkloadHealthMonitor, NPU parameter controller,
SPIR-V codegen safety rename.

**Status:** toadStool absorbed precision routing and dispatch patterns.
GlowPlug client wiring is the main remaining integration.

---

## Guidance for Primals

### For coralReef

All original 7 deliverables complete except GP_PUT DMA read. **Immediate priorities:**

1. **Ember per-device isolation**: The single-threaded ember daemon hangs entirely when one
   device enters D3cold. Move sysfs operations to per-device threads with D3cold pre-check
   (read `power_state` before any write — if D3cold, return error instead of blocking).
2. **RDNA validation**: `AmdRdnaLifecycle` uses conservative Vega 20 defaults — test on actual
   RX 5000/6000/7000 hardware to determine if RDNA's FLR support changes the picture.
3. **VendorProfile convergence**: RegisterMap (barraCuda) and VendorLifecycle (coral-ember)
   both dispatch from PCI vendor IDs. Unify into a single `VendorProfile` trait.

**Architecture facts:**
- `coral-ember`: standalone crate, modular `sysfs`, `swap`, `hold`, `ipc`, `vendor_lifecycle`
- `coral-glowplug`: library surface with typed `EmberError` — importable by toadStool
- `swap_device` RPC: single atomic orchestrator with `VendorLifecycle` hooks
- 6 vendor lifecycles: NVIDIA, AMD Vega 20, AMD RDNA, Intel Xe, BrainChip, Generic
- Hardened: capabilities + seccomp + namespaces + `NoNewPrivileges`
- Zero-sudo: `coralreef` group, socket 0660

Register references are in the falcon and PFIFO handoffs. Do NOT PMC-toggle
GR bit 12 on GV100. Do NOT change `boot_personality = "vfio"` in glowplug.toml.

### For toadStool

**Triangle architecture** — toadStool is the hub between coralReef and barraCuda:

1. **GlowPlug socket client**: Connect to `/run/coralreef/glowplug.sock` (no sudo needed if
   user is in `coralreef` group). Methods: `device.list`, `device.swap`, `device.health`,
   `health.check`, `daemon.status`, `daemon.shutdown`.
2. **Lifecycle-aware dispatch**: AMD round-trips are expensive (1/boot for Vega 20).
   Prefer keeping AMD on one personality per session. NVIDIA and Akida are unlimited.
3. **Hardware census via GlowPlug**: Replace manual BDF enumeration with RPC-based discovery.
4. **hw-learn health feed**: Feed GlowPlug health data (VRAM alive/dead, power state,
   domain faults, D3cold detection) into hw-learn for pattern learning.

### For barraCuda

**RegisterMap + VendorLifecycle convergence candidate.** Both traits dispatch from PCI
vendor IDs — consider a unified `VendorProfile` in the triangle architecture.

Pending absorption candidates: Polyakov loop shader, `su3_math_f64.wgsl`, ESN
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

90 superseded handoffs in `handoffs/archive/` (including PIN Mar 16,
V0632 Mar 13, Backend Analysis Mar 17, Comprehensive Audit Mar 17 — all
superseded by the Mar 20 trio handoff or absorbed into README; plus Mar 09
PFIFO GP_PUT and Mar 16 D3hot VRAM, superseded by Mar 21 PFIFO MMU handoff).
12 active handoffs at `handoffs/` root (including Mar 21 GCN5 E2E breakthrough,
dual-track, PFIFO MMU, and preswap/GLOBAL_LOAD handoffs). These document the full
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
