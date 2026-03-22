# wateringHole — Cross-Project Handoffs

**Project:** hotSpring (ecoPrimals)
**Last Updated:** March 22, 2026
**Status:** ACTIVE — Dual-track dispatch: sovereign VFIO (6/10 layers, MMU blocker) + DRM dispatch (**AMD GCN5 preswap: 6/6 PASS — f64 Lennard-Jones force verified, Newton's 3rd law confirmed**). **iommufd/cdev VFIO backend** (kernel 6.2+) — kernel-agnostic VFIO, resolves EBUSY on 6.17. **RTX 5060 Blackwell DRM cracked** (SM120, per-buffer fd, single mmap context). **Kepler (SM35) + Blackwell (SM120) ISA arches** in coral-reef. **Ember swap pipeline proven + hardened** — D-state resilient sysfs (process-isolated watchdog), IOMMU group peer release, EmberClient retry, DRM isolation auto-generation. **VRAM write-readback health check** (eliminates cold-boot false positives). **BDF allowlist** (ember rejects RPCs for unmanaged devices). **Pre-flight device checks** (D3cold/D3hot/0xFFFF config space guard before unbind). **nouveau ↔ vfio round-trip proven** on Titan V (both cards warm-swapped, HBM2 alive). **2× Titan V + RTX 5060** fleet. 74 experiments. 86 ember + 178 glowplug + 848 hotSpring tests. AMD D3cold characterized (1/boot Vega 20 limit), BrainChip Akida NPU integrated, zero-sudo coralctl

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

## Current State: Sovereign MMU Sprint — Ember Swap Pipeline Proven (March 2026)

hotSpring is **active at v0.6.32** (848 tests, 0 clippy warnings, 74 experiments).
The sovereign GPU lifecycle is production-grade across 2 vendors + 1 NPU.
**DRM dispatch achieved full GCN5 preswap validation: 6/6 phases PASS** (prior to MI50
removal) — WGSL → coral-reef → coral-driver PM4 → MI50 → readback verified.
**Ember swap pipeline fully operational**: D-state resilient sysfs (process-isolated
watchdog, 10s timeout), IOMMU group peer release for native driver swaps, EmberClient
retry with exponential backoff, DRM isolation auto-generated from device config at
startup. **nouveau ↔ vfio round-trip proven** on Titan V. **Sovereign command
submission** is 6/10 layers deep (MMU page table translation blocker).

### Hardware (biomeGate, March 22, 2026)

| Device | BDF | Role | Boot Driver | Round-trips |
|--------|-----|------|-------------|-------------|
| RTX 5060 (GB206) | varies | Display head + modern/vendor validator | nvidia | — |
| Titan V #1 (GV100) | 0000:03:00.0 | VFIO sovereign + swap oracle | vfio-pci | Unlimited |
| Titan V #2 (GV100) | 0000:4b:00.0 | VFIO sovereign + MMU diagnostics | vfio-pci | Unlimited |

### Ember Architecture (iommufd/cdev — kernel-agnostic, D-state resilient)

`coral-ember` is an immortal systemd service holding VFIO fds. `coral-glowplug`
connects via Unix socket, receives duplicated fds via `SCM_RIGHTS`.

**iommufd/cdev backend:** On kernel 6.2+ the legacy VFIO container/group API is
deprecated in favor of `iommufd`/`cdev`. `VfioDevice::open()` tries iommufd first,
falls back to legacy. Backend-agnostic: `VfioBackendKind`, `ReceivedVfioFds`,
`sendable_fds()`, `from_received()`. IPC sends 2 fds (iommufd) or 3 fds (legacy)
plus JSON metadata.

**D-state resilient sysfs (March 22, 2026):** Risky sysfs writes (driver/unbind,
bind, remove, rescan) spawn a child process via `/bin/sh`. Parent polls with
`try_wait()` and 10s timeout. If child enters D-state, parent kills it. Daemon stays
responsive. Safe config-space attributes (power/control, reset_method) use direct
writes with no fork overhead.

**IOMMU group peer handling:** Symmetric bind/release for multi-device IOMMU groups.
`release_iommu_group_from_vfio()` unbinds audio peers before native driver swap.
`bind_iommu_group_to_vfio()` reacquires peers on vfio swap.

- **Zero-sudo**: Users join `coralreef` group → full `coralctl` access (root:coralreef 0660)
- **Driver swaps are atomic**: ember drops fds → unbinds → binds target → reacquires
- **VendorLifecycle trait**: Vendor-specific hooks for each swap stage
- **DRM isolation auto-generated**: udev rules + Xorg config from device list at startup
- **EmberClient retry**: 3× backoff for EAGAIN/EINTR, `read_full_response()` for complete JSON
- **Backend-agnostic**: iommufd/cdev (modern) or container/group (legacy) — auto-detected

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

### toadStool Delivery Status (2 remaining)

| Task | Priority | Status |
|------|----------|--------|
| ~~GlowPlug socket client~~ | ~~P1~~ | **DELIVERED** — `glowplug_client.rs` in toadStool server crate. Runtime-discoverable (env/XDG/default), `ember.list`/`ember.status`/`ember.swap`/`ember.reacquire` RPCs, `SharedGlowPlugClient` via `Arc`. Follows `CoralReefClient` pattern. (Mar 22) |
| VFIO device in sysmon | P2 | Partially done (S150-S152 built VFIO infra) |
| hw-learn GlowPlug health feed | P3 | Pending |

### barraCuda — RegisterMap convergence candidate

IPC-first design works cleanly. hotSpring's 848 tests confirm stability at v0.3.7.
`GpuDriverProfile` deprecated → `DeviceCapabilities`. `ShaderTemplate::for_driver_auto` replaces
`for_driver_profile`. `BandwidthTier::NvLink` → `HighBandwidthInterconnect`.
RegisterMap trait and VendorLifecycle trait both dispatch from PCI vendor IDs — candidate
for unified `VendorProfile` in the trio triangle architecture.

### Next Milestones

**DRM dispatch (parallel fast track — Exp 072-073):**
1. ~~**AMD NOP dispatch**~~ — **PASSED**
2. ~~**GCN5 backend in coral-reef**~~ — **COMPLETE** — `Gcn5` arch, `ShaderModelRdna2` parameterized by GFX version, VOP1/VOP3/VOPC opcode translation (LLVM-validated), wave64 dispatch, GLOBAL segment, ACQUIRE_MEM L2 flush
3. ~~**GCN5 E2E compute dispatch**~~ — **PASSED** — WGSL → coral-reef → coral-driver PM4 → MI50 → 64/64 readback verified. Naga bypass validated.
4. ~~**Preswap Phase A/B/C**~~ — **PASSED** — f64 write, f64 arithmetic, multi-workgroup. 5 compiler bugs fixed.
5. ~~**GLOBAL_LOAD + remaining phases**~~ — **RESOLVED** — 6 additional compiler bugs fixed (VOP1 opcode table, f64 materialization in transcendentals, OpI2F f64 dest, is_f64_expr type resolution, VOP3 fneg/fabs modifiers, integer negation IAdd3). Phases D/E/F all pass.
6. ~~**DF64 Lennard-Jones via DRM**~~ — **PASSED** — f64 LJ force matches CPU reference (tol=1e-8). Newton's 3rd law verified.
7. ~~**RTX 5060 Blackwell DRM**~~ — **PIPELINE CRACKED** — NvUvmComputeDevice operational. SM120 class IDs, single-mmap fix, per-buffer-fd fix. 4/4 HW tests pass. ISA compilation pending (SM120 arch enum).
8. ~~**iommufd/cdev VFIO backend**~~ — **COMPLETE** — kernel-agnostic VFIO on 6.2+. Resolves persistent EBUSY on 6.17. Dual-path (iommufd + legacy) across ember/glowplug/driver. 607 tests pass. HW validated on Titan V.
9. ~~**Ember swap pipeline**~~ — **COMPLETE** — D-state resilient sysfs, IOMMU peer release, EmberClient retry, DRM isolation auto-gen. nouveau ↔ vfio round-trip proven on Titan V (Exp 074).
10. **K80 NVIDIA DRM** — legacy nouveau `CHANNEL_ALLOC` → `GEM_PUSHBUF` (no PMU needed)
11. **Titan V PMU investigation** — FECS-only channel? Compute-only type? K80 reference data

**Sovereign VFIO (ongoing — Exp 071):**
6. **MMU page table fix** — debug PDE/PTE encoding, verify IOMMU mapping, try BAR2-resident tables
7. **NOP GPFIFO fetch** — once MMU works, verify PBDMA fetches and executes NOP entries
8. **FECS/GPCCS firmware load** — after clean PBDMA dispatch, load compute engine firmware
9. **K80 sovereign** — no firmware signing, full 10-layer pipeline on unlocked hardware

**Infrastructure:**
10. ~~**Ember per-device isolation**~~: **DELIVERED** — `Arc<RwLock<HashMap>>` per-client threading, D3cold pre-checks in `reacquire`/`swap` (Mar 22)
11. ~~toadStool GlowPlug socket client~~ — **DELIVERED** — `glowplug_client.rs` with runtime discovery (Mar 22)
12. RDNA validation (RX 5000/6000/7000 series)

---

## Active Handoffs

### Strategic (action required)

| File | Date | Audience | What To Do |
|------|------|----------|------------|
| [`HOTSPRING_EMBER_HARDENING_HANDOFF_MAR22_2026.md`](handoffs/HOTSPRING_EMBER_HARDENING_HANDOFF_MAR22_2026.md) | Mar 22 | coralReef, toadStool, barraCuda | **START HERE.** VRAM write-readback health check, BDF allowlist (ember rejects unmanaged devices), pre-flight device state validation (D3cold/0xFFFF guard). 264 tests. Both Titans warm-swapped, HBM2 alive. Ember FD sharing validated. |
| [`HOTSPRING_EMBER_WATCHDOG_SWAP_PIPELINE_HANDOFF_MAR22_2026.md`](handoffs/HOTSPRING_EMBER_WATCHDOG_SWAP_PIPELINE_HANDOFF_MAR22_2026.md) | Mar 22 | coralReef, toadStool, barraCuda | D-state resilient ember, IOMMU peer swap, DRM isolation auto-gen, EmberClient retry. nouveau ↔ vfio round-trip proven. 2× Titan V + RTX 5060 fleet. |
| [`HOTSPRING_IOMMUFD_EMBER_EVOLUTION_HANDOFF_MAR22_2026.md`](handoffs/HOTSPRING_IOMMUFD_EMBER_EVOLUTION_HANDOFF_MAR22_2026.md) | Mar 22 | coralReef, toadStool, barraCuda | Kernel-agnostic VFIO backend: iommufd/cdev on 6.2+, legacy fallback. Ember/GlowPlug/Driver all evolved. Per-primal evolution items. |
| [`HOTSPRING_DRM_TRIO_PIPELINE_HANDOFF_MAR22_2026.md`](handoffs/HOTSPRING_DRM_TRIO_PIPELINE_HANDOFF_MAR22_2026.md) | Mar 22 | coralReef, toadStool, barraCuda | Three-GPU DRM pipeline: MI50 E2E (historical — MI50 since removed), RTX 5060 Blackwell DRM cracked, Titan V VFIO staged. iommufd Part 4 appended. |
| [`HOTSPRING_GCN5_COMPLETE_PRESWAP_HANDOFF_MAR2026.md`](handoffs/HOTSPRING_GCN5_COMPLETE_PRESWAP_HANDOFF_MAR2026.md) | Mar 2026 | coralReef, toadStool, barraCuda | GCN5 preswap 6/6 PASS — f64 write, f64 arith, multi-workgroup, multi-buffer, HBM2 bandwidth, **f64 LJ force (Newton's 3rd law verified)**. 18 bugs fixed. 85 coral-reef tests. Per-primal action items. |
| [`HOTSPRING_PFIFO_MMU_SOVEREIGN_DISPATCH_HANDOFF_MAR21_2026.md`](handoffs/HOTSPRING_PFIFO_MMU_SOVEREIGN_DISPATCH_HANDOFF_MAR21_2026.md) | Mar 21 | coralReef, toadStool, barraCuda | 54-config PFIFO diagnostic matrix, PFIFO re-init sequence (PMC+preempt+clear), root cause analysis (MMU 0xbad00200), 6/10 sovereign pipeline layers proven. Register reference. Per-primal action items. |
| [`HOTSPRING_TRIO_EVOLUTION_AMD_AKIDA_HANDOFF_MAR20_2026.md`](handoffs/HOTSPRING_TRIO_EVOLUTION_AMD_AKIDA_HANDOFF_MAR20_2026.md) | Mar 20 | coralReef, toadStool, barraCuda | Triangle architecture, AMD D3cold definitive resolution (4 strategies, 1 round-trip/boot limit), BrainChip AKD1000 NPU integration, zero-sudo coralctl, per-primal evolution priorities. |
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
| ~~8~~ | ~~GlowPlug socket client~~ | ~~toadStool~~ | **DELIVERED** — `glowplug_client.rs` runtime-discoverable client (Mar 22) |
| 9 | VFIO device in sysmon | toadStool (partial) | PIN handoff |
| ~~10~~ | ~~Privilege model (CAP_SYS_ADMIN)~~ | ~~coralReef~~ | **DELIVERED** — zero-sudo coralctl (Mar 20) |
| ~~11~~ | ~~Ember per-device isolation~~ | ~~coralReef~~ | **DELIVERED** — `Arc<RwLock<HashMap>>` per-client threading + D3cold guards (Mar 22) |
| 12 | RDNA validation | coralReef | AmdRdnaLifecycle untested on actual RDNA hardware |
| 13 | SM32 encoder `frnd.f32` | coralReef | Kepler codegen ICE on float rounding — add to instruction table |

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
**iommufd/cdev backend**: VfioDevice dual-path (iommufd first, legacy fallback),
backend-agnostic Ember→GlowPlug IPC (2-fd iommufd or 3-fd legacy + JSON metadata).
**Per-client threading**: `Arc<RwLock<HashMap>>` with D3cold pre-checks (Mar 22).
**Ember hardened (Mar 22)**: VRAM write-readback canary replaces read-nonzero check,
BDF allowlist rejects RPCs for unmanaged devices, pre-flight device checks
(sysfs existence, D0 power state, config space 0xFFFF guard). Display GPU safety
guard prevents unbind of active display devices. 86 ember + 178 glowplug tests pass.
**ISA arches**: `NvArch::Sm35` (Kepler) + `NvArch::Sm120` (Blackwell) with full
`wave_size` propagation through `CompiledKernel`/`KernelCacheEntry`/`ShaderInfo`.
Next: RDNA validation, SM32 encoder `frnd.f32` gap, VendorProfile convergence.

### hotSpring → toadStool (partial)

hotSpring contributed: PrecisionBrain, multi-adapter GPU selection,
StreamingDispatch, WorkloadHealthMonitor, NPU parameter controller,
SPIR-V codegen safety rename.

**Status:** toadStool absorbed precision routing and dispatch patterns.
GlowPlug client wiring is the main remaining integration.

---

## Guidance for Primals

### For coralReef

All original 7 deliverables complete except GP_PUT DMA read. **iommufd/cdev evolution shipped (Mar 22). Kepler/Blackwell ISA + ember threading shipped (Mar 22).** **Immediate priorities:**

1. ~~**SM120 ISA arch**~~: **DELIVERED** — `NvArch::Sm120` (Blackwell) and `NvArch::Sm35` (Kepler) added to coral-reef. Full propagation through coral-gpu: `sm_to_nvarch()`, `vfio_sm_from_device_id()`, PCI device ID ranges (GK110/GK210 for Kepler, GB20x for Blackwell). `wave_size` field propagated through `CompiledKernel`, `KernelCacheEntry`, `ShaderInfo`. SM32 encoder has `frnd.f32` gap for Kepler — codegen evolution needed.
2. ~~**Ember per-device isolation**~~: **DELIVERED** — `Arc<RwLock<HashMap>>` with `std::thread::spawn` per client. `sysfs::is_d3cold()` pre-check in `reacquire`/`swap` prevents operations on powered-off devices. Granular read/write locks per RPC method. 14 tests pass.
3. **RDNA validation**: `AmdRdnaLifecycle` uses conservative Vega 20 defaults — test on actual
   RX 5000/6000/7000 hardware to determine if RDNA's FLR support changes the picture.
4. **VendorProfile convergence**: RegisterMap (barraCuda) and VendorLifecycle (coral-ember)
   both dispatch from PCI vendor IDs. Unify into a single `VendorProfile` trait.
5. **SM32 encoder `frnd.f32`**: Kepler codegen triggers ICE on float rounding instructions. Add `FRND` to SM32 encoder instruction table.

**Architecture facts:**
- `coral-ember`: standalone crate, modular `sysfs`, `swap`, `hold`, `ipc`, `vendor_lifecycle`. **Per-client threaded** (`std::thread::spawn` per connection, `Arc<RwLock<HashMap>>` shared state)
- `coral-glowplug`: library surface with typed `EmberError` — importable by toadStool
- `coral-reef`: `NvArch` enum covers SM35 (Kepler) → SM120 (Blackwell). `wave_size` metadata flows through compilation pipeline
- `VfioDevice`: dual-path iommufd/cdev (kernel 6.2+) + legacy container/group (older kernels)
- IPC: `SCM_RIGHTS` sends 2 fds (iommufd) or 3 fds (legacy) + JSON `backend`/`ioas_id`
- `swap_device` RPC: single atomic orchestrator with `VendorLifecycle` hooks
- 6 vendor lifecycles: NVIDIA, AMD Vega 20, AMD RDNA, Intel Xe, BrainChip, Generic
- Hardened: capabilities + seccomp + namespaces + `NoNewPrivileges`
- Zero-sudo: `coralreef` group, socket 0660

Register references are in the falcon and PFIFO handoffs. Do NOT PMC-toggle
GR bit 12 on GV100. Do NOT change `boot_personality = "vfio"` in glowplug.toml.

### For toadStool

**Triangle architecture** — toadStool is the hub between coralReef and barraCuda:

1. ~~**GlowPlug socket client**~~: **DELIVERED** — `glowplug_client.rs` in `toadstool-server` crate.
   Runtime discovery: `CORALREEF_EMBER_SOCKET` env → `$XDG_RUNTIME_DIR/coralreef/ember.sock` → `/run/coralreef/ember.sock`.
   RPCs: `ember.list`, `ember.status`, `ember.swap`, `ember.reacquire`. `SharedGlowPlugClient` via `Arc<GlowPlugClient>`.
   **Next**: Wire into sysmon for hardware census, connect to hw-learn for health feed.
2. **Lifecycle-aware dispatch**: AMD round-trips are expensive (1/boot for Vega 20).
   Prefer keeping AMD on one personality per session. NVIDIA and Akida are unlimited.
3. **Hardware census via GlowPlug**: Wire `GlowPlugClient::list_devices()` into sysmon substrate discovery.
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

94 superseded handoffs in `handoffs/archive/` (including PIN Mar 16,
V0632 Mar 13, Backend Analysis Mar 17, Comprehensive Audit Mar 17 — all
superseded by the Mar 20 trio handoff or absorbed into README; plus Mar 09
PFIFO GP_PUT and Mar 16 D3hot VRAM, superseded by Mar 21 PFIFO MMU handoff;
plus Mar 19 Vendor Lifecycle AMD D3cold, Mar 21 GCN5 E2E Breakthrough,
Mar 21 DRM Sovereign Dual-Track, Mar 21 Preswap Global Load — all superseded
by later handoffs).
10 active handoffs at `handoffs/` root (including Mar 22 ember hardening,
iommufd evolution, DRM trio pipeline, ember watchdog swap; plus GCN5 complete
preswap, PFIFO MMU, trio evolution, vendor-agnostic hardened glowplug,
register maps absorption, and ember DRM isolation). These document the full
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
