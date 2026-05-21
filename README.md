# hotSpring

**Computational physics reproduction studies and control experiments.**

Named for the hot springs that gave us *Thermus aquaticus* and Taq polymerase — the origin story of the constrained evolution thesis. Professor Murillo's research domain is hot dense plasmas. A spring is a wellspring. This project draws from both.

---

## What This Is

hotSpring is where we reproduce published computational physics work from the Murillo Group (MSU) and benchmark it across consumer hardware. Every study has two phases:

- **Phase A (Control)**: Run the original Python code (Sarkas, mystic, TTM) on our hardware. Validate against reference data. Profile performance. Fix upstream bugs. **✅ Complete — 86/86 quantitative checks pass.**

- **Phase B (BarraCuda)**: Re-execute the same computation on ToadStool's BarraCuda engine — pure Rust, WGSL shaders, any GPU vendor. **✅ L1 validated (478× faster, better χ²). L2 validated (1.7× faster).**

- **Phase C (GPU MD)**: Run Sarkas Yukawa OCP molecular dynamics entirely on GPU using f64 WGSL shaders. **✅ 9/9 PP Yukawa DSF cases pass on RTX 4070. 0.000% energy drift at 80k production steps. Up to 259 steps/s sustained. 3.4× less energy per step than CPU at N=2000.**

- **Phase D (Native f64 Builtins + N-Scaling)**: Replaced software-emulated f64 transcendentals with hardware-native WGSL builtins. **✅ 2-6× throughput improvement. N=10,000 paper parity in 5.3 minutes. N=20,000 in 10.4 minutes. Full sweep (500→20k) in 34 minutes. 0.000% energy drift at all N. The f64 bottleneck is broken — double-float (f32-pair) on FP32 cores delivers 3.24 TFLOPS at 14-digit precision (9.9× native f64).**

- **Phase E (Paper-Parity Long Run + Toadstool Rewire)**: 9-case Yukawa OCP sweep at N=10,000, 80k production steps — matching the Dense Plasma Properties Database exactly. **✅ 9/9 cases pass, 0.000-0.002% energy drift, 3.66 hours total, $0.044 electricity. Cell-list 4.1× faster than all-pairs. Toadstool GPU ops (BatchedEighGpu, SsfGpu, PppmGpu) wired into hotSpring.**

- **Phase F (Kokkos-CUDA Parity + Verlet Neighbor List)**: Runtime-adaptive algorithm selection (AllPairs/CellList/VerletList) with DF64 precision on consumer GPUs. **✅ 9/9 cases pass, ≤0.004% drift. Verlet achieves 992 steps/s (κ=3) — gap vs Kokkos-CUDA closed from 27× to 3.7×. barraCuda v0.6.32.**

- **Phase G (Universal Substrate Deployment)**: guideStone-certified artifact deployable on any OS, any architecture, any filesystem. **✅ 59/59 checks x 5 substrates. Cross-architecture parity (x86_64 + aarch64, bit-identical). OCI container image. Windows WSL2/Docker + macOS Docker launchers. exFAT tmpdir fallback. `./hotspring` unified ecoBin entry point. benchScale 5-substrate validation (40/40 cross-substrate parity).**

hotSpring answers: *"Does our hardware produce correct physics?"*, *"Can Rust+WGSL replace the Python scientific stack?"*, and *"Can IPC-composed NUCLEUS primals reproduce what standalone Rust proves?"*

### Philosophical Evolution: Vendor Agnostic → Vendor Atheistic → Silicon Deistic

There is only math, energy, and silicon. Everything else is an abstraction.

The project's trajectory is to progressively eliminate abstraction layers between mathematical intent and physical compute:

- **Vendor agnostic** (current): abstraction layers support multiple vendors — AMD, NVIDIA, NPU. The `BootPipeline` trait, `dispatch_mode`, and WGSL shader portability.
- **Vendor atheistic** (in progress): no dependency on vendor toolchains, drivers, or firmware. Sovereign boot, sovereign compile (WGSL→native ISA via coralReef), sovereign dispatch (VFIO+PBDMA). The vendor's driver is not needed, not wanted, not trusted.
- **Silicon deistic** (target): only math, energy, and silicon exist. Rust compiles to machine instructions. WGSL compiles to native ISA. Instructions execute on transistors. No runtime, no interpreter, no VM, no driver ABI, no vendor firmware — just the laws of physics running on crystalline silicon. The entire software stack from `cargo build` to GPU register writes is a controlled, auditable chain from equation to electron.

### Eukaryotic UniBin: `hotspring_unibin`

hotSpring has evolved from the prokaryotic era of separate binaries into a eukaryotic UniBin — a single `hotspring_unibin` binary consolidating certification (L0–L6 guideStone organelle), validation scenarios (18 default / 24 with `barracuda-local`), and status reporting. Reference: primalSpring v0.9.25 interstadial eukaryotic wave.

```
hotspring certify              # L0-L6 composition certification
hotspring certify --bare       # L0 only, no primals needed
hotspring validate             # run all validation scenarios
hotspring validate --track nuclear-physics
hotspring validate --tier rust  # Tier 1 only (no IPC)
hotspring validate --list      # list all scenarios
hotspring status               # composition health summary
hotspring version              # version info
```

### guideStone Status: Level 6 — CERTIFIED (NUCLEUS Deployment Validation)

hotSpring is the reference implementation for the guideStone Composition Standard (primalSpring v0.9.25, guideStone v1.2.0). The guideStone is a self-validating deployable that carries its own benchmark — all six certified properties are satisfied:

| Property | Evidence |
|----------|----------|
| **1. Deterministic** | Same binary, same results. Cross-substrate parity (Python/CPU/GPU). `validation/` artifact: 59/59 checks × 5 substrates. |
| **2. Reference-traceable** | Every value traces to a paper or proof via `BaselineProvenance` / `AnalyticalProvenance`. DOIs for AME2020, Chabanat, Kortelainen, Bender, Lattimer & Prakash. |
| **3. Self-verifying** | BLAKE3 CHECKSUMS manifest verified via `primalspring::checksums::verify_manifest()`. Tampered inputs → non-zero exit. |
| **4. Environment-agnostic** | ecoBin compliant, static musl, no sudo, no network, no GPU required for core validation. |
| **5. Tolerance-documented** | 308 named constants in `tolerances/` module tree with physical/mathematical derivations. |
| **6. NUCLEUS deployment-validated** | **Level 6 — CERTIFIED**: IPC-composed NUCLEUS deployment validation complete; primals and composition patterns match direct Rust execution under the same tolerance methodology. |

**Validation ladder**: Python baseline (L1) → Rust proof (L2, DONE) → barraCuda CPU (L3) → barraCuda GPU (L4) → guideStone composition (L5) → **NUCLEUS deployment validation (L6) — CERTIFIED**.

**Pre-flight**: `hotspring_unibin certify` certifies composition correctness (6 layers). The legacy `hotspring_guidestone` binary is transitional — use `hotspring_unibin certify` instead.

**plasmidBin Deployment**: NUCLEUS primals ship as musl-static genomeBin binaries (46 binaries across 6 target triples, primalSpring v0.9.25) via `primals/biomeOS/plasmidBin/`. No compilation needed — deploy with `nucleus_launcher.sh --composition niche-hotspring`, then run `hotspring_guidestone` against the live stack. See `scripts/validate-primal-proof.sh` for the end-to-end workflow (auto-sets BEARDOG_FAMILY_SEED, SONGBIRD_SECURITY_PROVIDER, NESTGATE_JWT_SECRET).

**Composition Template (Phase 46)**: `tools/hotspring_composition.sh` implements event-driven QCD computation via the NUCLEUS composition library. Async tick model (convergence-based, not 60Hz), DAG memoization for parameter sweeps, ledger-sealed reproducible runs, and scientific provenance braids for peer-review audit. Run with `COMPOSITION_NAME=hotspring ./tools/hotspring_composition.sh` (requires NUCLEUS primals) or test in bare mode (graceful degradation, no crash).

> **For the physics**: See [`PHYSICS.md`](PHYSICS.md) for complete equation documentation
> with numbered references — every formula, every constant, every approximation.
>
> **For the methodology**: See [`whitePaper/METHODOLOGY.md`](whitePaper/METHODOLOGY.md)
> for the two-phase validation protocol and acceptance criteria.

---

## Paper Baseline Notebooks (13)

Publishable-grade Jupyter notebooks reproducing 25 peer-reviewed physics papers.
Live Python compute for small problems, frozen data for production simulations.
See `notebooks/papers/PAPER_NOTEBOOK_GUIDE.md` for the collaborator pattern.

| # | Notebook | Paper/Domain | Compute |
|---|----------|-------------|---------|
| 01 | SEMF Binding Energy | Chabanat (1998), AME2020 | Live |
| 02 | Yukawa Screening | Murillo & Weisheit (1998) | Live |
| 03 | Sarkas Yukawa MD | Stanton & Murillo (2016) | Live small + frozen |
| 04 | TTM Laser-Plasma | Chen et al. (2001) | Live |
| 05 | Transport Coefficients | Daligault (2012) | Live analytical |
| 06 | Surrogate Learning | Diaw et al. (2024) | Live demo + frozen |
| 07 | Quenched QCD | Wilson (1974), Creutz (1980) | Live 4^4 |
| 08 | Dynamical Fermions | Gottlieb (1987), HVP | Live small + frozen |
| 09 | Abelian Higgs | Bazavov (2015) | Live 8x8 |
| 10 | Spectral Theory | Anderson (1958), Hofstadter (1976) | Live |
| 11 | Gradient Flow | Luscher (2010), Chuna (2021) | Live 4^4 |
| 12 | Plasma Dielectric | Chuna & Murillo (2024) | Live |
| 13 | LTEE Anderson Fitness | Anderson & Wiser (2024) | Live statistics |

## Deploy Graphs (7)

Domain-specific NUCLEUS deployment profiles in `graphs/`:

| # | Graph | Composition | Domain |
|---|-------|-------------|--------|
| 1 | `hotspring_qcd_deploy` | Full NUCLEUS + skunkBat + Squirrel | Lattice QCD / HPC |
| 2 | `hotspring_plasma_md_deploy` | Tower + Node + skunkBat | Yukawa OCP, transport coefficients |
| 3 | `hotspring_md_deploy` | Tower + Node + Nest + skunkBat | GPU MD — Yukawa OCP, Sarkas validation |
| 4 | `hotspring_plasma_deploy` | Tower + Node + Nest + skunkBat | Dense plasma — dielectric, kinetic-fluid coupling |
| 5 | `hotspring_nuclear_eos_deploy` | Tower + Node + Nest + skunkBat | SEMF/HFB binding energies |
| 6 | `hotspring_spectral_deploy` | Tower + barraCuda + skunkBat | Anderson, Hofstadter, Lanczos |
| 7 | `hotspring_sovereign_gpu_deploy` | Full NUCLEUS + skunkBat | Sovereign GPU WGSL-to-SASS |

Deploy: `biomeos deploy --graph graphs/<name>.toml`

## Current Status (2026-05-21)

> **218 experiments** | **500+ quantitative checks** | **~$0.30 total science cost** | **700 (cylinder) / 596 (barracuda default) / 1,045 (barracuda-local) lib tests, 167 binaries, 65 validation suites (3 tiers: smoke/nucleus/silicon), 128 WGSL shaders** | **deny.toml** (ecoBin C-dep bans) | **21 RPC methods** (`sovereign.snapshot`, `sovereign.compare` added) | **zero `dyn` dispatch, `#[forbid(unsafe_code)]` on lib (unsafe confined to low-level GPU experiment bins), `#[expect]` over `#[allow]`** | **guideStone artifact: 59/59 checks x 5 substrates (x86_64 + aarch64)** | **OCI container image + Windows/macOS launchers** | **Fleet: 2× Titan V (GV100) + RTX 5060 (Blackwell)** — RTX 5060 dispatch PROVEN (8/8), **Titan V VFIO Tier 1 sovereign** (Exp 209-217: anchor-fd adoption, CE runlist discovery, PBDMA pipeline validated, PMU path closed, binary-patch warm handoff proven, sovereign driver rotation codified, pipeline consolidated, TPC wall confirmed firmware-dependent Exp 217) | **Sovereignty Tier Model** — Tier 0 (cold/vendor wall), Tier 1 (warm infrastructure — **validated**), Tier 2 (warm compute — **TPC PRI wall**, nvidia-470 nvsov dual-load path **active** Exp 218), Tier 3 (full sovereign) | **Vendor-atheistic target**: sovereign compute across generations (Volta GV100 → Blackwell SM120) — not just agnostic to vendor, but independent of vendor entirely | **Warm-catch pipeline in pure Rust** (`toadstool device warm-catch <BDF>` — ELF patcher + ember orchestrator) | **Vendor-agnostic `BootPipeline` trait** (KeplerInit + VoltaInit + VegaInit) | **SLM pool allocation (2 MiB)** | **AMD sovereign compiler: 24/24 QCD shaders** | **NVIDIA sovereign compiler: SM35 + SM70 + SM120** | **Ember gate + survivability hardening COMPLETE** | **SovereignInit Pipeline COMPLETE** | **NUCLEUS Composition Evolution COMPLETE** | **coralReef f64 transcendental lowering (SM32+)** | **Level 6 — CERTIFIED (NUCLEUS Deployment Validation)** | **GPU Generation Profile Architecture** | **unsafe audit: all NECESSARY** | **Diesel Engine Architecture: toadStool boot-time GPU management via plasmidBin ecoBin** | **Compile-then-dispatch pipeline (coralReef→toadStool) wired** | **Circuit-breaker discovery + TOML-driven capability aliases** | **PLX D3cold keepalive VALIDATED** (PlxKeepalive + PlxGuardian) | **VBIOS interpreter live HW validated** (Exp 204) | **PowerSafetyProfile** (K80 fire post-mortem → generation-aware PMC_ENABLE staging) | **Sovereign Boot Abstraction** — `SovereignBootState` enum (unified warm/cold model), `WarmKeepalive` facade, `sovereign.profile` RPC with µs-precision timing + register snapshots, twin-card profiling experiments on dual Titan V (Exp 207) | **Hardware Line Codified** — cold boot = power-on reset = boot ROM trains HBM2 = same wall vendor faces; warm keepalive systemd fd store prevents transitions back to cold | **Warm Keepalive PROVEN** (Exp 208) — **183ms warm pipeline** (76× faster than cold), falcon warm preservation eliminates 3.7s ACR re-boot, fd store end-to-end validated, GPUs stay warm across `systemctl restart`, cold early-exit 200ms | **GPC Boundary Analysis** (Exp 210) — PTOP_DEVICE_INFO_V2 parser fixed, CE runlist discovered, sovereignty tier model codified, all engine domains power-gated after nouveau unbind
>
> **Local Debt Resolution + Composition Evolution (2026-05-14):** Seven-item sprint resolving fragile composition patterns as hotSpring relies more on toadStool/coralReef/barraCuda IPC. **Compile-then-dispatch pipeline:** `compile_and_submit()` chains coralReef `shader.compile.wgsl` → toadStool `compute.dispatch.submit` with compiled `binary_b64` — fixes yukawa/plaquette validation failures. Legacy name-based `submit_workload()` deprecated. **Circuit-breaker discovery:** `PrimalEndpoint` gains `fail_count`/`dead_since`, `NucleusContext` gains `record_failure()`/`record_success()`/`maybe_reprobe()`/`refresh()` — 3 failures = mark dead, 30s cooldown before re-probe. `call_tracked()` for lifecycle-aware IPC. **Dispatch surface unification:** `compute_dispatch.rs` is now canonical for dispatch; `fleet_toadstool.rs` submit/dispatch deprecated; `glowplug_client/` module (types.rs + mod.rs) clarifies device-management-only scope. **FusedPipeline error handling:** `submit()` returns `FusedSubmitReport` with per-op `FusedOpSubmitOutcome::Submitted(id)` / `Failed(msg)` — no more fake `"error:{e}"` job IDs. **JSON-RPC helper:** `parse_jsonrpc_response()` replaces scattered `.get("result")` patterns with typed error handling (code/message extraction). **TOML-loaded aliases:** `PRIMAL_ALIASES` loaded from `config/capability_registry.toml` at runtime via `OnceLock`, compiled defaults as fallback. **Validation infra:** `validate_all` supports `--tier smoke|nucleus|silicon`, uses pre-built binaries from `target/release/`, 65 suites (35 smoke, 7 nucleus, 23 silicon). **596/596 lib tests pass (default). Zero clippy warnings.**
>
> **Compute Trio Rewire + Deep Debt Capability Evolution (2026-05-12):** Completed GAP-HS-087 (Compute Trio Rewire Sprint) and GAP-HS-088 (Deep Debt Capability Discovery). **Trio Rewire:** Local `PrecisionTier` (4) and `PhysicsDomain` (12) replaced with barraCuda upstream 15-tier/15-variant canonical enums via re-exports. `toadstool-dispatch` feature flag with `ToadStoolDispatchClient` in `fleet_toadstool.rs` — parallel IPC migration path for Phase C ember→toadStool cutover. `HardwareHint` field in `PrecisionRoute` for domain-based hardware routing. `validate_compute_trio_pipeline` binary: Yukawa force + Wilson plaquette through full barraCuda→coralReef→toadStool→hardware chain. Barrier shader validation: 9 WGSL shaders using `workgroupBarrier()` cataloged for coralReef `membar.{cta,gl}` emitter. **Deep Debt:** `detect_sovereign_available()` inverted to NUCLEUS `by_domain("shader")`-first (env vars as fallback). IPC provenance clients (`sweetgrass`, `rhizocrypt`, `loamspine`, `skunkbat`) evolved from hardcoded `biomeos/*.sock` paths to NUCLEUS `by_domain()` capability discovery. `certification/deployment.rs` `REQUIRED_PRIMALS` derived from `niche::DEPENDENCIES` (single source of truth). `compute_dispatch.rs` barrier validation uses `call_by_capability("shader", ...)` instead of direct socket IPC. `toadstool_report.rs` uses `by_domain("compute")` + `call_by_capability` for performance reporting. `low_level/bar0.rs` BAR0 map size discovered from file metadata (not hardcoded 16 MiB), sysfs path overridable via `HOTSPRING_SYSFS_PCI`. `Vec<&String>` → `Vec<&str>` in fleet_client.rs. PCI vendor IDs extracted to named constants. **1,045/1,045 lib tests pass (barracuda-local). Zero clippy warnings.** IPC transport evolution (GAP-HS-092): all IPC modules now use `call_by_capability()` for unified discovery + transport. `TierCapability::failed()`/`compiled_only()` constructors reduce calibration boilerplate.
>
> **Three-Tier Validation Architecture (2026-04-17):** Python baselines → Rust validation → NUCLEUS primal composition validation. **guideStone bare mode: 30/30 checks pass** (Property 3 BLAKE3 CHECKSUMS verified, deny.toml present, all 5 bare properties green). Only 3 SKIPs remain — expected NUCLEUS liveness probes when no primals deployed. The same tolerance-driven, exit-code-gated methodology that proved Rust matches Python now proves IPC-composed NUCLEUS patterns match direct Rust execution. Composition validators (`validate_nucleus_*`) run standalone (skip-pass for CI, exit 2 = all skipped) or against live primals (full IPC validation). `validate_science_probes()` validates compute, math, and provenance trio capabilities via IPC with Rust baseline parity. Pattern documented for sibling spring adoption in wateringHole handoffs.
>
> **Composition Evolution Wave 1-3 (2026-04-11, refined 2026-04-17):** Absorbed hardened patterns from primalSpring. **Wave 1 (contract):** `niche.rs` split `CAPABILITIES` into `LOCAL_CAPABILITIES` (21 served) + `ROUTED_CAPABILITIES` (26 proxied with canonical providers). `register_with_target()` sends `lifecycle.register` + `capability.register` to biomeOS. `plasmidBin/hotspring/metadata.toml` upgraded to full schema (`[provenance]`, `[compatibility]`, `[builds.*]`, `[genomeBin]`). `manifest.lock` entry added. **Wave 2 (validation harness):** `CompositionResult` with `check_skip()`, `exit_code_skip_aware()` (0/1/2), `ValidationSink` trait, `NdjsonSink`, `StdoutSink`. `OrExit<T>` trait for zero-panic binary patterns. **Wave 3 (hardening):** Cost estimate literals extracted to `tolerances::cost`. `config/capability_registry.toml` with bidirectional sync test. `HOTSPRING_NO_NUCLEUS=1` standalone mode. `cargo clippy --all-targets` clean. `cargo doc --lib --no-deps` clean. 993/993 lib tests pass (historical — current: 596/1,045).
>
> **Deep Debt Evolution Phase 2 (2026-05-08):** Smart refactoring wave: `pseudofermion/mod.rs` (926L) → split Hasenbusch mass-preconditioning into `hasenbusch.rs`. `npu_worker/handlers.rs` (839L) → split into `handlers/{mod,precompute,thermalization,inference,proxy}.rs`. `nuclear_eos_helpers/mod.rs` (821L) → display functions extracted to `display.rs`. Unsafe evolution: `exp070_register_dump.rs` mmap wrapped in `SafeBarMapping` struct with `Drop` impl (RAII munmap, bounds-checked accessors). Hardcoding elimination: `toadstool_report.rs` socket resolution migrated to `niche::socket_dirs()`, deprecated primal_bridge named accessors stripped of hardcoded name fallbacks. Test coverage: 9 new unit tests (primal_bridge + receipt_signing), 3 new integration tests (dielectric, spectral, lattice). Benchmark: nuclear EOS + spectral domains added to `validate_barracuda_cpu_gpu_parity` (8 domains total). Paper 45 `kinetic_fluid_control.json` committed. Tier 4 verified: `validate_fpeos` 18/19, `validate_atomec` 7/9. Downstream repos (`projectNUCLEUS`, `foundation`) cloned and audited → `docs/DOWNSTREAM_PATTERNS.md`. **1002/1002 lib tests pass (historical — current: 596/1,045).** Zero compilation errors.
>
> **Deep Debt Evolution Phase 1 (2026-04-27):** Capability-based primal discovery — `composition.rs` derives all primal requirements from `niche::DEPENDENCIES` (single source of truth), eliminating hardcoded name→domain maps. `primal_bridge.rs` named accessors (`toadstool()`, `beardog()`, etc.) deprecated in favor of `by_domain()`. Data-driven `PRIMAL_ALIASES` table replaces hardcoded alias checks. Smart file refactoring: `lattice/rhmc.rs` (989L) → `rhmc/mod.rs` (802L) + `rhmc/remez.rs` (190L, Remez exchange + Gauss elimination). `nuclear_eos_helpers.rs` (978L) → `nuclear_eos_helpers/mod.rs` (824L) + `objectives.rs` (174L, L1/L2 optimization). Pre-existing compile errors fixed in `nuclear_eos_l2_ref.rs` and `nuclear_eos_l2_hetero.rs` (upstream `DiscoveredDevice` API). 993/993 lib tests pass (historical — current: 596/1,045). Zero compilation errors.
>
> **Phase 46 Composition Template (2026-04-27):** Absorbed `primalSpring` Phase 46 composition patterns. `tools/hotspring_composition.sh` implements event-driven QCD computation lane with 5 domain hooks: async tick model (convergence-based, not fixed-rate), DAG memoization for parameter sweeps (`VERTEX_STACK`, `BRANCH_STACK`), scientific provenance via `sweetGrass` braids, compute dispatch through `toadStool`/`barraCuda`, ledger sealing via `loamSpine`. `tools/nucleus_composition_lib.sh` (41-function NUCLEUS wiring library) copied from primalSpring. Bare mode verified: all library functions gracefully degrade when primals absent.
>
> **NUCLEUS Composition Validation (2026-04-10):** Phase 2 transition complete — Rust+Python baselines now serve as validation targets for ecoPrimal NUCLEUS patterns. Four binaries (`validate_nucleus_composition`, `validate_nucleus_tower`, `validate_nucleus_node`, `validate_nucleus_nest`) prove atomic compositions via JSON-RPC IPC. `composition.rs` provides atomic health probes (Tower/Node/Nest/FullNucleus) and science probes. `mcp_tools.rs` exposes 5 MCP tool definitions. `harvest-ecobin.sh` builds musl-static binaries for `infra/plasmidBin/`.
>
> **SovereignInit Pipeline (Exp 164-165, 2026-04-08):** Full nouveau replacement pipeline implemented in pure Rust. `SovereignInit` orchestrates 8 stages: HBM2 Training (VBIOS DEVINIT via interpreter, auto cold/warm detection) → PMC Engine Gating → Topology Discovery (GPC/TPC/SM/FBP/PBDMA) → PFB Memory Controller → Falcon Boot Chain (SEC2→ACR→FECS/GPCCS solver, 15 strategies) → GR Engine Init (firmware BAR0 writes + FECS method probe) → PFIFO Discovery → GR Context Setup (optional, FECS bind + golden save). New entry point: `NvVfioComputeDevice::open_sovereign(bdf)` — zero nouveau, zero DRM, just Rust + VFIO + firmware blobs as ingredients. GR init functions extracted to standalone module. `SovereignInitResult` reports `compute_ready()` with structured diagnostics. 429 coral-driver tests pass, 176 toadstool-ember tests pass.
>
> **Ember Firmware Intermediary (2026-04-08):** Ember now replaces nouveau as the firmware management authority. Three new RPCs: `ember.firmware.inventory` (probes /lib/firmware/nvidia/{chip}/ per subsystem), `ember.firmware.load` (loads+validates ACR/GR firmware blobs), `ember.sovereign.init` (runs full 8-stage SovereignInit pipeline via fork-isolated MMIO). Firmware treated as ingredients — loaded by Rust, executed by GPU hardware. 40 RPC methods total. Pattern scales to any future GPU: add firmware recipe → ember manages lifecycle.
>
> **Firmware Boundary Pivot (Exp 159-163, 2026-04-07):** Architectural pivot — falcon firmware (PMU, SEC2, FECS, GPCCS) is the GPU's internal OS, to be interfaced with, not replaced. **NOP dispatch via nouveau DRM: SUCCEEDED** in both raw C and pure Rust. Pipeline: `VM_INIT → CHANNEL_ALLOC(VOLTA_COMPUTE_A) → SYNCOBJ → GEM_NEW → VM_BIND → mmap → EXEC → SYNCOBJ_WAIT`. PMU mailbox protocol mapped (register-based on GV100: MBOX0/MBOX1 + IRQSSET). `PmuInterface` struct created in coral-driver. Hot-handoff channel injection proven (CH 500 accepted by scheduler alongside nouveau). HBM2 training preserved through nouveau warm-cycle + `reset_method` clear. Firmware-agnostic interface pattern scales Kepler→Volta→Turing→GSP.
>
> **Ember Survivability Hardening (Exp 140-151, 2026-04-07):** Three-phase hardening eliminated all known lockup vectors. Fork-isolated MMIO, zero-I/O recovery, FdVault checkpoint/restore, GPU warm cycle resurrection. **Validated**: 8 consecutive fault runs — zero lockups. **Fleet:** 2× Titan V (GV100) + RTX 5060 (Blackwell). K80 retired (original caught fire Exp 199 → pulled, replaced by second Titan V).
>
> **SEC2 ACR Boot Investigation (Exp 141-151):** SEC2 falcon starts and executes BL code but does not achieve HS mode. Root cause narrowing: VBIOS DEVINIT contradicted (Exp 142-143). Crash vector hunt (Exp 150) isolated PRAMIN as lockup trigger. Cold VRAM detection graceful. **Superseded by DRM path** — firmware-agnostic DRM interface bypasses the ACR HS barrier entirely.
>
> **coralReef Deep Debt + Ember Evolution:** Deep Debt Plan (P1-P7) complete. Ember Survivability Hardening (3 phases, 12 tasks) complete. All toadstool-ember tests pass (170 pass, 4 ignored). All toadstool glowplug tests pass (285 lib + 3 doc). Both services deployed and validated. Key new capabilities: `ember.warm_cycle` RPC, `sysfs_warm_cycle` in glowplug resurrection, `FdVault` checkpoint/restore in ember lifecycle, `guarded_sysfs_read` with timeout.
>
> **Universal Substrate Deployment (April 2026):** guideStone artifact validated across 5 substrates — CPU-only Ubuntu, NVIDIA GPU, AMD GPU, Alpine musl, aarch64 qemu-user. Cross-architecture parity: 40/40 observable comparisons bit-identical between x86_64 and aarch64. OCI container image (`hotspring-guidestone.tar`) enables deployment on Windows (WSL2/Docker), macOS (Docker/Podman), and any Linux without ext4.
>
> **NVIDIA Sovereign Compute Breakthrough (2026-03-30):** RTX 3090 GPFIFO command submission pipeline **fully operational** through coralReef's sovereign driver. Key fixes via `ioctl` interception of CUDA: `NV906F_CTRL_CMD_BIND`, TSG scheduling, `GET_WORK_SUBMIT_TOKEN` via Volta class (0xC36F), VRAM USERD, 48-byte RM_ALLOC on 580.x GSP-RM.
>
> **AMD Sovereign Compute — Local Memory Breakthrough (2026-03-30):** Three-layer fix unlocks per-thread scratch memory on RDNA2. AMD sovereign compiler: 24/24 QCD shaders compiled (WGSL → native GFX10.3 ISA). 38/39 dispatch tests pass.
>
> **Science (Exp 096-103):** GPU RHMC production (Nf=2, Nf=2+1), gradient flow at volume (5 LSCFRK integrators), self-tuning RHMC (zero hand-tuned parameters). Silicon saturation profiling complete (Exp 105-106).
>
> See [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) for the full validation table and benchmark data.

| Domain | Status | Highlights |
|--------|--------|------------|
| **Dense Plasma MD** (Sarkas, 12 cases) | ✅ 60/60 | 9 PP Yukawa + 3 PPPM, paper-parity at N=10k |
| **Surrogate + Nuclear EOS** | ✅ 39/39 | BarraCuda 478× faster (χ²=2.27), HFB GPU, AME2020 |
| **Transport** (Stanton-Murillo) | ✅ 13/13 | GPU-only Green-Kubo D*/η*/λ* |
| **Lattice QCD** (quenched + dynamical) | ✅ 46/46 | HMC, Dirac CG, plaquette, SU(3) + U(1) Higgs |
| **GPU RHMC** (Nf=2, Nf=2+1) | ✅ Complete | True multi-shift CG, fermion force validated, ΔH=O(1), 8.5 GFLOP/s |
| **Gradient Flow** (Chuna 43) | ✅ Complete | 5 LSCFRK integrators, CK4 stability, t₀/w₀ |
| **Self-Tuning RHMC** | ✅ Complete | Zero hand-tuned parameters — spectral + acceptance-driven |
| **Spectral Theory** (Kachkovskiy) | ✅ 45/45 | Anderson 1D/2D/3D, Hofstadter, GPU Lanczos |
| **NPU** (AKD1000 hardware) | ✅ 34/35 | 10 SDK assumptions overturned, physics pipeline, phase detection |
| **Sovereign GPU** (toadStool + coralReef compile) | ✅ **Tier 1 Validated** | **Evolution: agnostic → atheistic (infra) → atheistic (compute) → deistic.** RTX 5060 dispatch live via DRM (12/12 QCD/HMC/MD on SM120). **Dual Titan V** VFIO sovereign Tier 1 proven (Exp 210-217): PFIFO, PBDMA, FECS, topology, CE channel — all working; TPC PRI stations are the Tier 2 wall (firmware-mediated, Exp 217). **Active path**: nvidia-470 nvsov dual-load injection (Exp 218) — proprietary driver seeds TPC init, warm swap preserves state. coralReef: SM70 + SM120 + GFX10.3. **183ms warm pipeline** (Exp 208). 21 RPC methods. See `SILICON_DEISM.md` for the full abstraction elimination thesis. |
| **Silicon Characterization** | ✅ Complete | TMU, ROP, L2, shader cores — AMD vs NVIDIA personalities |
| **Silicon Saturation Profiling** | ✅ Complete | TMU PRNG, subgroup reduce, ROP atomics, capacity analysis |
| **Chuna Papers 43-45** | ✅ **44/44** | Gradient flow + BGK dielectric + kinetic-fluid coupling |

Full validation table (218 rows) with per-experiment details: [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md)

### Science Ladder

Quenched SU(3) ✅ → Gradient Flow ✅ → LSCFRK Integrators ✅ → N_f=4 Infra ✅ → Chuna 44/44 ✅ → **N_f=2 ✅** → **N_f=2+1 ✅** → **Self-tuning ✅** → **True multi-shift CG ✅** → **Fermion force validated ✅** → **Silicon saturation profiling ✅** → **Sovereign NVIDIA GPFIFO ✅** → **AMD sovereign compiler 24/24 ✅** → **AMD scratch/local memory ✅** → **Livepatch warm handoff ✅** → **Dual GPU sovereign boot ✅** → **Deep Debt Evolution complete ✅** → **Sacrificial Ember Architecture ✅** → **Firmware Boundary Pivot ✅** → **NOP Dispatch (pure Rust DRM) ✅** → **GPU Generation Profile Architecture ✅** → **RTX 5060 sovereign dispatch ✅** → **K80 warm NOP dispatch ✅** → **K80 cold-boot PLL fix ✅** → **Titan V DMATRF FECS IMEM load ✅** → **SLM pool allocation ✅** → **unsafe audit (all NECESSARY) ✅** → **Warm-catch breakthrough (binary-patched nouveau) ✅** → **ALL 3 GPUs sovereign (Exp 190) ✅** → **Pure Rust warm-catch pipeline (toadstool device warm-catch) ✅** → **Sovereign Init RPC (Exp 197) ✅** → **Vendor-Agnostic BootPipeline + VBIOS Fixes (Exp 198) ✅** → **Diesel Engine Sovereign Boot (Exp 199) ✅** → **PowerSafetyProfile — K80 fire post-mortem (Exp 200) ✅** → **Volta Cold Boot CG Sweep (Exp 201) ✅** → **Bore-Agnostic Surface Rewire (Exp 202) ✅** → **Warm/Cold Boot Convergence + PLL activation (Exp 203) ✅** → **VBIOS Interpreter Live HW Validation (Exp 204) ✅** → **Dual Titan V Twin Study Baseline (Exp 205) ✅** → **Falcon ACR DMA Boot Solved (Exp 206) ✅** → **Sovereign Boot Abstraction + Profiling (Exp 207) ✅** → **Reboot-Efficient Sovereign Evolution — Warm Keepalive PROVEN (Exp 208) ✅** → **Sovereign VFIO Dispatch Bridge (Exp 209) ✅** → **GPC Boundary Analysis — Sovereignty Tier Model (Exp 210) ✅** → **PMU Mailbox Tier 2 Investigation (Exp 211) ⏳** → era-agnostic sovereign abstraction (coralReef + toadStool + barracuda trio). Cross-cutting sovereign validation matrix: [`specs/SOVEREIGN_VALIDATION_MATRIX.md`](specs/SOVEREIGN_VALIDATION_MATRIX.md).

## Evolution Architecture: Write → Absorb → Lean

hotSpring is a biome. ToadStool (barracuda) is the fungus — it lives in
every biome. hotSpring, neuralSpring, desertSpring each lean on toadstool
independently, evolve shaders and systems locally, and toadstool absorbs
what works. Springs don't reference each other — they learn from each other
by reviewing code in `ecoPrimals/`, not by importing.

```
hotSpring writes extension    → toadstool absorbs    → hotSpring leans on upstream
─────────────────────────       ──────────────────       ────────────────────────
Local GpuCellList (v0.5.13)  → CellListGpu fix (S25) → Deprecated local copy
Complex64 WGSL template      → complex_f64.wgsl      → First-class barracuda primitive
SU(3) WGSL template          → su3.wgsl              → First-class barracuda primitive
Wilson plaquette design       → plaquette_f64.wgsl    → GPU lattice shader
HMC force design             → su3_hmc_force.wgsl    → GPU lattice shader
Abelian Higgs design         → higgs_u1_hmc.wgsl     → GPU lattice shader
NAK eigensolve workarounds   → batched_eigh_nak.wgsl → Upstream shader
ReduceScalar feedback        → ReduceScalarPipeline  → Rewired in v0.5.12
Driver profiling feedback    → GpuDriverProfile      → Rewired in v0.5.15
```

**The cycle**: hotSpring implements physics on CPU with WGSL templates embedded
in the Rust source. Once validated, designs are handed to toadstool via
`ecoPrimals/wateringHole/handoffs/`. Toadstool absorbs them as GPU shaders. hotSpring
then rewires to use the upstream primitives and deletes local code. Each cycle
makes the upstream library richer and hotSpring leaner.

**What makes code absorbable**:
1. WGSL shaders in dedicated `.wgsl` files (loaded via `include_str!`)
2. Clear binding layout documentation (binding index, type, purpose)
3. Dispatch geometry documented (workgroup size, grid dimensions)
4. CPU reference implementation validated against known physics
5. Tolerance constants in `tolerances/` module tree (not inline magic numbers)
6. Handoff document with exact code locations and validation results

**Next absorption targets** (see `barracuda/ABSORPTION_MANIFEST.md`):
- Staggered Dirac shader — `lattice/dirac.rs` + `WGSL_DIRAC_STAGGERED_F64` (8/8 checks, Tier 1)
- CG solver shaders — `lattice/cg.rs` + 3 WGSL shaders (9/9 checks, Tier 1)
- Pseudofermion HMC — `lattice/pseudofermion/` (heat bath, force, combined leapfrog; 7/7 checks, Tier 1)
- ESN reservoir + readout — `md/reservoir/` (GPU+NPU validated, Tier 1)
- HFB shader suite — potentials + density + BCS bisection (14+GPU+6 checks, Tier 2)
- NPU substrate discovery — `metalForge/forge/src/probe.rs` (local evolution)

**Already leaning on upstream** (v0.6.32, synced to barraCuda v0.3.11 + toadStool S261 + coralReef Sprint 9+, wgpu 28, pollster 0.4, bytemuck 1.25, tokio 1.50):

toadStool **S261** adds `health.drain`, Kepler dispatch, VFIO IPC surface. coralReef **Sprint 9** adds HMMA GEMM codegen, subgroup ops, `health.version` RPC. barraCuda **Sprint 23** landed the f64 precision fix (production numerical parity on mixed pipelines). **Composition wired via compile-then-dispatch**: coralReef compiles WGSL → toadStool dispatches binary.

| Module | Upstream | Status |
|--------|----------|--------|
| `spectral/` | `barracuda::spectral::*` | **✅ Leaning** — 41 KB local deleted, re-exports + `CsrMatrix` alias |
| `md/celllist.rs` | `barracuda::ops::md::CellListGpu` | **✅ Leaning** — local `GpuCellList` deleted |

**Absorption-ready inventory** (v0.6.32):

| Module | Type | WGSL Shader | Status |
|--------|------|------------|--------|
| `lattice/dirac.rs` | Dirac SpMV | `WGSL_DIRAC_STAGGERED_F64` | (C) Ready — 8/8 checks |
| `lattice/cg.rs` | CG solver | `WGSL_COMPLEX_DOT_RE_F64` + 2 more | (C) Ready — 9/9 checks |
| `lattice/pseudofermion/` | Pseudofermion HMC | CPU (WGSL-ready pattern) | (C) Ready — 7/7 checks |
| `md/reservoir/` | ESN | `esn_reservoir_update.wgsl` + readout | (C) Ready — NPU validated |
| `physics/screened_coulomb.rs` | Sturm eigensolve | CPU only | (C) Ready — 23/23 checks |
| `physics/hfb_deformed_gpu/` | Deformed HFB | 5 WGSL shaders | (C) Ready — GPU-validated |

---

## BarraCuda Crate (v0.6.32)

The `barracuda/` directory is a standalone Rust crate providing the validation
environment, physics implementations, and GPU compute. Key architectural properties:

- **596 / 1,045 tests** (lib; **IPC-first default** / **barracuda-local**), **167 binaries**, **65 validation suites** (3 tiers: `smoke`/`nucleus`/`silicon` via `validate_all --tier`), **128 WGSL shaders** (all AGPL-3.0-only),
  **16 determinism tests** (rerun-identical for all stochastic algorithms). Includes
  lattice QCD (complex f64, SU(3), Wilson action, HMC, Dirac CG, pseudofermion HMC),
  Abelian Higgs (U(1) + Higgs, HMC), transport coefficients (Green-Kubo D*/η*/λ*,
  Sarkas-calibrated fits), HotQCD EOS tables, NPU quantization parity (f64→f32→int8→int4),
  and NPU beyond-SDK hardware capability validation. Test coverage: **74.9% region /
  83.8% function** (spectral tests upstream in barracuda; GPU modules require hardware
  for higher coverage). Measured with `cargo-llvm-cov`.
- **AGPL-3.0 only** — all `.rs` files and all 128 `.wgsl` shaders have
  `SPDX-License-Identifier: AGPL-3.0-only` on line 1.
- **Provenance** — centralized `BaselineProvenance` records trace hardcoded
  validation values to their Python origins (script path, git commit, date,
  exact command). `AnalyticalProvenance` references (DOIs, textbook citations)
  document mathematical ground truth for special functions, linear algebra,
  MD force laws, and GPU kernel correctness. All nuclear EOS binaries and
  library test modules source constants from `provenance::SLY4_PARAMS`,
  `NMP_TARGETS`, `L1_PYTHON_CHI2`, `MD_FORCE_REFS`, `GPU_KERNEL_REFS`, etc.
  DOIs for AME2020, Chabanat 1998, Kortelainen 2010, Bender 2003,
  Lattimer & Prakash 2016 are documented in `provenance.rs`.
- **Tolerances** — 308 centralized constants in the `tolerances/` module tree (6 submodules: physics 66, lattice 98, cost 32, core 38, md 51, npu 23) with physical
  justification (machine precision, numerical method, model, literature).
  Includes 12 physics guard constants (`DENSITY_FLOOR`, `SPIN_ORBIT_R_MIN`,
  `COULOMB_R_MIN`, `BCS_DENSITY_SKIP`, `DEFORMED_COULOMB_R_MIN`, etc.),
  8 solver configuration constants (`HFB_MAX_ITER`, `BROYDEN_WARMUP`,
  `BROYDEN_HISTORY`, `CELLLIST_REBUILD_INTERVAL`, etc.),
  plus validation thresholds for transport, lattice QCD, Abelian Higgs,
  NAK eigensolve, PPPM, screened Coulomb, spectral theory, ESN heterogeneous
  pipeline, NPU quantization, and NPU beyond-SDK hardware capabilities.
  Zero inline magic numbers — all validation binaries and solver loops wired to `tolerances::*`.
- **ValidationHarness** — structured pass/fail tracking with exit code 0/1.
  56 of 167 binaries use it (validation targets). Remaining binaries are optimization
  explorers, benchmarks, and diagnostics.
- **Shared data loading** — `data::EosContext` and `data::load_eos_context()`
  eliminate duplicated path construction across all nuclear EOS binaries.
  `data::chi2_per_datum()` centralizes χ² computation with `tolerances::sigma_theo`.
- **Typed errors** — `HotSpringError` enum with full `Result` propagation
  across all GPU pipelines, HFB solvers, and ESN prediction. Variants:
  `NoAdapter`, `NoShaderF64`, `DeviceCreation`, `DataLoad`, `Barracuda`,
  `GpuCompute`, `InvalidOperation`, `IoError`, `JsonError`.   **Zero `.unwrap()` and zero unannotated `.expect()`
  in library code** — `#![deny(clippy::expect_used, clippy::unwrap_used)]` enforced crate-wide;
  all fallible operations use `?` propagation. Documented binary-helper paths use `.expect()` with
  `#[expect(clippy::expect_used, reason = "...")]`. Provably
  unreachable byte-slice conversions annotated with SAFETY comments.
- **Shared physics** — `hfb_common.rs` consolidates BCS v², Coulomb exchange
  (Slater), CM correction, Skyrme t₀, Hermite polynomials, and Mat type.
  Shared across spherical, deformed, and GPU HFB solvers.
- **GPU helpers centralized** — `GpuF64` provides `upload_f64`, `read_back_f64`,
  `dispatch`, `create_bind_group`, `create_u32_buffer` methods. All shader
  compilation routes through ToadStool's `WgslOptimizer` with `GpuDriverProfile`
  for hardware-accurate ILP scheduling (loop unrolling, instruction reordering).
  No duplicate GPU helpers across binaries.
- **Zero duplicate math** — all linear algebra, quadrature, optimization,
  sampling, special functions, statistics, and spin-orbit coupling use
  BarraCuda primitives (`SpinOrbitGpu`, `compute_ls_factor`).
- **Plaquette variance** — `plaquette_variance` delegated to barraCuda.
- **Capability-based discovery** — runtime adapter enumeration by memory/capability
  (`discover_best_adapter`, `discover_primary_and_secondary_adapters`). Supports nvidia proprietary,
  NVK/nouveau, RADV, and any Vulkan driver. Buffer limits derived from
  `adapter.limits()`, not hardcoded. Data paths resolved via `HOTSPRING_DATA_ROOT`
  or directory discovery.
- **Capability-based primal routing** — `call_by_capability()` routes all functional IPC calls by capability domain, not primal identity.
- **NaN-safe** — all float sorting uses `f64::total_cmp()`.
- **Zero external commands** — pure-Rust ISO 8601 timestamps (Hinnant algorithm),
  no `date` shell-out. `nvidia-smi` calls degrade gracefully.
- **Unsafe confined** — library crate enforces `#![forbid(unsafe_code)]`. Low-level GPU experiment binaries (`exp169`–`exp184`, gated behind `required-features = ["low-level"]`) use audited `unsafe` for direct BAR0 mmap; production and science code is fully safe.
- **NUCLEUS composition** — `niche.rs` declares proto-nucleate (`downstream_manifest.toml`), capabilities, and dependencies.
  `composition.rs` validates atomic health (Tower/Node/Nest/FullNucleus) via IPC; `squirrel_client.rs` wires the Squirrel IPC client for primal communication.
  `mcp_tools.rs` exposes 5 MCP tool schemas for AI/LLM integration.
  `hotspring_unibin serve` JSON-RPC server serves `health.*`, `capability.*`, `composition.*`, `physics.*`, `compute.*`, and `mcp.tools.list` — all 13 physics/compute methods fully dispatched with `catch_unwind` safety. ecoBin packaging via `scripts/harvest-ecobin.sh`.
  Composition tolerances centralized: `COMPOSITION_SEMF_PARITY_REL` (1e-10) and `COMPOSITION_PLAQUETTE_PARITY_ABS` (1e-12) for science parity probes.
- **Quality gates**: Zero clippy warnings (lib), `#![forbid(unsafe_code)]` on lib (unsafe audited and confined to low-level experiment bins), zero `dyn` dispatch in production code, `#[expect(lint, reason)]` in all production binaries, `deny.toml` enforced (ecoBin C-dep bans + async-trait ban), 6 scoped EVOLUTION markers (4 B2 GPU-resident migration + 2 GPU HFB deformed), all files <1000 lines, AGPL-3.0-only consistent.

```bash
cd barracuda
cargo test               # 596 / 1,045 tests (lib; IPC-first default / barracuda-local), 6 ignored (~120s; spectral tests upstream)
cargo clippy --all-targets  # Zero warnings (pedantic + nursery via Cargo.toml workspace lints)
cargo doc --no-deps      # Full API documentation — 0 warnings
cargo run --release --bin validate_all  # 65 suites (--tier smoke|nucleus|silicon)
```

See [`barracuda/CHANGELOG.md`](barracuda/CHANGELOG.md) for version history.

---

## Quick Start

```bash
# Full regeneration — clones repos, downloads data, sets up envs, runs everything
# (~12 hours, ~30 GB disk space, GPU recommended)
bash scripts/regenerate-all.sh

# Or step by step:
bash scripts/regenerate-all.sh --deps-only   # Clone + download + env setup (~10 min)
bash scripts/regenerate-all.sh --sarkas      # Sarkas MD: 12 DSF cases (~3 hours)
bash scripts/regenerate-all.sh --surrogate   # Surrogate learning (~5.5 hours)
bash scripts/regenerate-all.sh --nuclear     # Nuclear EOS L1+L2 (~3.5 hours)
bash scripts/regenerate-all.sh --ttm         # TTM models (~1 hour)
bash scripts/regenerate-all.sh --dry-run     # See what would be done

# Or manually:
bash scripts/clone-repos.sh       # Clone + patch upstream repos
bash scripts/download-data.sh     # Download Zenodo archive (~6 GB)
bash scripts/setup-envs.sh        # Create Python environments
```

```bash
# guideStone Primal Proof (bare mode — no NUCLEUS required)
./scripts/validate-primal-proof.sh

# guideStone Primal Proof (full mode — against live NUCLEUS from plasmidBin)
export FAMILY_ID="hotspring-validation"
export BEARDOG_FAMILY_SEED="$(head -c 32 /dev/urandom | xxd -p)"
cd ../plasmidBin && ./nucleus_launcher.sh --family-id "$FAMILY_ID" --composition niche-hotspring
cd ../hotSpring && ./scripts/validate-primal-proof.sh --full
```

```bash
# Phase C: GPU Molecular Dynamics (requires SHADER_F64 GPU)
cd barracuda
cargo run --release --bin sarkas_gpu              # Quick: kappa=2, Gamma=158, N=500 (~30s)
cargo run --release --bin sarkas_gpu -- --full    # Full: 9 PP Yukawa cases, N=2000, 30k steps (~60 min)
cargo run --release --bin sarkas_gpu -- --long    # Long: 9 cases, N=2000, 80k steps (~71 min, recommended)
cargo run --release --bin sarkas_gpu -- --paper   # Paper parity: 9 cases, N=10k, 80k steps (~3.66 hrs)
cargo run --release --bin sarkas_gpu -- --scale   # GPU vs CPU scaling
```

### What gets regenerated

All large data (21+ GB) is gitignored but fully reproducible:

| Data | Size | Script | Time |
|------|------|--------|------|
| Upstream repos (Sarkas, TTM, Plasma DB) | ~500 MB | `clone-repos.sh` | 2 min |
| Zenodo archive (surrogate learning) | ~6 GB | `download-data.sh` | 5 min |
| Sarkas simulations (12 DSF cases) | ~15 GB | `regenerate-all.sh --sarkas` | 3 hr |
| TTM reproduction (3 species) | ~50 MB | `regenerate-all.sh --ttm` | 1 hr |
| **Total regeneratable** | **~22 GB** | `regenerate-all.sh` | **~12 hr** |

Upstream repos are pinned to specific versions and automatically patched:
- **Sarkas**: v1.0.0 + 3 patches (NumPy 2.x, pandas 2.x, Numba 0.60 compat)
- **TTM**: latest + 1 patch (NumPy 2.x `np.math.factorial` removal)

---

## Directory Structure

```
hotSpring/
├── README.md                           # This file
├── PHYSICS.md                          # Complete physics documentation (equations + references)
├── EXPERIMENT_INDEX.md                 # Full validation table, benchmark data
├── CHUNA_PARITY_STATUS.md             # Chuna paper parity tracking
├── CHUNA_REVIEW.md                    # Chuna paper review notes
├── LICENSE                             # AGPL-3.0
├── Dockerfile                          # OCI container image (Ubuntu 22.04 + Vulkan)
├── .gitignore
│
├── validation/                         # guideStone deployment artifact (v0.7.0)
│   ├── hotspring                      # Unified ecoBin entry point (./hotspring validate|benchmark|...)
│   ├── hotspring.bat                  # Windows launcher (WSL2 → Docker fallback)
│   ├── _lib.sh                        # Shared functions (integrity, arch/GPU/OS detect, container dispatch)
│   ├── GUIDESTONE.md                  # guideStone certification spec
│   ├── README                         # Artifact documentation (quick start, deployment matrix)
│   ├── CHECKSUMS                      # BLAKE3 source-integrity manifest (15 files)
│   ├── bin/
│   │   ├── x86_64/
│   │   │   ├── static/               # musl binaries (CPU-only, any Linux)
│   │   │   └── gpu/                   # glibc binaries (GPU-capable, Vulkan dlopen)
│   │   └── aarch64/
│   │       └── static/               # musl binaries (CPU-only, ARM Linux)
│   ├── container/
│   │   └── hotspring-guidestone.tar   # OCI container image (Docker/Podman)
│   └── results/                       # Validation + benchmark results (per-host)
│
├── whitePaper/                         # Public-facing study documents
│   ├── README.md                      # Document index
│   ├── STUDY.md                       # Main study — full writeup
│   ├── BARRACUDA_SCIENCE_VALIDATION.md # Phase B technical results
│   ├── CONTROL_EXPERIMENT_SUMMARY.md  # Phase A quick reference
│   ├── METHODOLOGY.md                # Two-phase validation protocol
│   ├── TECHNICAL_SUMMARY_FEB2026.md  # Technical summary snapshot
│   └── baseCamp/                      # Per-domain research briefings (19 docs — see baseCamp/README.md)
│       ├── murillo_plasma.md          # Murillo Group — dense plasma MD (Papers 1-6)
│       ├── murillo_lattice_qcd.md     # Lattice QCD — quenched & dynamical (Papers 7-12)
│       ├── kachkovskiy_spectral.md    # Spectral theory — Anderson, Hofstadter
│       ├── cross_spring_evolution.md  # Cross-spring shader ecosystem (164+ shaders)
│       ├── sovereign_gpu_compute.md   # GlowPlug, DRM, ACR, SovereignInit
│       ├── neuromorphic_silicon.md    # AKD1000 NPU — silicon behavior, cross-substrate ESN
│       ├── nucleus_composition_evolution.md  # NUCLEUS primal composition — three-tier validation
│       └── ...                        # 11 more: Chuna, self-tuning RHMC, ESN, reality ladder, etc.
│
├── CHANGELOG.md                        # Root changelog (spring-level changes)
│
├── graphs/                             # biomeOS deploy graphs (NUCLEUS composition deployment)
│   ├── hotspring_qcd_deploy.toml      # Primary deploy graph (10 primals, bonding policy)
│   └── README.md                      # Deploy graph documentation
│
├── docs/                               # Active documentation (7 files)
│   ├── PRIMAL_GAPS.md                # NUCLEUS composition gaps (handback to primalSpring)
│   ├── PRIMAL_PROOF_IPC_MAPPING.md   # Level 6: domain science → primal IPC method mapping
│   ├── DEGRADATION_BEHAVIOR.md       # Graceful degradation when primals absent
│   ├── CROSS_TIER_PARITY.md          # Cross-tier validation parity (default vs barracuda-local)
│   ├── DOWNSTREAM_PATTERNS.md        # Downstream repo patterns (projectNUCLEUS, foundation)
│   ├── DEPENDENCY_AUDIT.md           # Dependency audit and pin tracking
│   └── BASELINE_PROVENANCE_CATALOG.md # Provenance baseline catalog
│
├── barracuda/                          # BarraCuda Rust crate (596 / 1,045 lib tests, 167 binaries, 128 WGSL shaders)
│   ├── Cargo.toml                     # Dependencies (requires ecoPrimals/barraCuda)
│   ├── CHANGELOG.md                   # Version history
│   ├── ABSORPTION_MANIFEST.md         # Write → Absorb → Lean tracking
│   └── src/
│       ├── niche/                     # Self-knowledge (mod.rs: runtime, tables.rs: static capability data)
│       ├── composition.rs             # NUCLEUS atomic health probes and capability routing
│       ├── compute_dispatch/          # GPU dispatch (mod.rs: core, fused.rs: FusedPipeline)
│       ├── glowplug_client/           # toadStool RPC client (mod.rs: impl, types.rs: protocol types)
│       ├── mcp_tools.rs              # MCP tool schemas for AI/LLM integration
│       └── bin/                       # 167 binaries (validation, production, benchmarks, composition, guideStone)
│           └── _fossilized/          # hotspring_primal.rs (superseded by hotspring_unibin)
│
├── experiments/                        # 218 experiment journals (fossil record); 001-190 archived under experiments/archive/
│   ├── archive/                        # experiments 001-190 (archived journals + FOSSIL_RECORD summaries)
│   └── 191-218: active experiments (toadStool PBDMA, HW validation, PLX keepalive, cold/warm boot, driver lab, warm swap, diesel engine, power safety, CG sweep, surface rewire, warm/cold convergence, VBIOS live validation, dual Titan V, Falcon ACR DMA, sovereign boot abstraction + profiling, reboot-efficient sovereign evolution, VFIO dispatch bridge, GPC boundary, PMU mailbox Tier 2, sovereignty consolidation, live HW warm handoff, D-state hardening, TPC wall identification, kernel health preflight, TPC PRI station creation, nvidia-470 nvsov dual-load)
│
├── wateringHole/                       # Lab artifacts, handoffs, mmiotraces
│   ├── handoffs/                      # Dated evolution handoff documents for upstream teams
│   │   └── archive/                  # Older handoffs (May 7-11 era, fossil record)
│   └── mmiotraces/                    # GPU mmiotrace captures
│
├── scripts/                            # Build, regeneration, deployment, boot scripts
│   ├── boot/                          # Systemd units, wake scripts, install scripts
│   ├── validate-primal-proof.sh       # Primal proof validation (bare + NUCLEUS modes)
│   ├── build-guidestone.sh            # Build guideStone artifact (dual-arch, container, launchers)
│   ├── build-container.sh             # Build + export OCI container image
│   ├── prepare-usb.sh                 # Prepare USB liveSpore (ext4/exFAT modes)
│   ├── harvest-ecobin.sh             # ecoBin musl-static build + plasmidBin submission
│   ├── ci-coverage-gate.sh           # CI coverage threshold enforcement (90% line)
│   └── regenerate-all.sh             # Full science regeneration pipeline
│
├── sporeprint/                         # SporePrint / primals.eco publishing
├── tools/                              # Composition scripts, helpers
├── notebooks/                          # Jupyter notebooks (Phase A baselines)
├── specs/                              # Specifications, requirements, gap trackers
├── control/                            # Python control scripts (by domain)
├── metalForge/                         # Hardware characterization (GPU, NPU, nodes)
├── benchmarks/                         # Kokkos/LAMMPS parity, protocol
└── data/                               # Reference data (gitignored large files)
```

---

## Key Documents

| Document | Purpose |
|----------|---------|
| [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) | Full validation table, benchmark data, studies, document index |
| [`PHYSICS.md`](PHYSICS.md) | Complete physics documentation — every equation, constant, approximation |
| [`specs/PAPER_REVIEW_QUEUE.md`](specs/PAPER_REVIEW_QUEUE.md) | Papers to review/reproduce, prioritized by tier |
| [`specs/SOVEREIGN_VALIDATION_MATRIX.md`](specs/SOVEREIGN_VALIDATION_MATRIX.md) | Sovereign validation ladder / cross-cutting matrix (DRM, drivers, hardware) |
| [`whitePaper/baseCamp/`](whitePaper/baseCamp/) | Per-domain research briefings (19 docs) |
| [`validation/README`](validation/README) | guideStone artifact documentation — quick start, deployment matrix, cross-platform |
| [`validation/GUIDESTONE.md`](validation/GUIDESTONE.md) | guideStone certification spec (deterministic, traceable, self-verifying) |
| [`docs/PRIMAL_GAPS.md`](docs/PRIMAL_GAPS.md) | NUCLEUS composition gaps — handback to primalSpring |
| [`docs/DOWNSTREAM_PATTERNS.md`](docs/DOWNSTREAM_PATTERNS.md) | Downstream repository adoption patterns |
| [`docs/PRIMAL_PROOF_IPC_MAPPING.md`](docs/PRIMAL_PROOF_IPC_MAPPING.md) | Level 6 — CERTIFIED primal proof — domain science → IPC method mapping |
| [`scripts/validate-primal-proof.sh`](scripts/validate-primal-proof.sh) | Primal proof validation — bare + NUCLEUS modes, pre-flight integration |
| [`graphs/hotspring_qcd_deploy.toml`](graphs/hotspring_qcd_deploy.toml) | biomeOS deploy graph — 10 primals, bonding policy, spawn order |
| [`CHANGELOG.md`](CHANGELOG.md) | Root changelog — spring-level changes |
| [`barracuda/ABSORPTION_MANIFEST.md`](barracuda/ABSORPTION_MANIFEST.md) | Write → Absorb → Lean tracking for upstream absorption |
| [`Dockerfile`](Dockerfile) | OCI container image for universal substrate deployment |

---

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0).
See [LICENSE](LICENSE) for the full text.

Sovereign science: all source code, data processing scripts, and validation results are
freely available for inspection, reproduction, and extension. If you use this work in
a network service, you must make your source available under the same terms.

---

*218 experiments, 700 (cylinder) / 596 (barracuda default) / 1,045 (barracuda-local) lib tests, 167 binaries, 128 WGSL shaders, ~$0.30 total science cost.
Consumer GPUs reproduce HPC physics at paper parity. DF64 delivers 3.24 TFLOPS at
14-digit precision. GPU RHMC runs all-flavors dynamical QCD (Nf=2+1). Self-tuning
RHMC eliminates hand-tuned parameters. Chuna 44/44 checks pass. **Fleet: 2× Titan V (GV100) + RTX 5060 (Blackwell)** —
Tier 1 sovereign infrastructure **validated**, **183ms warm pipeline** (falcon
preservation, fd store, 76× faster than cold). RTX 5060 full dispatch LIVE
(QMD v5.0, SM120). Sovereignty tier model codified (Exp 210). TPC wall confirmed
firmware-dependent (Exp 217). nvidia-470 nvsov dual-load injection path **active**
(Exp 218) — module loads alongside host nvidia-580, 21 RPC methods, SymbolResolver
trait, structured twin-card comparison, PC32/PLT32 relocation normalization.
**Vendor-atheistic target**: not just agnostic to vendor, but independent of
vendor entirely — solving sovereign compute across Volta, Blackwell,
and AMD RDNA2. Three-tier validation: Python validates Rust. Rust validates NUCLEUS.
Peer-reviewed science runs on consumer hardware, composed via sovereign primal IPC.
guideStone artifact validated across 5 substrates.
Primal domain split follows Nest atomic pattern: coralReef owns compilation (HOW),
toadStool owns hardware access (WHERE), barraCuda owns physics (WHAT). Composed
via `by_domain()` IPC — no primal links another's crate at compile time.
The full science ladder — quenched through dynamical fermions with gradient flow
scale setting — runs on consumer hardware. The scarcity was artificial.*
