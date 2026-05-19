# hotSpring — Experiment & Validation Index

> **Last audited:** May 19, 2026 · **634 (cylinder) / 596 (default barracuda) / 1,045 (barracuda-local) lib tests** · **167 binaries** · **65 validation suites (smoke/nucleus/silicon tiers)** · **128 WGSL shaders** · **guideStone Level 6 CERTIFIED** (NUCLEUS Deployment Validation) · **7 deploy graphs** · **primalSpring v0.9.25** · **Compile-then-dispatch pipeline wired** · **Tier 4 IPC-first** (`default = []`) · **17/20 validation scenarios** (default / barracuda-local+sovereign-dispatch) · **Sovereignty Tier Model** — Tier 0 (cold), Tier 1 (warm infra — **validated**), Tier 2 (warm compute — blocked), Tier 3 (full sovereign) · **16 RPC methods** (`sovereign.ce_validate` added) · **Fleet: 2× Titan V (GV100) + RTX 5060 (Blackwell) + K80 incoming (GK210)**
>
> Experiments 001–210 validate Python→Rust fidelity, sovereign GPU compute, cross-generation WGSL→native ISA compilation, primal composition proof, vendor-agnostic boot abstraction, diesel engine ember-integrated sovereign boot, generation-aware power safety, Volta cold boot CG sweep (warm/cold convergence), bore-agnostic abstraction surface rewire, live VBIOS interpreter hardware validation, dual Titan V twin study, Falcon ACR DMA boot, sovereign boot state abstraction with twin-card profiling, reboot-efficient sovereign evolution with fd store warm keepalive validation, sovereign VFIO dispatch bridge with anchor-fd adoption, and GPC boundary analysis with sovereignty tier model. **Phase 2** (NUCLEUS composition validation) is tracked via `validate_nucleus_*` binaries and [`docs/PRIMAL_GAPS.md`](docs/PRIMAL_GAPS.md). **Phase 3** (primal composition proof) validates IPC-composed NUCLEUS patterns against direct Rust baselines.

> Updated May 19, 2026. This is the authoritative ledger of all
> experiments, validation suites, and benchmark data. For project overview, see [README.md](README.md).
> Experiments 001-190 archived to `experiments/archive/` — completed physics, benchmark, sovereign GPU, ember hardening, and coral-ember integration work, results absorbed into baseCamp and coralReef code.
> Note: Experiments 096-105 have dual-numbered IDs (physics + sovereign GPU tracks ran in parallel). Filenames are self-descriptive. Exp 136b disambiguated from 136.

**210 experiments** | **500+ quantitative checks** | **~$0.30 total science cost** | **AGPL-3.0-only**

---

## Validation Status Table

| Study | Status | Quantitative Checks |
|-------|--------|-------------------|
| **Sarkas MD** (12 cases) | ✅ Complete | 60/60 pass (DSF, RDF, SSF, VACF, Energy) |
| **TTM Local** (3 species) | ✅ Complete | 3/3 pass (Te-Ti equilibrium) |
| **TTM Hydro** (3 species) | ✅ Complete | 3/3 pass (radial profiles) |
| **Surrogate Learning** (9 functions) | ✅ Complete | 15/15 pass + iterative workflow |
| **Nuclear EOS L1** (Python, SEMF) | ✅ Complete | χ²/datum = 6.62 |
| **Nuclear EOS L2** (Python, HFB hybrid) | ✅ Complete | χ²/datum = 1.93 |
| **BarraCuda L1** (Rust+WGSL, f64) | ✅ Complete | χ²/datum = **2.27** (478× faster) |
| **BarraCuda L2** (Rust+WGSL+nalgebra) | ✅ Complete | χ²/datum = **16.11** best, 19.29 NMP-physical (1.7× faster) |
| **GPU MD PP Yukawa** (9 cases) | ✅ Complete | 45/45 pass (Energy, RDF, VACF, SSF, D*) |
| **N-Scaling + Native f64** (5 N values) | ✅ Complete | 16/16 pass (500→20k, 0.000% drift) |
| **Paper-Parity Long Run** (9 cases, 80k steps) | ✅ Complete | 9/9 pass (N=10k, 0.000-0.002% drift, 3.66 hrs, $0.044) |
| **ToadStool Rewire v1** (3 GPU ops) | ✅ Complete | BatchedEighGpu, SsfGpu, PppmGpu wired |
| **Nuclear EOS Full-Scale** (Phase F, AME2020) | ✅ Complete | 9/9 pass (L1 Pareto, L2 GPU 2042 nuclei, L3 deformed) |
| **BarraCuda MD Pipeline** (6 ops) | ✅ Complete | 12/12 pass (YukawaF64, VV, Berendsen, KE — 0.000% drift) |
| **BarraCuda HFB Pipeline** (3 ops) | ✅ Complete | 16/16 pass (BCS GPU 6.2e-11, Eigh 2.4e-12, single-dispatch) |
| **Stanton-Murillo Transport** (Paper 5) | ✅ Complete | 13/13 pass (D* Sarkas-calibrated, MSD≈VACF, Green-Kubo η*/λ*) |
| **GPU-Only Transport Pipeline** | ✅ Complete | Green-Kubo D*/η*/λ* entirely on GPU, ~493s |
| **HotQCD EOS Tables** (Paper 7) | ✅ Complete | Thermodynamic consistency, asymptotic freedom validated |
| **Pure Gauge SU(3)** (Paper 8) | ✅ Complete | 16/16 pass (HMC, Dirac CG, plaquette physics + sovereign GPU compile validation on SM35/SM70/SM120) |
| **Screened Coulomb** (Paper 6) | ✅ Complete | 23/23 pass (Sturm bisection, Python parity Δ≈10⁻¹², critical screening) |
| **Abelian Higgs** (Paper 13) | ✅ Complete | 17/17 pass (U(1)+Higgs HMC, phase structure, Rust 143× faster than Python) |
| **ToadStool Rewire v2** | ✅ Complete | WgslOptimizer + GpuDriverProfile wired into all shader compilation |
| **ToadStool Rewire v3** | ✅ Complete | CellListGpu fixed, Complex64+SU(3)+plaquette+HMC+Higgs GPU shaders, **FFT f64** — Tier 3 lattice QCD unblocked |
| **Kokkos-CUDA Parity** | ✅ Complete | Verlet neighbor list (992 steps/s peak), 27×→3.7× gap, 9/9 PASS |
| **Verlet Neighbor List** | ✅ Complete | Runtime-adaptive AllPairs/CellList/Verlet selection, DF64 + adaptive rebuild |
| **ToadStool Rewire v4** | ✅ Complete | Spectral module fully leaning on upstream (Sessions 25-31h absorbed). 41 KB local code deleted, `CsrMatrix` alias retained. BatchIprGpu now available |
| **ToadStool Session 42+ Catch-Up** | ✅ Reviewed | S42+: 612 shaders. Dirac+CG GPU absorbed. HFB shaders (10) + ESN weights absorbed. loop_unroller fixed, catch_unwind removed. Remaining: pseudofermion HMC |
| **NPU Quantization** (metalForge) | ✅ Complete | 6/6 pass (f32/int8/int4/act4 parity, sparsity, monotonic) |
| **NPU Beyond-SDK** (metalForge) | ✅ Complete | 29/29 pass (13 HW + 16 Rust math: channels, merge, batch, width, multi-out, mutation, determinism) |
| **NPU Physics Pipeline** (metalForge) | ✅ Complete | 20/20 pass (10 HW pipeline + 10 Rust math: MD→ESN→NPU→D*,η*,λ*) |
| **Lattice NPU Pipeline** (metalForge) | ✅ Complete | 10/10 pass (SU(3) HMC→ESN→NpuSimulator phase classification, β_c=5.715) |
| **Hetero Real-Time Monitor** (metalForge) | ✅ Complete | 9/9 pass (live HMC phase monitor, cross-substrate f64→f32→int4, 0.09% overhead, predictive steering 62% compute saved) |
| **Spectral Theory** (Kachkovskiy) | ✅ Complete | 10/10 pass (Anderson localization, almost-Mathieu, Herman γ=ln\|λ\|, Aubry-André transition, Poisson stats) |
| **Lanczos + 2D Anderson** (Kachkovskiy) | ✅ Complete | 11/11 pass (SpMV parity, Lanczos vs Sturm, full spectrum, GOE→Poisson transition, 2D bandwidth) |
| **3D Anderson** (Kachkovskiy) | ✅ Complete | 10/10 pass (mobility edge, GOE→Poisson transition, dimensional hierarchy 1D<2D<3D, spectrum symmetry) |
| **Hofstadter Butterfly** (Kachkovskiy) | ✅ Complete | 10/10 pass (band counting q=2,3,5, fractal Cantor measure, α↔1-α symmetry, gap opening) |
| **GPU SpMV + Lanczos** (Kachkovskiy GPU) | ✅ Complete | 14/14 pass (CSR SpMV parity 1.78e-15, Lanczos eigenvalues match CPU to 1e-15) |
| **GPU Dirac + CG** (Papers 9-12 GPU) | ✅ Complete | 17/17 pass (SU(3) Dirac 4.44e-16, CG iters match exactly, D†D positivity) |
| **Pure GPU QCD Workload** | ✅ Complete | 3/3 pass (HMC → GPU CG on thermalized configs, solution parity 4.10e-16) |
| **Dynamical Fermion QCD** (Paper 10) | ✅ Complete | 7/7 pass (pseudofermion HMC: ΔH scaling, plaquette, S_F>0, acceptance, mass dep, phase order) |
| **Python vs Rust CG** | ✅ Complete | **200× speedup**: identical iterations (5 cold, 37 hot), Dirac 0.023ms vs 4.59ms |
| **GPU Scaling (4⁴→16⁴)** | ✅ Complete | GPU **22.2× faster** at 16⁴ (24ms vs 533ms), crossover at V~2000, iters identical |
| **NPU HW Pipeline** | ✅ Complete | 10/10 on AKD1000: MD→ESN→NPU→D*,η*,λ*, 2469 inf/s, 8796× less energy |
| **NPU HW Beyond-SDK** | ✅ Complete | 13/13 on AKD1000: 10 SDK assumptions overturned, all validated on hardware |
| **NPU HW Quantization** | ✅ Complete | 4/4 on AKD1000: f32/int8/int4/act4 cascade, 685μs/inference |
| **NPU Lattice Phase** | ✅ 7/8 | β_c=5.715 on AKD1000, ESN 100% CPU, int4 NPU 60% (marginal as expected) |
| **Titan V NVK** | ✅ Complete | NVK built from Mesa 25.1.5. `cpu_gpu_parity` 6/6, `stanton_murillo` 40/40, `bench_gpu_fp64` pass |
| **Ember Architecture** | ✅ Complete | Immortal VFIO fd holder (`toadstool-ember`): `SCM_RIGHTS` fd passing, atomic `swap_device` RPC, DRM isolation preflight, external fd holder detection. Zero-crash driver hot-swap on live system |
| **DRM Isolation** | ✅ Complete | Xorg `AutoAddGPU=false` + udev seat tag removal (61-prefix) prevents compositor crash during driver swaps. Compute GPUs fully invisible to display manager |
| **Dual Titan Backend Matrix** (Exp 070) | ✅ Complete | Both Titans on GlowPlug/Ember. vfio↔nouveau swap validated (oracle). Full backend matrix: vfio, nouveau, nvidia × 2 cards. Register diff infrastructure ready |
| **PFIFO Diagnostic Matrix** (Exp 071) | ✅ Complete | 54-config matrix: 12 winning configs, 0 faults, scheduler-accepted. PFIFO re-init solved (PMC+preempt+clear). Root cause identified (PBDMA 0xbad00200 PBUS timeout) — resolved in Exp 076 (MMU fault buffer) + Exp 077 (init hardening). 6/10 sovereign pipeline layers proven. |
| **MMU Fault Buffer Breakthrough** (Exp 076) | ✅ Complete | **Layer 6 resolved.** Volta FBHUB requires configured non-replayable fault buffers before any MMU page table walk completes. Without them, PBUS returns 0xbad00200 and PBDMA stalls forever. Fix: FAULT_BUF0/1 configured in VfioChannel::create. Channel creation + DMA roundtrip + MMU translation all pass. Shader dispatch blocked at Layer 7 (GR/FECS context). |
| **PFIFO Init Hardening** (Exp 077) | ✅ Complete | Five failure modes documented and fixed. `PfifoInitConfig` unifies init paths, `GpuCapabilities` makes matrix arch-aware, `toadstool device reset` provides PCIe FLR recovery. |
| **Layer 7 Diagnostic Matrix** (Exp 078) | ✅ Complete | FECS/GPCCS confirmed in HRESET — sole Layer 7 blocker. |
| **Warm Handoff via Ember** (Exp 079) | ❌ Failed | nouveau teardown halts falcons before unbind. FECS IMEM does not survive swap. |
| **Sovereign FECS Boot** (Exp 080) | ❌ Blocked | Direct IMEM upload succeeds but falcon remains in HRESET. ACR-managed boot required. |
| **Falcon Boot Solver** (Exp 081) | ✅ Complete | SEC2 base fix (`0x87000`), EMEM PIO verified, CPUCTL v4+ bits corrected, `nvfw_bin_hdr` decoded. ACR boot solver built. |
| **Multi-Backend Oracle Campaign** (Exp 082) | ✅ Complete | Cross-card register profiling infrastructure. Oracle domain diff tooling. |
| **Nouveau Source Analysis** (Exp 083) | ✅ Complete | Root cause analysis of `bind_stat` failure. **4 bugs found (B1-B4).** |
| **B1-B4 Hardware Validation** (Exp 084) | ✅ Complete | All four bugs fixed. bind_inst accepts writes. |
| **B5-B7 Bind Trigger Validation** (Exp 085) | ✅ Complete | **Layer 7 SOLVED.** Three missing trigger writes discovered. **bind_stat reaches 5 on both Titans.** SEC2 DMA active. |
| **Cross-Driver Falcon Profile** (Exp 086) | ✅ Complete | **VERDICT: WPR is an INTERFACE problem, not key+lock.** |
| **WPR Format Analysis** (Exp 087) | ✅ Complete | **Layer 8 SOLVED.** 7 WPR construction bugs (W1-W7). ACR bootstraps FECS/GPCCS to cpuctl=0x12. |
| **Layer 9 Falcon Start** (Exp 088) | ✅ Complete | **Layer 9 SOLVED.** Both falcons transition from 0x12 (HRESET) to 0x00 (RUNNING). **9/10 sovereign layers solved.** |
| **DRM Dispatch Evolution** (Exp 072) | ✅ GCN5 Complete | **AMD GCN5 6/6 PASS.** **RTX 5060 Blackwell DRM cracked.** K80 validated (Exp 190), later retired (Exp 199 fire). |
| **iommufd/cdev VFIO Evolution** (Exp 073) | ✅ Complete | **Kernel-agnostic VFIO** on Linux 6.2+. 607 tests pass. |
| **Ember Swap Pipeline Evolution** (Exp 074) | ✅ Complete | **nouveau ↔ vfio round-trip proven** on Titan V. 86 ember + 178 glowplug tests. |
| **Deep Debt + Cross-Vendor Dispatch** (Exp 075) | ✅ Complete | **13 deep-debt items resolved.** Cross-vendor CUDA dispatch via glowplug daemon RPC. |
| **Vendor-Agnostic GlowPlug** | ✅ Complete | toadstool-ember standalone crate (absorbed into toadStool). RegisterMap trait (GV100 + GFX906/MI50). |
| **Privilege Hardening** | ✅ Complete | Capabilities + seccomp + namespaces. |
| **VendorLifecycle Trait** | ✅ Complete | Vendor-specific swap hooks. 157 tests pass. |
| **AMD D3cold Resolution** | ✅ Characterized | Vega 20 SMU firmware limitation: one vfio→amdgpu cycle per boot. |
| **BrainChip Akida NPU** | ✅ Complete | AKD1000 fully integrated. Unlimited round-trips, SimpleBind, no DRM. |
| **Zero-Sudo toadstool device** | ✅ Complete | `toadstool` unix group (legacy `coralreef` group deprecated), socket permissions. |
| **Experiment Loop Infrastructure** (Exp 092) | ✅ Complete | Adaptive experiment loop. **4,065 tests pass.** |
| **First Personality Sweep** (Exp 092) | ✅ Complete | Both Titan Vs swept. **Sub-1% cross-card variance.** |
| **GPU Streaming HMC** | ✅ Complete | 9/9 pass (4⁴→16⁴, streaming 67× CPU, dispatch parity, GPU PRNG) |
| **GPU Streaming Dynamical** | ✅ Complete | 13/13 pass (dynamical fermion streaming, GPU-resident CG, bidirectional stream) |
| **GPU-Resident CG** | ✅ Complete | 15,360× readback reduction, 30.7× speedup, α/β/rz GPU-resident |
| **biomeGate Prep** | ✅ Complete | Node profiles, env-var GPU selection, NVK setup guide, RTX 3090 characterization |
| **API Debt Fix** | ✅ Complete | solve_f64→CPU Gauss-Jordan, sampler/surrogate device args, 4 binaries fixed |
| **Production β-Scan (biomeGate)** | ✅ Complete | Titan V 16⁴ (9/9, 47 min, first NVK QCD). RTX 3090 32⁴ (12/12, 13.6h, $0.58). **Deconfinement transition: χ=40.1 at β=5.69** |
| **DF64 Core Streaming** | ✅ Complete | DF64 gauge force live on RTX 3090. 9.9× FP32 core throughput. |
| **Site-Indexing Standardization** | ✅ Complete | adopted toadStool t-major convention. |
| **DF64 Unleashed Benchmark** | ✅ Complete | 32⁴ at 7.7s/traj (2× faster). |
| **toadStool S60 DF64 Expansion** | ✅ Complete | FMA-optimized df64_core. 60% of HMC in DF64. |
| **Mixed Pipeline β-Scan** | ⏸️ Partial | 3-substrate (3090+NPU+Titan V). DF64 2× confirmed at 32⁴. |
| **Cross-Spring Rewiring** | ✅ Complete | GPU Polyakov loop (72× less transfer), NVK alloc guard, PRNG fix. |
| **Debt Reduction Audit** | ✅ Complete | 985 tests (lib), 82 validation binaries. |
| **DF64 Production Benchmark** (Exp 018) | ✅ Complete | 32⁴ at 7.1h mixed (vs 13.6h FP64-only). |
| **Forge Evolution Validation** (Exp 019) | ✅ Complete | metalForge streaming pipeline: 9/9 domains. |
| **NPU Characterization Campaign** (Exp 020) | ✅ Complete | 13/13: thermalization detector 87.5%, rejection predictor 96.2%. |
| **Cross-Substrate ESN Comparison** (Exp 021) | ✅ Complete | 35/35: First GPU ESN dispatch via WGSL. NPU 1000× faster streaming. |
| **NPU Offload Mixed Pipeline** (Exp 022) | ✅ Complete | 8⁴ validated. 32⁴ production on **live AKD1000 hardware NPU** via PCIe. |
| **NPU GPU-Prep + 11-Head** (Exp 023) | ✅ Complete | 11-head ESN, pipelined predictions, adaptive CG, intra-scan steering. |
| **HMC Parameter Sweep** (Exp 024) | ✅ Complete | Fermion force sign/factor fix (-2x). 160 configs, 2,400 trajectories. |
| **GPU Saturation Multi-Physics** (Exp 025) | ✅ Complete | 16⁴ validation, Titan V chains, Anderson 3D proxy. |
| **4D Anderson-Wegner Proxy** (Exp 026) | 📋 Planned | 4D Anderson + Wegner block proxy. |
| **Energy Thermal Tracking** (Exp 027) | 📋 Planned | RAPL + k10temp + nvidia-smi energy sidecar monitor. |
| **Brain Concurrent Pipeline** (Exp 028) | ✅ Complete | 4-layer brain: RTX 3090 + Titan V + CPU + NPU. |
| **NPU Steering Production** (Exp 029) | ✅ Complete | 4-seed baseline. Adaptive steering bug found and fixed. |
| **Adaptive Steering** (Exp 030) | ⏹ Superseded | Fixed adaptive steering, but auto_dt over-penalized mass. |
| **NPU-Controlled Parameters** (Exp 031) | ✅ Complete | NPU controls dt/n_md. |
| **Finite-Temp Deconfinement** (Exp 032) | ✅ 32³×8 Complete | 32³×8: 1,800 traj, crossover at β≈5.9. 64³×8: MILC-comparable. |
| **Wilson Gradient Flow** (Chuna) | ✅ Complete | t₀ + w₀ scale setting. LSCFRK3W6/W7/CK4 — 3rd-order coefficients **derived from first principles**. |
| **Flow Integrator Comparison** (Chuna) | ✅ Complete | 5 integrators validated. W7 ~2× more efficient for w₀. |
| **N_f=4 Staggered Dynamical GPU** | ✅ Infra Complete | GPU staggered Dirac + CG + pseudofermion + dynamical HMC trajectory. |
| **RHMC Infrastructure** | ✅ Complete | `RationalApproximation` + `multi_shift_cg_solve` for fractional flavors. |
| **GPU RHMC Production** (Exp 101) | ✅ Complete | Nf=2 at 4⁴/8⁴ + Nf=2+1 at 4⁴. First all-flavors dynamical QCD on consumer GPU. |
| **Gradient Flow at Volume** (Exp 102) | ✅ Complete | 16⁴ CK4 convergence orders verified. t₀/w₀ scale setting. |
| **Self-Tuning RHMC** (Exp 103) | ✅ Complete | `RhmcCalibrator`: zero hand-tuned magic numbers. |
| **Precision Stability** (Exp 046) | ✅ Complete | 9/9 cancellation families audited. |
| **Chuna Overnight** (Papers 43-45) | ✅ **44/44** | Core paper reproduction 41/41. Dynamical N_f=4 extension: 3/3 pass. |
| **coralReef Integration** | ✅ Complete | **45/46** shaders compile to SM70/SM86 SASS. Full `GpuBackend` impl. |
| **Precision Brain** (Exp 049) | ✅ Complete | Self-routing brain: safe hardware calibration. |
| **coralReef Hardware Data** (Exp 051) | ✅ Complete | NVK/Mesa 25.1.5 **unlocks Titan V**. |
| **NVK/Kokkos Parity** (Exp 052) | 🔄 Active | Multi-backend dispatch strategy. |
| **Live Kokkos Benchmark** (Exp 053) | ✅ Complete | **12.4× gap** measured. |
| **Kokkos N-Scaling** (Exp 054) | ✅ Complete | N=500→50k complexity benchmark. |
| **DF64 Naga Poisoning** (Exp 055) | ✅ Complete | Root cause: naga WGSL→SPIR-V codegen bug. coralReef sovereign bypass validated. |
| **Sovereign Dispatch** (Exp 056) | ✅ Complete | Backend-agnostic `MdEngine<B: GpuBackend>`. |
| **coralReef Ioctl Fix** (Exp 057) | ✅ Complete | 4 DRM ioctl struct ABI mismatches fixed. |
| **hwLearn Integration** | ✅ Complete | toadStool `hw-learn` crate: vendor-neutral GPU learning (46 tests). |
| **W1 Header + BOOTVEC Metadata** (Exp 093) | ✅ Complete | BL files parsed. IMEM layout fixed. |
| **Path B LS Mode Blocked** (Exp 094) | ❌ Dead | GV100 fuse-enforced LS mode authentication. Path B dead for Volta. |
| **Sysmem HS Mode Breakthrough** (Exp 095) | ✅ **BREAKTHROUGH** | **SEC2 enters Heavy Secure mode via system memory DMA.** |
| **Silicon Characterization** (Exp 096-100) | ✅ Complete | TMU 1.89x RTX 3090, AMD DF64 38% advantage, 4-phase pipeline, hardware personalities |
| **Silicon Routed QCD Revalidation** (Exp 105) | ✅ Complete | Revalidated quenched + Nf=2 + Nf=2+1 QCD with silicon routing. Unidirectional RHMC: 3.79x speedup (3090), 2.06x (6950 XT). **True multi-shift CG** (shared Krylov, 37% speedup). **Fermion force sign fix** (−η convention, ΔH: 1500→O(1)). `std::hint::black_box` for release-mode convergence |
| **Silicon Tier Routing + Legacy Cleanup** (Exp 106) | ✅ Complete | 7-tier routing spec, SiliconProfile system, GpuTelemetry, deprecated 6 sync-heavy functions, production binary migration |
| **Silicon Saturation Profiling** (Exp 107) | ✅ Complete | 7-phase full-card profiling on strandgate (RTX 3090 + RX 6950 XT). TMU PRNG (Box-Muller via textureLoad, Tier 0). Subgroup reduce (`subgroupAdd()` for CG dot products, Tier 4). ROP atomic scatter-add (fixed-point i32 `atomicAdd` for fermion force, Tier 3). NPU observation 11D (6D physics + 5D silicon tags). Capacity: RTX 3090 L=46⁴ dynamical (23.6 GB), RX 6950 XT L=40⁴ (13.5 GB). 6 new WGSL shaders, 10 new binaries |
| **Consolidation Matrix** (Exp 110) | ✅ Complete | biomeGate: sovereign pipeline consolidation and gap analysis |
| **VRAM Native Page Tables** (Exp 111) | ✅ Complete | biomeGate: native VRAM page table construction |
| **Dual Phase Boot** (Exp 112) | ✅ Complete | biomeGate: HS mode via dual-phase boot (SCTL=0x3002) |
| **Trap Analysis** (Exp 113) | ✅ Complete | biomeGate: PMU dependency confirmed |
| **LS Mailbox Pipeline** (Exp 114) | ✅ Complete | biomeGate: LS-mode WPR copy stall analysis |
| **Direct Boot + WPR Analysis** (Exp 115) | ✅ Complete | biomeGate: direct boot investigation |
| **WPR Reuse + Firmware Analysis** (Exp 116) | ✅ Complete | biomeGate: WPR reuse strategy |
| **WPR2 State Tracking** (Exp 117) | ✅ Complete | biomeGate: WPR2 valid at 12GB during nouveau, destroyed on swap |
| **WPR2 Preservation** (Exp 118) | ✅ Complete | biomeGate: WPR2 preservation attempts |
| **Cold Boot WPR2** (Exp 119) | ✅ Complete | biomeGate: cold boot WPR2 invalid |
| **Sovereign DEVINIT** (Exp 120) | ⚠️ Corrected | biomeGate: DEVINIT not needed on warm GPU (correct), BUT IS needed after SBR (see Exp 141 correction) |
| **WPR2 Resolution** (Exp 122) | ✅ Complete | biomeGate: **definitive root cause** — WPR2 registers hardware-locked by FWSEC |
| **Parasitic Compute** (Exp 123T) | ✅ Complete | biomeGate: parasitic compute probe |
| **K80 Sovereign Compute** (Exp 123) | 🔄 Active | biomeGate: Tesla K80 (GK210) — zero firmware security, direct PIO boot |
| **AMD Scratch/Local Memory Breakthrough** (Exp 124) | ✅ Complete | strandgate: AMD RX 6950 XT (RDNA2) scratch/local memory dispatch via coralReef DRM path |
| **VM Capture Cross-Analysis** (Exp 124b) | ✅ Complete | biomeGate: nvidia-470/535 VM captures for K80+Titan V, cross-driver register tracing |
| **Warm Handoff Livepatch** (Exp 125) | ✅ Complete | biomeGate: kernel livepatch NOP (mc_reset+gr_fini+falcon_fini+runl_commit), wired into ember/glowplug |
| **DRM Proprietary Tracing Matrix** (Exp 126) | ⏸ Paused | biomeGate: deprioritized vs VBIOS DEVINIT track (Exp 141-142) |
| **Warm FECS Dispatch Attack** (Exp 127) | ✅ Complete | biomeGate: FECS firmware preserved in IMEM but cannot be woken from HS+ halt state |
| **GPU Puzzle Box Matrix** (Exp 128) | ⏹ Superseded | biomeGate: converged to VBIOS DEVINIT as single remaining blocker (see Exp 141) |
| **No-FLR Recovery & PRI Ring Lessons** (Exp 130) | ✅ Complete | biomeGate: K80 GK210 PRI ring diagnostics, cold GPU detection, PMU/FECS falcon state analysis |
| **Reset Architecture Evolution** (Exp 131) | ✅ Complete | biomeGate: warm_fecs.rs → device.warm_handoff RPC, livepatch into ember, orphan cleanup |
| **Ember Frozen Warm Dispatch** (Exp 132) | ✅ Implemented | biomeGate: diesel engine pattern — glowplug orchestrates swap, ember keeps VFIO fds alive, `mmio.write` for active intervention, STOP_CTXSW freezes FECS scheduling |
| **Kepler Sovereign Compute** (Exp 133) | ✅ Implemented | biomeGate: K80 (GK210) Kepler-specific QMD v1.7, push buffer methods from `cla1c0.h`, architecture-aware dispatch branching |
| **K80 Sovereign Cold Boot Pipeline** (Exp 134) | ✅ Implemented | biomeGate: single-command cold boot (`toadstool device cold-boot <BDF> --recipe <path>`) — D3cold→FECS-running without any vendor driver |
| **Dual GPU Sovereign Boot Attempt** (Exp 135) | ✅ Complete | biomeGate: K80 needs VBIOS POST (memory training), Titan V SEC2 ROM rejects ACR — PMU/WPR chain required. FECS PIO upload works on K80 but PGRAPH CTXSW domain PRI-faults above 0x409504 |
| **Dual GPU Sovereign Boot Iteration** (Exp 136) | ✅ Complete | biomeGate: both GPUs hit known barriers. SEC2 DMA path analysis + FBHUB/FBPA discovery. FBIF locked in VIRT mode by HS+ |
| **SEC2 DMA Reconstruction** (Exp 137) | ✅ Complete | biomeGate: BOOTSTRAP_FALCON failure root cause confirmed, SEC2 communication protocol identified |
| **D-State Root Cause & Rewire Plan** (Exp 138) | ✅ Complete | biomeGate: D-state root cause traced, ember/glowplug rewire for resilient VFIO control |
| **Sovereign Dispatch ACR Lockdown** (Exp 139) | 🔴 Blocked | biomeGate: Titan V ACR lockdown confirmed, K80 cold/needs POST. FBIF locked in VIRT mode by HS+ |
| **Uncrashable GPU Safety Architecture** (Exp 140) | ✅ Validated | biomeGate: D-state resilience, timeout-guarded sysfs writes, ember process isolation |
| **ACR HS Auth Investigation** (Exp 141) | ⚠️ Refined | biomeGate: initially identified VBIOS DEVINIT as root cause (SEC2 crypto uninitialized after SBR). DMA path fully fixed (sysmem PTEs, FBIF VIRT, DMEM repair). **Exp 142-143 contradicted** — ACR fails even on BIOS-POSTed GPU. Actual root cause: SEC2 PTOP/PMC bit missing, falcon cannot start |
| **Sovereign Boot VBIOS DEVINIT** (Exp 142) | ⚠️ Ran | biomeGate: PM bridge reset did not cold-reset GPU. DEVINIT correctly skipped (GPU still POSTed). ACR fails — SEC2 POST-START FAULT. Root cause is SEC2 HAL, not DEVINIT. |
| **No-SBR Confirmation Test** (Exp 143) | ❌ Contradicted | biomeGate: ACR fails even on BIOS-POSTed GPU (no SBR, fresh cold boot). VBIOS DEVINIT is NOT the sole root cause. SEC2 falcon cannot start — PTOP missing SEC2 bit, PMC fallback may be wrong. |
| **PMC Bit5 ACR Progress** (Exp 144) | ✅ Complete | biomeGate: PMC bit 5 SEC2 enable/discovery, ACR pipeline progression |
| **Crash Vector Hunt** (Exp 150) | ✅ Complete | biomeGate: PRAMIN isolated as lockup trigger on cold VRAM. Graceful cold-VRAM detection. |
| **Revalidation & Next Stages** (Exp 151) | ✅ Complete | biomeGate: full revalidation pass + next-stage planning. Ember survivability hardening plan. |
| **Compute Dispatch Provenance** (Exp 152) | ✅ Complete | biomeGate: dispatch provenance validation, multi-backend parity confirmation |
| **Ember Flood Resurrection Proof** (Exp 153) | ✅ Complete | biomeGate: ember flood/resurrection under continuous fault injection |
| **SEC2 ACR PMU First Pipeline** (Exp 154) | ✅ Complete | biomeGate: SEC2→PMU first-boot pipeline, ACR chain ordering investigation |
| **K80 Warm FECS Dispatch** (Exp 155) | ✅ Complete | biomeGate: K80 warm-state FECS dispatch (Kepler PIO path) |
| **Reagent Trace Comparison** (Exp 156) | ✅ Complete | biomeGate: cross-reagent register trace comparison for DEVINIT analysis |
| **K80 DEVINIT Replay** (Exp 157) | ⚠️ Ran | biomeGate: K80 direct DEVINIT replay — PLL reprogramming risk identified |
| **SEC2 Real Firmware** (Exp 158) | ✅ Complete | biomeGate: SEC2 ACR bootloader executes but stalls on DMA (HBM2 not trained) |
| **Titan V VM-POST HBM2** (Exp 159) | ✅ Complete | biomeGate: HBM2 trained via nvidia-535 VM passthrough. **FLR kills training.** nouveau warm-cycle + `reset_method` clear preserves HBM2 through vfio-pci bind. |
| **Titan V MMIOTRACE Capture** (Exp 160) | ✅ Complete | biomeGate: mmiotrace register capture for GV100 nouveau init sequence |
| **Titan V NVDEC Sovereign Attempt** (Exp 161) | ✅ Complete | biomeGate: NVDEC engine sovereign dispatch attempt on GV100 |
| **Titan V Sovereign Compute Pipeline** (Exp 162) | ✅ Complete | biomeGate: full sovereign compute pipeline design for GV100 with firmware coexistence |
| **Firmware Boundary** (Exp 163) | ✅ Complete | biomeGate: **Architectural pivot.** Driver/firmware/hardware delineation. Falcon firmware = GPU's BIOS. PMU mailbox protocol mapped (register-based on GV100). Hot-handoff channel injection proven (CH 500 accepted by scheduler). **NOP dispatch via nouveau DRM: SUCCEEDED** (C + pure Rust). `PmuInterface` struct created. End-to-end: `VM_INIT → CHANNEL_ALLOC(VOLTA_COMPUTE_A) → GEM → VM_BIND → EXEC → SYNCOBJ`. |
| **Sovereign Compute Dispatch Proven** (Exp 164) | ✅ Complete | biomeGate: NOP dispatch proven via DRM + pure Rust ioctls. nouveau warm-cycle preserves HBM2 training. `reset_method` clear prevents FLR from destroying trained memory. Channel injection alongside nouveau scheduler validated. |
| **SovereignInit Full Pipeline** (Exp 165) | ✅ Complete | biomeGate: 8-stage `SovereignInit` pipeline replaces nouveau subsystem by subsystem. Stages: HBM2 Training → PMC Gating → Topology → PFB → Falcon Boot (15 strategies) → GR Init → PFIFO → GR Context. `open_sovereign()` entry point. GR init extracted to standalone fns. `SovereignInitResult` with `compute_ready()` + `diagnostic_summary()`. FECS method probe validates responsiveness. Optional Stage 7 GR context allocation + golden save. 429 coral-driver tests pass. |
| **Sovereign Boot Wiring** (Exp 166) | ✅ Complete | biomeGate: end-to-end sovereign boot wiring from SovereignInit through channel creation |
| **Warm Handoff** (Exp 167) | ✅ Complete | biomeGate: vfio → nouveau (HBM2 training) → vfio warm handoff. `skip_sysfs_unbind` PCI rescan path |
| **Sovereign Pipeline Complete** (Exp 168) | ✅ Complete | biomeGate: milestone — full sovereign compute pipeline operational on Volta |
| **Warm Handoff Validated** (Exp 169) | ✅ Complete | biomeGate: round-trip warm handoff validation — HBM2 state preserved through nouveau→vfio swap |
| **Sovereign Boot E2E** (Exp 170) | ✅ Complete | biomeGate: milestone — end-to-end sovereign boot without vendor driver |
| **K80 Sovereign Init** (Exp 171) | ⚠️ Blocked | biomeGate: K80 cold GDDR5 training blocked — needs nouveau warm cycle for VRAM |
| **No-ACR Warm Handoff** (Exp 172) | ✅ Complete | biomeGate: K80 pre-Kepler has no ACR — FECS PIO upload directly after warm cycle |
| **VM Reagent WPR Capture** (Exp 173) | ✅ Complete | biomeGate: nvidia-470 VM capture proves WPR NOT configured on GV100 (pre-GSP driver) |
| **K80 Sovereign Boot** (Exp 174) | ⚠️ In Progress | biomeGate: K80 GK210B sovereign boot — FECS boots, GPC topology discovered |
| **RTX 5060 Shared Compute** (Exp 175) | ✅ Complete | biomeGate: milestone — Blackwell (SM120) shared display+compute, first desktop GPU sovereign |
| **QCD Parity Benchmark** (Exp 176) | ✅ Complete | biomeGate: cross-generation QCD parity benchmark (K80/Titan V/RTX 5060) |
| **Blackwell Dispatch ABI Fixes** (Exp 177) | ✅ Complete | biomeGate: SM120 QMD ABI alignment fixes for Blackwell compute dispatch |
| **K80 PGOB Nvidia470 Analysis** (Exp 178) | ⚠️ Pivoted | biomeGate: GK210B PGOB analysis. Cold sovereign blocked → pivoted to nouveau warm-catch |
| **K80 Warm FECS Dispatch Pipeline** (Exp 179) | ✅ Complete | biomeGate: nouveau warm-catch → VFIO → FECS boot → PFIFO channel. SCHED_ERROR code=32 root-caused (missing RAMFC 0x3C/0x44) and fixed. HW validated: runlist works, SCHED_ERROR=0. Cold-boot sovereign achieved (udev PLX fix). GPC PGOB remains dispatch blocker |
| **Three-GPU Hardware Validation** (Exp 180) | ✅ Complete | biomeGate: RTX 5060 19/19 pass (CUDA+DRM+discovery), Titan V 20/20 standalone VFIO pass, K80 device open + runlist pass. PGOB GPC gating confirmed as K80 root blocker |
| **Sovereign Dispatch Pipeline Sweep** (Exp 181) | 🔧 In Progress | biomeGate: RTX 5060 8/8 PROVEN (WGSL→SM120→dispatch→readback). Titan V blocked: nouveau DRM (no PMU fw), VFIO warm handoff (FECS HRESET, HS-mode requires SEC2/ACR boot chain). K80 cold-boot sovereign (udev PLX fix), VFIO PGOB dispatch blocker remains. Ember Exclusive Device Gate live (all direct HW access routes through ember when active). nouveau+nvidia coexistence confirmed on kernel 6.17 |
| **K80 FECS PIO Boot** (Exp 182) | 🔧 Diagnostic | biomeGate: K80 GK210 FECS programmed I/O boot diagnostic. Direct BAR0 mmap (`low-level` feature). Falcon IMEM/DMEM load via PIO path. Validates FECS register state machine independently of interrupt-driven path |
| **K80 FECS Interrupt Boot** (Exp 183) | 🔧 Diagnostic | biomeGate: K80 GK210 FECS interrupt-driven boot. Direct BAR0 mmap (`low-level` feature). Falcon boot with interrupt signaling path. Counterpart to exp182 PIO path |
| **K80 GR Sovereign** (Exp 184) | ✅ Active | biomeGate: K80 GK210 sovereign GR initialization via ember RPC. Uses `sovereign_stages.rs` pipeline — modern ember-wired path (no direct BAR0 mmap). Kepler falcon boot with firmware from `/var/lib/coralreef/firmware/gk110`. Ember keepalive + switch preflight integrated |
| **K80 Nouveau GK210 Chipset Analysis** (Exp 185) | ✅ Complete | biomeGate: Root cause analysis of K80 nouveau failure. Upstream nouveau has NO `case 0x0f2:` — GK210 chip ID falls through to "unknown chipset" → `-ENODEV`. No subdevs init. GK210 is arch-identical to GK110B (`nvf1_chipset`). One-line kernel patch identified: `case 0x0f2: device->chip = &nvf1_chipset;` |
| **PMU Firmware Extraction Analysis** (Exp 186) | ✅ Complete | biomeGate: PMU firmware source analysis for K80 + Titan V. Kepler PMU from VBIOS (BIT tables, 62 KB ROM). Volta PMU NOT in linux-firmware/nvidia-580 — extraction target is nvidia-470 proprietary kernel module. Enhanced `exp168_pmu_firmware_probe` with Falcon v3, `--mode nv-ko`, `--mode vbios` |
| **Titan V nvidia-580 mmiotrace Prep** (Exp 187) | 🔧 Prepared | biomeGate: Capture script and analysis plan for nvidia-580 mmiotrace of Titan V GV100 FECS/SEC2/ACR boot. Determines if WPR is used, informs FalconBootSolver Volta branch. Existing nouveau trace analyzed (only 2 FECS reads — GR never started). Awaiting execution window |
| **K80 Warm-Catch Breakthrough** (Exp 188) | ✅ Breakthrough | biomeGate: Patched nouveau RECOGNIZED GK210 as GK110B — first-ever GR init on K80! `fb: 12288 MiB GDDR5`, 5 GPCs enrolled, 6 TPCs/GPC, DRM initialized. Post-VFIO rebind GPCs power-gated (livepatch incompatible w/ 6.17). PLX D3cold on ember stop — script updated to keep ember alive. nvidia-470 PMU scan: no Falcon UC headers — firmware embedded in RM data structs |
| **LTEE B2: Anderson Fitness Landscape** (Exp 189) | ✅ Complete | LTEE GuideStone: Wiser et al. 2013 Anderson disorder analogy — fitness trajectory as disordered potential, level spacing ratio diagnostics, sliding-window localization analysis, 12-population variance. Tier 1 Python baseline in `notebooks/papers/13-ltee-anderson-fitness.ipynb`. Tier 2 Rust validation: `s_ltee_anderson` scenario (self-contained tridiagonal eigensolver, 18 validation checks). Expected values JSON in `experiments/results/ltee/`. Feeds lithoSpore module 7 (anderson) + foundation Thread 7 |
| **Three-GPU Sovereign Validation** (Exp 190) | ✅ Complete | Post-power-cycle sovereign validation across RTX 5060 (SM120), Titan V (GV100), K80 (GK210). RTX 5060: **12/12 sovereign roundtrip PASS**, 154.2 steps/s Yukawa OCP MD. Titan V: warm-catch via binary-patched nouveau (nvidia-470 VM path, GAP-HS-073 resolved). K80: nouveau GK210→GK110B patch, 12 GiB GDDR5 trained, 5 GPCs active (GAP-HS-076 proven). benchScale VM isolation for multi-driver coexistence. See `experiments/190_THREE_GPU_SOVEREIGN_VALIDATION.md` |
| **toadStool S258+ PBDMA Dispatch Validation** (Exp 191) | ✅ Complete | Compute trio pipeline validation: toadStool PBDMA dispatch (S258-S261), VFIO IPC surface, QMD dispatch. barraCuda `compile_and_submit()` wiring. coralReef HMMA GEMM codegen. Circuit-breaker discovery, dispatch unification, TOML aliases. Titan V FECS warm-catch, K80 D3cold PLX recovery. plasmidBin ecoBin deployment for all 3 primals. See `experiments/191_TOADSTOOL_S258_PBDMA_VALIDATION.md` |
| **Hardware Validation Sprint — Compute Trio** (Exp 192) | ✅ Complete | RTX 5060 SM120 DRM **PASS** (architecture detection fixed `0x2900..=0x2FFF => "sm120"`, SemaphoreFence wiring confirmed, 24/24 generation profile tests). Titan V **PARTIAL** (FECS protocol corrected — PGRAPH-wrapped MAILBOX0 0x840 + MTHD_CMD 0x504 — 3 bugs fixed, but PGRAPH clock-gated after driver teardown, requires SovereignInit GR pipeline). K80 **BLOCKED** (PLX D3cold, PCI config all 0xFF, needs reboot). See `experiments/192_HARDWARE_VALIDATION_SPRINT_COMPUTE_TRIO.md` |
| **PLX D3cold Keepalive — K80 Warm Swap Survival** (Exp 193) | ✅ Proven | PLX D3cold root-cause diagnosed + **prevention validated on hardware**. `pin_bridge_hierarchy()` full ancestry walk prevents PLX PEX 8747 D3cold during unbind. Post-POR test: vfio-pci unbind → PLX stayed Gen3 8GT/s DLActive+ → vfio-pci rebind → both K80 dies healthy. `SwapGuard` burst keepalive (10ms CfgRd) wired. `NvidiaKeplerLifecycle` + `SysfsSwapExecutor` hierarchy pinning. 65/65 tests pass. See `experiments/193_PLX_D3COLD_KEEPALIVE_K80.md` |
| **Cold/Warm Boot Architecture** (Exp 194) | ✅ Complete | **No-FLR warm swap validated on Titan V** — 27/27 registers alive through nouveau→vfio-pci swap. PRI Ring + PGRAPH + GPC state fully preserved. Cold boot barrier: FLR kills PRI Ring (DEVINIT lost), Volta HS security (SCTL=0x3000) blocks falcon execution even with PIO. PMC_ENABLE writable from VFIO on warm GPU. K80 PLX D3cold during manual swap confirms SwapGuard required. GV100 topology: 6 GPCs, 40 TPCs, 6 ROPs. See `experiments/194_COLD_WARM_BOOT_ARCHITECTURE.md` |
| **Driver Lab — Mesa vs Vendor** (Exp 195) | 🔬 In Progress | **Driver comparison laboratory** using glowplug containment architecture. Trial 1→2 (cold→nouveau): 0 registers woke up, 92 PGRAPH_GPC changed, SEC2 untouched (all zeros). FECS HS-locked (SCTL=0x20204080: fuse-blown, debug disabled). Mapped security boundary: nouveau reaches PMC/PGRAPH hub/GPC/PBDMA/BAR access; FECS/GPCCS/PMU/SEC2 gated behind HS firmware chain. Trial 3 (nvidia-470 VM) pending. `DriverLabPlan` + `NV_BAR0_DOMAINS` in glowplug. See `experiments/195_DRIVER_LAB_MESA_VS_VENDOR.md` |
| **Warm Swap Validation + PLX Keepalive** (Exp 196) | ✅ Validated | **Root cause fix for PLX D3cold** — inactivity, not swaps. `PlxKeepalive` (ember): continuous CfgRd every 5s on device + bridge chain. `PlxGuardian` (glowplug): fleet-level auto-detect. Post-power-cycle warm swap validated: **Titan V 23 engines** (PMC_ENABLE=0x5fecdff1, SEC2 partially init), **K80 22 engines** (PMC_ENABLE=0xfc37b1ef, PGRAPH clock-gated). PLX survived full swap cycle (rev ca). K80 path to FECS: PGRAPH ungating → PIO boot (no HS security). 98 ember + 95 glowplug tests. See `experiments/196_WARM_SWAP_VALIDATION_PLX_KEEPALIVE.md` |
| **Sovereign Init RPC — Warm/Cold Cross-Hardware** (Exp 197) | ✅ Validated | **`sovereign.init` wired as JSON-RPC method** — first direct diesel engine invocation over IPC. `MappedBar::from_sysfs_rw()` constructor enables BAR0 access without full VFIO device open. **Titan V (warm)**: BAR0 probe OK (12ms), PMC enable OK (75ms), memory training **skipped** (warm detected via PRAMIN sentinel), falcon boot halted at StubGspBridge (expected — real bridge needed for FECS firmware). **K80 GPU0+GPU1 (cold VFIO)**: BAR0 probe OK, PMC enable OK, memory training **failed** ("PRAMIN dead" — GDDR5 DEVINIT replay needed, no driver has initialized since boot). Pipeline total: Titan V 88ms, K80 206-208ms. Stages 1-3 proven sovereign across GV100+GK210. K80 next step: VBIOS ROM extraction + DEVINIT replay for PRAMIN. Titan V next step: real GspBridge (coralReef IPC or warm-handoff FECS state). |
| **Vendor-Agnostic BootPipeline + VBIOS Fixes** (Exp 198) | ✅ Validated | **`BootPipeline` trait** — vendor-agnostic boot abstraction via `&dyn RegisterAccess`. Universal sequence: `probe → is_warm → devinit → engine_init → verify`. `DeviceTopology`/`DeviceFunction` replace NVIDIA-specific multi-die models. **KeplerInit + VoltaInit** implement both `InitPipeline` (NV-specific) and `BootPipeline` (generic). **VegaInit** AMD Vega 20 stub proves cross-vendor compatibility (GRBM_STATUS/SRBM_STATUS warm detect, 8 tests with FakeBar mock). **VBIOS interpreter fixes**: opcode 0x50 stride corrected (11+count*4), opcode 0x88 RAM-restrict added, `ram_restrict_group_count()` fixed to parse M table header, opcode 0x70 EON added. K80 Script 1 now parses correctly. Titan V warm re-validated (101ms, compute_ready=true). **591→606 tests** (15 new). |
| **Diesel Engine Sovereign Boot** (Exp 199) | ⚠️ K80 Fire | **`bar0_source=ember` pipeline** — sovereign.init routed through diesel engine's cached VFIO devices. `ComputeDevice::bar0()` + `dma_backend()` trait extensions. `DispatchHandler::sovereign_init_ember()`. Real `NvGspBridge::acr_boot` using `boot_falcon_hs` DMA for GPCCS+FECS. **Titan V**: bar0_probe OK, pmc_enable OK, memory_training FAILED (PGRAPH CG gated). **K80 x2**: pmc_enable OK (0xc0002020→0xfc37b1ef), memory_training FAILED → **K80 caught fire on reboot** (bulk PMC_ENABLE + uninitialised GDDR5 + aged VRM). See `experiments/199_DIESEL_ENGINE_SOVEREIGN_BOOT.md` |
| **Diesel Engine Power Safety** (Exp 200) | ✅ Validated | **`PowerSafetyProfile`** — generation-aware PMC_ENABLE sequencing derived from K80 fire post-mortem. `PRE_FIRMWARE` (Kepler/Maxwell): conservative mask 0xC000_2030, rollback on devinit failure. `FIRMWARE_MANAGED` (Pascal+): full 0xFFFF_FFFF. Staged pipeline: initial mask in stage 2, full ungating in new stage 3b only after devinit succeeds. `pmc_enable_rollback()` restores pre-pipeline value on failure. All 10 generation profiles annotated. **Builds clean**, validated with Titan V. See `experiments/200_DIESEL_ENGINE_POWER_SAFETY.md` |
| **Volta Cold Boot CG Sweep** (Exp 201) | ✅ Validated | **Warm/cold convergence** — extracted CG sweep from glowplug warm path into `MappedBar`-only functions in `sovereign_stages.rs`. New pipeline stages 2b/2c/2d: `cg_sweep` (PTHERM + PMC CG + PRIV_RING + PFB + PCLOCK + 4 FBPAs + 6 LTCs → `CG_DISABLE`), `pri_bus_recover` (PriBusMonitor fault ack), `pgob_ungating` (PGRAPH GPC broadcast). Runs for all non-NoAcr gens (Volta+) before memory_training. Unblocks cold HBM2 training and falcon DMA boot by clearing `0xBADF` PRI faults. `registers.rs::cg` module activated (removed `dead_code` expect). **Builds clean**. See `experiments/201_VOLTA_COLD_BOOT_CG_SWEEP.md` |
| **Experiment Surface Rewire** (Exp 202) | ✅ Implemented | **Bore-agnostic abstraction rewire** — 6 gaps closed between `SovereignStrategy` trait and pipeline internals. `falcon_boot()` dispatches on `FalconBootStyle` enum (not internal `is_kepler()`). `probe_identity()` + `verify_device()` trait methods replace hardcoded NVIDIA logic. `SovereignInitResult` fields neutralized (`identity_chip`, `identity_raw`, `training_writes`) with serde aliases for backward compat. `HaltBefore` expanded: `CgSweep` + `PgobUngate` variants between PMC and memory training. `pre_channel_init()` hook runs CG sweep before factory channel creation in ember path. **14 tests pass**, builds clean. See `experiments/202_EXPERIMENT_SURFACE_REWIRE.md` |
| **Warm/Cold Boot Convergence** (Exp 203) | ✅ Implemented | **Warm/cold convergence & firmware bridge freeze** — VBIOS interpreter: 6 PLL opcodes (0x79, 0x4B, 0x34, 0x4A, 0x59, 0x87) activated with BAR0 writes + 4 register copy opcodes (0x88, 0x8F, 0x90, 0x5F) implemented. `FalconWarmState` enum (`Cold`/`WarmPreserved`/`WarmRunning`/`Inconsistent`) replaces inline BAR0 register checks in `falcon_boot()`. `detect_falcon_warm_state()` on `SovereignStrategy` trait. `PfifoInitConfig::for_thermal_state()` unifies config selection. `pfifo_config()` trait method drives PFIFO selection from `FalconWarmState`. `NvGspBridge` + `GspBridge` trait documented as frozen dependency (pinned firmware blobs, hardware-defined upload mechanisms, glacial evolution). **611 tests pass**, builds clean. See `experiments/203_WARM_COLD_BOOT_CONVERGENCE.md` |
| **VBIOS Interpreter Live Validation** (Exp 204) | ✅ Validated | **First live cold-boot VBIOS interpreter execution on Titan V** — iterative opcode/stride debugging on real GV100 hardware. 422 ops, 231 BAR0 writes including PLL programming. Fixed 3 stride bugs: `0x56` (5→3), `0x3A` (3→3+size), `0x4F` (9→5). Added 4 undocumented Volta opcodes: `0xAC` (stride 13), `0xB0` (stride 10), `0xB1` (stride 3), `0x9E` (stride 1 prefix). Consecutive `0xFF` → end-of-script terminator. Graceful desync recovery (100 unknowns → clean script termination). Warm re-run: 489 ops, 378 writes. PMC_ENABLE=0x00000000 confirmed cold. Blocker: opcode `0x9E` at `0x8c2c`. See `experiments/204_VBIOS_INTERPRETER_LIVE_VALIDATION.md` |
| **Dual Titan V Twin Study Baseline** (Exp 205) | ✅ Validated | **Second Titan V installed in K80 slot** — dual GV100 sovereign pipeline validated. boot0=0x140000a1, PMC_ENABLE=0x5fecdff1, PGRAPH=0x00000000 all MATCH. VBIOS ROM byte-identical (sha256=af04a2c6). IOMMU groups 65+32 (clean isolation). Pipeline stage timing within 1ms. CG sweep converged after first run. Twin study surface live for A/B register experiments, warm/cold divergence, falcon ACR debugging. See `experiments/205_DUAL_TITAN_V_TWIN_STUDY_BASELINE.md` |
| **Falcon ACR DMA Boot Solved** (Exp 206) | ✅ Validated | **Falcon boot blocker solved** — ACR HS boot via DMA working on both Titan Vs. `boot_falcon_hs` loads GR firmware blobs via iommufd-mapped DMA buffers, GPCCS+FECS boot via bootloader descriptor. FECS cpuctl=0x10 (halted in command-wait). Three code fixes: DMA backend wiring in stateless handler, EmberGateBypass, gr_init skip after ACR. Infra: 02:00.0 bound to vfio-pci, stale coral-ember killed. Reboot needed for full warm pipeline. See `experiments/206_FALCON_ACR_DMA_BOOT_SOLVED.md` |
| **Sovereign Boot Abstraction + Profiling** (Exp 207) | ✅ Validated | **Unified warm/cold boot model** — `SovereignBootState` enum (Warm/Cold) with `ColdBootReason`, `BootCapability` flags, `probe_boot_state()` as single source of truth. `WarmKeepalive` facade (ember) wraps VfioAnchor + clutch + fd store. `sovereign.profile` RPC with µs-precision timing + register snapshots (BOOT0/PMC/PTIMER/FECS/GPCCS). **Hardware line codified**: cold = power-on reset = boot ROM trains HBM2 = vendor wall. Twin-card cold profiling: card #1 11.3s, card #2 13.0s (HBM2 training variance). See `experiments/207_SOVEREIGN_BOOT_ABSTRACTION_PROFILING.md` |
| **Reboot-Efficient Sovereign Evolution** (Exp 208) | ✅ Validated | **183ms warm pipeline (76× faster than cold)** — `sovereign.warm_status` RPC for instant anchor/boot visibility. Cold pipeline early-exit: 200ms vs 14s (70×). systemd FileDescriptorStore: 4 VFIO fds survive daemon restart. **Falcon warm preservation PROVEN**: early probe before pgraph_reset detects WarmRunning FECS (PC=0xB0+), skips pgraph_reset + falcon_boot entirely. Boot ROM firmware survives VFIO device open — first sovereign.init after power cycle: **185ms**. Twin-card validated (183ms / 183ms). FLR PC threshold (>= 0x40) prevents false positives. Warm state persists across `systemctl restart`. See `experiments/208_REBOOT_EFFICIENT_SOVEREIGN_EVOLUTION.md` |
| **Sovereign VFIO Dispatch Bridge** (Exp 209) | ✅ Pipeline proven | **Anchor-fd adoption bridges ember→dispatch gap** — VFIO device reconstructed from anchor fds when group EBUSY (ember holds it). `ComputeDevice::adopt_anchor_fds()` trait + `open_vfio_from_received()`. PBDMA pushbuffer submission proven on warm 184ms Titan V. coralReef SM70 compile (30ms, 240B binary). FECS setup protocol wired (INIT_CTXSW/BIND_CHANNEL/COMMIT). **PGRAPH power gating gap**: GR method mailbox returns PRI fault (0xbadf5545) after nouveau→vfio-pci handoff — FECS running but method interface gated. Nouveau warm handoff recovery proven. See `experiments/209_SOVEREIGN_VFIO_DISPATCH_BRIDGE.md` |
| **Sovereign GPC Boundary Analysis** (Exp 210) | ✅ Tier 1 Validated | **Hardware power domain boundary mapped** — All engine domains (GR, CE, NVDEC) gated after nouveau unbind. CE0 at 0x104000 → 0xbadf3000, GPCCS → 0xbadf5545. Fixed PTOP_DEVICE_INFO_V2 parser (runlist in kind==2 at bits [17:14]). CE runlist=10 discovered, channel created on non-GR runlist, but CE PBDMA returns DEVICE error (engine gated). **Sovereignty tier model**: Tier 0 (cold/vendor wall), Tier 1 (warm infra — **validated**), Tier 2 (warm compute — **blocked by GPC power**), Tier 3 (full sovereign). `SovereignTier` enum + `classify_tier()` + `TierEvidence` integrated into `sovereign.warm_status` API. NVK does NOT support Volta (SM75+ only). RTX 5060 DRM compute available via proprietary driver. See `experiments/210_SOVEREIGN_GPC_BOUNDARY.md` |
| **TOTAL** | **65 validation suites (smoke/nucleus/silicon)** | **634 (cylinder) / 596 (default barracuda) / 1,045 (barracuda-local) tests (lib)**, 167 binaries, 128 WGSL shaders, 7 deploy graphs. **guideStone Level 6 CERTIFIED** (NUCLEUS Deployment Validation, primalSpring v0.9.25). Zero clippy, `#![forbid(unsafe_code)]` on lib (unsafe confined to low-level experiment bins), zero `dyn` dispatch (prod), AGPL-3.0-only. `deny.toml` enforced (ecoBin C-dep bans). `#[expect(lint, reason)]` in all production code. Compile-then-dispatch pipeline wired. Tier 4 IPC-first (`default = []`). **Science ladder:** Quenched → Gradient Flow → Integrators → N_f=4 Infra → Chuna 44/44 → N_f=2 → N_f=2+1 → Self-tuning → Silicon saturation → 16⁴+ production → Firmware Boundary → NOP Dispatch → SovereignInit Pipeline → NUCLEUS Composition → Primal Composition Proof → Level 6 CERTIFIED → Tier 4 IPC-First → K80 Warm-Catch → Three-GPU Sovereign → LTEE B2 Anderson → Compute Trio Pipeline → HW Validation Sprint → PLX D3cold Keepalive → Cold/Warm Boot Architecture → Driver Lab (Mesa vs Vendor) → Warm Swap Validation + PLX Keepalive → Sovereign Init RPC → Vendor-Agnostic BootPipeline → Diesel Engine Power Safety → Volta Cold Boot CG Sweep → Experiment Surface Rewire → Warm/Cold Boot Convergence → VBIOS Interpreter Live HW Validation → Dual Titan V Twin Study → Falcon ACR DMA Boot → Sovereign Boot Abstraction + Profiling → **Warm Keepalive PROVEN (183ms)** → **Sovereign VFIO Dispatch Bridge** → **GPC Boundary Analysis — Sovereignty Tier Model**. **210 experiments**. Experiments 001-190 archived to `experiments/archive/`. Fleet: 2× Titan V (GV100) + RTX 5060 + K80 incoming (GK210). K80 #1 caught fire (Exp 199); replacement on order for cross-generational sovereign validation. |

---

## Benchmark Data

### Nuclear EOS Head-to-Head: BarraCuda vs Python

| Metric | Python L1 | BarraCuda L1 | Python L2 | BarraCuda L2 |
|--------|-----------|-------------|-----------|-------------|
| Best χ²/datum | 6.62 | **2.27** ✅ | **1.93** | **16.11** |
| Best NMP-physical | — | — | — | 19.29 (5/5 within 2σ) |
| Total evals | 1,008 | 6,028 | 3,008 | 60 |
| Total time | 184s | **2.3s** | 3.2h | 53 min |
| Throughput | 5.5 evals/s | **2,621 evals/s** | 0.28 evals/s | 0.48 evals/s |
| Speedup | — | **478×** | — | **1.7×** |

### The f64 Bottleneck: Broken

Double-float (f32-pair) on FP32 cores delivers 3.24 TFLOPS at 14-digit precision — 9.9x native f64. Consumer Ampere/Ada GPUs have hardware fp64:fp32 ~1:64 (both CUDA and Vulkan).

| Metric | Software f64 (before) | Native f64 (after) | Improvement |
|--------|----------------------|-------------------|-------------|
| N=500 steps/s | 169.0 | **998.1** | **5.9×** |
| N=2,000 steps/s | 76.0 | **361.5** | **4.8×** |
| N=5,000 steps/s | 66.9 | **134.9** | **2.0×** |
| N=10,000 steps/s | 24.6 | **110.5** | **4.5×** |
| N=20,000 steps/s | 8.6 | **56.1** | **6.5×** |
| Wall time (full sweep) | 113 min | **34 min** | **3.3×** |

### Paper Parity Assessment — ACHIEVED

| Capability | Murillo Group (HPC) | hotSpring (RTX 4070) | Gap |
|-----------|--------------------|--------------------|-----|
| Particle count | 10,000 | **10,000** ✅ | None |
| Production steps | 80,000-100,000+ | **80,000** ✅ | None |
| Energy conservation | ~0% | **0.000-0.002%** ✅ | None |
| 9 PP Yukawa cases | All pass | **9/9 pass** ✅ | None |
| Hardware cost | $M+ cluster | **$600 GPU** ✅ | 1000× cheaper |
| Total wall time | Not published | **3.66 hours** (9 cases) | Consumer GPU |
| Total energy cost | Not published | **$0.044** electricity | Sovereign science |

---

## Studies

### Study 1: Sarkas Molecular Dynamics

Reproduce plasma simulations from the Dense Plasma Properties Database. 12 cases: 9 Yukawa PP (κ=1,2,3 × Γ=low,mid,high) + 3 Coulomb PPPM (κ=0 × Γ=10,50,150).

- **Source**: [github.com/murillo-group/sarkas](https://github.com/murillo-group/sarkas) (MIT)
- **Reference**: [Dense Plasma Properties Database](https://github.com/MurilloGroupMSU/Dense-Plasma-Properties-Database)
- **Result**: 60/60 observable checks pass (DSF 8.5% mean error PP, 7.3% PPPM)

### Study 2: Surrogate Learning (Nature MI 2024)

Reproduce "Efficient learning of accurate surrogates for simulations of complex systems" (Diaw et al., 2024).

- **Paper**: [doi.org/10.1038/s42256-024-00839-1](https://doi.org/10.1038/s42256-024-00839-1)
- **Data**: [Zenodo: 10.5281/zenodo.10908462](https://doi.org/10.5281/zenodo.10908462)
- **Result**: 9/9 benchmark functions reproduced. Physics EOS from MD data converged.

### Study 3: Two-Temperature Model

Run the UCLA-MSU TTM for laser-plasma equilibration.

- **Source**: [github.com/MurilloGroupMSU/Two-Temperature-Model](https://github.com/MurilloGroupMSU/Two-Temperature-Model)
- **Result**: 6/6 checks pass (3 local + 3 hydro).

---

## Upstream Bugs Found and Fixed

| # | Bug | Where | Impact |
|---|-----|-------|--------|
| 1 | `np.int` removed in NumPy 2.x | `sarkas/tools/observables.py` | Silent DSF/SSF failure |
| 2 | `.mean(level=)` removed in pandas 2.x | `sarkas/tools/observables.py` | Silent DSF failure |
| 3 | Numba 0.60 `@jit` → `nopython=True` breaks pyfftw | `sarkas/potentials/force_pm.py` | PPPM method crashes |
| 4 | Thomas-Fermi `χ₁=NaN` poisons recombination | TTM `exp_setup.py` | Zbar solver diverges |
| 5 | DSF reference file naming (case sensitivity) | Plasma Properties DB | Validation script fails |
| 6 | Multithreaded dump corruption (v1.1.0) | Sarkas `4b561baa` | All `.npz` checkpoints NaN |

---

## Document Index

| Document | Purpose |
|----------|---------|
| [`PHYSICS.md`](PHYSICS.md) | Complete physics documentation — every equation, constant, approximation |
| [`whitePaper/CONTROL_EXPERIMENT_SUMMARY.md`](whitePaper/CONTROL_EXPERIMENT_SUMMARY.md) | Phase A summary with numbers |
| [`barracuda/CHANGELOG.md`](barracuda/CHANGELOG.md) | Crate version history |
| [`barracuda/EVOLUTION_READINESS.md`](barracuda/EVOLUTION_READINESS.md) | Module → shader → GPU promotion tier |
| [`specs/README.md`](specs/README.md) | Specification index |
| [`specs/PAPER_REVIEW_QUEUE.md`](specs/PAPER_REVIEW_QUEUE.md) | Papers to review/reproduce |
| [`whitePaper/README.md`](whitePaper/README.md) | White paper index |
| [`whitePaper/baseCamp/`](whitePaper/baseCamp/) | Per-domain research briefings |
| [`metalForge/README.md`](metalForge/README.md) | Hardware characterization |
| [`metalForge/npu/akida/BEYOND_SDK.md`](metalForge/npu/akida/BEYOND_SDK.md) | 10 overturned SDK assumptions |

### External References

| Reference | DOI / URL | Used For |
|-----------|-----------|----------|
| Diaw et al. (2024) *Nature Machine Intelligence* | [10.1038/s42256-024-00839-1](https://doi.org/10.1038/s42256-024-00839-1) | Surrogate learning methodology |
| Sarkas MD package | [github.com/murillo-group/sarkas](https://github.com/murillo-group/sarkas) (MIT) | DSF plasma simulations |
| Dense Plasma Properties Database | [github.com/MurilloGroupMSU](https://github.com/MurilloGroupMSU/Dense-Plasma-Properties-Database) | DSF reference spectra |
| Two-Temperature Model | [github.com/MurilloGroupMSU/Two-Temperature-Model](https://github.com/MurilloGroupMSU/Two-Temperature-Model) | Plasma equilibration |
| Zenodo surrogate archive | [10.5281/zenodo.10908462](https://doi.org/10.5281/zenodo.10908462) (CC-BY) | Convergence histories |
| AME2020 (Wang et al. 2021) | [IAEA Nuclear Data](https://www-nds.iaea.org/amdc/ame2020/) | Experimental binding energies |
