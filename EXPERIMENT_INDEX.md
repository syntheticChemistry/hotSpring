# hotSpring — Experiment & Validation Index

> Updated March 30, 2026. This is the authoritative ledger of all
> experiments, validation suites, and benchmark data. For project overview, see [README.md](README.md).
> Experiments 001-057 archived to `experiments/archive/` — completed physics and benchmark work, results absorbed into baseCamp.

**131+ experiments** | **500+ quantitative checks** | **~$0.30 total science cost** | **AGPL-3.0-only**

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
| **Pure Gauge SU(3)** (Paper 8) | ✅ Complete | 12/12 pass (HMC, Dirac CG, plaquette physics) |
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
| **Ember Architecture** | ✅ Complete | Immortal VFIO fd holder (`coral-ember`): `SCM_RIGHTS` fd passing, atomic `swap_device` RPC, DRM isolation preflight, external fd holder detection. Zero-crash driver hot-swap on live system |
| **DRM Isolation** | ✅ Complete | Xorg `AutoAddGPU=false` + udev seat tag removal (61-prefix) prevents compositor crash during driver swaps. Compute GPUs fully invisible to display manager |
| **Dual Titan Backend Matrix** (Exp 070) | ✅ Complete | Both Titans on GlowPlug/Ember. vfio↔nouveau swap validated (oracle). Full backend matrix: vfio, nouveau, nvidia × 2 cards. Register diff infrastructure ready |
| **PFIFO Diagnostic Matrix** (Exp 071) | ✅ Complete | 54-config matrix: 12 winning configs, 0 faults, scheduler-accepted. PFIFO re-init solved (PMC+preempt+clear). Root cause identified (PBDMA 0xbad00200 PBUS timeout) — resolved in Exp 076 (MMU fault buffer) + Exp 077 (init hardening). 6/10 sovereign pipeline layers proven. |
| **MMU Fault Buffer Breakthrough** (Exp 076) | ✅ Complete | **Layer 6 resolved.** Volta FBHUB requires configured non-replayable fault buffers before any MMU page table walk completes. Without them, PBUS returns 0xbad00200 and PBDMA stalls forever. Fix: FAULT_BUF0/1 configured in VfioChannel::create. Channel creation + DMA roundtrip + MMU translation all pass. Shader dispatch blocked at Layer 7 (GR/FECS context). |
| **PFIFO Init Hardening** (Exp 077) | ✅ Complete | Five failure modes documented and fixed. `PfifoInitConfig` unifies init paths, `GpuCapabilities` makes matrix arch-aware, `coralctl reset` provides PCIe FLR recovery. |
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
| **DRM Dispatch Evolution** (Exp 072) | ✅ GCN5 Complete | **AMD GCN5 6/6 PASS.** **RTX 5060 Blackwell DRM cracked.** K80 incoming. |
| **iommufd/cdev VFIO Evolution** (Exp 073) | ✅ Complete | **Kernel-agnostic VFIO** on Linux 6.2+. 607 tests pass. |
| **Ember Swap Pipeline Evolution** (Exp 074) | ✅ Complete | **nouveau ↔ vfio round-trip proven** on Titan V. 86 ember + 178 glowplug tests. |
| **Deep Debt + Cross-Vendor Dispatch** (Exp 075) | ✅ Complete | **13 deep-debt items resolved.** Cross-vendor CUDA dispatch via glowplug daemon RPC. |
| **Vendor-Agnostic GlowPlug** | ✅ Complete | coral-ember standalone crate. RegisterMap trait (GV100 + GFX906/MI50). |
| **Privilege Hardening** | ✅ Complete | Capabilities + seccomp + namespaces. |
| **VendorLifecycle Trait** | ✅ Complete | Vendor-specific swap hooks. 157 tests pass. |
| **AMD D3cold Resolution** | ✅ Characterized | Vega 20 SMU firmware limitation: one vfio→amdgpu cycle per boot. |
| **BrainChip Akida NPU** | ✅ Complete | AKD1000 fully integrated. Unlimited round-trips, SimpleBind, no DRM. |
| **Zero-Sudo coralctl** | ✅ Complete | `coralreef` unix group, socket permissions. |
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
| **Debt Reduction Audit** | ✅ Complete | 685 tests (lib), 47 validation binaries. |
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
| **Sovereign DEVINIT** (Exp 120) | ✅ Complete | biomeGate: sovereign DEVINIT not needed |
| **WPR2 Resolution** (Exp 122) | ✅ Complete | biomeGate: **definitive root cause** — WPR2 registers hardware-locked by FWSEC |
| **Parasitic Compute** (Exp 123T) | ✅ Complete | biomeGate: parasitic compute probe |
| **K80 Sovereign Compute** (Exp 123) | 🔄 Active | biomeGate: Tesla K80 (GK210) — zero firmware security, direct PIO boot |
| **AMD Scratch/Local Memory Breakthrough** (Exp 124) | ✅ Complete | strandgate: AMD RX 6950 XT (RDNA2) scratch/local memory dispatch via coralReef DRM path |
| **VM Capture Cross-Analysis** (Exp 124b) | ✅ Complete | biomeGate: nvidia-470/535 VM captures for K80+Titan V, cross-driver register tracing |
| **Warm Handoff Livepatch** (Exp 125) | 🔄 Active | biomeGate: kernel livepatch NOP (mc_reset+gr_fini+falcon_fini+runl_commit), dynamic enable/disable, reset_method sysfs fix, PBDMA warm mode |
| **DRM Proprietary Tracing Matrix** (Exp 126) | 🔄 Active | biomeGate: map all non-VFIO dispatch paths for Titan V (DRM+proprietary dual-use) |
| **Warm FECS Dispatch Attack** (Exp 127) | ✅ Complete | biomeGate: FECS firmware preserved in IMEM but cannot be woken from HS+ halt state |
| **GPU Puzzle Box Matrix** (Exp 128) | 🔄 Active | biomeGate: multi-path sovereign compute matrix — firmware interface approach |
| **No-FLR Recovery & PRI Ring Lessons** (Exp 130) | ✅ Complete | biomeGate: K80 GK210 PRI ring diagnostics, cold GPU detection, PMU/FECS falcon state analysis |
| **Reset Architecture Evolution** (Exp 131) | ✅ Complete | biomeGate: warm_fecs.rs → device.warm_handoff RPC, livepatch into ember, orphan cleanup |
| **TOTAL** | **39/39 Rust validation suites** | **870 tests (lib)**, 139 binaries, 99 WGSL shaders. Zero clippy, zero unsafe, AGPL-3.0-only. **Science ladder:** Quenched → Gradient Flow → Integrators → N_f=4 Infra → Chuna 44/44 → N_f=2 → N_f=2+1 → Self-tuning → Silicon saturation → 16⁴+ production. 130+ experiments. Experiments 001-057 archived to `experiments/archive/` (completed physics validation, absorbed into baseCamp). |

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
| [`CONTROL_EXPERIMENT_STATUS.md`](CONTROL_EXPERIMENT_STATUS.md) | Full status with numbers, evolution history |
| [`NUCLEAR_EOS_STRATEGY.md`](NUCLEAR_EOS_STRATEGY.md) | Nuclear EOS Phase A→B strategy |
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
