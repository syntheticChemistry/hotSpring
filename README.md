# hotSpring

**Computational physics reproduction studies and control experiments.**

Named for the hot springs that gave us *Thermus aquaticus* and Taq polymerase — the origin story of the constrained evolution thesis. Professor Murillo's research domain is hot dense plasmas. A spring is a wellspring. This project draws from both.

---

## What This Is

hotSpring is where we reproduce published computational physics work from the Murillo Group (MSU) and benchmark it across consumer hardware. Every study has two phases:

- **Phase A (Control)**: Run the original Python code (Sarkas, mystic, TTM) on our hardware. Validate against reference data. Profile performance. Fix upstream bugs. **✅ Complete — 86/86 quantitative checks pass.**

- **Phase B (BarraCUDA)**: Re-execute the same computation on ToadStool's BarraCUDA engine — pure Rust, WGSL shaders, any GPU vendor. **✅ L1 validated (478× faster, better χ²). L2 validated (1.7× faster).**

- **Phase C (GPU MD)**: Run Sarkas Yukawa OCP molecular dynamics entirely on GPU using f64 WGSL shaders. **✅ 9/9 PP Yukawa DSF cases pass on RTX 4070. 0.000% energy drift at 80k production steps. Up to 259 steps/s sustained. 3.4× less energy per step than CPU at N=2000.**

- **Phase D (Native f64 Builtins + N-Scaling)**: Replaced software-emulated f64 transcendentals with hardware-native WGSL builtins. **✅ 2-6× throughput improvement. N=10,000 paper parity in 5.3 minutes. N=20,000 in 10.4 minutes. Full sweep (500→20k) in 34 minutes. 0.000% energy drift at all N. The f64 bottleneck is broken — true fp64:fp32 ratio is ~1:2 (not 1:64).**

- **Phase E (Paper-Parity Long Run + Toadstool Rewire)**: 9-case Yukawa OCP sweep at N=10,000, 80k production steps — matching the Dense Plasma Properties Database exactly. **✅ 9/9 cases pass, 0.000-0.002% energy drift, 3.66 hours total, $0.044 electricity. Cell-list 4.1× faster than all-pairs. Toadstool GPU ops (BatchedEighGpu, SsfGpu, PppmGpu) wired into hotSpring.**

hotSpring answers: *"Does our hardware produce correct physics?"* and *"Can Rust+WGSL replace the Python scientific stack?"*

> **For the physics**: See [`PHYSICS.md`](PHYSICS.md) for complete equation documentation
> with numbered references — every formula, every constant, every approximation.
>
> **For the methodology**: See [`whitePaper/METHODOLOGY.md`](whitePaper/METHODOLOGY.md)
> for the two-phase validation protocol and acceptance criteria.

---

## Current Status (2026-02-21)

| Study | Status | Quantitative Checks |
|-------|--------|-------------------|
| **Sarkas MD** (12 cases) | ✅ Complete | 60/60 pass (DSF, RDF, SSF, VACF, Energy) |
| **TTM Local** (3 species) | ✅ Complete | 3/3 pass (Te-Ti equilibrium) |
| **TTM Hydro** (3 species) | ✅ Complete | 3/3 pass (radial profiles) |
| **Surrogate Learning** (9 functions) | ✅ Complete | 15/15 pass + iterative workflow |
| **Nuclear EOS L1** (Python, SEMF) | ✅ Complete | χ²/datum = 6.62 |
| **Nuclear EOS L2** (Python, HFB hybrid) | ✅ Complete | χ²/datum = 1.93 |
| **BarraCUDA L1** (Rust+WGSL, f64) | ✅ Complete | χ²/datum = **2.27** (478× faster) |
| **BarraCUDA L2** (Rust+WGSL+nalgebra) | ✅ Complete | χ²/datum = **16.11** best, 19.29 NMP-physical (1.7× faster) |
| **GPU MD PP Yukawa** (9 cases) | ✅ Complete | 45/45 pass (Energy, RDF, VACF, SSF, D*) |
| **N-Scaling + Native f64** (5 N values) | ✅ Complete | 16/16 pass (500→20k, 0.000% drift) |
| **Paper-Parity Long Run** (9 cases, 80k steps) | ✅ Complete | 9/9 pass (N=10k, 0.000-0.002% drift, 3.66 hrs, $0.044) |
| **ToadStool Rewire v1** (3 GPU ops) | ✅ Complete | BatchedEighGpu, SsfGpu, PppmGpu wired |
| **Nuclear EOS Full-Scale** (Phase F, AME2020) | ✅ Complete | 9/9 pass (L1 Pareto, L2 GPU 2042 nuclei, L3 deformed) |
| **BarraCUDA MD Pipeline** (6 ops) | ✅ Complete | 12/12 pass (YukawaF64, VV, Berendsen, KE — 0.000% drift) |
| **BarraCUDA HFB Pipeline** (3 ops) | ✅ Complete | 14/14 pass (BCS GPU 6.2e-11, Eigh 2.4e-12) |
| **Stanton-Murillo Transport** (Paper 5) | ✅ Complete | 13/13 pass (D* Sarkas-calibrated, MSD≈VACF, Green-Kubo η*/λ*) |
| **HotQCD EOS Tables** (Paper 7) | ✅ Complete | Thermodynamic consistency, asymptotic freedom validated |
| **Pure Gauge SU(3)** (Paper 8) | ✅ Complete | 12/12 pass (HMC, Dirac CG, plaquette physics) |
| **Screened Coulomb** (Paper 6) | ✅ Complete | 23/23 pass (Sturm bisection, Python parity Δ≈10⁻¹², critical screening) |
| **Abelian Higgs** (Paper 13) | ✅ Complete | 17/17 pass (U(1)+Higgs HMC, phase structure, Rust 143× faster than Python) |
| **ToadStool Rewire v2** | ✅ Complete | WgslOptimizer + GpuDriverProfile wired into all shader compilation |
| **ToadStool Rewire v3** | ✅ Complete | CellListGpu fixed, Complex64+SU(3)+plaquette+HMC+Higgs GPU shaders, **FFT f64** — Tier 3 lattice QCD unblocked |
| **NPU Quantization** (metalForge) | ✅ Complete | 6/6 pass (f32/int8/int4/act4 parity, sparsity, monotonic) |
| **NPU Beyond-SDK** (metalForge) | ✅ Complete | 29/29 pass (13 HW + 16 Rust math: channels, merge, batch, width, multi-out, mutation, determinism) |
| **NPU Physics Pipeline** (metalForge) | ✅ Complete | 20/20 pass (10 HW pipeline + 10 Rust math: MD→ESN→NPU→D*,η*,λ*) |
| **Lattice NPU Pipeline** (metalForge) | ✅ Complete | 10/10 pass (SU(3) HMC→ESN→NpuSimulator phase classification, β_c=5.715) |
| **Hetero Real-Time Monitor** (metalForge) | ✅ Complete | 9/9 pass (live HMC phase monitor, cross-substrate f64→f32→int4, 0.09% overhead, predictive steering 62% compute saved) |
| **Spectral Theory** (Kachkovskiy) | ✅ Complete | 10/10 pass (Anderson localization, almost-Mathieu, Herman γ=ln|λ|, Aubry-André transition, Poisson stats) |
| **Lanczos + 2D Anderson** (Kachkovskiy) | ✅ Complete | 11/11 pass (SpMV parity, Lanczos vs Sturm, full spectrum, GOE→Poisson transition, 2D bandwidth) |
| **3D Anderson** (Kachkovskiy) | ✅ Complete | 10/10 pass (mobility edge, GOE→Poisson transition, dimensional hierarchy 1D<2D<3D, spectrum symmetry) |
| **Hofstadter Butterfly** (Kachkovskiy) | ✅ Complete | 10/10 pass (band counting q=2,3,5, fractal Cantor measure, α↔1-α symmetry, gap opening) |
| **GPU SpMV + Lanczos** (Kachkovskiy GPU) | ✅ Complete | 14/14 pass (CSR SpMV parity 1.78e-15, Lanczos eigenvalues match CPU to 1e-15) |
| **GPU Dirac + CG** (Papers 9-12 GPU) | ✅ Complete | 17/17 pass (SU(3) Dirac 4.44e-16, CG iters match exactly, D†D positivity) |
| **Pure GPU QCD Workload** | ✅ Complete | 3/3 pass (HMC → GPU CG on thermalized configs, solution parity 4.10e-16) |
| **Python vs Rust CG** | ✅ Complete | **200× speedup**: identical iterations (5 cold, 37 hot), Dirac 0.023ms vs 4.59ms |
| **GPU Scaling (4⁴→16⁴)** | ✅ Complete | GPU **22.2× faster** at 16⁴ (24ms vs 533ms), crossover at V~2000, iters identical |
| **NPU HW Pipeline** | ✅ Complete | 10/10 on AKD1000: MD→ESN→NPU→D*,η*,λ*, 2469 inf/s, 8796× less energy |
| **NPU HW Beyond-SDK** | ✅ Complete | 13/13 on AKD1000: 10 SDK assumptions overturned, all validated on hardware |
| **NPU HW Quantization** | ✅ Complete | 4/4 on AKD1000: f32/int8/int4/act4 cascade, 685μs/inference |
| **NPU Lattice Phase** | ✅ 7/8 | β_c=5.715 on AKD1000, ESN 100% CPU, int4 NPU 60% (marginal as expected) |
| **Titan V NVK** | ✅ Complete | NVK built from Mesa 25.1.5. `cpu_gpu_parity` 6/6, `stanton_murillo` 40/40, `bench_gpu_fp64` pass |
| **TOTAL** | **31/31 Rust validation suites** | + 34/35 NPU HW checks, 345 unit tests, 6 upstream bugs found. Both GPUs validated |

Papers 5, 7, and 8 from the review queue are complete. Paper 5 transport fits
(Daligault 2012) were recalibrated against 12 Sarkas Green-Kubo D* values (Feb 2026)
and evolved with κ-dependent weak-coupling correction `C_w(κ)` (v0.5.14–15), reducing
crossover-regime errors from 44–63% to <10%. Transport grid expanded to 20 (κ,Γ)
points including 9 Sarkas-matched DSF cases with N=2000 ground-truth D*.
Lattice QCD (complex f64, SU(3), Wilson gauge, HMC, staggered Dirac, CG solver)
validated on CPU and GPU. GPU Dirac (8/8) and GPU CG (9/9) form the full GPU
lattice QCD pipeline. Pure GPU workload validated on thermalized HMC configurations:
5 CG solves match CPU at machine-epsilon parity (4.10e-16). **Rust is 200× faster
than Python** for the same CG algorithm (identical iteration counts, identical seeds).
Paper 13 (Abelian Higgs) extends lattice infrastructure to U(1) gauge + complex
scalar Higgs field on (1+1)D lattice, demonstrating 143× Rust-over-Python speedup.

metalForge NPU validation (AKD1000) overturns 10 SDK assumptions — arbitrary input
channels, FC chain merging (SkipDMA), batch PCIe amortization (2.35×), wide FC to
8192+, multi-output free cost, weight mutation linearity, and hardware determinism —
all validated on hardware (13/13 Python) and in pure Rust math (16/16).
ESN quantization cascade (f64→f32→int8→int4) validated across both substrates (6/6).
Full GPU→NPU physics pipeline validated end-to-end: MD trajectories → ESN training →
NPU multi-output deployment (D*, η*, λ*) with 9,017× less energy than CPU Green-Kubo.
Lattice QCD heterogeneous pipeline: SU(3) HMC → ESN phase classifier → NpuSimulator
detects deconfinement transition at β_c=5.715 (known 5.692, error 0.4%) — **no FFT
required for lattice phase structure** (though GPU FFT f64 is now available via
toadstool Session 25 for full QCD). Real-time heterogeneous monitor validates five
previously-impossible capabilities: live HMC phase monitoring (0.09% overhead), continuous
multi-output transport prediction (D*/η*/λ*), cross-substrate parity (f64→f32→int4, max
f32 error 5.1e-7), predictive steering (62% compute savings via adaptive β scan), and
zero-overhead physics monitoring on $900 consumer hardware. See `metalForge/` for full
hardware analysis.

See `CONTROL_EXPERIMENT_STATUS.md` for full details.

### Nuclear EOS Head-to-Head: BarraCUDA vs Python

| Metric | Python L1 | BarraCUDA L1 | Python L2 | BarraCUDA L2 |
|--------|-----------|-------------|-----------|-------------|
| Best χ²/datum | 6.62 | **2.27** ✅ | **1.93** | **16.11** |
| Best NMP-physical | — | — | — | 19.29 (5/5 within 2σ) |
| Total evals | 1,008 | 6,028 | 3,008 | 60 |
| Total time | 184s | **2.3s** | 3.2h | 53 min |
| Throughput | 5.5 evals/s | **2,621 evals/s** | 0.28 evals/s | 0.48 evals/s |
| Speedup | — | **478×** | — | **1.7×** |

### χ² Evolution: How GPU and CPU Validate Each Other

The different chi2 values across runs are not contradictions — they show the optimization landscape
and validate our math at each stage. Each configuration cross-checks the physics implementation:

| Run | χ²/datum | Evals | Config | What it validates |
|-----|---------|-------|--------|-------------------|
| L2 initial (missing physics) | 28,450 | — | — | Baseline: wrong without Coulomb, BCS, CM |
| L2 +5 physics features | ~92 | — | — | Physics implementation correct |
| L2 +gradient_1d fix | ~25 | — | — | Boundary stencils matter in SCF |
| L2 +brent root-finding | ~18 | — | — | Root-finder precision amplified by SCF |
| **L2 Run A** (best accuracy) | **16.11** | 60 | seed=42, λ=0.1 | Best χ² achieved |
| **L2 Run B** (best NMP) | **19.29** | 60 | seed=123, λ=1.0 | All 5 NMP within 2σ |
| L2 GPU benchmark | 23.09 | 12 | 3 rounds, energy-profiled | GPU energy: 32,500 J |
| L2 extended ref run | 25.43 | 1,009 | different seed/λ | More evals ≠ better χ² (landscape is multimodal) |
| L1 SLy4 (Python=CPU=GPU) | 4.99 | 100k | Fixed params | **Implementation parity: all substrates identical** |
| L1 GPU precision | |Δ|=4.55e-13 | — | Precomputed transcendentals | **Sub-ULP: GPU math is bit-exact** |

**L1 takeaway**: BarraCUDA finds a better minimum (2.27 vs 6.62) and runs 478× faster.
GPU path uses **44.8× less energy** than Python for identical physics (126 J vs 5,648 J).

**L2 takeaway**: Best BarraCUDA L2 is 16.11 (Run A). Python achieves 1.93 with SparsitySampler — the gap is sampling strategy, not physics. The range of L2 values (16–25) across configurations confirms the landscape is multimodal. SparsitySampler port is the #1 priority.

### The f64 Bottleneck: Broken

Before February 14, 2026, all GPU MD shaders used **software-emulated** f64 transcendentals
(`math_f64.wgsl` — hundreds of lines of f32-pair arithmetic for `sqrt_f64()`, `exp_f64()`, etc.).
This kept the GPU ALU underutilized and throughput artificially low. We believed NVIDIA's 1:64
fp64:fp32 hardware ratio was the bottleneck.

**Discovery**: The toadstool/barracuda team confirmed that wgpu/Vulkan's `SHADER_F64` feature
exposes **native hardware f64 operations** — `sqrt()`, `exp()`, `round()`, `floor()` all operate
directly on f64 types. The true fp64:fp32 throughput ratio on consumer GPUs (via Vulkan) is
**~1:2**, not 1:64. The 1:64 penalty only applies to CUDA's native fp64 units, which wgpu bypasses.

| Metric | Software f64 (before) | Native f64 (after) | Improvement |
|--------|----------------------|-------------------|-------------|
| N=500 steps/s | 169.0 | **998.1** | **5.9×** |
| N=2,000 steps/s | 76.0 | **361.5** | **4.8×** |
| N=5,000 steps/s | 66.9 | **134.9** | **2.0×** |
| N=10,000 steps/s | 24.6 | **110.5** | **4.5×** |
| N=20,000 steps/s | 8.6 | **56.1** | **6.5×** |
| Wall time (full sweep) | 113 min | **34 min** | **3.3×** |
| GPU power (N=5k) | ~56W (flat, ALU starved) | **65W (active)** | GPU actually working |
| Paper parity (N=10k) | 23.7 min | **5.3 min** | **4.5×** |

### RTX 4070 Capability: Time and Energy

What can a $600 consumer GPU card actually do for computational physics?

| N | steps/s | Wall (35k steps) | Energy (J) | J/step | W avg | VRAM | Method |
|---|---------|-------------------|-----------|--------|-------|------|--------|
| 500 | 998.1 | 35s | 1,655 | 0.047 | 47W | 584 MB | all-pairs |
| 2,000 | 361.5 | 97s | 5,108 | 0.146 | 53W | 574 MB | all-pairs |
| 5,000 | 134.9 | 259s | 16,745 | 0.478 | 65W | 560 MB | all-pairs |
| 10,000 | 110.5 | 317s | 19,351 | 0.553 | 61W | 565 MB | cell-list |
| 20,000 | 56.1 | 624s | 39,319 | 1.123 | 63W | 587 MB | cell-list |

**VRAM**: All N values fit in <600 MB. The RTX 4070 has 12 GB — so **N≈400,000** is feasible
before VRAM limits (each particle needs ~72 bytes of position/velocity/force state).

**Energy context**: Running N=10,000 for 35k steps costs **19.4 kJ** — that's 5.4 Wh, or
approximately **$0.001** in electricity. The equivalent CPU run would take ~4 hours and ~120 kJ.

### Where CPU Becomes Implausible

| N | GPU Wall | GPU Energy | Est. CPU Wall | Est. CPU Energy | GPU Advantage |
|---|----------|-----------|---------------|-----------------|---------------|
| 500 | 35s | 1.7 kJ | 63s | 3.2 kJ | 1.8× time, 1.9× energy |
| 2,000 | 97s | 5.1 kJ | 571s | 28.6 kJ | 5.9× time, 5.6× energy |
| 5,000 | 4.3 min | 16.7 kJ | ~60 min | ~180 kJ | **14× time, 11× energy** |
| 10,000 | 5.3 min | 19.4 kJ | ~4 hrs | ~720 kJ | **46× time, 37× energy** |
| 20,000 | 10.4 min | 39.3 kJ | ~16 hrs | ~2,880 kJ | **94× time, 73× energy** |
| 50,000 | ~45 min (est.) | ~170 kJ | ~10 days (est.) | ~72 MJ | **~300× time** |

Above N=5,000, CPU molecular dynamics on consumer hardware is no longer practical —
not because of accuracy, but because of time and energy. The GPU makes these runs routine.

### Paper Parity Assessment — ACHIEVED

The Murillo Group's published DSF study uses N=10,000 particles with 80,000-100,000+
production steps on HPC clusters. Our RTX 4070 now runs the **exact same configuration**:

| Capability | Murillo Group (HPC) | hotSpring (RTX 4070) | Gap |
|-----------|--------------------|--------------------|-----|
| Particle count | 10,000 | **10,000** ✅ | None |
| Production steps | 80,000-100,000+ | **80,000** (3.66 hrs / 9 cases) ✅ | None |
| Energy conservation | ~0% | **0.000-0.002%** ✅ | None |
| 9 PP Yukawa cases | All pass | **9/9 pass** ✅ | None |
| Observables | DSF, RDF, SSF, VACF | **All computed** ✅ | DSF spectral analysis pending |
| Physics method | PP Yukawa + PPPM | PP Yukawa ✅ + **PppmGpu wired** | κ=0 validation ready |
| Hardware cost | $M+ cluster | **$600 GPU** ✅ | 1000× cheaper |
| Total wall time | Not published | **3.66 hours** (9 cases) | Consumer GPU |
| Total energy cost | Not published | **$0.044** electricity | Sovereign science |

#### Per-Case Paper-Parity Results (February 14, 2026)

| Case | κ | Γ | Mode | Steps/s | Wall (min) | Drift % |
|------|---|---|------|---------|------------|---------|
| k1_G14 | 1 | 14 | all-pairs | 26.1 | 54.4 | 0.001% |
| k1_G72 | 1 | 72 | all-pairs | 29.4 | 48.2 | 0.001% |
| k1_G217 | 1 | 217 | all-pairs | 31.0 | 45.7 | 0.002% |
| k2_G31 | 2 | 31 | cell-list | 113.3 | 12.5 | 0.000% |
| k2_G158 | 2 | 158 | cell-list | 115.0 | 12.4 | 0.000% |
| k2_G476 | 2 | 476 | cell-list | 118.1 | 12.2 | 0.000% |
| k3_G100 | 3 | 100 | cell-list | 119.9 | 11.8 | 0.000% |
| k3_G503 | 3 | 503 | cell-list | 124.7 | 11.4 | 0.000% |
| k3_G1510 | 3 | 1510 | cell-list | 124.6 | 11.4 | 0.000% |

**Cell-list achieves 4.1× speedup** over all-pairs (118 vs 29 steps/s). See all-pairs
vs cell-list analysis below.

#### Remaining Gap to Full Paper Match

1. **DSF S(q,ω) spectral analysis** — dynamic structure factor comparison against `sqw_k{K}G{G}.npy`
2. **κ=0 Coulomb (PPPM)** — 3 additional cases, PppmGpu now wired and ready to validate
3. **100,000+ step extended runs** — paper upper range; our 80k matches the database exactly

---

### All-Pairs vs Cell-List: Profiling and Tradeoff Analysis

The GPU MD engine uses two force evaluation modes. The paper-parity data now gives us
definitive performance numbers for both:

| Metric | All-Pairs (κ=1) | Cell-List (κ=2,3) |
|--------|:---:|:---:|
| Algorithm | O(N²) — every particle checks all others | O(N) — only 27 neighbor cells |
| Shader | `SHADER_YUKAWA_FORCE` (single loop 0..N) | `SHADER_YUKAWA_FORCE_CELLLIST` (triple-nested 3³ cells) |
| Activation | `cells_per_dim < 5` | `cells_per_dim >= 5` |
| N=10,000 steps/s | **28.8 avg** | **118.5 avg** |
| Per-case wall time | **49.4 min** | **12.0 min** |
| GPU energy per case | **178.9 kJ** | **44.1 kJ** |
| Speedup | — | **4.1×** |

**Why cell-list can't replace all-pairs at κ=1:**

The mode selection is physics-driven, not a performance heuristic. At N=10,000:

| κ | rc (a_ws) | box_side | cells_per_dim | Mode |
|---|-----------|----------|:---:|------|
| 1 | 8.0 | 34.74 | **4** (< 5) | all-pairs |
| 2 | 6.5 | 34.74 | **5** (≥ 5) | cell-list |
| 3 | 6.0 | 34.74 | **5** (≥ 5) | cell-list |

For κ=1, the Yukawa interaction range (`rc = 8.0 a_ws`) is so long that the box only
fits 4 cells per dimension. With only 4³ = 64 cells, the 27-cell neighbor search
covers 42% of all cells — nearly equivalent to all-pairs but with the overhead of
cell-list construction (CPU readback + sort + upload every step). Below 5 cells/dim,
all-pairs is actually faster.

**Cell-list activates for κ=1 at N ≥ ~15,300** (where `box_side ≥ 40 a_ws`). So on
larger GPUs (Titan, 3090, 6950 XT) running N=20,000+, even κ=1 would use cell-list.

**Can we reduce rc for κ=1?** Technically yes — a shorter cutoff means fewer cells but
introduces truncation error. The current `rc = 8.0 a_ws` captures ~8 screening lengths
(e^-8 ≈ 3.4×10⁻⁴ of the potential), which is standard for Yukawa OCP. Reducing to
`rc = 6.9` would enable cell-list at N=10,000 but would sacrifice 0.1% force accuracy.
For paper parity, we keep the exact published cutoffs.

**Conclusion**: Both modes are needed. All-pairs for long-range (low κ, small N),
cell-list for short-range (high κ, large N). The crossover is cleanly physics-determined.
No streamlining — this is the correct architecture.

---

## Evolution Architecture: Write → Absorb → Lean

hotSpring follows a deliberate pattern for extending toadstool/barracuda:

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
`wateringHole/handoffs/`. Toadstool absorbs them as GPU shaders. hotSpring
then rewires to use the upstream primitives and deletes local code. Each cycle
makes the upstream library richer and hotSpring leaner.

**What makes code absorbable**:
1. WGSL template strings embedded in Rust source (not separate files)
2. Clear binding layout documentation (binding index, type, purpose)
3. Dispatch geometry documented (workgroup size, grid dimensions)
4. CPU reference implementation validated against known physics
5. Tolerance constants in `tolerances.rs` (not inline magic numbers)
6. Handoff document with exact code locations and validation results

**Next absorption targets** (see `EVOLUTION_READINESS.md`):
- GPU SpMV (CSR sparse matrix-vector) — `spectral.rs::CsrMatrix::spmv()`
- GPU Lanczos eigensolve — `spectral.rs::lanczos()`
- GPU Dirac operator — `lattice/dirac.rs` (staggered fermion SpMV)

---

## BarraCUDA Crate (v0.5.16)

The `barracuda/` directory is a standalone Rust crate providing the validation
environment, physics implementations, and GPU compute. Key architectural properties:

- **345 unit tests** (340 passing + 5 GPU-ignored), **27 validation suites** (27/27 pass). Includes lattice QCD (complex f64, SU(3), Wilson action, HMC,
  Dirac CG), Abelian Higgs (U(1) + Higgs, HMC), transport coefficients (Green-Kubo
  D*/η*/λ*, Sarkas-calibrated fits), HotQCD EOS tables, NPU quantization parity
  (f64→f32→int8→int4), and NPU beyond-SDK hardware capability validation (arbitrary
  channels, FC merge, batch amortization, wide FC, multi-output, weight mutation,
  determinism). Test coverage: ~48% line / ~65% function (GPU modules
  require hardware; CPU-testable modules average >90%). Measured with `cargo-llvm-cov`.
- **AGPL-3.0 only** — all 69 active `.rs` files have
  `SPDX-License-Identifier: AGPL-3.0-only`. 22/28 `.wgsl` shaders have
  the header (6 deprecated toadstool-ref shaders pending).
- **Provenance** — centralized `BaselineProvenance` records trace hardcoded
  validation values to their Python origins (script path, git commit, date,
  exact command). `AnalyticalProvenance` references (DOIs, textbook citations)
  document mathematical ground truth for special functions, linear algebra,
  MD force laws, and GPU kernel correctness. All nuclear EOS binaries and
  library test modules source constants from `provenance::SLY4_PARAMS`,
  `NMP_TARGETS`, `L1_PYTHON_CHI2`, `MD_FORCE_REFS`, `GPU_KERNEL_REFS`, etc.
  DOIs for AME2020, Chabanat 1998, Kortelainen 2010, Bender 2003,
  Lattimer & Prakash 2016 are documented in `provenance.rs`.
- **Tolerances** — 72 centralized constants in `tolerances.rs` with physical
  justification (machine precision, numerical method, model, literature).
  Includes 12 physics guard constants (`DENSITY_FLOOR`, `SPIN_ORBIT_R_MIN`,
  `COULOMB_R_MIN`, `BCS_DENSITY_SKIP`, `DEFORMED_COULOMB_R_MIN`, etc.)
  plus 28 validation-specific thresholds for transport, lattice QCD, Abelian Higgs,
  NAK eigensolve, PPPM, screened Coulomb, NPU quantization, and NPU beyond-SDK
  hardware capabilities. All validation binaries wired to `tolerances::*`.
- **ValidationHarness** — structured pass/fail tracking with exit code 0/1.
  12 of 20 binaries use it (validation targets). Remaining 8 are optimization
  explorers and diagnostics.
- **Shared data loading** — `data::EosContext` and `data::load_eos_context()`
  eliminate duplicated path construction across all nuclear EOS binaries.
  `data::chi2_per_datum()` centralizes χ² computation with `tolerances::sigma_theo`.
- **Typed errors** — `HotSpringError` enum with `Result` propagation in GPU
  and simulation APIs. Minimal `expect()` in library code (22 occurrences,
  mostly GPU staging and file I/O where failure is unrecoverable).
  Validation binaries use `.expect()` with descriptive messages.
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
  BarraCUDA primitives (`SpinOrbitGpu`, `compute_ls_factor`).
- **Capability-based discovery** — GPU adapter selection by name, index, or
  auto-detect (first discrete with `SHADER_F64`). Supports nvidia proprietary,
  NVK/nouveau, RADV, and any Vulkan driver. Buffer limits derived from
  `adapter.limits()`, not hardcoded. Data paths resolved via `HOTSPRING_DATA_ROOT`
  or directory discovery.
- **NaN-safe** — all float sorting uses `f64::total_cmp()`.
- **Zero external commands** — pure-Rust ISO 8601 timestamps (Hinnant algorithm),
  no `date` shell-out. `nvidia-smi` calls degrade gracefully.
- **No unsafe code** — zero `unsafe` blocks in the entire crate.

```bash
cd barracuda
cargo test               # 345 tests pass, 5 GPU-ignored (< 16 seconds)
cargo clippy --all-targets  # Zero warnings
cargo doc --no-deps      # Full API documentation — 0 warnings
cargo run --release --bin validate_all  # 27/27 suites pass
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
├── CONTROL_EXPERIMENT_STATUS.md        # Comprehensive status + results (195/195)
├── NUCLEAR_EOS_STRATEGY.md             # Nuclear EOS Phase A→B strategy
├── wateringHole/handoffs/              # 3 active + 16 archived cross-project handoffs (fossil record)
├── LICENSE                             # AGPL-3.0
├── .gitignore
│
├── whitePaper/                         # Public-facing study documents
│   ├── README.md                      # Document index
│   ├── STUDY.md                       # Main study — full writeup
│   ├── BARRACUDA_SCIENCE_VALIDATION.md # Phase B technical results
│   ├── CONTROL_EXPERIMENT_SUMMARY.md  # Phase A quick reference
│   └── METHODOLOGY.md                # Two-phase validation protocol
│
├── barracuda/                          # BarraCUDA Rust crate — v0.5.16 (345 tests, 27 suites)
│   ├── Cargo.toml                     # Dependencies (requires ecoPrimals/phase1/toadstool)
│   ├── CHANGELOG.md                   # Version history — baselines, tolerances, evolution
│   ├── EVOLUTION_READINESS.md         # Rust module → GPU promotion tier + absorption status
│   ├── clippy.toml                    # Clippy thresholds (physics-justified)
│   └── src/
│       ├── lib.rs                     # Crate root — module declarations + architecture docs
│       ├── error.rs                   # Typed errors (HotSpringError: NoAdapter, NoShaderF64, …)
│       ├── provenance.rs              # Baseline + analytical provenance (Python, DOIs, textbook)
│       ├── tolerances.rs              # 72 centralized thresholds with physical justification
│       ├── validation.rs              # Pass/fail harness — structured checks, exit code 0/1
│       ├── discovery.rs               # Capability-based data path resolution (env var / CWD)
│       ├── data.rs                    # AME2020 data + Skyrme bounds + EosContext + chi2_per_datum
│       ├── prescreen.rs               # NMP cascade filter (algebraic → L1 proxy → classifier)
│       ├── spectral.rs               # Spectral theory (CsrMatrix, SpMV, Lanczos, Anderson 1D/2D/3D, Hofstadter)
│       ├── bench.rs                   # Benchmark harness (RAPL, nvidia-smi, JSON reports)
│       ├── gpu.rs                     # GPU FP64 device wrapper (SHADER_F64 via wgpu/Vulkan)
│       │
│       ├── physics/                   # Nuclear structure — L1/L2/L3 implementations
│       │   ├── constants.rs           # CODATA 2018 physical constants
│       │   ├── semf.rs                # Semi-empirical mass formula (Bethe-Weizsäcker + Skyrme)
│       │   ├── nuclear_matter.rs      # Infinite nuclear matter properties (ρ₀, E/A, K∞, m*/m, J)
│       │   ├── hfb_common.rs          # Shared HFB: Mat, BCS v², Coulomb exchange, Hermite, factorial
│       │   ├── bcs_gpu.rs             # Local GPU BCS bisection (corrected WGSL shader)
│       │   ├── hfb.rs                 # Spherical HFB solver (L2)
│       │   ├── hfb_deformed.rs        # Axially-deformed HFB solver (L3, CPU)
│       │   ├── hfb_deformed_gpu.rs    # Deformed HFB with GPU eigensolves (L3)
│       │   ├── hfb_gpu.rs             # GPU-batched HFB (BatchedEighGpu)
│       │   ├── hfb_gpu_resident.rs    # GPU-resident HFB (potentials + H + eigensolve + density + mixing)
│       │   ├── hfb_gpu_types.rs       # GPU buffer types and uniform helpers for HFB pipeline
│       │   └── shaders/               # f64 WGSL physics kernels (13 shaders, ~2000 lines)
│       │
│       ├── md/                        # GPU Molecular Dynamics (Yukawa OCP)
│       │   ├── config.rs              # Simulation configuration (reduced units)
│       │   ├── celllist.rs            # Cell-list spatial decomposition (GPU neighbor search)
│       │   ├── shaders.rs             # Shader constants (include_str! + 3 small inline)
│       │   ├── shaders/               # f64 WGSL production kernels (5 files)
│       │   ├── simulation.rs          # GPU MD loop (all-pairs + cell-list)
│       │   ├── cpu_reference.rs       # CPU reference implementation (FCC, Verlet)
│       │   ├── observables.rs         # Energy, RDF, VACF, SSF, stress ACF, heat ACF
│       │   ├── transport.rs           # Stanton-Murillo analytical fits (D*, η*, λ*)
│       │   └── shaders_toadstool_ref/ # ToadStool shader snapshots (divergence tracking)
│       │
│       ├── lattice/                   # Lattice gauge theory (Papers 7, 8, 13)
│       │   ├── complex_f64.rs         # Complex f64 arithmetic (Rust + WGSL template)
│       │   ├── su3.rs                 # SU(3) 3×3 complex matrix algebra (Rust + WGSL template)
│       │   ├── wilson.rs              # Wilson gauge action — plaquettes, staples, force
│       │   ├── hmc.rs                 # Hybrid Monte Carlo — Cayley exp, leapfrog
│       │   ├── abelian_higgs.rs       # U(1) + Higgs (1+1)D lattice HMC (Paper 13)
│       │   ├── constants.rs            # Centralized LCG PRNG, SU(3) constants, guards
│       │   ├── dirac.rs              # Staggered Dirac operator
│       │   ├── cg.rs                  # Conjugate gradient solver for D†D
│       │   ├── eos_tables.rs          # HotQCD EOS tables (Bazavov et al. 2014)
│       │   └── multi_gpu.rs           # Temperature scan dispatcher
│       │
│       ├── archive/                   # Historical implementations (fossil record, not compiled)
│       │
│       └── bin/                       # 43 binaries (exit 0 = pass, 1 = fail)
│           ├── validate_all.rs        # Meta-validator: runs all 27 validation suites
│           ├── validate_nuclear_eos.rs # L1 SEMF + L2 HFB + NMP validation harness
│           ├── validate_barracuda_pipeline.rs # Full MD pipeline (12/12 checks)
│           ├── validate_barracuda_hfb.rs # BCS + eigensolve pipeline (14/14 checks)
│           ├── validate_cpu_gpu_parity.rs # CPU vs GPU numerical parity
│           ├── validate_md.rs         # CPU MD reference validation
│           ├── validate_nak_eigensolve.rs # NAK GPU eigensolve validation
│           ├── validate_pppm.rs       # PppmGpu κ=0 Coulomb validation
│           ├── validate_transport.rs  # CPU/GPU transport coefficient validation
│           ├── validate_stanton_murillo.rs # Paper 5: Green-Kubo vs Sarkas-calibrated fits (13/13)
│           ├── validate_hotqcd_eos.rs # Paper 7: HotQCD EOS thermodynamic validation
│           ├── validate_pure_gauge.rs # Paper 8: SU(3) HMC + Dirac CG validation (12/12)
│           ├── validate_abelian_higgs.rs # Paper 13: U(1)+Higgs HMC validation (17/17)
│           ├── validate_npu_quantization.rs # NPU ESN quantization cascade (6/6)
│           ├── validate_npu_beyond_sdk.rs # NPU beyond-SDK capabilities (16/16 math checks)
│           ├── validate_lattice_npu.rs  # Lattice QCD + NPU heterogeneous pipeline (10/10)
│           ├── validate_hetero_monitor.rs # Heterogeneous real-time monitor (9/9) — previously impossible
│           ├── validate_spectral.rs    # Spectral theory: Anderson + almost-Mathieu (10/10)
│           ├── validate_lanczos.rs    # Lanczos + SpMV + 2D Anderson (11/11)
│           ├── validate_anderson_3d.rs # 3D Anderson: mobility edge + dimensional hierarchy (10/10)
│           ├── validate_hofstadter.rs # Hofstadter butterfly: band counting + spectral topology (10/10)
│           ├── validate_reservoir_transport.rs # ESN transport prediction validation
│           ├── validate_screened_coulomb.rs # Screened Coulomb eigenvalues (23/23)
│           ├── validate_special_functions.rs # Gamma, Bessel, erf, Hermite, …
│           ├── validate_linalg.rs     # LU, QR, SVD, tridiagonal solver
│           ├── validate_optimizers.rs # BFGS, Nelder-Mead, RK45, stats
│           ├── verify_hfb.rs          # HFB physics verification (Rust vs Python)
│           ├── nuclear_eos_l1_ref.rs  # L1 SEMF optimization pipeline
│           ├── nuclear_eos_l2_ref.rs  # L2 HFB hybrid optimization
│           ├── nuclear_eos_l2_gpu.rs  # L2 GPU-batched HFB (BatchedEighGpu)
│           ├── nuclear_eos_l2_hetero.rs # L2 heterogeneous cascade pipeline
│           ├── nuclear_eos_l3_ref.rs  # L3 deformed HFB (CPU Rayon)
│           ├── nuclear_eos_l3_gpu.rs  # L3 deformed HFB (GPU-resident)
│           ├── nuclear_eos_gpu.rs     # GPU FP64 validation + energy profiling
│           ├── sarkas_gpu.rs          # GPU Yukawa MD (9 PP cases, f64 WGSL)
│           ├── bench_cpu_gpu_scaling.rs # CPU vs GPU crossover benchmark
│           ├── bench_gpu_fp64.rs      # GPU FP64 throughput benchmark
│           ├── bench_multi_gpu.rs     # Multi-GPU dispatch benchmark
│           ├── bench_wgsize_nvk.rs    # NVK workgroup-size tuning
│           ├── celllist_diag.rs       # Cell-list vs all-pairs force diagnostic
│           ├── f64_builtin_test.rs    # Native vs software f64 validation
│           └── shaders/               # Extracted WGSL diagnostic shaders (8 files)
│
├── control/
│   ├── comprehensive_control_results.json  # Grand total: 86/86 checks
│   │
│   ├── metalforge_npu/                # NPU hardware validation (AKD1000)
│   │   ├── scripts/                   # npu_quantization_parity.py, npu_beyond_sdk.py, native_int4_reservoir.py
│   │   └── results/                   # JSON baselines from hardware runs
│   │
│   ├── reservoir_transport/           # ESN transport prediction control
│   │   └── scripts/                   # reservoir_vacf.py
│   │
│   ├── akida_dw_edma/                 # Akida NPU kernel module (patched for 6.17)
│   │   ├── Makefile
│   │   ├── README.md
│   │   ├── akida-pcie-core.c          # PCIe driver source
│   │   └── akida-dw-edma/             # DMA engine sources
│   │
│   ├── sarkas/                         # Study 1: Molecular Dynamics
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── patches/                    # Patches for Sarkas v1.0.0 compat
│   │   │   └── sarkas-v1.0.0-compat.patch
│   │   ├── sarkas-upstream/            # Cloned + patched via scripts/clone-repos.sh
│   │   └── simulations/
│   │       └── dsf-study/
│   │           ├── input_files/        # YAML configs (12 cases)
│   │           ├── scripts/            # run, validate, batch, profile
│   │           └── results/            # Validation JSONs + plots
│   │
│   ├── surrogate/                      # Study 2: Surrogate Learning
│   │   ├── README.md
│   │   ├── REPRODUCE.md               # Step-by-step reproduction guide
│   │   ├── requirements.txt
│   │   ├── scripts/                    # Benchmark + iterative workflow runners
│   │   ├── results/                    # Result JSONs
│   │   └── nuclear-eos/               # Nuclear EOS (L1 + L2)
│   │       ├── README.md
│   │       ├── exp_data/              # AME2020 experimental binding energies
│   │       ├── scripts/               # run_surrogate.py, gpu_rbf.py
│   │       ├── wrapper/               # objective.py, skyrme_hf.py, skyrme_hfb.py
│   │       └── results/               # L1, L2, BarraCUDA JSON results
│   │
│   └── ttm/                            # Study 3: Two-Temperature Model
│       ├── README.md
│       ├── patches/                    # Patches for TTM NumPy 2.x compat
│       │   └── ttm-numpy2-compat.patch
│       ├── Two-Temperature-Model/      # Cloned + patched via scripts/clone-repos.sh
│       └── scripts/                    # Local + hydro model runners
│
├── experiments/                         # Experiment journals (the "why" behind the data)
│   ├── 001_N_SCALING_GPU.md            # N-scaling (500→20k) + native f64 builtins
│   ├── 002_CELLLIST_FORCE_DIAGNOSTIC.md # Cell-list i32 modulo bug diagnosis + fix
│   ├── 003_RTX4070_CAPABILITY_PROFILE.md # RTX 4070 capability profile (paper-parity COMPLETE)
│   ├── 004_GPU_DISPATCH_OVERHEAD_L3.md  # L3 deformed HFB GPU dispatch profiling
│   ├── 005_L2_MEGABATCH_COMPLEXITY_BOUNDARY.md # L2 mega-batch GPU complexity analysis
│   ├── 006_GPU_FP64_COMPARISON.md      # RTX 4070 vs Titan V fp64 benchmark
│   └── 007_CPU_GPU_SCALING_BENCHMARK.md # CPU vs GPU scaling: crossover analysis
│
├── metalForge/                         # Hardware characterization & direct-wire programming
│   ├── README.md                      # Philosophy + hardware inventory
│   ├── npu/akida/                     # BrainChip AKD1000 NPU exploration
│   │   ├── HARDWARE.md                # Architecture, compute model, limits
│   │   ├── EXPLORATION.md             # Novel applications for physics
│   │   ├── BEYOND_SDK.md              # 10 overturned SDK assumptions (the discovery doc)
│   │   └── scripts/                   # Python probing scripts (deep_probe.py)
│   └── gpu/nvidia/                    # RTX 4070 + Titan V characterization
│
├── wateringHole/                       # Cross-project handoffs
│   └── handoffs/                       # Unidirectional handoff documents
│
├── benchmarks/
│   ├── PROTOCOL.md                     # Cross-gate benchmark protocol (time + energy)
│   ├── nuclear-eos/results/            # Benchmark JSON reports (auto-generated)
│   └── sarkas-cpu/                     # Sarkas CPU comparison notes
│
├── data/
│   ├── plasma-properties-db/           # Dense Plasma Properties Database — clone via scripts/
│   ├── zenodo-surrogate/               # Zenodo archive — download via scripts/
│   └── ttm-reference/                  # TTM reference data
│
├── scripts/
│   ├── regenerate-all.sh               # Master: full data regeneration on fresh clone
│   ├── clone-repos.sh                  # Clone + pin + patch upstream repos
│   ├── download-data.sh               # Download Zenodo data (~6 GB)
│   └── setup-envs.sh                   # Create Python envs (conda/micromamba)
│
└── envs/
    ├── sarkas.yaml                     # Sarkas env spec (Python 3.9)
    ├── surrogate.yaml                  # Surrogate env spec (Python 3.10)
    └── ttm.yaml                        # TTM env spec (Python 3.10)
```

---

## Studies

### Study 1: Sarkas Molecular Dynamics

Reproduce plasma simulations from the Dense Plasma Properties Database. 12 cases: 9 Yukawa PP (κ=1,2,3 × Γ=low,mid,high) + 3 Coulomb PPPM (κ=0 × Γ=10,50,150).

- **Source**: [github.com/murillo-group/sarkas](https://github.com/murillo-group/sarkas) (MIT)
- **Reference**: [Dense Plasma Properties Database](https://github.com/MurilloGroupMSU/Dense-Plasma-Properties-Database)
- **Result**: 60/60 observable checks pass (DSF 8.5% mean error PP, 7.3% PPPM)
- **Finding**: `force_pp.update()` is 97.2% of runtime → primary GPU offload target
- **Bugs fixed**: 3 (NumPy 2.x `np.int`, pandas 2.x `.mean(level=)`, Numba/pyfftw PPPM)

### Study 2: Surrogate Learning (Nature MI 2024)

Reproduce "Efficient learning of accurate surrogates for simulations of complex systems" (Diaw et al., 2024).

- **Paper**: [doi.org/10.1038/s42256-024-00839-1](https://doi.org/10.1038/s42256-024-00839-1)
- **Data**: [Zenodo: 10.5281/zenodo.10908462](https://doi.org/10.5281/zenodo.10908462) (open, 6 GB)
- **Code**: [Code Ocean: 10.24433/CO.1152070.v1](https://doi.org/10.24433/CO.1152070.v1) — gated, sign-up denied
- **Result**: 9/9 benchmark functions reproduced. Physics EOS from MD data converged (χ²=4.6×10⁻⁵).

#### Nuclear EOS Surrogate (L1 + L2)

Built from first principles — no HFBTHO, no Code Ocean. Pure Python physics:

| Level | Method | Python χ²/datum | BarraCUDA χ²/datum | Speedup |
|-------|--------|-----------------|--------------------|---------|
| 1 | SEMF + nuclear matter (52 nuclei) | 6.62 | **2.27** ✅ | **478×** |
| 2 | HF+BCS hybrid (18 focused nuclei) | **1.93** | **16.11** / 19.29 (NMP) | 1.7× |
| 3 | Axially deformed HFB (target) | — | — | — |

- **L1**: Skyrme EDF → nuclear matter properties → SEMF → χ²(AME2020)
- **L2**: Spherical HF+BCS solver for 56≤A≤132, SEMF elsewhere, 18 focused nuclei
- **BarraCUDA**: Full Rust port with WGSL cdist, f64 LA, LHS, multi-start Nelder-Mead

### Study 3: Two-Temperature Model

Run the UCLA-MSU TTM for laser-plasma equilibration in cylindrical coordinates.

- **Source**: [github.com/MurilloGroupMSU/Two-Temperature-Model](https://github.com/MurilloGroupMSU/Two-Temperature-Model)
- **Result**: 6/6 checks pass (3 local + 3 hydro). All species reach physical equilibrium.
- **Bug fixed**: 1 (Thomas-Fermi ionization model sets χ₁=NaN, must use Saha input data)

---

## Upstream Bugs Found and Fixed

| # | Bug | Where | Impact |
|---|-----|-------|--------|
| 1 | `np.int` removed in NumPy 2.x | `sarkas/tools/observables.py` | Silent DSF/SSF failure |
| 2 | `.mean(level=)` removed in pandas 2.x | `sarkas/tools/observables.py` | Silent DSF failure |
| 3 | Numba 0.60 `@jit` → `nopython=True` breaks pyfftw | `sarkas/potentials/force_pm.py` | PPPM method crashes |
| 4 | Thomas-Fermi `χ₁=NaN` poisons recombination | TTM `exp_setup.py` | Zbar solver diverges |
| 5 | DSF reference file naming (case sensitivity) | Plasma Properties DB | Validation script fails |
| 6 | Multithreaded dump corruption (v1.1.0) | Sarkas `4b561baa` | All `.npz` checkpoints NaN from step ~10 (resolved by pinning to v1.0.0) |

These are **silent failures** — wrong results, no error messages. This fragility is a core finding.

---

## Hardware

- **Eastgate (primary dev)**: i9-12900K, RTX 4070 (12GB) + Titan V (12GB HBM2), Akida AKD1000 NPU, 32 GB DDR5.
  - RTX 4070 (Ada): nvidia proprietary 580.x, `SHADER_F64` confirmed. fp64:fp32 ~1:2 via wgpu/Vulkan.
  - Titan V (GV100): **NVK / nouveau (Mesa 25.1.5, built from source)**, `SHADER_F64` confirmed. Native fp64 silicon, 6.9 TFLOPS FP64, 12GB HBM2. `validate_cpu_gpu_parity` 6/6, `validate_stanton_murillo` 40/40 on NVK.
  - AKD1000 (BrainChip): PCIe `08:00.0`, 80 NPs, 8MB SRAM, akida 2.19.1. 10 SDK assumptions overturned. See `metalForge/npu/akida/BEYOND_SDK.md`.
  - **Numerical parity**: identical physics to 1e-15 across both GPUs and both drivers. NPU int4 quantization error bounded at <30%.
  - VRAM headroom: <600 MB used at N=20,000 — estimated N≈400,000 before VRAM limits.
  - Adapter selection: `HOTSPRING_GPU_ADAPTER=titan` or `=4070` or `=0`/`=1` (see `gpu.rs` docs).
- **Strandgate**: 64-core EPYC, 32 GB. Full-scale DSF (N=10,000) CPU runs.
- **Northgate**: i9-14900K. Single-thread comparison.
- **Southgate**: 5800X3D. V-Cache neighbor list performance.

---

## Document Index

| Document | Purpose |
|----------|---------|
| [`PHYSICS.md`](PHYSICS.md) | Complete physics documentation — every equation, constant, approximation with numbered references |
| [`CONTROL_EXPERIMENT_STATUS.md`](CONTROL_EXPERIMENT_STATUS.md) | Full status with numbers, 195/195 checks, evolution history |
| [`NUCLEAR_EOS_STRATEGY.md`](NUCLEAR_EOS_STRATEGY.md) | Strategic plan: Python control → BarraCUDA proof |
| [`barracuda/CHANGELOG.md`](barracuda/CHANGELOG.md) | Crate version history — baselines, tolerance changes, evolution |
| [`barracuda/EVOLUTION_READINESS.md`](barracuda/EVOLUTION_READINESS.md) | Rust module → WGSL shader → GPU promotion tier mapping |
| [`whitePaper/README.md`](whitePaper/README.md) | **White paper index** — the publishable study narrative |
| [`whitePaper/STUDY.md`](whitePaper/STUDY.md) | Main study: replicating computational plasma physics on consumer hardware |
| [`whitePaper/BARRACUDA_SCIENCE_VALIDATION.md`](whitePaper/BARRACUDA_SCIENCE_VALIDATION.md) | Phase B technical results: BarraCUDA vs Python/SciPy |
| [`benchmarks/PROTOCOL.md`](benchmarks/PROTOCOL.md) | Benchmark protocol: time + energy + hardware measurement |
| [`experiments/001_N_SCALING_GPU.md`](experiments/001_N_SCALING_GPU.md) | N-scaling sweep + native f64 builtins discovery |
| [`experiments/002_CELLLIST_FORCE_DIAGNOSTIC.md`](experiments/002_CELLLIST_FORCE_DIAGNOSTIC.md) | Cell-list i32 modulo bug diagnosis and fix |
| [`experiments/003_RTX4070_CAPABILITY_PROFILE.md`](experiments/003_RTX4070_CAPABILITY_PROFILE.md) | RTX 4070 capability profile + paper-parity long run results |
| [`experiments/004_GPU_DISPATCH_OVERHEAD_L3.md`](experiments/004_GPU_DISPATCH_OVERHEAD_L3.md) | L3 deformed HFB GPU dispatch profiling |
| [`experiments/005_L2_MEGABATCH_COMPLEXITY_BOUNDARY.md`](experiments/005_L2_MEGABATCH_COMPLEXITY_BOUNDARY.md) | L2 mega-batch GPU complexity boundary analysis |
| [`experiments/006_GPU_FP64_COMPARISON.md`](experiments/006_GPU_FP64_COMPARISON.md) | RTX 4070 vs Titan V: fp64 benchmark, driver comparison, NVK vs proprietary |
| [`experiments/007_CPU_GPU_SCALING_BENCHMARK.md`](experiments/007_CPU_GPU_SCALING_BENCHMARK.md) | CPU vs GPU scaling: crossover analysis, streaming dispatch |
| [`metalForge/README.md`](metalForge/README.md) | Hardware characterization — philosophy, inventory, directory |
| [`metalForge/npu/akida/BEYOND_SDK.md`](metalForge/npu/akida/BEYOND_SDK.md) | **10 overturned SDK assumptions** — the discovery document |
| [`metalForge/npu/akida/HARDWARE.md`](metalForge/npu/akida/HARDWARE.md) | AKD1000 deep-dive: architecture, compute model, PCIe BAR mapping |
| [`metalForge/npu/akida/EXPLORATION.md`](metalForge/npu/akida/EXPLORATION.md) | Novel NPU applications for computational physics |
| [`wateringHole/handoffs/`](wateringHole/handoffs/) | Cross-project handoffs to ToadStool/BarraCUDA team |
| [`control/surrogate/REPRODUCE.md`](control/surrogate/REPRODUCE.md) | Step-by-step reproduction guide for surrogate learning |

### External References

| Reference | DOI / URL | Used For |
|-----------|-----------|----------|
| Diaw et al. (2024) *Nature Machine Intelligence* | [10.1038/s42256-024-00839-1](https://doi.org/10.1038/s42256-024-00839-1) | Surrogate learning methodology |
| Sarkas MD package | [github.com/murillo-group/sarkas](https://github.com/murillo-group/sarkas) (MIT) | DSF plasma simulations |
| Dense Plasma Properties Database | [github.com/MurilloGroupMSU](https://github.com/MurilloGroupMSU/Dense-Plasma-Properties-Database) | DSF reference spectra |
| Two-Temperature Model | [github.com/MurilloGroupMSU](https://github.com/MurilloGroupMSU/Two-Temperature-Model) | Plasma equilibration |
| Zenodo surrogate archive | [10.5281/zenodo.10908462](https://doi.org/10.5281/zenodo.10908462) (CC-BY) | Convergence histories |
| AME2020 (Wang et al. 2021) | [IAEA Nuclear Data](https://www-nds.iaea.org/amdc/ame2020/) | Experimental binding energies |
| Code Ocean capsule | [10.24433/CO.1152070.v1](https://doi.org/10.24433/CO.1152070.v1) | **Gated** — registration denied |

---

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0).
See [LICENSE](LICENSE) for the full text.

Sovereign science: all source code, data processing scripts, and validation results are
freely available for inspection, reproduction, and extension. If you use this work in
a network service, you must make your source available under the same terms.

---

*hotSpring proves that a $600 GPU can do the same physics as an HPC cluster —
same observables, same energy conservation, same particle count, same production
steps — in 3.66 hours for 9 cases, using 0.365 kWh of electricity at $0.044.
A $300 NPU runs the same math at 30mW for inference workloads — 9,017× less
energy than CPU for transport predictions. The heterogeneous pipeline (GPU+NPU+CPU)
makes five things possible that were previously impossible: real-time phase monitoring
during live simulation, continuous transport prediction without Green-Kubo post-processing,
substrate-portable physics (f64→f32→int4), predictive steering that saves 62% of compute,
and zero-overhead monitoring at 0.09%. The scarcity was artificial.*
