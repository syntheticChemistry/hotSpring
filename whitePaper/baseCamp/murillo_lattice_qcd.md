# Lattice QCD — Quenched & Dynamical Fermions

**Papers:** 7-12 (HotQCD EOS, Pure Gauge, Production QCD, Dynamical Fermions, HVP g-2, Freeze-Out)
**Updated:** March 8, 2026
**Status:** ✅ Production QCD on consumer GPU, deconfinement at β_c=5.69, asymmetric 64³×8, N_f=4 infrastructure complete

**Chuna-specific papers** (43, 44, 45) have dedicated baseCamp documents:
- [`chuna_gradient_flow.md`](chuna_gradient_flow.md) — Paper 43 (11/11 core; dynamical ext 1/3)
- [`chuna_bgk_dielectric.md`](chuna_bgk_dielectric.md) — Paper 44 (20/20 checks)
- [`chuna_kinetic_fluid.md`](chuna_kinetic_fluid.md) — Paper 45 (10/10 checks)

---

## What We Reproduced

Lattice QCD is the numerical method for computing the strong nuclear force from
first principles. Our reproduction starts from the HotQCD equation of state
(Bazavov et al. 2014) and builds up to full dynamical fermion HMC.

| Study | Paper | Checks | Result |
|-------|-------|--------|--------|
| HotQCD EOS Tables | Paper 7 | Pass | Thermodynamic consistency, asymptotic freedom |
| Pure Gauge SU(3) | Paper 8 | 16/16 | HMC, Dirac CG, plaquette physics + sovereign GPU compile (SM35/SM70/SM120) |
| Production QCD (quenched) | Paper 9 | 12/12 | 32⁴ on RTX 3090 (13.6h, $0.58), β_c=5.69 |
| Dynamical Fermion HMC | Paper 10 | 7/7 | Pseudofermion HMC: ΔH scaling, plaquette, acceptance |
| Abelian Higgs | Paper 13 | 17/17 | U(1)+Higgs (1+1)D, Rust 143× faster than Python |

## Evolution Path

1. **CPU lattice** (Paper 8): Complex f64, SU(3) 3×3 matrices, Wilson gauge action, HMC with Cayley exponential map. Staggered Dirac operator and CG solver for D†D.

2. **GPU lattice** (v0.6.0-v0.6.8): 24 WGSL shaders covering all lattice QCD operations. GPU Dirac (8/8), GPU CG (9/9), pure GPU QCD workload (3/3). Rust CG is 200× faster than Python.

3. **GPU streaming** (v0.6.9): Streaming HMC eliminates CPU round-trips. GPU-resident CG reduces readback by 15,360×. Bidirectional streaming for dynamical fermions.

4. **DF64 core streaming** (v0.6.10-v0.6.11): Double-float arithmetic on FP32 cores delivers 3.24 TFLOPS at 14-digit precision — 9.9× native f64. Site-indexing standardized to toadStool t-major convention.

5. **DF64 expansion** (v0.6.12): toadStool S60 absorption. DF64 plaquette, kinetic energy, transcendentals. 60% of HMC in DF64 (up from 40%). 8-12% additional speedup.

6. **Cross-spring rewiring** (v0.6.13): GPU Polyakov loop (72× less data transfer). NVK allocation guard. PRNG fix. 13/13 validation checks.

## Production Results (Experiment 013)

**RTX 3090, 32⁴ lattice, 12-point β-scan:**
- 200 measurements per point, 50 HMC thermalization
- Wall time: 13.6 hours ($0.58 electricity)
- **Deconfinement transition: χ=40.1 at β=5.69** — matches known β_c=5.692
- Finite-size scaling confirmed: 16⁴ vs 32⁴ shows expected volume dependence

**DF64 unleashed (Experiment 014):**
- 32⁴ at 7.7s/trajectory (2× faster than v0.6.8)
- Same physics quality, half the time, lower power draw

## Key Finding

A lone scientist in a basement, using a $1,500 consumer GPU (RTX 3090) and
open-source Rust/WGSL code, reproduces the deconfinement phase transition of
quantum chromodynamics — the same physics that historically required
million-dollar supercomputers. The DF64 hybrid strategy delivers 9.9× the
throughput of native f64 on the same silicon.

## Cross-Spring Contributions

- **Complex64 WGSL** → toadStool `complex_f64.wgsl` → used everywhere
- **SU(3) math** → toadStool `su3_f64.wgsl` / `su3_math_f64.wgsl` → hotSpring-native, toadStool-absorbed
- **DF64 core streaming discovery** → toadStool `df64_core.wgsl` → used by all springs needing precision
- **GPU Polyakov loop** → toadStool `polyakov_loop_f64.wgsl` → bidirectional evolution: toadStool → hotSpring → toadStool
- **NVK allocation guard** → toadStool `driver_profile.rs` → protects all nouveau users

## Asymmetric Lattices & Finite Temperature (Exp 032, March 6 2026)

Moved from hypercubic (L⁴) to physically relevant asymmetric (N_s³ × N_t) geometries
for finite-temperature QCD. N_t sets the temperature: T = 1/(a × N_t).

| Lattice | Sites | GPU ms/traj | Speedup | Status |
|---------|:-----:|:-----------:|:-------:|--------|
| 32³×8 | 262K | 2,840 | 26.5× | ✅ Complete — 1,800 traj, 3.5h |
| 64³×8 | 2.10M | ~23,000 | ~26× | ✅ Production run (overnight) |

32³×8 shows susceptibility peak near β≈5.9 — crossover behavior as expected
for N_s/N_t = 4. 64³×8 (N_s/N_t = 8, MILC-comparable) should resolve sharp
first-order transition at β_c(N_t=8) ≈ 6.062.

## Chuna Papers (43-45) — See Dedicated Documents

The three Chuna/Murillo papers are now covered in their own baseCamp artifacts:

| Paper | Document | Checks |
|-------|----------|:------:|
| 43 — Gradient Flow | [`chuna_gradient_flow.md`](chuna_gradient_flow.md) | 11/11 core (ext 3/3) |
| 44 — BGK Dielectric | [`chuna_bgk_dielectric.md`](chuna_bgk_dielectric.md) | 20/20 |
| 45 — Kinetic-Fluid | [`chuna_kinetic_fluid.md`](chuna_kinetic_fluid.md) | 10/10 |

Total: **44/44 checks pass** via `validate_chuna_overnight` (core 41/41; dynamical ext 3/3).

## Science Ladder

| Level | Physics | Status |
|-------|---------|--------|
| 0 | Quenched HMC (32⁴, 32³×8, 64³×8) | ✅ Complete |
| 1 | Gradient flow (t₀, w₀, 5 integrators) | ✅ Complete — see [`chuna_gradient_flow.md`](chuna_gradient_flow.md) |
| 2 | Flow integrator convergence (Chuna paper) | ✅ Validated |
| 2b | Dynamical N_f=4 staggered gradient flow | ✅ Complete |
| 3 | N_f=4 staggered dynamical (GPU CG) | ✅ Infrastructure complete |
| 3b | Chuna Papers 43-45 core parity (41/41) | ✅ **Complete** (dynamical ext 1/3, in progress) |
| 4 | N_f=2 dynamical (RHMC rooting trick) | ✅ **Production complete** — Exp 101: 4^4 (78% accept) + 8^4 (50% accept) |
| 5 | N_f=2+1 staggered (strange quark mass) | ✅ **Production complete** — Exp 101: 4^4, 2-sector, correct ordering Q < 2+1 < 2 |
| 5b | Gradient flow convergence at volume | 🔄 Exp 102: CK4 stability confirmed, 16^4 t₀/w₀ in progress |
| 5c | Gradient flow on dynamical (Nf=2+1) configs | ✅ **Exp 103** — `production_rhmc_flow` binary, E(t) measured on 8^4 RHMC configs |
| 6 | N_f=2+1+1 HISQ (full physical QCD) | Long-term |

## What's Next (March 26, 2026)

- ✅ **GPU RHMC production**: Nf=2 at 4^4 and 8^4, Nf=2+1 at 4^4 — Exp 101 complete
- 🔄 **Gradient flow at volume**: t₀/w₀ on 16⁴ — Exp 102 in progress (16^4 quenched flow running)
- 🔄 **Flow convergence**: CK4 stability at ε=0.1 confirmed (error 2.3e-6 vs 1.6 for W6/W7)
- ✅ **Gradient flow on RHMC configs** (Exp 103): W7 on Nf=2 and Nf=2+1 at 8^4, E(t) measured
- **Scale to 32⁴**: RHMC + flow overnight runs for production physics
- **64³×8 analysis**: Polyakov loop jump at β_c, finite-size scaling (32³/48³/64³ × 8)
- **Silicon characterization on HBM2 fleet**: Titan V, MI50, Tesla P80 (Exp 100)
- **Meet/exceed Chuna scale**: Volume roadmap in `silicon_characterization_at_scale.md`
