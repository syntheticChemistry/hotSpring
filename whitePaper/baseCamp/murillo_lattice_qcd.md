# Lattice QCD — Quenched & Dynamical Fermions

**Papers:** 7-12 (HotQCD EOS, Pure Gauge, Production QCD, Dynamical Fermions, HVP g-2, Freeze-Out)
**Updated:** February 25, 2026
**Status:** ✅ Production 32⁴ quenched β-scan complete, deconfinement transition observed

---

## What We Reproduced

Lattice QCD is the numerical method for computing the strong nuclear force from
first principles. Our reproduction starts from the HotQCD equation of state
(Bazavov et al. 2014) and builds up to full dynamical fermion HMC.

| Study | Paper | Checks | Result |
|-------|-------|--------|--------|
| HotQCD EOS Tables | Paper 7 | Pass | Thermodynamic consistency, asymptotic freedom |
| Pure Gauge SU(3) | Paper 8 | 12/12 | HMC, Dirac CG, plaquette physics |
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

## What's Next

- **Dynamical QCD at scale**: 16⁴-32⁴ with full pseudofermion HMC (light quarks)
- **Mixed pipeline**: NPU-driven adaptive β selection (Experiment 015 Phase 2+)
- **Titan V validation**: NVK fp64 parity with proprietary driver at production scale
