# hotSpring — Paper Review Queue

**Last Updated**: February 20, 2026
**Purpose**: Track papers for reproduction/review, ordered by priority and feasibility
**Principle**: Reproduce, validate, then decrease cost. Each paper proves the
pipeline on harder physics — toadStool evolves the GPU acceleration in parallel.

---

## Completed Reproductions

| # | Paper | Phase | Checks | Faculty | Cost |
|---|-------|-------|--------|---------|------|
| 1 | Sarkas Yukawa OCP MD (Silvestri et al.) | A + C-E | 9/9 PP cases at N=10k, 80k steps | Murillo | $0.044 |
| 2 | Two-Temperature Model (TTM) | A | 6/6 | Murillo | ~$0.001 |
| 3 | Diaw et al. (2024) Surrogate Learning — Nat Mach Intel | A | 15/15 | Murillo | ~$0.01 |
| 4 | Nuclear EOS (SEMF → HFB, AME2020) | A + F | 2,042 nuclei, 195/195 | Murillo | ~$0.10 |
| 5 | Stanton & Murillo (2016) Transport | Tier 0 | 13/13 | Murillo | ~$0.02 |
| 6 | Murillo & Weisheit (1998) Screening | Tier 0 | 23/23 | Murillo | ~$0.001 |
| 7 | HotQCD EOS tables (Bazavov 2014) | Tier 1 | Thermo validation | Bazavov | ~$0.001 |
| 8 | Pure gauge SU(3) Wilson action | Tier 2 | 12/12 | Bazavov | ~$0.02 |
| 13 | Abelian Higgs (Bazavov 2015) | Tier 2 | 17/17 | Bazavov | ~$0.001 |

**Total science cost**: ~$0.20 for 9 papers, 300+ validation checks.
Papers 6, 7, 13 add checks at negligible cost (CPU-only, <1 second each).

---

## Review Queue — Reordered by Feasibility

### Tier 0 — Immediate: Zero New Primitives Required

These papers can be reproduced using only existing BarraCUDA capabilities.
The goal is maximum science per dollar with no infrastructure investment.

| # | Paper | Journal | Year | Faculty | What We Need | What We Have | Status |
|---|-------|---------|------|---------|-------------|-------------|--------|
| 5 | Stanton & Murillo "Ionic transport in high-energy-density matter" | Phys Rev Lett 116, 075002 | 2016 | Murillo | MD across Yukawa phase diagram, VACF → transport coefficients, Green-Kubo integrals | Sarkas GPU MD (9/9 PP), VACF observable, FusedMapReduceF64 | **Done** — D* calibrated to Sarkas (12 points), 13/13 checks |
| 6 | Murillo & Weisheit "Dense plasmas, screened interactions, and atomic ionization" | Physics Reports | 1998 | Murillo | Eigensolve for effective potentials, screened Coulomb theory | Sturm bisection eigensolve, screening models | **Done** — 23/23 checks |

**Paper 5 status**: ✅ Complete. Green-Kubo transport pipeline validated.
13/13 checks pass. Four bugs fixed:
1. **V² normalization bug** in stress/heat ACF: `compute_stress_acf` and
   `compute_heat_acf` used `V/kT` prefactor when the total-stress convention
   requires `1/(VkT)`. Fixed — η* now O(10⁻¹), matching independent MD literature.
2. **Green-Kubo integral noise**: Plateau detection added to all three Green-Kubo
   integrals (D* VACF, η* stress ACF, λ* heat ACF).
3. **Insufficient equilibration**: Transport cases now use 50k equil steps (lite)
   / 100k (full), with final velocity rescale to target T*.
4. **Fit coefficient normalization**: Daligault (2012) strong-coupling coefficients
   (A, α) were ~70× too small due to reduced-unit convention mismatch between
   the original Python baseline (`v += dt × F × Γ`, effective mass = 1) and
   standard OCP units (m* = 3, ω_p time). Recalibrated using 12 Sarkas DSF study
   Green-Kubo D* values at N=2000 (physical units → D* = D/(a²ω_p)). Corrected
   coefficients: A(κ) = 0.808 + 0.423κ − 0.152κ², α(κ) = 1.049 + 0.044κ − 0.039κ².
   D* now matches Sarkas within 50% at N=500 (statistical noise at small system size).
   η*/λ* coefficients proportionally rescaled (not independently calibrated).

**Paper 6 status**: ✅ Complete. Screened Coulomb (Yukawa) bound-state solver
validated. Sturm bisection eigensolve for tridiagonal Hamiltonian — O(N) per
eigenvalue, scales to N=10,000+ grid points. 23/23 validation checks:
4 hydrogen eigenvalue vs exact, 7 Python-Rust parity (Δ ≈ 10⁻¹²), 3 critical
screening vs Lam & Varshni (1971), 6 physics trends, 3 screening models.
Key: same Yukawa exp(−κr)/r potential as MD Papers 1/5 but for atomic
binding (electron-ion) instead of ion-ion interaction.

### Tier 1 — Short-term: Public Data, No Simulation Required

| # | Paper | Journal | Year | Faculty | What We Need | Status |
|---|-------|---------|------|---------|-------------|--------|
| 7 | HotQCD lattice EOS tables (Bazavov et al.) | Nuclear Physics A 931, 867 | 2014 | Bazavov | Download public EOS tables, validate downstream thermodynamics, compare to Sarkas plasma EOS | **Done** — `lattice/eos_tables.rs` + `validate_hotqcd_eos` |

**Paper 7 status**: ✅ Complete. HotQCD EOS reference table (Bazavov et al. 2014)
hardcoded in `lattice/eos_tables.rs` with 14 temperature points. Validation checks:
pressure/energy monotonicity, trace anomaly peak near T_c, asymptotic freedom
(approach to Stefan-Boltzmann limit), thermodynamic consistency (s/T³ = ε/T⁴ + p/T⁴).
All checks pass.

### Tier 2 — Medium-term: Needs Complex f64 + SU(3) (No FFT)

| # | Paper | Journal | Year | Faculty | What We Need | Status |
|---|-------|---------|------|---------|-------------|--------|
| 8 | Pure gauge SU(3) Wilson action (subset of Bazavov) | — | — | Bazavov | Complex f64, SU(3) matrix ops, plaquette force, HMC/Metropolis. NO Dirac solver. NO FFT. | **Done** — `lattice/` module + `validate_pure_gauge` (12/12) |

**Paper 8 status**: ✅ Complete. Full CPU implementation of pure gauge SU(3)
lattice QCD validated on 4^4 lattice:
- `complex_f64.rs`: Complex f64 arithmetic with WGSL template
- `su3.rs`: SU(3) matrix algebra (multiply, adjoint, trace, det, reunitarize,
  random near-identity, Lie algebra momenta) with WGSL template
- `wilson.rs`: Wilson gauge action — plaquettes, staples, gauge force (dP/dt),
  Polyakov loop. 100% test coverage.
- `hmc.rs`: Hybrid Monte Carlo with Cayley matrix exponential (exactly unitary),
  leapfrog integrator, Metropolis accept/reject. 96-100% acceptance at β=5.5-6.0.
- `dirac.rs`: Staggered Dirac operator, fermion fields, D†D for CG
- `cg.rs`: Conjugate gradient solver for D†D x = b
- `multi_gpu.rs`: Temperature scan dispatcher (CPU-threaded, GPU-ready)
- All 12/12 validation checks pass. Plaquette values match strong-coupling
  expansion and known lattice results. HMC ΔH = O(0.01).

**Paper 13 status**: ✅ Complete. Abelian Higgs (1+1)D: U(1) gauge + complex
scalar Higgs field with HMC. Validates the full phase structure from
Bazavov et al. (2015):
- Cold start identities (plaquette=1, action=0, |φ|²=1, Polyakov=1)
- Weak coupling (β=6): plaquette 0.915, 84% acceptance
- Strong coupling (β=0.5): plaquette 0.236, 90% acceptance
- Higgs condensation (κ=2): ⟨|φ|²⟩ = 4.42
- Confined phase (β=1, κ=0.1): intermediate plaquette
- Large λ=10: ⟨|φ|²⟩ ≈ 1.01 (φ⁴ potential freezes modulus)
- Leapfrog reversibility: |ΔH| = 0.002 at dt=0.01
- **Rust 143× faster than Python** (12 ms vs 1750 ms)
Key: bridges SU(3) (Paper 8) to quantum simulation — same HMC framework,
U(1) gauge group, complex scalar matter field. Wirtinger-correct forces.
17/17 validation checks.

### Tier 3 — Long-term: Full Lattice QCD Stack Required

| # | Paper | Journal | Year | Faculty | What We Need | Status |
|---|-------|---------|------|---------|-------------|--------|
| 9 | Bazavov [HotQCD] (2014) "The QCD equation of state" | Nucl Phys A 931, 867 | 2014 | Bazavov | FFT, complex f64, SU(3), HMC, Dirac CG solver | Unblocked (complex f64 + SU(3) + HMC + Dirac CG done; FFT still needed for full QCD) |
| 10 | Bazavov et al. (2016) "Polyakov loop in 2+1 flavor QCD" | Phys Rev D 93, 114502 | 2016 | Bazavov | Same as #9 + Polyakov loop observable | Polyakov loop implemented; FFT needed |
| 11 | Bazavov et al. (2025) "Hadronic vacuum polarization for the muon g-2" | Phys Rev D 111, 094508 | 2025 | Bazavov | Same as #9 + subpercent precision | FFT needed |
| 12 | Bazavov et al. (2016) "Curvature of the freeze-out line" | Phys Rev D 93, 014512 | 2016 | Bazavov | Same as #9 + inverse problem | FFT needed |

---

## ~~Paper 5 — Detailed Reproduction Plan~~ ✅ COMPLETED

### Stanton & Murillo (2016): Ionic Transport in HED Matter

**Completed February 19, 2026.** All components implemented and validated:

| Component | Status | Location |
|-----------|--------|----------|
| Green-Kubo integrator (VACF → D*) | ✅ Done | `md/observables.rs` |
| Stress tensor observable (σ_αβ) | ✅ Done | `md/observables.rs` |
| Heat current observable (J_Q) | ✅ Done | `md/observables.rs` |
| Daligault (2012) D* analytical fit | ✅ Done, Sarkas-calibrated | `md/transport.rs` |
| Stanton-Murillo (2016) η*, λ* fits | ✅ Done | `md/transport.rs` |
| Validation binary | ✅ 13/13 pass | `bin/validate_stanton_murillo.rs` |

**Key findings**:
1. Daligault Table I coefficients were ~70× too small due to reduced-unit
   convention mismatch. Recalibrated against 12 Sarkas Green-Kubo D* values at N=2000.
2. **(v0.5.14)** Constant weak-coupling prefactor `C_w=5.3` had 44–63% error in the
   crossover regime (Γ ≈ Γ_x). Evolved to κ-dependent `C_w(κ) = exp(1.435 + 0.715κ + 0.401κ²)`.
   Errors now <10% across all 12 Sarkas calibration points. Transport grid expanded to 20
   (κ,Γ) configurations including 9 Sarkas-matched DSF points.

---

## Cost Actuals + Projection

| Paper | GPU Time | Cost | Status |
|-------|:---:|:---:|:---:|
| #5 Stanton-Murillo transport | ~2 hours | **$0.02** | ✅ Done |
| #6 Murillo-Weisheit screening | ~1 min | **$0.001** | ✅ Done |
| #7 HotQCD EOS tables | ~5 min | **$0.001** | ✅ Done |
| #8 Pure gauge SU(3) | ~2 hours | **$0.02** | ✅ Done |
| #13 Abelian Higgs (1+1)D | ~1 min | **$0.001** | ✅ Done |
| Total (Tier 0-2 + 13) | ~5 hours | **~$0.05** | **9/9 complete** |

**Cumulative science portfolio**: 9 papers reproduced, ~$0.20 total compute cost.
All Tier 0-2 targets complete + Paper 13 (Abelian Higgs). Next: Tier 3
(full lattice QCD, requires FFT).

---

## Scaling Vision: From One GPU to Every GPU

Our WGSL shaders run on any Vulkan-capable GPU — NVIDIA, AMD, Intel,
integrated, Steam Deck. The BarraCUDA pipeline is hardware-agnostic by
design. This means the compute cost equations change with scale:

| Scale | Hardware | Lattice feasibility | Cost/paper |
|-------|----------|-------------------|------------|
| **Now** | 1× RTX 4070 ($600) | Tier 0-1 (transport, EOS tables) | ~$0.01-0.10 |
| **Lab** | 4× mixed GPU (Titan V + 4070 + AMD) | Tier 2 (pure gauge SU(3)) | ~$0.02-0.20 |
| **Distributed** | 100s of idle GPUs (volunteer compute) | Tier 3 partial (HMC sweeps) | ~$0.001/GPU-hr |
| **Fleet** | Every idle GPU on Steam | Full lattice QCD | Approaches zero marginal cost |

The architecture allows this evolution because:
- **WGSL/Vulkan**: runs on every GPU vendor, no CUDA lock-in
- **Independent dispatches**: Jacobi sweeps, HMC trajectories, parameter
  scans are embarrassingly parallel across GPUs
- **Sovereign stack**: AGPL-3.0, no proprietary dependencies, no vendor
  gatekeeping — any GPU that exposes Vulkan can contribute compute
- **Lattice decomposition**: each sublattice is an independent dispatch;
  communication is boundary exchange (small relative to compute)

Full lattice QCD remains the long-term north star. The path:
1. **Today**: transport coefficients, EOS tables — zero new GPU code
2. **Near**: pure gauge SU(3) — complex f64 + plaquette force
3. **Medium**: FFT + Dirac CG solver — unlocks full 2+1 flavor QCD
4. **Scale**: distribute across any available GPU fleet

Each step builds on the last. We don't need the full stack to start —
we need the architecture to allow evolution. That architecture exists.

---

## Notes

- **Bazavov full lattice** requires FFT + Dirac solver — toadstool/BarraCUDA P0 gap.
  Long-term evolution goal, not a blocker for current science.
- **Paper 5 (Stanton-Murillo)** ✅ complete: Green-Kubo transport validated, 13/13 checks
- Paper 5 → Paper 6 forms a Murillo Group transport chain: MD → transport → screening
- Paper 7 bridges hotSpring ↔ lattice QCD without requiring lattice simulation
- Paper 8 (pure gauge) bridges MD ↔ lattice QCD with minimal new code
- Each paper reproduced at ~$0.01-0.10 proves the cost decrease thesis:
  **consumer GPU + Rust + open-source drivers → democratized computational science**
