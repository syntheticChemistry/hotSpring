# hotSpring — Paper Review Queue

**Last Updated**: February 18, 2026
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

**Total science cost**: ~$0.15 for 4 papers, 225+ validation checks.

---

## Review Queue — Reordered by Feasibility

### Tier 0 — Immediate: Zero New Primitives Required

These papers can be reproduced using only existing BarraCUDA capabilities.
The goal is maximum science per dollar with no infrastructure investment.

| # | Paper | Journal | Year | Faculty | What We Need | What We Have | Status |
|---|-------|---------|------|---------|-------------|-------------|--------|
| 5 | Stanton & Murillo "Ionic transport in high-energy-density matter" | Phys Rev Lett 116, 075002 | 2016 | Murillo | MD across Yukawa phase diagram, VACF → transport coefficients, Green-Kubo integrals | Sarkas GPU MD (9/9 PP), VACF observable, FusedMapReduceF64 | **NEXT** |
| 6 | Murillo & Weisheit "Dense plasmas, screened interactions, and atomic ionization" | Physics Reports | 1998 | Murillo | Eigensolve for effective potentials, screened Coulomb theory | BatchedEighGpu, BCS bisection, nuclear EOS framework | Queued |

**Paper 5 rationale**: Stanton & Murillo 2016 computes transport coefficients
(diffusion, viscosity, thermal conductivity) from Yukawa MD simulations —
exactly what we already run. They validate against an effective Boltzmann
equation. We reproduce their MD results with BarraCUDA, compute the same
Green-Kubo transport coefficients from our VACF data, and compare to their
published fits. This is a direct extension of Phase C-E with no new code.

**Paper 6 rationale**: Murillo & Weisheit 1998 is a foundational review.
The reproducible numerical content involves effective potentials and
eigenvalue problems — BatchedEighGpu handles this. Partial reproduction
validates our eigensolve infrastructure on a different physics domain.

### Tier 1 — Short-term: Public Data, No Simulation Required

| # | Paper | Journal | Year | Faculty | What We Need | Status |
|---|-------|---------|------|---------|-------------|--------|
| 7 | HotQCD lattice EOS tables (Bazavov et al.) | Nuclear Physics A 931, 867 | 2014 | Bazavov | Download public EOS tables, validate downstream thermodynamics, compare to Sarkas plasma EOS | Queued |

**Paper 7 rationale**: HotQCD EOS tables are publicly available
(jnoronhahostler/Equation-of-State on GitHub). We do NOT run lattice QCD —
we use the published EOS as input to validate thermodynamic analysis
pipelines. This proves the computational overlap between lattice QCD and
plasma physics without needing FFT, Dirac solvers, or SU(3) ops.

### Tier 2 — Medium-term: Needs Complex f64 + SU(3) (No FFT)

| # | Paper | Journal | Year | Faculty | What We Need | Status |
|---|-------|---------|------|---------|-------------|--------|
| 8 | Pure gauge SU(3) Wilson action (subset of Bazavov) | — | — | Bazavov | Complex f64, SU(3) matrix ops, plaquette force, HMC/Metropolis. NO Dirac solver. NO FFT. | Queued |

**Paper 8 rationale**: A pure gauge SU(3) simulation on a small lattice
(e.g., 4^4) requires only: complex f64, SU(3) link multiplication, plaquette
computation, and an HMC-like integrator (adapt Velocity Verlet). It does NOT
need the expensive Dirac solver that dominates full QCD. This is the minimal
lattice simulation that proves shared MD structure between plasma and QCD.

### Tier 3 — Long-term: Full Lattice QCD Stack Required

| # | Paper | Journal | Year | Faculty | What We Need | Status |
|---|-------|---------|------|---------|-------------|--------|
| 9 | Bazavov [HotQCD] (2014) "The QCD equation of state" | Nucl Phys A 931, 867 | 2014 | Bazavov | FFT, complex f64, SU(3), HMC, Dirac CG solver | Blocked (FFT P0) |
| 10 | Bazavov et al. (2016) "Polyakov loop in 2+1 flavor QCD" | Phys Rev D 93, 114502 | 2016 | Bazavov | Same as #9 + Polyakov loop observable | Blocked (FFT P0) |
| 11 | Bazavov et al. (2025) "Hadronic vacuum polarization for the muon g-2" | Phys Rev D 111, 094508 | 2025 | Bazavov | Same as #9 + subpercent precision | Blocked (FFT P0) |
| 12 | Bazavov et al. (2016) "Curvature of the freeze-out line" | Phys Rev D 93, 014512 | 2016 | Bazavov | Same as #9 + inverse problem | Blocked (FFT P0) |
| 13 | Bazavov et al. (2015) "Gauge-invariant Abelian Higgs on optical lattices" | Phys Rev D 92, 076003 | 2015 | Bazavov | Complex f64, gauge theory ops | Blocked (complex f64) |

---

## Paper 5 — Detailed Reproduction Plan

### Stanton & Murillo (2016): Ionic Transport in HED Matter

**What they did**: Computed diffusion coefficients, viscosity, and thermal
conductivity for Yukawa OCP plasmas across coupling strengths Γ = 0.1–175
and screening κ = 0–4. Validated effective Boltzmann equation against MD.

**What we do**:

1. **Run existing Sarkas GPU MD** across Γ-κ grid (we already have the
   infrastructure from Phase C-E, just extend the parameter range)
2. **Compute VACF** from trajectories (observable already implemented)
3. **Green-Kubo integration**: D = (1/3) ∫₀^∞ <v(0)·v(t)> dt
   - Simple numerical integration of VACF — no new GPU primitives
   - Viscosity from stress tensor autocorrelation (needs stress observable)
   - Thermal conductivity from heat current autocorrelation
4. **Compare to Stanton-Murillo fits** — they publish analytical expressions
   for transport coefficients as functions of (Γ, κ)
5. **Cost estimate**: ~20 parameter points × ~5 min each = ~100 min GPU time,
   ~$0.01 electricity on RTX 4070

### New Code Needed

| Component | Effort | Description |
|-----------|--------|-------------|
| Green-Kubo integrator | Low | Trapezoidal integration of VACF → diffusion coefficient |
| Stress tensor observable | Medium | Σ_i F_ij ⊗ r_ij — pair force outer product sum |
| Heat current observable | Medium | J_Q = Σ_i [e_i v_i + (1/2) Σ_j (F_ij · v_i) r_ij] |
| Parameter sweep harness | Low | Extend sarkas_gpu with Γ-κ grid mode |
| Comparison framework | Low | Load Stanton-Murillo analytical fits, compute relative error |

### Validation Target

| Observable | Stanton-Murillo | BarraCUDA | Tolerance |
|-----------|----------------|-----------|-----------|
| Self-diffusion D*(Γ,κ) | Published fit, Table I | Green-Kubo from VACF | < 5% relative |
| Viscosity η*(Γ,κ) | Published fit | Stress autocorrelation | < 10% relative |
| Thermal conductivity λ*(Γ,κ) | Published fit | Heat current autocorrelation | < 10% relative |

---

## Cost Projection

| Paper | Est. GPU Time | Est. Cost | New Code |
|-------|:---:|:---:|:---:|
| #5 Stanton-Murillo transport | ~2 hours | **$0.02** | Green-Kubo, stress tensor |
| #6 Murillo-Weisheit screening | ~30 min | **$0.005** | Effective potential evaluation |
| #7 HotQCD EOS tables | ~5 min | **$0.001** | Table loader, thermodynamic comparison |
| #8 Pure gauge SU(3) | ~8 hours | **$0.08** | Complex f64, SU(3), plaquette force |
| Total (Tier 0-2) | ~11 hours | **~$0.11** | |

**Cumulative science portfolio**: 8 papers reproduced, ~$0.26 total compute cost.

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
- **Paper 5 (Stanton-Murillo)** is the natural next step: same MD code, new observables,
  new physics (transport coefficients), validates Green-Kubo methodology
- Paper 5 → Paper 6 forms a Murillo Group transport chain: MD → transport → screening
- Paper 7 bridges hotSpring ↔ lattice QCD without requiring lattice simulation
- Paper 8 (pure gauge) bridges MD ↔ lattice QCD with minimal new code
- Each paper reproduced at ~$0.01-0.10 proves the cost decrease thesis:
  **consumer GPU + Rust + open-source drivers → democratized computational science**
