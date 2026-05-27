# Sovereign CompChem Evolution

**Status:** Active — Papers 50-58 baseline tracking
**Last Updated:** 2026-05-26
**Prerequisite:** Paper 50 (FES Gaussian summation) ✅ PROVEN on GPU
**Long-term target:** Sovereign all-atom MD at enzyme scale (93K atoms)

## Philosophy

Sovereign all-atom MD is achieved by evolving individual GPU kernels through the
same pipeline as Lattice QCD: Python Control → BarraCuda CPU → BarraCuda GPU →
metalForge. Each paper entry in the queue isolates exactly one mathematical kernel,
validates it at machine-epsilon against the industry control (GROMACS), then composes
with prior kernels. Nature doesn't build the enzyme in one step — it evolves each
domain independently and composes.

The hardware exists (see `gen3/about/HARDWARE.md`). The scale exists (32⁴ lattice =
1,048,576 sites on consumer GPU, N=20K Yukawa). The gap is specific algorithms, not
fundamental capability.

---

## Kernel Decomposition: All-Atom MD

All-atom MD at the GROMACS level decomposes into 7 independent mathematical kernels.
Each becomes a WGSL shader, proven at parity, then composed into a sovereign integrator.

### K1: Non-bonded short-range (Lennard-Jones + Coulomb direct)

| Property | Specification |
|----------|--------------|
| **Math** | `F_ij = 4ε[(σ/r)¹² − (σ/r)⁶] + q_i·q_j / (4πε₀·r²)` with switching function |
| **Parallelism** | O(N²) all-pairs or O(N) with cell-list (Verlet neighbor list) |
| **Existing base** | `yukawa_force_f64.wgsl` (all-pairs, N=20K), `cell_list_f64.wgsl` (neighbor search) |
| **Delta to sovereign** | Replace Yukawa `exp(-κr)/r` with LJ 12-6 + direct Coulomb. Trivial kernel swap. |
| **Paper** | Extension of Paper 1 (Sarkas Yukawa). Nearest evolution step. |
| **Parity target** | GROMACS `nb_kernel_ElecCoul_VdwLJ_F` on same topology |
| **Status** | P1 — nearest-term (kernel shape identical to Yukawa, different potential) |

### K2: Bonded interactions (bonds, angles, dihedrals, impropers, 1-4)

| Property | Specification |
|----------|--------------|
| **Math — harmonic bond** | `E = ½ k_b (r − r₀)²` |
| **Math — harmonic angle** | `E = ½ k_θ (θ − θ₀)²` |
| **Math — periodic dihedral** | `E = k_φ [1 + cos(nφ − δ)]` |
| **Math — improper** | `E = ½ k_ξ (ξ − ξ₀)²` |
| **Math — 1-4 scaling** | `fudgeLJ × LJ + fudgeQQ × Coulomb` (AMBER: 0.5/0.8333) |
| **Parallelism** | Per-bond independent (SIMD over bond list). N_bonds ≈ N_atoms for proteins. |
| **Existing base** | None in barracuda (bonded not needed for plasma/QCD) |
| **Delta to sovereign** | New shaders: `harmonic_bond_f64.wgsl`, `angle_f64.wgsl`, `dihedral_f64.wgsl`, `improper_f64.wgsl` |
| **Paper** | 51 |
| **Parity target** | GROMACS bonded kernels on alanine dipeptide (22 atoms), then enzyme (93K) |
| **Status** | P2 — each kernel is simple; the integration list parsing is the work |

### K3: Constraint solver (LINCS)

| Property | Specification |
|----------|--------------|
| **Math** | Linear Constraint Solver: project bond vectors to satisfy `|r_i − r_j| = d₀` after unconstrained step. Iterative: `B = A·S⁻¹·A^T`, solve `B·λ = (|r'| − d₀)`, correct positions. |
| **Reference** | Hess et al., J. Comput. Chem. 18:1463 (1997) |
| **Parallelism** | Per-constraint independent at each iteration (coupled across iterations). LINCS order=4 typically converges in 1-2 matrix-vector products. |
| **Existing base** | None. New domain for barracuda. |
| **Delta to sovereign** | `lincs_project_f64.wgsl` — constraint projection + correction pass |
| **Paper** | 52 |
| **Parity target** | GROMACS LINCS (4th order, 1 iteration) on TIP3P water + alanine dipeptide |
| **Scaling note** | 93K-atom enzyme: ~31K constraints (all H-bonds + TIP3P SETTLE). Each constraint is 3 multiply-adds. Total: ~93K FLOP/step — trivially GPU-parallel. |
| **Status** | P2 — well-defined algorithm, clean parallelism after graph coloring |

### K4: Long-range electrostatics (PME)

| Property | Specification |
|----------|--------------|
| **Math** | Particle Mesh Ewald: split Coulomb into short-range (real-space, erfc) + long-range (reciprocal-space, FFT on mesh). `E_recip = (1/2V) Σ_k |S(k)|² · f(k)`, with charge spreading (B-spline) and force interpolation. |
| **Reference** | Darden et al., J. Chem. Phys. 98:10089 (1993); Essmann et al., J. Chem. Phys. 103:8577 (1995) |
| **Parallelism** | Charge spreading: per-atom (scatter). FFT: collective (3D). Force interpolation: per-atom (gather). |
| **Existing base** | `validate_pppm` (Ewald PPPM, N=64), `Fft3DF64` (toadStool, validated roundtrip 1e-10) |
| **Delta to sovereign** | B-spline charge spreading shader (order=4), `pme_reciprocal_f64.wgsl` (multiply + FFT), force gather shader. FFT3D infrastructure already GPU-validated. |
| **Paper** | 53 |
| **Parity target** | GROMACS PME (grid=72³, order=4, β=3.38 nm⁻¹) on TIP3P box |
| **Scaling note** | Grid 72³ = 373K points. FFT3D already proven at scale in QCD (lattice volumes to 64³×8 = 2M sites). |
| **Status** | P2 — hardest individual kernel, but FFT infrastructure exists |

### K5: Velocity Verlet integrator

| Property | Specification |
|----------|--------------|
| **Math** | `v(t+½dt) = v(t) + ½·F(t)/m·dt`; `r(t+dt) = r(t) + v(t+½dt)·dt`; forces at t+dt; `v(t+dt) = v(t+½dt) + ½·F(t+dt)/m·dt` |
| **Parallelism** | Per-atom fully independent |
| **Existing base** | `velocity_verlet_f64.wgsl` — proven at N=20K, 80K steps, 0.000% energy drift |
| **Delta to sovereign** | Already implemented. Only change: composite force gather from K1-K4. |
| **Paper** | Part of 54 (integrator composition) |
| **Parity target** | GROMACS `md-vv` mode |
| **Status** | ✅ DONE — exists and is validated |

### K6: Thermostat / barostat

| Property | Specification |
|----------|--------------|
| **Math — Berendsen** | `λ = [1 + (dt/τ)(T_target/T − 1)]^½` |
| **Math — V-rescale** | Bussi-Donadio-Parrinello stochastic velocity rescaling (correct canonical ensemble) |
| **Parallelism** | Global reduction (T_kinetic) then per-atom scale |
| **Existing base** | `berendsen_f64.wgsl`, `nose_hoover_f64.wgsl` |
| **Delta to sovereign** | V-rescale shader (stochastic term requires GPU PRNG — already have `GpuPrng` from streaming HMC) |
| **Paper** | Part of 54 |
| **Parity target** | GROMACS v-rescale thermostat |
| **Status** | P3 — Berendsen exists, V-rescale is straightforward extension with existing PRNG |

### K7: FES reconstruction (analysis kernel)

| Property | Specification |
|----------|--------------|
| **Math** | `F(x,y) = −Σ_g h_g · exp(−(x−x_g)²/2σ²_x) · exp(−(y−y_g)²/2σ²_y)` |
| **Parallelism** | Embarrassingly parallel over grid points (N_grid ≫ N_gaussians typical) |
| **Existing base** | `fes_gaussian_sum_f64.wgsl` — 110² grid × 20K Gaussians validated |
| **Delta to sovereign** | DONE |
| **Paper** | 50 |
| **Status** | ✅ PROVEN — 11-14× GPU speedup, RMSD 1e-14 vs CPU |

---

## Composition Path: Papers → Sovereign MD

```
Paper 50: K7 (FES)                    ✅ PROVEN (May 2026)
   ↓
Paper 51: K2 (bonded)                 NEXT — isolated kernel, simple math
   ↓
Paper 52: K3 (LINCS)                  parallel constraint projection
   ↓
Paper 53: K4 (PME)                    FFT3D foundation exists (toadStool)
   ↓
Paper 1 extension: K1 (LJ+Coulomb)   Yukawa → LJ kernel swap
   ↓
Paper 54: K5+K6 + compose(K1-K4)     full integrator = sovereign mdrun
   ↓
Paper 58: 93K-atom enzyme             industry-parity with GROMACS
```

Each step validates against GROMACS on the same system. The GuideStone wraps the
parity evidence: industry output (GROMACS) vs sovereign output (barracuda) with
tolerance class and RMSD < ε.

---

## Scale Precedent (why this is feasible)

| Domain | Achieved scale | Hardware | Kernel count |
|--------|---------------|----------|:---:|
| Lattice QCD (quenched) | 32⁴ = 1,048,576 sites | Single GPU (AMD RX 6950 XT) | 6 shaders |
| Lattice QCD (asymmetric) | 64³×8 = 2,097,152 sites | Single GPU | 6 shaders |
| Yukawa MD (production) | N=10,000 particles | Single GPU (RTX 3090) | 4 shaders |
| Yukawa MD (scaling) | N=20,000 particles | Single GPU | 4 shaders |
| All-atom enzyme (target) | 93,000 atoms | Dual GPU (AMD+NVIDIA) | 7 shaders |
| CompChem FES (validated) | 12,100 grid × 20K Gaussians | Single GPU | 1 shader |

The enzyme system (93K atoms) is ~4.5× the proven Yukawa scale and ~11× smaller than
the proven QCD scale. The gap is algorithmic (bonded + constraints + PME), not scale.

---

## Cross-Spring Integration

| Spring | Contribution to sovereign MD |
|--------|------------------------------|
| hotSpring | All physics kernels (K1-K7), GPU validation, parity evidence |
| toadStool | FFT3D (PME), dispatch infrastructure, multi-backend routing |
| coralReef | WGSL → native compilation (future: bypass wgpu for raw performance) |
| barracuda | Shader library, GPU abstraction, hardware profiling |
| NUCLEUS | Primal composition, workload scheduling, multi-GPU orchestration |

---

## Validation Strategy Per Paper

Each paper follows the established pattern:

1. **Python/Industry Control** — Run the kernel in GROMACS/PLUMED, capture reference output
2. **BarraCuda CPU** — Implement in pure Rust, validate against control (machine-epsilon)
3. **BarraCuda GPU** — WGSL shader, dispatch via wgpu, validate against CPU (machine-epsilon)
4. **GuideStone** — Package parity evidence as self-verifying pseudoSpore module

Tolerance classes (from `pseudoSpore v1.6.1`, originally established in v1.5.0):

| Class | Threshold | Application |
|-------|-----------|-------------|
| `strict` | RMSD < 0.5 kJ/mol | FES reconstruction (Paper 50) |
| `relaxed` | RMSD < 2.0 kJ/mol | Sampling-dependent (metadynamics convergence) |
| `statistical` | Overlap > 95% | Basin/barrier identification |
| `kernel-parity` | RMSD < 1e-12 | Individual kernel isolation tests (Papers 51-54) |

---

## Hardware Dispatch Plan (Sovereign MD)

Based on GPU profiling (bench_device_pair, validate_dual_gpu_qcd):

| Workload | Target device | Rationale |
|----------|--------------|-----------|
| K1 (non-bonded) | AMD RX 6950 XT | Largest FLOP budget, AMD 4.5× faster (HMC parity) |
| K2 (bonded) | Either | Low compute, bandwidth-bound |
| K3 (LINCS) | AMD RX 6950 XT | Iterative solver, f64 precision critical |
| K4 (PME FFT) | NVIDIA RTX 3090 | Large VRAM (24 GB), FFT is memory-bound |
| K4 (PME spread/gather) | Split | Per-atom ops, any device |
| K5+K6 (integrator) | Either | Per-atom, trivial |
| K7 (FES analysis) | AMD RX 6950 XT | Proven on both, AMD preferred (NVK zero-poison avoidance) |

Multi-GPU composition via `toadStool` dispatch:
- Bonded + non-bonded → AMD (compute-heavy)
- PME FFT → NVIDIA (memory-heavy)
- Position sync: 93K × 3 × f64 = 2.2 MB per step (PCIe: ~0.07 ms)

---

## Next Actions (Priority Order)

1. **Paper 51 (bonded)** — Implement `harmonic_bond_f64.wgsl`, validate on alanine dipeptide vs GROMACS
2. **Paper 1 extension** — LJ + direct Coulomb shader (swap Yukawa kernel)
3. **Paper 52 (LINCS)** — Constraint projection shader, validate on TIP3P water
4. **Paper 53 (PME)** — B-spline charge spread + reciprocal multiply (FFT3D exists)
5. **Paper 54 (composition)** — Wire K1-K6 into single `sovereign_mdrun` dispatch
6. **Paper 58 (enzyme scale)** — 93K-atom enzyme system at GROMACS parity
