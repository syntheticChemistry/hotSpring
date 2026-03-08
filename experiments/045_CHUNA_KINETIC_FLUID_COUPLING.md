# Experiment 045: Chuna Multi-Species Kinetic-Fluid Coupling

**Date**: March 6, 2026
**Paper**: Haack, Murillo, Sagert & Chuna, J. Comput. Phys. (2024), DOI:10.1016/j.jcp.2024.112908
**Related**: Haack, Hauck, Murillo, J. Stat. Phys. 168:826 (2017) — conservative multi-species BGK model
**Status**: ✅ CPU + GPU — Python (18/18) + CPU (16 tests + 20/20) + GPU BGK pipeline — **322× CPU vs Py**
**Priority**: P3 — new domain (kinetic-fluid coupling for HED)

---

## Objective

Reproduce the multi-species kinetic-fluid coupling method of Haack, Murillo,
Sagert & Chuna. The paper couples a conservative multi-species BGK kinetic
equation solver with Euler fluid equations for high-energy-density (HED)
simulations. The key innovation is moment-matching at the kinetic-fluid
interface while preserving mass, momentum, and energy conservation globally.

## Evolution Path

```
Python (kinetic_fluid_control.py) → BarraCuda CPU (kinetic_fluid.rs) → BarraCuda GPU → sovereign
```

## Physics

### Multi-Species BGK Kinetic Equation

For species s with distribution function f_s(x,v,t):

  ∂f_s/∂t + v · ∇f_s = ν_s (M_s - f_s)

where ν_s is the collision frequency and M_s is the target Maxwellian
determined by conservation constraints. The target Maxwellian parameters
(density n_s*, velocity u_s*, temperature T_s*) are chosen to conserve:
- Species mass: ∫ M_s dv = n_s
- Total momentum: Σ_s m_s ∫ v M_s dv = Σ_s m_s n_s u_s
- Total kinetic energy: Σ_s ½m_s ∫ v² M_s dv = Σ_s ½m_s n_s (u_s² + 3T_s/m_s)

This differs from the naive BGK operator where M_s is a simple Maxwellian
at species-local temperature — the conservation-preserving target couples
all species through the constraint system.

### Euler Fluid Equations (1D)

  ∂ρ/∂t + ∂(ρu)/∂x = 0
  ∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0
  ∂E/∂t + ∂((E + p)u)/∂x = 0

where E = ρu²/2 + p/(γ-1) is the total energy density.

### Coupling Interface

At the kinetic-fluid boundary:
1. Kinetic → Fluid: compute moments of f_s at the interface (n, nu, E)
   and set as boundary data for the Euler solver
2. Fluid → Kinetic: construct a Maxwellian from the fluid state and
   set as incoming distribution for the BGK solver
3. Conservation: total mass, momentum, energy flux must match across
   the interface to machine precision

### IMEX Time Integration

The collision term is stiff (ν → ∞ in the fluid limit). The paper uses
Implicit-Explicit (IMEX) Runge-Kutta:
- Advection: explicit (upwind / MUSCL)
- Collision: implicit (fixed-point iteration on moment equations)
- The implicit solve converges under mild CFL-like time step restrictions

## Validation Checks

### Phase 1: Homogeneous BGK Relaxation (no spatial transport)
1. Two-species relaxation: different initial T₁, T₂ → common T_eq
2. Mass conservation: |Σ n_s(t) - Σ n_s(0)| < 1e-14
3. Momentum conservation: |Σ m_s n_s u_s(t) - Σ m_s n_s u_s(0)| < 1e-14
4. Energy conservation: |E_total(t) - E_total(0)| < 1e-14
5. H-theorem: entropy S(t) is monotonically non-decreasing
6. Equilibrium: T₁(t_final) ≈ T₂(t_final) within tolerance

### Phase 2: 1D Euler Shock Tube (fluid-only)
7. Sod shock tube: density/velocity/pressure profiles at t=0.2
8. Contact discontinuity position: x_contact ≈ 0.68 ± 0.02
9. Shock speed: correct Rankine-Hugoniot jump conditions
10. Total mass/momentum/energy conservation through the shock

### Phase 3: Coupled Kinetic-Fluid
11. Domain split: kinetic (x < x_interface) + fluid (x > x_interface)
12. Interface flux continuity: mass/momentum/energy flux match
13. Global conservation: total conserved quantities preserved
14. Consistency: coupled solution approaches fluid-only in the limit ν → ∞

## Connection to Existing Work

- **Paper 44** (BGK dielectric): Same BGK framework, but Paper 44 computes
  linear response (dielectric function) while Paper 45 solves the full
  nonlinear kinetic equation in velocity space
- **Paper 1/5** (Sarkas MD, transport): Provides the particle-level physics
  that the kinetic equation models statistically
- **TTM** (Paper 2): Two-temperature relaxation is a limiting case of
  multi-species BGK relaxation
- **LANL Multi-BGK**: Reference implementation (github.com/lanl/Multi-BGK)
  provides validation data

## Files

- `control/kinetic_fluid/scripts/kinetic_fluid_control.py` — Python control
- `barracuda/src/physics/kinetic_fluid.rs` — BarraCuda CPU module
- `barracuda/src/bin/validate_kinetic_fluid.rs` — Validation binary

## GPU Promotion (March 6, 2026)

`physics/gpu_kinetic_fluid.rs` — GPU-accelerated BGK relaxation pipeline.

### Architecture

Two-pass design via `bgk_relaxation_f64.wgsl`:
- **Pass 1** (`compute_moments`): Each thread computes per-velocity-point
  moment contributions (n·dv, n·v·dv, n·v²·dv). CPU reduces.
- **Pass 2** (`bgk_relax`): Each thread evaluates target Maxwellian and
  applies f → f + dt·ν·(f_target - f). GPU-parallel per (species, v-point).

Target parameters (n*, u*, T*) computed on CPU from reduced moments
via `bgk_target_params()` — O(N_species) work, not worth GPU dispatch.

### Files

| File | Purpose |
|------|---------|
| `physics/gpu_kinetic_fluid.rs` | Host pipeline, buffer management |
| `physics/shaders/bgk_relaxation_f64.wgsl` | WGSL compute shader |

### Remaining Extensions

- [ ] GPU Euler update (spatial mesh — currently small N_x, CPU sufficient)
- [ ] Full coupled kinetic-fluid on GPU with ToadStool streaming
