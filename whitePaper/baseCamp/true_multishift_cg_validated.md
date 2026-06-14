# True Multi-Shift CG + Fermion Force Validation

**Date:** 2026-03-28
**Project:** hotSpring (ecoPrimals)
**Experiments:** 105 (Silicon-Routed QCD Revalidation)
**Status:** Production-validated — ΔH=O(1), all trajectories accepted

---

## Summary

Two critical bugs in the RHMC fermion force, plus an optimizer-related CG convergence
failure, were identified and fixed through systematic debugging:

1. **Fermion force sign convention** — The staggered fermion force shader used coefficient
   +η/2, but the gauge force convention in this codebase outputs ∂S/∂U (positive gradient).
   The per-pole fermion contribution must match: F = −η·TA[U·(x_fwd⊗y†−y_fwd⊗x†)].
   The original +η/2 had both wrong sign and wrong magnitude.

2. **True multi-shift CG** — Shares a single Krylov subspace across all shifted systems,
   reducing the dominant cost from N_shifts × I D†D applications to just I. Uses ζ-recurrence
   (Jegerlehner, hep-lat/9612014) for shifted scalar tracking, shifted-base approach with
   σ_min for improved base convergence, and exponential back-off convergence checking.

3. **Compiler optimizer fix** — Release-mode Rust optimized away the convergence check
   `rz_new < tol_sq` when the value wasn't subsequently "used." Fixed with
   `std::hint::black_box(rz_new)` — the idiomatic Rust barrier against dead-code elimination.

## Debugging Methodology

The fermion force sign was traced through a 5-step process:
- Quenched (pure gauge) ΔH ≈ 0 → gauge force + integrator correct
- A/B test: sequential CG vs multi-shift CG → same ΔH → CG is not the problem
- Timestep invariance: dt → dt/10 with ΔH unchanged → not integration error, fundamental force bug
- Hamiltonian repeatability: same config computed twice → identical → H computation deterministic
- Re-derivation comparing gauge force convention (−β/3·TA[U·staple]) to fermion force → sign mismatch found

## Key Files

| File | Change |
|------|--------|
| `staggered_fermion_force_f64.wgsl` | +η/2 → −η (GPU RHMC force) |
| `pseudofermion_force_f64.wgsl` | +η/2 → −η (GPU HMC force) |
| `pseudofermion/mod.rs` | +η/2 → −η (CPU reference) |
| `true_multishift_cg.rs` | New: shared Krylov multi-shift CG |
| `ms_zeta_update_f64.wgsl` | New: ζ-recurrence for shifted scalars |
| `ms_x_update_f64.wgsl` | New: shifted solution update |
| `ms_p_update_f64.wgsl` | New: shifted direction update |
| `cg_compute_alpha_shifted_f64.wgsl` | New: shifted-base α computation |
| `cg_update_xr_shifted_f64.wgsl` | New: shifted-base x/r update |

## Production Results (RTX 3090, 8⁴, β=5.5, Nf=2, m=0.1)

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| ΔH | ~1500 (always rejected) | ~2-3 (all accepted) |
| CG convergence | 350 iter/solve (correct) | 350 iter/solve (correct) |
| Wall time/traj | 26.5s (with diagnostics) | 16.5s (clean, 37% faster) |
| Throughput | 5.3 GFLOP/s | 8.5 GFLOP/s |

## Handoff Notes for Primal Teams

### For barraCuda (primal compute engine)
- The `std::hint::black_box` pattern should be adopted wherever GPU staging buffer
  readbacks are used in convergence loops. This is a general wgpu/release-mode issue.
- True multi-shift CG is general-purpose — any lattice QCD codebase using partial
  fraction rational approximation benefits.
- The sign convention (shader outputs ∂S/∂U, not −∂S/∂U) should be documented as a
  project-wide standard.

### For coralReef (sovereign shader compiler)
- 8 new WGSL shaders validated in production. All use the `f64` precision path.
- The `ms_zeta_update_f64.wgsl` kernel is a single-workgroup scalar recurrence —
  potential target for warp-level optimization in the SASS compiler.

### For springs (downstream consumers)
- Any spring using RHMC should pull the fermion force fix.
- The `black_box` fix applies to any GPU-resident convergence loop.

---

*Fossil record: traced a 1500× ΔH error to a single sign flip in a matrix outer product.
The gauge force convention (∂S/∂U with P += dt·F) is the Rosetta stone.*
