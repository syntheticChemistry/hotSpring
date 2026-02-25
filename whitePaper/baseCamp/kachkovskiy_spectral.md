# Kachkovskiy — Spectral Theory & Anderson Localization

**Domain:** Spectral theory, Anderson localization, almost-Mathieu, Hofstadter butterfly
**Updated:** February 25, 2026
**Status:** ✅ 45/45 checks pass, GPU Lanczos validated

---

## What We Reproduced

Professor Kachkovskiy's research at Michigan State covers spectral properties of
quasi-periodic and random Schrödinger operators. Our reproduction implements
these from first principles in Rust with GPU acceleration.

| Study | Checks | Result |
|-------|--------|--------|
| Anderson localization (1D) | 10/10 | GOE→Poisson transition, Herman γ=ln|λ|, Aubry-André |
| Lanczos + 2D Anderson | 11/11 | SpMV parity 1.78e-15, full spectrum, bandwidth |
| 3D Anderson | 10/10 | Mobility edge, dimensional hierarchy 1D<2D<3D |
| Hofstadter butterfly | 10/10 | Band counting q=2,3,5, Cantor measure, α↔1-α symmetry |
| GPU SpMV + Lanczos | 14/14 | CSR SpMV parity 1.78e-15, Lanczos eigenvalues match CPU to 1e-15 |

## Evolution Path

1. **CPU spectral**: Sparse matrix (CSR), Lanczos eigensolve, Anderson/almost-Mathieu/Hofstadter Hamiltonians.

2. **GPU SpMV**: CSR sparse matrix-vector multiply on GPU. Parity with CPU at 1.78e-15.

3. **GPU Lanczos**: Full Lanczos eigensolve on GPU. Eigenvalues match CPU to 1e-15.

4. **toadStool absorption** (v0.6.9): 41 KB of local spectral code deleted. Now leans entirely on upstream `barracuda::spectral::*`. Only a `CsrMatrix` type alias retained.

## Key Finding

The spectral module demonstrates the full "Write → Absorb → Lean" cycle. All
spectral code originated in hotSpring, was absorbed by toadStool, and hotSpring
now re-exports from upstream — zero local implementation, full validation parity.

## Cross-Spring Contributions

- **CSR SpMV GPU shader** → toadStool `spmv_f64.wgsl` → used by neuralSpring for graph operations
- **Lanczos eigensolve** → toadStool `LanczosGpu` → available to all springs
- **Anderson Hamiltonian** → toadStool `spectral::anderson` → educational and research tool
