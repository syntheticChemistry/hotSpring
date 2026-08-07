# Spec: Lanczos Scaling Benchmark

**Binary**: `bench_lanczos_scaling` (extend existing `validate_gpu_lanczos`)  
**Module**: `barracuda::spectral::lanczos`  
**Target reviewer**: Ilya Kachkovskiy (MSU Mathematics)

---

## Purpose

Demonstrate GPU Lanczos eigensolve at N=10,000+ sites with timing data
and direct SciPy `eigsh` comparison. This is the P0 gate for Kachkovskiy
outreach — must answer "faster than what I use?"

---

## Interface

```bash
# Full scaling sweep
cargo run --release --features barracuda-local --bin bench_lanczos_scaling

# Single size
cargo run --release --features barracuda-local --bin bench_lanczos_scaling -- \
  --model anderson3d --L 22 --k 50 --disorder 4.0
```

---

## Benchmarks Required

### Phase 1: Scaling Sweep

| Model | L | N (sites) | k (eigenvalues) | W (disorder) |
|---|---|---|---|---|
| Anderson 3D | 10 | 1,000 | 20 | 4.0 |
| Anderson 3D | 14 | 2,744 | 50 | 4.0 |
| Anderson 3D | 18 | 5,832 | 50 | 4.0 |
| Anderson 3D | 22 | 10,648 | 50 | 4.0 |
| Anderson 3D | 26 | 17,576 | 50 | 16.5 (near W_c) |
| Anderson 3D | 30 | 27,000 | 50 | 16.5 |
| Hofstadter | — | 5,000 | 100 | — |
| Hofstadter | — | 10,000 | 100 | — |

### Phase 2: SciPy Comparison

For each (model, L, k) above:
1. Export sparse matrix to SciPy-readable format (NPZ or MatrixMarket)
2. Python script: `scipy.sparse.linalg.eigsh(H, k=k, which='SM')`
3. Time on same CPU (EPYC 7452, single-threaded for fair comparison)
4. Report:
   - Wall time (GPU vs CPU-Rust vs SciPy)
   - Eigenvalue parity (max |λ_gpu - λ_scipy|)
   - Memory usage

### Phase 3: Two-Particle (Kachkovskiy headline)

| Model | L | N (Hilbert dim = L²) | k | Parameters |
|---|---|---|---|---|
| Two-particle Anderson 1D | 50 | 2,500 | 20 | W=2.0, U=1.0 |
| Two-particle Anderson 1D | 100 | 10,000 | 50 | W=2.0, U=1.0 |
| Two-particle Anderson 1D | 150 | 22,500 | 50 | W=2.0, U=1.0 |
| Two-particle Anderson 1D | 200 | 40,000 | 50 | W=2.0, U=1.0 |

**Hamiltonian**: H = H₁ ⊗ I + I ⊗ H₂ + U·δ(x₁−x₂)  
where H₁, H₂ are single-particle Anderson tight-binding.

---

## Output Format

```
═══ Lanczos Scaling Benchmark ═══

  Model: Anderson 3D (tight-binding, W=4.0)
  GPU: NVIDIA GeForce RTX 3090

  L=10  N=1,000     k=20    GPU: 0.05s   CPU: 0.12s   SciPy: 0.34s   Speedup: 6.8×
  L=14  N=2,744     k=50    GPU: 0.12s   CPU: 0.89s   SciPy: 2.1s    Speedup: 17.5×
  L=18  N=5,832     k=50    GPU: 0.31s   CPU: 3.2s    SciPy: 8.7s    Speedup: 28×
  L=22  N=10,648    k=50    GPU: 0.67s   CPU: 9.1s    SciPy: 24s     Speedup: 36×
  ...

  Eigenvalue parity: max |Δλ| = X.Xe-15 (machine precision)
```

---

## Demo Package (for Kachkovskiy in-person)

What to show on laptop at Wells Hall:

1. **Hofstadter butterfly** — live rendering, full resolution in <5 seconds
2. **Anderson 3D mobility edge** — W sweep showing GOE→Poisson transition
3. **This benchmark output** — GPU vs SciPy timing table
4. **Two-particle localization** — if Phase 3 complete

---

## Files

- `barracuda::spectral::lanczos` — GPU Lanczos implementation (validated to N=1000)
- `barracuda::spectral::anderson` — Anderson model builders
- `src/bin/bench_lanczos_scaling.rs` — Binary (to be written)
- `specs/LANCZOS_SCALING.md` — This document
- `scripts/scipy_eigsh_comparison.py` — SciPy timing script (to be written)
