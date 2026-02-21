#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Lattice QCD CG Solver — Python Control Baseline

Independent NumPy implementation of the staggered Dirac operator and
conjugate gradient solver. Uses identical algorithms and LCG PRNG as
the Rust implementation, producing bit-identical lattice configurations.

Purpose: establish Python baseline for Rust/GPU parity validation.
  Python (interpreted) → Rust CPU (compiled) → Rust GPU (portable)

References:
  - Kogut & Susskind, PRD 11, 395 (1975) — staggered fermions
  - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 5, 8.4
"""

import numpy as np
import time
import sys

# ═══════════════════════════════════════════════════════════════════
#  LCG PRNG — matches Rust lattice/constants.rs exactly
# ═══════════════════════════════════════════════════════════════════

LCG_A = np.uint64(6_364_136_223_846_793_005)
LCG_C = np.uint64(1_442_695_040_888_963_407)
LCG_53_DIVISOR = float(1 << 53)


def lcg_step(seed):
    return np.uint64(seed * LCG_A + LCG_C)


def lcg_uniform(seed):
    seed = lcg_step(seed)
    return seed, float(np.uint64(seed) >> np.uint64(11)) / LCG_53_DIVISOR


# ═══════════════════════════════════════════════════════════════════
#  SU(3) Matrix Operations
# ═══════════════════════════════════════════════════════════════════

def su3_identity():
    return np.eye(3, dtype=np.complex128)


def su3_random_near_identity(seed, epsilon):
    """Generate random SU(3) near identity — matches Rust Su3Matrix::random_near_identity."""
    m = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            seed, re = lcg_uniform(seed)
            seed, im = lcg_uniform(seed)
            m[i, j] = complex((re - 0.5) * epsilon, (im - 0.5) * epsilon)
    m = np.eye(3, dtype=np.complex128) + m
    # Gram-Schmidt reunitarization
    u0 = m[0] / np.linalg.norm(m[0])
    u1 = m[1] - np.dot(np.conj(u0), m[1]) * u0
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(np.conj(u0), np.conj(u1))
    return seed, np.array([u0, u1, u2])


# ═══════════════════════════════════════════════════════════════════
#  Lattice
# ═══════════════════════════════════════════════════════════════════

class Lattice:
    def __init__(self, dims, beta):
        self.dims = dims
        self.beta = beta
        self.volume = dims[0] * dims[1] * dims[2] * dims[3]
        self.links = None  # [volume * 4] list of 3x3 complex arrays

    @staticmethod
    def cold_start(dims, beta):
        lat = Lattice(dims, beta)
        lat.links = [su3_identity() for _ in range(lat.volume * 4)]
        return lat

    @staticmethod
    def hot_start(dims, beta, seed):
        lat = Lattice(dims, beta)
        rng = np.uint64(seed)
        lat.links = []
        for _ in range(lat.volume * 4):
            rng, u = su3_random_near_identity(rng, 1.5)
            lat.links.append(u)
        return lat

    def site_index(self, x):
        return x[0] + self.dims[0] * (x[1] + self.dims[1] * (x[2] + self.dims[2] * x[3]))

    def site_coords(self, idx):
        x0 = idx % self.dims[0]
        rem = idx // self.dims[0]
        x1 = rem % self.dims[1]
        rem2 = rem // self.dims[1]
        x2 = rem2 % self.dims[2]
        x3 = rem2 // self.dims[2]
        return [x0, x1, x2, x3]

    def neighbor(self, x, mu, forward):
        y = list(x)
        if forward:
            y[mu] = (x[mu] + 1) % self.dims[mu]
        else:
            y[mu] = (x[mu] - 1) % self.dims[mu]
        return y

    def link(self, x, mu):
        idx = self.site_index(x)
        return self.links[idx * 4 + mu]


# ═══════════════════════════════════════════════════════════════════
#  Staggered Dirac Operator
# ═══════════════════════════════════════════════════════════════════

def staggered_phase(x, mu):
    s = sum(x[:mu])
    return 1.0 if s % 2 == 0 else -1.0


def apply_dirac(lattice, psi, mass):
    """D_st ψ — matches Rust apply_dirac() exactly."""
    vol = lattice.volume
    result = np.zeros((vol, 3), dtype=np.complex128)

    for idx in range(vol):
        x = lattice.site_coords(idx)
        out = mass * psi[idx]

        for mu in range(4):
            eta = staggered_phase(x, mu)
            half_eta = 0.5 * eta

            x_fwd = lattice.neighbor(x, mu, True)
            idx_fwd = lattice.site_index(x_fwd)
            u_fwd = lattice.link(x, mu)
            fwd = u_fwd @ psi[idx_fwd]

            x_bwd = lattice.neighbor(x, mu, False)
            idx_bwd = lattice.site_index(x_bwd)
            u_bwd = lattice.link(x_bwd, mu)
            bwd = u_bwd.conj().T @ psi[idx_bwd]

            out += half_eta * (fwd - bwd)

        result[idx] = out

    return result


def apply_dirac_adjoint(lattice, psi, mass):
    """D† — same as D but with flipped hopping sign."""
    vol = lattice.volume
    result = np.zeros((vol, 3), dtype=np.complex128)

    for idx in range(vol):
        x = lattice.site_coords(idx)
        out = mass * psi[idx]

        for mu in range(4):
            eta = staggered_phase(x, mu)
            half_eta = 0.5 * eta

            x_fwd = lattice.neighbor(x, mu, True)
            idx_fwd = lattice.site_index(x_fwd)
            u_fwd = lattice.link(x, mu)
            fwd = u_fwd @ psi[idx_fwd]

            x_bwd = lattice.neighbor(x, mu, False)
            idx_bwd = lattice.site_index(x_bwd)
            u_bwd = lattice.link(x_bwd, mu)
            bwd = u_bwd.conj().T @ psi[idx_bwd]

            out -= half_eta * (fwd - bwd)

        result[idx] = out

    return result


def apply_dirac_sq(lattice, psi, mass):
    """D†D — positive definite for CG."""
    dpsi = apply_dirac(lattice, psi, mass)
    return apply_dirac_adjoint(lattice, dpsi, mass)


# ═══════════════════════════════════════════════════════════════════
#  Conjugate Gradient Solver
# ═══════════════════════════════════════════════════════════════════

def fermion_dot(a, b):
    """<a|b> = Σ conj(a) * b"""
    return np.sum(np.conj(a) * b)


def fermion_norm_sq(a):
    return fermion_dot(a, a).real


def cg_solve(lattice, b, mass, tol, max_iter):
    """Solve D†D x = b using standard CG."""
    vol = lattice.volume
    x = np.zeros((vol, 3), dtype=np.complex128)

    ax = apply_dirac_sq(lattice, x, mass)
    r = b - ax

    b_norm_sq = fermion_norm_sq(b)
    if b_norm_sq < 1e-30:
        return x, 0, 0.0

    r_norm_sq = fermion_norm_sq(r)
    tol_sq = tol * tol * b_norm_sq

    if r_norm_sq < tol_sq:
        return x, 0, (r_norm_sq / b_norm_sq) ** 0.5

    p = r.copy()
    iterations = 0

    for it in range(max_iter):
        iterations = it + 1

        ap = apply_dirac_sq(lattice, p, mass)
        p_ap = fermion_dot(p, ap).real

        if abs(p_ap) < 1e-30:
            break

        alpha = r_norm_sq / p_ap
        x += alpha * p
        r -= alpha * ap

        r_norm_sq_new = fermion_norm_sq(r)
        if r_norm_sq_new < tol_sq:
            r_norm_sq = r_norm_sq_new
            break

        beta = r_norm_sq_new / r_norm_sq
        r_norm_sq = r_norm_sq_new
        p = r + beta * p

    final_res = (r_norm_sq / b_norm_sq) ** 0.5
    return x, iterations, final_res


# ═══════════════════════════════════════════════════════════════════
#  Random Fermion Field — matches Rust FermionField::random()
# ═══════════════════════════════════════════════════════════════════

def random_fermion(volume, seed):
    rng = np.uint64(seed)
    data = np.zeros((volume, 3), dtype=np.complex128)
    for i in range(volume):
        for c in range(3):
            rng, re = lcg_uniform(rng)
            rng, im = lcg_uniform(rng)
            data[i, c] = complex(re - 0.5, im - 0.5)
    return data


# ═══════════════════════════════════════════════════════════════════
#  Main — Validation + Benchmark
# ═══════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Lattice QCD CG — Python Control Baseline (NumPy)         ║")
    print("║  Staggered Dirac D†D x = b on 4⁴ lattice                 ║")
    print("║  Algorithm-identical to Rust: same LCG, same physics      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    checks_passed = 0
    checks_total = 0

    # ── Check 1: Cold lattice CG ─────────────────────────────────────
    print("═══ Check 1: Cold lattice CG (4⁴, mass=1.0, tol=1e-8) ══════")
    lat = Lattice.cold_start([4, 4, 4, 4], 6.0)
    b = random_fermion(lat.volume, 42)

    t0 = time.perf_counter()
    x, iters, res = cg_solve(lat, b, 1.0, 1e-8, 500)
    t1 = time.perf_counter()
    dt_cold = t1 - t0

    print(f"  iters={iters}, residual={res:.2e}, time={dt_cold*1000:.1f}ms")
    checks_total += 1
    if res < 1e-8:
        print("  ✓ Converged")
        checks_passed += 1
    else:
        print("  ✗ FAILED to converge")

    # Verify solution
    ax = apply_dirac_sq(lat, x, 1.0)
    diff = np.max(np.abs(ax - b))
    print(f"  max |D†D x - b|: {diff:.2e}")
    checks_total += 1
    if diff < 1e-6:
        print("  ✓ Solution verified")
        checks_passed += 1
    else:
        print("  ✗ Solution verification FAILED")

    # ── Check 2: Hot lattice CG ──────────────────────────────────────
    print()
    print("═══ Check 2: Hot lattice CG (4⁴, mass=0.5, tol=1e-6) ═══════")
    lat_hot = Lattice.hot_start([4, 4, 4, 4], 6.0, 42)
    b_hot = random_fermion(lat_hot.volume, 99)

    t0 = time.perf_counter()
    x_hot, iters_hot, res_hot = cg_solve(lat_hot, b_hot, 0.5, 1e-6, 2000)
    t1 = time.perf_counter()
    dt_hot = t1 - t0

    print(f"  iters={iters_hot}, residual={res_hot:.2e}, time={dt_hot*1000:.1f}ms")
    checks_total += 1
    if res_hot < 1e-6:
        print("  ✓ Converged")
        checks_passed += 1
    else:
        print("  ✗ FAILED to converge")

    # ── Check 3: D on cold lattice ───────────────────────────────────
    print()
    print("═══ Check 3: Dirac operator on cold lattice (sanity) ════════")
    psi = random_fermion(lat.volume, 77)
    dpsi = apply_dirac(lat, psi, 0.5)
    norm_dpsi = fermion_norm_sq(dpsi) ** 0.5
    print(f"  ||D ψ|| = {norm_dpsi:.6f} (should be > 0)")
    checks_total += 1
    if norm_dpsi > 0:
        print("  ✓ Nonzero")
        checks_passed += 1
    else:
        print("  ✗ FAILED")

    # ── Check 4: D†D positive definite ───────────────────────────────
    print()
    print("═══ Check 4: D†D positive definite ══════════════════════════")
    ddpsi = apply_dirac_sq(lat, psi, 0.5)
    inner = fermion_dot(psi, ddpsi).real
    print(f"  <ψ|D†D|ψ> = {inner:.6e} (must be > 0)")
    checks_total += 1
    if inner > 0:
        print("  ✓ Positive definite")
        checks_passed += 1
    else:
        print("  ✗ FAILED")

    # ── Benchmark: Dirac apply timing ────────────────────────────────
    print()
    print("═══ Benchmark: Dirac apply timing (4⁴, 100 iterations) ═════")
    n_reps = 100
    t0 = time.perf_counter()
    for _ in range(n_reps):
        _ = apply_dirac(lat, psi, 0.5)
    t1 = time.perf_counter()
    dt_dirac = (t1 - t0) / n_reps * 1000
    print(f"  {n_reps} Dirac applies: {(t1-t0)*1000:.1f}ms total, {dt_dirac:.2f}ms/apply")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("═══ Summary ═══════════════════════════════════════════════════")
    print(f"  {checks_passed}/{checks_total} checks passed")
    print(f"  Cold CG: {iters} iters in {dt_cold*1000:.1f}ms")
    print(f"  Hot CG:  {iters_hot} iters in {dt_hot*1000:.1f}ms")
    print(f"  Dirac apply: {dt_dirac:.2f}ms/apply")
    print()
    print("  These are the Python baselines. Rust should match iterations")
    print("  and residuals exactly, while being significantly faster.")
    print()

    # JSON output for machine consumption
    results = {
        "cold_iters": iters,
        "cold_residual": res,
        "cold_time_ms": dt_cold * 1000,
        "hot_iters": iters_hot,
        "hot_residual": res_hot,
        "hot_time_ms": dt_hot * 1000,
        "dirac_apply_ms": dt_dirac,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
    }
    print(f"  JSON: {results}")

    sys.exit(0 if checks_passed == checks_total else 1)


if __name__ == "__main__":
    main()
