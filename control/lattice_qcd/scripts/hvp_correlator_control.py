#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Hadronic Vacuum Polarization (HVP) for muon g-2 — Python Control Baseline (Paper 11)

Independent NumPy implementation of the lattice HVP pipeline:
  1. Thermalize quenched gauge configs with leapfrog HMC
  2. Point-to-all staggered propagator via CG solve of (D†D)x = src
  3. Time-slice correlator C(t) = Σ_{x spatial} Σ_c |G_c(x,t)|²
  4. HVP kernel K(t) ∝ t²(T/2 - t)² / T⁴  (Bernecker-Meyer discretization)
  5. HVP integral a_μ ∝ Σ_t K(t) C(t)

Uses the same SU(3) infrastructure as quenched_beta_scan.py and
dynamical_fermion_control.py.  Algorithm-identical to Rust: same LCG
PRNG, same Cayley matrix exp, same staggered phases, same CG.

Quenched approximation on 4⁴ for speed; the Rust validator runs 8⁴.
Physics quantities (correlator shape, HVP sign, mass ordering) are
lattice-size independent, so 4⁴ is a valid baseline.

References:
  - Bernecker & Meyer, EPJA 47, 148 (2011) — time-momentum representation
  - Blum, PRL 91, 052001 (2003) — lattice HVP method
  - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 8
"""

import numpy as np
import time
import json
import sys
import os

# ═══════════════════════════════════════════════════════════════════
#  LCG PRNG — matches Rust lattice/constants.rs exactly
# ═══════════════════════════════════════════════════════════════════

LCG_A = 6_364_136_223_846_793_005
LCG_C = 1_442_695_040_888_963_407
LCG_MOD = 1 << 64
LCG_53_DIVISOR = float(1 << 53)
LATTICE_DIVISION_GUARD = 1e-15


def lcg_step(seed):
    return (seed * LCG_A + LCG_C) % LCG_MOD


def lcg_uniform(seed):
    seed = lcg_step(seed)
    return seed, float(seed >> 11) / LCG_53_DIVISOR


def lcg_gaussian(seed):
    seed, u1 = lcg_uniform(seed)
    seed, u2 = lcg_uniform(seed)
    r = np.sqrt(-2.0 * np.log(max(u1, LATTICE_DIVISION_GUARD)))
    theta = 2.0 * np.pi * u2
    return seed, float(r * np.cos(theta))


# ═══════════════════════════════════════════════════════════════════
#  SU(3) Matrix Operations
# ═══════════════════════════════════════════════════════════════════

def su3_identity():
    return np.eye(3, dtype=np.complex128)


def su3_random_near_identity(seed, epsilon):
    m = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            seed, re = lcg_uniform(seed)
            seed, im = lcg_uniform(seed)
            m[i, j] = complex((re - 0.5) * epsilon, (im - 0.5) * epsilon)
    m = np.eye(3, dtype=np.complex128) + m
    u0 = m[0] / np.linalg.norm(m[0])
    u1 = m[1] - np.dot(np.conj(u0), m[1]) * u0
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(np.conj(u0), np.conj(u1))
    return seed, np.array([u0, u1, u2])


def su3_random_algebra(seed):
    scale = 1.0 / np.sqrt(2.0)
    coeffs = []
    for _ in range(8):
        seed, g = lcg_gaussian(seed)
        coeffs.append(scale * g)
    a3, a8, re01, im01, re02, im02, re12, im12 = coeffs
    sqrt3 = np.sqrt(3.0)
    h = np.zeros((3, 3), dtype=np.complex128)
    h[0, 0] = complex(a3 + a8 / sqrt3, 0.0)
    h[1, 1] = complex(-a3 + a8 / sqrt3, 0.0)
    h[2, 2] = complex(-2.0 * a8 / sqrt3, 0.0)
    h[0, 1] = complex(re01, im01)
    h[1, 0] = complex(re01, -im01)
    h[0, 2] = complex(re02, im02)
    h[2, 0] = complex(re02, -im02)
    h[1, 2] = complex(re12, im12)
    h[2, 1] = complex(re12, -im12)
    return seed, 1j * h


# ═══════════════════════════════════════════════════════════════════
#  Lattice
# ═══════════════════════════════════════════════════════════════════

class Lattice:
    def __init__(self, dims, beta):
        self.dims = list(dims)
        self.beta = beta
        self.volume = dims[0] * dims[1] * dims[2] * dims[3]
        self.links = None

    @staticmethod
    def hot_start(dims, beta, seed):
        lat = Lattice(dims, beta)
        rng = int(seed)
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

    def set_link(self, x, mu, u):
        idx = self.site_index(x)
        self.links[idx * 4 + mu] = u

    def plaquette(self, x, mu, nu):
        x_mu = self.neighbor(x, mu, True)
        x_nu = self.neighbor(x, nu, True)
        return (self.link(x, mu) @ self.link(x_mu, nu)
                @ self.link(x_nu, mu).conj().T @ self.link(x, nu).conj().T)

    def average_plaquette(self):
        total = 0.0
        count = 0
        for idx in range(self.volume):
            x = self.site_coords(idx)
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    total += np.trace(self.plaquette(x, mu, nu)).real / 3.0
                    count += 1
        return total / count

    def staple(self, x, mu):
        s = np.zeros((3, 3), dtype=np.complex128)
        x_mu = self.neighbor(x, mu, True)
        for nu in range(4):
            if nu == mu:
                continue
            x_nu = self.neighbor(x, nu, True)
            x_mu_bnu = self.neighbor(x_mu, nu, False)
            x_bnu = self.neighbor(x, nu, False)
            upper = (self.link(x_mu, nu) @ self.link(x_nu, mu).conj().T
                     @ self.link(x, nu).conj().T)
            lower = (self.link(x_mu_bnu, nu).conj().T
                     @ self.link(x_bnu, mu).conj().T @ self.link(x_bnu, nu))
            s += upper + lower
        return s

    def wilson_action(self):
        total = 0.0
        for idx in range(self.volume):
            x = self.site_coords(idx)
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    total += 1.0 - np.trace(self.plaquette(x, mu, nu)).real / 3.0
        return self.beta * total

    def gauge_force(self, x, mu):
        u = self.link(x, mu)
        v = self.staple(x, mu)
        w = u @ v
        wd = w.conj().T
        diff = 0.5 * (w - wd)
        tr = np.trace(diff)
        for i in range(3):
            diff[i, i] -= tr / 3.0
        return -self.beta / 3.0 * diff


# ═══════════════════════════════════════════════════════════════════
#  HMC
# ═══════════════════════════════════════════════════════════════════

def exp_su3_cayley(p, dt):
    half = 0.5 * dt * p
    plus = np.eye(3, dtype=np.complex128) + half
    minus = np.eye(3, dtype=np.complex128) - half
    result = plus @ np.linalg.inv(minus)
    u0 = result[0] / np.linalg.norm(result[0])
    u1 = result[1] - np.dot(np.conj(u0), result[1]) * u0
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(np.conj(u0), np.conj(u1))
    return np.array([u0, u1, u2])


def kinetic_energy(momenta):
    t = 0.0
    for p in momenta:
        t -= 0.5 * np.trace(p @ p).real
    return t


def hmc_trajectory(lattice, seed, n_md_steps, dt):
    vol = lattice.volume
    old_links = [u.copy() for u in lattice.links]
    action_before = lattice.wilson_action()

    momenta = []
    for _ in range(vol * 4):
        seed, p = su3_random_algebra(seed)
        momenta.append(p)
    ke_before = kinetic_energy(momenta)
    h_old = action_before + ke_before

    half_dt = 0.5 * dt
    for idx in range(vol):
        x = lattice.site_coords(idx)
        for mu in range(4):
            momenta[idx * 4 + mu] += half_dt * lattice.gauge_force(x, mu)

    for step in range(n_md_steps):
        for idx in range(vol):
            x = lattice.site_coords(idx)
            for mu in range(4):
                lattice.set_link(x, mu,
                                 exp_su3_cayley(momenta[idx * 4 + mu], dt) @ lattice.link(x, mu))
        p_dt = dt if step < n_md_steps - 1 else half_dt
        for idx in range(vol):
            x = lattice.site_coords(idx)
            for mu in range(4):
                momenta[idx * 4 + mu] += p_dt * lattice.gauge_force(x, mu)

    ke_after = kinetic_energy(momenta)
    h_new = lattice.wilson_action() + ke_after
    delta_h = h_new - h_old

    if delta_h <= 0.0:
        accept = True
    else:
        seed, r = lcg_uniform(seed)
        accept = r < np.exp(-delta_h)

    if not accept:
        lattice.links = old_links

    return seed, lattice.average_plaquette(), delta_h, accept


# ═══════════════════════════════════════════════════════════════════
#  Staggered Dirac Operator + CG
# ═══════════════════════════════════════════════════════════════════

def staggered_phase(x, mu):
    s = sum(x[:mu])
    return 1.0 if s % 2 == 0 else -1.0


def apply_dirac(lattice, psi, mass):
    vol = lattice.volume
    result = [mass * psi[i].copy() for i in range(vol)]
    for idx in range(vol):
        x = lattice.site_coords(idx)
        for mu in range(4):
            eta = staggered_phase(x, mu)
            fwd = lattice.neighbor(x, mu, True)
            bwd = lattice.neighbor(x, mu, False)
            fwd_idx = lattice.site_index(fwd)
            bwd_idx = lattice.site_index(bwd)
            result[idx] += 0.5 * eta * (lattice.link(x, mu) @ psi[fwd_idx])
            result[idx] -= 0.5 * eta * (lattice.link(bwd, mu).conj().T @ psi[bwd_idx])
    return result


def apply_dirac_adjoint(lattice, psi, mass):
    vol = lattice.volume
    result = [mass * psi[i].copy() for i in range(vol)]
    for idx in range(vol):
        x = lattice.site_coords(idx)
        for mu in range(4):
            eta = staggered_phase(x, mu)
            fwd = lattice.neighbor(x, mu, True)
            bwd = lattice.neighbor(x, mu, False)
            fwd_idx = lattice.site_index(fwd)
            bwd_idx = lattice.site_index(bwd)
            result[idx] -= 0.5 * eta * (lattice.link(x, mu) @ psi[fwd_idx])
            result[idx] += 0.5 * eta * (lattice.link(bwd, mu).conj().T @ psi[bwd_idx])
    return result


def apply_ddag_d(lattice, psi, mass):
    return apply_dirac_adjoint(lattice, apply_dirac(lattice, psi, mass), mass)


def cg_solve(lattice, phi, mass, tol=1e-8, max_iter=5000):
    vol = lattice.volume
    x = [np.zeros(3, dtype=np.complex128) for _ in range(vol)]
    r = [phi[i].copy() for i in range(vol)]
    p = [r[i].copy() for i in range(vol)]

    rr = sum(np.vdot(r[i], r[i]).real for i in range(vol))
    phi_norm = sum(np.vdot(phi[i], phi[i]).real for i in range(vol))

    for it in range(max_iter):
        ap = apply_ddag_d(lattice, p, mass)
        pap = sum(np.vdot(p[i], ap[i]).real for i in range(vol))
        if abs(pap) < 1e-30:
            break
        alpha = rr / pap
        for i in range(vol):
            x[i] += alpha * p[i]
            r[i] -= alpha * ap[i]
        rr_new = sum(np.vdot(r[i], r[i]).real for i in range(vol))
        if rr_new / max(phi_norm, 1e-30) < tol * tol:
            return x, it + 1, True
        beta = rr_new / max(rr, 1e-30)
        for i in range(vol):
            p[i] = r[i] + beta * p[i]
        rr = rr_new

    return x, max_iter, False


# ═══════════════════════════════════════════════════════════════════
#  HVP Correlator + Kernel
# ═══════════════════════════════════════════════════════════════════

def point_propagator_correlator(lattice, mass, cg_tol=1e-8, cg_max=5000):
    """Compute C(t) = Σ_{x spatial} Σ_c |G_c(x,t)|² from point source at origin."""
    vol = lattice.volume
    nt = lattice.dims[3]

    source = [np.zeros(3, dtype=np.complex128) for _ in range(vol)]
    source[0][0] = 1.0 + 0.0j

    x_sol, n_iter, converged = cg_solve(lattice, source, mass, cg_tol, cg_max)

    corr = np.zeros(nt)
    for idx in range(vol):
        t = lattice.site_coords(idx)[3]
        corr[t] += np.vdot(x_sol[idx], x_sol[idx]).real

    return corr, n_iter, converged


def hvp_kernel(t, nt):
    """K(t) ∝ t²(T/2 - t)² / T⁴ — Bernecker-Meyer discretization."""
    half = nt // 2
    if t == 0 or t >= half:
        return 0.0
    tf = t / half
    return tf * tf * (1.0 - tf) * (1.0 - tf)


def hvp_integral(correlator):
    """a_μ^HVP ∝ Σ_t K(t) C(t)"""
    nt = len(correlator)
    return sum(hvp_kernel(t, nt) * correlator[t] for t in range(1, nt // 2))


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("+" + "=" * 62 + "+")
    print("|  HVP g-2 Correlator -- Python Control Baseline (Paper 11)    |")
    print("|  Point-source staggered propagator -> HVP integral on 4^4    |")
    print("+" + "=" * 62 + "+")
    print()

    dims = [4, 4, 4, 4]
    beta = 6.0
    mass = 0.1
    seed = 42
    n_therm = 30
    n_md_steps = 15
    dt = 0.05

    t0 = time.perf_counter()

    # Phase 1: Thermalize
    print("=== Phase 1: Thermalize (30 HMC trajectories, 4^4, beta=6.0) ===")
    lat = Lattice.hot_start(dims, beta, seed)
    for i in range(n_therm):
        seed, plaq, dh, acc = hmc_trajectory(lat, seed, n_md_steps, dt)
        if i % 10 == 0 or i == n_therm - 1:
            print(f"  traj {i:3d}: plaq={plaq:.6f}, dH={dh:+.4e}, {'ACC' if acc else 'REJ'}")
    therm_plaq = lat.average_plaquette()
    print(f"  Thermalized <P>: {therm_plaq:.6f}")
    print()

    # Phase 2: Single-config HVP
    print("=== Phase 2: Single-config HVP (m=0.1, CG tol=1e-8) ===")
    corr, cg_iter, cg_conv = point_propagator_correlator(lat, mass)
    nt = dims[3]
    print(f"  CG: {cg_iter} iters, converged={cg_conv}")
    print(f"  C(t): {[f'{c:.6e}' for c in corr]}")

    hvp_val = hvp_integral(corr)
    print(f"  HVP integral: {hvp_val:.6e}")
    print()

    # Phase 3: Multi-config average (5 configs, 5 HMC sweeps between)
    print("=== Phase 3: Multi-config HVP average (5 configs) ===")
    n_configs = 5
    hvp_values = [hvp_val]
    correlators = [corr.tolist()]
    all_cg_ok = cg_conv

    for i in range(1, n_configs):
        for _ in range(5):
            seed, _, _, _ = hmc_trajectory(lat, seed, n_md_steps, dt)
        c, n_it, conv = point_propagator_correlator(lat, mass)
        h = hvp_integral(c)
        hvp_values.append(h)
        correlators.append(c.tolist())
        if not conv:
            all_cg_ok = False
        print(f"  Config {i+1}: HVP={h:.6e}, CG iters={n_it}")

    hvp_mean = np.mean(hvp_values)
    hvp_std = np.std(hvp_values, ddof=1) if len(hvp_values) > 1 else 0.0
    print(f"  HVP mean: {hvp_mean:.6e} +/- {hvp_std:.6e}")
    print()

    # Phase 4: Mass dependence (m=0.1 vs m=1.0)
    print("=== Phase 4: Mass dependence (m=0.1 vs m=1.0) ===")
    corr_heavy, _, conv_heavy = point_propagator_correlator(lat, 1.0)
    hvp_heavy = hvp_integral(corr_heavy)
    print(f"  m=0.1: HVP={hvp_mean:.6e}")
    print(f"  m=1.0: HVP={hvp_heavy:.6e}")
    print()

    # Phase 5: Kernel shape
    print("=== Phase 5: HVP kernel shape ===")
    kernel = [hvp_kernel(t, nt) for t in range(nt)]
    k_max = max(kernel)
    k_max_t = kernel.index(k_max)
    print(f"  K(t) max at t={k_max_t}: {k_max:.6e}")
    print(f"  K(0)={kernel[0]:.6e}, K(T/2)={kernel[nt//2]:.6e}")
    print()

    wall_s = time.perf_counter() - t0

    # Validation checks
    checks_passed = 0
    checks_total = 0

    checks_total += 1
    if cg_conv:
        print("  [PASS] CG converges on single config")
        checks_passed += 1
    else:
        print("  [FAIL] CG did not converge")

    checks_total += 1
    if all(c >= 0.0 for c in corr):
        print("  [PASS] C(t) all positive")
        checks_passed += 1
    else:
        print("  [FAIL] C(t) has negative values")

    checks_total += 1
    monotone = all(corr[t] <= corr[t-1] + 1e-12 for t in range(1, nt // 2))
    if monotone:
        print("  [PASS] C(t) monotone decreasing for t=1..T/2")
        checks_passed += 1
    else:
        print("  [FAIL] C(t) not monotone")

    checks_total += 1
    if hvp_val > 0:
        print("  [PASS] HVP integral > 0")
        checks_passed += 1
    else:
        print("  [FAIL] HVP integral not positive")

    checks_total += 1
    if all_cg_ok:
        print("  [PASS] All CG converge across configs")
        checks_passed += 1
    else:
        print("  [FAIL] CG failed on some config")

    checks_total += 1
    if all(h > 0 for h in hvp_values):
        print("  [PASS] All HVP values positive")
        checks_passed += 1
    else:
        print("  [FAIL] Some HVP values negative")

    checks_total += 1
    if hvp_mean > hvp_heavy:
        print("  [PASS] Lighter quarks give larger HVP (m=0.1 > m=1.0)")
        checks_passed += 1
    else:
        print("  [FAIL] Mass dependence wrong")

    checks_total += 1
    if k_max_t > 0 and k_max_t < nt // 2:
        print("  [PASS] HVP kernel peaks in middle")
        checks_passed += 1
    else:
        print("  [FAIL] HVP kernel peak location wrong")

    print(f"\n  {checks_passed}/{checks_total} checks passed")
    print(f"  Wall time: {wall_s:.1f}s")

    output = {
        "lattice_dims": dims,
        "beta": beta,
        "mass": mass,
        "n_thermalization": n_therm,
        "n_configs": n_configs,
        "plaquette": float(therm_plaq),
        "correlator": corr.tolist(),
        "correlators": correlators,
        "hvp_value": float(hvp_val),
        "hvp_values": [float(h) for h in hvp_values],
        "hvp_mean": float(hvp_mean),
        "hvp_std": float(hvp_std),
        "hvp_heavy": float(hvp_heavy),
        "kernel": kernel,
        "cg_iterations": cg_iter,
        "cg_converged": cg_conv,
        "mass_ordering_correct": bool(hvp_mean > hvp_heavy),
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "wall_time_s": float(wall_s),
    }

    out_path = "../results/hvp_correlator_control.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    sys.exit(0 if checks_passed == checks_total else 1)


if __name__ == "__main__":
    main()
