#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Dynamical Fermion QCD — Python Control Baseline (Paper 10)

Independent NumPy implementation of pseudofermion HMC for staggered QCD.
Extends the quenched control baseline (quenched_beta_scan.py) with:
  - Staggered Dirac operator D
  - CG solver for (D†D)x = b
  - Pseudofermion heat bath, action, and force
  - Combined gauge + fermion leapfrog

Algorithm-identical to Rust: same LCG PRNG, same Cayley matrix exp,
same staggered phases, same CG convergence criterion.

Heavy quarks (m=2.0) are used to keep the fermion force manageable on
a coarse 4^4 lattice without multi-timescale integration.

References:
  - Gottlieb et al., PRD 35, 2531 (1987) — pseudofermion HMC
  - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 8.1-8.3
"""

import numpy as np
import time
import json
import sys

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

    def average_polyakov_loop(self):
        ns = self.dims[:3]
        total = 0.0
        count = 0
        for ix in range(ns[0]):
            for iy in range(ns[1]):
                for iz in range(ns[2]):
                    prod = su3_identity()
                    for t in range(self.dims[3]):
                        prod = prod @ self.link([ix, iy, iz, t], 3)
                    total += abs(np.trace(prod) / 3.0)
                    count += 1
        return total / count


# ═══════════════════════════════════════════════════════════════════
#  Staggered Dirac Operator
# ═══════════════════════════════════════════════════════════════════

def staggered_phase(x, mu):
    """η_μ(x) = (−1)^{x_0 + ... + x_{μ−1}}"""
    s = sum(x[:mu])
    return 1.0 if s % 2 == 0 else -1.0


def apply_dirac(lattice, psi, mass):
    """(Dψ)_x = m ψ_x + (1/2) Σ_μ η_μ(x)[U_μ(x)ψ(x+μ) − U†_μ(x−μ)ψ(x−μ)]"""
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
    """D† = m + (1/2) Σ_μ η_μ(x)[−U†_μ(x−μ)·ψ(x−μ) + U_μ(x)ψ(x+μ)]†"""
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
    """Apply D†D to ψ."""
    return apply_dirac_adjoint(lattice, apply_dirac(lattice, psi, mass), mass)


def cg_solve(lattice, phi, mass, tol=1e-8, max_iter=5000):
    """CG for (D†D)x = φ. Returns (x, n_iter, converged)."""
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
#  Pseudofermion HMC
# ═══════════════════════════════════════════════════════════════════

def pseudofermion_heatbath(lattice, mass, seed):
    """Generate φ = D†η where η is Gaussian random."""
    vol = lattice.volume
    eta = []
    for _ in range(vol):
        c = np.zeros(3, dtype=np.complex128)
        for a in range(3):
            seed, re = lcg_gaussian(seed)
            seed, im = lcg_gaussian(seed)
            c[a] = complex(re, im)
        eta.append(c)
    phi = apply_dirac_adjoint(lattice, eta, mass)
    return seed, phi


def pseudofermion_action(lattice, phi, mass, cg_tol=1e-8, cg_max_iter=5000):
    """S_F = φ†(D†D)⁻¹φ = φ†·x where (D†D)x = φ."""
    vol = lattice.volume
    x, n_iter, converged = cg_solve(lattice, phi, mass, cg_tol, cg_max_iter)
    action = sum(np.vdot(phi[i], x[i]).real for i in range(vol))
    return action, x, n_iter, converged


def pseudofermion_force(lattice, x_field, mass):
    """Compute dS_F/dU_μ(x) — fermion force for all links."""
    vol = lattice.volume
    y_field = apply_dirac(lattice, x_field, mass)
    force = [np.zeros((3, 3), dtype=np.complex128) for _ in range(vol * 4)]

    for idx in range(vol):
        x = lattice.site_coords(idx)
        for mu in range(4):
            eta = staggered_phase(x, mu)
            fwd = lattice.neighbor(x, mu, True)
            fwd_idx = lattice.site_index(fwd)
            u = lattice.link(x, mu)

            m_mat = np.zeros((3, 3), dtype=np.complex128)
            for a in range(3):
                for b in range(3):
                    contrib = (x_field[fwd_idx][a] * np.conj(y_field[idx][b])
                               - y_field[fwd_idx][a] * np.conj(x_field[idx][b]))
                    m_mat[a, b] += contrib * eta * 0.5

            w = u @ m_mat
            wh = w.conj().T
            ta = 0.5 * (w - wh)
            tr = np.trace(ta)
            for a in range(3):
                ta[a, a] -= tr / 3.0

            force[idx * 4 + mu] = ta

    return force


# ═══════════════════════════════════════════════════════════════════
#  HMC Integration
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


def update_total_momenta(lattice, momenta, phi_fields, dt, mass, cg_tol):
    """Gauge force + fermion force."""
    vol = lattice.volume
    for idx in range(vol):
        x = lattice.site_coords(idx)
        for mu in range(4):
            momenta[idx * 4 + mu] += dt * lattice.gauge_force(x, mu)

    for phi in phi_fields:
        x_field, _, _ = cg_solve(lattice, phi, mass, cg_tol)
        ferm_force = pseudofermion_force(lattice, x_field, mass)
        for k in range(len(momenta)):
            momenta[k] += dt * ferm_force[k]


def dynamical_leapfrog(lattice, momenta, phi_fields, n_steps, dt, mass, cg_tol):
    vol = lattice.volume
    half_dt = 0.5 * dt

    update_total_momenta(lattice, momenta, phi_fields, half_dt, mass, cg_tol)

    for step in range(n_steps):
        for idx in range(vol):
            x = lattice.site_coords(idx)
            for mu in range(4):
                p = momenta[idx * 4 + mu]
                u = lattice.link(x, mu)
                lattice.set_link(x, mu, exp_su3_cayley(p, dt) @ u)

        p_dt = dt if step < n_steps - 1 else half_dt
        update_total_momenta(lattice, momenta, phi_fields, p_dt, mass, cg_tol)


def dynamical_hmc_trajectory(lattice, seed, n_md_steps, dt, mass, cg_tol=1e-8):
    """One dynamical fermion HMC trajectory."""
    vol = lattice.volume
    old_links = [u.copy() for u in lattice.links]

    seed, phi = pseudofermion_heatbath(lattice, mass, seed)
    phi_fields = [phi]

    gauge_before = lattice.wilson_action()
    sf_before, _, _, _ = pseudofermion_action(lattice, phi, mass, cg_tol)

    momenta = []
    for _ in range(vol * 4):
        seed, p = su3_random_algebra(seed)
        momenta.append(p)
    ke_before = kinetic_energy(momenta)
    h_old = ke_before + gauge_before + sf_before

    dynamical_leapfrog(lattice, momenta, phi_fields, n_md_steps, dt, mass, cg_tol)

    gauge_after = lattice.wilson_action()
    sf_after, _, _, _ = pseudofermion_action(lattice, phi, mass, cg_tol)
    ke_after = kinetic_energy(momenta)
    h_new = ke_after + gauge_after + sf_after

    delta_h = h_new - h_old

    if delta_h <= 0.0:
        accept = True
    else:
        seed, r = lcg_uniform(seed)
        accept = r < np.exp(-delta_h)

    if not accept:
        lattice.links = old_links

    plaq = lattice.average_plaquette()
    return seed, plaq, delta_h, accept, sf_after


def quenched_hmc_trajectory(lattice, seed, n_md_steps, dt):
    """One quenched HMC trajectory (for pre-thermalization)."""
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
                lattice.set_link(x, mu, exp_su3_cayley(momenta[idx * 4 + mu], dt) @ lattice.link(x, mu))
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
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Dynamical Fermion QCD — Python Control Baseline            ║")
    print("║  Pseudofermion HMC on 4⁴, staggered, heavy quarks (m=2.0)  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    dims = [4, 4, 4, 4]
    beta = 5.5
    mass = 2.0
    seed = 42

    t0 = time.perf_counter()

    # Phase 1: Quenched pre-thermalization
    print("═══ Phase 1: Quenched pre-thermalization (15 traj) ═══")
    lat = Lattice.hot_start(dims, beta, seed)
    for i in range(15):
        seed, plaq, dh, acc = quenched_hmc_trajectory(lat, seed, 15, 0.05)
        if i % 5 == 0 or i == 14:
            print(f"  traj {i:3d}: plaq={plaq:.6f}, ΔH={dh:+.4e}, {'ACC' if acc else 'REJ'}")
    quenched_plaq = lat.average_plaquette()
    print(f"  Quenched <P>: {quenched_plaq:.6f}")
    print()

    # Phase 2: Dynamical HMC
    print("═══ Phase 2: Dynamical HMC (10 therm + 12 meas) ═══")
    n_therm = 10
    n_meas = 12
    dt = 0.001
    n_steps = 100

    plaquettes = []
    sf_values = []
    accepted = 0

    for i in range(n_therm + n_meas):
        seed, plaq, dh, acc, sf = dynamical_hmc_trajectory(
            lat, seed, n_steps, dt, mass
        )
        label = "therm" if i < n_therm else "meas "
        if i % 5 == 0 or i == n_therm + n_meas - 1:
            print(f"  {label} {i:3d}: plaq={plaq:.6f}, S_F={sf:.2f}, "
                  f"ΔH={dh:+.4e}, {'ACC' if acc else 'REJ'}")

        if i >= n_therm:
            plaquettes.append(plaq)
            sf_values.append(sf)
            if acc:
                accepted += 1

    mean_plaq = np.mean(plaquettes)
    acc_rate = accepted / n_meas
    poly = lat.average_polyakov_loop()
    wall_s = time.perf_counter() - t0

    print()
    print("═══ Summary ═══")
    print(f"  Dynamical <P>: {mean_plaq:.6f}")
    print(f"  Quenched  <P>: {quenched_plaq:.6f}")
    print(f"  Shift: {abs(mean_plaq - quenched_plaq):.6f}")
    print(f"  Acceptance: {accepted}/{n_meas} ({acc_rate*100:.0f}%)")
    print(f"  Polyakov |L|: {poly:.6f}")
    print(f"  Mean S_F: {np.mean(sf_values):.2f}")
    print(f"  Wall time: {wall_s:.1f}s")

    checks_passed = 0
    checks_total = 0

    # Check 1: plaquette in (0, 1)
    checks_total += 1
    if all(0 < p < 1 for p in plaquettes):
        print("  ✓ All plaquettes in (0, 1)")
        checks_passed += 1
    else:
        print("  ✗ Plaquette out of range")

    # Check 2: S_F > 0
    checks_total += 1
    if all(s > 0 for s in sf_values):
        print("  ✓ All S_F > 0")
        checks_passed += 1
    else:
        print("  ✗ S_F not always positive")

    # Check 3: some acceptance
    checks_total += 1
    if accepted > 0:
        print(f"  ✓ At least one trajectory accepted ({accepted})")
        checks_passed += 1
    else:
        print("  ✗ No trajectories accepted")

    print(f"\n  {checks_passed}/{checks_total} checks passed")

    output = {
        "lattice_dims": dims,
        "beta": beta,
        "mass": mass,
        "dt": dt,
        "n_md_steps": n_steps,
        "n_thermalization": n_therm,
        "n_measurement": n_meas,
        "dynamical_plaquette": float(mean_plaq),
        "quenched_plaquette": float(quenched_plaq),
        "acceptance_rate": float(acc_rate),
        "polyakov_loop": float(poly),
        "wall_time_s": float(wall_s),
        "checks_passed": checks_passed,
        "checks_total": checks_total,
    }

    import os
    out_path = "../results/dynamical_fermion_control.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    sys.exit(0 if checks_passed == checks_total else 1)


if __name__ == "__main__":
    main()
