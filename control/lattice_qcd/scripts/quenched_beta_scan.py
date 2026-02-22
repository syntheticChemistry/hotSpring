#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Quenched SU(3) β-Scan — Python Control Baseline

Independent NumPy implementation of HMC for pure gauge SU(3) lattice
theory. Produces plaquette and Polyakov loop as functions of β (inverse
bare coupling) to establish the Python baseline for Rust/GPU validation.

Algorithm-identical to Rust: same LCG PRNG, same Cayley matrix exp,
same leapfrog integrator, same Metropolis step.

The β-scan crosses the deconfinement transition at β_c ≈ 5.69 (4^4).
Below β_c, |L| ≈ 0 (confined); above β_c, |L| > 0 (deconfined).

References:
  - Wilson, PRD 10, 2445 (1974) — Wilson gauge action
  - Creutz, PRD 21, 2308 (1980) — SU(3) Monte Carlo
  - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 3, 8
  - HotQCD, PRD 90, 094503 (2014) — 2+1 flavor EOS
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


def lcg_step(seed):
    return (seed * LCG_A + LCG_C) % LCG_MOD


LATTICE_DIVISION_GUARD = 1e-15


def lcg_uniform(seed):
    seed = lcg_step(seed)
    return seed, float(seed >> 11) / LCG_53_DIVISOR


def lcg_gaussian(seed):
    """Box-Muller Gaussian deviate — matches Rust lcg_gaussian exactly."""
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
    """Generate random SU(3) near identity — matches Rust."""
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
    """Random traceless anti-Hermitian 3x3 via Gell-Mann basis — matches Rust."""
    scale = 1.0 / np.sqrt(2.0)

    def rg():
        nonlocal seed
        seed, g = lcg_gaussian(seed)
        return scale * g

    h = np.zeros((3, 3), dtype=np.complex128)

    a3 = rg()
    a8 = rg()
    sqrt3 = np.sqrt(3.0)
    h[0, 0] = complex(a3 + a8 / sqrt3, 0.0)
    h[1, 1] = complex(-a3 + a8 / sqrt3, 0.0)
    h[2, 2] = complex(-2.0 * a8 / sqrt3, 0.0)

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        re = rg()
        im = rg()
        h[i, j] = complex(re, im)
        h[j, i] = complex(re, -im)

    # Return iH (anti-Hermitian, traceless)
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
    def cold_start(dims, beta):
        lat = Lattice(dims, beta)
        lat.links = [su3_identity() for _ in range(lat.volume * 4)]
        return lat

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
        u1 = self.link(x, mu)
        u2 = self.link(x_mu, nu)
        u3 = self.link(x_nu, mu).conj().T
        u4 = self.link(x, nu).conj().T
        return u1 @ u2 @ u3 @ u4

    def average_plaquette(self):
        total = 0.0
        count = 0
        for idx in range(self.volume):
            x = self.site_coords(idx)
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    p = self.plaquette(x, mu, nu)
                    total += np.trace(p).real / 3.0
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
            upper = self.link(x_mu, nu) @ self.link(x_nu, mu).conj().T @ self.link(x, nu).conj().T
            lower = self.link(x_mu_bnu, nu).conj().T @ self.link(x_bnu, mu).conj().T @ self.link(x_bnu, nu)
            s += upper + lower
        return s

    def wilson_action(self):
        total = 0.0
        for idx in range(self.volume):
            x = self.site_coords(idx)
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    p = self.plaquette(x, mu, nu)
                    total += 1.0 - np.trace(p).real / 3.0
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

    def polyakov_loop(self, x_spatial):
        nt = self.dims[3]
        prod = su3_identity()
        for t in range(nt):
            x = [x_spatial[0], x_spatial[1], x_spatial[2], t]
            prod = prod @ self.link(x, 3)
        return np.trace(prod) / 3.0

    def average_polyakov_loop(self):
        ns = self.dims[:3]
        spatial_vol = ns[0] * ns[1] * ns[2]
        total = 0.0
        for ix in range(ns[0]):
            for iy in range(ns[1]):
                for iz in range(ns[2]):
                    l = self.polyakov_loop([ix, iy, iz])
                    total += abs(l)
        return total / spatial_vol


# ═══════════════════════════════════════════════════════════════════
#  HMC
# ═══════════════════════════════════════════════════════════════════

def exp_su3_cayley(p, dt):
    """Cayley matrix exponential: (I + X/2)(I - X/2)^{-1}."""
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
        p2 = p @ p
        t -= 0.5 * np.trace(p2).real
    return t


def update_momenta(lattice, momenta, dt):
    for idx in range(lattice.volume):
        x = lattice.site_coords(idx)
        for mu in range(4):
            f = lattice.gauge_force(x, mu)
            momenta[idx * 4 + mu] += dt * f


def leapfrog(lattice, momenta, n_steps, dt):
    half_dt = 0.5 * dt
    update_momenta(lattice, momenta, half_dt)
    for step in range(n_steps):
        for idx in range(lattice.volume):
            x = lattice.site_coords(idx)
            for mu in range(4):
                p = momenta[idx * 4 + mu]
                u = lattice.link(x, mu)
                exp_p = exp_su3_cayley(p, dt)
                lattice.set_link(x, mu, exp_p @ u)
        p_dt = dt if step < n_steps - 1 else half_dt
        update_momenta(lattice, momenta, p_dt)


def hmc_trajectory(lattice, seed, n_md_steps, dt):
    vol = lattice.volume
    import copy
    old_links = [u.copy() for u in lattice.links]
    action_before = lattice.wilson_action()

    momenta = []
    for _ in range(vol * 4):
        seed, p = su3_random_algebra(seed)
        momenta.append(p)

    ke_before = kinetic_energy(momenta)
    h_old = action_before + ke_before

    leapfrog(lattice, momenta, n_md_steps, dt)

    action_after = lattice.wilson_action()
    ke_after = kinetic_energy(momenta)
    h_new = action_after + ke_after
    delta_h = h_new - h_old

    if delta_h <= 0.0:
        accept = True
    else:
        seed, r = lcg_uniform(seed)
        accept = r < np.exp(-delta_h)

    if not accept:
        lattice.links = old_links

    plaq = lattice.average_plaquette()
    return seed, plaq, delta_h, accept


def run_beta_scan(dims, beta_values, n_therm, n_traj, n_md_steps, dt, seed_base):
    """Run independent HMC at each β and collect observables."""
    results = []
    for i, beta in enumerate(beta_values):
        seed = int(seed_base + i)
        lat = Lattice.hot_start(dims, beta, seed_base + i)
        plaquettes = []
        accepted = 0

        t0 = time.perf_counter()

        for traj in range(n_therm + n_traj):
            seed, plaq, dh, acc = hmc_trajectory(lat, seed, n_md_steps, dt)
            if traj >= n_therm:
                plaquettes.append(plaq)
                if acc:
                    accepted += 1

            if traj % 10 == 0:
                print(f"  β={beta:.2f} traj {traj}: plaq={plaq:.6f}, ΔH={dh:.4e}, {'ACC' if acc else 'REJ'}")

        poly = lat.average_polyakov_loop()
        wall_s = time.perf_counter() - t0
        mean_plaq = np.mean(plaquettes)
        std_plaq = np.std(plaquettes, ddof=1) if len(plaquettes) > 1 else 0.0
        acc_rate = accepted / n_traj if n_traj > 0 else 0.0

        results.append({
            "beta": beta,
            "mean_plaquette": float(mean_plaq),
            "std_plaquette": float(std_plaq),
            "polyakov_loop": float(poly),
            "acceptance_rate": float(acc_rate),
            "wall_time_s": float(wall_s),
            "n_trajectories": n_traj,
            "n_thermalization": n_therm,
        })
        print(f"  β={beta:.2f}: <plaq>={mean_plaq:.6f}±{std_plaq:.6f}, |L|={poly:.6f}, acc={acc_rate:.1%}, {wall_s:.1f}s\n")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Quenched SU(3) β-Scan — Python Control Baseline           ║")
    print("║  Pure gauge HMC on 4⁴ lattice across deconfinement         ║")
    print("║  β_c ≈ 5.69 for SU(3) on N_t = 4                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    dims = [4, 4, 4, 4]
    beta_values = [4.0, 4.5, 5.0, 5.5, 5.7, 5.8, 6.0, 6.2, 6.5]
    n_therm = 20
    n_traj = 30
    n_md_steps = 15
    dt = 0.05
    seed_base = 42

    results = run_beta_scan(dims, beta_values, n_therm, n_traj, n_md_steps, dt, seed_base)

    print("═══ Summary ════════════════════════════════════════════════════")
    print(f"  {'β':>6} {'<plaq>':>10} {'σ(plaq)':>10} {'|L|':>10} {'acc%':>8}")
    for r in results:
        print(f"  {r['beta']:>6.2f} {r['mean_plaquette']:>10.6f} {r['std_plaquette']:>10.6f} "
              f"{r['polyakov_loop']:>10.6f} {r['acceptance_rate']*100:>7.1f}%")

    checks_passed = 0
    checks_total = 0

    # Check 1: plaquette increases with β
    checks_total += 1
    monotonic = all(results[i+1]["mean_plaquette"] > results[i]["mean_plaquette"]
                    for i in range(len(results) - 1))
    if monotonic:
        print("\n  ✓ Plaquette monotonically increases with β")
        checks_passed += 1
    else:
        print("\n  ✗ Plaquette NOT monotonic with β")

    # Check 2: Polyakov loop increases through transition
    checks_total += 1
    confined = [r for r in results if r["beta"] < 5.5]
    deconfined = [r for r in results if r["beta"] > 6.0]
    if confined and deconfined:
        avg_conf = np.mean([r["polyakov_loop"] for r in confined])
        avg_deconf = np.mean([r["polyakov_loop"] for r in deconfined])
        if avg_deconf > avg_conf:
            print(f"  ✓ Polyakov: confined <|L|>={avg_conf:.4f} < deconfined <|L|>={avg_deconf:.4f}")
            checks_passed += 1
        else:
            print(f"  ✗ Polyakov transition NOT detected")

    # Check 3: acceptance > 10% everywhere
    checks_total += 1
    all_accept = all(r["acceptance_rate"] > 0.10 for r in results)
    if all_accept:
        print("  ✓ HMC acceptance > 10% at all β")
        checks_passed += 1
    else:
        print("  ✗ HMC acceptance below 10% at some β")

    print(f"\n  {checks_passed}/{checks_total} checks passed")

    out_path = "../results/quenched_beta_scan_4x4.json"
    output = {
        "lattice_dims": dims,
        "beta_values": beta_values,
        "n_thermalization": n_therm,
        "n_trajectories": n_traj,
        "n_md_steps": n_md_steps,
        "dt": dt,
        "seed_base": seed_base,
        "results": results,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
    }

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    sys.exit(0 if checks_passed == checks_total else 1)


if __name__ == "__main__":
    main()
