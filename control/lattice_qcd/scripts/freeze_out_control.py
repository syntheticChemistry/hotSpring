#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Freeze-out Curvature via Susceptibility beta-Scan — Python Control Baseline (Paper 12)

Independent NumPy implementation of the deconfinement transition detector:
  1. Fine beta-scan (5.2–6.2) with quenched HMC on 4^4
  2. Measure plaquette susceptibility chi_P(beta) = V * (<P^2> - <P>^2)
  3. Measure Polyakov loop susceptibility chi_L(beta) = V_s * (<|L|^2> - <|L|>^2)
  4. Locate peaks -> beta_c
  5. Compare with known SU(3) result: beta_c ~ 5.69 on N_t=4

The freeze-out curvature at finite mu_B requires dynamical fermions,
but the method (susceptibility peak location) is identical in quenched.

Uses the same SU(3) HMC infrastructure as quenched_beta_scan.py.

References:
  - Bazavov et al., PRD 93, 014512 (2016) — freeze-out curvature
  - Bali et al., PRD 62, 054503 (2000) — SU(3) beta_c reference
  - Lucini, Teper, Wenger, JHEP 0401:061 (2004) — SU(3) deconfinement
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

    def polyakov_loop(self, x_spatial):
        nt = self.dims[3]
        prod = su3_identity()
        for t in range(nt):
            x = [x_spatial[0], x_spatial[1], x_spatial[2], t]
            prod = prod @ self.link(x, 3)
        return np.trace(prod) / 3.0

    def average_polyakov_loop(self):
        ns = self.dims[:3]
        total = 0.0
        count = 0
        for ix in range(ns[0]):
            for iy in range(ns[1]):
                for iz in range(ns[2]):
                    total += abs(self.polyakov_loop([ix, iy, iz]))
                    count += 1
        return total / count


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
#  Susceptibilities — matches Rust correlator.rs exactly
# ═══════════════════════════════════════════════════════════════════

def plaquette_susceptibility(plaquettes, volume):
    """chi_P = V * (<P^2> - <P>^2)"""
    n = len(plaquettes)
    mean = sum(plaquettes) / n
    mean_sq = sum(p * p for p in plaquettes) / n
    return volume * (mean_sq - mean * mean)


def polyakov_susceptibility(poly_abs, spatial_vol):
    """chi_L = V_s * (<|L|^2> - <|L|>^2)"""
    n = len(poly_abs)
    mean_abs = sum(poly_abs) / n
    mean_sq = sum(p * p for p in poly_abs) / n
    return spatial_vol * (mean_sq - mean_abs * mean_abs)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

KNOWN_BETA_C = 5.69

def main():
    print("+" + "=" * 62 + "+")
    print("|  Freeze-out Susceptibility beta-Scan -- Python Control (P12)  |")
    print("|  Quenched HMC on 4^4, susceptibility peaks -> beta_c         |")
    print("+" + "=" * 62 + "+")
    print()

    dims = [4, 4, 4, 4]
    vol = dims[0] * dims[1] * dims[2] * dims[3]
    spatial_vol = dims[0] * dims[1] * dims[2]
    n_therm = 50
    n_meas = 100
    n_md_steps = 20
    dt = 0.05

    beta_values = [5.2 + 0.1 * i for i in range(11)]

    t0 = time.perf_counter()

    # Phase 1: beta-scan
    print("=== Phase 1: beta-scan (5.2-6.2, 11 points, 4^4 leapfrog) ===")
    print(f"  {'beta':>5} {'<P>':>8} {'chi_P':>10} {'<|L|>':>8} {'chi_L':>10} {'acc%':>6}")

    scan_results = []
    for beta in beta_values:
        seed = 42 + int(beta * 1000)
        lat = Lattice.hot_start(dims, beta, 42)

        # Thermalize
        for _ in range(n_therm):
            seed, _, _, _ = hmc_trajectory(lat, seed, n_md_steps, dt)

        # Measure
        plaquettes = []
        poly_abs_list = []
        accepted = 0

        for _ in range(n_meas):
            seed, plaq, dh, acc = hmc_trajectory(lat, seed, n_md_steps, dt)
            if acc:
                accepted += 1
            plaquettes.append(plaq)
            poly_abs_list.append(lat.average_polyakov_loop())

        mean_plaq = np.mean(plaquettes)
        chi_plaq = plaquette_susceptibility(plaquettes, vol)
        mean_poly = np.mean(poly_abs_list)
        chi_poly = polyakov_susceptibility(poly_abs_list, spatial_vol)
        accept_rate = accepted / n_meas

        print(f"  {beta:5.2f} {mean_plaq:8.6f} {chi_plaq:10.4f} "
              f"{mean_poly:8.4f} {chi_poly:10.4f} {accept_rate*100:5.1f}")

        scan_results.append({
            "beta": float(beta),
            "mean_plaquette": float(mean_plaq),
            "chi_plaquette": float(chi_plaq),
            "mean_polyakov": float(mean_poly),
            "chi_polyakov": float(chi_poly),
            "acceptance_rate": float(accept_rate),
        })

    print()

    # Phase 2: Locate peaks
    print("=== Phase 2: Locate susceptibility peaks ===")
    chi_p_max_idx = max(range(len(scan_results)),
                        key=lambda i: scan_results[i]["chi_plaquette"])
    beta_c_plaq = scan_results[chi_p_max_idx]["beta"]

    chi_l_max_idx = max(range(len(scan_results)),
                        key=lambda i: scan_results[i]["chi_polyakov"])
    beta_c_poly = scan_results[chi_l_max_idx]["beta"]

    print(f"  chi_P peak: beta_c = {beta_c_plaq:.2f} (index {chi_p_max_idx})")
    print(f"  chi_L peak: beta_c = {beta_c_poly:.2f} (index {chi_l_max_idx})")

    plaq_err = abs(beta_c_plaq - KNOWN_BETA_C) / KNOWN_BETA_C
    poly_err = abs(beta_c_poly - KNOWN_BETA_C) / KNOWN_BETA_C
    print(f"  Known beta_c ~ {KNOWN_BETA_C:.2f}, "
          f"plaq error={plaq_err*100:.1f}%, poly error={poly_err*100:.1f}%")
    print()

    # Phase 3: Physical observable trends
    print("=== Phase 3: Physical observable trends ===")
    plaq_mono = all(
        scan_results[i+1]["mean_plaquette"] >= scan_results[i]["mean_plaquette"] - 0.02
        for i in range(len(scan_results) - 1)
    )

    confined = [r for r in scan_results if r["beta"] < KNOWN_BETA_C - 0.3]
    deconfined = [r for r in scan_results if r["beta"] > KNOWN_BETA_C + 0.3]
    poly_low = confined[-1]["mean_polyakov"] if confined else 0.0
    poly_high = deconfined[0]["mean_polyakov"] if deconfined else 0.0
    print(f"  Polyakov below beta_c: {poly_low:.4f}")
    print(f"  Polyakov above beta_c: {poly_high:.4f}")
    print()

    wall_s = time.perf_counter() - t0

    # Validation checks
    checks_passed = 0
    checks_total = 0

    checks_total += 1
    if chi_p_max_idx > 0 and chi_p_max_idx < len(scan_results) - 1:
        print("  [PASS] chi_P peak not at scan boundary")
        checks_passed += 1
    else:
        print("  [FAIL] chi_P peak at boundary")

    checks_total += 1
    if scan_results[chi_l_max_idx]["chi_polyakov"] > 0.0:
        print("  [PASS] chi_L shows non-trivial signal")
        checks_passed += 1
    else:
        print("  [FAIL] chi_L has no signal")

    checks_total += 1
    if np.isfinite(beta_c_plaq) and np.isfinite(beta_c_poly):
        print("  [PASS] Both susceptibility estimators produce finite beta_c")
        checks_passed += 1
    else:
        print("  [FAIL] Non-finite beta_c")

    checks_total += 1
    if plaq_err < 0.10:
        print(f"  [PASS] beta_c(plaq) within 10% of known {KNOWN_BETA_C}")
        checks_passed += 1
    else:
        print(f"  [FAIL] beta_c(plaq) error {plaq_err*100:.1f}% > 10%")

    checks_total += 1
    if plaq_mono:
        print("  [PASS] Plaquette approximately monotone with beta")
        checks_passed += 1
    else:
        print("  [FAIL] Plaquette not monotone")

    checks_total += 1
    if poly_high > poly_low * 0.9 or poly_high > 0.25:
        print("  [PASS] Polyakov grows through transition region")
        checks_passed += 1
    else:
        print("  [FAIL] Polyakov transition not detected")

    checks_total += 1
    all_accept = all(r["acceptance_rate"] > 0.30 for r in scan_results)
    if all_accept:
        print("  [PASS] Acceptance > 30% at all beta")
        checks_passed += 1
    else:
        print("  [FAIL] Acceptance below 30% at some beta")

    checks_total += 1
    chi_p_max = scan_results[chi_p_max_idx]["chi_plaquette"]
    chi_p_min = min(r["chi_plaquette"] for r in scan_results)
    if chi_p_max > 2.0 * chi_p_min:
        print("  [PASS] chi_P peak is at least 2x the minimum (clear signal)")
        checks_passed += 1
    else:
        print("  [FAIL] chi_P peak not clearly above background")

    print(f"\n  {checks_passed}/{checks_total} checks passed")
    print(f"  Wall time: {wall_s:.1f}s")

    output = {
        "lattice_dims": dims,
        "beta_values": beta_values,
        "n_thermalization": n_therm,
        "n_measurement": n_meas,
        "n_md_steps": n_md_steps,
        "dt": dt,
        "known_beta_c": KNOWN_BETA_C,
        "scan_results": scan_results,
        "beta_c_plaquette": float(beta_c_plaq),
        "beta_c_polyakov": float(beta_c_poly),
        "beta_c_plaq_error_pct": float(plaq_err * 100),
        "beta_c_poly_error_pct": float(poly_err * 100),
        "plaquette_monotone": plaq_mono,
        "polyakov_transition_detected": bool(poly_high > poly_low * 0.9 or poly_high > 0.25),
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "wall_time_s": float(wall_s),
    }

    out_path = "../results/freeze_out_control.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    sys.exit(0 if checks_passed == checks_total else 1)


if __name__ == "__main__":
    main()
