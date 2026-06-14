#!/usr/bin/env python3
"""
Abelian Higgs model on a (1+1)D lattice — Python reference HMC.

Python control for Paper 13: Bazavov et al., Phys. Rev. D 92, 076003 (2015).

Implements U(1) gauge + complex scalar Higgs with:
  S_gauge = β_pl Σ (1 − Re U_p)
  S_higgs = Σ_x [ −2κ Σ_μ Re(φ* U_μ φ') + |φ|² + λ(|φ|²−1)² ]

HMC with leapfrog integrator, Metropolis accept/reject.

Output: reference observables (plaquette, ⟨|φ|²⟩, acceptance, ΔH) as JSON
for Rust validation parity check.

Usage:
  python abelian_higgs_hmc.py [--json output.json]
"""

import json
import time
import argparse
import numpy as np

# ── LCG PRNG (matches barracuda constants.rs) ────────────────────────

LCG_MUL = np.uint64(6_364_136_223_846_793_005)
LCG_INC = np.uint64(1_442_695_040_888_963_407)
LCG_53_DIV = float(1 << 53)


class LCG:
    """Deterministic LCG matching barracuda's constants.rs."""

    def __init__(self, seed: int):
        self.state = np.uint64(seed)

    def step(self):
        self.state = self.state * LCG_MUL + LCG_INC

    def uniform(self) -> float:
        self.step()
        return float(self.state >> np.uint64(11)) / LCG_53_DIV

    def gaussian(self) -> float:
        u1 = max(self.uniform(), 1e-30)
        u2 = self.uniform()
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


# ── Lattice ──────────────────────────────────────────────────────────

class U1HiggsLattice:
    """(1+1)D lattice with U(1) links (angles) and complex Higgs field."""

    def __init__(self, nt, ns, beta_pl, kappa, lam, mu=0.0):
        self.nt = nt
        self.ns = ns
        self.beta_pl = beta_pl
        self.kappa = kappa
        self.lam = lam
        self.mu = mu
        self.vol = nt * ns
        self.links = np.zeros(self.vol * 2)       # link angles
        self.higgs = np.ones(self.vol, dtype=complex)  # complex scalar

    @classmethod
    def cold(cls, nt, ns, beta_pl, kappa, lam, mu=0.0):
        return cls(nt, ns, beta_pl, kappa, lam, mu)

    @classmethod
    def hot(cls, nt, ns, beta_pl, kappa, lam, rng, mu=0.0):
        lat = cls(nt, ns, beta_pl, kappa, lam, mu)
        lat.links = np.array([2.0 * np.pi * rng.uniform() - np.pi
                              for _ in range(lat.vol * 2)])
        lat.higgs = np.array([complex(rng.gaussian() * 0.5 + 1.0,
                                      rng.gaussian() * 0.5)
                              for _ in range(lat.vol)])
        return lat

    def idx(self, t, x):
        return t * self.ns + x

    def fwd(self, t, x, mu):
        if mu == 0:
            return (t + 1) % self.nt, x
        return t, (x + 1) % self.ns

    def bwd(self, t, x, mu):
        if mu == 0:
            return (t - 1) % self.nt, x
        return t, (x - 1) % self.ns

    def link_angle(self, t, x, mu):
        return self.links[self.idx(t, x) * 2 + mu]

    def link(self, t, x, mu):
        return np.exp(1j * self.link_angle(t, x, mu))

    def plaquette_phase(self, t, x):
        t1, _ = self.fwd(t, x, 0)
        _, x1 = self.fwd(t, x, 1)
        return (self.link_angle(t, x, 0) + self.link_angle(t1, x, 1)
                - self.link_angle(t, x1, 0) - self.link_angle(t, x, 1))

    def average_plaquette(self):
        s = sum(np.cos(self.plaquette_phase(t, x))
                for t in range(self.nt) for x in range(self.ns))
        return s / self.vol

    def gauge_action(self):
        s = sum(1.0 - np.cos(self.plaquette_phase(t, x))
                for t in range(self.nt) for x in range(self.ns))
        return self.beta_pl * s

    def higgs_action(self):
        s_kin = 0.0
        s_pot = 0.0
        for t in range(self.nt):
            for x in range(self.ns):
                i = self.idx(t, x)
                phi = self.higgs[i]
                phi_sq = abs(phi) ** 2
                s_pot += phi_sq + self.lam * (phi_sq - 1.0) ** 2
                for mu in range(2):
                    tf, xf = self.fwd(t, x, mu)
                    phi_f = self.higgs[self.idx(tf, xf)]
                    u = self.link(t, x, mu)
                    hop = np.conj(phi) * u * phi_f
                    cw = np.exp(self.mu) if mu == 0 else 1.0
                    s_kin += cw * hop.real
        return -2.0 * self.kappa * s_kin + s_pot

    def total_action(self):
        return self.gauge_action() + self.higgs_action()

    def average_higgs_sq(self):
        return np.mean(np.abs(self.higgs) ** 2)

    # ── HMC forces ────────────────────────────────────────────────────

    def gauge_force(self, t, x, mu):
        nu = 1 - mu
        # Forward plaquette (link is first in its direction)
        tm, xm = self.fwd(t, x, mu)
        tn, xn = self.fwd(t, x, nu)
        phase_fwd = (self.link_angle(t, x, mu) + self.link_angle(tm, xm, nu)
                     - self.link_angle(tn, xn, mu) - self.link_angle(t, x, nu))
        f = self.beta_pl * np.sin(phase_fwd)

        # Backward plaquette
        tb, xb = self.bwd(t, x, nu)
        tbm, xbm = self.fwd(tb, xb, mu)
        phase_bwd = (self.link_angle(tb, xb, nu) + self.link_angle(t, x, mu)
                     - self.link_angle(tbm, xbm, nu)
                     - self.link_angle(tb, xb, mu))
        f += self.beta_pl * np.sin(phase_bwd)
        return -f

    def higgs_link_force(self, t, x, mu):
        tf, xf = self.fwd(t, x, mu)
        phi = self.higgs[self.idx(t, x)]
        phi_f = self.higgs[self.idx(tf, xf)]
        u = self.link(t, x, mu)
        hop = np.conj(phi) * u * phi_f
        cw = np.exp(self.mu) if mu == 0 else 1.0
        return -2.0 * self.kappa * cw * hop.imag

    def higgs_field_force(self, t, x):
        i = self.idx(t, x)
        phi = self.higgs[i]
        phi_sq = abs(phi) ** 2
        pot_grad = phi * (1.0 + 2.0 * self.lam * (phi_sq - 1.0))
        kin = 0j
        for mu in range(2):
            tf, xf = self.fwd(t, x, mu)
            u_f = self.link(t, x, mu)
            cf = np.exp(self.mu) if mu == 0 else 1.0
            kin += u_f * self.higgs[self.idx(tf, xf)] * cf
            tb, xb = self.bwd(t, x, mu)
            u_b = np.conj(self.link(tb, xb, mu))
            cb = np.exp(-self.mu) if mu == 0 else 1.0
            kin += u_b * self.higgs[self.idx(tb, xb)] * cb
        return 2.0 * (self.kappa * kin - pot_grad)


def kinetic_energy(pi_links, pi_higgs):
    return 0.5 * (np.sum(pi_links ** 2) + np.sum(np.abs(pi_higgs) ** 2))


def hmc_trajectory(lat, n_md, dt, rng):
    """Run one HMC trajectory. Returns (accepted, delta_h, plaq, higgs_sq)."""
    old_links = lat.links.copy()
    old_higgs = lat.higgs.copy()

    pi_links = np.array([rng.gaussian() for _ in range(lat.vol * 2)])
    pi_higgs = np.array([complex(rng.gaussian(), rng.gaussian())
                         for _ in range(lat.vol)])

    ke0 = kinetic_energy(pi_links, pi_higgs)
    s0 = lat.total_action()
    h0 = ke0 + s0

    def update_momenta(fdt):
        for t in range(lat.nt):
            for x in range(lat.ns):
                for mu in range(2):
                    idx = lat.idx(t, x) * 2 + mu
                    fg = lat.gauge_force(t, x, mu)
                    fh = lat.higgs_link_force(t, x, mu)
                    pi_links[idx] += fdt * (fg + fh)
                idx = lat.idx(t, x)
                pi_higgs[idx] += fdt * lat.higgs_field_force(t, x)

    # Leapfrog
    update_momenta(dt / 2.0)
    for step in range(n_md):
        lat.links += dt * pi_links
        for i in range(lat.vol):
            lat.higgs[i] += dt * pi_higgs[i]
        if step < n_md - 1:
            update_momenta(dt)
    update_momenta(dt / 2.0)

    ke1 = kinetic_energy(pi_links, pi_higgs)
    s1 = lat.total_action()
    h1 = ke1 + s1
    dh = h1 - h0

    accepted = dh <= 0.0 or rng.uniform() < np.exp(-dh)
    if not accepted:
        lat.links = old_links
        lat.higgs = old_higgs

    return accepted, dh, lat.average_plaquette(), lat.average_higgs_sq()


# ── Scan parameters ──────────────────────────────────────────────────

SCAN_CONFIGS = [
    # (label, nt, ns, beta, kappa, lambda, n_therm, n_traj, n_md, dt, seed)
    ("weak_coupling",     8, 8, 6.0, 0.3, 1.0, 50, 100, 10, 0.08, 42),
    ("strong_coupling",   8, 8, 0.5, 0.3, 1.0, 50, 100, 10, 0.08, 42),
    ("higgs_condensed",   8, 8, 2.0, 2.0, 1.0, 50, 100, 10, 0.05, 42),
    ("confined",          8, 8, 1.0, 0.1, 1.0, 50, 100, 10, 0.08, 42),
    ("large_lambda",      8, 8, 2.0, 0.5, 10.0, 50, 100, 10, 0.05, 42),
    ("cold_start_check",  4, 4, 2.0, 0.5, 1.0,  0,   1, 10, 0.05, 42),
]


def run_scan(configs):
    results = []
    for label, nt, ns, beta, kappa, lam, n_th, n_tr, n_md, dt, seed in configs:
        rng = LCG(seed)
        lat = U1HiggsLattice.hot(nt, ns, beta, kappa, lam, rng)

        t0 = time.perf_counter()

        # Thermalization
        for _ in range(n_th):
            hmc_trajectory(lat, n_md, dt, rng)

        plaq_sum = 0.0
        hsq_sum = 0.0
        dh_sum = 0.0
        n_acc = 0
        for _ in range(n_tr):
            acc, dh, plaq, hsq = hmc_trajectory(lat, n_md, dt, rng)
            plaq_sum += plaq
            hsq_sum += hsq
            dh_sum += abs(dh)
            n_acc += int(acc)

        elapsed = time.perf_counter() - t0
        r = {
            "label": label,
            "nt": nt, "ns": ns,
            "beta_pl": beta, "kappa": kappa, "lambda": lam,
            "n_therm": n_th, "n_traj": n_tr, "n_md": n_md, "dt": dt,
            "acceptance": n_acc / n_tr,
            "avg_plaquette": plaq_sum / n_tr,
            "avg_higgs_sq": hsq_sum / n_tr,
            "avg_abs_delta_h": dh_sum / n_tr,
            "elapsed_s": elapsed,
        }
        results.append(r)
        print(f"  {label:20s}  plaq={r['avg_plaquette']:.6f}  "
              f"|φ²|={r['avg_higgs_sq']:.6f}  acc={r['acceptance']:.2f}  "
              f"|ΔH|={r['avg_abs_delta_h']:.4f}  {elapsed:.3f}s")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="abelian_higgs_reference.json")
    args = parser.parse_args()

    print("Abelian Higgs (1+1)D — Python reference HMC")
    print("=" * 60)
    results = run_scan(SCAN_CONFIGS)

    with open(args.json, "w") as f:
        json.dump({"paper": "Bazavov et al. 2015, PRD 92, 076003",
                   "model": "U(1) Abelian Higgs, (1+1)D",
                   "configs": results}, f, indent=2)
    print(f"\nReference data → {args.json}")


if __name__ == "__main__":
    main()
