#!/usr/bin/env python3
"""
Standalone Yukawa OCP MD baseline for transport coefficient validation.

Self-contained velocity-Verlet MD in reduced units with:
  - FCC lattice initialization (thermalized)
  - Berendsen thermostat during equilibration only
  - NVE production with full velocity storage
  - VACF via Green-Kubo relation
  - Reduced self-diffusion coefficient D*

Provides exact algorithmic parity with barracuda/src/md/observables.rs
for cross-validation.

Units: reduced (a_ws = 1, omega_p = 1, kBT = 3/(2*Gamma))

Reference:
  Stanton & Murillo, PRE 93, 043203 (2016)
  Daligault, PRE 86, 047401 (2012)
"""

import json
import sys
import datetime
import argparse
import numpy as np


def fcc_lattice(n_particles, box_side):
    """Generate FCC lattice positions scaled to fill the simulation box."""
    n_cells = int(np.ceil((n_particles / 4) ** (1.0 / 3.0)))
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ])
    positions = []
    cell_len = box_side / n_cells
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                for b in basis:
                    pos = (np.array([ix, iy, iz]) + b) * cell_len
                    positions.append(pos)
                    if len(positions) >= n_particles:
                        return np.array(positions[:n_particles])
    return np.array(positions[:n_particles])


def yukawa_forces(positions, n, box, kappa, rc):
    """
    Compute Yukawa pair forces and potential energy in reduced units.

    V(r) = (Gamma / r) * exp(-kappa * r)
    F_r = -dV/dr = Gamma * exp(-kappa*r) * (1/r^2 + kappa/r)
    """
    pe = 0.0
    forces = np.zeros_like(positions)
    rc2 = rc * rc

    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]

            dx -= box * round(dx / box)
            dy -= box * round(dy / box)
            dz -= box * round(dz / box)

            r2 = dx * dx + dy * dy + dz * dz
            if r2 < rc2:
                r = np.sqrt(r2)
                exp_kr = np.exp(-kappa * r)
                v = exp_kr / r
                pe += v
                f_over_r = exp_kr * (1.0 / r2 + kappa / r) / r
                fx = dx * f_over_r
                fy = dy * f_over_r
                fz = dz * f_over_r
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[i, 2] += fz
                forces[j, 0] -= fx
                forces[j, 1] -= fy
                forces[j, 2] -= fz

    return pe, forces


try:
    from numba import njit
    yukawa_forces = njit(yukawa_forces)
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def run_md(gamma, kappa, n_particles, eq_steps, prod_steps, dt, rc_aws,
           dump_step, berendsen_tau, seed=42):
    """Run a complete Yukawa OCP MD simulation in reduced units."""
    rng = np.random.default_rng(seed)
    t_star = 1.0 / gamma
    box = (4.0 * np.pi * n_particles / 3.0) ** (1.0 / 3.0)
    rc = rc_aws

    print(f"  Gamma={gamma}, kappa={kappa}, N={n_particles}")
    print(f"  T*={t_star:.6f}, box={box:.4f}, rc={rc:.2f} a_ws")
    print(f"  dt={dt}, eq={eq_steps}, prod={prod_steps}, dump={dump_step}")
    print(f"  numba: {'yes' if HAS_NUMBA else 'no (SLOW)'}")

    positions = fcc_lattice(n_particles, box)
    velocities = rng.normal(0.0, np.sqrt(t_star), (n_particles, 3))
    velocities -= velocities.mean(axis=0)

    ke_target = 1.5 * n_particles * t_star
    ke_actual = 0.5 * np.sum(velocities ** 2)
    velocities *= np.sqrt(ke_target / ke_actual)

    pe, forces = yukawa_forces(positions, n_particles, box, kappa, rc)
    pe *= gamma

    print(f"  Initial PE/N = {pe/n_particles:.4f}, KE/N = {ke_target/n_particles:.4f}")
    print(f"  Equilibrating...")

    for step in range(eq_steps):
        velocities += 0.5 * dt * forces * gamma
        positions += dt * velocities
        positions %= box
        pe, forces = yukawa_forces(positions, n_particles, box, kappa, rc)
        pe *= gamma
        velocities += 0.5 * dt * forces * gamma

        if berendsen_tau > 0:
            ke = 0.5 * np.sum(velocities ** 2)
            t_inst = 2.0 * ke / (3.0 * n_particles)
            lam = np.sqrt(1.0 + dt / berendsen_tau * (t_star / t_inst - 1.0))
            velocities *= lam

        if step % 1000 == 0:
            ke = 0.5 * np.sum(velocities ** 2)
            t_inst = 2.0 * ke / (3.0 * n_particles)
            print(f"    eq step {step}: T*={t_inst:.4f} (target {t_star:.4f}), PE/N={pe/n_particles:.4f}")

    velocities -= velocities.mean(axis=0)
    ke = 0.5 * np.sum(velocities ** 2)
    print(f"  Equilibration done. KE/N={ke/n_particles:.4f}, PE/N={pe/n_particles:.4f}")
    print(f"  Production (NVE)...")

    n_dumps = prod_steps // dump_step
    vel_trajectory = np.zeros((n_dumps, n_particles, 3))
    dump_idx = 0

    for step in range(prod_steps):
        velocities += 0.5 * dt * forces * gamma
        positions += dt * velocities
        positions %= box
        pe, forces = yukawa_forces(positions, n_particles, box, kappa, rc)
        pe *= gamma
        velocities += 0.5 * dt * forces * gamma

        if (step + 1) % dump_step == 0:
            vel_trajectory[dump_idx] = velocities.copy()
            dump_idx += 1

        if step % 5000 == 0:
            ke = 0.5 * np.sum(velocities ** 2)
            etot = ke + pe
            print(f"    prod step {step}: KE/N={ke/n_particles:.4f}, E/N={etot/n_particles:.4f}")

    print(f"  Production done. {dump_idx} velocity snapshots.")
    return vel_trajectory, dt * dump_step


def compute_vacf_and_dstar(vel_trajectory, dt_dump, max_lag_frac=0.5):
    """Compute VACF and D* from velocity trajectory via Green-Kubo."""
    n_frames, n_particles, _ = vel_trajectory.shape
    max_lag = min(int(n_frames * max_lag_frac), 2000)

    print(f"  Computing VACF (max_lag={max_lag}, n_frames={n_frames})...")
    vacf = np.zeros(max_lag)
    for tau in range(max_lag):
        n_origins = n_frames - tau
        dot = np.sum(vel_trajectory[:n_origins] * vel_trajectory[tau:tau + n_origins], axis=2)
        vacf[tau] = np.mean(dot)

    c0 = vacf[0]
    c_tail = vacf[-1]
    tail_ratio = c_tail / c0 if abs(c0) > 1e-30 else 0.0

    d_star = np.trapezoid(vacf, dx=dt_dump) / 3.0

    print(f"  C(0) = {c0:.6e}, C(t_max)/C(0) = {tail_ratio:.6f}")
    print(f"  D* = {d_star:.6e}")

    return vacf, d_star, tail_ratio


def daligault_d_star(gamma, kappa):
    """
    Daligault (2012) analytical fit for D*(Gamma, kappa).

    Matches barracuda/src/md/transport.rs exactly.
    Reference: Daligault, PRE 86, 047401 (2012)
    """
    gamma_eff = gamma * np.exp(-kappa)
    if gamma_eff < 0.1:
        cl = max(np.log(1.0 / gamma_eff), 1.0)
    else:
        cl = max(np.log(1.0 + 1.0 / gamma_eff), 0.1)

    d_weak = 3.0 * np.sqrt(np.pi) / 4.0 / (gamma ** 2.5 * cl)

    a = 0.0094 + 0.018 * kappa - 0.0025 * kappa ** 2
    alpha = 1.09 + 0.12 * kappa - 0.019 * kappa ** 2
    d_strong = a * gamma ** (-alpha)

    gamma_x = 10.0 * np.exp(0.5 * kappa)
    f = 1.0 / (1.0 + (gamma / gamma_x) ** 2)

    return d_weak * f + d_strong * (1.0 - f)


def main():
    parser = argparse.ArgumentParser(description="Yukawa MD baseline for transport validation")
    parser.add_argument("--lite", action="store_true", help="Lite cases (N=500, shorter runs)")
    parser.add_argument("--case", type=str, help="Run single case: k<kappa>_G<gamma>")
    args = parser.parse_args()

    cases = [
        {"kappa": 1, "gamma": 50, "rc_aws": 8.0},
        {"kappa": 2, "gamma": 100, "rc_aws": 6.5},
        {"kappa": 3, "gamma": 100, "rc_aws": 6.0},
    ]

    if args.case:
        parts = args.case.replace("k", "").replace("G", " ").split("_")
        k = int(parts[0])
        g = int(parts[1])
        cases = [c for c in cases if c["kappa"] == k and c["gamma"] == g]

    if args.lite:
        n_particles = 500
        eq_steps = 5000
        prod_steps = 20000
        dump_step = 1
    else:
        n_particles = 2000
        eq_steps = 10000
        prod_steps = 100000
        dump_step = 5

    dt = 0.01
    berendsen_tau = 5.0

    results = []

    for case in cases:
        kappa = case["kappa"]
        gamma = case["gamma"]
        rc = case["rc_aws"]

        print(f"\n{'=' * 60}")
        print(f"Case: kappa={kappa}, Gamma={gamma}")
        print(f"{'=' * 60}")

        vel_traj, dt_dump = run_md(
            gamma, kappa, n_particles, eq_steps, prod_steps,
            dt, rc, dump_step, berendsen_tau,
        )

        vacf, d_star, tail_ratio = compute_vacf_and_dstar(vel_traj, dt_dump)
        d_fit = daligault_d_star(gamma, kappa)
        rel_err = abs(d_star - d_fit) / d_fit if d_fit != 0 else float("inf")

        print(f"\n  D*(MD)       = {d_star:.6e}")
        print(f"  D*(Daligault) = {d_fit:.6e}")
        print(f"  Rel error     = {rel_err:.2%}")
        passf = "PASS" if rel_err < 0.15 else "FAIL"
        print(f"  Verdict: {passf}")

        results.append({
            "case": f"k{kappa}_G{gamma}",
            "kappa": kappa,
            "gamma": gamma,
            "n_particles": n_particles,
            "D_star_md": d_star,
            "D_star_daligault": d_fit,
            "relative_error": rel_err,
            "C0": float(vacf[0]),
            "C_tail_ratio": tail_ratio,
            "eq_steps": eq_steps,
            "prod_steps": prod_steps,
            "dt": dt,
            "dump_step": dump_step,
            "pass": bool(rel_err < 0.15),
        })

    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "_lite" if args.lite else ""

    output = {
        "provenance": {
            "generator": "hotSpring/control/sarkas/simulations/transport-study/scripts/yukawa_md_baseline.py",
            "method": "velocity_verlet_reduced_units_fcc_init",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "numba": HAS_NUMBA,
            "units": "reduced (a_ws=1, omega_p=1)",
        },
        "results": results,
    }

    out_path = results_dir / f"transport_baseline_standalone{suffix}.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\n{'=' * 60}")
    print(f"Output: {out_path}")
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {r['case']}: D*={r['D_star_md']:.4e} vs fit={r['D_star_daligault']:.4e} ({status})")

    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
