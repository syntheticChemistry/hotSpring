#!/usr/bin/env python3
"""
Ratcheting Step 1: Python validation of 9 PP Yukawa DSF cases.

Exact parameter match with barracuda/src/md/config.rs::dsf_pp_cases.
Self-contained velocity-Verlet in reduced units (a_ws=1, omega_p=1).
Vectorized NumPy for performance (no numba dependency).

Output: JSON results for comparison with barraCuda CPU/GPU and Kokkos/LAMMPS.
"""

import json
import sys
import time
import datetime
import numpy as np

DSF_CASES = [
    {"label": "k1_G14",   "kappa": 1.0, "gamma":   14.0, "rc": 8.0},
    {"label": "k1_G72",   "kappa": 1.0, "gamma":   72.0, "rc": 8.0},
    {"label": "k1_G217",  "kappa": 1.0, "gamma":  217.0, "rc": 8.0},
    {"label": "k2_G31",   "kappa": 2.0, "gamma":   31.0, "rc": 6.5},
    {"label": "k2_G158",  "kappa": 2.0, "gamma":  158.0, "rc": 6.5},
    {"label": "k2_G476",  "kappa": 2.0, "gamma":  476.0, "rc": 6.5},
    {"label": "k3_G100",  "kappa": 3.0, "gamma":  100.0, "rc": 6.0},
    {"label": "k3_G503",  "kappa": 3.0, "gamma":  503.0, "rc": 6.0},
    {"label": "k3_G1510", "kappa": 3.0, "gamma": 1510.0, "rc": 6.0},
]


def fcc_lattice(n_particles, box_side):
    n_cells = int(np.ceil((n_particles / 4) ** (1.0 / 3.0)))
    basis = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    positions = []
    cell_len = box_side / n_cells
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                for b in basis:
                    positions.append((np.array([ix, iy, iz]) + b) * cell_len)
                    if len(positions) >= n_particles:
                        return np.array(positions[:n_particles])
    return np.array(positions[:n_particles])


def yukawa_forces_vectorized(positions, n, box, kappa, rc):
    """Vectorized Yukawa force computation using NumPy broadcasting."""
    rc2 = rc * rc
    forces = np.zeros_like(positions)
    pe = 0.0

    for i in range(n - 1):
        dr = positions[i] - positions[i + 1:]  # r_i - r_j (force points FROM j TO i)
        dr -= box * np.round(dr / box)
        r2 = np.sum(dr * dr, axis=1)

        mask = r2 < rc2
        if not np.any(mask):
            continue

        dr_m = dr[mask]
        r2_m = r2[mask]
        r_m = np.sqrt(r2_m)
        exp_kr = np.exp(-kappa * r_m)

        pe += np.sum(exp_kr / r_m)

        f_over_r = exp_kr * (1.0 / r2_m + kappa / r_m) / r_m
        f_vec = dr_m * f_over_r[:, np.newaxis]

        forces[i] += np.sum(f_vec, axis=0)
        idx = np.where(mask)[0] + i + 1
        np.subtract.at(forces, idx, f_vec)

    return pe, forces


def run_md(gamma, kappa, n_particles, rc, equil_steps, prod_steps, dt=0.01,
           dump_step=10, berendsen_tau=5.0, seed=42):
    rng = np.random.default_rng(seed)
    t_star = 1.0 / gamma
    box = (4.0 * np.pi * n_particles / 3.0) ** (1.0 / 3.0)

    positions = fcc_lattice(n_particles, box)
    velocities = rng.normal(0.0, np.sqrt(t_star), (n_particles, 3))
    velocities -= velocities.mean(axis=0)
    ke_target = 1.5 * n_particles * t_star
    ke_actual = 0.5 * np.sum(velocities ** 2)
    velocities *= np.sqrt(ke_target / ke_actual)

    pe, forces = yukawa_forces_vectorized(positions, n_particles, box, kappa, rc)
    pe *= gamma

    for step in range(equil_steps):
        velocities += 0.5 * dt * forces * gamma
        positions += dt * velocities
        positions %= box
        pe, forces = yukawa_forces_vectorized(positions, n_particles, box, kappa, rc)
        pe *= gamma
        velocities += 0.5 * dt * forces * gamma
        if berendsen_tau > 0:
            ke = 0.5 * np.sum(velocities ** 2)
            t_inst = 2.0 * ke / (3.0 * n_particles)
            lam = np.sqrt(1.0 + dt / berendsen_tau * (t_star / t_inst - 1.0))
            velocities *= lam
        if step % 500 == 0:
            ke = 0.5 * np.sum(velocities ** 2)
            t_inst = 2.0 * ke / (3.0 * n_particles)
            print(f"      eq {step}: T*={t_inst:.4f} (target {t_star:.4f})")

    velocities -= velocities.mean(axis=0)

    energy_history = []
    t_wall_start = time.time()

    for step in range(prod_steps):
        velocities += 0.5 * dt * forces * gamma
        positions += dt * velocities
        positions %= box
        pe, forces = yukawa_forces_vectorized(positions, n_particles, box, kappa, rc)
        pe *= gamma
        velocities += 0.5 * dt * forces * gamma

        if (step + 1) % dump_step == 0:
            ke = 0.5 * np.sum(velocities ** 2)
            energy_history.append({"step": step + 1, "ke": float(ke),
                                   "pe": float(pe), "etot": float(ke + pe)})

        if step % 2000 == 0:
            ke = 0.5 * np.sum(velocities ** 2)
            etot = ke + pe
            elapsed = time.time() - t_wall_start
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"      prod {step}: E/N={etot/n_particles:.6f}, {sps:.0f} steps/s")

    wall_prod = time.time() - t_wall_start
    steps_per_sec = prod_steps / wall_prod if wall_prod > 0 else 0.0
    return energy_history, steps_per_sec, wall_prod


def validate_energy(history):
    if len(history) < 10:
        return 0.0, False
    etots = np.array([h["etot"] for h in history])
    e_mean = np.mean(etots)
    if abs(e_mean) < 1e-30:
        return 0.0, True
    e_first = np.mean(etots[:len(etots) // 10])
    e_last = np.mean(etots[-len(etots) // 10:])
    drift_pct = abs(e_last - e_first) / abs(e_mean) * 100.0
    return drift_pct, drift_pct < 5.0


def main():
    n_particles = 500
    equil_steps = 2000
    prod_steps = 10000
    dump_step = 10
    dt = 0.01

    print("=" * 70)
    print(f"  Ratcheting Step 1: Python Yukawa OCP Validation")
    print(f"  N={n_particles}, {len(DSF_CASES)} DSF cases, vectorized NumPy")
    print("=" * 70)

    results = []
    total_wall = 0.0

    for i, case in enumerate(DSF_CASES):
        label = case["label"]
        kappa = case["kappa"]
        gamma = case["gamma"]
        rc = case["rc"]

        print(f"\n{'─' * 60}")
        print(f"  Case {i+1}/{len(DSF_CASES)}: {label} (κ={kappa}, Γ={gamma})")
        print(f"{'─' * 60}")

        t0 = time.time()
        history, steps_per_sec, wall_prod = run_md(
            gamma, kappa, n_particles, rc,
            equil_steps, prod_steps, dt, dump_step
        )
        wall_total = time.time() - t0
        total_wall += wall_total

        drift_pct, passed = validate_energy(history)
        status = "PASS" if passed else "FAIL"

        etots = [h["etot"] for h in history]
        pe_final = history[-1]["pe"] if history else 0.0
        ke_final = history[-1]["ke"] if history else 0.0

        print(f"    E/N = {np.mean(etots)/n_particles:.6f}, "
              f"drift = {drift_pct:.3f}%, "
              f"steps/s = {steps_per_sec:.1f}, "
              f"wall = {wall_total:.1f}s  [{status}]")

        results.append({
            "label": label,
            "kappa": kappa,
            "gamma": gamma,
            "rc": rc,
            "n_particles": n_particles,
            "equil_steps": equil_steps,
            "prod_steps": prod_steps,
            "dt": dt,
            "energy_per_particle": float(np.mean(etots) / n_particles),
            "pe_per_particle": float(pe_final / n_particles),
            "ke_per_particle": float(ke_final / n_particles),
            "drift_pct": float(drift_pct),
            "steps_per_sec": float(steps_per_sec),
            "wall_seconds": float(wall_total),
            "passed": bool(passed),
        })

    n_passed = sum(1 for r in results if r["passed"])
    print(f"\n{'=' * 70}")
    print(f"  PYTHON VALIDATION: {n_passed}/{len(results)} cases passed")
    print(f"  Total wall time: {total_wall:.1f}s")
    print(f"{'=' * 70}")

    for r in results:
        s = "PASS" if r["passed"] else "FAIL"
        print(f"    {r['label']:>12}: drift={r['drift_pct']:.3f}%, "
              f"E/N={r['energy_per_particle']:.4f}, "
              f"{r['steps_per_sec']:.0f} steps/s  [{s}]")

    output = {
        "provenance": {
            "generator": "hotSpring/benchmarks/kokkos-lammps/validate_9cases_python.py",
            "method": "velocity_verlet_reduced_units_vectorized_numpy",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "n_particles": n_particles,
            "units": "reduced (a_ws=1, omega_p=1)",
        },
        "results": results,
        "summary": {
            "n_passed": n_passed,
            "n_total": len(results),
            "total_wall_seconds": total_wall,
        },
    }

    from pathlib import Path
    out_path = Path(__file__).parent / "results_python_9cases.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\n  Output: {out_path}")
    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
