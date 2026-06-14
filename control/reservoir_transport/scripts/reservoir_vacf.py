#!/usr/bin/env python3
"""
Reservoir Computing for MD Transport Prediction — Python Control.

Echo State Network (ESN) trained on velocity features from Yukawa OCP
molecular dynamics. Validates the ESN pipeline end-to-end in Python,
providing the scaffold for Rust CPU and Akida NPU implementations.

Units: OCP reduced (a_ws=1, omega_p=1, E_0 = q^2/(4pieps0 a_ws))
  Mass: m* = 4*pi*n*a_ws^3 = 3.0  (derived, not a magic number)
  Temp: T* = k_BT / E_0 = 1/Gamma
  Accel: a = F/m* (force prefactor 1.0, mass 3.0)

This matches barracuda/src/md/cpu_reference.rs exactly for cross-
validation. Future evolution: migrate both Python and Rust to the
Sarkas/standard convention (m=1, V=Gamma*phi) for ecosystem parity.

Reference:
  Jaeger (2001) "The echo state approach to recurrent neural networks"
  Stanton & Murillo, PRE 93, 043203 (2016)

License: AGPL-3.0-only
"""

import json
import sys
import time
import datetime
import numpy as np
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
#  Yukawa OCP MD (reused from yukawa_md_baseline.py)
# ═══════════════════════════════════════════════════════════════════

def fcc_lattice(n_particles, box_side):
    n_cells = int(np.ceil((n_particles / 4) ** (1.0 / 3.0)))
    basis = np.array([
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
    ])
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


def yukawa_forces(positions, n, box, kappa, rc):
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


def run_md(gamma, kappa, n_particles, eq_steps, prod_steps, dt, rc,
           dump_step, berendsen_tau, seed=42):
    """
    Velocity-Verlet MD in OCP E_0 reduced units.

    Matches barracuda/src/md/cpu_reference.rs exactly:
      m* = 3.0, force_prefactor = 1.0, T* = 1/Gamma
    """
    rng = np.random.default_rng(seed)
    t_star = 1.0 / gamma
    mass = 3.0
    inv_m = 1.0 / mass
    box = (4.0 * np.pi * n_particles / 3.0) ** (1.0 / 3.0)

    positions = fcc_lattice(n_particles, box)
    velocities = rng.normal(0.0, np.sqrt(t_star / mass), (n_particles, 3))
    velocities -= velocities.mean(axis=0)
    ke_target = 1.5 * n_particles * t_star
    ke_actual = 0.5 * mass * np.sum(velocities ** 2)
    velocities *= np.sqrt(ke_target / ke_actual)

    pe, forces = yukawa_forces(positions, n_particles, box, kappa, rc)

    for step in range(eq_steps):
        velocities += 0.5 * dt * forces * inv_m
        positions += dt * velocities
        positions %= box
        pe, forces = yukawa_forces(positions, n_particles, box, kappa, rc)
        velocities += 0.5 * dt * forces * inv_m
        if berendsen_tau > 0 and step % 10 == 0:
            ke = 0.5 * mass * np.sum(velocities ** 2)
            t_inst = 2.0 * ke / (3.0 * n_particles)
            ratio = 1.0 + dt / berendsen_tau * (t_star / t_inst - 1.0)
            lam = np.sqrt(max(ratio, 0.0))
            velocities *= lam

    # Exact temperature rescale at production start (matches Rust)
    ke = 0.5 * mass * np.sum(velocities ** 2)
    t_inst = 2.0 * ke / (3.0 * n_particles)
    if t_inst > 1e-30:
        velocities *= np.sqrt(t_star / t_inst)

    n_dumps = prod_steps // dump_step
    vel_trajectory = np.zeros((n_dumps, n_particles, 3))
    dump_idx = 0

    for step in range(prod_steps):
        velocities += 0.5 * dt * forces * inv_m
        positions += dt * velocities
        positions %= box
        pe, forces = yukawa_forces(positions, n_particles, box, kappa, rc)
        velocities += 0.5 * dt * forces * inv_m
        if (step + 1) % dump_step == 0:
            vel_trajectory[dump_idx] = velocities.copy()
            dump_idx += 1

    return vel_trajectory, dt * dump_step


def compute_vacf_and_dstar(vel_trajectory, dt_dump, max_lag=None):
    """
    Compute VACF and D* with plateau detection, matching Rust exactly.

    D* = (1/3) integral C(t) dt, with plateau cutoff to avoid noise
    accumulation beyond the correlation time (Allen & Tildesley).
    """
    n_frames, n_particles, _ = vel_trajectory.shape
    if max_lag is None:
        max_lag = min(n_frames // 2, 2000)
    vacf = np.zeros(max_lag)
    for tau in range(max_lag):
        n_origins = n_frames - tau
        dot = np.sum(
            vel_trajectory[:n_origins] * vel_trajectory[tau:tau + n_origins],
            axis=2,
        )
        vacf[tau] = np.mean(dot)

    # Plateau-detection D* (matches Rust compute_vacf)
    integral = 0.0
    d_star_max = 0.0
    plateau_count = 0
    plateau_window = int(np.ceil(20.0 / dt_dump))

    for i in range(1, max_lag):
        integral += 0.5 * (vacf[i - 1] + vacf[i]) * dt_dump
        d_star_running = integral / 3.0
        if d_star_running > d_star_max:
            d_star_max = d_star_running
            plateau_count = 0
        else:
            plateau_count += 1
            if plateau_count > plateau_window:
                break

    return vacf, d_star_max


# ═══════════════════════════════════════════════════════════════════
#  Echo State Network (NumPy reference)
# ═══════════════════════════════════════════════════════════════════

class EchoStateNetwork:
    """
    Echo State Network matching ToadStool's barracuda::esn_v2::ESNConfig.

    The reservoir is a sparse random recurrent network. Input is fed through
    W_in, mixed with recurrent state through W_res, and the readout W_out
    is trained via ridge regression (no backpropagation).
    """

    def __init__(self, input_size, reservoir_size, output_size,
                 spectral_radius=0.95, connectivity=0.1, leak_rate=0.3,
                 regularization=1e-6, seed=42):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.leak_rate = leak_rate
        self.regularization = regularization

        rng = np.random.default_rng(seed)

        self.w_in = rng.uniform(-0.5, 0.5, (reservoir_size, input_size))

        mask = rng.random((reservoir_size, reservoir_size)) < connectivity
        w = rng.standard_normal((reservoir_size, reservoir_size)) * mask
        eigenvalues = np.linalg.eigvals(w)
        sr = np.max(np.abs(eigenvalues))
        if sr > 1e-10:
            w = w * (spectral_radius / sr)
        self.w_res = w

        self.w_out = None
        self.state = np.zeros(reservoir_size)

    def reset_state(self):
        self.state = np.zeros(self.reservoir_size)

    def _update(self, u):
        """Single reservoir update: x(t+1) = (1-a)*x(t) + a*tanh(W_in*u + W_res*x)"""
        pre = self.w_in @ u + self.w_res @ self.state
        self.state = ((1.0 - self.leak_rate) * self.state
                      + self.leak_rate * np.tanh(pre))
        return self.state.copy()

    def collect_states(self, input_sequence):
        """
        Drive reservoir with input sequence, collect states.

        Args:
            input_sequence: (T, input_size) array

        Returns:
            states: (T, reservoir_size) array
        """
        T = len(input_sequence)
        states = np.zeros((T, self.reservoir_size))
        for t in range(T):
            states[t] = self._update(input_sequence[t])
        return states

    def train(self, input_sequences, targets):
        """
        Train readout via ridge regression on collected reservoir states.

        Args:
            input_sequences: list of (T_i, input_size) arrays
            targets: list of (output_size,) arrays — one target per sequence
        """
        all_states = []
        for seq in input_sequences:
            self.reset_state()
            states = self.collect_states(seq)
            all_states.append(states[-1])

        X = np.array(all_states)
        Y = np.array(targets)

        XtX = X.T @ X + self.regularization * np.eye(self.reservoir_size)
        XtY = X.T @ Y
        self.w_out = np.linalg.solve(XtX, XtY).T

    def predict(self, input_sequence):
        """
        Predict output from input sequence.

        Returns:
            (output_size,) array
        """
        self.reset_state()
        states = self.collect_states(input_sequence)
        return self.w_out @ states[-1]


# ═══════════════════════════════════════════════════════════════════
#  Feature extraction: velocity stream → ESN input
# ═══════════════════════════════════════════════════════════════════

def velocity_features(vel_trajectory, kappa, gamma, n_features=8):
    """
    Extract per-frame features from velocity trajectory for ESN input.

    From each frame of (N, 3) velocities, compute:
      [mean_vx, mean_vy, mean_vz, mean_speed, ke_per_particle, v_rms, kappa, gamma_scaled]

    kappa and gamma are constant physics parameters that allow the ESN to
    generalize across the (kappa, Gamma) phase diagram.

    Returns: (n_frames, n_features) array
    """
    n_frames = vel_trajectory.shape[0]
    features = np.zeros((n_frames, n_features))
    gamma_scaled = np.log10(gamma) / 3.0
    for t in range(n_frames):
        v = vel_trajectory[t]
        speeds = np.linalg.norm(v, axis=1)
        ke = 0.5 * np.mean(np.sum(v ** 2, axis=1))
        features[t, 0] = np.mean(v[:, 0])
        features[t, 1] = np.mean(v[:, 1])
        features[t, 2] = np.mean(v[:, 2])
        features[t, 3] = np.mean(speeds)
        features[t, 4] = ke
        features[t, 5] = np.sqrt(np.mean(speeds ** 2))
        features[t, 6] = kappa / 3.0
        features[t, 7] = gamma_scaled
    return features


# ═══════════════════════════════════════════════════════════════════
#  Scan configurations
# ═══════════════════════════════════════════════════════════════════

CASES = [
    {"kappa": 1, "gamma": 50,  "rc": 8.0},
    {"kappa": 1, "gamma": 72,  "rc": 8.0},
    {"kappa": 2, "gamma": 31,  "rc": 6.5},
    {"kappa": 2, "gamma": 100, "rc": 6.5},
    {"kappa": 2, "gamma": 158, "rc": 6.5},
    {"kappa": 3, "gamma": 100, "rc": 6.0},
]

TRAIN_INDICES = [0, 2, 4, 5]
TEST_INDICES = [1, 3]

N_FEATURES = 8


def main():
    t0 = time.time()
    n_particles = 256
    eq_steps = 5000
    prod_steps_full = 4000
    prod_steps_short = 500
    dump_step = 1
    dt = 0.01
    berendsen_tau = 5.0

    print("=" * 70)
    print("Reservoir Computing for MD Transport Prediction")
    print(f"  {len(CASES)} cases, train on {len(TRAIN_INDICES)}, test on {len(TEST_INDICES)}")
    print(f"  N={n_particles}, full={prod_steps_full} steps, short={prod_steps_short} steps")
    print(f"  numba: {'yes' if HAS_NUMBA else 'no (SLOW)'}")
    print("=" * 70)

    # ── Phase 1: Generate data ────────────────────────────────────
    case_data = []
    for i, case in enumerate(CASES):
        kappa, gamma, rc = case["kappa"], case["gamma"], case["rc"]
        print(f"\n─── Case {i}: κ={kappa}, Γ={gamma} ───")

        t_case = time.time()
        vel_full, dt_dump = run_md(
            gamma, kappa, n_particles, eq_steps, prod_steps_full,
            dt, rc, dump_step, berendsen_tau, seed=42 + i,
        )
        vacf_full, d_star_full = compute_vacf_and_dstar(vel_full, dt_dump)

        vel_short = vel_full[:prod_steps_short]
        vacf_short, d_star_short = compute_vacf_and_dstar(vel_short, dt_dump,
                                                          max_lag=min(1000, prod_steps_short // 2))

        features_full = velocity_features(vel_full, kappa, gamma, N_FEATURES)
        features_short = velocity_features(vel_short, kappa, gamma, N_FEATURES)

        elapsed = time.time() - t_case
        print(f"  D*(full)  = {d_star_full:.6e}  ({prod_steps_full} steps)")
        print(f"  D*(short) = {d_star_short:.6e}  ({prod_steps_short} steps)")
        print(f"  Time: {elapsed:.1f}s")

        case_data.append({
            "kappa": kappa, "gamma": gamma,
            "d_star_full": d_star_full,
            "d_star_short": d_star_short,
            "features_full": features_full,
            "features_short": features_short,
            "vacf_full": vacf_full[:200],
        })

    # ── Phase 2: Train ESN ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Training Echo State Network")
    print("=" * 70)

    reservoir_size = 50
    esn = EchoStateNetwork(
        input_size=N_FEATURES,
        reservoir_size=reservoir_size,
        output_size=1,
        spectral_radius=0.95,
        connectivity=0.2,
        leak_rate=0.3,
        regularization=1e-2,
        seed=42,
    )

    train_sequences = []
    train_targets = []
    for idx in TRAIN_INDICES:
        cd = case_data[idx]
        train_sequences.append(cd["features_short"])
        train_targets.append([cd["d_star_full"]])

    esn.train(train_sequences, train_targets)
    print(f"  Reservoir size: {reservoir_size}")
    print(f"  Training cases: {len(TRAIN_INDICES)}")
    print(f"  W_out shape: {esn.w_out.shape}")

    # ── Phase 3: Evaluate ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Evaluation: ESN predictions vs Green-Kubo ground truth")
    print("=" * 70)

    results = []
    print(f"\n  {'Case':<15} {'D*(GK)':<12} {'D*(ESN)':<12} {'D*(short)':<12} {'ESN err':>8} {'Short err':>9}  {'Set':<5}")
    print("  " + "-" * 75)

    for i, cd in enumerate(case_data):
        d_gk = cd["d_star_full"]
        d_short = cd["d_star_short"]

        pred = esn.predict(cd["features_short"])
        d_esn = float(pred[0])

        esn_err = abs(d_esn - d_gk) / abs(d_gk) if abs(d_gk) > 1e-30 else float("inf")
        short_err = abs(d_short - d_gk) / abs(d_gk) if abs(d_gk) > 1e-30 else float("inf")

        split = "TRAIN" if i in TRAIN_INDICES else "TEST"
        label = f"k{cd['kappa']}_G{cd['gamma']}"

        print(f"  {label:<15} {d_gk:<12.4e} {d_esn:<12.4e} {d_short:<12.4e} {esn_err:>7.1%} {short_err:>8.1%}  {split}")

        results.append({
            "case": label,
            "kappa": cd["kappa"],
            "gamma": cd["gamma"],
            "d_star_green_kubo": d_gk,
            "d_star_esn": d_esn,
            "d_star_short_gk": d_short,
            "esn_relative_error": esn_err,
            "short_gk_relative_error": short_err,
            "split": split.lower(),
        })

    # ── Summary ───────────────────────────────────────────────────
    test_results = [r for r in results if r["split"] == "test"]
    train_results = [r for r in results if r["split"] == "train"]

    train_err = np.mean([r["esn_relative_error"] for r in train_results]) if train_results else 0
    test_err = np.mean([r["esn_relative_error"] for r in test_results]) if test_results else 0
    short_test_err = np.mean([r["short_gk_relative_error"] for r in test_results]) if test_results else 0

    total_time = time.time() - t0

    print(f"\n  Train mean error: {train_err:.1%}")
    print(f"  Test mean error:  {test_err:.1%}")
    print(f"  Short GK test error: {short_test_err:.1%}")
    print(f"  Total time: {total_time:.1f}s")

    esn_config = {
        "input_size": N_FEATURES,
        "reservoir_size": reservoir_size,
        "output_size": 1,
        "spectral_radius": 0.95,
        "connectivity": 0.2,
        "leak_rate": 0.3,
        "regularization": 1e-2,
        "seed": 42,
    }

    output = {
        "provenance": {
            "generator": "hotSpring/control/reservoir_transport/scripts/reservoir_vacf.py",
            "method": "echo_state_network_transport_prediction",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "numba": HAS_NUMBA,
            "units": "reduced (a_ws=1, omega_p=1)",
        },
        "esn_config": esn_config,
        "md_config": {
            "n_particles": n_particles,
            "eq_steps": eq_steps,
            "prod_steps_full": prod_steps_full,
            "prod_steps_short": prod_steps_short,
            "dt": dt,
            "dump_step": dump_step,
        },
        "summary": {
            "train_cases": len(TRAIN_INDICES),
            "test_cases": len(TEST_INDICES),
            "train_mean_error": train_err,
            "test_mean_error": test_err,
            "short_gk_test_error": short_test_err,
            "total_time_s": total_time,
        },
        "results": results,
        "w_out_shape": list(esn.w_out.shape) if esn.w_out is not None else None,
    }

    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "reservoir_transport_baseline.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")

    print(f"\n  Output: {out_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
