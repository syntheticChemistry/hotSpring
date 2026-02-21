"""
Native int4+ReLU Reservoir — metalForge Control Experiment

Designs and validates a reservoir computing system that is native to the
AKD1000's actual compute model: int4 ternary weights, ReLU activation,
event-based sparsity. Instead of quantizing a float ESN, we design one
that IS the hardware's math.

Key differences from standard ESN:
  - Weights: ternary {-1, 0, +1} at 4-bit (most weights zero for sparsity)
  - Activation: ReLU (hardware-native) instead of tanh
  - No negative states (ReLU clips to 0)
  - Echo State Property: requires different conditions than tanh reservoir
  - Readout: still trained via ridge regression on float states

This is the "shader-originating math" philosophy applied to neuromorphic
hardware: the int4+ReLU reservoir IS the true computation, matching what
the transistors actually execute.

Outputs: control/metalforge_npu/results/native_reservoir_baseline.json
"""

import os
import sys
import json
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class NativeInt4Reservoir:
    """
    Reservoir with ternary weights and ReLU activation.

    Matches AKD1000 hardware: int4 weights (here ternary for maximal
    sparsity), ReLU activation (bounded to [0, max_act]), leak_rate
    for state decay.
    """

    def __init__(self, input_size, reservoir_size, output_size=1,
                 connectivity=0.2, leak_rate=0.3, spectral_radius=0.95,
                 max_activation=7.0, regularization=1e-2, seed=42):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.leak_rate = leak_rate
        self.max_activation = max_activation
        self.regularization = regularization

        rng = np.random.default_rng(seed)

        # W_in: ternary {-1, 0, +1} with ~50% sparsity
        self.w_in = np.zeros((reservoir_size, input_size), dtype=np.int8)
        for i in range(reservoir_size):
            for j in range(input_size):
                r = rng.random()
                if r < 0.25:
                    self.w_in[i, j] = -1
                elif r < 0.50:
                    self.w_in[i, j] = 1

        # W_res: ternary sparse recurrent weights
        self.w_res = np.zeros((reservoir_size, reservoir_size), dtype=np.int8)
        for i in range(reservoir_size):
            for j in range(reservoir_size):
                if rng.random() < connectivity:
                    self.w_res[i, j] = rng.choice([-1, 1])

        # Scale recurrent weights for echo state property
        sr = self._spectral_radius()
        if sr > 1e-10:
            scale = spectral_radius / sr
            # For ternary, we can't scale directly. Instead we use the
            # leak_rate to control effective spectral radius:
            # effective_sr = (1 - alpha) + alpha * sr ≈ spectral_radius
            # when alpha = (spectral_radius - (1-alpha)) / sr
            # In practice, ternary reservoirs are stable with ReLU if
            # connectivity is low enough.
            self.input_scale = min(scale, 2.0)
        else:
            self.input_scale = 1.0

        self.w_out = None

    def _spectral_radius(self):
        """Estimate spectral radius via power iteration."""
        n = self.reservoir_size
        v = np.ones(n) / np.sqrt(n)
        w = self.w_res.astype(np.float64)
        lam = 0.0
        for _ in range(100):
            wv = w @ v
            norm = np.linalg.norm(wv)
            if norm < 1e-30:
                return 0.0
            lam = norm
            v = wv / norm
        return lam

    def forward(self, input_sequence, return_states=False):
        """Run reservoir forward with int4 + ReLU dynamics.

        The computation mirrors what the AKD1000 hardware does:
          pre = W_in * input + W_res * state     (int4 × int4 MACs)
          state = (1-α) * state + α * ReLU(pre)  (leaky update)
          state = clip(state, 0, max_act)         (bounded ReLU)
        """
        rs = self.reservoir_size
        state = np.zeros(rs, dtype=np.float32)
        states = []

        for inp in input_sequence:
            inp_scaled = np.array(inp, dtype=np.float32) * self.input_scale

            pre = (self.w_in.astype(np.float32) @ inp_scaled +
                   self.w_res.astype(np.float32) @ state)

            activated = np.maximum(pre, 0.0)
            activated = np.minimum(activated, self.max_activation)

            state = (1.0 - self.leak_rate) * state + self.leak_rate * activated

            if return_states:
                states.append(state.copy())

        if return_states:
            return state, states
        return state

    def train(self, input_sequences, targets):
        """Train readout via ridge regression (same as standard ESN)."""
        rs = self.reservoir_size
        n = len(input_sequences)

        x_mat = np.zeros((n, rs), dtype=np.float64)
        for i, seq in enumerate(input_sequences):
            x_mat[i] = self.forward(seq).astype(np.float64)

        y_mat = np.array(targets).reshape(-1, self.output_size)

        xtx = x_mat.T @ x_mat + self.regularization * np.eye(rs)
        xty = x_mat.T @ y_mat
        self.w_out = np.linalg.solve(xtx, xty).T

    def predict(self, input_sequence):
        """Predict using trained readout."""
        state = self.forward(input_sequence).astype(np.float64)
        return (self.w_out @ state).item()

    def sparsity_stats(self):
        """Report weight and activation sparsity."""
        w_in_sparse = np.sum(self.w_in == 0) / self.w_in.size
        w_res_sparse = np.sum(self.w_res == 0) / self.w_res.size
        w_in_unique = len(np.unique(self.w_in))
        w_res_unique = len(np.unique(self.w_res))
        return {
            "w_in_sparsity": float(w_in_sparse),
            "w_res_sparsity": float(w_res_sparse),
            "w_in_unique_values": w_in_unique,
            "w_res_unique_values": w_res_unique,
            "w_in_ternary": w_in_unique <= 3,
            "w_res_ternary": w_res_unique <= 3,
        }


def generate_md_like_data(n_cases=8, n_frames=100, n_features=8, seed=123):
    """
    Generate synthetic data mimicking MD velocity features → D* prediction.

    Features are scaled to [0, 1] (not [-1, 1]) to match ReLU reservoir's
    non-negative activation space.
    """
    rng = np.random.default_rng(seed)
    sequences = []
    targets = []

    kappa_gamma_cases = [
        (1.0, 10.0), (1.0, 50.0), (1.0, 175.0),
        (2.0, 10.0), (2.0, 50.0), (2.0, 175.0),
        (1.5, 30.0), (1.5, 100.0),
    ]

    for kappa, gamma in kappa_gamma_cases[:n_cases]:
        d_star = 0.5 / (gamma ** 0.5) * np.exp(-0.3 * kappa)

        seq = np.zeros((n_frames, n_features))
        for t in range(n_frames):
            speed = np.sqrt(1.0 / gamma) * (1 + 0.1 * rng.standard_normal())
            ke = 0.5 * speed ** 2
            seq[t] = [
                abs(rng.standard_normal() * speed),
                abs(rng.standard_normal() * speed),
                abs(rng.standard_normal() * speed),
                speed,
                ke,
                speed * 1.1,
                kappa / 3.0,
                np.log10(gamma) / 3.0,
            ]

        seq = np.abs(seq)
        seq_max = seq.max(axis=0)
        seq_max[seq_max < 1e-10] = 1.0
        seq = seq / seq_max

        sequences.append(seq)
        targets.append(d_star)

    return sequences, targets, kappa_gamma_cases[:n_cases]


def main():
    print("=" * 60)
    print("  Native int4+ReLU Reservoir — metalForge Control")
    print("=" * 60)
    print()

    sequences, targets, cases = generate_md_like_data()

    train_idx = list(range(6))
    test_idx = [6, 7]

    # --- Standard tanh ESN for comparison ---
    print("[1] Standard tanh ESN (float, for comparison)")
    from npu_quantization_parity import build_esn_weights, esn_forward
    w_in_f, w_res_f = build_esn_weights(input_size=8, reservoir_size=50)

    tanh_states = []
    for seq in sequences:
        state = esn_forward(w_in_f, w_res_f, None, seq)
        tanh_states.append(state)

    x_tanh = np.array([tanh_states[i] for i in train_idx])
    y_tanh = np.array([targets[i] for i in train_idx]).reshape(-1, 1)
    xtx = x_tanh.T @ x_tanh + 1e-2 * np.eye(50)
    w_out_tanh = np.linalg.solve(xtx, x_tanh.T @ y_tanh).T

    tanh_preds = [float(w_out_tanh @ s) for s in tanh_states]
    tanh_train_err = np.mean([abs(tanh_preds[i] - targets[i]) / max(abs(targets[i]), 1e-10)
                              for i in train_idx])
    tanh_test_err = np.mean([abs(tanh_preds[i] - targets[i]) / max(abs(targets[i]), 1e-10)
                             for i in test_idx])
    print(f"  Train error: {tanh_train_err*100:.1f}%  Test error: {tanh_test_err*100:.1f}%")

    # --- Native int4+ReLU reservoir ---
    print("\n[2] Native int4+ReLU reservoir")
    reservoir = NativeInt4Reservoir(
        input_size=8,
        reservoir_size=50,
        connectivity=0.2,
        leak_rate=0.3,
        spectral_radius=0.9,
        max_activation=7.0,
        regularization=1e-2,
        seed=42,
    )

    sparsity = reservoir.sparsity_stats()
    print(f"  W_in sparsity: {sparsity['w_in_sparsity']*100:.1f}%, "
          f"ternary: {sparsity['w_in_ternary']}")
    print(f"  W_res sparsity: {sparsity['w_res_sparsity']*100:.1f}%, "
          f"ternary: {sparsity['w_res_ternary']}")

    reservoir.train(
        [sequences[i] for i in train_idx],
        [targets[i] for i in train_idx],
    )

    native_preds = [reservoir.predict(seq) for seq in sequences]
    native_train_err = np.mean([abs(native_preds[i] - targets[i]) / max(abs(targets[i]), 1e-10)
                                for i in train_idx])
    native_test_err = np.mean([abs(native_preds[i] - targets[i]) / max(abs(targets[i]), 1e-10)
                               for i in test_idx])
    print(f"  Train error: {native_train_err*100:.1f}%  Test error: {native_test_err*100:.1f}%")

    # --- Activation sparsity analysis ---
    print("\n[3] Activation sparsity (ReLU advantage)")
    total_activations = 0
    zero_activations = 0
    for seq in sequences[:3]:
        _, states = reservoir.forward(seq, return_states=True)
        for s in states:
            total_activations += len(s)
            zero_activations += np.sum(s < 1e-6)
    act_sparsity = zero_activations / total_activations
    print(f"  Activation sparsity: {act_sparsity*100:.1f}%")
    print(f"  (AKD1000 skips {act_sparsity*100:.0f}% of MACs — free speedup)")

    # --- MAC count comparison ---
    print("\n[4] MAC count comparison")
    rs = 50
    is_ = 8
    dense_macs = rs * is_ + rs * rs  # W_in*x + W_res*s
    sparse_macs = int(dense_macs * (1 - sparsity['w_res_sparsity']) *
                      (1 - act_sparsity))
    print(f"  Dense MACs per step: {dense_macs}")
    print(f"  Sparse MACs per step: ~{sparse_macs} "
          f"({sparse_macs/dense_macs*100:.0f}% of dense)")
    print(f"  Savings: {(1 - sparse_macs/dense_macs)*100:.0f}%")

    # --- Per-case results ---
    print("\n[5] Per-case results")
    case_results = []
    for i, (kappa, gamma) in enumerate(cases):
        label = "train" if i in train_idx else "test"
        target = targets[i]
        p_native = native_preds[i]
        err = abs(p_native - target) / max(abs(target), 1e-10) * 100
        case_results.append({
            "case": i,
            "kappa": kappa,
            "gamma": gamma,
            "split": label,
            "target_d_star": target,
            "pred_native_int4": p_native,
            "pred_tanh_f64": tanh_preds[i],
            "err_native_pct": err,
        })
        print(f"  ({kappa:.1f}, {gamma:.0f}) {label}: target={target:.4f} "
              f"native={p_native:.4f} err={err:.1f}%")

    # --- Save results ---
    output = {
        "provenance": {
            "generator": "control/metalforge_npu/scripts/native_int4_reservoir.py",
            "method": "Native int4+ReLU reservoir (hardware-native design)",
            "date": time.strftime("%Y-%m-%d"),
            "numpy_version": np.__version__,
        },
        "config": {
            "input_size": 8,
            "reservoir_size": 50,
            "connectivity": 0.2,
            "leak_rate": 0.3,
            "spectral_radius": 0.9,
            "max_activation": 7.0,
            "weight_type": "ternary_int4",
            "activation": "bounded_relu",
        },
        "sparsity": sparsity,
        "activation_sparsity": float(act_sparsity),
        "macs": {
            "dense_per_step": dense_macs,
            "sparse_per_step": sparse_macs,
            "savings_pct": float((1 - sparse_macs / dense_macs) * 100),
        },
        "cases": case_results,
        "summary": {
            "native_train_err_pct": float(native_train_err * 100),
            "native_test_err_pct": float(native_test_err * 100),
            "tanh_train_err_pct": float(tanh_train_err * 100),
            "tanh_test_err_pct": float(tanh_test_err * 100),
        },
    }

    out_path = os.path.join(RESULTS_DIR, "native_reservoir_baseline.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # --- Pass/fail ---
    print("\n" + "=" * 60)
    s = output["summary"]
    checks = [
        ("Native train error < 50%", s["native_train_err_pct"] < 50),
        ("Native test error < 100%", s["native_test_err_pct"] < 100),
        ("Weights are ternary", sparsity["w_in_ternary"] and sparsity["w_res_ternary"]),
        ("Activation sparsity > 15%", act_sparsity > 0.15),
        ("MAC savings > 50%", output["macs"]["savings_pct"] > 50),
    ]
    all_pass = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}")

    if all_pass:
        print("\n  ALL CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
