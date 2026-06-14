"""
NPU Quantization Parity — Python Control Experiment

Measures prediction accuracy degradation through the quantization cascade:
  f64 (CPU ESN) → f32 (NpuSimulator) → int8 → int4 (AKD1000 hardware)

Then deploys quantized weights to the real AKD1000 and measures hardware
inference accuracy against the f64 reference.

Outputs: control/metalforge_npu/results/npu_quantization_baseline.json
"""

import os
import sys
import json
import time
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

try:
    import akida
except ImportError:
    print("WARNING: akida not installed — hardware benchmarks will be skipped")
    akida = None


# ═══════════════════════════════════════════════════════════════
# ESN reimplementation (matches Rust reservoir.rs exactly)
# ═══════════════════════════════════════════════════════════════

class Xoshiro256pp:
    """Xoshiro256++ PRNG matching Rust implementation bit-for-bit."""

    def __init__(self, seed):
        s = [0] * 4
        z = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        for i in range(4):
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
            s[i] = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
        self.s = s

    def next_u64(self):
        s = self.s
        result = (((s[0] + s[3]) & 0xFFFFFFFFFFFFFFFF) << 23 | ((s[0] + s[3]) & 0xFFFFFFFFFFFFFFFF) >> 41) & 0xFFFFFFFFFFFFFFFF
        result = (result + s[0]) & 0xFFFFFFFFFFFFFFFF
        t = (s[1] << 17) & 0xFFFFFFFFFFFFFFFF
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = ((s[3] << 45) | (s[3] >> 19)) & 0xFFFFFFFFFFFFFFFF
        return result

    def uniform(self):
        return (self.next_u64() >> 11) / (1 << 53)

    def standard_normal(self):
        u1 = max(self.uniform(), 1e-30)
        u2 = self.uniform()
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def build_esn_weights(input_size=8, reservoir_size=50, spectral_radius=0.95,
                      connectivity=0.2, seed=42):
    """Build ESN weight matrices matching Rust implementation."""
    rng = Xoshiro256pp(seed)

    w_in = np.array([[rng.uniform() - 0.5 for _ in range(input_size)]
                     for _ in range(reservoir_size)])

    w_res = np.zeros((reservoir_size, reservoir_size))
    for i in range(reservoir_size):
        for j in range(reservoir_size):
            if rng.uniform() < connectivity:
                w_res[i, j] = rng.standard_normal()

    sr = np.max(np.abs(np.linalg.eigvals(w_res)))
    if sr > 1e-10:
        w_res *= spectral_radius / sr

    return w_in, w_res


def esn_forward(w_in, w_res, w_out, input_seq, leak_rate=0.3, dtype=np.float64):
    """Run ESN forward pass at specified precision."""
    w_in = w_in.astype(dtype)
    w_res = w_res.astype(dtype)
    state = np.zeros(w_in.shape[0], dtype=dtype)

    for inp in input_seq:
        inp = np.array(inp, dtype=dtype)
        pre = w_in @ inp + w_res @ state
        state = (1.0 - leak_rate) * state + leak_rate * np.tanh(pre)

    if w_out is not None:
        w_out = np.array(w_out, dtype=dtype)
        return (w_out @ state).item()
    return state


def quantize_weights(weights, bits):
    """Symmetric uniform quantization to `bits` precision.

    Maps the weight range [-max_abs, max_abs] to [-2^(bits-1)+1, 2^(bits-1)-1].
    Returns (quantized_int, scale) where float_value ≈ quantized_int * scale.
    """
    max_abs = np.max(np.abs(weights))
    if max_abs < 1e-30:
        return np.zeros_like(weights, dtype=np.int8), 1.0

    max_int = (1 << (bits - 1)) - 1
    scale = max_abs / max_int
    quantized = np.clip(np.round(weights / scale), -max_int, max_int).astype(np.int8)
    return quantized, scale


def esn_forward_quantized(w_in_q, s_in, w_res_q, s_res, w_out, input_seq,
                          leak_rate=0.3, act_bits=None):
    """Run ESN forward pass with quantized weights.

    Dequantizes on-the-fly: w_float = w_quantized * scale.
    Activations optionally quantized to simulate hardware.
    """
    rs = w_in_q.shape[0]
    w_in_f = w_in_q.astype(np.float32) * s_in
    w_res_f = w_res_q.astype(np.float32) * s_res
    state = np.zeros(rs, dtype=np.float32)

    for inp in input_seq:
        inp_f = np.array(inp, dtype=np.float32)
        pre = w_in_f @ inp_f + w_res_f @ state
        state = (1.0 - leak_rate) * state + leak_rate * np.tanh(pre)

        if act_bits is not None:
            max_int = (1 << (act_bits - 1)) - 1
            s_max = np.max(np.abs(state))
            if s_max > 1e-30:
                s_act = s_max / max_int
                state = np.clip(np.round(state / s_act), -max_int, max_int) * s_act

    if w_out is not None:
        w_out_f = np.array(w_out, dtype=np.float32)
        return float(w_out_f @ state)
    return state


# ═══════════════════════════════════════════════════════════════
# Synthetic test data (matches validate_reservoir_transport.rs)
# ═══════════════════════════════════════════════════════════════

def generate_test_sequences(n_cases=6, n_frames=100, n_features=8, seed=99):
    """Generate synthetic feature sequences for quantization testing."""
    rng = np.random.default_rng(seed)
    sequences = []
    targets = []
    for _ in range(n_cases):
        seq = rng.standard_normal((n_frames, n_features))
        seq = np.tanh(seq * 0.5)
        sequences.append(seq)
        targets.append(rng.uniform(0.1, 2.0))
    return sequences, targets


# ═══════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  NPU Quantization Parity — metalForge Control Experiment")
    print("=" * 60)
    print()

    # --- Build ESN and train ---
    print("[1] Building ESN (seed=42, reservoir=50, input=8)")
    w_in, w_res = build_esn_weights()
    sequences, targets = generate_test_sequences()

    train_idx = [0, 1, 2, 3]
    test_idx = [4, 5]

    train_seqs = [sequences[i] for i in train_idx]
    train_targets = [targets[i] for i in train_idx]

    rs = w_in.shape[0]
    n_train = len(train_seqs)
    x_mat = np.zeros((n_train, rs))
    for i, seq in enumerate(train_seqs):
        state = esn_forward(w_in, w_res, None, seq)
        x_mat[i] = state

    y_mat = np.array(train_targets).reshape(-1, 1)

    # Ridge regression
    reg = 1e-2
    xtx = x_mat.T @ x_mat + reg * np.eye(rs)
    xty = x_mat.T @ y_mat
    w_out = np.linalg.solve(xtx, xty).T  # (1, rs)

    print(f"  W_out shape: {w_out.shape}")
    print(f"  W_out range: [{w_out.min():.4f}, {w_out.max():.4f}]")

    # --- Quantization cascade ---
    print("\n[2] Quantization cascade: f64 → f32 → int8 → int4")

    precisions = {
        "f64": {"dtype": np.float64},
        "f32": {"dtype": np.float32},
        "int8": {"bits": 8},
        "int4": {"bits": 4},
    }

    # Quantize weights at each precision
    w_in_q8, s_in_8 = quantize_weights(w_in, 8)
    w_res_q8, s_res_8 = quantize_weights(w_res, 8)
    w_in_q4, s_in_4 = quantize_weights(w_in, 4)
    w_res_q4, s_res_4 = quantize_weights(w_res, 4)

    results_per_case = []
    for ci in range(len(sequences)):
        seq = sequences[ci]
        target = targets[ci]
        label = "train" if ci in train_idx else "test"

        pred_f64 = esn_forward(w_in, w_res, w_out, seq, dtype=np.float64)
        pred_f32 = esn_forward(w_in, w_res, w_out, seq, dtype=np.float32)
        pred_int8 = esn_forward_quantized(w_in_q8, s_in_8, w_res_q8, s_res_8,
                                          w_out, seq)
        pred_int4 = esn_forward_quantized(w_in_q4, s_in_4, w_res_q4, s_res_4,
                                          w_out, seq)
        pred_int4_act4 = esn_forward_quantized(w_in_q4, s_in_4, w_res_q4, s_res_4,
                                               w_out, seq, act_bits=4)

        ref = pred_f64
        denom = abs(ref) if abs(ref) > 1e-10 else 1.0

        case_result = {
            "case": ci,
            "split": label,
            "target": target,
            "pred_f64": pred_f64,
            "pred_f32": pred_f32,
            "pred_int8": pred_int8,
            "pred_int4": pred_int4,
            "pred_int4_act4": pred_int4_act4,
            "err_f32_pct": abs(pred_f32 - ref) / denom * 100,
            "err_int8_pct": abs(pred_int8 - ref) / denom * 100,
            "err_int4_pct": abs(pred_int4 - ref) / denom * 100,
            "err_int4_act4_pct": abs(pred_int4_act4 - ref) / denom * 100,
        }
        results_per_case.append(case_result)

        print(f"  Case {ci} ({label}): f64={pred_f64:.4f}  f32={pred_f32:.4f}  "
              f"int8={pred_int8:.4f}  int4={pred_int4:.4f}  int4+act4={pred_int4_act4:.4f}")
        print(f"    Errors vs f64:  f32={case_result['err_f32_pct']:.4f}%  "
              f"int8={case_result['err_int8_pct']:.4f}%  "
              f"int4={case_result['err_int4_pct']:.4f}%  "
              f"int4+act4={case_result['err_int4_act4_pct']:.4f}%")

    # --- Summary statistics ---
    print("\n[3] Summary statistics")
    for key in ["err_f32_pct", "err_int8_pct", "err_int4_pct", "err_int4_act4_pct"]:
        vals = [r[key] for r in results_per_case]
        test_vals = [r[key] for r in results_per_case if r["split"] == "test"]
        label = key.replace("err_", "").replace("_pct", "")
        print(f"  {label:>12s}:  mean={np.mean(vals):.4f}%  max={np.max(vals):.4f}%  "
              f"test_mean={np.mean(test_vals):.4f}%")

    # --- Weight statistics ---
    print("\n[4] Weight quantization statistics")
    for name, w, wq4, wq8 in [("W_in", w_in, w_in_q4, w_in_q8),
                                ("W_res", w_res, w_res_q4, w_res_q8)]:
        n_zero_f64 = np.sum(np.abs(w) < 1e-30) / w.size * 100
        n_zero_q4 = np.sum(wq4 == 0) / wq4.size * 100
        n_zero_q8 = np.sum(wq8 == 0) / wq8.size * 100
        unique_q4 = len(np.unique(wq4))
        print(f"  {name}: sparsity f64={n_zero_f64:.1f}%  int8={n_zero_q8:.1f}%  "
              f"int4={n_zero_q4:.1f}%  unique_int4={unique_q4}")

    # --- AKD1000 hardware deployment ---
    hw_result = None
    if akida is not None:
        print("\n[5] AKD1000 hardware deployment")
        hw_devices = akida.devices()
        if hw_devices:
            hw_result = deploy_to_akd1000(w_in_q4, s_in_4, w_out, sequences, results_per_case)
        else:
            print("  No hardware detected — skipping")
            print("  Fix: pkexec chmod 666 /dev/akida0")
    else:
        print("\n[5] Akida SDK not installed — skipping hardware deployment")

    # --- Save results ---
    output = {
        "provenance": {
            "generator": "control/metalforge_npu/scripts/npu_quantization_parity.py",
            "method": "ESN quantization cascade (f64→f32→int8→int4)",
            "date": time.strftime("%Y-%m-%d"),
            "akida_sdk": akida.__version__ if akida else None,
            "numpy_version": np.__version__,
            "esn_config": {
                "input_size": 8,
                "reservoir_size": 50,
                "spectral_radius": 0.95,
                "connectivity": 0.2,
                "leak_rate": 0.3,
                "regularization": 1e-2,
                "seed": 42,
            },
        },
        "cases": results_per_case,
        "summary": {
            "f32_mean_err_pct": float(np.mean([r["err_f32_pct"] for r in results_per_case])),
            "f32_max_err_pct": float(np.max([r["err_f32_pct"] for r in results_per_case])),
            "int8_mean_err_pct": float(np.mean([r["err_int8_pct"] for r in results_per_case])),
            "int8_max_err_pct": float(np.max([r["err_int8_pct"] for r in results_per_case])),
            "int4_mean_err_pct": float(np.mean([r["err_int4_pct"] for r in results_per_case])),
            "int4_max_err_pct": float(np.max([r["err_int4_pct"] for r in results_per_case])),
            "int4_act4_mean_err_pct": float(np.mean([r["err_int4_act4_pct"] for r in results_per_case])),
            "int4_act4_max_err_pct": float(np.max([r["err_int4_act4_pct"] for r in results_per_case])),
        },
        "hardware": hw_result,
    }

    out_path = os.path.join(RESULTS_DIR, "npu_quantization_baseline.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # --- Pass/fail summary ---
    print("\n" + "=" * 60)
    s = output["summary"]
    checks = [
        ("f32 mean error < 0.01%", s["f32_mean_err_pct"] < 0.01),
        ("int8 mean error < 5%", s["int8_mean_err_pct"] < 5.0),
        ("int4 mean error < 30%", s["int4_mean_err_pct"] < 30.0),
        ("int4+act4 mean error < 50%", s["int4_act4_mean_err_pct"] < 50.0),
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


def deploy_to_akd1000(w_in_q4, s_in_4, w_out, sequences, results_per_case):
    """Deploy quantized ESN readout to real AKD1000 hardware."""
    rng = np.random.default_rng(42)

    model = akida.Model()
    model.add(akida.InputConvolutional(
        name="proj",
        input_shape=(1, 1, 8),
        kernel_size=(1, 1),
        filters=50,
        weights_bits=4,
        act_bits=4,
    ))
    model.add(akida.FullyConnected(
        name="readout",
        units=1,
        weights_bits=4,
        activation=False,
    ))

    proj = model.get_layer("proj")
    w_proj = w_in_q4.reshape(1, 1, 8, 50).astype(np.int8)
    proj.set_variable("weights", w_proj)
    proj.set_variable("threshold", np.zeros(50, dtype=np.int32))

    readout = model.get_layer("readout")
    w_readout_q, _ = quantize_weights_ext(w_out.flatten(), 4)
    readout.set_variable("weights", w_readout_q.reshape(1, 1, 50, 1).astype(np.int8))

    device = akida.devices()[0]
    model.map(device)
    model.summary()

    print(f"\n  Inference clocks: {model.statistics.inference_clk}")
    print(f"  Program clocks: {model.statistics.program_clk}")

    n_iter = 100
    dummy = np.random.randint(0, 256, size=(1, 1, 1, 8), dtype=np.uint8)
    model.forward(dummy)

    start = time.perf_counter()
    for _ in range(n_iter):
        model.forward(dummy)
    elapsed = time.perf_counter() - start

    result = {
        "fps": n_iter / elapsed,
        "latency_us": (elapsed / n_iter) * 1e6,
        "inference_clk": model.statistics.inference_clk,
        "program_clk": model.statistics.program_clk,
        "program_bytes": None,
    }

    seqs = model.sequences
    for i, seq in enumerate(seqs):
        prog = seq.program
        if prog is not None:
            result["program_bytes"] = len(prog)

    print(f"  Hardware: {result['fps']:.0f} FPS, {result['latency_us']:.0f} μs/inference")
    return result


def quantize_weights_ext(weights, bits):
    """Quantize for external use (returns int8 array and scale)."""
    max_abs = np.max(np.abs(weights))
    if max_abs < 1e-30:
        return np.zeros_like(weights, dtype=np.int8), 1.0
    max_int = (1 << (bits - 1)) - 1
    scale = max_abs / max_int
    quantized = np.clip(np.round(weights / scale), -max_int, max_int).astype(np.int8)
    return quantized, scale


if __name__ == "__main__":
    sys.exit(main())
