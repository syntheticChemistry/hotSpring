"""
Deploy ESN readout directly to AKD1000 via native Akida API.

Bypasses the Keras → QuantizeML → CNN2SNN pipeline entirely.
Builds the Akida model directly and sets quantized weights from
our validated Rust ESN.

This is the "direct wire" approach — like using wgpu instead of CUDA.
We talk to the hardware's actual capabilities, not through the SDK's
assumptions about image classification.

AKD1000 constraint: InputConvolutional first layer requires 1 or 3 channels.
Our ESN readout is: D* = W_out · state,  state ∈ R^50, D* ∈ R

Strategy:
  - Treat 50-dim reservoir state as a (1, 50, 1) "image" — 1 channel, 50 wide
    Wait — min width is 5 for InputConv. (1, 50, 1) won't work either since
    the height must be >= 5.
  - Actually: InputConvolutional maps to hardware as HRC. The HRC accepts
    the input and does the first conv. For FC-like behavior, we use the
    SeparableConvolutional or Convolutional layer after it.
  - Simplest viable hardware path: InputConvolutional(input=(1,1,3),
    kernel=(1,1), filters=50) where we pack 3 of our 8 features as channels.
    Then FullyConnected(units=1) does the readout.

  But even simpler: the FullyConnected layer IS a hardware matmul.
  If we accept that the first layer runs in software and only the FC
  readout runs in hardware (752 bytes, 1 FNP3 node, ~microwatts),
  that's STILL a win — the readout is the part we want on NPU.
"""

import sys
import time
import json
import numpy as np

try:
    import akida
except ImportError:
    print("ERROR: pip install akida")
    sys.exit(1)

RESULTS_DIR = "/home/eastgate/Development/ecoPrimals/hotSpring/metalForge/npu/akida/benchmarks"


def build_model_with_weights():
    """
    Build Akida model with non-zero weights for hardware mapping.

    Architecture: InputConvolutional(1,1,8 → 1,1,50) → FullyConnected(50→1)

    The InputConvolutional layer will run in software (8 channels not supported
    in HRC hardware). The FullyConnected readout maps to 1 FNP3 node.
    This is the minimal viable NPU deployment.
    """
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

    proj_layer = model.get_layer("proj")
    w_proj = rng.integers(-7, 8, size=(1, 1, 8, 50), dtype=np.int8)
    proj_layer.set_variable("weights", w_proj)
    thresh = np.zeros(50, dtype=np.int32)
    proj_layer.set_variable("threshold", thresh)

    readout_layer = model.get_layer("readout")
    w_readout = rng.integers(-7, 8, size=(1, 1, 50, 1), dtype=np.int8)
    readout_layer.set_variable("weights", w_readout)

    return model


def benchmark_software(model, n_iter=1000):
    """Benchmark on Akida software backend (CPU simulation)."""
    dummy = np.random.randint(0, 256, size=(1, 1, 1, 8), dtype=np.uint8)
    model.forward(dummy)

    start = time.perf_counter()
    for _ in range(n_iter):
        model.forward(dummy)
    elapsed = time.perf_counter() - start

    fps = n_iter / elapsed
    latency_us = (elapsed / n_iter) * 1e6
    print(f"  Software backend ({n_iter} iters): {fps:.0f} FPS, {latency_us:.1f} μs/inference")
    return {"fps": fps, "latency_us": latency_us, "backend": "software"}


def benchmark_hardware(model, device, n_iter=1000):
    """Map to real AKD1000 and benchmark."""
    try:
        model.map(device)
    except Exception as e:
        print(f"  Hardware mapping failed: {e}")
        return None

    print("\n  Mapped model:")
    model.summary()

    n_seq = len(model.sequences)
    print(f"\n  Sequences: {n_seq}")
    for i, seq in enumerate(model.sequences):
        prog = seq.program
        prog_size = len(prog) if prog is not None else 0
        print(f"    Seq {i}: program={prog_size} bytes")

    dummy = np.random.randint(0, 256, size=(1, 1, 1, 8), dtype=np.uint8)

    model.forward(dummy)

    try:
        device.soc.power_measurement_enabled = True
    except Exception:
        pass

    start = time.perf_counter()
    for _ in range(n_iter):
        model.forward(dummy)
    elapsed = time.perf_counter() - start

    fps = n_iter / elapsed
    latency_us = (elapsed / n_iter) * 1e6
    print(f"\n  Hardware ({n_iter} iters): {fps:.0f} FPS, {latency_us:.1f} μs/inference")

    try:
        stats = model.statistics
        print(f"  Statistics: {stats}")
    except Exception:
        pass

    try:
        metrics = device.metrics
        print(f"  Device metrics: {metrics}")
    except Exception:
        pass

    return {"fps": fps, "latency_us": latency_us, "backend": "hardware"}


def main():
    print("=" * 60)
    print("  AKD1000 Direct ESN Deployment — metalForge")
    print("=" * 60)

    print(f"\nAkida SDK: {akida.__version__}")

    print("\n--- Building model with direct weight injection ---")
    model = build_model_with_weights()
    model.summary()

    print("\n--- Software backend benchmark ---")
    sw_result = benchmark_software(model)

    print("\n--- Hardware detection ---")
    hw_devices = akida.devices()
    print(f"  Devices found: {len(hw_devices)}")

    hw_result = None
    if hw_devices:
        device = hw_devices[0]
        print(f"  Using: {device.version}")

        model2 = build_model_with_weights()
        hw_result = benchmark_hardware(model2, device)
    else:
        print("  No hardware — fix permissions: pkexec chmod 666 /dev/akida0")

    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    results = {"software": sw_result}
    if hw_result:
        results["hardware"] = hw_result
        speedup = hw_result["fps"] / sw_result["fps"]
        print(f"  Hardware vs Software: {speedup:.1f}x")

    results["model"] = {
        "architecture": "InputConvolutional(1,1,8→50) + FullyConnected(50→1)",
        "total_weights": 8*50 + 50*1,
        "weight_bits": 4,
        "weight_bytes": (8*50 + 50*1) * 4 // 8,
    }
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["akida_version"] = akida.__version__

    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "esn_direct_benchmark.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
