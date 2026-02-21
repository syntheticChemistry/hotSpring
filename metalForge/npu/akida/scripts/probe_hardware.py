"""
AKD1000 hardware probe and characterization script.

Discovers hardware, enumerates NP mesh, maps a trivial model for baseline
latency/power measurements, then maps our ESN-equivalent model and benchmarks.

Prerequisites:
  - pip install akida
  - /dev/akida0 permissions (chmod 666 or udevadm trigger)
  - AKD1000 PCIe board installed (lspci shows 1e7c:bca1)
"""

import sys
import time
import numpy as np

try:
    import akida
except ImportError:
    print("ERROR: akida package not installed. Run: pip install akida")
    sys.exit(1)


def probe_devices():
    """Enumerate all Akida hardware and virtual devices."""
    print(f"Akida SDK version: {akida.__version__}")
    print()

    hw_devices = akida.devices()
    print(f"Hardware devices: {len(hw_devices)}")
    for i, d in enumerate(hw_devices):
        print(f"  [{i}] version={d.version}")
        nps = list(d.mesh.nps)
        np_types = {}
        for np_info in nps:
            for t in np_info.types:
                np_types[t.name] = np_types.get(t.name, 0) + 1
        print(f"    NPs: {len(nps)} total")
        for name, count in sorted(np_types.items()):
            print(f"      {name}: {count}")
    print()

    vdev = akida.AKD1000()
    nps = list(vdev.mesh.nps)
    np_types = {}
    for np_info in nps:
        for t in np_info.types:
            np_types[t.name] = np_types.get(t.name, 0) + 1
    print(f"Virtual AKD1000: version={vdev.version}")
    print(f"  NPs: {len(nps)} total")
    for name, count in sorted(np_types.items()):
        print(f"    {name}: {count}")

    return hw_devices, vdev


def build_esn_equivalent_model():
    """
    Build an Akida model approximating the ESN readout with injected weights.

    AKD1000 finding: InputConvolutional with >3 channels falls to software.
    Only FullyConnected maps to hardware (FNP3 node). Weights must be non-zero
    for hardware mapping (all-zero weights rejected by HRC).
    """
    print("\n--- Building ESN-equivalent Akida model ---")

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
    proj.set_variable("weights", rng.integers(-7, 8, size=(1,1,8,50), dtype=np.int8))
    proj.set_variable("threshold", np.zeros(50, dtype=np.int32))
    readout = model.get_layer("readout")
    readout.set_variable("weights", rng.integers(-7, 8, size=(1,1,50,1), dtype=np.int8))

    model.summary()
    return model


def benchmark_model(model, device, n_iter=100):
    """Map model to device and benchmark inference."""
    print(f"\n--- Benchmarking on {device.version} ---")

    try:
        model.map(device)
    except Exception as e:
        print(f"  Mapping failed: {e}")
        return None

    model.summary()

    input_shape = (1,) + tuple(model.input_shape)
    dummy_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)

    model.forward(dummy_input)

    start = time.perf_counter()
    for _ in range(n_iter):
        model.forward(dummy_input)
    elapsed = time.perf_counter() - start

    fps = n_iter / elapsed
    us_per_inference = (elapsed / n_iter) * 1e6

    print(f"\n  Results ({n_iter} iterations):")
    print(f"    FPS: {fps:.1f}")
    print(f"    Latency: {us_per_inference:.1f} μs/inference")
    print(f"    Total: {elapsed*1000:.1f} ms")

    try:
        stats = model.statistics
        print(f"    Statistics: {stats}")
    except Exception:
        pass

    return {"fps": fps, "latency_us": us_per_inference}


def main():
    print("=" * 60)
    print("  AKD1000 Hardware Probe — metalForge/npu/akida")
    print("=" * 60)
    print()

    hw_devices, vdev = probe_devices()

    try:
        model = build_esn_equivalent_model()
    except Exception as e:
        print(f"\nModel build failed: {e}")
        print("Trying simpler model construction...")
        model = None

    if model is None:
        print("\nFalling back to pre-built model download approach.")
        print("This requires the quantizeml/cnn2snn toolchain.")
        return

    results = {}

    vdev_result = benchmark_model(model, vdev)
    if vdev_result:
        results["virtual"] = vdev_result

    for i, hw_dev in enumerate(hw_devices):
        hw_result = benchmark_model(model, hw_dev)
        if hw_result:
            results[f"hw_{i}"] = hw_result

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name}: {r['fps']:.1f} FPS, {r['latency_us']:.1f} μs/inference")

    if not hw_devices:
        print("\n  NOTE: No hardware detected. Possible causes:")
        print("    1. /dev/akida0 permissions: run 'sudo chmod 666 /dev/akida0'")
        print("       or 'sudo udevadm trigger' to apply udev rules")
        print("    2. Driver version mismatch with SDK v" + akida.__version__)
        print("    3. Board not properly seated in PCIe slot")


if __name__ == "__main__":
    main()
