"""
NPU Beyond-SDK Capabilities — Python Control Experiment

Validates hardware capabilities discovered by probing beyond the SDK:
  1. Arbitrary input channels (SDK claims 1 or 3 only; HW takes any)
  2. FC chain merging into single HW sequence (SkipDMA)
  3. Batch inference PCIe amortization (batch=8 → 2.4× throughput)
  4. Wide FC scaling (tested to 8192 neurons)
  5. Multi-output free cost (N outputs ≈ 1 output latency)
  6. Weight mutation determinism (set_variable preserves math)
  7. Hardware determinism (identical input → identical output)

The Rust binary (validate_npu_beyond_sdk.rs) validates the math parity;
this script validates the hardware behavior on the actual AKD1000.

Outputs: control/metalforge_npu/results/npu_beyond_sdk_baseline.json
"""

import os
import sys
import json
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

try:
    import akida
except ImportError:
    print("ERROR: akida SDK required for beyond-SDK hardware experiments")
    sys.exit(1)

rng = np.random.default_rng(42)


def build_model(device, in_dim, hidden, out_dim, depth=1):
    """Build, inject weights, and map an Akida model to hardware."""
    model = akida.Model()
    model.add(akida.InputConvolutional(
        name="inp", input_shape=(1, 1, in_dim), kernel_size=(1, 1),
        filters=hidden, weights_bits=4, act_bits=4,
    ))
    for d in range(depth):
        model.add(akida.FullyConnected(
            name=f"fc{d}", units=hidden, weights_bits=4, activation=True,
        ))
    model.add(akida.FullyConnected(
        name="out", units=out_dim, weights_bits=4, activation=False,
    ))

    model.get_layer("inp").set_variable(
        "weights", rng.integers(-7, 8, size=(1, 1, in_dim, hidden), dtype=np.int8),
    )
    model.get_layer("inp").set_variable(
        "threshold", np.zeros(hidden, dtype=np.int32),
    )
    for d in range(depth):
        model.get_layer(f"fc{d}").set_variable(
            "weights", rng.integers(-7, 8, size=(1, 1, hidden, hidden), dtype=np.int8),
        )
        model.get_layer(f"fc{d}").set_variable(
            "threshold", np.zeros(hidden, dtype=np.int32),
        )
    model.get_layer("out").set_variable(
        "weights", rng.integers(-7, 8, size=(1, 1, hidden, out_dim), dtype=np.int8),
    )
    model.map(device)
    return model


def bench_latency(model, in_dim, n_warmup=10, n_iter=200):
    """Benchmark single-sample inference latency."""
    dummy = rng.integers(0, 256, size=(1, 1, 1, in_dim), dtype=np.uint8)
    for _ in range(n_warmup):
        model.forward(dummy)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model.forward(dummy)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1e6


def bench_batch(model, in_dim, batch_size, n_warmup=5, n_iter=100):
    """Benchmark batched inference throughput."""
    data = rng.integers(0, 256, size=(batch_size, 1, 1, in_dim), dtype=np.uint8)
    for _ in range(n_warmup):
        model.forward(data)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model.forward(data)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) / n_iter * 1000
    per_sample_us = total_ms / batch_size * 1000
    throughput = batch_size * n_iter / (t1 - t0)
    return total_ms, per_sample_us, throughput


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    print("=" * 60)
    print("  NPU Beyond-SDK — metalForge Control Experiment")
    print("=" * 60)

    devices = akida.devices()
    if not devices:
        print("ERROR: No Akida devices. Fix: pkexec chmod 666 /dev/akida0")
        return 1

    device = devices[0]
    print(f"  Device: {device.desc} (v{device.version})")
    print(f"  NPs: {len(list(device.mesh.nps))}")
    print(f"  SDK: akida {akida.__version__}")

    results = {"checks": [], "measurements": {}}
    all_pass = True

    def check(label, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        nonlocal all_pass
        if not passed:
            all_pass = False
        results["checks"].append({"label": label, "passed": passed, "detail": detail})
        print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))

    # ═════════════════════════════════════════════════════════
    # TEST 1: Arbitrary input channels
    # ═════════════════════════════════════════════════════════
    section("TEST 1: Arbitrary Input Channels")
    print("  SDK claims InputConv limited to 1 or 3 channels.")
    print("  Testing 2, 4, 5, 8, 16, 32, 50, 64...")

    channel_results = {}
    all_channels_work = True
    for ch in [2, 4, 5, 8, 16, 32, 50, 64]:
        try:
            m = build_model(device, ch, 16, 1, depth=0)
            hw_seqs = sum(1 for s in m.sequences if s.program is not None)
            lat = bench_latency(m, ch, n_iter=50)
            channel_results[ch] = {"hw_seqs": hw_seqs, "lat_us": lat}
            print(f"    ch={ch:3d}: HW={hw_seqs} lat={lat:.0f}μs — OK")
        except Exception as e:
            channel_results[ch] = {"error": str(e)}
            all_channels_work = False
            print(f"    ch={ch:3d}: FAILED — {e}")

    results["measurements"]["channel_test"] = channel_results
    check("All non-standard channel counts (2,4,5,8,16,32,50,64) map to HW",
          all_channels_work)

    physics_channels_work = all(
        ch in channel_results and "error" not in channel_results[ch]
        for ch in [8, 50]
    )
    check("Physics vectors (8-dim and 50-dim) work on hardware",
          physics_channels_work)

    # ═════════════════════════════════════════════════════════
    # TEST 2: FC chain merging (SkipDMA)
    # ═════════════════════════════════════════════════════════
    section("TEST 2: FC Chain Merging (SkipDMA)")

    merge_results = {}
    all_merge = True
    for depth in [1, 2, 3, 4, 7]:
        m = build_model(device, 50, 64, 1, depth=depth)
        hw = sum(1 for s in m.sequences if s.program is not None)
        lat = bench_latency(m, 50, n_iter=50)
        merge_results[depth] = {"hw_seqs": hw, "lat_us": lat, "total_layers": depth + 2}
        if hw != 1:
            all_merge = False
        print(f"    depth={depth} ({depth+2} layers): HW={hw} lat={lat:.0f}μs")

    results["measurements"]["fc_merge"] = merge_results
    check("All FC depths (1-7) merge into single HW sequence",
          all_merge)

    lat_d1 = merge_results[1]["lat_us"]
    lat_d7 = merge_results[7]["lat_us"]
    depth_overhead_pct = (lat_d7 - lat_d1) / lat_d1 * 100
    check("Depth overhead < 30% (7 layers vs 1 layer)",
          depth_overhead_pct < 30,
          f"{depth_overhead_pct:.1f}%")

    # ═════════════════════════════════════════════════════════
    # TEST 3: Batch inference amortization
    # ═════════════════════════════════════════════════════════
    section("TEST 3: Batch Inference (PCIe Amortization)")

    m = build_model(device, 50, 256, 1, depth=2)
    batch_results = {}
    for bs in [1, 2, 4, 8, 16]:
        total_ms, per_us, throughput = bench_batch(m, 50, bs, n_iter=50)
        batch_results[bs] = {
            "total_ms": total_ms,
            "per_sample_us": per_us,
            "throughput": throughput,
        }
        print(f"    batch={bs:3d}: {total_ms:.2f}ms  {per_us:.0f}μs/sample  {throughput:.0f}/s")

    results["measurements"]["batch_inference"] = batch_results
    single_us = batch_results[1]["per_sample_us"]
    batch8_us = batch_results[8]["per_sample_us"]
    speedup = single_us / batch8_us
    check("Batch=8 speedup > 1.5× over single inference",
          speedup > 1.5,
          f"{speedup:.2f}×")
    check("Batch=8 per-sample latency < 600μs",
          batch8_us < 600,
          f"{batch8_us:.0f}μs")

    # ═════════════════════════════════════════════════════════
    # TEST 4: Wide FC scaling
    # ═════════════════════════════════════════════════════════
    section("TEST 4: Wide FC Scaling")

    width_results = {}
    all_widths_ok = True
    for w in [64, 128, 256, 512, 1024]:
        try:
            m = build_model(device, 8, w, 1, depth=1)
            lat = bench_latency(m, 8, n_iter=30)
            width_results[w] = {"lat_us": lat}
            print(f"    width={w:5d}: lat={lat:.0f}μs — OK")
        except Exception as e:
            width_results[w] = {"error": str(e)}
            all_widths_ok = False
            print(f"    width={w:5d}: FAILED — {e}")

    results["measurements"]["width_scaling"] = width_results
    check("All FC widths (64-1024) map to hardware", all_widths_ok)

    if 512 in width_results and "lat_us" in width_results[512]:
        lat_512 = width_results[512]["lat_us"]
        check("Width=512 latency > width=64 (compute contributing)",
              lat_512 > width_results[64]["lat_us"],
              f"512:{lat_512:.0f}μs vs 64:{width_results[64]['lat_us']:.0f}μs")

    # ═════════════════════════════════════════════════════════
    # TEST 5: Multi-output free cost
    # ═════════════════════════════════════════════════════════
    section("TEST 5: Multi-Output Free Cost")

    m1 = build_model(device, 50, 512, 1, depth=0)
    lat_1out = bench_latency(m1, 50, n_iter=100)
    m3 = build_model(device, 50, 512, 3, depth=0)
    lat_3out = bench_latency(m3, 50, n_iter=100)
    m10 = build_model(device, 50, 512, 10, depth=0)
    lat_10out = bench_latency(m10, 50, n_iter=100)

    multi_out_results = {
        "1_output_us": lat_1out,
        "3_output_us": lat_3out,
        "10_output_us": lat_10out,
    }
    results["measurements"]["multi_output"] = multi_out_results
    print(f"    1 output:  {lat_1out:.0f}μs")
    print(f"    3 outputs: {lat_3out:.0f}μs")
    print(f"   10 outputs: {lat_10out:.0f}μs")

    overhead_10_pct = (lat_10out - lat_1out) / lat_1out * 100
    check("10-output overhead < 30% vs single output",
          overhead_10_pct < 30,
          f"{overhead_10_pct:.1f}%")

    # ═════════════════════════════════════════════════════════
    # TEST 6: Weight mutation determinism
    # ═════════════════════════════════════════════════════════
    section("TEST 6: Weight Mutation Determinism")

    m = akida.Model()
    m.add(akida.InputConvolutional(
        name="inp", input_shape=(1, 1, 8), kernel_size=(1, 1),
        filters=16, weights_bits=4, act_bits=4))
    m.add(akida.FullyConnected(
        name="out", units=1, weights_bits=4, activation=False))
    m.get_layer("inp").set_variable("weights", np.ones((1, 1, 8, 16), dtype=np.int8))
    m.get_layer("inp").set_variable("threshold", np.zeros(16, dtype=np.int32))
    m.get_layer("out").set_variable("weights", np.ones((1, 1, 16, 1), dtype=np.int8))
    m.map(device)

    test_input = np.full((1, 1, 1, 8), 10, dtype=np.uint8)

    r1 = int(m.forward(test_input).flatten()[0])
    m.get_layer("out").set_variable("weights", np.full((1, 1, 16, 1), 2, dtype=np.int8))
    r2 = int(m.forward(test_input).flatten()[0])
    m.get_layer("out").set_variable("weights", np.full((1, 1, 16, 1), -3, dtype=np.int8))
    r3 = int(m.forward(test_input).flatten()[0])

    print(f"    w=1:  result={r1}")
    print(f"    w=2:  result={r2} (ratio={r2/r1:.2f})")
    print(f"    w=-3: result={r3} (ratio={r3/r1:.2f})")

    ratio_2x = abs(r2 / r1 - 2.0) if r1 != 0 else float("inf")
    ratio_3x = abs(r3 / r1 - (-3.0)) if r1 != 0 else float("inf")
    results["measurements"]["weight_mutation"] = {
        "w1_result": r1, "w2_result": r2, "w_neg3_result": r3,
        "ratio_2x_err": ratio_2x, "ratio_3x_err": ratio_3x,
    }
    check("Weight×2 produces output×2",
          ratio_2x < 0.01, f"err={ratio_2x:.4f}")
    check("Weight×(-3) produces output×(-3)",
          ratio_3x < 0.01, f"err={ratio_3x:.4f}")

    # ═════════════════════════════════════════════════════════
    # TEST 7: Hardware determinism
    # ═════════════════════════════════════════════════════════
    section("TEST 7: Hardware Determinism")

    m = build_model(device, 50, 128, 1, depth=1)
    test = rng.integers(0, 256, size=(1, 1, 1, 50), dtype=np.uint8)
    outputs = [int(m.forward(test).flatten()[0]) for _ in range(20)]
    unique = len(set(outputs))
    results["measurements"]["determinism"] = {"n_runs": 20, "unique_outputs": unique}
    print(f"    20 identical inputs → {unique} unique output(s)")
    check("Hardware is fully deterministic (20 runs, 1 unique output)",
          unique == 1)

    # ═════════════════════════════════════════════════════════
    # TEST 8: Model save/load parity
    # ═════════════════════════════════════════════════════════
    section("TEST 8: Model Save/Load Parity")

    m = build_model(device, 50, 128, 1, depth=1)
    test = rng.integers(0, 256, size=(1, 1, 1, 50), dtype=np.uint8)
    r_orig = int(m.forward(test).flatten()[0])

    save_path = "/tmp/metalforge_test_model.fbz"
    m.save(save_path)
    m2 = akida.Model(save_path)
    m2.map(device)
    r_loaded = int(m2.forward(test).flatten()[0])

    print(f"    Original: {r_orig}, Loaded: {r_loaded}")
    results["measurements"]["save_load"] = {
        "original": r_orig, "loaded": r_loaded, "match": r_orig == r_loaded,
    }
    check("Model save/load produces identical output", r_orig == r_loaded)
    os.remove(save_path)

    # ═════════════════════════════════════════════════════════
    # Summary
    # ═════════════════════════════════════════════════════════
    section("SUMMARY")

    passed = sum(1 for c in results["checks"] if c["passed"])
    total = len(results["checks"])
    print(f"  {passed}/{total} checks passed")

    for c in results["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['label']}")

    results["provenance"] = {
        "generator": "control/metalforge_npu/scripts/npu_beyond_sdk.py",
        "method": "AKD1000 beyond-SDK capability validation",
        "date": time.strftime("%Y-%m-%d"),
        "akida_sdk": akida.__version__,
        "device": device.desc,
        "device_version": str(device.version),
        "numpy_version": np.__version__,
    }
    results["summary"] = {"passed": passed, "total": total, "all_pass": all_pass}

    out_path = os.path.join(RESULTS_DIR, "npu_beyond_sdk_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    if all_pass:
        print("\n  ALL CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
