"""
Build, train, quantize, and convert an ESN-equivalent model for AKD1000.

The Akida deployment pipeline:
  1. Build a Keras model matching our ESN topology
  2. Train with synthetic data (or real reservoir outputs)
  3. Quantize to int4/int8 via QuantizeML
  4. Convert to Akida model via CNN2SNN
  5. Map to AKD1000 (virtual or real) and benchmark

Our ESN readout is: D* = W_out · state
where state ∈ R^50 (reservoir state) and D* ∈ R (scalar output).

On AKD1000 this maps to: FullyConnected(input=50, output=1, weights_bits=4)
"""

import os
import sys
import time
import json
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    import tf_keras as keras
    import akida
    from quantizeml.models import quantize, QuantizationParams
    from cnn2snn import convert
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install akida quantizeml cnn2snn tensorflow tf-keras")
    sys.exit(1)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_synthetic_esn_data(n_samples=500, reservoir_size=50, seed=42):
    """
    Generate synthetic data mimicking ESN reservoir states → D* prediction.

    Uses a linear mapping with noise to simulate what the trained readout does.
    The actual W_out weights are random but fixed — what matters is the quantized
    model can approximate this mapping.
    """
    rng = np.random.default_rng(seed)

    w_true = rng.standard_normal(reservoir_size).astype(np.float32) * 0.1
    bias_true = 0.5

    states = rng.standard_normal((n_samples, reservoir_size)).astype(np.float32)
    states = np.tanh(states)  # reservoir states are post-tanh

    d_star = states @ w_true + bias_true
    d_star += rng.standard_normal(n_samples).astype(np.float32) * 0.01

    return states, d_star, w_true, bias_true



def build_akida_compatible_model(input_dim=50, output_dim=1):
    """
    Build a tf_keras model that maps to Akida hardware.

    Akida 1.0 InputConvolutional only accepts 1 or 3 input channels, and
    CNN2SNN Reshape only allows (N,) → (1,1,N). Since our ESN state is
    50-dimensional, we use Dense layers throughout. QuantizeML explicitly
    supports Dense as a first layer for uint8 quantization.

    Architecture: Dense(50→16, ReLU) → Dense(16→1)
    This is functionally equivalent to the ESN readout: a linear projection
    from reservoir state to D*, with a hidden layer for quantization headroom.
    """
    model = keras.Sequential([
        keras.layers.Dense(16, activation="relu", input_shape=(input_dim,),
                           name="proj"),
        keras.layers.Dense(output_dim, name="readout"),
    ], name="esn_readout")
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    print("=" * 60)
    print("  ESN Model: Build → Train → Quantize → Convert → Map")
    print("=" * 60)

    # --- 1. Generate data ---
    print("\n[1] Generating synthetic ESN data...")
    states, d_star, w_true, bias_true = generate_synthetic_esn_data()
    n_train = 400
    x_train, y_train = states[:n_train], d_star[:n_train]
    x_test, y_test = states[n_train:], d_star[n_train:]
    print(f"  Train: {n_train}, Test: {len(x_test)}")
    print(f"  D* range: [{d_star.min():.3f}, {d_star.max():.3f}]")

    # --- 2. Build and train Keras model ---
    print("\n[2] Building Akida-compatible Keras model...")
    model = build_akida_compatible_model(input_dim=50, output_dim=1)
    model.summary()

    print("\n  Training...")
    model.fit(x_train, y_train, epochs=50, batch_size=32,
              validation_data=(x_test, y_test), verbose=0)

    train_loss = model.evaluate(x_train, y_train, verbose=0)
    test_loss = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Train MSE: {train_loss:.6f}")
    print(f"  Test MSE:  {test_loss:.6f}")

    # --- 3. Quantize ---
    print("\n[3] Quantizing model...")
    try:
        qparams = QuantizationParams(
            input_weight_bits=8,
            weight_bits=4,
            activation_bits=4,
            output_bits=8,
            input_dtype="uint8",
        )
        qmodel = quantize(model, qparams=qparams)
        qmodel.compile(optimizer="adam", loss="mse")

        print("  Fine-tuning quantized model...")
        qmodel.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

        qtrain_loss = qmodel.evaluate(x_train, y_train, verbose=0)
        qtest_loss = qmodel.evaluate(x_test, y_test, verbose=0)
        print(f"  Quantized Train MSE: {qtrain_loss:.6f}")
        print(f"  Quantized Test MSE:  {qtest_loss:.6f}")
    except Exception as e:
        print(f"  Quantization failed: {e}")
        import traceback; traceback.print_exc()
        print("  Proceeding with float model conversion...")
        qmodel = model

    # --- 4. Convert to Akida ---
    print("\n[4] Converting to Akida model...")
    try:
        akida_model = convert(qmodel)
        akida_model.summary()
    except Exception as e:
        print(f"  Conversion failed: {e}")
        import traceback; traceback.print_exc()
        return

    # --- 5. Map and benchmark ---
    print("\n[5] Mapping to virtual AKD1000...")
    vdev = akida.AKD1000()
    try:
        akida_model.map(vdev)
        akida_model.summary()
    except Exception as e:
        print(f"  Mapping failed: {e}")

    # Benchmark on CPU backend (Akida software simulation)
    print("\n[6] Benchmarking (Akida software backend)...")
    x_uint8 = np.clip(
        (x_test * 50 + 128), 0, 255
    ).astype(np.uint8)

    start = time.perf_counter()
    n_iter = len(x_test)
    for i in range(n_iter):
        akida_model.forward(x_uint8[i:i+1])
    elapsed = time.perf_counter() - start

    print(f"  Software backend: {n_iter/elapsed:.0f} FPS, "
          f"{elapsed/n_iter*1e6:.0f} us/inference")

    # Try hardware
    hw_devices = akida.devices()
    if hw_devices:
        print(f"\n[7] Mapping to real AKD1000...")
        hw_dev = hw_devices[0]
        try:
            akida_model.map(hw_dev)
            akida_model.summary()

            start = time.perf_counter()
            for i in range(n_iter):
                akida_model.forward(x_uint8[i:i+1])
            elapsed = time.perf_counter() - start

            print(f"  Hardware: {n_iter/elapsed:.0f} FPS, "
                  f"{elapsed/n_iter*1e6:.0f} us/inference")
            print(f"  Statistics: {akida_model.statistics}")
        except Exception as e:
            print(f"  Hardware mapping failed: {e}")
    else:
        print("\n[7] No hardware device found — skipping HW benchmark")
        print("    Fix: sudo chmod 666 /dev/akida0  (or sudo udevadm trigger)")

    # --- Save results ---
    results = {
        "model": "esn_readout_equivalent",
        "input_dim": 50,
        "output_dim": 1,
        "keras_train_mse": float(train_loss),
        "keras_test_mse": float(test_loss),
        "akida_sdk_version": akida.__version__,
        "hardware_detected": len(hw_devices) > 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    results_path = os.path.join(RESULTS_DIR, "esn_model_benchmark.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Save Akida model
    model_path = os.path.join(RESULTS_DIR, "esn_readout.fbz")
    akida_model.save(model_path)
    print(f"  Akida model saved to {model_path}")


if __name__ == "__main__":
    main()
