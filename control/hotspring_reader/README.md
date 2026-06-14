# hotspring_reader — Python interface to hotSpring output

**Dependency**: NumPy only. No pip install, no setup.py, no pandas.

Load any hotSpring JSON output (ensemble manifests, per-config measurements,
analysis summaries) into NumPy arrays. Export to TSV for paper tables or
`numpy.loadtxt()`.

## Quick Start

```python
import sys
sys.path.insert(0, "control/hotspring_reader")
import hotspring_reader as hs

# Load ensemble-level analysis (from chuna_analyze)
ana = hs.load_analysis("results/analysis.json")
print(f"<P> = {ana['plaquette']['mean']:.6f} ± {ana['plaquette']['error']:.2e}")

# Export to TSV for paper table
hs.to_tsv(ana, "results/analysis.tsv")
```

## Loading Measurements

```python
# Load all per-config JSONs from a directory (chuna_measure output)
meas = hs.load_measurements("data/b6.0_L8/measurements/")

# NumPy arrays ready for analysis
plaq = meas["plaquette"]       # float64 [N]
t0   = meas["flow_t0"]         # float64 [N] (NaN where absent)
corr = meas["correlator"]      # float64 [N, Nt] or None (HVP)

# Timeseries TSV for quick plotting
hs.timeseries_to_tsv(meas, "timeseries.tsv")
```

## Loading Ensemble Manifests

```python
# Load ensemble.json (from chuna_generate)
ens = hs.load_ensemble("data/b6.0_L8/ensemble.json")
print(f"beta={ens['beta']}, dims={ens['dims']}, {len(ens['configs'])} configs")
print(f"plaquettes: {ens['config_plaquettes']}")
```

## Flow Curves

```python
# Extract gradient flow curves for all configs
flow = hs.extract_flow_curves(meas)
if flow:
    import matplotlib.pyplot as plt
    for i in range(flow["t2_e"].shape[0]):
        plt.plot(flow["t"], flow["t2_e"][i], alpha=0.3)
    plt.axhline(0.3, ls="--", color="k", label="t² E = 0.3 (t₀)")
    plt.xlabel("flow time t")
    plt.ylabel("t² ⟨E⟩")
    plt.show()
```

## Substrate Parity Comparison

Compare Python control output against Rust binary output:

```bash
python compare_substrates.py --paper=43 \
    --python-result=../gradient_flow/results/gradient_flow_control.json \
    --rust-result=../../results/analysis.json
```

Tolerances are physically derived per paper (finite-size, quadrature, grid resolution).
Exit code 0 = all pass. See `TOLERANCES` dict in `compare_substrates.py` for derivations.

## Available Fields

### `load_measurements()` returns:

| Key | Type | Source |
|-----|------|--------|
| `plaquette` | `float64[N]` | gauge |
| `polyakov_abs` | `float64[N]` | gauge |
| `polyakov_re` | `float64[N]` | gauge |
| `polyakov_im` | `float64[N]` | gauge |
| `action_density` | `float64[N]` | gauge |
| `flow_t0` | `float64[N]` | flow (NaN if absent) |
| `flow_w0` | `float64[N]` | flow (NaN if absent) |
| `topo_charge` | `float64[N]` | topology (NaN if absent) |
| `condensate` | `float64[N]` | fermion (NaN if absent) |
| `correlator` | `float64[N, Nt]` | HVP (None if absent) |
| `hvp_integral` | `float64[N]` | HVP (NaN if absent) |
| `cg_iterations` | `int64[N]` | HVP (-1 if absent) |
| `cg_residual` | `float64[N]` | HVP (NaN if absent) |
| `wall_seconds` | `float64[N]` | timing |
| `trajectory` | `int64[N]` | config index |
| `wilson_loops` | `list` or None | W(R,T) grid |

### `load_analysis()` returns:

Observables as dicts with `mean`, `error`, `tau_int`, `tau_int_error`, `n_values`.
Susceptibilities as plain floats. All fields match `chuna_analyze` JSON exactly.

### `load_ensemble()` returns:

Ensemble metadata with `dims` as `int64[4]`, `config_plaquettes` as `float64[N]`,
plus all manifest fields as native Python types.
