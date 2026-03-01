# Handoff: Nautilus Shell + Brain Architecture → toadStool

**Date:** March 1, 2026
**From:** hotSpring
**To:** toadStool / barracuda team
**License:** AGPL-3.0-only
**Covers:** v0.6.15, bingoCube-nautilus v0.1.0, brain architecture (Exp 024–029)

---

## Executive Summary

- **Nautilus Shell** is an evolutionary reservoir computing architecture where populations of BingoCube boards replace temporal recurrence. Validated at **5.3% LOO generalization error** on Exp 024+028 QCD data. Quenched→dynamical transfer achieves **540× cost reduction** for proxy CG prediction.
- **NautilusBrain** integration module provides the API for embedding the Nautilus Shell into the NPU worker thread alongside the existing ESN. ESN handles fast temporal dynamics; Nautilus handles slow structural adaptation.
- **4-layer brain architecture** (RTX 3090 + Titan V + CPU + NPU) is running in Exp 029 with NPU-steered adaptive β insertion. See `specs/BIOMEGATE_BRAIN_ARCHITECTURE.md`.
- **Concept edge detection** via LOO cross-validation identifies β regions where physics models break down — the computational equivalent of mapping phase boundaries.
- **Drift monitor** implements the N_e·s boundary from population genetics: warns when the board population is too small for selection to outweigh genetic drift.

---

## Part 1: Nautilus Shell — What It Is

### Architecture

The Nautilus Shell (`ecoPrimals/primalTools/bingoCube/nautilus/`) is a feed-forward reservoir computing system that uses evolutionary generations as its time step instead of temporal recurrence.

```
Input stream ("caller")     Board population ("boards")     Linear readout ("player")
─────────────────────       ───────────────────────────     ─────────────────────────
β, plaquette, CG, ...  →   24 boards project input    →    Ridge regression extracts
                            into 24 response vectors        predictions from concatenated
                            via BLAKE3 hashing              response vectors
```

Key properties:
- **Feed-forward only** — no recurrence, maps directly to AKD1000 int4 weights
- **Integer arithmetic** — board cells are u8 (1–75), column ranges enforce constraint
- **Human-verifiable** — board weights are literal bingo boards, inspectable by hand
- **Portable** — entire evolutionary history serializes to JSON, transfers between instances
- **Mergeable** — shells from different instances combine via population union + re-evolution

### Crate Structure

```
bingoCube/nautilus/
├── src/
│   ├── lib.rs           # Public API
│   ├── response.rs      # Board projection: input → ResponseVector via BLAKE3
│   ├── population.rs    # Board ensemble, fitness evaluation (Pearson correlation)
│   ├── evolution.rs     # Selection (elitism/tournament/roulette), crossover, mutation
│   ├── readout.rs       # Linear readout: ridge regression, Cholesky solver
│   ├── shell.rs         # NautilusShell: history, transfer, merge
│   ├── constraints.rs   # DriftMonitor (N_e·s), EdgeSeeder (directed mutagenesis)
│   └── brain.rs         # NautilusBrain: integration API for NPU worker thread
├── examples/
│   ├── shell_lifecycle.rs         # Within/between instance demo
│   ├── live_qcd_prediction.rs     # Exp 024+028 data replay
│   └── quenched_to_dynamical.rs   # 540× cost reduction demo
└── Cargo.toml           # Dependencies: bingocube-core, serde, serde_json
```

### Validated Results

| Metric | Value | Source |
|--------|-------|--------|
| Training MSE (CG) | 1.1% mean error | Exp 024+028, 25 β points |
| LOO generalization | 5.3% mean CG error | Leave-one-out cross-validation |
| Quenched→dynamical transfer | 4.4% LOO CG error | Quenched features only |
| Cost reduction | **540×** | Quenched gauge update vs dynamical HMC |
| Board population | 24 boards × 5×5 grid | Configurable |
| Evolution generations | 20 per cycle | Configurable |
| Concept edges detected | β ≈ 5.6, 6.0 | LOO error spikes at phase boundaries |
| Tests | 26 | Unit tests across all modules |

---

## Part 2: NautilusBrain Integration API

The `NautilusBrain` struct is the integration point for the NPU worker thread.

### Key Types

```rust
pub struct BetaObservation {
    pub beta: f64,
    pub quenched_plaq: Option<f64>,
    pub quenched_plaq_var: Option<f64>,
    pub plaquette: f64,
    pub cg_iters: f64,
    pub acceptance: f64,
    pub delta_h_abs: f64,
    pub anderson_r: Option<f64>,
    pub anderson_lambda_min: Option<f64>,
}

pub struct NautilusBrain {
    pub config: NautilusBrainConfig,
    pub shell: NautilusShell,
    pub observations: Vec<BetaObservation>,
    pub drift: DriftMonitor,
    pub concept_edges: Vec<f64>,
    pub trained: bool,
}
```

### API Methods

| Method | Purpose |
|--------|---------|
| `NautilusBrain::new(config, instance)` | Create fresh brain with random board population |
| `NautilusBrain::from_shell(config, shell, instance)` | Bootstrap from inherited shell (cross-run) |
| `brain.observe(obs)` | Add a measured β point |
| `brain.train()` | Evolve boards + fit readout (when ≥ min_training_points) |
| `brain.predict_dynamical(quenched_features)` | Predict CG/plaquette/acceptance from quenched proxies |
| `brain.estimate_cg(beta)` | Quick CG estimate for β-steering decisions |
| `brain.screen_candidates(betas)` | Rank candidate β values by predicted information gain |
| `brain.detect_concept_edges()` | LOO cross-validation → identify breakdown regions |
| `brain.is_drifting()` | Check N_e·s boundary, recommend population increase |
| `brain.export_shell()` | Serialize for cross-instance transfer |
| `brain.to_json() / from_json()` | Full brain state persistence |

### Integration with Existing ESN

The Nautilus Shell complements the ESN — they operate at different timescales:

| Aspect | ESN (existing) | Nautilus Shell (new) |
|--------|---------------|---------------------|
| Timescale | Within-run (per trajectory) | Cross-run (per generation) |
| Adaptation | Weight updates (linear readout) | Population evolution + readout |
| Memory | Hidden state (temporal) | Evolutionary history (structural) |
| Input | Trajectory features | β-point aggregated observables |
| Hardware | GPU (large RS) or NPU (streaming) | CPU (evolution) → NPU (inference) |

---

## Part 3: ESN Absorption Guide

### What toadStool Should Absorb from hotSpring

| Component | Location | Priority |
|-----------|----------|----------|
| `reservoir.rs` ESN module | `barracuda/src/md/reservoir.rs` | High — 11-head ESN, cross-substrate |
| ESN WGSL shader | `esn_reservoir_update.wgsl` | High — GPU reservoir dispatch |
| NPU worker pattern | `production_dynamical_mixed.rs` | High — spawn_npu_worker thread |
| NautilusBrain API | `bingoCube/nautilus/src/brain.rs` | Medium — add to `ai.*` JSON-RPC |
| Drift monitor | `bingoCube/nautilus/src/constraints.rs` | Medium — population health check |
| Edge seeder | `bingoCube/nautilus/src/constraints.rs` | Low — directed mutagenesis |

### NPU Evolution: Current State

- **11-head ESN** with QUENCHED_LENGTH and QUENCHED_THERM heads
- **Disagreement logging** — when ESN and NPU predictions diverge, log for later investigation
- **Concept edge detection** — LOO error spikes at β ≈ 5.6, 6.0 indicate phase structure boundaries
- **Cross-run bootstrap** — Exp 029 loads weights from Exp 024+028 combined training data
- **Adaptive β steering** — NPU prioritizes β points by predicted information gain

---

## Part 4: AKD1000 Hardware Mapping

The Nautilus Shell is designed for direct AKD1000 deployment:

| Shell Component | AKD1000 Mapping |
|----------------|-----------------|
| Board cells (u8, 1–75) | int4 weights (4-bit quantization of cell values) |
| Response vector (f64) | int8 activations (post-projection quantization) |
| Population (24 boards) | 24 FC layers in SkipDMA chain |
| Linear readout | Final FC layer (ridge regression weights) |
| Fitness evaluation | CPU-side (evolution is offline) |

The feed-forward design avoids the AKD1000's lack of on-chip recurrence. Board population inference maps to a single forward pass through the SkipDMA chain.

---

## Part 5: Action Items

### toadStool Priority Actions

1. **Add `bingocube-nautilus` as workspace dependency**
   - Wire `bingoCube/nautilus/` into toadStool's `Cargo.toml` workspace
   - The crate has no unsafe code, no GPU dependencies — pure Rust + serde

2. **Absorb NautilusBrain into `ai.*` JSON-RPC namespace**
   - `ai.nautilus.observe` → `brain.observe(obs)`
   - `ai.nautilus.train` → `brain.train()`
   - `ai.nautilus.predict` → `brain.predict_dynamical(features)`
   - `ai.nautilus.edges` → `brain.detect_concept_edges()`
   - `ai.nautilus.export` → `brain.export_shell()`

3. **Integrate drift monitor into MultiDevicePool scheduler**
   - When `brain.is_drifting()` returns true, log warning and optionally increase population
   - This parallels Anderson's N_e·s boundary (see thesis Chapter 3)

4. **Map board populations to AKD1000 via akida-models pipeline**
   - Quantize board response vectors from f64 → int4
   - Use existing SkipDMA FC chain pattern from `akida-models`
   - Benchmark: latency target < 100μs per population inference

5. **Wire ESN reservoir module from hotSpring**
   - `md/reservoir.rs` + `esn_reservoir_update.wgsl` → `barracuda::ai::esn`
   - Keep GPU and NPU dispatch paths

### Lower Priority

6. **Edge seeder integration** — when concept edges detected, generate edge-targeted boards
7. **Shell merge protocol** — enable combining shells from different nodes (distributed evolution)
8. **Quenched cost estimator** — toadStool scheduler can use Nautilus to estimate CG cost before committing GPU time

---

## Cross-References

| Document | Location | Relevance |
|----------|----------|-----------|
| Brain architecture spec | `specs/BIOMEGATE_BRAIN_ARCHITECTURE.md` | Full 4-layer design |
| Nautilus Shell paper | `whitePaper/gen3/baseCamp/11_bingocube_nautilus_shell.md` | Technical details + origin story |
| Constrained evolution thesis | `whitePaper/gen3/thesis/03_theoretical_framework.md` | Theoretical foundation |
| Exp 024 results | `experiments/024_HMC_PARAMETER_SWEEP.md` | Training data source |
| Exp 028 results | `experiments/028_BRAIN_CONCURRENT_PIPELINE.md` | 4-layer brain validation |
| Exp 029 (running) | `experiments/029_NPU_STEERING_PRODUCTION.md` | Live NPU-steered production |

---

**License:** AGPL-3.0-only — All source, data, and documentation freely available.
