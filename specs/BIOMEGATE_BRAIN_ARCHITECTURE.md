# Spec: biomeGate Brain Architecture

**Status:** DRAFT
**Date:** February 28, 2026
**License:** AGPL-3.0-only
**Depends on:** Exp 024 (dynamical production), Exp 025 (multi-GPU), Exp 026 (4D proxy)

---

## Problem

The production dynamical pipeline (`production_dynamical_mixed.rs`) is
serial: GPU does CG, CPU blocks, NPU predicts, CPU blocks, GPU does
next CG. During a 40-second CG solve on the RTX 3090:

| Substrate | State | Capacity |
|-----------|-------|----------|
| RTX 3090 | 10% occupied (CG on 8⁴) | 90% idle shader cores |
| Titan V | Off | 12 GB HBM2, 5120 CUDA cores |
| CPU | Blocked on GPU readback | 32 cores, 256 GB RAM |
| NPU | Idle | 80 NPUs, 30 mW |

Four substrates. One working. Three dark.

## Design Principle: Treat biomeGate as a Brain

A human learning to walk doesn't dedicate the entire brain to leg
coordination. The cerebellum handles routine motor patterns at low
power. The cortex handles novel situations and planning. The motor
cortex executes specific movements. When something unexpected happens
(tripping), the cerebellum escalates to conscious processing.

biomeGate should work the same way:

| Brain Region | biomeGate Substrate | Role |
|-------------|--------------------|----|
| Cerebellum | NPU (AKD1000, 30 mW) | Continuous monitoring, learned predictions, fast reflex |
| Motor cortex | RTX 3090 (370W peak) | Primary execution (CG solver, HMC trajectory) |
| Pre-motor cortex | Titan V (250W peak) | Prepare next movement (quenched pre-therm, warm configs) |
| Prefrontal cortex | CPU (280W TDP, 256 GB) | Deep analysis (eigendecomposition, ESN retrain, planning) |

## Architecture

### Thread Model

```
Main thread (brainstem):
  Coordinates all substrates. Runs the beta-scan loop.
  Checks interrupt channel at each CG batch boundary.

NPU worker thread (cerebellum):
  Existing thread (spawn_npu_worker). Extended with:
  - Continuous residual monitoring (Layer 1)
  - Attention state machine (Layer 4)
  - Asynchronous interrupt emission

Titan V worker thread (pre-motor):
  New thread. Owns a second GpuF64 instance on the Titan V.
  Receives next-beta commands, runs quenched pre-therm, sends
  warm gauge configs back to main thread.

CPU cortex thread (prefrontal):
  New thread. Runs Anderson 4D / Wegner block eigendecomposition
  concurrently with GPU work. Sends proxy features to NPU thread.
```

### Channel Topology

```
                    ┌─────────────────────────┐
                    │   Main Thread            │
                    │   (brainstem)            │
                    └──┬──────┬──────┬────────┘
                       │      │      │
              npu_tx/rx│      │      │interrupt_rx
                       │      │      │
                    ┌──▼──┐   │   ┌──▼──────────┐
                    │ NPU │   │   │ Interrupt    │
                    │ Wkr │◄──┼───│ Channel      │
                    └──┬──┘   │   └──────────────┘
                       │      │
              proxy_tx │      │titan_tx/rx
                       │      │
                    ┌──▼──┐ ┌─▼───────┐
                    │ CPU │ │ Titan V  │
                    │ Ctx │ │ Pre-Mtr  │
                    └─────┘ └─────────┘
```

| Channel | Type | Direction | Payload |
|---------|------|-----------|---------|
| `npu_tx` / `npu_rx` | `mpsc::channel` | Main → NPU → Main | `NpuRequest` / `NpuResponse` (existing) |
| `residual_tx` | `mpsc::Sender` | Main → NPU | `CgResidualUpdate { iter, rz_new, batch_wall_us }` |
| `interrupt_rx` | `mpsc::Receiver` | NPU → Main | `BrainInterrupt { kind, severity, suggestion }` |
| `titan_tx` / `titan_rx` | `mpsc::channel` | Main → Titan → Main | `TitanRequest` / `TitanResponse` |
| `proxy_tx` | `mpsc::Sender` | CPU → NPU | `ProxyFeatures { r, lambda_min, ipr, phase }` |
| `cortex_tx` | `mpsc::Sender` | Main → CPU | `CortexRequest { beta, mass, plaq_var }` |

### Data Types

```rust
struct CgResidualUpdate {
    iteration: usize,
    rz_new: f64,
    batch_wall_us: u64,
    beta: f64,
    traj_idx: usize,
}

enum AttentionState {
    Green,   // nominal — no intervention
    Yellow,  // alert — increase monitoring frequency
    Red,     // interrupt — send corrective action to main thread
}

enum InterruptKind {
    CgDiverging,       // residual increasing
    CgStalled,         // residual plateau (no progress)
    CgSlowerThanPred,  // converging but much slower than NPU predicted
    DeltaHExplosion,   // |ΔH| >> expected
    AcceptanceCollapse, // acceptance rate dropped sharply
    TitanPrethermFail, // Titan V pre-therm produced bad config
}

struct BrainInterrupt {
    kind: InterruptKind,
    severity: AttentionState,
    suggestion: InterruptAction,
    context: String,
}

enum InterruptAction {
    KillCg,                          // abort current CG solve
    RestartCgWithSmallerDt(f64),     // retry with different parameters
    SkipBeta,                        // abandon this beta point
    IncreaseCheckInterval(usize),    // check convergence less often
    DecreaseCheckInterval(usize),    // check convergence more often
    NoAction,                        // informational only
}

enum TitanRequest {
    PreThermalize {
        beta: f64,
        mass: f64,
        lattice: usize,
        n_quenched: usize,
        seed: u64,
    },
    Shutdown,
}

enum TitanResponse {
    WarmConfig {
        beta: f64,
        gauge_links: Vec<f64>,
        plaquette: f64,
    },
    Failed(String),
}

struct CortexRequest {
    beta: f64,
    mass: f64,
    lattice: usize,
    plaq_var: f64,
}

struct ProxyFeatures {
    beta: f64,
    level_spacing_ratio: f64,
    lambda_min: f64,
    ipr: f64,
    bandwidth: f64,
    phase: String,
    tier: u8,        // 1=3D scalar, 2=4D scalar, 3=4D Wegner
    wall_ms: f64,
}
```

## Layer 1: NPU During CG (Residual Watcher)

### Current flow

```
Main:  [submit CG batch] → [block on readback] → [check convergence] → [repeat]
NPU:   idle ────────────────────────────────────────────────────────────
```

### Proposed flow

```
Main:  [submit CG batch] → [non-blocking readback] → [send residual to NPU] → [check interrupt] → [repeat]
NPU:   [receive residual] → [classify decay curve] → [update attention] → [maybe send interrupt]
```

### Implementation

1. Replace `gpu_cg_solve_resident` call in `gpu_dynamical_hmc_trajectory_resident`
   with a new function `gpu_cg_solve_brain` that:
   - Uses the async readback pattern from `gpu_cg_solve_resident_async`
   - At each batch boundary, sends `CgResidualUpdate` to the NPU via `residual_tx`
   - Checks `interrupt_rx.try_recv()` for interrupts
   - If `BrainInterrupt` with `KillCg` arrives, breaks the CG loop early

2. Add NPU Head 15: `CG_RESIDUAL_MONITOR`
   - Input: last N residual values (sliding window)
   - Output: convergence ETA (iterations remaining) and anomaly score
   - Training data: all CG residual curves from the production run

3. NPU worker thread adds a new handler for `CgResidualUpdate`:
   - Maintains a per-trajectory residual history `Vec<(usize, f64)>`
   - After every update, feeds the history to Head 15
   - If anomaly score > threshold → emit `BrainInterrupt` on interrupt channel

### Check interval strategy

| Attention state | Check interval | Rationale |
|----------------|---------------|-----------|
| Green | 100 iterations | Normal — batch is efficient |
| Yellow | 20 iterations | Suspicious — check more often |
| Red | 5 iterations | Anomalous — prepare to abort |

The NPU can dynamically adjust `check_interval` by including it in
the `InterruptAction`.

## Layer 2: Titan V Pre-Motor

### Current flow

```
3090:  [measure β_i] → [quenched pre-therm β_{i+1}] → [dynamical therm β_{i+1}] → [measure β_{i+1}]
Titan:  idle ────────────────────────────────────────────────────────────────────────
```

### Proposed flow

```
3090:  [measure β_i] ──────────────────────→ [dynamical therm β_{i+1}] → [measure β_{i+1}]
Titan:  [quenched pre-therm β_{i+1}] → done ↗
                                        (warm config transferred)
```

### Implementation

1. At start of run, create second `GpuF64` on Titan V:
   ```rust
   std::env::set_var("HOTSPRING_GPU_ADAPTER", "titan");
   let titan_gpu = GpuF64::new().await;
   ```

2. Spawn Titan V worker thread with `titan_rx` / `titan_tx` channels.

3. When the NPU decides the next beta (adaptive steering), the main
   thread sends `TitanRequest::PreThermalize` immediately — while the
   3090 is still measuring at the current beta.

4. The Titan V thread:
   - Creates a hot-start gauge configuration
   - Runs `n_quenched` quenched HMC trajectories using
     `gpu_hmc_trajectory_streaming` on the Titan V GPU
   - Reads the gauge links back to CPU
   - Sends `TitanResponse::WarmConfig` with the warm gauge links

5. When the main thread is ready for the next beta, it checks
   `titan_rx.try_recv()`. If the warm config is ready, it uploads to
   the 3090 and skips quenched pre-therm entirely.

### Config transfer cost

| Lattice | Gauge links (f64s) | Bytes | PCIe transfer | Fraction of trajectory |
|---------|--------------------|-------|---------------|----------------------|
| 8⁴ | 4,096 × 4 × 18 = 294,912 | 2.4 MB | 0.1 ms (Gen4 x16) | 0.0003% |
| 16⁴ | 65,536 × 4 × 18 = 4,718,592 | 37.7 MB | 1.5 ms | 0.001% |
| 32⁴ | 1,048,576 × 4 × 18 = 75,497,472 | 604 MB | 24 ms | 0.003% |

Transfer cost is negligible at all scales.

## Layer 3: CPU Cortex (Proxy Pipeline)

### Current flow

```
CPU:   [blocked on GPU readback] ─────── idle ──────── [blocked again]
Proxy: [separate binary, run offline]
```

### Proposed flow

```
CPU:   [spawn cortex thread] → [receive beta] → [anderson_4d] → [send features to NPU]
GPU:   [CG solve] ──────────────────────────── [next CG solve]
NPU:   [receive proxy features] → [update Head 14 prediction]
```

### Implementation

1. Refactor `gpu_physics_proxy.rs` internals into library functions:
   - `pub fn anderson_4d_proxy(beta: f64, mass: f64, lattice: usize, plaq_var: f64) -> ProxyFeatures`
   - `pub fn potts_proxy(beta: f64, lattice: usize) -> ProxyFeatures`

2. Spawn cortex thread at run start. It receives `CortexRequest` from
   the main thread and sends `ProxyFeatures` to the NPU worker.

3. After each trajectory, the main thread sends the current beta and
   plaquette variance to the cortex thread. The cortex runs the proxy
   and delivers features before the next trajectory begins.

4. The NPU worker receives proxy features and uses them as additional
   input to Head 14 (Anderson CG predictor) for the next CG prediction.

### Timing budget

| Proxy tier | Est. wall time | Fits in 40s CG? |
|-----------|---------------|-----------------|
| 3D scalar Anderson | 200 ms | Yes (200× margin) |
| 4D scalar Anderson | 2-5 s | Yes (8-20× margin) |
| 4D Wegner block | 10-30 s | Yes (1.3-4× margin) |

Even the most expensive proxy (4D Wegner, 30s) completes within a
single CG solve (40s at 8⁴). At 32⁴ (CG ~ 8-15 min), all proxies
complete with massive margin.

## Layer 4: Attention State Machine (Interrupt Protocol)

### State transitions

```
         ┌──────────────────────────────────────────┐
         │                                          │
         ▼                                          │
  ┌─────────────┐   anomaly score > 0.3   ┌────────┴────┐
  │   GREEN     │ ──────────────────────► │   YELLOW     │
  │ (cerebellum)│                         │   (alert)    │
  └─────────────┘ ◄────────────────────── └──────┬───────┘
     ▲            3 consecutive normal            │
     │                                     anomaly > 0.7
     │                                    or CG diverging
     │            ┌─────────────┐                 │
     └────────────│    RED      │ ◄───────────────┘
      corrective  │ (interrupt) │
      action      └─────────────┘
      taken
```

### Inputs to the attention state machine

| Input | Source | Update frequency |
|-------|--------|-----------------|
| CG residual | Layer 1 (residual_tx) | Every check_interval iterations |
| Plaquette, ΔH, acceptance | Main thread (after each trajectory) | Every trajectory |
| Titan V pre-therm status | Layer 2 (titan_rx) | Once per beta point |
| Proxy features | Layer 3 (proxy_tx) | Once per trajectory |
| ESN internal state | NPU worker | Continuous |

### Decision logic

The NPU worker maintains:
- `residual_history: Vec<(usize, f64)>` — CG residual curve for current trajectory
- `trajectory_history: VecDeque<TrajectoryRecord>` — last 20 trajectories
- `attention: AttentionState` — current state
- `predictions: HashMap<String, f64>` — latest predictions from all heads

Every time a new residual arrives:
1. Append to `residual_history`
2. Run Head 15 on the history → get `convergence_eta` and `anomaly_score`
3. Compare `anomaly_score` against thresholds
4. If transition needed, emit `BrainInterrupt` on interrupt channel

Every time a trajectory completes:
1. Compare actual CG iterations to predicted
2. Compare ΔH to expected range
3. Update `trajectory_history`
4. If multiple consecutive anomalies → escalate attention

### Interrupt delivery

The main thread checks `interrupt_rx.try_recv()` at each CG batch
boundary (non-blocking). If an interrupt arrives:

```rust
match interrupt_rx.try_recv() {
    Ok(interrupt) => match interrupt.suggestion {
        InterruptAction::KillCg => break,
        InterruptAction::DecreaseCheckInterval(n) => check_interval = n,
        InterruptAction::SkipBeta => { skip_beta = true; break; }
        _ => {}
    },
    Err(TryRecvError::Empty) => {} // no interrupt, continue
    Err(TryRecvError::Disconnected) => break,
}
```

## NPU Head Expansion (14 → 15) — Gen 1 "Lizard Brain"

| Head | Name | Phase | New? |
|------|------|-------|------|
| 0-13 | (existing) | various | No |
| 14 | CG Residual Monitor | During CG | **Yes** |

Head 14 input: sliding window of last 10 residual values (normalized).
Head 14 output: two values — convergence ETA (iterations), anomaly score (0-1).

Note: This renumbers the existing head indices. The current
`ANDERSON_CG` at index 13 stays at 13. The new head is appended at 14.
`NUM_HEADS` increases from 14 to 15.

---

## Evolution: Overlapping Head Groups — Gen 2 "Developed Organism"

**Status:** DRAFT
**Date:** February 25, 2026
**Depends on:** Gen 1 brain architecture, Exp 024/028/029 data

### Motivation

Gen 1 is a lizard brain — each head has one job, one opinion. A lizard
can catch flies but can't evaluate its own certainty. It reacts but
doesn't reflect.

A developed organism has **redundant, overlapping** neural circuits.
Vision and proprioception both estimate body position. When they agree,
confidence is high. When they disagree (vestibular conflict, motion
sickness), the brain knows something is wrong — disagreement itself is
information.

The Gen 2 NPU applies this principle: multiple heads predict the **same
quantity** from **different physics models**. Their agreement measures
confidence. Their disagreement measures epistemic uncertainty — and
drives the attention state machine.

### Architecture: From Single-Purpose to Head Groups

Gen 1 (15 heads, single-purpose):

```
Reservoir ──→ [H0: priority] [H1: dt] [H2: therm] ... [H14: residual]
               each head = one job, one answer
```

Gen 2 (36 heads, 5 overlapping groups + meta-mixer):

```
Reservoir ──→ ┌── Group A: Anderson-informed (6 heads) ──┐
              ├── Group B: QCD-empirical    (6 heads) ──┤
              ├── Group C: Potts-informed   (6 heads) ──├──→ Disagreement ──→ Meta Group
              ├── Group D: Steering/Control (6 heads) ──┤
              ├── Group E: Brain/Monitor    (6 heads) ──┘
              └── Group M: Meta-mixer       (6 heads) ← reads from all
```

### Head Group Taxonomy

#### Group A: Anderson-Informed (Heads 0–5)

Trained with Anderson proxy features as **additional input context**.
These heads see the world through the lens of disorder → localization →
spectral statistics.

| Head | Name | Predicts | Training target |
|------|------|----------|-----------------|
| A0 | Anderson CG Cost | CG iterations | `actual_cg / max_cg` |
| A1 | Anderson Phase | confined/transition/deconfined | phase label from ⟨r⟩ thresholds |
| A2 | Anderson λ_min | smallest eigenvalue magnitude | `proxy.lambda_min` |
| A3 | Anderson Anomaly | is this config atypical? | `\|actual_cg - predicted_cg\| / σ` |
| A4 | Anderson Therm | has CG "thermalized"? | convergence from residual curve |
| A5 | Anderson Priority | which β needs more data? | information gain from Anderson landscape |

**Input encoding for Group A** (5D → extended with proxy features):
```
[β_norm, plaq, mass, susceptibility/1000, proxy_r_spacing]
```
The 5th input slot carries the Anderson level-spacing ratio ⟨r⟩ when
available, otherwise 0.5 (GOE midpoint). This encodes the proxy's
physics directly into the feature vector.

#### Group B: QCD-Empirical (Heads 6–11)

Trained purely on HMC observables — no proxy. These heads learn the
empirical mapping `(β, plaquette, acceptance, CG history) → prediction`
from the data itself.

| Head | Name | Predicts | Training target |
|------|------|----------|-----------------|
| B0 | QCD CG Cost | CG iterations | `actual_cg / max_cg` |
| B1 | QCD Phase | confined/transition/deconfined | phase from plaquette thresholds |
| B2 | QCD Acceptance | will trajectory be accepted? | `1 - acceptance` |
| B3 | QCD Anomaly | is this measurement unusual? | anomaly score from plaquette distribution |
| B4 | QCD Therm | has Markov chain thermalized? | convergence from plaquette window |
| B5 | QCD Priority | which β needs more data? | gap in plaquette curve (curvature) |

**Input encoding for Group B** (5D → standard):
```
[β_norm, plaq, mass, susceptibility/1000, acceptance]
```
The 5th slot carries the running acceptance rate instead of proxy
features. This group has no knowledge of Anderson or Potts — it's pure
empiricism.

#### Group C: Potts-Informed (Heads 12–17)

Trained with Z(3) Potts Monte Carlo results as input context. These
heads see the world through the Svetitsky-Yaffe universality lens.

| Head | Name | Predicts | Training target |
|------|------|----------|-----------------|
| C0 | Potts CG Cost | CG iterations | `actual_cg / max_cg` |
| C1 | Potts Phase | confined/transition/deconfined | Potts MC phase label |
| C2 | Potts β_c | critical coupling estimate | Potts susceptibility peak location |
| C3 | Potts Anomaly | does QCD deviate from Potts? | `\|qcd_phase - potts_phase\|` |
| C4 | Potts Order | order parameter prediction | Polyakov loop magnitude |
| C5 | Potts Priority | which β near transition? | Potts susceptibility gradient |

**Input encoding for Group C** (5D → with Potts magnetization):
```
[β_norm, plaq, mass, susceptibility/1000, potts_magnetization]
```

#### Group D: Steering/Control (Heads 18–23)

Decision-making heads — these read the same features but are trained
on **action targets** (what should the system do?) rather than
**observable targets** (what will happen?).

| Head | Name | Decides | Training target |
|------|------|---------|-----------------|
| D0 | Next β | which β point to measure next | information gain ranking |
| D1 | Optimal dt | integrator step size | acceptance-optimal dt from history |
| D2 | Optimal n_md | trajectory length | acceptance-optimal n_md from history |
| D3 | Check Interval | CG monitoring frequency | optimal interval from hindsight |
| D4 | Kill Decision | abort CG solve? | hindsight: was CG productive? |
| D5 | Skip Decision | skip this β point? | hindsight: was measurement useful? |

#### Group E: Brain/Monitor (Heads 24–29)

Real-time monitoring heads that operate during CG solves. Fed by the
residual stream (Layer 1).

| Head | Name | Monitors | Training target |
|------|------|----------|-----------------|
| E0 | Residual ETA | iterations to convergence | `remaining_iters / max_iters` |
| E1 | Residual Anomaly | is convergence abnormal? | residual curve anomaly score |
| E2 | Convergence Rate | log-slope of residual | `d(log r) / d(iter)` |
| E3 | Stall Detector | has CG stalled? | plateau detection |
| E4 | Divergence Detector | is CG diverging? | residual increasing flag |
| E5 | Quality Forecast | measurement quality from trajectory | `quality_score` |

#### Group M: Meta-Mixer (Heads 30–35)

These heads read the **same reservoir state** but are trained on targets
derived from **cross-group agreement**. They don't predict physics —
they predict **which group to trust**.

| Head | Name | Predicts | Training target |
|------|------|----------|-----------------|
| M0 | CG Consensus | best CG estimate | weighted average of A0, B0, C0 (weights from hindsight) |
| M1 | Phase Consensus | best phase label | majority vote of A1, B1, C1 (with confidence) |
| M2 | CG Uncertainty | epistemic uncertainty on CG | `max(A0,B0,C0) - min(A0,B0,C0)` |
| M3 | Phase Uncertainty | epistemic uncertainty on phase | disagreement count among A1, B1, C1 |
| M4 | Proxy Trust | which proxy is most reliable? | `\|proxy_pred - actual\|` hindsight score |
| M5 | Attention Level | Green/Yellow/Red recommendation | optimal attention from hindsight |

### The Disagreement Signal

After each inference pass, compute disagreement across overlapping heads:

```rust
struct HeadGroupDisagreement {
    delta_cg: f64,       // max(A0,B0,C0) - min(A0,B0,C0)
    delta_phase: f64,    // number of groups disagreeing on phase label
    delta_anomaly: f64,  // max(A3,B3,C3) - min(A3,B3,C3)
    delta_priority: f64, // max(A5,B5,C5) - min(A5,B5,C5)
}
```

These disagreement signals feed into the attention state machine:

```
             ┌──────────────────────────────────────────────┐
             │                                              │
             ▼                                              │
      ┌─────────────┐   Δ_cg > 0.3 OR Δ_phase > 0   ┌────┴────────┐
      │   GREEN     │ ──────────────────────────────► │   YELLOW     │
      │             │                                 │              │
      └─────────────┘ ◄───────────────────────────── └──────┬───────┘
         ▲            3 consecutive low-Δ                    │
         │                                          Δ_cg > 0.6
         │                                         OR M5 → Red
         │            ┌─────────────┐                       │
         └────────────│    RED      │ ◄─────────────────────┘
          corrective  │ (interrupt) │
          action      └─────────────┘
```

Gen 1 triggers on raw observables (CG diverging, ΔH explosion).
Gen 2 triggers on **model disagreement** — a fundamentally more
informative signal. If Anderson says "easy" but QCD says "hard," the
brain detects confusion before the CG solver even starts struggling.

### Cross-Run Learning: How Heads Accumulate Knowledge

The key: **all 36 heads share the same reservoir**. The reservoir is a
universal nonlinear feature extractor (fixed random weights). Only
the 36 readout vectors (36 × 50 = 1,800 floats) are trained.

```
Run N-1:
  Save: w_out (36 × 50 matrix) + trajectory JSONL

Run N:
  Load:  w_out from Run N-1  (immediate — all knowledge restored)
  Load:  trajectory JSONL from Run N-1, N-2, ... (retrain for refinement)

  During run:
    - Each head makes predictions
    - Actual outcomes recorded (CG cost, acceptance, phase, etc.)
    - After each β point, retrain readout with new data appended
    - Meta-heads (M0–M5) retrain on NEW disagreement patterns

  Save: updated w_out + trajectory JSONL → feed Run N+1
```

Over successive runs, each group specializes:
- **Group A** learns when Anderson proxy is predictive (strong coupling)
  vs when it's noisy (weak coupling where disorder mapping breaks down)
- **Group B** learns the empirical shape of this specific lattice/mass/action
- **Group C** learns where Svetitsky-Yaffe universality holds and where it fails
- **Group M** learns which group to trust in which regime

This is **dynamic programming over the physics landscape** — the
readout weights are the memoization table, and each run fills in more
of the table while correcting past errors.

### Multi-Reservoir Extension (Gen 2.5 — Future)

The current design uses a single reservoir (50 neurons, leak_rate=0.3).
All heads see the same temporal dynamics. A future extension:

```
Reservoir F (fast):  leak=0.5, spectral=0.90, size=30
  → Short memory: trajectory-level features, CG convergence curves
  → Best for: Group E (monitoring), Group D (kill/skip decisions)

Reservoir S (slow):  leak=0.1, spectral=0.99, size=30
  → Long memory: scan-level patterns, plaquette curve shape
  → Best for: Group A/B/C (physics predictions), Group D (steering)

Reservoir M (medium): leak=0.3, spectral=0.95, size=30  [current]
  → Balanced memory: general purpose
  → Best for: Group M (meta-mixing), fallback for all groups
```

Head groups can read from any reservoir or concatenated states:
```
h_fast  ∈ ℝ³⁰  ─┐
h_slow  ∈ ℝ³⁰  ─┤─→ h_concat ∈ ℝ⁹⁰ ─→ readout (36 heads × 90 = 3,240 weights)
h_medium ∈ ℝ³⁰ ─┘
```

On Akida: three separate reservoir blocks, merged at the readout layer.
Hardware supports this natively — each reservoir is an independent
spiking network, and SkipDMA handles the concatenation.

### Implementation Plan

**Phase 1: Head expansion (36 heads, single reservoir)**
1. Update `heads` module: `NUM_HEADS = 36`, named constants for all groups
2. Update `build_training_data()` to generate 36-target vectors
3. Update `ExportedWeights` serialization (backward-compatible: detect
   output_size on load, pad missing heads with zeros)
4. Add `HeadGroupDisagreement` computation after each `predict_all_heads()`
5. Wire disagreement signals into attention state machine

**Phase 2: Disagreement-driven attention**
1. Replace raw-observable triggers with disagreement triggers
2. Add `M5` (Attention Level) head to the interrupt decision logic
3. Log disagreement signals to trajectory JSONL for post-hoc analysis
4. Add `--head-group-log` CLI option for detailed per-head logging

**Phase 3: Meta-head training loop**
1. After each β point, compute hindsight: which group was closest?
2. Generate meta-head training targets from hindsight
3. Online retrain meta-heads (M0–M5) mid-run
4. Log proxy trust evolution across the scan

**Phase 4: Multi-reservoir (Gen 2.5)**
1. Refactor `EchoStateNetwork` to support reservoir pools
2. Add `ReservoirPool { reservoirs: Vec<EchoStateNetwork>, concat_size: usize }`
3. Head groups select which reservoir(s) to read from
4. Akida deployment: configure three spiking sub-networks

### Hardware Budget

| Component | Gen 1 | Gen 2 | Gen 2.5 |
|-----------|-------|-------|---------|
| Reservoir neurons | 50 | 50 | 90 (3×30) |
| Readout heads | 15 | 36 | 36 |
| Readout weights | 750 | 1,800 | 3,240 |
| Inference latency (Akida) | ~8 μs | ~12 μs | ~18 μs |
| Training time (ridge regression) | ~2 ms | ~5 ms | ~8 ms |
| Saved weights file | ~25 KB | ~60 KB | ~108 KB |

All well within Akida's 300 μs inference budget. The CG solver takes
40,000,000 μs — the NPU has a 2,000,000× speed advantage.

## Validation Criteria

| Layer | Test | Pass condition |
|-------|------|---------------|
| 1 | CG residual streaming | NPU receives residual updates every check_interval |
| 1 | Interrupt delivery | Main thread receives and acts on interrupt within 1 batch |
| 2 | Titan V pre-therm | Warm config produces valid plaquette at target beta |
| 2 | Overlap | Titan V pre-therm completes before 3090 finishes measurement |
| 3 | Proxy concurrent | Anderson 4D completes within one CG solve wall time |
| 3 | Feature delivery | NPU receives proxy features before next trajectory |
| 4 | State machine | GREEN → YELLOW on injected anomaly, YELLOW → RED on sustained |
| 4 | Kill CG | Injected divergence triggers CG abort within 2 batches |

## Nautilus Shell Integration (Gen 2.5 — Evolutionary Reservoir)

The Nautilus Shell (`primalTools/bingoCube/nautilus/`) is a complementary
subsystem that runs alongside the ESN. While the ESN handles fast within-run
temporal dynamics, the Nautilus Shell handles slow cross-run structural learning.

### Architecture

```
                   Brain Architecture
                        │
         ┌──────────────┼──────────────┐
         │              │              │
    ESN (temporal)  Nautilus (structural)  Meta-mixer
    - CG prediction   - Cross-run learning    - Which to trust
    - Phase class.    - Quenched→dynamical    - Disagreement
    - Anomaly det.    - Concept edge detect   - Attention
    - Steering        - Shell propagation     - Regime switch
         │              │              │
         └──────────────┼──────────────┘
                   NPU Worker Thread
```

### Validated Results

| Test | Result |
|------|--------|
| CG cost prediction (training, 21 β) | 1.1% mean relative error |
| CG cost prediction (LOO, 21 β) | 5.3% mean relative error |
| Quenched→dynamical transfer (LOO, 21 β) | **4.4% mean relative error** |
| Concept edge detection (β=6.131) | Correctly identified phase boundary |
| Shell serialization roundtrip | Predictions identical after deserialize |
| Instance transfer (2 machines) | Lineage preserved, evolution continues |
| Shell merge (2 populations) | Best boards from both, diversity maintained |
| Drift monitor (N_e·s boundary) | Detects when selection < drift |
| Edge seeding (directed mutagenesis) | Generates constraint-valid boards at edges |

### Quenched→Dynamical Cost Reduction

The Nautilus Shell trained on quenched pretherm data (pure gauge, ~10 seconds
per β, zero CG cost) can predict dynamical CG solver iterations (with fermions,
~90 minutes per β) at 4.4% LOO error. This enables:

- Pre-screening β points before committing dynamical GPU budget
- Estimating wall time for resource allocation
- Identifying concept edges for priority measurement
- Guiding adaptive steering before any dynamical data exists
- Cost ratio: 540× cheaper than dynamical measurement

### Integration API (NautilusBrain)

```rust
// Create brain subsystem alongside ESN
let nautilus = NautilusBrain::new(config, "northgate-exp030");

// Feed observations after each beta measurement
nautilus.observe(BetaObservation { beta, plaquette, cg_iters, ... });

// Train shell (20 generations per cycle)
nautilus.train();

// Predict CG cost from quenched proxy
let (cg, plaq, acc) = nautilus.predict_dynamical(5.5, Some(quenched_plaq));

// Screen candidate betas by information value
let ranked = nautilus.screen_candidates(&[4.5, 5.0, 5.5, 6.0]);

// Detect concept edges (expensive, run between measurement cycles)
let edges = nautilus.detect_concept_edges();

// Export shell for next run
let json = nautilus.to_json();
```

### Crate: primalTools/bingoCube/nautilus/ (26 tests, 3 examples)

| Module | Purpose |
|--------|---------|
| `response.rs` | Board as reservoir projection (BLAKE3 scalar field) |
| `population.rs` | Board ensembles + Pearson fitness evaluation |
| `evolution.rs` | Column-range-preserving crossover + mutation |
| `readout.rs` | Ridge regression readout (Cholesky solver) |
| `shell.rs` | Nautilus shell: layered history + instance transfer |
| `constraints.rs` | Drift monitor, edge seeder, constraint variants |
| `brain.rs` | NautilusBrain: integration API for the NPU worker |

## Files

| File | Role |
|------|------|
| `specs/BIOMEGATE_BRAIN_ARCHITECTURE.md` | This document |
| `experiments/028_BRAIN_CONCURRENT_PIPELINE.md` | Validation experiments |
| `barracuda/src/md/reservoir.rs` | Head 15, NUM_HEADS → 15 |
| `barracuda/src/lattice/gpu_hmc/resident_cg.rs` | `gpu_cg_solve_brain` function |
| `barracuda/src/bin/production_dynamical_mixed.rs` | Brain architecture main binary |
| `barracuda/src/bin/gpu_physics_proxy.rs` | Refactor to library functions |
| `primalTools/bingoCube/nautilus/` | Nautilus Shell evolutionary reservoir crate |
| `whitePaper/gen3/baseCamp/11_bingocube_nautilus_shell.md` | baseCamp paper 11 |
