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

## NPU Head Expansion (14 → 15)

| Head | Name | Phase | New? |
|------|------|-------|------|
| 0-13 | (existing) | various | No |
| 14 | CG Residual Monitor | During CG | **Yes** |

Head 14 input: sliding window of last 10 residual values (normalized).
Head 14 output: two values — convergence ETA (iterations), anomaly score (0-1).

Note: This renumbers the existing head indices. The current
`ANDERSON_CG` at index 13 stays at 13. The new head is appended at 14.
`NUM_HEADS` increases from 14 to 15.

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

## Files

| File | Role |
|------|------|
| `specs/BIOMEGATE_BRAIN_ARCHITECTURE.md` | This document |
| `experiments/028_BRAIN_CONCURRENT_PIPELINE.md` | Validation experiments |
| `barracuda/src/md/reservoir.rs` | Head 15, NUM_HEADS → 15 |
| `barracuda/src/lattice/gpu_hmc/resident_cg.rs` | `gpu_cg_solve_brain` function |
| `barracuda/src/bin/production_dynamical_mixed.rs` | Brain architecture main binary |
| `barracuda/src/bin/gpu_physics_proxy.rs` | Refactor to library functions |
