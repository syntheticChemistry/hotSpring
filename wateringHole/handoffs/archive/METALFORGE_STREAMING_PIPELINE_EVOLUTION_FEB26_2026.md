# metalForge Evolution: Streaming Heterogeneous Pipeline

**Date:** February 26, 2026
**From:** hotSpring v0.6.14 (Exp 018 benchmark campaign)
**To:** ToadStool core team / metalForge evolution
**Priority:** P0 — next architectural evolution
**License:** AGPL-3.0-only

---

## Executive Summary

hotSpring's Exp 018 benchmark campaign demonstrated that we can run physics on
three substrates simultaneously (RTX 3090 DF64, Titan V native f64, NPU ESN).
But the current `production_mixed_pipeline` is **phase-sequential** — each
substrate waits for the previous phase to complete. The Titan V sat idle for
4+ hours while the 3090 ran seed scans.

This handoff proposes evolving metalForge's forge crate from point-to-point
workload routing (`route()`) to a **streaming heterogeneous pipeline** where
all substrates execute concurrently, work flows between them through typed
channels, and the pipeline topology can be reconfigured at runtime.

---

## Part 1: What We Have (forge v0.2.0)

```
Substrate::discover() → Vec<Substrate>
    ↓
route(workload, &substrates) → Decision
```

Forge discovers hardware and routes individual workloads. One workload, one
substrate, one answer. This is correct for single-shot dispatch but cannot
express:
- Two GPUs producing trajectories in parallel at different β values
- NPU scoring trajectory N-1 while 3090 computes trajectory N
- Titan V validating trajectory N-2 from the 3090's DF64 output
- The pipeline reordering itself to find optimal silicon efficiency

## Part 2: What We Need — Streaming Pipeline

### Core Abstraction: `Stage`

A stage wraps a substrate + computation + typed I/O:

```rust
pub struct Stage<In, Out> {
    substrate: SubstrateRef,
    transform: Box<dyn Fn(In) -> Out + Send>,
    config: StageConfig,
}

pub struct StageConfig {
    /// How many items can queue before backpressure
    buffer_depth: usize,
    /// Whether this stage can be skipped when downstream is saturated
    droppable: bool,
    /// Power budget in milliwatts (NPU: 30, 3090: 350_000, Titan V: 250_000)
    power_budget_mw: Option<u32>,
}
```

### Core Abstraction: `Pipeline`

A pipeline is a directed graph of stages connected by channels:

```rust
pub struct Pipeline {
    stages: Vec<AnyStage>,
    edges: Vec<Edge>,
    topology: Topology,
}

pub enum Topology {
    /// A → B → C (current mixed pipeline, sequential)
    Chain(Vec<StageId>),
    /// A and B run in parallel, both feed C
    FanIn { producers: Vec<StageId>, consumer: StageId },
    /// A feeds B and C in parallel
    FanOut { producer: StageId, consumers: Vec<StageId> },
    /// Arbitrary directed graph
    Graph(Vec<Edge>),
}

pub struct Edge {
    from: StageId,
    to: StageId,
    channel: ChannelKind,
}

pub enum ChannelKind {
    /// Bounded mpsc — backpressure when full
    Bounded { capacity: usize },
    /// Unbounded — producer never blocks (careful with memory)
    Unbounded,
    /// Broadcast — every consumer gets every item
    Broadcast,
    /// Filtered — consumer only gets items matching a predicate
    Filtered { predicate: String },
}
```

### The QCD Pipeline — Concrete Example

```rust
let pipeline = PipelineBuilder::new()
    // Both GPUs produce HMC trajectories in parallel
    .stage("3090_hmc", gpu_3090, HmcProducer::new(32, Fp64Strategy::Hybrid))
    .stage("titan_hmc", titan_v, HmcProducer::new(16, Fp64Strategy::Native))

    // NPU scores every trajectory for phase transition signal
    .stage("npu_screen", npu, EsnClassifier::new(reservoir_size: 50))

    // Cross-validation: each GPU spot-checks the other's work
    .stage("3090_validate", gpu_3090, PrecisionChecker::df64_vs_f64())
    .stage("titan_validate", titan_v, PrecisionChecker::native_f64())

    // Steering: decides what β to run next based on all results
    .stage("steer", cpu, BayesianSteering::new(beta_range: 4.0..7.0))

    // Edges: the data flow
    .edge("3090_hmc", "npu_screen", ChannelKind::Bounded { capacity: 4 })
    .edge("titan_hmc", "npu_screen", ChannelKind::Bounded { capacity: 4 })
    .edge("npu_screen", "steer", ChannelKind::Bounded { capacity: 8 })
    .edge("steer", "3090_hmc", ChannelKind::Bounded { capacity: 1 })
    .edge("steer", "titan_hmc", ChannelKind::Bounded { capacity: 1 })

    // Cross-validation edges (filtered: only critical-region configs)
    .edge("3090_hmc", "titan_validate",
          ChannelKind::Filtered { predicate: "chi > 10.0" })
    .edge("titan_hmc", "3090_validate",
          ChannelKind::Filtered { predicate: "chi > 10.0" })

    .build();
```

This runs **all substrates concurrently**:
- 3090 produces 32⁴ DF64 trajectories at 7.64s/traj
- Titan V produces 16⁴ native f64 trajectories at 1.27s/traj
- NPU scores every trajectory as it arrives (<1ms)
- CPU steers both GPUs to the most informative β values
- When χ > 10 (near the phase transition), configs cross-validate between GPUs

### Contiguous Work Units

The physics requires **Markov chain continuity** — each trajectory depends on
the previous one. You can't split a single Markov chain across GPUs. But you
CAN run **independent chains** on different substrates:

```
3090:     chain_A[β=5.69, 32⁴] → traj_1 → traj_2 → traj_3 → ...
Titan V:  chain_B[β=5.72, 16⁴] → traj_1 → traj_2 → traj_3 → ...
```

Each chain is contiguous on its substrate. The steering layer decides which
β value each chain explores. Chains can be **migrated** between substrates
at checkpoint boundaries (serialize the gauge config, transfer, deserialize).

### Runtime Reconfiguration

The topology can evolve mid-run based on observed performance:

```rust
impl Pipeline {
    /// Reconfigure the topology while running.
    /// Stages are paused, edges rewired, stages resumed.
    pub async fn reconfigure(&mut self, new_topology: Topology) { ... }

    /// Metrics collected per stage for steering decisions.
    pub fn stage_metrics(&self, id: StageId) -> StageMetrics { ... }
}

pub struct StageMetrics {
    pub throughput_per_sec: f64,
    pub latency_p50_ms: f64,
    pub latency_p99_ms: f64,
    pub utilization_pct: f64,
    pub power_watts: f64,
    pub queue_depth: usize,
    pub items_processed: u64,
}
```

The steering stage can observe these metrics and reconfigure:

```rust
// If Titan V queue is backing up, reduce its workload
if metrics["titan_validate"].queue_depth > 6 {
    pipeline.reconfigure(Topology::Graph(vec![
        // Remove titan cross-validation, let it focus on production
        Edge::new("3090_hmc", "npu_screen", bounded(4)),
        Edge::new("titan_hmc", "npu_screen", bounded(4)),
        Edge::new("npu_screen", "steer", bounded(8)),
    ])).await;
}

// If 3090 acceptance rate drops below 15%, switch its chain to a
// different β where it's more efficient
if results["3090_hmc"].acceptance < 0.15 {
    pipeline.send_config("3090_hmc", HmcConfig { beta: next_beta }).await;
}
```

---

## Part 3: Pipeline Orderings — Finding Silicon Efficiency

The same physics can be computed with different substrate orderings.
Each ordering has different throughput, latency, power, and cost
characteristics. The pipeline should be able to try all orderings
and measure which extracts the most physics per joule.

### Order A: Production (3090-led, Titan V validates)

```
3090 DF64 [32⁴] → NPU screen → Titan V f64 spot-check
```
- **Throughput**: limited by 3090 at 7.64s/traj
- **Power**: ~350W (3090) + 0.03W (NPU) + ~200W (Titan V intermittent)
- **Use case**: production β-scan, maximum lattice size

### Order B: Discovery (Titan V-led, 3090 follows up)

```
Titan V f64 [16⁴] → NPU screen → 3090 DF64 [32⁴] focused
```
- **Throughput**: Titan V at 1.27s/traj (fast iteration on small lattice)
- NPU identifies interesting β values quickly
- 3090 only runs the β values that matter (saves hours)
- **Use case**: exploring unknown phase diagrams

### Order C: Parallel Discovery (both GPUs, NPU-steered)

```
NPU Bayesian model → dispatch to 3090 AND Titan V → collect → update model
```
- Both GPUs produce independent Markov chains at different β
- NPU maintains a Bayesian posterior over the phase diagram
- Dispatches β values to maximize information gain across all substrates
- **Use case**: maximum throughput when exploring wide β range

### Order D: Precision Cascade (DF64 → f64 continuous)

```
3090 DF64 → every config → Titan V f64 re-evaluate
```
- No NPU, no screening — every config gets precision-checked
- Measures DF64 vs native f64 precision drift continuously
- **Use case**: validating DF64 precision for publication

### Order E: Power-Optimized (NPU-heavy, GPU on-demand)

```
NPU maintains ESN model (30mW) → GPUs wake only when uncertainty is high
```
- GPUs idle most of the time, NPU runs continuously
- When the ESN uncertainty crosses a threshold, the GPU fires one trajectory
- **Use case**: long-running monitoring, minimal power consumption

### Measuring Silicon Efficiency

For each ordering, measure:

```rust
pub struct SiliconEfficiency {
    /// Physics results per second (trajectories/s, measurements/s)
    pub throughput: f64,
    /// Joules per physics result
    pub energy_per_result: f64,
    /// Total power draw (all substrates, watts)
    pub total_power_w: f64,
    /// Fraction of each substrate's peak capability used
    pub utilization: HashMap<SubstrateId, f64>,
    /// Physics quality (effective sample size, χ² per measurement)
    pub quality: f64,
    /// Cost efficiency: quality / energy
    pub quality_per_joule: f64,
}
```

The pipeline can run each ordering for a calibration period, measure
efficiency, and auto-select the best for the remaining run.

---

## Part 4: DF64 as Extension, Not Replacement

A critical insight from this campaign: DF64 **extends** native fp64, it
doesn't replace it. On every GPU:

1. **Saturate FP64 units** with precision-critical work (accumulations,
   convergence tests, Cayley exponential, RNG)
2. **Overflow** bulk math to FP32 cores via DF64 (SU(3) products, staple
   sums, kinetic energy P²)
3. Both execution unit types fire **simultaneously** within the same SM

The `Fp64Strategy` enum should evolve:

```rust
pub enum Fp64Strategy {
    Native,     // fp64-only (wastes FP32 cores on Titan V)
    Hybrid,     // DF64 bulk + fp64 accumulations (current on 3090)
    Concurrent, // saturate fp64 AND overflow to DF64 (evolution target)
}
```

On the **Titan V** with Concurrent:
- FP64 units: 0.60 TFLOPS (current, improving as NAK evolves)
- FP32 via DF64: 1.41 TFLOPS (measured)
- **Combined: ~2.0 TFLOPS** (3.3× over current Native-only)

On the **RTX 3090** with Concurrent:
- The current Hybrid already approximates this
- Making it explicit enables better scheduling of which operations go where

---

## Part 5: What hotSpring Proved (Exp 018 Evidence)

| Finding | Evidence |
|---------|----------|
| DF64 production-ready | 6.37h vs 13.6h (2.13×), physics matches baseline |
| Titan V 30⁴ recovered | NVK allocation guard enables previously-crashing lattice |
| NAK improving | fp64 throughput 0.25→0.60 TFLOPS (2.4×) without driver changes |
| Cross-GPU physics consistent | 32⁴ DF64 and 30⁴ native f64 χ profiles match |
| Timing rock-solid | DF64 per-point variance ±0.7% vs ±15% native |
| All 3 substrates concurrent | Mixed pipeline discovers 3090 + Titan V + NPU |

---

## Part 6: Implementation Roadmap

### Phase 1: Streaming Channels (forge v0.3.0)

Add `tokio::mpsc`-based inter-stage channels to forge:

- `Stage<In, Out>` trait with async `process(In) -> Out`
- `Pipeline::run()` spawns all stages as tokio tasks
- Bounded channels with configurable backpressure
- `StageMetrics` collection via atomic counters

**Deliverable**: Two GPUs running independent HMC chains simultaneously,
both feeding results to a CPU steering stage.

### Phase 2: Topology Builder (forge v0.4.0)

- `PipelineBuilder` with type-safe stage/edge wiring
- `Topology` enum (Chain, FanIn, FanOut, Graph)
- `ChannelKind::Filtered` for selective forwarding
- Serialization of pipeline topology for reproducibility

**Deliverable**: The QCD pipeline example from Part 2, running live.

### Phase 3: Runtime Reconfiguration (forge v0.5.0)

- `Pipeline::reconfigure()` with stage pause/resume
- `SiliconEfficiency` metrics collection
- Auto-calibration: run each ordering for N trajectories, measure, select
- Mid-run topology changes based on observed performance

**Deliverable**: Pipeline that discovers optimal ordering automatically.

### Phase 4: Concurrent Fp64Strategy (barracuda evolution)

- `Fp64Strategy::Concurrent` in driver_profile.rs
- Shader-level interleaving of fp64 and DF64 operations
- Benchmark on Titan V: measure combined throughput vs Native-only

**Deliverable**: Titan V at ~2.0 TFLOPS effective (vs 0.60 current).

---

## Part 7: What Exists to Build On

| Component | Location | Status |
|-----------|----------|--------|
| `Substrate` discovery | `forge/src/probe.rs` | ✅ GPU + NPU + CPU |
| Capability model | `forge/src/substrate.rs` | ✅ 12 capabilities |
| `route()` dispatch | `forge/src/dispatch.rs` | ✅ point-to-point |
| Workload profiles | `forge/src/dispatch.rs` | ✅ 7 profiles |
| Bridge to barracuda | `forge/src/bridge.rs` | ✅ device ↔ substrate |
| `GpuHmcStreamingPipelines` | `barracuda/lattice/gpu_hmc/` | ✅ single-GPU |
| `production_mixed_pipeline` | `barracuda/src/bin/` | ✅ phase-sequential |
| `Fp64Strategy` | `barracuda/device/driver_profile.rs` | ✅ Native/Hybrid |
| ESN phase classifier | `barracuda/src/esn/` | ✅ trained in <1ms |
| NPU simulator | `barracuda/src/md/npu.rs` | ✅ ESN inference |

The streaming pipeline is the **next layer up** — it composes these
existing components into concurrent execution.

---

## Acceptance Criteria

### forge v0.3.0
- [ ] Two GPU stages run concurrently producing trajectories
- [ ] Results flow through bounded channels to a steering stage
- [ ] `StageMetrics` reports throughput and queue depth per stage
- [ ] No physics regression — results match sequential pipeline

### forge v0.4.0
- [ ] `PipelineBuilder` compiles typed topology
- [ ] At least 3 orderings from Part 3 work (A, B, C)
- [ ] Filtered channels deliver only critical-region configs to validators

### forge v0.5.0
- [ ] Pipeline reconfigures mid-run without losing state
- [ ] `SiliconEfficiency` measured for at least 2 orderings
- [ ] Auto-calibration selects best ordering after warm-up period

---

## Addendum: Empirical NPU Placement Data (Exp 020)

The NPU Characterization Campaign (Exp 020, Feb 26 2026) tested 6 pipeline placements
with real ESN models on 1800 trajectories. Key findings that inform the streaming pipeline:

### Placement Results

| Position | Accuracy | Traj Saved | Implication for Streaming |
|---|---|---|---|
| A: Pre-thermalization | 83.3% | 390 (21.7%) | **Highest ROI** — NPU monitors plaquette convergence |
| B: Mid-trajectory exit | 95.8% | 0 | Best accuracy — useful as post-facto classifier |
| C: Post-trajectory (baseline) | 83.3% | 0 | Current approach, NPU after GPU completion |
| D: Inter-beta steering | 45.5% | 0 | Needs more training data; promising direction |
| E: Pre-run bootstrap | 50.0% | 0 | Use historical data to warm-start |
| F: All combined | 87.5% | 390 | Maximum savings with combined approach |

### Recommended Streaming Topology (from Exp 020)

Position A should be the **default NPU placement** in the streaming pipeline:

```
[GPU: Thermalization] → [NPU: Convergence Monitor (A)] → early stop signal
                                                            ↓
[GPU: Measurement] ←───────────────────────────────────────┘
        ↓
[NPU: Post-trajectory Classify (C)]
        ↓
[CPU: Inter-β Steering (D)]
```

### NPU Latency Budget

- Single inference: 331 µs (p50), 520 µs (p99) — validated on NpuSimulator
- Batch=8: 3.5 inf/ms (+18% over single) — hardware: 2.4× expected
- Weight mutation: 0.015ms (sim) / 14ms (hw) — acceptable once per β-point
- PCIe roundtrip: ~1.3ms — negligible vs 7.64s/trajectory

### Thermalization Detector Performance

The ESN thermalization detector achieved 87.5% accuracy with 61.8% savings of
thermalization trajectories. At production scale, this projects to **3.15 hours saved**
from the 5.1h thermalization budget — the single largest optimization available.

See `experiments/020_NPU_CHARACTERIZATION_CAMPAIGN.md` and
`wateringHole/handoffs/AKIDA_BEHAVIOR_REPORT_FEB26_2026.md` for full data.
