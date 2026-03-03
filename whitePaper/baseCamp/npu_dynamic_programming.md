# NPU as Dynamic Programming: Activation Parity & Subproblem Memoization for Hot QCD

**Date:** March 3, 2026
**Binary:** `barracuda/src/bin/esn_baseline_validation.rs` (v2, with activation comparison)
**Depends on:** `esn_baseline_validation.md`, `neuromorphic_silicon.md`, `metalForge/npu/akida/HARDWARE.md`

---

## 1. Activation Comparison: One System Works

We added a configurable `Activation` enum to the core ESN (`Tanh` vs `ReluTanhApprox`).
The ReLU variant is a 5-segment piecewise-linear approximation of tanh built entirely
from ReLU operations — directly deployable to the AKD1000 as a 2-layer FC chain
(Discovery 2: merged FC chains cost ~3 µs extra latency).

### Head-to-Head Results

| Dataset | Head | tanh R² | ReLU R² | Δ R² | Verdict |
|---|---|---|---|---|---|
| **Sine (smooth)** | REJECT_PREDICT | 0.984 | 0.985 | +0.001 | Parity |
| | PLAQUETTE | 0.989 | 0.999 | +0.010 | ReLU wins |
| | ACCEPTANCE | 1.000 | 0.999 | -0.001 | Parity |
| **Step (sharp)** | All heads | 0.000 | 0.000 | 0.000 | Both fail — target issue |
| **Power-Law** | REJECT_PREDICT | 0.998 | 1.000 | +0.002 | Parity |
| | LOG_CG | -3.437 | -0.458 | +2.98 | ReLU less bad |
| **Volume Scaling** | LOG_CG | 1.000 | 1.000 | 0.000 | Parity |
| | PLAQUETTE | 1.000 | 0.986 | -0.014 | Slight tanh edge |
| **Noisy** | LOG_CG | 0.986 | 0.968 | -0.018 | Slight tanh edge (smoothing helps) |
| **Real Physics** | REJECT_PREDICT | 0.987 | 0.958 | -0.029 | Both EXCELLENT |
| | PARAM_SUGGEST | 0.548 | 0.857 | **+0.310** | ReLU wins significantly |
| | PLAQUETTE | 1.000 | 0.999 | -0.001 | Parity |

### Conclusions

1. **Every head that works in tanh also works in ReLU-approx-tanh.** Max degradation
   on a working head is 0.029 (REJECT_PREDICT on real data) — both remain EXCELLENT.
2. **PARAM_SUGGEST improves +0.31 with ReLU.** The piecewise-linear activation preserves
   the discrete dt-suggestion boundaries better than smooth tanh.
3. **Tanh has a ~2% edge on noisy data** — smoothness acts as implicit regularization.
   This is the only case where tanh measurably helps.
4. **One system validated.** Train with `Activation::ReluTanhApprox`, deploy the same
   weights to the AKD1000 via FC chain. If Akida 2.0 ships native tanh, flip the flag
   and retrain.

---

## 2. What the NPU Actually Learns

From the full validation harness across 116 points, 4 volumes, 5 masses:

### The NPU knows (R² > 0.95):
- **Acceptance rate** at any (β, mass, volume) it has seen or interpolates between
- **Plaquette value** — the equilibrium gauge field energy at a given coupling
- **Rejection probability** — whether an HMC trajectory will be rejected

### The NPU partially knows (R² 0.3–0.85):
- **Phase region** — confined vs deconfined, but only when test data overlaps training
- **Optimal dt** — the right HMC step size, especially with ReLU activation

### The NPU cannot learn (R² < 0):
- **CG cost extrapolation** across orders of magnitude
- **Susceptibility ranking** at unseen volumes
- **Physics at volumes/masses it hasn't seen**

### The asymmetry is the key insight

| Transfer Direction | REJECT R² | Works? |
|---|---|---|
| Heavy mass → Light mass | **0.973** | Yes |
| Light mass → Heavy mass | -0.614 | No |
| Small volume → Large volume | Degrades | Partially |
| Large volume → Small volume | Works | Yes |

The NPU predicts **downward** (from hard to easy, from complex to simple) but not
**upward** (from simple to complex). It can memoize a solution and apply it to
problems that are the same or easier, but it cannot generalize to harder problems
it hasn't seen.

---

## 3. This Is Dynamic Programming

### The classical DP structure

Dynamic programming solves large problems by:
1. Decomposing into overlapping subproblems
2. Solving each subproblem once
3. Caching the result (memoization)
4. Building up the solution from cached sub-solutions

### The NPU-HMC mapping

| DP Concept | NPU-HMC Equivalent |
|---|---|
| **Subproblem** | One (β, mass, volume) point fully thermalized |
| **Memoization table** | Trained ESN weights — encode the solved parameter space |
| **Table lookup** | NPU inference (~µs) — predicts observables at known points |
| **Optimal substructure** | Physics at volume L contains information about physics at L/2 |
| **Overlapping subproblems** | Many (β, mass) points share similar dynamics across volumes |
| **New subproblem** | The point at the frontier that hasn't been seen — requires full HMC |

### Why this works for lattice QCD

Lattice QCD has a natural hierarchical structure:

```
Volume:    2^4  →  4^4  →  8^4  →  16^4  →  32^4
              ↓       ↓       ↓        ↓
           cheap   medium  expensive  production
```

At each volume, the physics is a **refinement** of the smaller-volume result:
- Plaquette converges as V→∞ (NPU: R²=0.999 at all volumes)
- Acceptance rate decreases with volume but follows a learnable curve
- Phase transition sharpens with volume — crossover becomes first-order
- CG cost scales with volume but the scaling exponent is learnable

The NPU doesn't need to solve the 32^4 problem from scratch. It needs to predict:
- "Based on what I know from 2^4, 4^4, 8^4, 16^4, what should I expect at 32^4?"
- The HMC only computes the **residual** — the difference between prediction and reality.

---

## 4. Architecture: NPU-Assisted Multigrid HMC

### Phase 1: Build the Memoization Table (Bottom-Up)

```
Rung 0: Solve 2^4 at full (β, mass) grid          → Train NPU   → NPU knows 2^4
Rung 1: NPU predicts interesting 4^4 points        → Solve those → NPU knows 2^4 + 4^4
Rung 2: NPU predicts interesting 8^4 points        → Solve those → NPU knows 2^4..8^4
Rung 3: NPU predicts interesting 16^4 points       → Solve those → NPU knows 2^4..16^4
```

At each rung, the NPU:
1. **Predicts** acceptance, CG cost, phase for the next volume from accumulated knowledge
2. **Prioritizes** which (β, mass) points to solve first (highest uncertainty = highest information gain)
3. **Sets HMC parameters** (dt, n_md, thermalization) based on prediction
4. **Detects anomalies** — points where reality diverges from prediction are the frontier

### Phase 2: Exploit Memoization (Steering)

Once the table has sufficient coverage, the NPU enables:

**Subproblem Skip:**
For a new (β, mass) on a large lattice, if the NPU recognizes it as "already solved"
(high confidence on all heads), skip thermalization probes entirely. Use NPU-predicted
dt and n_md. Start measurement immediately. Saves 50-80% of wasted trajectories.

**Frontier Focus:**
The NPU identifies which (β, mass) points are genuinely new — where its predictions
have low confidence or high disagreement between head groups. Route 100% of compute to
these frontier points. Everything else is cache hits.

**Subproblem Decomposition for Hard Points:**
When the NPU encounters a point where CG cost explodes (near critical slowing down),
it recognizes the pattern from smaller volumes:
- "At 4^4, (β=5.69, m=0.01) had CG=21,000 and 38% acceptance"
- "At 8^4, the same point will be ~4× harder"
- "Use Hasenbusch preconditioning, dt=0.002, n_md=100"
- The NPU doesn't solve the hard problem — it **prevents surprise** and sets up optimal parameters.

### Phase 3: NP Decomposition (The Deep Insight)

For genuinely NP-hard aspects of lattice QCD (sign problem, critical slowing down,
topological freezing), the NPU enables a decomposition:

```
Problem: Solve 16^4 at (β_c, m_light) — the hardest point
                     |
         NPU predicts: "this is CSD territory"
         NPU predicts: acceptance ~20%, CG ~100,000
         NPU suggests: Hasenbusch with mass_heavy=0.5
                     |
         ┌──────────┴──────────┐
    KNOWN subproblem:        NEW subproblem:
    Heavy-mass contrib.      Light-mass residual
    (NPU has seen this)      (must compute)
    dt, n_md from cache      Only target for HMC
         └──────────┬──────────┘
                    |
         Result: only the light-mass residual
         is "new compute". Everything else is
         memoized from the DP table.
```

The Hasenbusch factorization itself is a form of subproblem decomposition:
```
det(D†D) = det(D†D / D_heavy†D_heavy) × det(D_heavy†D_heavy)
            └── light residual ──────┘   └── heavy (NPU-memoized) ──┘
```

The NPU tells us **how to factor the problem** based on what it has already learned
about the parameter space. The heavy-mass factor is a "solved subproblem" in the DP
sense — we know its behavior from direct computation at smaller volumes.

---

## 5. Quantitative Estimates

### Compute Savings from Memoization

At each volume, the NPU eliminates redundant exploration:

| Volume | Full Grid (β × mass) | NPU-Selected Points | Savings |
|---|---|---|---|
| 2^4 | 50 × 5 = 250 points | 250 (solve all — cheap) | 0% |
| 4^4 | 50 × 5 = 250 points | ~100 (NPU-guided) | 60% |
| 8^4 | 50 × 5 = 250 points | ~40 (frontier only) | 84% |
| 16^4 | 50 × 5 = 250 points | ~15 (NPU-high-uncertainty) | 94% |

Each volume is ~16× more expensive per trajectory than the previous. The NPU
savings compound: 84% savings at 16× cost means the 8^4 rung costs roughly
the same as the 4^4 rung would have cost without memoization.

### Memoization Table Size

The ESN with 50 neurons, 6 inputs, and 8 heads stores:
- W_in: 50 × 6 = 300 parameters
- W_res: 50 × 50 = 2,500 parameters (sparse, ~500 non-zero)
- W_out: 8 × 50 = 400 parameters
- Total: ~1,200 meaningful parameters encoding 116+ data points

On the AKD1000, this fits in a single neuromorphic core. Inference: ~10 µs.
An HMC trajectory at 8^4 costs ~30 seconds. The NPU lookup is 3,000,000× faster
than re-computing.

---

## 6. What This Means for the Brain

The 4-layer brain architecture (Cortex, Cerebellum, Motor, Pre-motor) maps naturally:

| Brain Layer | DP Role |
|---|---|
| **Cerebellum (AKD1000)** | Memoization table. Stores the learned parameter space. Lookup is ~10 µs. |
| **Cortex (CPU)** | DP controller. Decides which subproblems to solve, manages the build-up. Implements `HeadConfidence` gating. |
| **Motor (RTX 3090)** | HMC executor. Solves the "new subproblem" — the frontier point that the NPU can't predict. |
| **Pre-motor (Titan V)** | Warm start provider. Pre-computes gauge configs for predicted-similar points. A second level of memoization: approximate configs from similar past solves. |

### Concrete Next Steps

1. **Implement volume-aware DP steering** in `npu_worker.rs`: when the NPU's
   HeadConfidence for a new volume drops below threshold, trigger a "fill grid"
   at the smaller volume first to seed the memoization table.

2. **Add a "novelty score" head** (head D0): measures how different the current
   (β, mass, V) is from the nearest training point. High novelty = frontier.
   Route compute there.

3. **Hasenbusch mass suggestion from NPU**: when CG prediction is high-confidence
   and exceeds a threshold, the NPU suggests the Hasenbusch heavy mass based on
   the mass transfer asymmetry (heavy→light works, light→heavy doesn't — so
   always factor toward the heavy side the NPU already knows).

4. **Cross-volume weight transfer**: when moving to a new volume, initialize
   readout weights from the previous volume's trained weights (warm start the
   memoization table). The reservoir captures similar dynamics; only the readout
   needs adjustment.

---

## 7. Activation Path to Hardware

The ReLU-approx-tanh validation confirms the deploy path:

```
CPU training (ReluTanhApprox)
     │
     ├── Identical dynamics to native tanh (Δ R² < 0.03 on all working heads)
     │
     ├── Weights export to ExportedWeights { activation: ReluTanhApprox }
     │
     └── AKD1000 deployment:
              ├── W_in, W_res: standard FC layers (int4 quantized)
              ├── Activation: 2-layer FC chain (10 ReLU neurons)
              │   approximating tanh, merged into reservoir update
              │   pass at ~3 µs extra latency
              └── W_out: readout layer (swappable via set_variable, ~14 ms)
```

No dual system. No conversion. Same weights, same reservoir, same readout.
The hardware maps the math.

---

## 8. Forward Reference: Neuromorphic-Native Field Theory

The DP memoization architecture positions the NPU as a steering layer for GPU-driven
HMC. The logical extrapolation asks: if the NPU understands the lattice well enough
to predict it, can it eventually *be* the lattice?

See [`neuromorphic_native_field_theory.md`](neuromorphic_native_field_theory.md) for
the 5-level incremental path from NPU-as-steering to NPU-as-simulation, the structural
isomorphism between lattice sites and neuromorphic neurons, and the hardware roadmap
through quenched SU(3) to dynamical fermions on spiking hardware.
