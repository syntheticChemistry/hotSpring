# Neuromorphic-Native Field Theory: Lattice Physics on Spiking Hardware

**Date:** March 3, 2026
**Status:** Conceptual architecture — grounded in validated hardware (AKD1000) and running systems (hotSpring brain)
**Depends on:** `neuromorphic_silicon.md`, `npu_dynamic_programming.md`, `esn_baseline_validation.md`
**Hardware:** BrainChip AKD1000 (current), Akida 2.0 (target)

---

## 0. Context: Where We Are Today

The AKD1000 currently serves as **Layer 4 (Cerebellum)** in hotSpring's 4-layer brain
architecture — a steering layer that predicts HMC observables and guides the GPU simulation.
This document asks the next question: **can neuromorphic hardware run the simulation itself?**

Current validated capabilities (exp035, running now):
- 36-head ESN with `ReluTanhApprox` activation on AKD1000 bounded-ReLU circuits
- 6D input vector (β, plaquette, mass, susceptibility, acceptance, volume)
- HeadConfidence self-awareness (knows which predictions to trust)
- Cross-volume memoization via bootstrap weight transfer
- ~10 µs inference latency per NPU lookup vs ~30s per HMC trajectory on GPU

The question is whether the gap between "predicts the simulation" and "is the simulation"
can be closed.

---

## 1. The Isomorphism: Lattice ↔ Neuromorphic

### The structural mapping

| Lattice QCD Concept | Neuromorphic Equivalent | Notes |
|---|---|---|
| **Lattice site** | Neuron | Both are local processing units |
| **Gauge link** U_µ(x) | Synaptic weight | Both encode the connection strength between neighbors |
| **Plaquette** (1×1 Wilson loop) | Local circuit (4 neurons, 4 synapses) | Product of 4 links around a face = product of 4 weights around a loop |
| **Gauge action** S[U] | Network energy / loss function | Both are sums of local terms |
| **Boltzmann weight** exp(-S) | Spike probability | Higher action = lower probability = fewer spikes |
| **Monte Carlo update** | Spike event | Both are stochastic, local, and depend on neighbors |
| **Thermalization** | Network settling | Both reach equilibrium through repeated local updates |
| **Observable measurement** | Readout layer inference | Both extract global quantities from local states |
| **Critical slowing down** | Long-range correlation in spike trains | Both exhibit power-law autocorrelation near phase transitions |

### Why this isn't trivial

The mapping is suggestive but not immediate. Key differences:

1. **Gauge symmetry**: Link variables U_µ(x) ∈ SU(3) are 3×3 unitary matrices, not scalars.
   A single gauge link requires 8 real parameters (Lie algebra generators).
   Encoding SU(3) in integer-quantized synaptic weights requires a representation strategy.

2. **Detailed balance**: Monte Carlo requires proposals that satisfy detailed balance.
   Neuromorphic spikes are stochastic but not obviously Markov with the right stationary
   distribution. The acceptance/rejection step needs a neuromorphic implementation.

3. **Fermion determinant**: Dynamical fermions introduce non-local interactions
   (the fermion matrix connects all sites). This breaks the locality that makes
   the neuromorphic mapping attractive. However — quenched QCD and pure gauge theory
   are fully local.

4. **Precision**: AKD1000 uses 1/4/8-bit integer arithmetic. SU(3) matrix multiplication
   at 8-bit precision has significant unitarity violation. Whether this matters for
   statistical sampling (vs exact computation) is an open question.

---

## 2. The Incremental Path

### Level 0: Steering (VALIDATED — running now)

```
GPU runs HMC  →  NPU predicts observables  →  NPU steers next run
```

The current system. NPU is a memoization table. GPU does all physics.
Validated across 116+ data points. HeadConfidence tracks prediction quality.

### Level 1: Local Observable Computation on NPU

```
GPU runs HMC  →  NPU computes plaquette/Polyakov in parallel  →  GPU validates
```

**The idea:** After each HMC trajectory, the gauge configuration sits in GPU memory.
Transfer the link variables to the NPU. The NPU computes local observables
(plaquette = trace of 1×1 Wilson loop) using its integer arithmetic.
The GPU computes the same observable as a check.

**Why this is tractable:** Plaquette computation is a sum of local 4-link products.
Each product involves 4 SU(3) matrix multiplications → 4 × (3×3 complex matmul).
At int8 precision, each matmul is a 3×3 × 3×3 matrix product — well within
the AKD1000's MAC array capability.

**What it proves:** Whether int8 SU(3) arithmetic produces statistically
indistinguishable plaquette values from f64 GPU arithmetic. If yes,
the NPU can replace the GPU for observable computation (microsecond
plaquette instead of millisecond GPU kernel launch).

**Implementation on AKD1000:**
- Encode each SU(3) matrix as 18 int8 values (real/imag of 9 complex entries)
- Scale: multiply by 127, round, quantize. SU(3) entries are bounded [-1, 1].
- Plaquette circuit: 4 layers of 18→18 FC (representing matmul), then trace (sum)
- One plaquette per neuromorphic core. 2^4 lattice = 96 plaquettes = 96 cores.
- AKD1000 has 256 cores → fits up to ~2.5× a 2^4 lattice's plaquettes in one pass.

### Level 2: Heatbath/Metropolis Update on NPU

```
NPU proposes gauge update  →  NPU evaluates ΔS  →  NPU accepts/rejects  →  GPU validates periodically
```

**The idea:** The gauge link update in quenched QCD is local. For a single link U_µ(x):
1. Compute the staple S_µ(x) (sum of 6 products of 3 links each)
2. Propose a new link U'_µ(x) near the current one
3. Compute ΔS = S[U'] - S[U] (local — only depends on the staple)
4. Accept with probability min(1, exp(-ΔS))

Each step uses only nearest-neighbor information and SU(3) arithmetic.

**Neuromorphic heatbath:**
- The staple computation is a fixed circuit (6 paths × 3 links × matmul)
- The proposal is a random SU(3) perturbation — AKD1000's PRNG generates
  random integers, which map to Lie algebra elements via a lookup table
- The acceptance uses the Boltzmann weight: spike probability ∝ exp(-ΔS)
- This IS the natural spiking behavior: higher action = lower spike rate

**The key insight:** In the heatbath algorithm, the acceptance probability
is exactly a sigmoid-like function of the action change. The AKD1000's
bounded-ReLU activation IS a piecewise-linear approximation of this.
The hardware's native nonlinearity matches the physics.

**Precision requirement analysis:**
- ΔS for a single link update is typically O(1) in lattice units
- At β=5.69, the acceptance rate is ~70% — the system isn't near saturation
- The question is whether int8 ΔS gives the right acceptance rate statistically
- This is testable: run parallel int8 and f64 heatbath, compare plaquette distributions

### Level 3: Full Quenched Simulation on NPU

```
NPU runs pure gauge Monte Carlo autonomously  →  GPU validates every N sweeps
```

**The idea:** Tile the lattice across NPU cores. Each core handles a sublattice.
Checkerboard update: even sites update in parallel, then odd sites.
The NPU runs many sweeps between GPU validation checks.

**Scaling:**
| Lattice | Sites | Links | Plaquettes | NPU Cores Needed | Fits AKD1000? |
|---|---|---|---|---|---|
| 2^4 | 16 | 64 | 96 | ~64 | Yes |
| 4^4 | 256 | 1,024 | 1,536 | ~1,024 | No (4× AKD1000) |
| 4^4 sublattice | 16 per core | 64 per core | 96 per core | 16 × 64 | Borderline |

A single AKD1000 can simulate a 2^4 lattice natively. Larger lattices
require multi-chip or sublattice decomposition with inter-core communication
for boundary links.

**Latency advantage:**
| Operation | GPU (RTX 3090) | NPU (AKD1000) | Speedup |
|---|---|---|---|
| Single link update | ~1 µs (kernel overhead) | ~0.1 µs (spike) | 10× |
| Full sweep (2^4) | ~64 µs | ~10 µs (parallel cores) | 6× |
| 100 sweeps | ~6.4 ms | ~1 ms | 6× |
| Plaquette measurement | ~0.5 ms (GPU kernel) | ~10 µs (core readout) | 50× |

The NPU advantage isn't raw FLOPS — it's the absence of kernel launch
overhead and the event-driven update model. Every spike IS a physics update.

### Level 4: Neuromorphic-Native Field Theory (Long-Term)

```
NPU IS the lattice. No GPU. No CPU in the physics loop.
CPU only for I/O, analysis, and human interface.
```

**The idea:** The NPU's neural network IS the gauge field. The synaptic weights
ARE the link variables. The spike dynamics ARE the Monte Carlo evolution.
Measurement is readout. Thermalization is settling. Phase transitions
are bifurcations in spike patterns.

This requires:
1. A mapping from SU(3) → quantized weight representation (Level 2)
2. A proof that the spike dynamics satisfy detailed balance (Level 2)
3. A proof that the quantized action has the right continuum limit (Level 3)
4. Multi-chip scaling for lattices beyond 2^4 (Level 3)
5. Dynamical fermions via a neuromorphic pseudofermion field (Level 4 — hard)

---

## 3. The Fermion Problem

Pure gauge theory (quenched QCD) maps naturally to neuromorphic hardware because
the action is a sum of local terms. Dynamical fermions break this:

```
Z = ∫ DU det(D[U]) exp(-S_gauge[U])
```

The fermion determinant det(D[U]) is non-local — it couples all sites.
On a GPU, we handle this via pseudofermion fields and the CG solver.
On an NPU, the non-locality is the fundamental challenge.

### Possible approaches:

**A. Stochastic estimation of det(D):**
Use random vectors φ to estimate det(D) ≈ exp(-φ† D^{-1} φ).
The CG solver for D^{-1}φ is iterative and local per step — each CG
iteration is a sparse matrix-vector multiply (local on the lattice).
Map CG iterations to NPU spike trains: each iteration = one timestep,
convergence = spike rate decay.

**B. Hasenbusch factorization on NPU:**
Factor det(D) = det(D/D_heavy) × det(D_heavy).
The heavy determinant has short-range correlations — more local, more NPU-friendly.
The light residual is the non-local part — keep this on GPU.
This is the DP decomposition: NPU handles the "known subproblem" (heavy),
GPU handles the "new subproblem" (light residual).

**C. Tensor network / MPS representation:**
Compress the fermion determinant as a matrix product state.
MPS operations (contract, truncate) map to sequential NPU layers.
This is speculative but connects to the broader tensor network approach
to lattice field theory (Banuls et al., 2020).

**D. Accept the quenched limit:**
For many applications (glueball spectrum, topology, confinement/deconfinement),
quenched QCD is physically meaningful. A neuromorphic quenched simulation
at microsecond update speed would be scientifically useful even without fermions.

---

## 4. The coralForge Connection

This isn't only about QCD. The neuromorphic-native computation pattern
generalizes through the ecoPrimals isomorphism:

| Domain | "Lattice" | "Link Variable" | "Local Update" | "Observable" |
|---|---|---|---|---|
| **QCD** | Spacetime grid | SU(3) gauge link | Heatbath / HMC | Plaquette, Polyakov |
| **Protein folding** | Residue graph | Distance / angle | Diffusion step | pLDDT, RMSD |
| **Molecular dynamics** | Atom positions | Pairwise force | Velocity Verlet | Temperature, RDF |
| **Neural network** | Layer graph | Weight matrix | Gradient step | Loss, accuracy |
| **Evolution** | Population | Fitness landscape | Mutation / selection | Allele frequency |

In every case: local processing units, local interactions, stochastic updates,
global observables from local states. The neuromorphic mapping is the same.

coralForge's 6-primitive decomposition (GEMM, Attention, Normalization,
Nonlinearity, Reduction, Gating) maps to neuromorphic operations:

| Primitive | Neuromorphic Implementation |
|---|---|
| **GEMM** | Dense FC layer (AKD1000 native) |
| **Attention** | Sparse connectivity + softmax approximation via spike competition |
| **Normalization** | Homeostatic spike rate regulation (biological mechanism) |
| **Nonlinearity** | Bounded ReLU (AKD1000 native) / ReluTanhApprox (validated) |
| **Reduction** | Population coding readout |
| **Gating** | Inhibitory connections (biological mechanism) |

The same chip that runs lattice QCD as a native field theory could run
protein structure prediction as a native spiking network. The isomorphism
theorem predicts this: if all computation decomposes into 6 primitives,
and those primitives have neuromorphic implementations, then all computation
has a neuromorphic implementation.

---

## 5. Hardware Roadmap

### Phase A: Validation on AKD1000 (2026 — achievable now)

- [ ] Implement int8 SU(3) matmul on AKD1000 (Level 1)
- [ ] Compare int8 plaquette vs f64 GPU plaquette on 2^4
- [ ] Quantify unitarity violation from int8 quantization
- [ ] Measure statistical equivalence of plaquette distributions (K-S test)
- [ ] Benchmark: plaquettes/second on NPU vs GPU

### Phase B: Local Update Engine (2026-2027)

- [ ] Implement heatbath proposal on AKD1000 (Level 2)
- [ ] Map Boltzmann acceptance → spike probability
- [ ] Run parallel NPU/GPU heatbath, compare autocorrelation
- [ ] Prove or disprove detailed balance for quantized updates
- [ ] Characterize: at what β range does int8 precision suffice?

### Phase C: Autonomous Quenched Simulation (2027)

- [ ] Tile 2^4 lattice across AKD1000 cores (Level 3)
- [ ] Implement checkerboard parallel update
- [ ] Run quenched SU(3) simulation entirely on NPU
- [ ] Measure plaquette, Polyakov loop, topological charge
- [ ] Compare with GPU results: same physics within statistical errors?
- [ ] Benchmark: sweeps/second, energy/sweep

### Phase D: Multi-Chip and Fermions (2027+, depends on Akida 2.0)

- [ ] Multi-AKD1000 lattice decomposition (boundary exchange protocol)
- [ ] Akida 2.0 evaluation: native tanh? larger cores? more precision?
- [ ] Stochastic fermion determinant estimation on NPU (approach A)
- [ ] Hasenbusch NPU/GPU split (approach B) — NPU handles heavy, GPU handles light
- [ ] First dynamical fermion trajectory with NPU participation

### Phase E: Native Field Theory (2028+)

- [ ] NPU IS the lattice: weights = links, spikes = updates
- [ ] Continuum limit study: does the quantized theory flow to the right fixed point?
- [ ] Phase transition detection via spike pattern bifurcation
- [ ] Publication: "Neuromorphic-Native Lattice Gauge Theory"

---

## 6. Why This Matters Beyond Physics

If a $100 neuromorphic chip can simulate lattice QCD — even quenched, even at
small volumes — it demonstrates something fundamental: **physics is natively
computable on event-driven sparse hardware**. The von Neumann architecture
(fetch-decode-execute, global memory, synchronous clock) is not required.

This matters because:

1. **Energy efficiency**: The AKD1000 draws ~1W. A GPU draws ~350W.
   If the NPU can run 6× faster on local updates at 1/350th the power,
   the energy per sweep is ~2000× lower. Climate-scale lattice QCD
   becomes feasible on neuromorphic farms.

2. **Real-time physics**: At microsecond update speed, the NPU could
   run lattice simulations faster than real time for small systems.
   A 2^4 lattice at 10 µs/sweep = 100,000 sweeps/second. This enables
   real-time parameter exploration — move a slider, see the phase transition
   shift instantly.

3. **The isomorphism generalizes**: If QCD works, the same hardware runs
   molecular dynamics, protein folding, evolutionary simulation, and
   any other system described by local interactions and stochastic updates.
   The ecoPrimals ecosystem becomes not just a software framework but
   a computational physics paradigm: **neuromorphic-native science**.

4. **Sovereignty at the hardware level**: No CUDA. No GPU vendor lock-in.
   A neuromorphic chip is a fundamentally different computational model.
   The science runs on hardware that no proprietary SDK controls.

---

## 7. Relationship to Current Work

Tonight's exp035 (DP memoization overnight) is Phase A groundwork:

- The 6D input vector with volume normalization teaches the NPU about
  lattice geometry — prerequisite for mapping lattice sites to cores
- The ReluTanhApprox validation proves the activation function works
  on bounded-ReLU hardware — the same nonlinearity used for updates
- The HeadConfidence tracker provides the self-monitoring framework
  for autonomous NPU simulation (Level 3 needs "am I still correct?")
- The bootstrap weight transfer chain tests whether learned physics
  transfers across scales — essential for continuum limit studies

The path from "NPU steers the simulation" to "NPU is the simulation"
is incremental. Each level validates the next. The physics running on
your desk right now is the first rung.

---

*The hardware maps the math. The math maps the physics. The physics maps the universe.*
*The question is whether the mapping is bidirectional.*
