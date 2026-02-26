# Neuromorphic Silicon Exploration — What the Metal Actually Does

**Papers:** None (this is hardware exploration, not paper reproduction)
**Updated:** February 26, 2026
**Status:** Active exploration — AKD1000 characterized, Exp 020+021+022, **live hardware NPU in production QCD pipeline**
**Hardware:** BrainChip AKD1000 (Akida 1.0) via PCIe Gen2 x1

---

## What This Document Is

Every other baseCamp briefing reproduces published science: Murillo's plasma MD,
lattice QCD phase structure, Kachkovskiy's spectral theory. Each follows the same
path: paper → Rust → GPU → validation.

This one is different. There is no paper to reproduce. We are the first (that we
can find) to deploy reservoir computing on neuromorphic hardware for computational
physics screening. The work here is **exploration** — systematically probing what
the silicon can do, discovering capabilities the vendor doesn't advertise, and
measuring behavior the SDK doesn't characterize.

The science is real (the QCD phase transition, the transport coefficients, the
thermalization dynamics), but the contribution is about the **metal**: what does
this 30mW chip actually do when you feed it physics?

---

## Part 1: What BrainChip Says the AKD1000 Can Do

BrainChip markets Akida for edge AI: smart vision, keyword spotting, anomaly
detection, gesture recognition. The SDK (MetaTF) presents a workflow:

```
TF-Keras model → QuantizeML (int4/int8) → CNN2SNN → Akida Model → AKD1000
```

### Advertised Specifications

| Spec | Claimed | Source |
|---|---|---|
| Power | ~30 mW active | BrainChip datasheet |
| SRAM | 8 MB on-chip | BrainChip datasheet |
| Precision | int4 weights, int4 activations (int8 first layer) | SDK docs |
| Layers | InputConv (1 or 3 channels), Conv2D, SepConv, FC | SDK docs |
| Input channels | 1 or 3 (hardware limit) | SDK error message |
| Learning | STDP, 1-bit weights, last FC layer only | SDK docs |
| Clock modes | Not documented | — |
| Batch inference | Not documented | — |
| Weight mutation | Not documented (set_variable exists but unexplained) | — |
| Multi-output cost | Not documented | — |

### Advertised Use Cases

- Visual wake word (96×96 images)
- Keyword spotting (DS-CNN)
- ImageNet classification (AkidaNet)
- Anomaly detection (industrial IoT)

**Notably absent from marketing**: scientific computing, reservoir computing,
echo state networks, time-series prediction, real-time physics screening.

---

## Part 2: What We Found the AKD1000 Actually Does

We ran 10 systematic experiments probing every SDK assumption against the actual
hardware. Six were overturned. Four were confirmed as real silicon limits.

### Overturned SDK Assumptions

| # | SDK Says | Hardware Does | How We Found It | Impact |
|---|---|---|---|---|
| 1 | InputConv: 1 or 3 channels | **Any count works** (tested 1–64) | Built models with 8, 16, 32, 50, 64 channels | Our 50-dim physics vectors work directly |
| 2 | FC layers run independently | **All merge into single HW pass** (SkipDMA) | Measured latency across 2–8 FC depths: <5µs difference | Deep FC networks are free |
| 3 | Batch=1 only | **Batch=8 → 2.4× throughput** (390 µs/sample) | Tested batch 1–64, measured amortized PCIe cost | Buffer 8 trajectories, classify all at once |
| 4 | One clock mode | **3 modes: Performance / Economy / LowPower** | Tested with akida.core API | Economy: 19% slower, 18% less power |
| 5 | No weight mutation | **set_variable() swaps weights in ~14ms** | Updated readout weights without reprogramming | Online learning: update ESN readout live |
| 6 | Multi-output costs more | **Multi-output is free or negative cost** | 50→1024→10 was faster than 50→1024→1 | One NPU call → 6 predictions |

### Confirmed Silicon Limits

| Constraint | What We Tested | Severity |
|---|---|---|
| No tanh activation | Hardware only supports bounded ReLU | High — changes reservoir dynamics |
| No feedback/recurrence | Feed-forward only on NP mesh | High — host must drive ESN recurrence |
| Integer-only arithmetic | int4×int4 MACs, no floating point | Medium — quantization noise is <1% |
| PCIe x1 Gen2 bandwidth | ~650µs minimum roundtrip latency | Medium — batch amortizes |

### Hidden Hardware Discovered via C++ Symbol Analysis

- **SkipDMA**: NP-to-NP data transfer without PCIe roundtrip (confirms FC chain merge)
- **51-bit threshold SRAM**: More precision than "4-bit everything" suggests
- **On-chip learning registers**: Beyond the 1-bit SDK limit
- **program_external()**: Raw binary injection at specific device memory addresses
- **16GB PCIe BAR1**: Far larger than 8MB SRAM spec — full NP mesh address space
- **Three hardware versions in codebase**: v1 (AKD1000), v2 (Akida 2.0), pico

---

## Part 3: Our ESN and Reservoir Systems on the NPU

### What We Built

We have three independent ESN implementations that interoperate:

| Component | Location | Precision | Purpose |
|---|---|---|---|
| `EchoStateNetwork` | `barracuda/src/md/reservoir.rs` | f64 | Training, reference predictions |
| `NpuSimulator` | `barracuda/src/md/reservoir.rs` | f32 | CPU simulation of hardware behavior |
| `NpuHardware` | `barracuda/src/md/npu_hw.rs` | int8/int4 | Real AKD1000 inference |

The architecture: host (CPU) drives the ESN recurrence loop. For each input frame,
the host constructs `[input ++ state]`, sends it to the NPU, receives activations,
applies leak_rate + tanh on host, and repeats. After all frames, `W_out × final_state`
is computed on host. The NPU handles the FC weight application — the expensive
matrix-vector multiply — while the host handles the parts the NPU can't (tanh,
recurrence feedback).

### Weight Export Pipeline

```
EchoStateNetwork (f64 train) → export_weights() → ExportedWeights (f32 flat arrays)
                                                        ↓
                                            NpuSimulator::from_exported()  [CPU validation]
                                            NpuHardware::from_exported()   [real hardware]
```

### How They Behave (Exp 020 Measurements)

**Latency** (NpuSimulator, 50-reservoir, 8-input, 6-output):

| Percentile | Value | Notes |
|---|---|---|
| p50 | 331 µs | Consistent, narrow distribution |
| p95 | 403 µs | Low tail latency |
| p99 | 520 µs | Rare outliers from OS scheduling |
| mean | 341 µs | Matches hardware batch=8 measurements (390 µs) |

**Determinism**: Identical input → identical output, every time. Zero drift over
50 sequential batches. The NP mesh is fully deterministic digital logic.

**Quantization Error**: CPU f64 vs NpuSimulator f32 divergence < 1% on all tested
models. The physics is robust to f32→int8 truncation because the ESN readout is
a simple linear combination — quantization noise averages out.

**Weight Mutation**: 0.015ms on simulator, ~14ms on hardware. The readout weights
(`W_out`) can be hot-swapped without reprogramming the reservoir weights
(`W_in`, `W_res`). This enables online learning: retrain the readout layer as
new physics data arrives, push updated weights to NPU in 14ms.

---

## Part 4: What Workloads the NPU Manages

### Tested Workloads (Exp 020 Campaign)

| Workload | ESN Architecture | Accuracy | Throughput | Verdict |
|---|---|---|---|---|
| **Thermalization detection** | 10→50→1 | 87.5% | 3,000/s | **Best ROI** — saves 3.15h |
| **Rejection prediction** | 5→50→1 | 96.2% | 3,000/s | Near-perfect on small lattice |
| **Phase classification** | 8→50→1 | 100% (at n≥10) | 3,000/s | Reliable with adequate data |
| **β_c estimation** | 8→50→1 (multi-output) | ε=0.0098 | 3,000/s | Precise regression |
| **Anomaly scoring** | 8→50→6 (multi-output) | AUC=0.50 | 3,000/s | Needs more anomaly training data |
| **CG iteration prediction** | 8→50→6 (multi-output) | ε=0.0 | 3,000/s | Trivial for quenched (cg=0) |

### Pipeline Placement Results (6 positions tested)

| Position | Description | Trajectories Saved | Physics Accuracy |
|---|---|---|---|
| **A: Pre-thermalization** | Monitor plaquette convergence | **390 (21.7%)** | 83.3% |
| B: Mid-trajectory | Predict accept/reject early | 0 | 95.8% |
| C: Post-trajectory | Classify after completion | 0 (baseline) | 83.3% |
| D: Inter-beta | Steer next β-point | 0 | 45.5% |
| E: Pre-run bootstrap | Warm-start from historical data | 0 | 50.0% |
| **F: All combined** | A+B+C | **390** | **87.5%** |

### Where the NPU Wins

1. **Pre-thermalization screening**: The ESN learns plaquette convergence faster
   than fixed-N thermalization. At production scale (32⁴, 200 therm traj/β),
   this saves ~3.15h from the 5.1h thermalization budget. The NPU's contribution
   is a 341µs inference call vs a 7.64s GPU trajectory — 0.004% overhead.

2. **Continuous monitoring at zero GPU cost**: While the GPU runs HMC, the NPU
   independently classifies trajectory quality. No GPU cycles stolen.

3. **Multi-output for free**: A single 341µs inference produces phase label,
   β_c estimate, thermalization flag, acceptance probability, anomaly score,
   and CG iteration prediction simultaneously. On GPU or CPU, these would be
   6 separate computations.

### Where the NPU Does Not Help (Yet)

1. **Mid-trajectory early exit**: The high acceptance rate on 4⁴ lattices (96.5%)
   means almost no trajectories are rejected. On larger lattices with lower
   acceptance, the rejection predictor becomes valuable.

2. **Inter-beta steering**: Needs more training data. With 12 β-points, the ESN
   can't reliably predict which direction to steer. With 50+ points from
   historical data, this becomes viable.

3. **Direct force computation**: The NPU can't replace the GPU for the actual
   physics. SU(3) matrix operations require f64 precision and custom compute —
   exactly what GPUs excel at.

---

## Part 5: How It Compares to Loihi and Other Systems

### The Neuromorphic Landscape (2025-2026)

| Chip | Vendor | Architecture | Power | Availability | Learning |
|---|---|---|---|---|---|
| **AKD1000** (ours) | BrainChip | Digital, event-based, 80 NPs | ~30 mW | PCIe board, $300 | STDP (1-bit) |
| **Akida 2.0** | BrainChip | Same arch + TENNs, 8-bit | ~30 mW | IP license only | Enhanced STDP |
| **Loihi 2** | Intel | 128 async cores, programmable neurons | 30-80 mW/core | Research program | Microcode plasticity |
| **Speck** | SynSense | Event-driven vision processor | ~1 mW | Commercial | Limited |
| **TrueNorth** | IBM | 1M neurons, 256M synapses | 70 mW | Research only | None on-chip |

### Direct Comparison: AKD1000 vs Loihi 2

| Dimension | AKD1000 (ours) | Loihi 2 | Notes |
|---|---|---|---|
| **Access** | Commercial ($300 PCIe) | Intel research program | We own ours; Loihi requires NDA |
| **Neuron model** | Fixed (integrate-and-fire) | Programmable (microcode) | Loihi can implement arbitrary dynamics |
| **Reservoir computing** | FC-only, host-driven recurrence | Native spiking reservoir (Sigma-Pi neurons) | Loihi is architecturally superior for RC |
| **ESN performance** | 87.5% therm detection, 96.2% reject pred | Nature 2025: principled RC with polynomial features | Loihi demonstrated on chaotic dynamics |
| **PDE solving** | Not attempted | NeuroFEM (Sandia): Poisson equation on Loihi 2 | Loihi mapped FEM to spiking dynamics |
| **On-chip learning** | 1-bit STDP, last layer only | Full programmable plasticity | Loihi: 70× faster, 5600× more efficient than GPU |
| **Quantization** | int4 weights/activations | Configurable per neuron | Loihi more flexible |
| **Multi-output** | Free (all FC merge to one HW pass) | Per-core overhead | AKD1000 wins here |
| **Batch inference** | 2.4× at batch=8 | Not typical (event-driven) | Different paradigm |
| **Weight mutation** | 14ms via set_variable() | In-situ via plasticity engine | Loihi faster |
| **Power** | ~30 mW (but PCIe board is 900 mW) | 30-80 mW per core | Comparable at chip level |
| **Ecosystem** | Python SDK, thin Rust FFI | Lava framework (open-source) | Lava is richer |

### What Loihi Has That We Don't

1. **Programmable neuron models**: Loihi 2's microcode engine can implement
   arbitrary neuron dynamics — including reservoir computing natively on-chip
   without host-driven recurrence. Our AKD1000 requires the host to drive
   the ESN loop (tanh + state feedback), which adds PCIe roundtrip per frame.

2. **NeuroFEM (Sandia)**: Loihi 2 has been demonstrated solving the Poisson
   equation by mapping sparse FEM interactions to spiking neural populations.
   The AKD1000's fixed neuron model can't express this — it needs FC-compatible
   workloads.

3. **Principled reservoir computing** (Nature 2025): Loihi 2 implements
   Sigma-Pi neurons that compute higher-order polynomial features, enabling
   better scaling than traditional ESN approaches. Our reservoir is limited
   to linear input projection + tanh nonlinearity on host.

4. **Continual learning at 5600× efficiency** (arxiv 2511.01553): Loihi 2
   achieves 0.33ms/0.05mJ for continual learning tasks vs GPU's 23.2ms/281mJ.

### What We Have That Loihi Doesn't

1. **The hardware**: We own an AKD1000. Loihi 2 requires Intel's research
   program. We can probe, experiment, break things, and iterate at our pace.

2. **Multi-output inference for free**: Our 6-output model runs at the same
   latency as a 1-output model. This is an AKD1000-specific advantage from
   FC chain merging.

3. **Demonstrated physics pipeline**: No one has published neuromorphic-accelerated
   lattice QCD thermalization detection. Our pipeline (GPU HMC → NPU ESN screening)
   is, as far as we can find, novel.

4. **Weight mutation for online physics**: The 14ms set_variable() hot-swap
   enables the ESN to refine its predictions as the physics evolves — without
   reprogramming the device.

5. **Beyond-SDK characterization**: Our 10-discovery probe of the AKD1000
   revealed capabilities (batch inference, FC merge, clock modes, 16GB BAR)
   that aren't in BrainChip's public documentation.

### Other Relevant Work in the Field

| Project | Platform | What They Did | Relevance to Us |
|---|---|---|---|
| **NeuroFEM** (Sandia, 2024-25) | Loihi 2 | Solved Poisson equation via spiking FEM | Direct model for our Tier 3 exploration |
| **Principled RC** (Nature 2025) | Loihi 2 | Sigma-Pi reservoir with polynomial features | Better reservoir architecture than our ESN |
| **NeuroBench** (Nature 2025) | Multi-platform | Standardized neuromorphic benchmarks | Framework for comparing our results |
| **Resistive memory lattice QCD** (arxiv 2509.12812) | Analog RRAM | Normalizing flows for lattice field theory | 8-14× faster than HMC, 73-138× energy savings |
| **Particle tracking SNN** (arxiv 2502.06771) | Neuromorphic | STDP-based unsupervised particle tracking | Event-driven detector readout |
| **Dual memory pathway** (arxiv 2512.07602) | General | Fast-slow cortical organization | 40-60% fewer parameters, 4× throughput |
| **ESN vs LSM comparison** (MDPI 2025) | Software | ESN overfits; LSM generalizes under quantization | Suggests spiking models may be better long-term |

---

## Part 6: The Reservoir Question — ESN vs Spiking vs Hardware-Native

A critical finding from the 2025 literature: **ESNs and Liquid State Machines (LSMs)
have fundamentally different behavior under quantization.** ESNs achieve lower error
with small reservoirs but saturate and overfit as size increases. LSMs (spiking
reservoirs) demonstrate better generalization and robust performance under aggressive
quantization — making them architecturally better suited for neuromorphic hardware.

### Implications for Our Work

Our current approach — train ESN in f64, export to f32, quantize to int8 — works
(< 1% error). But the literature suggests we may be fighting the hardware:

| Approach | Pros | Cons | Hardware Fit |
|---|---|---|---|
| **ESN (current)** | Simple, proven, fast training | Host-driven recurrence, tanh on CPU | Mediocre — NPU just does FC readout |
| **LSM (spiking)** | Recurrence on-chip, quantization-robust | More complex training, needs Loihi-class hardware | AKD1000 can't do this |
| **Hardware-native int4+ReLU** | Maximizes AKD1000 silicon | Different dynamics than ESN | Best fit for AKD1000 |
| **Sigma-Pi reservoir** | Best scaling (Nature 2025) | Requires programmable neurons (Loihi 2) | Not possible on AKD1000 |

### What We Should Try Next

1. **Hardware-native reservoir**: Design reservoir dynamics that are optimal for
   the AKD1000's actual compute model — int4 weights, bounded ReLU activations,
   no tanh, no recurrence. This means the reservoir IS the FC chain, not an
   approximation of a continuous ESN.

2. **Unrolled recurrence**: Instead of host-driven loops, unroll the ESN time
   steps into FC depth. With FC chains being free (Discovery 2), a 10-step
   unrolled ESN would be 10 FC layers — all executing in one hardware pass.
   Trade model size for elimination of PCIe roundtrips.

3. **Quantization-aware training**: Train the ESN directly in int8/int4 rather
   than training in f64 and quantizing post-hoc. The 2025 ESN vs LSM comparison
   suggests this matters for generalization.

---

## Part 7: Akida 1.0 vs 2.0 — Where Our Hardware Sits

Our AKD1000 is **Akida 1.0**. BrainChip has released the Akida 2.0 IP (available
as licensable silicon, not as a standalone PCIe board yet). The differences matter:

| Feature | Akida 1.0 (ours) | Akida 2.0 |
|---|---|---|
| Weight precision | 1, 2, 4-bit | 1, 2, 4, **8-bit** |
| Activation precision | 1, 2, 4-bit | 1, 2, 4, **8-bit** |
| TENNs | No | **Yes** — temporal event-based neural networks |
| Skip connections | No | **Yes** — residual paths |
| LUT activations | No | **Yes** — GeLU, SiLU, LeakyReLU, PReLU, tanh |
| Dense1D | No | **Yes** — 1D dense layers up to 2048 dim |
| BufferTempConv | No | **Yes** — temporal convolution with FIFO |
| Vision Transformers | No | **Yes** — attention via SNN encoding |

### What Akida 2.0 Would Give Us

1. **LUT tanh**: The single biggest limitation of our current ESN pipeline is
   that tanh runs on host. Akida 2.0's lookup-table activation includes tanh,
   meaning the full ESN recurrence could run on-chip.

2. **8-bit precision**: Double the dynamic range for weights and activations.
   Our ESN's quantization error would drop from <1% to negligible.

3. **TENNs for time-series**: BrainChip's Temporal Event-based Neural Networks
   are purpose-built for the kind of streaming time-series prediction we're
   doing. They process only motion-relevant data — exactly the plaquette
   convergence detection we need.

4. **BufferTempConv**: Hardware FIFO-based temporal convolution would eliminate
   the host-side sliding window management.

### Feedback for BrainChip

Our Exp 020 results and Akida feedback report (`wateringHole/handoffs/AKIDA_BEHAVIOR_REPORT_FEB26_2026.md`)
raise specific questions:

1. Is GPU↔NPU peer-to-peer DMA possible via PCIe? (Eliminates CPU staging)
2. Is model-switching latency documented? (We need therm detector AND multi-output)
3. Does Economy clock degrade p99 latency under sustained hours-long load?
4. Is set_variable() atomic with respect to inference? (Torn read risk)
5. What's the practical model size limit for single-pass inference?
6. When will Akida 2.0 silicon be available as a PCIe board?

---

## Part 8: What We Don't Know Yet

| Question | Why It Matters | How We'd Test |
|---|---|---|
| Real hardware latency under sustained physics load | Thermal throttling? p99 drift? | Run 10,000+ consecutive inferences on AKD1000 |
| Actual chip power during inference | Board floor (900mW) hides chip compute | Need external current shunt or BrainChip's help |
| Economy vs Performance under different model sizes | Maybe Economy wins for small, Performance for large | Sweep model sizes × clock modes on hardware |
| Multi-model switching latency | We need fast swap between therm and multi-output | Load two .fbz files, measure switch time |
| BAR1 16GB address space mapping | Could enable direct NP SRAM read/write | mmap BAR1, probe for data at NP offsets |
| Hardware-native int4 reservoir quality | Current ESN is float approximation | Train int4 reservoir from scratch |
| Unrolled recurrence quality | Could eliminate host-driven loop | Build 10-depth FC model, compare to iterative ESN |
| PCIe Gen3 vs Gen2 improvement | We're on x1 Gen2 — is the bus the bottleneck? | Test on a Gen3 riser if AKD1000 supports it |

---

## Summary: The Metal Speaks

| What We Learned | Significance |
|---|---|
| AKD1000 handles 50-dim physics vectors directly | SDK channel limit is software, not silicon |
| FC chains merge into single hardware pass | Deep networks are free after PCIe cost is paid |
| Batch=8 gives 2.4× throughput | Buffer physics data, amortize PCIe |
| Multi-output is free | One inference → six physics predictions |
| Weight mutation works at 14ms | Online learning in the physics pipeline |
| Pre-thermalization screening saves 3.15h | **Biggest practical win** — ESN detects equilibrium |
| Thermalization detector: 87.5% accuracy | Reliable enough for production use with fallback |
| Rejection predictor: 96.2% accuracy | Ready for large-lattice deployment |
| PCIe latency dominates, not compute | The chip is fast; the bus is slow |
| No tanh, no recurrence in hardware | Host must drive ESN — Akida 2.0 may fix this |

The AKD1000 is not a GPU replacement. It is a **physics co-processor** — a 30mW
screening engine that watches the physics while the GPU does the heavy work. Its
value is not raw compute but **continuous low-overhead monitoring at points in the
pipeline where the GPU is doing something else**.

The neuromorphic landscape is evolving fast. Loihi 2 has architectural advantages
for reservoir computing (programmable neurons, on-chip recurrence, NeuroFEM).
Akida 2.0's TENNs and LUT activations would close the gap. But we have what we
have — a $300 commercial board, no NDA required, with capabilities that exceed
what the SDK advertises. The exploration continues.

---

## Cross-Substrate ESN Comparison (Exp 021)

A natural follow-up to Exp 020: if the NPU can run ESN inference, what does
that same workload look like on CPU and GPU? This experiment ran identical
ESN workloads on all three substrates, discovering where each silicon excels.

### GPU as ESN Reservoir

The RTX 3090 can run the same WGSL ESN shaders (`esn_reservoir_update.wgsl`,
`esn_readout.wgsl`) that were designed for ToadStool dispatch. This is the
first time these shaders were actually dispatched to GPU from hotSpring.

At RS=1024 with 200-step sequences, the GPU achieves **8.2× speedup** over
CPU-f64. The GPU "becomes the ESN" when the reservoir is large enough to
amortize dispatch overhead.

### The Crossover

GPU dispatch overhead is ~3.5ms per submit cycle. CPU ESN scales as O(RS²).
These curves cross at **RS ≈ 512** — below this, CPU wins; above, GPU wins.

This means the NPU's optimal zone (RS ≤ 200) has no GPU competition, while
the GPU's optimal zone (RS ≥ 512) has no NPU competition. The substrates
are naturally complementary.

### NPU Streaming Advantage

The NPU's 2.8 μs/step streaming inference is **1,000× faster** than GPU
per-step dispatch (3,170 μs). For "examine every trajectory as it arrives"
workloads, the NPU is the only viable real-time substrate.

### Substrate Hierarchy

1. **NPU**: Screening, classification, threshold detection — anything that
   needs per-sample decisions at microsecond latency
2. **GPU**: Large reservoir models, high-dimensional physics embedding,
   batch inference over many samples
3. **CPU**: Precision arbiter (f64), small models (RS < 200), training

---

## Files

| File | Purpose |
|---|---|
| `metalForge/npu/akida/HARDWARE.md` | AKD1000 architecture deep-dive |
| `metalForge/npu/akida/BEYOND_SDK.md` | 10 SDK assumptions tested against hardware |
| `metalForge/npu/akida/EXPLORATION.md` | Novel application analysis |
| `barracuda/src/md/reservoir.rs` | ESN + NpuSimulator implementation |
| `barracuda/src/md/npu_hw.rs` | Real AKD1000 hardware adapter |
| `barracuda/src/bin/npu_experiment_campaign.rs` | Exp 020 campaign binary |
| `barracuda/src/bin/cross_substrate_esn_benchmark.rs` | Exp 021 cross-substrate benchmark |
| `barracuda/src/gpu/buffers.rs` | f32 GPU buffer helpers (new for Exp 021) |
| `experiments/020_NPU_CHARACTERIZATION_CAMPAIGN.md` | Full experiment results |
| `experiments/021_CROSS_SUBSTRATE_ESN_COMPARISON.md` | Cross-substrate comparison results |
| `wateringHole/handoffs/AKIDA_BEHAVIOR_REPORT_FEB26_2026.md` | Akida feedback report |

## Part 9: Live NPU in Production QCD (Exp 022)

Experiment 022 is the culmination: the AKD1000 hardware NPU is live in the
production 32⁴ lattice QCD mixed pipeline via PCIe transfer. This is the first
time real neuromorphic silicon is integrated into a production lattice QCD run.

### What Changed

| Before (Exp 020-021) | After (Exp 022) |
|---|---|
| NpuSimulator (CPU f32 math) | **AKD1000 hardware via PCIe** |
| NPU characterization only | **NPU in production physics pipeline** |
| Single-run ESN | **Cross-run learning (bootstrap + export weights)** |
| NPU on main thread | **Dedicated NPU worker thread + mpsc channels** |

### Live Hardware Integration

The `npu-hw` cargo feature enables `akida-driver` — a pure Rust driver for the
AKD1000 that maps the PCIe BAR, programs NP mesh weights, and reads inference
results. The kernel module (`akida-pcie.ko`) was built from source for kernel
6.17 and loaded via `pkexec`. `/dev/akida0` is the device node.

### Cross-Run Learning Loop

Each run now produces trained ESN weights that the next run absorbs:

```
Run N: GPU HMC → trajectory log → ESN trains during run → --save-weights=esn.json
                                                                  ↓
Run N+1: --bootstrap-from=esn.json → ESN starts pre-trained → refines further
```

The ESN accumulates knowledge across runs. The 8⁴ validation showed 60%
thermalization early-exit rate; with cross-run bootstrap from historical data,
the 32⁴ run starts with a trained model from 749 simulator data points.

### 8⁴ Validation Results

| Metric | Value |
|---|---|
| Thermalization early-exits | 6/10 β points (60%) |
| Rejection prediction accuracy | 86.0% |
| Phase classifications | 10/10 |
| Total NPU calls | 5,947 |
| NPU overhead per trajectory | ~1.2ms (0.016% of 7.6s trajectory) |

### Production Status

The 32⁴ run with live AKD1000 hardware is currently in progress on biomeGate:
- RTX 3090 (DF64 HMC) + AKD1000 (hardware NPU) + Titan V (f64 oracle)
- ESN bootstrapped from 749 simulator data points
- Cross-run weights will be exported for future runs

---

## References

### Our Experiments
- Exp 022: NPU Offload Mixed Pipeline — live AKD1000 hardware, cross-run ESN, 4 placements
- Exp 021: Cross-Substrate ESN Comparison — GPU ESN dispatch, scaling crossover, capability envelope
- Exp 020: NPU Characterization Campaign — thermalization, rejection, multi-output, placement
- Exp 018: DF64 Production Benchmark (baselines to beat)
- Exp 013: biomeGate Production β-Scan (13.6h baseline)

### External
- Sandia NeuroFEM: "Solving sparse finite element problems on neuromorphic hardware" (Nature Machine Intelligence, 2025)
- "Principled neuromorphic reservoir computing" (Nature Communications, Jan 2025) — Sigma-Pi neurons on Loihi 2
- "Real-time Continual Learning on Intel Loihi 2" (arxiv 2511.01553, Nov 2025) — 70× faster, 5600× more efficient
- NeuroBench: "The neurobench framework for benchmarking neuromorphic computing" (Nature Communications, Feb 2025)
- "Modeling and Optimizing Performance Bottlenecks for Neuromorphic Accelerators" (arxiv 2511.21549) — AKD1000/Speck/Loihi 2 comparison
- "Reservoir Computing: Foundations, Advances, and Challenges Toward Neuromorphic Intelligence" (MDPI AI, 2025) — ESN vs LSM quantization comparison
- "Efficient lattice field theory simulation using adaptive normalizing flow on a resistive memory-based neural DE solver" (arxiv 2509.12812) — neuromorphic-adjacent lattice QCD
- "Unsupervised Particle Tracking with Neuromorphic Computing" (arxiv 2502.06771) — SNN for HEP detectors
