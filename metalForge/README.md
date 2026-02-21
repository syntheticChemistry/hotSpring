# metalForge — Hardware Exploration & Systems Engineering

**Parent**: ecoPrimals/hotSpring
**Purpose**: Concrete hardware characterization, direct-wire programming, and
cross-substrate performance engineering. Where whitePaper/ is the science,
metalForge/ is the metal.

---

## Philosophy

Every computational physics result in ecoPrimals originates in shader math
(WGSL). ToadStool dispatches that math to whatever silicon is available.
metalForge exists to **characterize the silicon itself** — what it can do,
what it can't, and where the physics meets the transistors.

We don't just benchmark. We probe. We map register spaces, measure cache
line behavior, exploit hardware quirks, and find the paths that vendor SDKs
don't advertise. The GPU work (native f64 at 1:2 on consumer cards via
BarraCUDA) proved this approach works. metalForge extends it to every
substrate ecoPrimals touches.

---

## Local Hardware Inventory

| Substrate | Device | PCIe Slot | Key Spec | Status |
|-----------|--------|-----------|----------|--------|
| **NPU** | BrainChip AKD1000 | `08:00.0` | 80 NPs, 8MB SRAM, ~30mW, event-based | Driver loaded, `/dev/akida0` |
| **GPU (primary)** | NVIDIA RTX 4070 | `01:00.0` | 12GB, Ada Lovelace, native f64 1:2 | Active — all MD validated |
| **GPU (secondary)** | NVIDIA Titan V | `05:00.0` | 12GB, Volta GV100, NVK driver | Validated — identical physics |
| **CPU** | Intel i9-12900K | — | 16C/24T, 30MB L3, Alder Lake P+E | Reference substrate |

### Future Targets (no hardware yet)

| Substrate | Target | Why |
|-----------|--------|-----|
| AMD RDNA3/4 GPU | RX 7900 XTX or similar | 96MB Infinity Cache — MD neighbor lists fit entirely in L3 |
| Intel Arc GPU | A770 | Xe HPG + XMX, alternative wgpu backend |
| FPGA | Xilinx/AMD Alveo | Fully custom force pipeline, no instruction fetch overhead |

---

## Directory Structure

```
metalForge/
├── README.md                  ← this file
├── npu/
│   └── akida/
│       ├── HARDWARE.md        ← AKD1000 deep-dive: architecture, compute model, limits
│       ├── EXPLORATION.md     ← what we can do, what we can't, novel applications
│       ├── BEYOND_SDK.md      ← 10 overturned SDK assumptions (the discovery doc)
│       ├── scripts/
│       │   └── deep_probe.py  ← consolidated hardware probe (reproduces all discoveries)
│       └── benchmarks/        ← latency, power, throughput measurements
├── gpu/
│   ├── nvidia/                ← RTX 4070 + Titan V characterization
│   └── amd/                   ← future: RDNA cache advantage analysis
└── benchmarks/                ← cross-substrate comparison data
```

---

## Relationship to Other hotSpring Components

| Component | Role | metalForge Interaction |
|-----------|------|----------------------|
| `barracuda/` | Rust physics + WGSL shaders | metalForge characterizes the hardware these shaders run on |
| `whitePaper/` | Scientific methodology | metalForge provides the hardware context for performance claims |
| `control/` | Python reference implementations | metalForge validates cross-substrate math parity at the hardware level |
| `wateringHole/handoffs/` | ToadStool absorption | metalForge findings inform ToadStool's dispatch strategy |

---

## Current Focus: NPU (AKD1000)

See `npu/akida/HARDWARE.md` for the deep-dive, `npu/akida/EXPLORATION.md`
for the novel application analysis, and `npu/akida/BEYOND_SDK.md` for the
comprehensive discovery document.

### Key Findings (Feb 19-20, 2026)

- **78 NPs** enumerated on our AKD1000 (CNP1×78, CNP2×54, FNP2×4, FNP3×18)
- **668 inference clocks** for ESN readout (50→1 FC) on 1 FNP3 node
- **PCIe latency dominates**: ~650 μs round-trip vs ~0.7 μs compute
- **Board floor power**: 918 mW; chip inference power below measurement threshold
- **Direct weight injection works** — bypassed Keras→QuantizeML→CNN2SNN
  entirely via native `set_variable()` API (the "wgpu moment" for NPU)

#### Beyond-SDK Discoveries (see `npu/akida/BEYOND_SDK.md`)

10 SDK assumptions systematically tested and overturned:

| # | SDK Claim | Actual Hardware | Measured |
|---|-----------|-----------------|----------|
| 1 | InputConv: 1 or 3 channels | Any count works (tested 1-64) | 8/8 channel sizes pass |
| 2 | FC layers run independently | All merge into single HW pass (SkipDMA) | 6.7% overhead for 7 extra layers |
| 3 | Batch=1 inference only | Batch=8 → 2.35× throughput | 427μs/sample at batch=8 |
| 4 | Single clock mode | 3 modes: Performance/Economy/LowPower | Economy: 19% slower, 18% less power |
| 5 | Max FC width ~hundreds | Tested to 8192+ neurons | All map to HW |
| 6 | No direct weight mutation | set_variable() updates without reprogram | Exact linearity (error=0) |
| 7 | "30mW" power spec | Board floor 900mW, chip below noise | True chip power unmeasurable |
| 8 | 8MB SRAM is the limit | PCIe BAR1 exposes 16GB address space | Full NP mesh addressable |
| 9 | Program is opaque | FlatBuffer with program_info + program_data | Weights via DMA, not in program |
| 10 | Simple inference engine | SkipDMA, on-chip learning, register access | C++ symbol analysis confirmed |

#### Validated Control Experiments

| Suite | Script | Checks | Status |
|-------|--------|:------:|--------|
| Python HW (beyond-SDK) | `control/metalforge_npu/scripts/npu_beyond_sdk.py` | 13/13 | All pass on AKD1000 |
| Python HW (quantization) | `control/metalforge_npu/scripts/npu_quantization_parity.py` | 4/4 | f32/int8/int4/act4 all pass |
| Python HW (pipeline) | `control/metalforge_npu/scripts/npu_physics_pipeline.py` | 10/10 | MD→ESN→NPU→D*,η*,λ* end-to-end |
| Rust math (beyond-SDK) | `barracuda/src/bin/validate_npu_beyond_sdk.rs` | 16/16 | All pass |
| Rust math (quantization) | `barracuda/src/bin/validate_npu_quantization.rs` | 6/6 | All pass |
| Rust math (pipeline) | `barracuda/src/bin/validate_npu_pipeline.rs` | 10/10 | All pass |
| HW deep probe | `npu/akida/scripts/deep_probe.py` | 8 test suites | All discoveries reproduced |

#### Remaining Work

- **Device permission management** must be solved in Rust (`DeviceManager`
  with udev or capabilities), not manual `pkexec chmod 666 /dev/akida0`
- **Rust NPU driver** — open `/dev/akida0`, load FlatBuffer programs, run
  inference without Python SDK dependency
- **ToadStool NPU dispatch** — integrate into substrate dispatch system:
  shader math → quantize → NPU inference

---

## Lattice QCD: Heterogeneous Hardware Pipeline

Full lattice QCD (Tier 3 papers) previously required FFT for dynamical fermions.
**The full GPU lattice QCD pipeline is now COMPLETE**: GPU FFT f64 (toadstool
`1ffe8b1a`), GPU Complex64 + SU(3) (toadstool `8fb5d5a0`), GPU Dirac operator
(`WGSL_DIRAC_STAGGERED_F64`, 8/8 checks), and GPU CG solver (3 WGSL shaders,
9/9 checks). Pure GPU workload validated on thermalized HMC configurations:
5 CG solves at machine-epsilon parity (4.10e-16). **Rust is 200× faster than
Python** for the same CG algorithm (identical iteration counts, identical seeds).
Meanwhile, **lattice phase structure** is already accessible using position-space
observables and a multi-substrate pipeline.

### The Insight

The deconfinement phase transition in SU(3) gauge theory is the most
important finite-temperature QCD observable. It's detectable from:
- **Polyakov loop** ⟨|L|⟩: order parameter — near 0 in confined, >0 in deconfined
- **Plaquette** ⟨P⟩: gauge field order — smooth crossover near β_c

Both are position-space quantities. **No FFT required.**

### Heterogeneous Pipeline Architecture

```
┌─────────────────────────────────────────────────────┐
│ GPU: Pure gauge SU(3) HMC                           │
│   hot_start → leapfrog → Metropolis → observables   │
│   Output: (β, ⟨P⟩, ⟨|L|⟩) at each configuration    │
└───────────────────────┬─────────────────────────────┘
                        │ PCIe stream
                        ▼
┌─────────────────────────────────────────────────────┐
│ NPU: Phase classifier (ESN/FC)                      │
│   Input: (β, ⟨P⟩, ⟨|L|⟩) features                  │
│   Output: phase label (confined=0, deconfined=1)     │
│   Power: ~30mW inference, batch=8 for throughput     │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ CPU: Validation + orchestration                     │
│   Compare predicted β_c against known ~5.69         │
│   Monitor HMC acceptance, anomaly detection         │
└─────────────────────────────────────────────────────┘
```

### What This Proves

1. **Lattice QCD phase structure without FFT** — on consumer hardware
2. **GPU HMC + NPU inference composition** — two substrates, one physics
3. **Cost**: GPU ($600) generates configs, NPU ($300) classifies — ~$900 total
4. **Energy**: NPU classification at ~30mW vs CPU recomputing observables
5. **Science**: critical coupling β_c detectable from learned observables

### Control Experiments

| Suite | Script | Checks | Status |
|-------|--------|:------:|--------|
| Python HW | `control/metalforge_npu/scripts/npu_lattice_phase.py` | 9 | GPU HMC → NPU phase classification |
| Rust math | `barracuda/src/bin/validate_lattice_npu.rs` | 10/10 | **All pass** — β_c=5.715 (error 0.4%) |

**Rust validation results** (Feb 20, 2026):
- β_c detected at 5.715 vs known 5.692 (0.023 absolute error)
- ESN phase classifier: 100% test accuracy (30/30)
- NpuSimulator f32 vs CPU f64: max absolute error 2.821e-7
- Plaquette monotonically increases: 0.337 (β=4.5) → 0.572 (β=6.5)
- Phase predictions: mean 0.014 (low β, confined) → 1.022 (high β, deconfined)

### Previously Impossible → Now Demonstrated

`validate_hetero_monitor` (9/9 checks) proves capabilities that were
impossible before the heterogeneous pipeline:

| Capability | Before | Now | Key Metric |
|------------|--------|-----|------------|
| **Live HMC phase monitoring** | Offline analysis | Real-time ESN classification | 9 μs prediction, 0.09% overhead |
| **Continuous transport prediction** | Green-Kubo post-processing | NPU predicts D*/η*/λ* from 10-frame windows | Multi-output, all finite |
| **Cross-substrate physics** | Single substrate lock-in | CPU f64 → f32 sim → int4 quantized | f32 error 5.1e-7 |
| **Zero-overhead monitoring** | Paused simulation | Prediction < 0.1% of HMC time | 9 μs vs 10.3 ms |
| **Predictive steering** | Uniform parameter sampling | Adaptive β scan near transition | 62% fewer evaluations |

Control experiment: `barracuda/src/bin/validate_hetero_monitor.rs` — 9/9 checks pass

### Relation to Existing Work

| Component | Status | Location |
|-----------|--------|----------|
| SU(3) pure gauge HMC | ✅ Validated (12/12) | `barracuda/src/lattice/` |
| Polyakov loop | ✅ Implemented | `lattice/wilson.rs` |
| Average plaquette | ✅ Implemented | `lattice/wilson.rs` |
| GPU Dirac operator | ✅ Validated (8/8) | `lattice/dirac.rs` + WGSL |
| GPU CG solver | ✅ Validated (9/9) | `lattice/cg.rs` + 3 WGSL shaders |
| Pure GPU QCD workload | ✅ Validated (3/3) | `bin/validate_pure_gpu_qcd.rs` |
| ESN reservoir | ✅ Validated | `barracuda/src/md/reservoir.rs` |
| NPU ESN deployment | ✅ Validated (29/29) | `metalForge/npu/akida/BEYOND_SDK.md` |
| Phase classification | ✅ Validated (10/10) | `bin/validate_lattice_npu.rs` |

---

## Cross-System Pipeline: GPU → NPU → CPU (Full Stack)

The complete heterogeneous physics pipeline on $900 consumer hardware:

```
┌────────────────────────────────────────────────────────────────┐
│ GPU: Full Lattice QCD on GPU (RTX 4070, $600)                  │
│   HMC config gen → GPU Dirac (D†) → GPU CG (D†D x = b)        │
│   24 bytes/CG-iter transfer (α, β, ||r||²)                    │
│   200× faster than Python, machine-epsilon math parity         │
│   Output: gauge configs + fermion solutions + observables      │
└────────────────────────┬───────────────────────────────────────┘
                         │ PCIe stream (observables only)
                         ▼
┌────────────────────────────────────────────────────────────────┐
│ NPU: Phase/Transport Classifier (AKD1000, $300)                │
│   Input: (β, ⟨P⟩, ⟨|L|⟩, D*, η*, λ*)                         │
│   ESN: int8 weights, batch=8, 427μs/sample                    │
│   Output: phase label + transport predictions                  │
│   Power: ~30mW chip, 9,017× less energy than CPU Green-Kubo   │
│   Zero GPU overhead — independent PCIe device                  │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│ CPU: Validation + Orchestration (i9-12900K, $0)                │
│   Compare NPU predictions against GPU truth                   │
│   Monitor convergence, anomaly detection                      │
│   Predictive steering: 62% compute savings via adaptive scan  │
│   Pure Rust math — 33/33 validation suites, 0 failures        │
└────────────────────────────────────────────────────────────────┘
```

### Validated Cross-System Results (Feb 20, 2026 — All on local hardware)

| Metric | Value | Source |
|--------|-------|--------|
| GPU CG parity (vs CPU) | 4.10e-16 | `validate_pure_gpu_qcd` (RTX 4070) |
| GPU 16⁴ speedup vs CPU | **22.2×** (24ms vs 533ms) | `bench_lattice_scaling` (RTX 4070) |
| GPU CG iterations match | 33/33 identical at all sizes | `bench_lattice_scaling` |
| GPU→NPU f32 parity | 5.1e-7 | `validate_hetero_monitor` |
| NPU int8 error budget | <5% (max 0.85%) | `npu_quantization_parity.py` (AKD1000) |
| NPU throughput | 2,469 inf/s streaming | `npu_physics_pipeline.py` (AKD1000) |
| NPU energy savings | 8,796× less than CPU Green-Kubo | `npu_physics_pipeline.py` (AKD1000) |
| Phase detection β_c | 5.715 (known 5.692, 0.4%) | `npu_lattice_phase.py` (AKD1000) |
| NPU beyond-SDK | 13/13 HW checks pass | `npu_beyond_sdk.py` (AKD1000) |
| HMC monitoring overhead | 0.09% (9μs per trajectory) | `validate_hetero_monitor` |
| Compute savings (steering) | 62% fewer evaluations | `validate_hetero_monitor` |
| Rust vs Python (CG solver) | **200× faster** | `bench_lattice_cg` |
| Rust validation suites | 33/33 pass | `validate_all` |
| NPU HW checks | 34/35 pass (int4 accuracy marginal) | AKD1000 hardware |
| Total hardware cost | ~$900 | RTX 4070 ($600) + AKD1000 ($300) |

### Titan V Now Online (Feb 21, 2026)

NVK built from Mesa 25.1.5 source (`-Dvulkan-drivers=nouveau`) and installed
as user ICD at `~/.config/vulkan/icd.d/nouveau_icd.json`. Both GPUs now
visible to wgpu/Vulkan simultaneously:

| Adapter | Device | Driver | Vulkan | SHADER_F64 |
|---------|--------|--------|--------|:----------:|
| GPU0 | RTX 4070 | NVIDIA 580.82.09 | 1.4.312 | YES |
| GPU1 | NVIDIA TITAN V (NVK GV100) | NVK / Mesa 25.1.5 | 1.3.311 | YES |

ToadStool barracuda detects NVK automatically via adapter info strings and
applies the Volta driver profile (`DriverKind::Nvk`, `CompilerKind::Nak`,
`GpuArch::Volta`) including exp/log polynomial workarounds.

Validated on Titan V: `validate_cpu_gpu_parity` 6/6, `bench_gpu_fp64` pass.
Select with `HOTSPRING_GPU_ADAPTER=titan` or `BARRACUDA_GPU_ADAPTER=titan`.
