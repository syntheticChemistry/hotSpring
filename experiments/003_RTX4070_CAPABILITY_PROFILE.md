# Experiment 003: RTX 4070 Capability Profile

**Date**: February 15, 2026  
**Hardware**: RTX 4070 (12 GB GDDR6X), i9-12900K, 64 GB DDR5  
**Prerequisite**: Native f64 builtins confirmed (Experiment 001, Phase D)  
**Status**: PLANNED — ready to execute  
**Estimated run time**: 4-8 hours (depending on sweep range)

---

## Motivation

Experiment 001 established N-scaling from 500 to 20,000 particles with native f64 builtins.
The bottleneck is broken, and the RTX 4070 achieves paper parity (N=10,000) in 5.3 minutes.
But we don't yet know the **practical ceiling** — what is the maximum N, step count, and
parameter diversity this card can handle before time, energy, or VRAM become limiting?

This experiment profiles the full capability envelope of a single consumer GPU for
computational plasma physics. The goal is to answer:

1. **Max N**: Where does the RTX 4070 hit VRAM limits? Where does throughput drop below 1 step/s?
2. **Max steps**: At paper-parity N=10,000, how long can we run? What's the energy budget for overnight runs?
3. **Exploration space**: How many distinct (κ, Γ) parameter combinations can we sweep in a day?
4. **Reference nuclei scaling**: What happens to L1/L2 accuracy when we expand from 52 to 200+ AME2020 nuclei?
5. **Cost model**: Time and energy per experiment → how does this compare to HPC cluster allocations?

---

## Part 1: N-Scaling Beyond 20,000

### Protocol

Extend the Experiment 001 sweep to larger N values. Use κ=2, Γ=158 (textbook OCP case).
35,000 production steps per N. Cell-list kernel for N >= 5,000.

### Sweep values

| N | Expected Method | Expected VRAM | Estimated steps/s | Estimated Wall |
|---|----------------|---------------|-------------------|----------------|
| 20,000 | cell-list | ~600 MB | ~56 (measured) | ~10 min |
| 50,000 | cell-list | ~1.5 GB | ~15-25 | ~25-40 min |
| 100,000 | cell-list | ~3 GB | ~5-15 | ~40-120 min |
| 200,000 | cell-list | ~6 GB | ~3-8 | ~1-3 hrs |
| 400,000 | cell-list | ~12 GB | ~1-4 | ~3-10 hrs |
| 500,000 | cell-list | ~15 GB (OOM?) | — | OOM test |

### Metrics to capture per N

- steps/s (sustained, excluding setup)
- Wall time (total including setup)
- GPU power (avg W, peak W)
- GPU energy (J, kWh)
- GPU temperature (avg °C, peak °C)
- VRAM usage (MB)
- Energy conservation (relative drift)
- Energy per step (J/step)

### Acceptance criteria

- Energy drift < 0.01% (relaxed from 0.005% for very large N)
- GPU temperature < 85°C sustained
- No OOM or driver timeout

---

## Part 2: Long Production Runs (N=10,000)

### Protocol

At paper-parity N=10,000, run extended production to determine:
- Observable convergence (how many steps until RDF/VACF/SSF stabilize?)
- Energy conservation over very long runs (>1M steps)
- Overnight budget: energy and time

### Sweep values

| Steps | Estimated Wall | Estimated Energy | Estimated Cost |
|-------|----------------|------------------|----------------|
| 35,000 | 5.3 min | 19.4 kJ | $0.001 |
| 100,000 | ~15 min | ~56 kJ | $0.003 |
| 500,000 | ~75 min | ~280 kJ | $0.015 |
| 1,000,000 | ~2.5 hrs | ~560 kJ | $0.030 |
| 5,000,000 | ~12.5 hrs | ~2.8 MJ | $0.150 |
| 10,000,000 | ~25 hrs | ~5.6 MJ | $0.300 |

At $0.30 for 10 million steps at paper-parity particle count, the question becomes:
is there scientific value in longer production? (Yes — statistical convergence of
observables improves as √N_steps.)

---

## Part 3: Parameter Space Sweep

### Protocol

Sweep a grid of (κ, Γ) values at N=10,000 to map the full Yukawa OCP phase space.
The Murillo Group's published study covers 9 PP cases (κ=1,2,3 × 3 Γ values).
We can cover much more.

### Proposed sweep grid

| κ | Γ values | Cases | Estimated Time | Notes |
|---|----------|-------|----------------|-------|
| 0.5 | 5, 10, 20, 50, 100, 200 | 6 | ~32 min | Weak screening |
| 1 | 5, 14, 30, 72, 150, 217, 500 | 7 | ~37 min | Paper values + extensions |
| 2 | 10, 31, 80, 158, 300, 476, 1000 | 7 | ~37 min | Paper values + extensions |
| 3 | 30, 100, 250, 503, 1000, 1510, 3000 | 7 | ~37 min | Paper values + extensions |
| 4 | 50, 200, 500, 1000, 2000 | 5 | ~27 min | Strong screening (novel) |
| 5 | 100, 500, 1000, 5000 | 4 | ~21 min | Very strong screening (novel) |
| **Total** | | **36 cases** | **~3 hours** | 4× the Murillo study |

35,000 production steps per case. This would be the most comprehensive consumer-GPU
Yukawa OCP phase space survey — done in a single afternoon for ~$0.04 in electricity.

### Novel parameter space

The Murillo Group tested κ ∈ {1, 2, 3}. Testing κ ∈ {0.5, 4, 5} explores:
- Weak screening (κ=0.5): transition toward Coulomb behavior
- Strong screening (κ=4, 5): approaching hard-sphere limit

These are regimes where published simulation data is sparse. Consumer GPU makes
this exploration trivial.

---

## Part 4: Reference Nuclei Scaling (L1/L2)

### Protocol

The current L1/L2 pipeline uses 52 nuclei from AME2020. The full AME2020 dataset
contains **2,457** experimentally measured masses. More reference nuclei means:
- Tighter constraints on Skyrme parameters (more data points to fit)
- Better coverage of shell effects, deformation regions
- Higher-confidence NMP extrapolation

### Sweep values

| N_nuclei | Expected L1 eval time | Expected L1 sweep (512 pts) | Notes |
|----------|----------------------|----------------------------|-------|
| 52 | 39.7 μs | 3.75s | Current (validated) |
| 100 | ~76 μs | ~7.3s | Double the constraints |
| 200 | ~153 μs | ~14.6s | Major shell coverage |
| 500 | ~382 μs | ~36.5s | Most of light/medium |
| 1,000 | ~765 μs | ~73s | Heavy nuclei included |
| 2,457 | ~1.9 ms | ~180s | Full AME2020 |

Even at the full 2,457 nuclei, a 512-point LHS sweep takes only 3 minutes on GPU.
This is a computational trivial expansion that may significantly improve accuracy.

### Questions to answer

1. Does expanding from 52 → 200 nuclei change the optimal Skyrme parameters?
2. Does the χ²/datum improve or worsen? (More data can expose model deficiencies.)
3. How do NMP values change with more constraints?
4. Is there a "sweet spot" number of nuclei that balances accuracy and physical insight?

---

## Part 5: Cost Model Comparison

### GPU vs HPC

| Resource | RTX 4070 (Eastgate) | MSU HPCC (iCER) | Ratio |
|----------|:-------------------:|:----------------:|:-----:|
| Cost/hour | ~$0.01 electricity | ~$0.10-1.00 SU | 10-100× cheaper |
| Setup time | Seconds (local) | Minutes-hours (queue) | 10-100× faster |
| Availability | 24/7 | Queue-dependent | Always available |
| Max N (practical) | ~200-400K | ~1M+ | HPC wins for huge N |
| FP64 TFLOPS | ~0.3 (bandwidth-limited ~1:2) | ~10-100 (A100/H100) | HPC wins for compute |
| Energy per paper-parity run | 19.4 kJ | ~100-500 kJ (shared infra) | 5-25× more efficient |

**The key insight**: For N ≤ 100,000 and exploration-phase work (parameter sweeps,
prototyping, validation), the consumer GPU is not just cheaper — it's faster to start
and produces identical physics. HPC becomes necessary only for N > 200,000 or
production runs requiring maximum throughput.

---

## Execution Plan

1. **Part 1 first** (N-scaling): ~2-4 hours. Establishes the hardware ceiling.
2. **Part 3 next** (parameter sweep): ~3 hours. Produces the most novel data.
3. **Part 2 overnight** (long production): 12-25 hours. Set and forget.
4. **Part 4 during analysis** (nuclei scaling): ~30 min. Quick investigation.
5. **Part 5** is analysis, not computation.

Total estimated GPU time: **1-2 days** for the full capability profile.
Total estimated electricity cost: **< $1.00**.

---

## Expected Outcomes

1. A complete capability map of the RTX 4070 for Yukawa OCP molecular dynamics
2. The most comprehensive consumer-GPU phase space survey of Yukawa plasmas
3. Data on whether more reference nuclei improve L1/L2 accuracy
4. A quantitative cost model comparing consumer GPU to HPC cluster
5. Identification of the N and step-count boundaries where Titan V is needed

---

*This experiment plan was designed after confirming native f64 builtins on RTX 4070.
The $0.001/run economics make all of these experiments feasible on a PhD student budget.*
