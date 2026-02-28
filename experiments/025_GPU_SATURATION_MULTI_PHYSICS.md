# Experiment 025: GPU Saturation & Multi-Physics Multi-GPU Strategy

**Status:** IN PROGRESS
**Date:** February 28, 2026
**Depends on:** Exp 024 (fermion force fix, production 8^4 run)
**License:** AGPL-3.0-only

---

## Motivation

The Exp 024 production run demonstrated stable dynamical fermion HMC with
NPU adaptive steering, but exposed massive hardware under-utilization:

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| VRAM (3090) | 13.7 MB | 24 GB | 0.06% |
| SM cores | 16/82 workgroups | 82 SMs | 20% |
| DRAM bandwidth | L2-resident (2.8 MB) | 936 GB/s | ~0% |
| Titan V | Idle | 12 GB HBM2 | 0% |

This experiment systematically saturates the hardware through:
1. Larger lattices (16^4, 32^4) on the 3090
2. Independent dynamical chains on the Titan V
3. Physics proxy models (Anderson 3D, Potts) on Titan V + CPU feeding NPU
4. Expanded NPU heads (12-14) for multi-physics predictions

## PCIe Topology (biomeGate, measured)

| Device | Slot | Link Speed | Width | Bandwidth |
|--------|------|------------|-------|-----------|
| RTX 3090 | 21:00.0 | Gen4 (16 GT/s) | x16 | ~25 GB/s |
| Titan V | 4b:00.0 | Gen3 (8 GT/s) | x8 | ~7.9 GB/s |
| Akida NPU | — | Gen2 | x1 | ~500 MB/s |

Titan V has x16 capability but the TRX40 slot provides x8. Independent
chains are unaffected (no cross-GPU traffic). Domain decomposition would
be bandwidth-limited.

## dt Scaling Analysis for 16^4

From Exp 024 production data (8^4, dt=0.01, m=0.1):
- Mean |ΔH| = 0.568, acceptance = 59%

Omelyan scaling: ΔH ∝ dt^4 × V. Volume ratio 16^4/8^4 = 16.

| dt | n_md | τ | Predicted |ΔH| | Predicted acc |
|----|------|---|-----------|-------------|
| 0.010 | 100 | 1.0 | 9.09 | ~0% |
| 0.005 | 200 | 1.0 | 0.57 | ~45% |
| 0.004 | 250 | 1.0 | 0.23 | ~63% |
| 0.003 | 333 | 1.0 | 0.07 | ~79% |

**Selected**: dt=0.005, n_md=200 (τ=1.0). Predicted 45-70% acceptance,
comparable to the 8^4 run. Conservative fallback: dt=0.003.

## CG Iteration Predictions

| Lattice | Sites | Predicted CG/traj | Est. wall/traj |
|---------|-------|--------------------|----------------|
| 8^4 | 4,096 | 57,000 (measured) | ~50s |
| 16^4 | 65,536 | ~114,000 | ~2-5 min |
| 32^4 | 1,048,576 | ~228,000 | ~8-15 min |

## Run A: 16^4 Single-Beta Validation

Quick validation to measure real scaling before committing to a long run.

```bash
HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin production_dynamical_mixed -- \
  --lattice=16 --betas=5.69 --mass=0.1 --dt=0.005 --n-md=200 \
  --therm=10 --quenched-pretherm=5 --meas=20 --seed=137 \
  --trajectory-log=../results/exp025_16x4_validation.jsonl \
  2>&1 | tee ../results/exp025_16x4_validation.log
```

**Success criteria:**
- [ ] Acceptance > 40%
- [ ] |ΔH| < 2
- [ ] CG iterations within 2x of prediction (~114K)
- [ ] SM utilization visibly higher than 8^4
- [ ] Total wall time measured

## Run B: Dual-GPU Independent Chains

Simultaneous dynamical HMC on both GPUs, different beta points.

```bash
# Terminal 1: 3090 on 16^4
HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin production_dynamical_mixed -- \
  --lattice=16 --betas=5.5,5.69,6.0 --mass=0.1 --dt=0.005 --n-md=200 \
  --therm=10 --quenched-pretherm=5 --meas=30 --seed=137 \
  --trajectory-log=../results/exp025_dual_3090.jsonl \
  2>&1 | tee ../results/exp025_dual_3090.log

# Terminal 2: Titan V on 8^4
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin production_dynamical_mixed -- \
  --lattice=8 --betas=5.0,5.3,5.6 --mass=0.1 --dt=0.01 --n-md=100 \
  --therm=10 --quenched-pretherm=5 --meas=30 --seed=42 \
  --trajectory-log=../results/exp025_dual_titan.jsonl \
  2>&1 | tee ../results/exp025_dual_titan.log
```

**Success criteria:**
- [ ] Both GPUs run simultaneously without contention
- [ ] Titan V produces valid physics (plaquette, Polyakov match 3090 at overlapping β)
- [ ] Combined throughput > 1.5x single-GPU

## Run C: Physics Proxy Pipeline (Anderson + Potts on Titan V)

See `barracuda/src/bin/gpu_physics_proxy.rs` for implementation.

## Run D: 32^4 Scale Test

```bash
HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin production_dynamical_mixed -- \
  --lattice=32 --betas=5.69 --mass=0.1 --dt=0.003 --n-md=333 \
  --therm=5 --quenched-pretherm=5 --meas=10 --seed=137 \
  --trajectory-log=../results/exp025_32x4_scaletest.jsonl \
  2>&1 | tee ../results/exp025_32x4_scaletest.log
```

**Success criteria:**
- [ ] Fits in 3090 VRAM (predicted 3.5 GB)
- [ ] CG converges (may need dt=0.002 if 0.003 is unstable)
- [ ] Wall time per trajectory measured for production planning

## NPU Head Expansion (11 → 14)

| Head | Name | Input | Output | Physics Basis |
|------|------|-------|--------|---------------|
| 12 | RMT Spectral | β, m, L | predicted λ_min, ⟨r⟩ | Random Matrix Theory |
| 13 | Potts Phase | β, L | phase label, β_c | Z(3) Potts universality |
| 14 | Anderson CG | β, m, plaq_var | predicted CG iters | Anderson localization proxy |

## Files

- `experiments/025_GPU_SATURATION_MULTI_PHYSICS.md` — this document
- `barracuda/src/bin/gpu_physics_proxy.rs` — Titan V proxy pipeline
- `results/exp025_*.jsonl` — run output data
- `results/exp025_*.log` — run logs
