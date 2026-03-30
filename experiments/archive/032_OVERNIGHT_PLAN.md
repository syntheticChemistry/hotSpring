# Experiment 032: Overnight Run Plan

**Date**: 2026-03-02
**Hardware**: RTX 3090 (NVK/NAK GA102) + AKD1000 NPU
**Lattice**: 8^4 (4096 sites)

## Validated Parameters

| Run | dt | n_md | traj_len | delta_H | accept% | wall/traj |
|-----|-----|------|----------|---------|---------|-----------|
| v2 | 0.02 | 20 | 0.4 | 2.8 | 0% | 70s |
| v6 | 0.0125 | 40 | 0.5 | 1.0 | 0% | 131s |
| v7 | 0.01 | 50 | 0.5 | 0.75 | 50% | 168s |

## Overnight Parameters

Based on delta_H ∝ V × dt^4 scaling:

```
dt=0.008, n_md=63, trajectory_length=0.504
→ Expected delta_H ≈ 0.31 → ~75% acceptance
→ 189 CG calls/trajectory × ~1.0s/call = ~189s/trajectory
→ 8 hours = 28800s → ~150 trajectories
→ After 20 therm: ~130 measurement configurations
```

## Command

```bash
cd /home/biomegate/Development/ecoPrimals/hotSpring && \
PATH="/home/biomegate/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/local/bin:/usr/bin:/bin" \
HOTSPRING_GPU_ADAPTER=3090 \
nohup ./barracuda/target/release/production_dynamical_mixed \
  --lattice=8 \
  --betas=5.69 \
  --mass=0.1 \
  --dt=0.008 \
  --n-md=63 \
  --therm=20 \
  --meas=130 \
  --quenched-pretherm=10 \
  --seed=20260302 \
  --no-titan \
  --no-npu-control \
  --trajectory-log=results/exp032_8x8_overnight.jsonl \
  > results/exp032_8x8_overnight.log 2>&1 &
```

## Key Engineering in This Run

1. **Latency-adaptive CG**: Auto-scales check_interval when readback > 5ms (NVK)
2. **Dispatch coalescing**: Pre-CG and post-CG GPU work batched into single submits
3. **CG-precise compilation**: Dirac/dot/axpy/xpay via WGSL-text (no FMA fusion)
4. **Sovereign f64**: Transcendental shaders (sin, cos, exp) via SPIR-V passthrough
5. **NPU parameter override disabled**: Manual dt/n_md for volume-appropriate stepping

## Monitoring

```bash
# Trajectory count and acceptance
wc -l results/exp032_8x8_overnight.jsonl
grep -c '"accepted":true' results/exp032_8x8_overnight.jsonl

# Last trajectory
tail -1 results/exp032_8x8_overnight.jsonl | python3 -m json.tool

# Process health
ps aux | grep production_dynamical | grep -v grep
```

## Physics Expectations

- β=5.69, m=0.1 on 8^4: transition/crossover region
- ⟨P⟩ ≈ 0.45-0.50
- Polyakov loop: small magnitude (confined-ish at this β/volume)
- CG iterations: ~32k per trajectory (~210/solve × 152 solves)
