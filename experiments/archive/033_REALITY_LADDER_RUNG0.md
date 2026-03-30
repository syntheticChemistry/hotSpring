# Experiment 033: Reality Ladder — Rung 0 (Mass Scan)

**Date:** 2026-03-02 → 2026-03-03
**Status:** COMPLETE
**Script:** `experiments/033_reality_ladder_rung0.sh`
**Results:** `results/exp033_{2x2,4x4,6x6}_m{mass}.jsonl`
**Analysis:** `whitePaper/baseCamp/reality_ladder_rung0.md`

## Summary

479 trajectories across 5 quark masses (0.01–0.5), 3–5 coupling strengths
(beta 5.0–6.0), and 3 lattice volumes (2^4, 4^4, 6^4) at Nf=4.

## Key Results

- Mass shifts the confinement transition: heavier quarks keep the system confined
- CG cost scales as ~1/m: 13k (m=0.5) → 25k (m=0.05)
- Mass=0.01 is completely broken (delta_H ~ 10^9) — proves need for Hasenbusch
- Volume amplifies difficulty: CG doubles from 4^4 → 6^4
- NPU now has 3D training manifold (mass x beta x volume)

## What Comes Next

- Rung 1: Multi-field comparison (Nf=4 vs 8 vs 12) — code ready
- Rung 2: GPU Hasenbusch to unlock m < 0.05 — code ready
- Rung 3: RHMC for physical Nf=2, 2+1 — code ready
