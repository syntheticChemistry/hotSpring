# hotSpring White Paper

**Status**: Working draft — reviewed for PII, suitable for public repository  
**Purpose**: Document the replication of Murillo Group computational plasma physics on consumer hardware using BarraCUDA  
**Date**: February 2026

---

## Documents

| Document | Description | Audience |
|----------|-------------|----------|
| [STUDY.md](STUDY.md) | **Main study** — full writeup of the two-phase validation, data sources, results, and path to paper parity | Reviewers, collaborators |
| [BARRACUDA_SCIENCE_VALIDATION.md](BARRACUDA_SCIENCE_VALIDATION.md) | Phase B technical results — BarraCUDA vs Python/SciPy numbers | Technical reference |
| [CONTROL_EXPERIMENT_SUMMARY.md](CONTROL_EXPERIMENT_SUMMARY.md) | Phase A summary — Python reproduction of published work | Quick reference |
| [METHODOLOGY.md](METHODOLOGY.md) | Two-phase validation protocol | Methodology review |

---

## What This Study Is

hotSpring replicates published computational plasma physics from the Murillo Group (Michigan State University) on consumer hardware, then re-executes the computations using BarraCUDA — a Pure Rust scientific computing library with zero external dependencies.

The study answers two questions:
1. **Can published computational science be independently reproduced?** (Answer: yes, but it required fixing 5 silent bugs and rebuilding physics that was behind a gated platform)
2. **Can Rust + WebGPU replace the Python scientific stack for real physics?** (Answer: yes — BarraCUDA achieves 3.8-8.3x better accuracy than Python/SciPy)

---

## Key Results

### Phase A (Python Control): 81/81 checks pass

- Sarkas MD: 12 cases, 60 observable checks, 8.3% mean DSF peak error
- TTM: 6/6 equilibration checks pass
- Surrogate learning: 9/9 benchmark functions converge
- 5 silent upstream bugs found and fixed

### Phase B (BarraCUDA): Python parity exceeded

| Level | BarraCUDA | Python/SciPy | Improvement |
|-------|-----------|-------------|-------------|
| L1 (SEMF) | 0.80 chi2/datum | 6.62 | 8.3x better, 400x faster |
| L2 (HFB) | 16.11 chi2/datum | 61.87 | 3.8x better, zero deps |
| L2 (NMP-physical) | 19.29 chi2/datum | 61.87 | 3.2x better, all NMP within 2sigma |

---

## Relation to Other Documents

- **`whitePaper/barraCUDA/`** (main repo, gated): The BarraCUDA evolution story — how scientific workloads drove the library's development. Sections 04 and 04a reference hotSpring data.
- **`whitePaper/gen3/`** (main repo, gated): The constrained evolution thesis — hotSpring provides quantitative evidence for convergent evolution between ML and physics math.
- **`wateringHole/handoffs/`** (internal): Detailed technical handoffs to the ToadStool/BarraCUDA team with code locations, bug fixes, and GPU roadmap.
- **This directory** (`hotSpring/whitePaper/`): Public-facing study focused on the science replication itself.

---

## Reproduction

```bash
# Phase A (Python, ~12 hours total)
bash scripts/regenerate-all.sh

# Phase B (BarraCUDA, ~2 hours total)
cd barracuda
cargo run --release --bin nuclear_eos_l1_ref          # L1: ~3 seconds
cargo run --release --bin nuclear_eos_l2_ref -- --seed=42 --lambda=0.1   # L2: ~55 min
```

No institutional access required. No Code Ocean account. No Fortran compiler. AGPL-3.0 licensed.

---

## GPU FP64 Status (Feb 13, 2026)

Native FP64 GPU compute confirmed on RTX 4070 via `wgpu::Features::SHADER_F64` (Vulkan backend):
- **Precision**: True IEEE 754 double precision (0 ULP error vs CPU f64)
- **Performance**: ~2x FP64:FP32 ratio for bandwidth-limited operations (not the CUDA-reported 1:64)
- **Implication**: The RTX 4070 is usable for FP64 science compute today via BarraCUDA's wgpu shaders
