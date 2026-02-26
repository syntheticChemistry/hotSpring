# Nuclear EOS: Independent Reproduction (From-Scratch)

**Status**: ✅ Complete — L1 SEMF + L2 spherical HFB + L3 deformed HFB (BarraCUDA GPU)
**Goal**: Build a callable `objective(skyrme_params) → χ²` without Code Ocean

---

## What This Does

The Diaw et al. (2024) paper uses a nuclear Equation of State (EOS) as their
headline application for surrogate learning. The objective function:

1. Takes Skyrme energy density functional (EDF) parameters as input
2. Runs a Hartree-Fock-Bogoliubov (HFB) nuclear structure calculation
3. Compares predicted nuclear masses to experimental data (AME2020)
4. Returns a χ² score

The Code Ocean capsule wraps this calculation but is behind gated access.
We rebuild it from open components.

---

## Components

### 1. From-Scratch HFB (replacing HFBTHO)

HFBTHO (Fortran) was the original target but requires gated CPC Program
Library access. We built the physics from scratch instead:
- **L1**: `skyrme_hf.py` — SEMF + nuclear matter properties (Python)
- **L2**: `skyrme_hfb.py` — Spherical HF+BCS with separate p/n (Python)
- **L3**: BarraCUDA — Axially deformed HFB in WGSL (Rust GPU)
- See `hfbtho/README.md` for the from-scratch evolution path

### 2. Experimental Data (AME2020)
- **What**: Atomic Mass Evaluation 2020 — binding energies of all known nuclei
- **Source**: IAEA Nuclear Data Services (public, no login)
- **URL**: https://www-nds.iaea.org/amdc/
- **Format**: Text tables of (Z, N, binding_energy, uncertainty)

### 3. Python Wrapper
- **What**: `objective.py` — objective function for surrogate learning
- **Interface**: `objective(x) → float` where x = Skyrme parameters
- **Flow**: params → L1/L2 physics → χ² against AME2020

### 4. Skyrme Parameter Bounds
- **What**: Published parameter ranges for known Skyrme parametrizations
- **Reference**: SLy4, UNEDF0, UNEDF1, SkM*, SGII
- **Format**: JSON with bounds per parameter

---

## Directory Structure

```
nuclear-eos/
├── README.md              ← this file
├── hfbtho/                ← From-scratch evolution (README explains why)
│   └── README.md          ← HFBTHO → from-scratch L1/L2/L3 evolution path
├── wrapper/
│   ├── objective.py       ← Python wrapper: params → χ²
│   ├── skyrme_hf.py       ← Level 1: SEMF + nuclear matter
│   ├── skyrme_hfb.py      ← Level 2: Spherical HF+BCS
│   ├── gpu_accel.py       ← GPU acceleration helpers
│   └── skyrme_bounds.json ← Parameter ranges
├── exp_data/
│   ├── download_ame2020.sh ← Download script for AME2020
│   └── ame2020_selected.json ← Experimental binding energies
├── scripts/
│   └── run_surrogate.py   ← Run full iterative workflow on nuclear EOS
└── results/               ← Output from surrogate runs
```

---

## Status

| Component | Status |
|-----------|--------|
| Directory structure | ✅ Ready |
| Skyrme bounds (JSON) | ✅ Ready |
| AME2020 data | ✅ Ready |
| Level 1 (SEMF, Python) | ✅ Complete — χ²/datum = 6.62 |
| Level 2 (HFB, Python) | ✅ Complete — χ²/datum = 1.93 |
| Level 1 (BarraCUDA GPU) | ✅ Complete — χ²/datum = 2.27, 478× faster |
| Level 2 (BarraCUDA GPU) | ✅ Complete — χ²/datum = 16.11 |
| Level 3 (deformed HFB GPU) | ✅ Complete — 295/2036 nuclei improved |
| Full AME2020 (2042 nuclei) | ✅ Complete on RTX 4070 |

---

## Comparison Target

From Zenodo `results/orig/score.txt`:
- 30 rounds, 1000 evals/round, 30,000 total evaluations
- Final χ² = 9.2 × 10⁻⁶
- Monotonic convergence from NaN → 4.1e-3 → ... → 9e-6

Our Physics EOS (from Sarkas MD, quick mode):
- 30 rounds, 100 evals/round, 3,008 total evaluations
- Final χ² = 6.6 × 10⁻⁴
- Monotonic convergence (all rounds trending down)

The nuclear EOS will be a more challenging objective (higher dimension,
more expensive per evaluation), but the methodology is identical.

