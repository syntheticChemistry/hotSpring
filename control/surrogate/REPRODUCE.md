# How to Reproduce the Surrogate Learning Study

**Paper**: Diaw et al. (2024) "Efficient learning of accurate surrogates for
simulations of complex systems" *Nature Machine Intelligence*  
**DOI**: [10.1038/s42256-024-00839-1](https://doi.org/10.1038/s42256-024-00839-1)

**What this reproduces**: The paper's optimizer-directed surrogate methodology
on open benchmark functions + a physics-based EOS derived from validated
molecular dynamics simulations.

**What this cannot reproduce**: The paper's nuclear EOS objective function,
which is locked behind a Code Ocean capsule that denies sign-up (see §4a in
the white paper for full documentation of this access failure).

**Time**: ~30 min (quick mode) or ~5 hours (full mode, 1000 evals/round)  
**Hardware**: Any x86_64 Linux machine with ≥8 GB RAM  
**License**: AGPL-3.0

---

## Prerequisites

- Python 3.10+ (tested on 3.10.19)
- ~7 GB disk space (for Zenodo data)
- Internet access (one-time, to download Zenodo data)

---

## Step 1: Clone the repository

```bash
git clone https://github.com/ecoPrimals/ecoPrimals.git
cd ecoPrimals/hotSpring
```

---

## Step 2: Create the Python environment

### Option A: pip (simplest)

```bash
python3 -m venv .venv-surrogate
source .venv-surrogate/bin/activate
pip install -r control/surrogate/requirements.txt
```

### Option B: conda / micromamba

```bash
micromamba create -n surrogate python=3.10 -y
micromamba activate surrogate
pip install -r control/surrogate/requirements.txt
```

### Verify

```bash
python -c "from mystic.samplers import SparsitySampler; print('mystic OK')"
python -c "from scipy.interpolate import RBFInterpolator; print('scipy OK')"
```

---

## Step 3: Download the Zenodo data (one-time)

```bash
bash scripts/download-data.sh
```

This downloads ~6 GB from Zenodo (DOI: 10.5281/zenodo.10908462) into
`data/zenodo-surrogate/`. No authentication required.

### Verify

```bash
ls data/zenodo-surrogate/results/
# Should list: easom/ hartmann6/ michal/ nick/ orig/ plateau/ rast/ rosen/ rosen8/
```

---

## Step 4: Run the surrogate workflow

### Quick validation (~30 min)

```bash
python control/surrogate/scripts/full_iterative_workflow.py --quick
```

Uses 100 evaluations/round (vs paper's 1000). Good for verifying the
methodology works.

### Full reproduction (~5 hours)

```bash
python control/surrogate/scripts/full_iterative_workflow.py
```

Uses 1000 evaluations/round, matching the paper's `1000SS` configuration.
This produces publication-quality results.

### Results

Output saved to `control/surrogate/results/full_iterative_workflow_results.json`

---

## Step 5: Verify results

```bash
python control/surrogate/scripts/verify_results.py
```

This script checks:
- All functions were tested (≥7 objectives)
- Physics EOS converged (χ² < 1e-4)
- Convergence histories are monotonically improving
- Results file is well-formed and complete

---

## Step 6 (optional): Reproduce the physics EOS data

The physics EOS objective is built from validated Sarkas molecular dynamics
simulations. To reproduce those from scratch:

### 6a. Create Sarkas environment

```bash
# Sarkas requires Python 3.9 + specific NumPy/Numba versions
micromamba create -n sarkas python=3.9 -y
micromamba activate sarkas
pip install -r control/sarkas/requirements.txt

# Install Sarkas from our patched fork
cd control/sarkas/sarkas-upstream
pip install -e .
cd ../../..
```

### 6b. Run the 12 MD simulations

```bash
# Generate lite input files (N=2000 particles, ~15 min each)
python control/sarkas/simulations/dsf-study/scripts/generate_lite_inputs.py

# Run all 9 PP cases
bash control/sarkas/simulations/dsf-study/scripts/batch_run_lite.sh

# Run all 3 PPPM cases
bash control/sarkas/simulations/dsf-study/scripts/batch_run_pppm_lite.sh
```

### 6c. Validate observables

```bash
python control/sarkas/simulations/dsf-study/scripts/validate_all_observables.py
```

This produces `results/all_observables_validation.json`, which is the input
data for the physics EOS objective.

---

## What each script does

| Script | Purpose | Runtime |
|--------|---------|---------|
| `full_iterative_workflow.py` | Main reproduction: 9 objectives × 30 rounds | 30min–5hr |
| `run_benchmark_functions.py` | Quick single-round validation on 5 functions | ~2 min |
| `verify_results.py` | Automated check that results match expectations | ~1 sec |

---

## Expected results

### Quick mode (--quick, 100 evals/round)

| Function | Converges? | Score range |
|----------|:----------:|:-----------:|
| Rastrigin 2D | ⚠️ (slow) | ~2–3 |
| Rosenbrock 2D | ⚠️ (slow) | ~10–15 |
| Easom 2D | ✅ or close | ~2e-4 |
| MultiscaleNDFunc 2D | ⚠️ | ~0.5–1 |
| Hartmann6 | ⚠️ | ~0.02–0.05 |
| Michalewicz 2D | ✅ or close | ~1e-4 |
| Rosenbrock 8D | ⚠️ | ~1e4–1e6 |
| MultiscaleNDFunc 5D | ⚠️ | ~1–2 |
| **Physics EOS** | **✅** | **< 1e-4** |

### Full mode (1000 evals/round)

Scores should improve significantly. The paper's N16 configs achieve:
- Rastrigin: χ²=0 (exact)
- Rosenbrock 2D: χ²=1e-6
- Easom: χ²=1e-6
- MultiscaleNDFunc: χ²=1.3e-4
- Hartmann6: χ²=7.3e-5
- Rosenbrock 8D: χ²=1.5e-4

---

## Comparison to Code Ocean

| Aspect | Code Ocean capsule | This repository |
|--------|-------------------|-----------------|
| Login required | Yes (denied on some OS) | No |
| Nuclear EOS available | Yes (gated) | No (replaced with open physics EOS) |
| Methodology reproducible | Unknown (can't access) | Yes (fully open) |
| Dependencies pinned | Unknown | Yes (`requirements.txt`) |
| Results inspectable | Unknown | Yes (JSON, logged) |
| License | Unknown | AGPL-3.0 |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'mystic'"
→ Install requirements: `pip install -r control/surrogate/requirements.txt`

### "FileNotFoundError: perlin_nick.py"
→ Download Zenodo data: `bash scripts/download-data.sh`

### "MemoryError during RBF training"
→ Use `--quick` mode or reduce `max_rounds` in the script

### Results differ slightly from saved JSON
→ Expected. Random seed is pinned (`np.random.seed(42)`) but
`SparsitySampler` uses internal randomness that may vary across
mystic versions. Check that convergence trends match, not exact values.

---

## Citation

If you use this reproduction in your work:

```
@misc{ecoprimalsSurrogateRepro2026,
  title={Open Reproduction of Optimizer-Directed Surrogate Learning},
  author={ecoPrimals Team},
  year={2026},
  howpublished={\url{https://github.com/ecoPrimals/ecoPrimals}},
  note={Reproduces Diaw et al. (2024) without Code Ocean access}
}
```

Original paper:
```
@article{diaw2024efficient,
  title={Efficient learning of accurate surrogates for simulations of complex systems},
  author={Diaw, A. and McKerns, M. and Sagert, I. and Stanton, L. and Murillo, M.S.},
  journal={Nature Machine Intelligence},
  year={2024},
  doi={10.1038/s42256-024-00839-1}
}
```

