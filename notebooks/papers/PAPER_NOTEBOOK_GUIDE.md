# Paper Baseline Notebooks — Collaborator Guide

Publishable-grade Jupyter notebooks reproducing peer-reviewed physics
on consumer hardware. Each notebook is a self-contained entry point
for a published paper — run it, see the physics, compare to Rust.

## For Collaborators

You don't need primals, NUCLEUS, or GPU hardware to contribute.
Each notebook runs on plain Python (numpy + scipy + matplotlib).

**To add a new paper:**
1. Create `NN-short-name.ipynb` in this directory
2. Follow the cell structure below
3. Put any heavy pre-computed data in `experiments/results/papers/`
4. Submit a PR — the notebook must execute cleanly via `jupyter nbconvert --execute`

## Cell Structure

Every paper notebook follows this pattern:

1. **Title cell** (markdown): Paper citation with DOI, one-paragraph summary of
   what we reproduce, data sources, adaptation note for other springs
2. **Physics context** (markdown): Key equations in LaTeX, physical setup,
   why this paper matters
3. **Imports + constants** (code): numpy, scipy, matplotlib — physical
   constants matching the paper and matching `barracuda/src/` Rust constants
4. **Core computation** (code): The physics algorithm. Live compute for
   problems that finish in <60s. Load frozen JSON for expensive runs.
5. **Visualization** (code): matplotlib charts. Compare Python results to
   published values. Color palette: `#2ecc71` (pass), `#e74c3c` (fail),
   `#3498db` (info), `#9b59b6` (GPU), `#f39c12` (Python), `#1abc9c` (Rust)
6. **Rust parity** (markdown + optional code): Show Rust results alongside
   Python, note speedup, link to the `validate_*` binary
7. **Provenance summary** (markdown): Full citation, DOI link, ecoPrimal
   validation path, "next: primal composition" note

## Data Loading Pattern

```python
import json
from pathlib import Path

RESULTS = Path('..') / '..' / 'experiments' / 'results' / 'papers'

def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)
```

## Visualization Standards

- `matplotlib` only (renders to static PNG, works everywhere)
- Save to `/tmp/hotspring_paper_<notebook>_<chart>.png`
- Always include chart titles with key numbers
- LaTeX labels for axes where appropriate (`$\\beta$`, `$\\chi^2$`)
- Reference lines for published values in dashed gray

## Live vs Frozen

**Live compute** — run the physics in the notebook:
- Eigenvalue problems (scipy, <5s)
- SEMF binding energies (numpy, <1s)
- Small lattice HMC (4^4, <30s for short runs)
- ODE integration (scipy, <1s)
- Analytical dielectric functions (<1s)

**Frozen results** — load from `experiments/results/papers/*.json`:
- HFB self-consistent field (minutes per nucleus)
- Production lattice QCD (hours)
- Full MD simulations (minutes)
- Parameter sweep grids (hours)

## Dependencies

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
```

No GPU, no Rust, no primals required. That's the point.

## Evolution Path

These notebooks are Tier 1 (Python baselines). The evolution path:

1. **Tier 1**: Python notebook — anyone can run it, see the physics
2. **Tier 2**: Rust validation — `cargo test --lib` proves Rust matches Python
3. **Tier 3**: NUCLEUS composition — primals prove IPC matches direct Rust
4. **Tier 4**: JupyterHub compute — notebooks dispatch to GPU via primal composition

When a collaborator (like Chuna) wants to submit cluster-scale work,
they extend from Tier 1 — their expertise is the physics, not the
infrastructure. The infrastructure evolves separately.
