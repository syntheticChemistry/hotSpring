# Public Notebook Pattern — hotSpring

How to create public-facing notebooks for hotSpring. Adapted from the
primalSpring/wetSpring exemplar pattern for computational physics domains.

## Directory Convention

```
hotSpring/
  notebooks/
    NOTEBOOK_PATTERN.md          <- this file
    01-composition-validation.ipynb   <- NUCLEUS composition + guideStone validation
    02-benchmark-comparison.ipynb     <- Python vs Rust vs GPU performance
    03-experiment-evidence.ipynb      <- 181 experiments, science ladder, paper reproductions
    04-cross-spring-connections.ipynb  <- ecosystem connections, patterns handed back
    05-physics-deep-dive.ipynb        <- QCD, nuclear EOS, sovereign GPU pipeline
```

## Cell Structure

Every notebook follows the same structure:

1. **Title cell** (markdown): Title, one-paragraph context, data sources, "for other springs" adaptation note
2. **Imports + data loading** (code): Load from `../experiments/results/*.json`
3. **Domain-specific cells** (code + markdown): Visualization and analysis
4. **Summary cell** (markdown): Validation table, provenance note, links to primals.eco

## Data Loading Pattern

```python
import json
from pathlib import Path

RESULTS = Path('..') / 'experiments' / 'results'

def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)

data = load('composition_validation.json')
```

Notebooks load **frozen data** (committed JSON artifacts), not live API responses.
This means they work without primals running.

## Frozen Data for hotSpring

| File | Contents |
|------|----------|
| `composition_validation.json` | Deploy graph, atomic types, guideStone bare checks, capability routing |
| `test_suite_report.json` | Module-level test counts (993), timings, physics categories |
| `experiment_catalog.json` | All 181 experiments, 12 categories, science ladder, papers reproduced |
| `benchmark_timing.json` | Rust vs Python, GPU vs CPU, DF64 throughput, energy/cost estimates |
| `cross_spring_matrix.json` | Primal consumption, patterns handed back, ecosystem flows |
| `security_convergence.json` | guideStone Level 5, BTSP posture, gap summary, evolution timeline |

## Visualization Standards

- Use `matplotlib` (available everywhere, renders to static PNG)
- Save figures to `/tmp/hotspring_<notebook>_<chart>.png`
- Color palette: `#2ecc71` (pass/ok), `#e74c3c` (fail), `#3498db` (info)
- Additional domain colors: `#9b59b6` (GPU), `#f39c12` (Python baseline), `#1abc9c` (Rust)
- Always include chart titles with key numbers

## Adapting for Your Spring

1. Copy this directory structure
2. Replace data paths with your `experiments/results/` JSONs
3. Update the narrative for your domain (QCD → your science)
4. Keep the cell structure (title → load → analyze → summary)
5. Add your spring to `shared/abg/commons/<spring>-public/notebooks/` symlink
