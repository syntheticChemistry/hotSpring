# Surrogate Learning Control Experiment

**Study 2**: Reproduce "Efficient learning of accurate surrogates for simulations of complex systems"  
Diaw, McKerns, Sagert, Stanton, Murillo — *Nature Machine Intelligence*, May 2024

---

## Status: ✅ Methodology Reproduced (10/11 functions)

| Category | Status |
|----------|--------|
| Paper methodology (SparsitySampler + RBF) | ✅ Fully reconstructed |
| Benchmark functions (9/9 from paper) | ✅ All reproduced |
| Physics EOS (our addition) | ✅ Converged, χ²=4.6×10⁻⁵ |
| Nuclear EOS (paper's headline) | ❌ Blocked (Code Ocean gated) |
| Reproducibility infrastructure | ✅ Complete |

---

## Resources

| Resource | URL | Status |
|----------|-----|--------|
| **Paper** | [doi.org/10.1038/s42256-024-00839-1](https://doi.org/10.1038/s42256-024-00839-1) | ✅ Open |
| **Datasets** | [Zenodo: 10.5281/zenodo.10908462](https://doi.org/10.5281/zenodo.10908462) | ✅ Open (6 GB) |
| **Code** | [Code Ocean: 10.24433/CO.1152070.v1](https://doi.org/10.24433/CO.1152070.v1) | ❌ **Gated** |

### Code Ocean Access Failure

Code Ocean denies sign-up on some operating systems ("OS is denied"). This means:
- The `_workflow.py` orchestration script is inaccessible
- The nuclear EOS objective function (wrapping LANL simulation data) is inaccessible
- **The paper's headline result cannot be independently reproduced**

We fully reconstructed the methodology from the paper text + Zenodo config files,
and replaced the nuclear EOS with a physics EOS from our validated Sarkas MD data.
See `REPRODUCE.md` for the complete reproduction guide.

---

## Quick Start

```bash
# Full reproduction (see REPRODUCE.md for detailed steps)
pip install -r requirements.txt
python scripts/full_iterative_workflow.py --quick   # ~30 min
python scripts/verify_results.py                     # automated check
```

---

## Directory Layout

```
control/surrogate/
├── README.md                   # This file
├── REPRODUCE.md                # Step-by-step reproduction guide
├── requirements.txt            # Pinned Python dependencies
├── code-ocean-capsule/         # (empty — Code Ocean access denied)
├── scripts/
│   ├── full_iterative_workflow.py  # Main reproduction (9 objectives × 30 rounds)
│   ├── run_benchmark_functions.py  # Quick single-round validation (5 functions)
│   ├── run_reproduction.py         # Original capsule-based runner (unused)
│   └── verify_results.py           # Automated result verification (7 checks)
└── results/
    ├── full_iterative_workflow_results.json    # Iterative workflow results
    ├── benchmark_functions_results.json        # Single-round benchmark results
    ├── extended_benchmark_results.json         # Extended benchmark results
    └── physics_surrogate_LOO.json              # LOO cross-validation on MD data
```

---

## What Each Script Does

| Script | What | Runtime |
|--------|------|---------|
| `full_iterative_workflow.py` | 9 objectives × 30 rounds, matching paper's methodology | 30 min (quick) / 5 hr (full) |
| `run_benchmark_functions.py` | Quick validation: 5 functions, 3 sampling strategies | ~2 min |
| `verify_results.py` | Automated check: 7 assertions on result file | ~1 sec |
| `run_reproduction.py` | Original Code Ocean capsule runner (requires capsule) | N/A (blocked) |

---

## Key Findings

1. **SparsitySampler works**: Optimizer-directed sampling consistently outperforms random/LHS for training surrogates.

2. **Code Ocean is a reproducibility failure**: The paper's headline result (nuclear EOS surrogate) is locked behind a gated platform that restricts access by operating system.

3. **Physics EOS converges fast**: Our 3-parameter EOS from 12 MD simulations converges in 11 rounds with 176 evaluations — significantly faster than the paper's nuclear EOS (30 rounds, 30,000 evaluations).

4. **The methodology is the contribution, not the platform**: The entire algorithm (SparsitySampler → cache → RBF → test → loop) is reconstructible from the paper text alone. Code Ocean adds nothing.
