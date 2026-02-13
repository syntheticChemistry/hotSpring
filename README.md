# hotSpring

**Computational physics reproduction studies and control experiments.**

Named for the hot springs that gave us *Thermus aquaticus* and Taq polymerase — the origin story of the constrained evolution thesis. Professor Murillo's research domain is hot dense plasmas. A spring is a wellspring. This project draws from both.

---

## What This Is

hotSpring is where we reproduce published computational physics work from the Murillo Group (MSU) and benchmark it across consumer hardware. Every study has two phases:

- **Phase A (Control)**: Run the original Python code (Sarkas, mystic, TTM) on our hardware. Validate against reference data. Profile performance. Fix upstream bugs. **✅ Complete — 86/86 quantitative checks pass.**

- **Phase B (BarraCUDA)**: Re-execute the same computation on ToadStool's BarraCUDA engine — pure Rust, WGSL shaders, any GPU vendor. **✅ L1 validated (478× faster, better χ²). L2 validated (1.7× faster).**

hotSpring answers: *"Does our hardware produce correct physics?"* and *"Can Rust+WGSL replace the Python scientific stack?"*

> **For the physics**: See [`PHYSICS.md`](PHYSICS.md) for complete equation documentation
> with numbered references — every formula, every constant, every approximation.
>
> **For the methodology**: See [`whitePaper/METHODOLOGY.md`](whitePaper/METHODOLOGY.md)
> for the two-phase validation protocol and acceptance criteria.

---

## Current Status (2026-02-11)

| Study | Status | Quantitative Checks |
|-------|--------|-------------------|
| **Sarkas MD** (12 cases) | ✅ Complete | 60/60 pass (DSF, RDF, SSF, VACF, Energy) |
| **TTM Local** (3 species) | ✅ Complete | 3/3 pass (Te-Ti equilibrium) |
| **TTM Hydro** (3 species) | ✅ Complete | 3/3 pass (radial profiles) |
| **Surrogate Learning** (9 functions) | ✅ Complete | 15/15 pass + iterative workflow |
| **Nuclear EOS L1** (Python, SEMF) | ✅ Complete | χ²/datum = 6.62 |
| **Nuclear EOS L2** (Python, HFB hybrid) | ✅ Complete | χ²/datum = 1.93 |
| **BarraCUDA L1** (Rust+WGSL, f64) | ✅ Complete | χ²/datum = **2.27** (478× faster) |
| **BarraCUDA L2** (Rust+WGSL+nalgebra) | ✅ Complete | χ²/datum = 25.43 (1.7× faster) |
| **TOTAL** | **86/86 checks pass** | 5 upstream bugs found and fixed |

See `CONTROL_EXPERIMENT_STATUS.md` for full details.

### Nuclear EOS Head-to-Head: BarraCUDA vs Python

| Metric | Python L1 | BarraCUDA L1 | Python L2 | BarraCUDA L2 |
|--------|-----------|-------------|-----------|-------------|
| Best χ²/datum | 6.62 | **2.27** ✅ | **1.93** | 25.43 |
| Total evals | 1,008 | 6,028 | 3,008 | 1,009 |
| Total time | 184s | **2.3s** | 3.2h | 2091s |
| Throughput | 5.5 evals/s | **2,621 evals/s** | 0.28 evals/s | 0.48 evals/s |
| Speedup | — | **478×** | — | **1.7×** |

**L1 takeaway**: BarraCUDA (Rust + WGSL) beats the Python/PyTorch control on both accuracy *and* throughput. Latin Hypercube Sampling + multi-start Nelder-Mead + f64 dual-precision on consumer hardware.

**L2 takeaway**: The throughput advantage (1.7×) is real but the accuracy gap traces to sampling strategy (Python uses mystic `SparsitySampler`). SparsitySampler port is the #1 priority for L2 parity.

---

## Quick Start

```bash
# Full regeneration — clones repos, downloads data, sets up envs, runs everything
# (~12 hours, ~30 GB disk space, GPU recommended)
bash scripts/regenerate-all.sh

# Or step by step:
bash scripts/regenerate-all.sh --deps-only   # Clone + download + env setup (~10 min)
bash scripts/regenerate-all.sh --sarkas      # Sarkas MD: 12 DSF cases (~3 hours)
bash scripts/regenerate-all.sh --surrogate   # Surrogate learning (~5.5 hours)
bash scripts/regenerate-all.sh --nuclear     # Nuclear EOS L1+L2 (~3.5 hours)
bash scripts/regenerate-all.sh --ttm         # TTM models (~1 hour)
bash scripts/regenerate-all.sh --dry-run     # See what would be done

# Or manually:
bash scripts/clone-repos.sh       # Clone + patch upstream repos
bash scripts/download-data.sh     # Download Zenodo archive (~6 GB)
bash scripts/setup-envs.sh        # Create Python environments
```

### What gets regenerated

All large data (21+ GB) is gitignored but fully reproducible:

| Data | Size | Script | Time |
|------|------|--------|------|
| Upstream repos (Sarkas, TTM, Plasma DB) | ~500 MB | `clone-repos.sh` | 2 min |
| Zenodo archive (surrogate learning) | ~6 GB | `download-data.sh` | 5 min |
| Sarkas simulations (12 DSF cases) | ~15 GB | `regenerate-all.sh --sarkas` | 3 hr |
| TTM reproduction (3 species) | ~50 MB | `regenerate-all.sh --ttm` | 1 hr |
| **Total regeneratable** | **~22 GB** | `regenerate-all.sh` | **~12 hr** |

Upstream repos are pinned to specific versions and automatically patched:
- **Sarkas**: v1.0.0 + 3 patches (NumPy 2.x, pandas 2.x, Numba 0.60 compat)
- **TTM**: latest + 1 patch (NumPy 2.x `np.math.factorial` removal)

---

## Directory Structure

```
hotSpring/
├── README.md                           # This file
├── CONTROL_EXPERIMENT_STATUS.md        # Comprehensive status + results (86/86)
├── NUCLEAR_EOS_STRATEGY.md             # Nuclear EOS Phase A→B strategy
├── .gitignore
│
├── control/
│   ├── comprehensive_control_results.json  # Grand total: 86/86 checks
│   │
│   ├── akida_dw_edma/                 # Akida NPU kernel module (patched for 6.17)
│   │   ├── Makefile
│   │   ├── README.md
│   │   ├── akida-pcie-core.c          # PCIe driver source
│   │   └── akida-dw-edma/             # DMA engine sources
│   │
│   ├── sarkas/                         # Study 1: Molecular Dynamics
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── patches/                    # Patches for Sarkas v1.0.0 compat
│   │   │   └── sarkas-v1.0.0-compat.patch
│   │   ├── sarkas-upstream/            # Cloned + patched via scripts/clone-repos.sh
│   │   └── simulations/
│   │       └── dsf-study/
│   │           ├── input_files/        # YAML configs (12 cases)
│   │           ├── scripts/            # run, validate, batch, profile
│   │           └── results/            # Validation JSONs + plots
│   │
│   ├── surrogate/                      # Study 2: Surrogate Learning
│   │   ├── README.md
│   │   ├── REPRODUCE.md               # Step-by-step reproduction guide
│   │   ├── requirements.txt
│   │   ├── scripts/                    # Benchmark + iterative workflow runners
│   │   ├── results/                    # Result JSONs
│   │   └── nuclear-eos/               # Nuclear EOS (L1 + L2)
│   │       ├── README.md
│   │       ├── exp_data/              # AME2020 experimental binding energies
│   │       ├── scripts/               # run_surrogate.py, gpu_rbf.py
│   │       ├── wrapper/               # objective.py, skyrme_hf.py, skyrme_hfb.py
│   │       └── results/               # L1, L2, BarraCUDA JSON results
│   │
│   └── ttm/                            # Study 3: Two-Temperature Model
│       ├── README.md
│       ├── patches/                    # Patches for TTM NumPy 2.x compat
│       │   └── ttm-numpy2-compat.patch
│       ├── Two-Temperature-Model/      # Cloned + patched via scripts/clone-repos.sh
│       └── scripts/                    # Local + hydro model runners
│
├── benchmarks/
│   ├── PROTOCOL.md                     # Cross-gate benchmark protocol
│   ├── sarkas-cpu/                     # Sarkas CPU comparison notes
│   ├── sarkas-gpu/                     # Future: BarraCUDA vs Python
│   ├── surrogate-gpu/                  # Surrogate training GPU benchmarks
│   └── cross-substrate/               # Same workload, all hardware
│
├── data/
│   ├── plasma-properties-db/           # Dense Plasma Properties Database — clone via scripts/
│   ├── zenodo-surrogate/               # Zenodo archive — download via scripts/
│   └── ttm-reference/                  # TTM reference data
│
├── scripts/
│   ├── regenerate-all.sh               # Master: full data regeneration on fresh clone
│   ├── clone-repos.sh                  # Clone + pin + patch upstream repos
│   ├── download-data.sh               # Download Zenodo data (~6 GB)
│   └── setup-envs.sh                   # Create Python envs (conda/micromamba)
│
└── envs/
    ├── sarkas.yaml                     # Sarkas env spec (Python 3.9)
    ├── surrogate.yaml                  # Surrogate env spec (Python 3.10)
    └── ttm.yaml                        # TTM env spec (Python 3.10)
```

---

## Studies

### Study 1: Sarkas Molecular Dynamics

Reproduce plasma simulations from the Dense Plasma Properties Database. 12 cases: 9 Yukawa PP (κ=1,2,3 × Γ=low,mid,high) + 3 Coulomb PPPM (κ=0 × Γ=10,50,150).

- **Source**: [github.com/murillo-group/sarkas](https://github.com/murillo-group/sarkas) (MIT)
- **Reference**: [Dense Plasma Properties Database](https://github.com/MurilloGroupMSU/Dense-Plasma-Properties-Database)
- **Result**: 60/60 observable checks pass (DSF 8.5% mean error PP, 7.3% PPPM)
- **Finding**: `force_pp.update()` is 97.2% of runtime → primary GPU offload target
- **Bugs fixed**: 3 (NumPy 2.x `np.int`, pandas 2.x `.mean(level=)`, Numba/pyfftw PPPM)

### Study 2: Surrogate Learning (Nature MI 2024)

Reproduce "Efficient learning of accurate surrogates for simulations of complex systems" (Diaw et al., 2024).

- **Paper**: [doi.org/10.1038/s42256-024-00839-1](https://doi.org/10.1038/s42256-024-00839-1)
- **Data**: [Zenodo: 10.5281/zenodo.10908462](https://doi.org/10.5281/zenodo.10908462) (open, 6 GB)
- **Code**: [Code Ocean: 10.24433/CO.1152070.v1](https://doi.org/10.24433/CO.1152070.v1) — gated, sign-up denied
- **Result**: 9/9 benchmark functions reproduced. Physics EOS from MD data converged (χ²=4.6×10⁻⁵).

#### Nuclear EOS Surrogate (L1 + L2)

Built from first principles — no HFBTHO, no Code Ocean. Pure Python physics:

| Level | Method | Python χ²/datum | BarraCUDA χ²/datum | Speedup |
|-------|--------|-----------------|--------------------|---------|
| 1 | SEMF + nuclear matter (52 nuclei) | 6.62 | **2.27** ✅ | **478×** |
| 2 | HF+BCS hybrid (18 focused nuclei) | **1.93** | 25.43 | 1.7× |
| 3 | Axially deformed HFB (target) | — | — | — |

- **L1**: Skyrme EDF → nuclear matter properties → SEMF → χ²(AME2020)
- **L2**: Spherical HF+BCS solver for 56≤A≤132, SEMF elsewhere, 18 focused nuclei
- **BarraCUDA**: Full Rust port with WGSL cdist, f64 LA, LHS, multi-start Nelder-Mead

### Study 3: Two-Temperature Model

Run the UCLA-MSU TTM for laser-plasma equilibration in cylindrical coordinates.

- **Source**: [github.com/MurilloGroupMSU/Two-Temperature-Model](https://github.com/MurilloGroupMSU/Two-Temperature-Model)
- **Result**: 6/6 checks pass (3 local + 3 hydro). All species reach physical equilibrium.
- **Bug fixed**: 1 (Thomas-Fermi ionization model sets χ₁=NaN, must use Saha input data)

---

## Upstream Bugs Found and Fixed

| # | Bug | Where | Impact |
|---|-----|-------|--------|
| 1 | `np.int` removed in NumPy 2.x | `sarkas/tools/observables.py` | Silent DSF/SSF failure |
| 2 | `.mean(level=)` removed in pandas 2.x | `sarkas/tools/observables.py` | Silent DSF failure |
| 3 | Numba 0.60 `@jit` → `nopython=True` breaks pyfftw | `sarkas/potentials/force_pm.py` | PPPM method crashes |
| 4 | Thomas-Fermi `χ₁=NaN` poisons recombination | TTM `exp_setup.py` | Zbar solver diverges |
| 5 | DSF reference file naming (case sensitivity) | Plasma Properties DB | Validation script fails |

These are **silent failures** — wrong results, no error messages. This fragility is a core finding.

---

## Hardware

- **Eastgate (primary dev)**: i9-12900K, RTX 4070, Akida AKD1000 NPU. All development and validation.
- **Titan V ×2 (on order)**: GV100, 12GB HBM2, 6.9 TFLOPS FP64 each. Native f64 GPU compute.
- **Strandgate**: 64-core EPYC, 32 GB. Full-scale DSF (N=10,000) runs.
- **Northgate**: i9-14900K. Single-thread comparison.
- **Southgate**: 5800X3D. V-Cache neighbor list performance.

---

## Related

- `CONTROL_EXPERIMENT_STATUS.md` — Full status with numbers, 86/86 checks
- `NUCLEAR_EOS_STRATEGY.md` — Strategic plan: Python control → BarraCUDA proof
- `PHYSICS.md` — Complete physics model documentation with equations and references
- BarraCUDA L1/L2 source: `barracuda/src/bin/nuclear_eos_l{1,2}_ref.rs`
- Handoff: `wateringHole/handoffs/BARRACUDA_SCIENTIFIC_COMPUTING_MIDDLEWARE_FEB11_2026.md`

---

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0).
See [LICENSE](LICENSE) for the full text.

Sovereign science: all source code, data processing scripts, and validation results are
freely available for inspection, reproduction, and extension. If you use this work in
a network service, you must make your source available under the same terms.
