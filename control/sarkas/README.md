# Sarkas Control Experiment

**Study 1**: Reproduce Sarkas molecular dynamics simulations on the basement HPC.

---

## Setup

```bash
# 1. Clone Sarkas upstream (if not already done)
cd hotSpring/
bash scripts/clone-repos.sh

# 2. Create conda environment
bash scripts/setup-envs.sh

# 3. Activate
conda activate sarkas
```

## Directory Layout

```
control/sarkas/
├── README.md               # This file
├── sarkas-upstream/        # Cloned Sarkas repo (via clone-repos.sh)
├── simulations/            # Our simulation input/output
│   ├── quickstart/         # From Sarkas quickstart tutorial
│   ├── yukawa/             # Yukawa (screened Coulomb) systems
│   ├── coulomb/            # Pure Coulomb systems
│   └── binary-mixtures/    # Binary ionic mixtures
├── validation/             # Comparison against Dense Plasma Properties Database
└── profiling/              # cProfile output, timing data
```

## Workflow

### 1. Quickstart Validation

Run the Sarkas quickstart to confirm installation works:

```bash
conda activate sarkas
cd sarkas-upstream/
jupyter lab
# Run the Quickstart notebook from docs
```

### 2. Tutorial Simulations

Follow the full Sarkas tutorial. Create YAML input files for:
- Yukawa system (screened Coulomb, κ = 1, 2, 4)
- Pure Coulomb system
- Binary ionic mixture

Save inputs and outputs in `simulations/`.

### 3. Validate Against Reference Data

Compare our results to the Dense Plasma Properties Database:
- Yukawa VACF → compare to `Yukawa_Dynamic_Structure_Factors/`
- Yukawa g(r) → compare to `Yukawa_Intermediate_Scattering_Functions/`
- Yukawa susceptibilities → compare to `Yukawa_Susceptibilities/`

Save comparison plots in `validation/`.

### 4. DSF Study (Yukawa Dynamic Structure Factor Database)

Run the full 12-case DSF reproduction against the Dense Plasma Properties Database:

```bash
cd simulations/dsf-study/

# Generate input files (already done, see input_files/MANIFEST.txt)
python scripts/generate_inputs.py

# Run a single case
python scripts/run_case.py input_files/dsf_k1_G14_mks.yaml

# Run all 12 cases in batch
python scripts/run_batch.py                      # all cases
python scripts/run_batch.py --cases k1_G14,k2_G31  # specific cases
python scripts/run_batch.py --skip-completed     # resume interrupted batch
python scripts/run_batch.py --dry-run            # see case list without running

# Validate against reference data
python scripts/validate_dsf.py
```

### 5. Profile

```bash
# Profile a simulation to identify compute hotspots for BarraCUDA Phase B
cd simulations/dsf-study/
python scripts/profile_sarkas.py input_files/dsf_k1_G14_mks.yaml --top 30

# Output: profiling/dsf_k1_G14_mks.prof (snakeviz-compatible)
#         profiling/dsf_k1_G14_mks_summary.txt (MD pipeline breakdown)
```

Categorizes compute time into: force calculation, FFT/PPPM, neighbor list,
integration, I/O, and numba JIT. This data drives Phase B (BarraCUDA) shader prioritization.

## Hardware Notes

- **Eastgate (local dev)**: RTX 4070 + Akida NPU. Good for validation.
- **Strandgate**: 64-core EPYC. Primary benchmark target for CPU-parallel MD.
- **Northgate**: i9-14900K. Single-thread comparison.
- **Southgate**: 5800X3D. V-Cache neighbor list performance.
