# Two-Temperature Model Control Experiment

**Study 3**: Reproduce UCLA-MSU Two-Temperature Model simulations.

---

## Source

[github.com/MurilloGroupMSU/Two-Temperature-Model](https://github.com/MurilloGroupMSU/Two-Temperature-Model) (updated January 2025)

## Setup

```bash
# 1. Clone upstream (if not already done)
cd hotSpring/
bash scripts/clone-repos.sh

# 2. Create environment
bash scripts/setup-envs.sh

# 3. Activate
conda activate ttm
```

## Directory Layout

```
control/ttm/
├── README.md               # This file
├── Two-Temperature-Model/  # Cloned upstream repo
├── scripts/
│   ├── run_local_model.py  # Headless ODE (local) model runner
│   └── run_hydro_model.py  # Headless hydro (spatial) model runner
└── reproduction/
    ├── local-model/        # Local model results (Te/Ti vs time)
    └── hydro-model/        # Hydro model results (Te/Ti vs r,t)
```

## What It Does

Evolves electron and ion temperatures in cylindrical coordinates. Models a plasma formed by a laser passing through a dense gas. SI units throughout.

Key files in upstream:
- `core/physics.py` - Model parameters (Fermi Energy, transport models: JT_GMS, SMT, Fraley)
- `core/constants.py` - SI numerical constants and unit conversions
- `core/local_ODE_solver.py` - ODE solver (spatially uniform, equilibration only)
- `core/Hydro_solver.py` - Full cylindrical hydro solver with diffusion
- `core/exp_setup.py` - Experiment class (grid, initial profiles, ionization)
- `jupyter/` - Notebooks (primary interface)
- `data/` - Reference data from UCLA collaboration

## Workflow

### 1. Local Model (ODE — fast, no spatial resolution)

Run Te/Ti equilibration for Argon, Xenon, Helium:

```bash
conda activate ttm
cd control/ttm/
python scripts/run_local_model.py --species argon,xenon,helium --model SMT
```

Produces CSV time series + PNG evolution plots in `reproduction/local-model/`.

### 2. Hydro Model (full spatial + hydrodynamics)

Run the cylindrical two-temperature model with diffusion and expansion:

```bash
python scripts/run_hydro_model.py --species argon --model SMT
python scripts/run_hydro_model.py --species xenon,helium  # all species
```

Produces radial profiles at multiple time snapshots, center temperature evolution, and FWHM tracking in `reproduction/hydro-model/`.

### 3. Jupyter Exploration

```bash
cd Two-Temperature-Model/
jupyter lab
# Explore: jupyter/Local_model_solver/ and jupyter/Hydro_Runner/
```

### 4. Parameter Sweeps

Run systematic sweeps over initial conditions (density, temperature, ionization model) for cross-gate benchmarking after NUCLEUS stabilizes.
