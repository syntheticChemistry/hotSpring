# Baseline Provenance Catalog

**Version:** v0.6.32 | **Last audited:** 2026-05-15

Every validation binary traces its expected values to one of four source types:

| Source Type | Code Reference | Description |
|-------------|---------------|-------------|
| **Python** | `provenance::BaselineProvenance` | Values from `control/` Python scripts with (script, commit, date, command) tuples |
| **Literature** | `provenance::*_REFS` / `provenance::*_DOI` | Published data tables, analytical formulas (e.g. AME2020, HotQCD) |
| **Analytical** | `provenance::MD_FORCE_REFS` etc. | Exact mathematical identities (e.g. cold plaquette = 1.0, Gamma(n) = (n-1)!) |
| **GPU Parity** | `provenance::GPU_KERNEL_REFS` | CPU f64 reference computed in the same binary; GPU output compared within tolerance |

## Provenance Records (machine-readable, in `barracuda/src/provenance/`)

| Record Constant | Module | Script | Commit | Date |
|-----------------|--------|--------|--------|------|
| `L1_PYTHON_CHI2` | `eos` | `surrogate/nuclear-eos/wrapper/objective.py` | `fd908c41` | 2026-01-15 |
| `L1_PYTHON_CANDIDATES` | `eos` | `surrogate/nuclear-eos/wrapper/objective.py` | `fd908c41` | 2026-01-15 |
| `L2_PYTHON_CHI2` | `eos` | `surrogate/nuclear-eos/wrapper/objective.py` | `fd908c41` | 2026-01-15 |
| `L2_PYTHON_CANDIDATES` | `eos` | `surrogate/nuclear-eos/wrapper/objective.py` | `fd908c41` | 2026-01-15 |
| `L2_PYTHON_TOTAL_CHI2` | `eos` | `surrogate/nuclear-eos/wrapper/objective.py` | `fd908c41` | 2026-01-15 |
| `QUENCHED_BETA_SCAN_PROVENANCE` | `lattice` | `lattice_qcd/scripts/quenched_beta_scan.py` | `e047444` | 2026-02-22 |
| `ABELIAN_HIGGS_PYTHON_TIMING_MS` | `lattice` | `abelian_higgs/scripts/abelian_higgs_hmc.py` | `3f0d36d` | 2026-02-22 |
| `HOTQCD_EOS_PROVENANCE` | `lattice` | N/A (published data) | N/A | 2014-11-01 |
| `SCREENED_COULOMB_PROVENANCE` | `lattice` | `screened_coulomb/scripts/yukawa_eigenvalues.py` | `3f0d36d` | 2026-02-19 |
| `HFB_TEST_NUCLEI_PROVENANCE` | `hfb` | `surrogate/nuclear-eos/wrapper/skyrme_hf.py` | `fd908c41` | 2026-01-15 |
| `DALIGAULT_FIT_PROVENANCE` | `transport` | `sarkas/.../scripts/daligault_fit.py` | `0a6405f` | 2026-02-19 |
| `DALIGAULT_CALIBRATION_PROVENANCE` | `transport` | `sarkas/.../scripts/calibrate_daligault_fit.py` | `0a6405f` | 2026-02-19 |
| `TRANSPORT_MD_BASELINE_PROVENANCE` | `transport` | `sarkas/.../scripts/yukawa_md_baseline.py` | `381fdb64` | 2026-02-19 |
| `TTM_ARGON_EQUILIBRIUM_K` | `transport` | `ttm/scripts/run_local_model.py` | `03c0403` | 2026-02-26 |
| `TTM_XENON_EQUILIBRIUM_K` | `transport` | `ttm/scripts/run_local_model.py` | `03c0403` | 2026-02-26 |
| `TTM_HELIUM_EQUILIBRIUM_K` | `transport` | `ttm/scripts/run_local_model.py` | `03c0403` | 2026-02-26 |

## Validation Binary → Baseline Source Map

### Python-sourced baselines

| Binary | Physics Domain | Python Script | Provenance Record |
|--------|---------------|---------------|-------------------|
| `validate_nuclear_eos` | Nuclear EOS (L1/L2) | `objective.py`, `skyrme_hf.py` | `L1_PYTHON_*`, `L2_PYTHON_*`, `HFB_TEST_NUCLEI_PROVENANCE` |
| `validate_barracuda_evolution` | Cross-domain CPU/GPU | Multiple | `QUENCHED_BETA_SCAN_PROVENANCE`, `ABELIAN_HIGGS_PYTHON_TIMING_MS` |
| `validate_production_qcd` | Quenched SU(3) | `quenched_beta_scan.py` | `QUENCHED_BETA_SCAN_PROVENANCE` |
| `validate_abelian_higgs` | U(1)+scalar HMC | `abelian_higgs_hmc.py` | `ABELIAN_HIGGS_PYTHON_TIMING_MS` |
| `validate_screened_coulomb` | Yukawa bound states | `yukawa_eigenvalues.py` | `SCREENED_COULOMB_PROVENANCE` |
| `validate_dielectric` | BGK dielectric | `dielectric_bgk.py` | `BaselineProvenance` in binary |
| `validate_fpeos` | Militzer FPEOS | `militzer_pre2021_table.py` | `BaselineProvenance` in binary |
| `validate_transport` | Sarkas MD transport | `yukawa_md_baseline.py`, `daligault_fit.py` | `TRANSPORT_MD_BASELINE_PROVENANCE`, `DALIGAULT_*` |
| `validate_stanton_murillo` | Ionic transport | `daligault_fit.py` | `DALIGAULT_FIT_PROVENANCE` |
| `validate_transport_gpu_resident` | GPU VACF transport | `yukawa_md_baseline.py` | `TRANSPORT_MD_BASELINE_PROVENANCE` |
| `validate_reservoir_transport` | ESN transport | `reservoir_vacf.py` | Cited in binary docstring |
| `validate_ttm` | Two-temperature model | `run_local_model.py` | `TTM_*_EQUILIBRIUM_K` |
| `validate_hotqcd_eos` | HotQCD EOS tables | Published data (PRD 90) | `HOTQCD_EOS_PROVENANCE` |
| `validate_lattice_npu` | Lattice+NPU | `npu_lattice_phase.py` | Cited in binary docstring |
| `validate_npu_beyond_sdk` | NPU beyond-SDK | `npu_beyond_sdk.py` | Cited in binary docstring |
| `validate_npu_pipeline` | NPU physics | `npu_physics_pipeline.py` | Cited in binary docstring |
| `validate_npu_quantization` | NPU quantization | `npu_quantization_parity.py` | Cited in binary docstring |
| `validate_primal_proof` | Primal IPC proof | Inline Python baselines | In binary |

### Analytical/literature baselines (no Python control)

| Binary | Physics Domain | Reference Source | Provenance Record |
|--------|---------------|-----------------|-------------------|
| `validate_special_functions` | Gamma, erf, Bessel | A&S 1964, NIST DLMF | `SPECIAL_FUNCTION_REFS` |
| `validate_linalg` | Dense linear algebra | Mathematical identities | `LINALG_REFS` |
| `validate_optimizers` | NM, BFGS, RK45 | scipy.optimize parity | `OPTIMIZER_REFS` |
| `validate_md` | LJ/Coulomb/Morse forces | Allen & Tildesley 1987 | `MD_FORCE_REFS` |
| `validate_pure_gauge` | SU(3) pure gauge | Creutz, Wilson, Bali | `PURE_GAUGE_REFS` |
| `validate_gradient_flow` | Gradient flow CPU | Analytical convergence | Documented in binary |
| `validate_anderson_3d` | 3D Anderson MIT | Kachkovskiy theory | `BaselineProvenance` (literature) |
| `validate_lanczos` | Lanczos eigensolve | Mathematical identities | `tolerances::lattice` |
| `validate_spectral` | Spectral theory | Kachkovskiy/Wigner-Dyson | `tolerances::lattice` |
| `validate_hofstadter` | Hofstadter butterfly | Symmetry identities | `tolerances::lattice` |

### GPU parity baselines (CPU f64 reference computed in same binary)

| Binary | Physics Domain | Reference | Provenance Record |
|--------|---------------|-----------|-------------------|
| `validate_barracuda_hfb` | BCS/HFB GPU pipeline | CPU f64 | `GPU_KERNEL_REFS` |
| `validate_barracuda_pipeline` | MD/Yukawa GPU | CPU f64 | `GPU_KERNEL_REFS` |
| `validate_cpu_gpu_parity` | Cross-domain | CPU f64 | `tolerances::core::GPU_VS_CPU_F64` |
| `validate_gpu_beta_scan` | GPU beta scan | CPU f64 | `tolerances::lattice` |
| `validate_gpu_cg` | GPU CG solver | CPU f64 | `tolerances::lattice` |
| `validate_gpu_dielectric` | GPU dielectric | CPU f64 | `tolerances::physics` |
| `validate_gpu_dirac` | GPU staggered Dirac | CPU f64 | `tolerances::lattice` |
| `validate_gpu_gradient_flow` | GPU gradient flow | CPU f64 | `tolerances::lattice` |
| `validate_gpu_lanczos` | GPU Lanczos | CPU f64 | `tolerances::lattice` |
| `validate_gpu_spmv` | GPU SpMV | CPU f64 | `tolerances::lattice` |
| `validate_gpu_streaming` | GPU streaming HMC | CPU f64 | `tolerances::lattice` |
| `validate_gpu_streaming_dyn` | GPU dynamical HMC | CPU f64 | `tolerances::lattice` |
| `validate_gpu_dynamical_hmc` | GPU dynamical HMC | CPU f64 | `tolerances::lattice` |
| `validate_pure_gpu_hmc` | Pure GPU HMC | CPU f64 | `tolerances::lattice` |
| `validate_pure_gpu_qcd` | Pure GPU QCD | CPU f64 | `tolerances::lattice` |
| `validate_nak_eigensolve` | NAK GPU eigensolve | CPU f64 | `tolerances::lattice::NAK_*` |
| `validate_pppm` | PPPM Ewald | Direct Coulomb sum | `GPU_KERNEL_REFS` |
| `validate_sovereign_roundtrip` | WGSL round-trip | Raw vs sovereign output | `tolerances::core::SOVEREIGN_*` |
| `validate_silicon_capabilities` | GPU reduction tests | CPU f64 | `tolerances::core` |

### IPC/composition/dispatch validators (no physics baselines)

| Binary | Role | Baseline Type |
|--------|------|---------------|
| `validate_all` | Meta-runner | Delegates to all above |
| `validate_compute_dispatch` | Exp 152 dispatch | DAG provenance hash |
| `validate_compute_trio_pipeline` | barraCuda→coralReef→toadStool | IPC liveness + physics result |
| `validate_ember_resilience` | Ember fleet resilience | IPC contract |
| `validate_nucleus_composition` | NUCLEUS Phase 2 | IPC contract + physics parity |
| `validate_nucleus_nest` | Nest atomic | Blake3 provenance |
| `validate_nucleus_node` | Node atomic | IPC contract |
| `validate_nucleus_tower` | Tower atomic | IPC contract |
| `validate_squirrel_roundtrip` | Squirrel inference | IPC contract |
| `validate_sovereign_compile` | Sovereign WGSL compile | naga parse success |
| `validate_cross_vendor_dispatch` | Cross-vendor SAXPY | IPC contract + f32 parity |

## Commit Verification

```
git log --oneline fd908c41 -1   # Nuclear EOS control (pinned)
git log --oneline 0a6405f -1    # Transport study (Paper 5)
git log --oneline 381fdb64 -1   # MD baseline (standalone)
git log --oneline 3f0d36d -1    # Screened Coulomb + Abelian Higgs
git log --oneline e047444 -1    # Quenched beta scan (Paper 9)
git log --oneline 03c0403 -1    # TTM control
```

## External Data Sources

| Dataset | DOI / Accession | Usage |
|---------|----------------|-------|
| AME2020 mass table | [10.1088/1674-1137/abddaf](https://doi.org/10.1088/1674-1137/abddaf) | HFB binding energies |
| SLy4 parametrization | [10.1016/S0375-9474(98)00180-8](https://doi.org/10.1016/S0375-9474(98)00180-8) | Skyrme HF reference |
| UNEDF0 parametrization | [10.1103/PhysRevC.82.024313](https://doi.org/10.1103/PhysRevC.82.024313) | Alternative Skyrme |
| Bender, Heenen, Reinhard | [10.1103/RevModPhys.75.121](https://doi.org/10.1103/RevModPhys.75.121) | E/A NMP target |
| Lattimer & Prakash | [10.1016/j.physrep.2015.12.005](https://doi.org/10.1016/j.physrep.2015.12.005) | Symmetry energy J |
| Sarkas MD software | [10.1016/j.cpc.2021.108245](https://doi.org/10.1016/j.cpc.2021.108245) | Transport control runs |
| Surrogate learning archive | [10.5281/zenodo.10908462](https://doi.org/10.5281/zenodo.10908462) | Nuclear EOS convergence |
| HotQCD EOS | [10.1103/PhysRevD.90.094503](https://doi.org/10.1103/PhysRevD.90.094503) | Lattice QCD thermodynamics |
| Stanton & Murillo transport | [10.1103/PhysRevE.93.043203](https://doi.org/10.1103/PhysRevE.93.043203) | Ionic transport D* |
| Daligault D* model | [10.1103/PhysRevE.86.047401](https://doi.org/10.1103/PhysRevE.86.047401) | Weak-coupling correction |
