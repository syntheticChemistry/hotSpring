# hotSpring Deep Debt Resolution + Evolution Sprint — May 13, 2026

**Sprint ID:** GAP-HS-097
**Spring:** hotSpring v0.6.32
**Target:** primalSpring Deep Debt Directive
**Status:** COMPLETE

---

## Sprint Summary

7 work items completed in response to primalSpring's "ecoPrimals — Deep Debt Resolution + Evolution Sprint" directive. hotSpring was already in strong shape (Edition 2024, MSRV 1.87, `#![forbid(unsafe_code)]` on lib, clippy pedantic+nursery, zero TODO/FIXME markers). This sprint addressed remaining code health, dependency compliance, and CI automation gaps.

---

## Work Items Completed

### 1. println/eprintln → log Migration

10 library-core modules migrated from `eprintln!`/`println!` to structured logging:

| Module | Calls | Level |
|--------|-------|-------|
| precision_brain.rs | 3 | info, debug |
| hardware_calibration.rs | 3 | warn, error |
| compute_dispatch.rs | 6 | info, error |
| composition.rs | 6 | info, warn |
| gpu/adapter.rs | 1 | warn |
| low_level/bar0.rs | 2 | warn |
| certification/mod.rs | 2 | warn |
| receipt_signing.rs | 3 | info, warn |
| dag_provenance.rs | 2 | warn |
| data.rs | 1 | warn |
| validation/harness.rs | 1 | warn |

Display/reporting helpers in `bin_helpers/`, `nuclear_eos_helpers/display.rs`, and `bench/report.rs` retain `println!` — these format user-visible output, not diagnostics.

### 2. Hardcoded Lab BDFs Evolved

Three validation scenarios evolved from hardcoded PCI BDFs to env-var-first discovery:

- **`s_compute_trio.rs`**: `HOTSPRING_RTX5060_BDF`, `HOTSPRING_TITAN_V_BDF`, `HOTSPRING_K80_BDF` env vars with lab defaults
- **`s_vfio_dispatch.rs`**: `discover_vfio_targets()` function with env var override per GPU
- **`s_hotqcd_dispatch.rs`**: `qcd_shader_dir()` and `routing_shader_dir()` resolve from `CARGO_MANIFEST_DIR` with `HOTSPRING_QCD_SHADER_DIR` / `HOTSPRING_ROUTING_SHADER_DIR` overrides

### 3. blake3 Pure Rust

`blake3 = { version = "1", default-features = false }` drops the `cc` C build dependency. Performance trade-off is acceptable for hash verification use. Default build now has **zero C dependencies**.

### 4. Boot Scripts Migrated

| Action | Files |
|--------|-------|
| **Created** | `toadstool-ember.service`, `toadstool-glowplug.service` |
| **Updated** | `k80-wake-and-run.sh` (TOADSTOOL_* vars, toadstoolctl, toadstool-ember/glowplug services) |
| **Archived** | 9 coral-era scripts → `scripts/archive/` |

### 5. unwrap() Evolution in Binary Targets

Top 10 most concerning binaries evolved (28 `unwrap()` calls total):

| Binary | Count | Pattern |
|--------|-------|---------|
| validate_gpu_gradient_flow | 8 | `let Some(..) = .. else { check_bool(false); return }` |
| validate_precision_matrix | 4 | `.expect("context")` for infallible byte slices |
| validate_gradient_flow | 4 | `.map_or(f64::NAN, ..)` |
| gpu_physics_proxy | 3 | `if let Ok(json) = ..` |
| validate_sovereign_roundtrip | 3 | `.expect("context")` |
| meta_table_scan | 2 | `.expect("mode invariant")` |
| validate_multi_observable_npu | 2 | `.is_ok_and(..)` |
| validate_silicon_science | 2 | `.expect("map async")` |
| compare_flow_integrators | 2 | `.map_or(f64::NAN, ..)` |

### 6. CI Gate

Created `.github/workflows/ci.yml`:
- `cargo check --lib --tests`
- `cargo clippy --lib --tests -- -D warnings`
- `cargo test --lib`
- `cargo fmt --all -- --check`
- Rust stable toolchain, rust-cache for CI speed

### 7. Dependency Audit

Created `docs/DEPENDENCY_AUDIT.md`:

| Category | Status |
|----------|--------|
| Default build C deps | **Zero** (blake3 pure-Rust) |
| wgpu/tokio | Ecosystem boundaries (documented) |
| cudarc | Feature-gated (`cuda-validation`) |
| rustix | Feature-gated (`low-level`) |
| `#![forbid(unsafe_code)]` | Library + uniBin |

---

## Audit Answers (for upstream handoff)

### Python baselines for barraCuda CPU parity

Extensive coverage in `control/`:
- SEMF/binding energy, lattice QCD (quenched + dynamical), screened Coulomb (SciPy eigensolver)
- Transport (Sarkas/DSF/Yukawa), gradient flow (Chuna 43), BGK dielectric (Chuna 44)
- Kinetic fluid (Chuna 45), spectral theory (Anderson/Hofstadter)
- Reservoir transport, surrogate EOS, TTM, Abelian Higgs
- **Gaps**: HotQCD uses published tables only (no simulation control). Sulfolobus is wetSpring-owned. Large 3D Anderson/Hofstadter may be Rust-first.

### Industry benchmarks for barraCuda GPU parity

- **Kokkos/LAMMPS**: Wired in `benchmarks/kokkos-lammps/` (9 cases, validation script)
- **SciPy**: First-class reference for Coulomb eigensolver, optimizers, linalg, special functions
- **Galaxy/OpenMM/GROMACS**: Explicitly not targeted (different physics domains)
- **Gap**: No formal DF64 NVK GPU parity benchmark yet; tensor-core GEMM routing baseline needed when coralReef HMMA codegen is ready

### Not implemented / verified / validated / tested

- NVIDIA sovereign VFIO dispatch (FECS-gated, `NvVfioComputeDevice` returns `Unsupported`)
- DF64 NVK hardware validation
- Tensor-core GEMM routing
- Multi-GPU OOM recovery in fleet router
- `sovereign-dispatch` feature is inert (awaits toadStool Phase D NVIDIA path)

### Papers unreviewed

- `specs/PAPER_REVIEW_QUEUE.md` is authoritative
- HotQCD (7) is published-tables-only
- Sulfolobus (23) is wetSpring
- All 15+ Chuna-family papers have active control scripts
- Papers 25–31 (Folding@home, SETI@home, BOINC), 32–42 (Tier 4 warm-dense-matter / NIF roadmap), B9 (DFE evolution LTEE) remain queued

### Datasets to examine

| Dataset | Status |
|---------|--------|
| AME2020 (nuclear binding masses) | In-tree |
| HotQCD EOS tables | In-tree |
| Dense Plasma Properties Database | Off-repo download pending |
| Zenodo surrogate archive (10.5281/zenodo.10908462) | Pinned |
| Militzer FPEOS corpus | Partial |
| atoMEC | Partial (7/9) |

---

## Validation

- **591/591** library tests pass
- **Zero** clippy warnings
- **Zero** C dependencies in default build
- CI workflow active
