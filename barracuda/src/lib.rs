//! hotSpring Nuclear EOS — BarraCUDA L1/L2/L3 validation environment
//!
//! Validates BarraCUDA library against Python/scipy controls using nuclear
//! equation-of-state workloads (Skyrme energy density functional).
//!
//! ## Active modules
//!   - `data` — AME2020 experimental data and Skyrme parameter bounds
//!   - `physics` — SEMF, nuclear matter properties, spherical HFB, deformed HFB
//!   - `gpu` — GPU FP64 device wrapper (SHADER_F64 via wgpu/Vulkan)
//!   - `prescreen` — NMP cascade filter for L2 heterogeneous pipeline
//!
//! ## Validation binaries
//!   - `nuclear_eos_l1_ref` — L1 (SEMF) with NMP-constrained objective, Pareto sweep
//!   - `nuclear_eos_l2_ref` — L2 (HFB) with L1-seeded DirectSampler
//!   - `nuclear_eos_l2_hetero` — L2 heterogeneous pipeline with cascade
//!   - `nuclear_eos_gpu` — GPU FP64 three-way comparison (Python/CPU/GPU)
//!   - `nuclear_eos_l3_ref` — L3 (deformed HFB) architecture test
//!   - `sarkas_gpu` — GPU Yukawa OCP molecular dynamics (9 PP cases, f64 WGSL)
//!   - `verify_hfb` — HFB physics cross-check (Rust vs Python)
//!   - `validate_special_functions` — 77 special function tests
//!   - `validate_linalg` — LU, QR, SVD, tridiagonal solver tests
//!   - `validate_optimizers` — BFGS, NM, bisection, RK45, stats tests
//!   - `validate_md` — MD forces + Velocity-Verlet tests
//!
//! ## Archived (superseded by BarraCUDA native APIs)
//!   Reference implementations that drove BarraCUDA evolution are preserved
//!   in `src/archive/` for historical record. BarraCUDA now has native
//!   implementations of: LOO-CV auto-smoothing, DirectSampler, PenaltyFilter,
//!   chi2_decomposed_weighted, bootstrap_ci, convergence_diagnostics.

pub mod bench;
pub mod data;
pub mod gpu;
pub mod md;
pub mod physics;
pub mod prescreen;

