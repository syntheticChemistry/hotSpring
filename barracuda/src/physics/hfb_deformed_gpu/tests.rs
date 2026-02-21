// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the GPU-accelerated deformed HFB solver.

use super::types::{HamiltonianParamsGpu, NucleusSetup};

#[test]
fn test_nucleus_setup() {
    let s = NucleusSetup::new(8, 8);
    println!(
        "O-16: grid={}×{}={}, states={}, blocks={}",
        s.n_rho,
        s.n_z,
        s.n_grid,
        s.states.len(),
        s.omega_blocks.len()
    );
    assert!(s.n_grid > 4000);
    assert!(s.states.len() > 10);
}

#[test]
fn test_params_gpu_layout() {
    let p = HamiltonianParamsGpu::new(60, 80, 20, 8, 0.2, 0.35);
    assert_eq!(std::mem::size_of_val(&p), 32);
    let d = f64::from_bits(u64::from(p.d_rho_hi) << 32 | u64::from(p.d_rho_lo));
    assert!((d - 0.2).abs() < 1e-15);
}
