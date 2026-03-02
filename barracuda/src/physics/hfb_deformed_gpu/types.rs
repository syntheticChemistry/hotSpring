// SPDX-License-Identifier: AGPL-3.0-only

//! Data types for the GPU-accelerated deformed HFB solver.
//!
//! Contains: `GpuResidentL3Result`, `HamiltonianParamsGpu`, `BasisState`,
//! `NucleusSetup` with grid/basis construction.

use super::super::constants::{HBAR_C, M_NUCLEON};
use super::super::hfb_deformed_common::deformation_guess;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Result from GPU-resident L3 evaluation.
#[derive(Debug)]
pub struct GpuResidentL3Result {
    /// Per-nucleus results: (Z, N, binding_MeV, converged, wall_s).
    pub results: Vec<(usize, usize, f64, bool, f64)>,
    /// Total wall time (seconds).
    pub wall_time_s: f64,
    /// Number of eigensolver dispatches.
    pub eigh_dispatches: usize,
    /// Total GPU dispatches.
    pub total_gpu_dispatches: usize,
    /// Number of nuclei evaluated.
    pub n_nuclei: usize,
}

/// Matches the WGSL `HamiltonianParams` struct byte-for-byte.
/// Layout: 4×u32 (16 bytes) + 2×f64 (16 bytes) = 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct HamiltonianParamsGpu {
    pub(super) n_rho: u32,
    pub(super) n_z: u32,
    pub(super) block_size: u32,
    pub(super) n_blocks: u32,
    pub(super) d_rho_lo: u32,
    pub(super) d_rho_hi: u32,
    pub(super) d_z_lo: u32,
    pub(super) d_z_hi: u32,
}

impl HamiltonianParamsGpu {
    #[allow(clippy::cast_possible_truncation)] // f64 bit-splitting for WGSL uniform; no truncation
    #[allow(dead_code)] // EVOLUTION(GPU): used in test_params_gpu_layout; will wire to deformed_*.wgsl when GPU pipeline is complete
    pub(super) const fn new(
        n_rho: u32,
        n_z: u32,
        block_size: u32,
        n_blocks: u32,
        d_rho: f64,
        d_z: f64,
    ) -> Self {
        let dr = d_rho.to_bits();
        let dz = d_z.to_bits();
        Self {
            n_rho,
            n_z,
            block_size,
            n_blocks,
            d_rho_lo: dr as u32,
            d_rho_hi: (dr >> 32) as u32,
            d_z_lo: dz as u32,
            d_z_hi: (dz >> 32) as u32,
        }
    }
}

pub(super) struct BasisState {
    pub(super) n_z: u32,
    pub(super) n_perp: u32,
    pub(super) abs_lambda: u32,
    pub(super) lambda: i32,
    pub(super) sigma: i32,
    pub(super) omega_x2: i32,
    #[allow(dead_code)]
    // EVOLUTION(GPU): will be used when deformed_*.wgsl shaders need shell truncation
    pub(super) _n_shell: u32,
}

pub(super) struct NucleusSetup {
    pub(super) z: usize,
    pub(super) n_neutrons: usize,
    pub(super) a: usize,
    pub(super) n_rho: usize,
    pub(super) n_z: usize,
    pub(super) n_grid: usize,
    pub(super) d_rho: f64,
    pub(super) d_z: f64,
    pub(super) z_min: f64,
    #[allow(dead_code)]
    // EVOLUTION(GPU): will be used when deformed_*.wgsl shaders wire grid bounds
    pub(super) _rho_max: f64,
    pub(super) hw_z: f64,
    pub(super) hw_perp: f64,
    pub(super) b_z: f64,
    pub(super) b_perp: f64,
    pub(super) delta_p: f64,
    pub(super) delta_n: f64,
    pub(super) states: Vec<BasisState>,
    pub(super) omega_blocks: HashMap<i32, Vec<usize>>,
}

impl NucleusSetup {
    pub(super) fn new(z: usize, n: usize) -> Self {
        let a = z + n;
        let a_f = a as f64;
        let hw0 = 41.0 * a_f.powf(-1.0 / 3.0);
        let beta2_init = deformation_guess(z, n);
        let hw_z = hw0 * (1.0 - 2.0 * beta2_init / 3.0);
        let hw_perp = hw0 * (1.0 + beta2_init / 3.0);
        let b_z = HBAR_C / (M_NUCLEON * hw_z).sqrt();
        let b_perp = HBAR_C / (M_NUCLEON * hw_perp).sqrt();
        let r0 = 1.2 * a_f.cbrt();
        let rho_max = (r0 + 8.0).max(12.0);
        let z_max = (r0 * (1.0 + beta2_init.abs()) + 8.0).max(14.0);
        let n_rho = ((rho_max * 8.0) as usize).max(60);
        let n_z_val = ((2.0 * z_max * 8.0) as usize).max(80);
        let d_rho = rho_max / n_rho as f64;
        let d_z = 2.0 * z_max / n_z_val as f64;
        let delta = 12.0 / a_f.max(4.0).sqrt();
        let n_shells = ((2.0 * a_f.cbrt()) as usize + 5).clamp(10, 16);

        let mut setup = Self {
            z,
            n_neutrons: n,
            a,
            n_rho,
            n_z: n_z_val,
            n_grid: n_rho * n_z_val,
            d_rho,
            d_z,
            z_min: -z_max,
            _rho_max: rho_max,
            hw_z,
            hw_perp,
            b_z,
            b_perp,
            delta_p: delta,
            delta_n: delta,
            states: Vec::new(),
            omega_blocks: HashMap::new(),
        };
        setup.build_basis(n_shells);
        setup
    }

    #[allow(clippy::cast_possible_truncation)] // Basis quantum numbers: n_z, n_perp, abs_l, n_shell ≤ 16
    pub(super) fn build_basis(&mut self, n_shells: usize) {
        for n_sh in 0..n_shells {
            for n_z_v in 0..=n_sh {
                let rem = n_sh - n_z_v;
                for n_perp in 0..=(rem / 2) {
                    let abs_l = rem - 2 * n_perp;
                    let lams = if abs_l == 0 {
                        vec![0i32]
                    } else {
                        vec![abs_l as i32]
                    };
                    for &lam in &lams {
                        for &sig in &[1i32, -1i32] {
                            let omega_x2 = 2 * lam + sig;
                            if omega_x2 <= 0 {
                                continue;
                            }
                            self.states.push(BasisState {
                                n_z: n_z_v as u32,
                                n_perp: n_perp as u32,
                                abs_lambda: abs_l as u32,
                                lambda: lam,
                                sigma: sig,
                                omega_x2,
                                _n_shell: n_sh as u32,
                            });
                        }
                    }
                }
            }
        }
        for (i, s) in self.states.iter().enumerate() {
            self.omega_blocks.entry(s.omega_x2).or_default().push(i);
        }
    }

    pub(super) fn volume_element(&self, i_rho: usize) -> f64 {
        let rho = (i_rho + 1) as f64 * self.d_rho;
        2.0 * PI * rho * self.d_rho * self.d_z
    }

    pub(super) const fn grid_idx(&self, i_rho: usize, i_z: usize) -> usize {
        i_rho * self.n_z + i_z
    }
}
