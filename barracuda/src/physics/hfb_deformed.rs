//! Axially-Deformed Hartree-Fock + BCS solver (Level 3)
//!
//! Extends the spherical L2 solver to handle nuclear deformation:
//!   - Axially-symmetric HFB in cylindrical coordinates (rho, z)
//!   - Nilsson quantum numbers: n_z, n_perp, Lambda, Omega
//!   - Deformation parameter beta_2 from quadrupole moment Q20
//!   - Block-diagonal structure by Omega (K quantum number)
//!   - Self-consistent deformed densities
//!
//! Uses (all BarraCUDA native):
//!   - barracuda::linalg::eigh_f64 — block diagonalization
//!   - barracuda::optimize::brent — BCS chemical potential
//!   - barracuda::numerical::{trapz, gradient_1d} — integrals & gradients
//!   - barracuda::special::{gamma, laguerre} — basis functions
//!
//! References:
//!   - Ring & Schuck, "The Nuclear Many-Body Problem" (2004), Ch. 11
//!   - Vautherin & Brink, PRC 5, 626 (1972)
//!   - Bender, Heenen, Reinhard, Rev. Mod. Phys. 75, 121 (2003)

use super::constants::*;
use barracuda::linalg::eigh_f64;
use barracuda::numerical::trapz;
use barracuda::special::{gamma, laguerre};
use std::collections::HashMap;
use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// Deformed basis state
// ═══════════════════════════════════════════════════════════════════

/// Nilsson-like quantum numbers for axially-deformed harmonic oscillator
#[derive(Debug, Clone)]
struct DeformedState {
    n_z: usize,     // quantum number along symmetry axis
    n_perp: usize,  // perpendicular oscillator quanta
    lambda: i32,    // projection of orbital angular momentum on z-axis
    sigma: i32,     // +1 or -1 (spin projection: +1/2 or -1/2)
    omega_x2: i32,  // 2*Omega = 2*Lambda + sigma (block index, always > 0)
    parity: i32,    // (-1)^(n_z + 2*n_perp + |lambda|) = (-1)^N
    n_shell: usize, // major shell N = n_z + 2*n_perp + |lambda|
}

impl DeformedState {
    fn omega(&self) -> f64 {
        self.omega_x2 as f64 / 2.0
    }
}

// ═══════════════════════════════════════════════════════════════════
// Row-major matrix helper (same pattern as spherical HFB)
// ═══════════════════════════════════════════════════════════════════

struct Mat {
    data: Vec<f64>,
    n: usize,
}

impl Mat {
    fn zeros(n: usize) -> Self {
        Mat { data: vec![0.0; n * n], n }
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.n + c]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.n + c] = v;
    }

    #[inline]
    fn add(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.n + c] += v;
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2D grid for cylindrical coordinates
// ═══════════════════════════════════════════════════════════════════

/// Cylindrical (rho, z) mesh for deformed densities
struct CylindricalGrid {
    rho: Vec<f64>,  // perpendicular coordinate
    z: Vec<f64>,    // symmetry axis coordinate
    n_rho: usize,
    n_z: usize,
    d_rho: f64,
    d_z: f64,
}

impl CylindricalGrid {
    fn new(rho_max: f64, z_max: f64, n_rho: usize, n_z: usize) -> Self {
        let d_rho = rho_max / n_rho as f64;
        let d_z = 2.0 * z_max / n_z as f64;
        let rho: Vec<f64> = (1..=n_rho).map(|i| i as f64 * d_rho).collect();
        let z: Vec<f64> = (0..n_z).map(|i| -z_max + (i as f64 + 0.5) * d_z).collect();

        CylindricalGrid { rho, z, n_rho, n_z, d_rho, d_z }
    }

    /// Volume element: 2*pi*rho * d_rho * d_z
    fn volume_element(&self, i_rho: usize, _i_z: usize) -> f64 {
        2.0 * PI * self.rho[i_rho] * self.d_rho * self.d_z
    }

    /// Flatten (i_rho, i_z) -> index
    #[inline]
    fn idx(&self, i_rho: usize, i_z: usize) -> usize {
        i_rho * self.n_z + i_z
    }

    /// Total grid points
    fn total(&self) -> usize {
        self.n_rho * self.n_z
    }
}

// ═══════════════════════════════════════════════════════════════════
// Deformed HFB solver
// ═══════════════════════════════════════════════════════════════════

/// Axially-symmetric deformed HFB solver
pub struct DeformedHFB {
    z: usize,
    n_neutrons: usize,
    a: usize,
    grid: CylindricalGrid,
    hw_z: f64,     // HO frequency along z (can differ from perp for deformation)
    hw_perp: f64,  // HO frequency perpendicular
    b_z: f64,      // HO length parameter along z
    b_perp: f64,   // HO length parameter perpendicular
    beta2: f64,    // deformation parameter
    delta_p: f64,  // proton pairing gap
    delta_n: f64,  // neutron pairing gap
    states: Vec<DeformedState>,
    /// Block structure: omega_x2 → state indices
    omega_blocks: HashMap<i32, Vec<usize>>,
}

/// Result from deformed HFB solve
#[derive(Debug, Clone)]
pub struct DeformedHFBResult {
    pub binding_energy_mev: f64,
    pub converged: bool,
    pub iterations: usize,
    pub delta_e: f64,
    pub beta2: f64,      // final deformation
    pub q20_fm2: f64,    // quadrupole moment
    pub rms_radius_fm: f64,
}

impl DeformedHFB {
    /// Create deformed HFB solver with adaptive parameters
    pub fn new_adaptive(z: usize, n: usize) -> Self {
        let a = z + n;
        let a_f = a as f64;

        // HO frequencies: axially-symmetric
        let hw0 = 41.0 * a_f.powf(-1.0 / 3.0);

        // Initial deformation guess: prolate for typical deformed nuclei
        let beta2_init = Self::initial_deformation_guess(z, n);

        // Deformed oscillator parameters
        // omega_z = omega_0 * (1 - 2*beta2/3)
        // omega_perp = omega_0 * (1 + beta2/3)
        // This preserves volume: omega_z * omega_perp^2 = omega_0^3
        let hw_z = hw0 * (1.0 - 2.0 * beta2_init / 3.0);
        let hw_perp = hw0 * (1.0 + beta2_init / 3.0);

        let b_z = HBAR_C / (M_NUCLEON * hw_z).sqrt();
        let b_perp = HBAR_C / (M_NUCLEON * hw_perp).sqrt();

        // Grid: adaptive to nucleus size
        let r0 = 1.2 * a_f.powf(1.0 / 3.0);
        let rho_max = (r0 + 8.0).max(12.0);
        let z_max = (r0 * (1.0 + beta2_init.abs()) + 8.0).max(14.0);
        let n_rho = ((rho_max * 8.0) as usize).max(60);
        let n_z = ((2.0 * z_max * 8.0) as usize).max(80);

        let grid = CylindricalGrid::new(rho_max, z_max, n_rho, n_z);

        // Pairing gaps
        let delta = 12.0 / a_f.max(4.0).sqrt();

        // Number of shells (slightly more than spherical to capture deformation)
        let n_shells = (2.0 * a_f.powf(1.0 / 3.0)) as usize + 5;
        let n_shells = n_shells.max(10).min(16);

        let mut solver = DeformedHFB {
            z,
            n_neutrons: n,
            a,
            grid,
            hw_z,
            hw_perp,
            b_z,
            b_perp,
            beta2: beta2_init,
            delta_p: delta,
            delta_n: delta,
            states: Vec::new(),
            omega_blocks: HashMap::new(),
        };

        solver.build_deformed_basis(n_shells);
        solver
    }

    /// Heuristic initial deformation based on nuclear chart
    fn initial_deformation_guess(z: usize, n: usize) -> f64 {
        let a = z + n;
        // Known deformed regions:
        // 1. Rare earths: 150 < A < 190 (beta2 ~ 0.2-0.35)
        // 2. Actinides: A > 222 (beta2 ~ 0.2-0.3)
        // 3. Light deformed: 20 < A < 28 (beta2 ~ 0.3-0.5)
        // Magic numbers: Z,N = 2,8,20,28,50,82,126 → spherical

        let magic = [2, 8, 20, 28, 50, 82, 126];
        let z_magic = magic.iter().any(|&m| (z as i32 - m as i32).unsigned_abs() <= 2);
        let n_magic = magic.iter().any(|&m| (n as i32 - m as i32).unsigned_abs() <= 2);

        if z_magic && n_magic {
            0.0  // doubly magic → spherical
        } else if a > 222 {
            0.25  // actinides
        } else if a > 150 && a < 190 {
            0.28  // rare earths
        } else if a > 20 && a < 28 {
            0.35  // sd-shell deformed
        } else if z_magic || n_magic {
            0.05  // near magic → weakly deformed
        } else {
            0.15  // generic
        }
    }

    /// Build deformed HO basis: enumerate (n_z, n_perp, Lambda, sigma)
    fn build_deformed_basis(&mut self, n_shells: usize) {
        self.states.clear();
        self.omega_blocks.clear();

        for n_sh in 0..n_shells {
            // N = n_z + 2*n_perp + |Lambda|
            for n_z in 0..=n_sh {
                let remaining = n_sh - n_z;
                // remaining = 2*n_perp + |Lambda|
                for n_perp in 0..=(remaining / 2) {
                    let abs_lambda = remaining - 2 * n_perp;

                    // Lambda can be +/- abs_lambda, but we only need Lambda >= 0
                    // since (Lambda, sigma) and (-Lambda, -sigma) give same Omega
                    let lambdas = if abs_lambda == 0 { vec![0_i32] }
                                  else { vec![abs_lambda as i32] };

                    for &lambda in &lambdas {
                        for &sigma in &[1_i32, -1_i32] {
                            let omega_x2 = 2 * lambda + sigma;

                            // Only positive Omega blocks (time-reversal partner is implicit)
                            if omega_x2 <= 0 { continue; }

                            let parity = if n_sh % 2 == 0 { 1 } else { -1 };

                            self.states.push(DeformedState {
                                n_z,
                                n_perp,
                                lambda,
                                sigma,
                                omega_x2,
                                parity,
                                n_shell: n_sh,
                            });
                        }
                    }
                }
            }
        }

        // Build Omega-block index
        for (i, s) in self.states.iter().enumerate() {
            self.omega_blocks.entry(s.omega_x2).or_default().push(i);
        }
    }

    /// Evaluate deformed HO wavefunction on the cylindrical grid
    /// psi(rho, z) = phi_nz(z/b_z) * phi_{n_perp,|Lambda|}(rho/b_perp)
    fn evaluate_wavefunction(&self, state: &DeformedState) -> Vec<f64> {
        let mut psi = vec![0.0; self.grid.total()];

        for i_rho in 0..self.grid.n_rho {
            for i_z in 0..self.grid.n_z {
                let rho = self.grid.rho[i_rho];
                let z = self.grid.z[i_z];

                // z-part: Hermite oscillator
                let phi_z = Self::hermite_oscillator(state.n_z, z, self.b_z);

                // rho-part: 2D oscillator with |Lambda|
                let phi_rho = Self::laguerre_oscillator(
                    state.n_perp, state.lambda.unsigned_abs() as usize, rho, self.b_perp
                );

                psi[self.grid.idx(i_rho, i_z)] = phi_z * phi_rho;
            }
        }

        psi
    }

    /// 1D harmonic oscillator along z: H_n(z/b) * exp(-z²/2b²) * normalization
    fn hermite_oscillator(n: usize, z: f64, b: f64) -> f64 {
        let xi = z / b;
        // Use Laguerre-based Hermite connection for numerical stability:
        // H_{2m}(x) related to L_m^{-1/2}(x²)
        // H_{2m+1}(x) related to L_m^{1/2}(x²) * x
        // But simpler: direct Hermite via recurrence
        let h_n = hermite_value(n, xi);
        let norm = 1.0 / (b * PI.sqrt() * (1 << n) as f64 * factorial_f64(n)).sqrt();
        norm * h_n * (-xi * xi / 2.0).exp()
    }

    /// 2D oscillator radial part: Laguerre basis
    /// R_{n_perp, |Lambda|}(rho) = sqrt(2 n! / (b² Gamma(n+|L|+1))) * (rho/b)^|L|
    ///                              * exp(-rho²/2b²) * L_n^|L|(rho²/b²)
    fn laguerre_oscillator(n_perp: usize, abs_lambda: usize, rho: f64, b: f64) -> f64 {
        let eta = (rho / b).powi(2);
        let alpha = abs_lambda as f64;

        let n_fact = factorial_f64(n_perp);
        let gamma_val = gamma(n_perp as f64 + alpha + 1.0).unwrap_or(1.0);
        let norm = (2.0 * n_fact / (b * b * gamma_val)).sqrt();

        let lag = laguerre(n_perp, alpha, eta);
        norm * (rho / b).powi(abs_lambda as i32) * (-eta / 2.0).exp() * lag
    }

    /// Self-consistent HFB solve
    pub fn solve(&mut self, params: &[f64]) -> DeformedHFBResult {
        let max_iter = 200;
        let tol = 1e-6; // MeV

        let mut e_prev = 0.0;
        let mut converged = false;
        let mut binding_energy = 0.0;

        // Precompute all wavefunctions on the grid
        let wavefunctions: Vec<Vec<f64>> = self.states.iter()
            .map(|s| self.evaluate_wavefunction(s))
            .collect();

        // Initialize with zero mean-field → just oscillator
        let n_grid = self.grid.total();
        let mut rho_p = vec![0.0; n_grid];
        let mut rho_n = vec![0.0; n_grid];

        let mut iter = 0;
        let mut delta_e = 0.0;

        for iteration in 0..max_iter {
            iter = iteration + 1;
            let total_rho: Vec<f64> = rho_p.iter().zip(&rho_n)
                .map(|(&p, &n)| p + n)
                .collect();

            // Compute mean-field potentials
            let v_p = self.mean_field_potential(params, &rho_p, &rho_n, &total_rho, true);
            let v_n = self.mean_field_potential(params, &rho_p, &rho_n, &total_rho, false);

            // Diagonalize block-by-block for each Omega
            let (eigs_p, occ_p) = self.diagonalize_blocks(
                &v_p, &wavefunctions, self.z, self.delta_p
            );
            let (eigs_n, occ_n) = self.diagonalize_blocks(
                &v_n, &wavefunctions, self.n_neutrons, self.delta_n
            );

            // Update densities
            let (new_rho_p, new_rho_n) = self.compute_densities(
                &wavefunctions, &occ_p, &occ_n
            );

            // Mix densities for convergence (simple linear mixing)
            let alpha_mix = if iteration < 5 { 0.3 } else { 0.5 };
            for i in 0..n_grid {
                rho_p[i] = (1.0 - alpha_mix) * rho_p[i] + alpha_mix * new_rho_p[i];
                rho_n[i] = (1.0 - alpha_mix) * rho_n[i] + alpha_mix * new_rho_n[i];
            }

            // Compute total energy
            binding_energy = self.total_energy(
                params, &rho_p, &rho_n, &eigs_p, &eigs_n, &occ_p, &occ_n
            );

            delta_e = (binding_energy - e_prev).abs();
            if iteration > 0 && delta_e < tol {
                converged = true;
                break;
            }
            e_prev = binding_energy;
        }

        // Compute deformation observables
        let rho_total: Vec<f64> = rho_p.iter().zip(&rho_n)
            .map(|(&p, &n)| p + n)
            .collect();
        let q20 = self.quadrupole_moment(&rho_total);
        let beta2 = self.beta2_from_q20(q20);
        let rms_r = self.rms_radius(&rho_total);

        DeformedHFBResult {
            binding_energy_mev: binding_energy,
            converged,
            iterations: iter,
            delta_e,
            beta2,
            q20_fm2: q20,
            rms_radius_fm: rms_r,
        }
    }

    /// Compute mean-field potential on cylindrical grid
    fn mean_field_potential(
        &self,
        params: &[f64],
        rho_p: &[f64],
        rho_n: &[f64],
        rho_total: &[f64],
        is_proton: bool,
    ) -> Vec<f64> {
        let n = self.grid.total();
        let mut v = vec![0.0; n];

        let t0 = params[0];
        let t3 = params[3];
        let x0 = params[4];
        let x3 = params[7];
        let alpha = params[8];

        let rho_q: &[f64] = if is_proton { rho_p } else { rho_n };
        let rho_q_prime: &[f64] = if is_proton { rho_n } else { rho_p };

        for i in 0..n {
            let rho = rho_total[i].max(0.0);
            let rq = rho_q[i].max(0.0);
            let rqp = rho_q_prime[i].max(0.0);

            // Central Skyrme
            let v_central = t0 * ((1.0 + x0 / 2.0) * rho - (0.5 + x0) * rq)
                + t3 / 12.0 * rho.powf(alpha) *
                    ((2.0 + alpha) * (1.0 + x3 / 2.0) * rho
                     - (2.0 * (0.5 + x3) * rq + alpha * (1.0 + x3 / 2.0) * rho));

            v[i] = v_central;

            // Coulomb for protons
            if is_proton && rho_p.iter().any(|&x| x > 0.0) {
                // Approximate spherical Coulomb (simplified for deformed case)
                let r_eff = self.effective_radius_at(i);
                if r_eff > 0.01 {
                    v[i] += E2 * rho_p.iter().zip(0..n)
                        .map(|(&rp, j)| {
                            let r_j = self.effective_radius_at(j);
                            rp * self.grid.volume_element(j / self.grid.n_z, j % self.grid.n_z)
                                / r_eff.max(r_j).max(0.01)
                        })
                        .sum::<f64>() * 0.1; // approximate scaling
                }
                // Slater exchange
                v[i] -= E2 * (3.0 / PI).powf(1.0 / 3.0) * rho_p[i].max(0.0).powf(1.0 / 3.0);
            }
        }

        v
    }

    /// Effective radius at grid point (for Coulomb approximation)
    fn effective_radius_at(&self, idx: usize) -> f64 {
        let i_rho = idx / self.grid.n_z;
        let i_z = idx % self.grid.n_z;
        if i_rho < self.grid.n_rho {
            let rho = self.grid.rho[i_rho];
            let z = self.grid.z[i_z];
            (rho * rho + z * z).sqrt()
        } else {
            0.0
        }
    }

    /// Diagonalize Hamiltonian block-by-block in Omega
    fn diagonalize_blocks(
        &self,
        v_potential: &[f64],
        wavefunctions: &[Vec<f64>],
        n_particles: usize,
        delta_pair: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n_states = self.states.len();
        let mut all_eigenvalues = vec![0.0; n_states];
        let mut all_occupations = vec![0.0; n_states];

        // Temporary storage: eigenvalue/eigenvector for each block
        let mut block_eigs: Vec<(usize, f64)> = Vec::new(); // (state_idx, eigenvalue)

        for (_, block_indices) in &self.omega_blocks {
            let block_size = block_indices.len();
            if block_size == 0 { continue; }

            // Build block Hamiltonian
            let mut h = Mat::zeros(block_size);

            for bi in 0..block_size {
                let i = block_indices[bi];
                for bj in bi..block_size {
                    let j = block_indices[bj];

                    // Kinetic energy (diagonal for HO basis)
                    let t_ij = if i == j {
                        let s = &self.states[i];
                        self.hw_z * (s.n_z as f64 + 0.5)
                            + self.hw_perp * (2.0 * s.n_perp as f64 + s.lambda.unsigned_abs() as f64 + 1.0)
                    } else {
                        0.0
                    };

                    // Potential matrix element: <i|V|j> = integral psi_i * V * psi_j * dV
                    let v_ij = self.potential_matrix_element(
                        &wavefunctions[i], &wavefunctions[j], v_potential
                    );

                    let h_ij = t_ij + v_ij;
                    h.set(bi, bj, h_ij);
                    if bi != bj { h.set(bj, bi, h_ij); }
                }
            }

            // Diagonalize this block
            let eig = eigh_f64(&h.data, block_size).expect("eigh_f64 failed for deformed block");

            for (bi, &eval) in eig.eigenvalues.iter().enumerate() {
                let state_idx = block_indices[bi];
                all_eigenvalues[state_idx] = eval;
                // Each Omega block has degeneracy 2 (time reversal: Omega and -Omega)
                block_eigs.push((state_idx, eval));
            }
        }

        // BCS occupations
        block_eigs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let degs: Vec<f64> = (0..n_states).map(|_| 2.0).collect(); // time-reversal degeneracy

        if delta_pair > 1e-10 {
            // BCS with pairing
            let fermi = Self::approx_fermi(&block_eigs, n_particles, &degs);
            for (state_idx, eval) in &block_eigs {
                let eps = eval - fermi;
                let e_qp = (eps * eps + delta_pair * delta_pair).sqrt();
                let v2 = 0.5 * (1.0 - eps / e_qp);
                all_occupations[*state_idx] = v2.clamp(0.0, 1.0);
            }
        } else {
            // Sharp Fermi surface
            let mut particles_left = n_particles as f64;
            for (state_idx, _eval) in &block_eigs {
                let deg = degs[*state_idx];
                if particles_left >= deg {
                    all_occupations[*state_idx] = 1.0;
                    particles_left -= deg;
                } else if particles_left > 0.0 {
                    all_occupations[*state_idx] = particles_left / deg;
                    particles_left = 0.0;
                }
            }
        }

        (all_eigenvalues, all_occupations)
    }

    /// Approximate Fermi energy from eigenvalues
    fn approx_fermi(
        sorted_eigs: &[(usize, f64)],
        n_particles: usize,
        _degs: &[f64],
    ) -> f64 {
        let mut count = 0.0;
        for (_, eval) in sorted_eigs {
            count += 2.0; // degeneracy
            if count >= n_particles as f64 {
                return *eval;
            }
        }
        sorted_eigs.last().map(|(_, e)| *e).unwrap_or(0.0)
    }

    /// Potential matrix element: <i|V|j> via numerical integration
    fn potential_matrix_element(
        &self,
        psi_i: &[f64],
        psi_j: &[f64],
        v: &[f64],
    ) -> f64 {
        let mut integral = 0.0;
        for i_rho in 0..self.grid.n_rho {
            for i_z in 0..self.grid.n_z {
                let idx = self.grid.idx(i_rho, i_z);
                let dv = self.grid.volume_element(i_rho, i_z);
                integral += psi_i[idx] * v[idx] * psi_j[idx] * dv;
            }
        }
        integral
    }

    /// Compute densities from wavefunctions and occupations
    fn compute_densities(
        &self,
        wavefunctions: &[Vec<f64>],
        occ_p: &[f64],
        occ_n: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = self.grid.total();
        let mut rho_p = vec![0.0; n];
        let mut rho_n = vec![0.0; n];

        for (i, s) in self.states.iter().enumerate() {
            let occ_proton = occ_p[i] * 2.0; // 2 for time-reversal degeneracy
            let occ_neutron = occ_n[i] * 2.0;

            if occ_proton > 1e-15 || occ_neutron > 1e-15 {
                for k in 0..n {
                    let psi2 = wavefunctions[i][k] * wavefunctions[i][k];
                    rho_p[k] += occ_proton * psi2;
                    rho_n[k] += occ_neutron * psi2;
                }
            }
        }

        (rho_p, rho_n)
    }

    /// Compute total energy (Skyrme EDF)
    fn total_energy(
        &self,
        params: &[f64],
        rho_p: &[f64],
        rho_n: &[f64],
        eigs_p: &[f64],
        eigs_n: &[f64],
        occ_p: &[f64],
        occ_n: &[f64],
    ) -> f64 {
        // Sum of single-particle energies weighted by occupation
        let mut e_sp = 0.0;
        for (i, s) in self.states.iter().enumerate() {
            let deg = 2.0; // time-reversal
            e_sp += deg * occ_p[i] * eigs_p[i] + deg * occ_n[i] * eigs_n[i];
        }

        // Potential energy contribution (avoid double counting)
        let mut e_pot = 0.0;
        let t0 = params[0];
        let t3 = params[3];
        let x0 = params[4];
        let x3 = params[7];
        let alpha = params[8];

        for i_rho in 0..self.grid.n_rho {
            for i_z in 0..self.grid.n_z {
                let idx = self.grid.idx(i_rho, i_z);
                let dv = self.grid.volume_element(i_rho, i_z);
                let rho = (rho_p[idx] + rho_n[idx]).max(0.0);
                let rp = rho_p[idx].max(0.0);
                let rn = rho_n[idx].max(0.0);

                // Skyrme volume energy
                let e_vol = t0 / 4.0 * ((2.0 + x0) * rho * rho - (1.0 + 2.0 * x0) * (rp * rp + rn * rn))
                    + t3 / 24.0 * rho.powf(alpha) *
                        ((2.0 + x3) * rho * rho - (1.0 + 2.0 * x3) * (rp * rp + rn * rn));

                e_pot += e_vol * dv;
            }
        }

        // E_total ≈ E_sp - E_pot_double_counting + pairing
        let e_pair_p = -0.25 * self.delta_p * self.delta_p * self.z as f64 / 10.0;
        let e_pair_n = -0.25 * self.delta_n * self.delta_n * self.n_neutrons as f64 / 10.0;

        // CM correction (approximate: 0.75 * 41/A^(1/3))
        let e_cm = 0.75 * 41.0 / (self.a as f64).powf(1.0 / 3.0);

        e_sp - e_pot + e_pair_p + e_pair_n - e_cm
    }

    /// Quadrupole moment Q20 = integral rho(rho,z) * (2z² - rho²) dV
    fn quadrupole_moment(&self, rho_total: &[f64]) -> f64 {
        let mut q20 = 0.0;
        for i_rho in 0..self.grid.n_rho {
            for i_z in 0..self.grid.n_z {
                let idx = self.grid.idx(i_rho, i_z);
                let dv = self.grid.volume_element(i_rho, i_z);
                let rho = self.grid.rho[i_rho];
                let z = self.grid.z[i_z];
                q20 += rho_total[idx] * (2.0 * z * z - rho * rho) * dv;
            }
        }
        q20
    }

    /// Extract beta2 deformation parameter from Q20
    fn beta2_from_q20(&self, q20: f64) -> f64 {
        let a = self.a as f64;
        let r0 = 1.2 * a.powf(1.0 / 3.0);
        // beta2 = sqrt(5*pi) * Q20 / (3 * A * R0²)
        (5.0 * PI).sqrt() * q20 / (3.0 * a * r0 * r0)
    }

    /// RMS radius from density
    fn rms_radius(&self, rho_total: &[f64]) -> f64 {
        let mut sum_r2 = 0.0;
        let mut sum_rho = 0.0;
        for i_rho in 0..self.grid.n_rho {
            for i_z in 0..self.grid.n_z {
                let idx = self.grid.idx(i_rho, i_z);
                let dv = self.grid.volume_element(i_rho, i_z);
                let rho = self.grid.rho[i_rho];
                let z = self.grid.z[i_z];
                let r2 = rho * rho + z * z;
                sum_r2 += rho_total[idx] * r2 * dv;
                sum_rho += rho_total[idx] * dv;
            }
        }
        if sum_rho > 0.0 { (sum_r2 / sum_rho).sqrt() } else { 0.0 }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════════

/// Hermite polynomial H_n(x) via recurrence
fn hermite_value(n: usize, x: f64) -> f64 {
    if n == 0 { return 1.0; }
    if n == 1 { return 2.0 * x; }

    let mut h_prev = 1.0;
    let mut h_curr = 2.0 * x;
    for k in 2..=n {
        let h_next = 2.0 * x * h_curr - 2.0 * (k - 1) as f64 * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }
    h_curr
}

/// Factorial for f64
fn factorial_f64(n: usize) -> f64 {
    if n <= 1 { return 1.0; }
    (2..=n).fold(1.0, |acc, k| acc * k as f64)
}

/// Public API: L3 binding energy from deformed HFB
pub fn binding_energy_l3(z: usize, n: usize, params: &[f64]) -> (f64, bool, f64) {
    let mut solver = DeformedHFB::new_adaptive(z, n);
    let result = solver.solve(params);
    (result.binding_energy_mev, result.converged, result.beta2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hermite_values() {
        assert!((hermite_value(0, 1.0) - 1.0).abs() < 1e-10);
        assert!((hermite_value(1, 1.0) - 2.0).abs() < 1e-10);
        assert!((hermite_value(2, 1.0) - 2.0).abs() < 1e-10); // 4x²-2 = 2
        assert!((hermite_value(3, 1.0) - (-4.0)).abs() < 1e-10); // 8x³-12x = -4
    }

    #[test]
    fn test_deformed_basis_count() {
        let solver = DeformedHFB::new_adaptive(8, 8); // O-16
        // Should have states in multiple Omega blocks
        assert!(solver.states.len() > 10);
        assert!(solver.omega_blocks.len() >= 2);
        println!("O-16 deformed basis: {} states, {} Omega blocks",
            solver.states.len(), solver.omega_blocks.len());
    }

    #[test]
    fn test_deformation_guess() {
        // Doubly magic
        assert_eq!(DeformedHFB::initial_deformation_guess(20, 20), 0.0);
        // Rare earth
        assert!(DeformedHFB::initial_deformation_guess(66, 96) > 0.2);
        // Actinide
        assert!(DeformedHFB::initial_deformation_guess(92, 146) > 0.2);
    }

    #[test]
    fn test_deformed_hfb_runs() {
        // Just verify it doesn't crash
        let sly4: [f64; 10] = [
            -2488.91, 486.82, -546.39, 13777.0,
            0.834, -0.344, -1.0, 1.354, 0.1667, 123.0,
        ];
        let (be, conv, beta2) = binding_energy_l3(8, 8, &sly4);
        println!("O-16 deformed: B={:.2} MeV, conv={}, beta2={:.4}", be, conv, beta2);
        // O-16 is doubly magic, should be nearly spherical
        assert!(beta2.abs() < 0.5, "O-16 should be nearly spherical");
    }
}
