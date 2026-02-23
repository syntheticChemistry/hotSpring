// SPDX-License-Identifier: AGPL-3.0-only

//! Abelian Higgs model on a (1+1)D lattice.
//!
//! Implements the `U(1)` gauge + complex scalar Higgs model from
//! Bazavov et al., Phys. Rev. D 92, 076003 (2015).
//!
//! # Action
//!
//! S = `S_gauge` + `S_higgs`
//!
//! `S_gauge` = `β_pl` Σ\_{plaq} (1 − Re `U_p`)
//!
//! `S_higgs` = Σ\_x [ −κ Σ\_μ (φ\*(x) `U_μ`(x) φ(x+μ̂) + h.c.)
//!                  + |φ(x)|² + λ(|φ(x)|² − 1)² ]
//!
//! where `U_μ`(x) = e^{i`θ_μ`(x)} are `U(1)` link variables and φ(x) are
//! complex scalar Higgs fields.
//!
//! # Phase structure
//!
//! The model exhibits three regimes:
//! - **Confined** (small β, small κ): disordered links, small Higgs `VEV`
//! - **Higgs** (large κ): Higgs condensation, ⟨|φ|²⟩ → 1
//! - **Coulomb** (large β, small κ): ordered links, small Higgs `VEV`
//!
//! In (1+1)D with λ → ∞ and spin-1 truncation, the model maps to
//! a two-species Bose-Hubbard model.

use super::complex_f64::Complex64;
use super::constants::{lcg_gaussian, lcg_uniform_f64};
use std::f64::consts::PI;

/// Parameters for the Abelian Higgs model.
#[derive(Clone, Debug)]
pub struct AbelianHiggsParams {
    /// Inverse gauge coupling: `β_pl` = 1/g².
    pub beta_pl: f64,
    /// Hopping parameter (gauge–Higgs coupling).
    pub kappa: f64,
    /// Scalar self-coupling.
    pub lambda: f64,
    /// Chemical potential (temporal hopping weight).
    pub mu: f64,
}

impl AbelianHiggsParams {
    /// Create new Abelian Higgs parameters.
    #[must_use]
    pub const fn new(beta_pl: f64, kappa: f64, lambda: f64) -> Self {
        Self {
            beta_pl,
            kappa,
            lambda,
            mu: 0.0,
        }
    }

    /// Set chemical potential (temporal hopping weight).
    #[must_use]
    pub const fn with_mu(mut self, mu: f64) -> Self {
        self.mu = mu;
        self
    }
}

/// (1+1)D lattice with U(1) gauge links and complex scalar Higgs field.
///
/// Layout: `nt` temporal × `ns` spatial sites with periodic boundaries.
/// Links: 2 per site (temporal μ=0, spatial μ=1).
#[allow(missing_docs)]
pub struct U1HiggsLattice {
    pub nt: usize,
    pub ns: usize,
    pub params: AbelianHiggsParams,
    /// Link angles `θ_μ`(x) ∈ [−π, π). Layout: `links[site * 2 + mu]`.
    pub links: Vec<f64>,
    /// Higgs field φ(x). Layout: `higgs[site]`.
    pub higgs: Vec<Complex64>,
}

impl U1HiggsLattice {
    /// Total number of lattice sites.
    #[must_use]
    pub const fn volume(&self) -> usize {
        self.nt * self.ns
    }

    /// Flat site index from coordinates (t, x).
    #[inline]
    #[must_use]
    pub const fn site_index(&self, t: usize, x: usize) -> usize {
        t * self.ns + x
    }

    /// Neighbor site in direction mu (+1) with periodic boundaries.
    /// mu=0: temporal, mu=1: spatial.
    #[inline]
    #[must_use]
    pub const fn neighbor_fwd(&self, t: usize, x: usize, mu: usize) -> (usize, usize) {
        if mu == 0 {
            ((t + 1) % self.nt, x)
        } else {
            (t, (x + 1) % self.ns)
        }
    }

    /// Neighbor site in direction -mu with periodic boundaries.
    #[inline]
    #[must_use]
    pub const fn neighbor_bwd(&self, t: usize, x: usize, mu: usize) -> (usize, usize) {
        if mu == 0 {
            ((t + self.nt - 1) % self.nt, x)
        } else {
            (t, (x + self.ns - 1) % self.ns)
        }
    }

    /// Access link angle `θ_μ` at site (t, x).
    #[inline]
    #[must_use]
    pub fn link_angle(&self, t: usize, x: usize, mu: usize) -> f64 {
        self.links[self.site_index(t, x) * 2 + mu]
    }

    /// Access link `U_μ` = e^{iθ} at site (t, x).
    #[inline]
    pub fn link(&self, t: usize, x: usize, mu: usize) -> Complex64 {
        Complex64::from_polar(self.link_angle(t, x, mu))
    }

    /// Cold start: all links θ=0 (ordered), all Higgs φ=1.
    #[must_use]
    pub fn cold_start(nt: usize, ns: usize, params: AbelianHiggsParams) -> Self {
        let vol = nt * ns;
        Self {
            nt,
            ns,
            params,
            links: vec![0.0; vol * 2],
            higgs: vec![Complex64::ONE; vol],
        }
    }

    /// Hot start: random links θ ∈ [−π, π), random Higgs with |φ| ~ 1.
    pub fn hot_start(nt: usize, ns: usize, params: AbelianHiggsParams, seed: &mut u64) -> Self {
        let vol = nt * ns;
        let links: Vec<f64> = (0..vol * 2)
            .map(|_| (2.0 * PI).mul_add(lcg_uniform_f64(seed), -PI))
            .collect();
        let higgs: Vec<Complex64> = (0..vol)
            .map(|_| {
                let re = lcg_gaussian(seed).mul_add(0.5, 1.0);
                let im = lcg_gaussian(seed) * 0.5;
                Complex64::new(re, im)
            })
            .collect();
        Self {
            nt,
            ns,
            params,
            links,
            higgs,
        }
    }

    // ── Plaquette ────────────────────────────────────────────────────

    /// Plaquette `U_p` = `U_0`(x) `U_1`(x+0̂) `U_0`†(x+1̂) `U_1`†(x).
    ///
    /// In (1+1)D there's exactly one plaquette orientation (temporal × spatial).
    /// Returns the phase: `θ_0`(x) + `θ_1`(x+0̂) − `θ_0`(x+1̂) − `θ_1`(x).
    #[must_use]
    pub fn plaquette_phase(&self, t: usize, x: usize) -> f64 {
        let (t_fwd, _) = self.neighbor_fwd(t, x, 0);
        let (_, x_fwd) = self.neighbor_fwd(t, x, 1);
        self.link_angle(t, x, 0) + self.link_angle(t_fwd, x, 1)
            - self.link_angle(t, x_fwd, 0)
            - self.link_angle(t, x, 1)
    }

    /// `Re`(`U_p`) = cos(`plaquette_phase`).
    #[must_use]
    pub fn plaquette_re(&self, t: usize, x: usize) -> f64 {
        self.plaquette_phase(t, x).cos()
    }

    /// Average plaquette ⟨Re `U_p`⟩ over all sites.
    #[must_use]
    pub fn average_plaquette(&self) -> f64 {
        let mut sum = 0.0;
        for t in 0..self.nt {
            for x in 0..self.ns {
                sum += self.plaquette_re(t, x);
            }
        }
        sum / self.volume() as f64
    }

    // ── Action ───────────────────────────────────────────────────────

    /// Wilson gauge action: `S_gauge` = `β_pl` Σ (1 − Re `U_p`).
    #[must_use]
    pub fn gauge_action(&self) -> f64 {
        let mut s = 0.0;
        for t in 0..self.nt {
            for x in 0..self.ns {
                s += 1.0 - self.plaquette_re(t, x);
            }
        }
        self.params.beta_pl * s
    }

    /// Higgs action: kinetic + potential.
    ///
    /// `S_higgs` = Σ\_x [ −κ Σ\_μ (φ\*(x) `U_μ`(x) φ(x+μ̂) + h.c.)
    ///                  + |φ(x)|² + λ(|φ(x)|² − 1)² ]
    ///
    /// Chemical potential: temporal hopping gets exp(±μ) weight.
    #[must_use]
    pub fn higgs_action(&self) -> f64 {
        let kappa = self.params.kappa;
        let lambda = self.params.lambda;
        let mu = self.params.mu;
        let mut s_kin = 0.0;
        let mut s_pot = 0.0;

        for t in 0..self.nt {
            for x in 0..self.ns {
                let idx = self.site_index(t, x);
                let phi = self.higgs[idx];
                let phi_sq = phi.abs_sq();

                s_pot += (lambda * (phi_sq - 1.0)).mul_add(phi_sq - 1.0, phi_sq);

                for mu_dir in 0..2 {
                    let (t_fwd, x_fwd) = self.neighbor_fwd(t, x, mu_dir);
                    let idx_fwd = self.site_index(t_fwd, x_fwd);
                    let phi_fwd = self.higgs[idx_fwd];
                    let u = self.link(t, x, mu_dir);

                    let hop = phi.conj() * u * phi_fwd;
                    let chem_weight = if mu_dir == 0 { mu.exp() } else { 1.0 };
                    s_kin += chem_weight * hop.re;
                }
            }
        }
        (-2.0 * kappa).mul_add(s_kin, s_pot)
    }

    /// Total action S = `S_gauge` + `S_higgs`.
    #[must_use]
    pub fn total_action(&self) -> f64 {
        self.gauge_action() + self.higgs_action()
    }

    // ── Observables ──────────────────────────────────────────────────

    /// Average Higgs field modulus squared ⟨|φ|²⟩.
    #[must_use]
    pub fn average_higgs_sq(&self) -> f64 {
        let sum: f64 = self.higgs.iter().map(|phi| phi.abs_sq()).sum();
        sum / self.volume() as f64
    }

    /// Average Higgs field ⟨φ⟩ (complex).
    pub fn average_higgs(&self) -> Complex64 {
        let sum: Complex64 = self
            .higgs
            .iter()
            .copied()
            .fold(Complex64::ZERO, |a, b| a + b);
        sum.scale(1.0 / self.volume() as f64)
    }

    /// Polyakov loop: product of temporal links along a spatial slice.
    /// L(x) = Π\_{t=0}^{Nt-1} `U_0`(t, x)
    pub fn polyakov_loop(&self, x: usize) -> Complex64 {
        let mut prod = Complex64::ONE;
        for t in 0..self.nt {
            prod *= self.link(t, x, 0);
        }
        prod
    }

    /// Average Polyakov loop ⟨|L|⟩.
    #[must_use]
    pub fn average_polyakov_loop(&self) -> f64 {
        let sum: f64 = (0..self.ns).map(|x| self.polyakov_loop(x).abs()).sum();
        sum / self.ns as f64
    }

    // ── HMC forces ───────────────────────────────────────────────────

    /// Gauge force on link angle `θ_μ`(t, x): −d`S_gauge`/dθ.
    ///
    /// For the single plaquette orientation in (1+1)D, each link appears
    /// in exactly two plaquettes (forward and backward).
    fn gauge_force(&self, t: usize, x: usize, mu: usize) -> f64 {
        let mut force = 0.0;
        let nu = 1 - mu; // the other direction

        // Forward plaquette: link appears as U_μ(x)
        {
            let (t_mu, x_mu) = self.neighbor_fwd(t, x, mu);
            let phase = self.link_angle(t, x, mu) + self.link_angle(t_mu, x_mu, nu)
                - self.link_angle(
                    self.neighbor_fwd(t, x, nu).0,
                    self.neighbor_fwd(t, x, nu).1,
                    mu,
                )
                - self.link_angle(t, x, nu);
            force += self.params.beta_pl * phase.sin();
        }

        // Backward plaquette: link appears in the plaquette starting at (x − ν̂)
        {
            let (t_bwd, x_bwd) = self.neighbor_bwd(t, x, nu);
            let (t_bwd_mu, x_bwd_mu) = self.neighbor_fwd(t_bwd, x_bwd, mu);
            let phase = self.link_angle(t_bwd, x_bwd, nu) + self.link_angle(t, x, mu)
                - self.link_angle(t_bwd_mu, x_bwd_mu, nu)
                - self.link_angle(t_bwd, x_bwd, mu);
            force += self.params.beta_pl * phase.sin();
        }

        -force
    }

    /// Higgs force on link angle `θ_μ`(t, x): −d`S_higgs`/dθ.
    ///
    /// d/dθ Re[φ\*(x) e^{iθ} φ(x+μ̂)] = −Im[φ\*(x) e^{iθ} φ(x+μ̂)]
    /// so −d`S_higgs`/dθ = −2κ × Im(`hop`) × `chem_weight`.
    fn higgs_link_force(&self, t: usize, x: usize, mu: usize) -> f64 {
        let (t_fwd, x_fwd) = self.neighbor_fwd(t, x, mu);
        let idx = self.site_index(t, x);
        let idx_fwd = self.site_index(t_fwd, x_fwd);
        let phi = self.higgs[idx];
        let phi_fwd = self.higgs[idx_fwd];
        let u = self.link(t, x, mu);

        let hop = phi.conj() * u * phi_fwd;
        let chem_weight = if mu == 0 { self.params.mu.exp() } else { 1.0 };

        -2.0 * self.params.kappa * chem_weight * hop.im
    }

    /// Force on Higgs field φ(x): −2 dS/dφ* (complex).
    ///
    /// The factor of 2 arises because dp/dt = −(∂S/∂`φ_R` + i ∂S/∂`φ_I`) = −2 ∂S/∂φ\*
    /// when using Wirtinger derivatives with the kinetic energy T = ½|p|².
    fn higgs_field_force(&self, t: usize, x: usize) -> Complex64 {
        let idx = self.site_index(t, x);
        let phi = self.higgs[idx];
        let phi_sq = phi.abs_sq();
        let kappa = self.params.kappa;
        let lambda = self.params.lambda;
        let mu = self.params.mu;

        // ∂S_pot/∂φ* = φ(1 + 2λ(|φ|² − 1))
        let pot_grad = phi.scale((2.0 * lambda).mul_add(phi_sq - 1.0, 1.0));

        // ∂S_kin/∂φ* = −κ Σ_μ [U_μ(x) φ(x+μ̂) + U†_μ(x−μ̂) φ(x−μ̂)]
        let mut kin_neighbors = Complex64::ZERO;
        for mu_dir in 0..2 {
            let (t_fwd, x_fwd) = self.neighbor_fwd(t, x, mu_dir);
            let idx_fwd = self.site_index(t_fwd, x_fwd);
            let u_fwd = self.link(t, x, mu_dir);
            let chem_fwd = if mu_dir == 0 { mu.exp() } else { 1.0 };
            kin_neighbors += (u_fwd * self.higgs[idx_fwd]).scale(chem_fwd);

            let (t_bwd, x_bwd) = self.neighbor_bwd(t, x, mu_dir);
            let idx_bwd = self.site_index(t_bwd, x_bwd);
            let u_bwd = self.link(t_bwd, x_bwd, mu_dir).conj();
            let chem_bwd = if mu_dir == 0 { (-mu).exp() } else { 1.0 };
            kin_neighbors += (u_bwd * self.higgs[idx_bwd]).scale(chem_bwd);
        }

        // −2 ∂S/∂φ* = 2(κ × neighbors − potential_gradient)
        (kin_neighbors.scale(kappa) - pot_grad).scale(2.0)
    }

    // ── HMC ──────────────────────────────────────────────────────────

    /// Run one HMC trajectory.
    ///
    /// Returns `(accepted, delta_h, action_before, action_after)`.
    pub fn hmc_trajectory(&mut self, n_md_steps: usize, dt: f64, seed: &mut u64) -> HmcResult {
        let vol = self.volume();

        // Save old configuration for Metropolis reject
        let old_links = self.links.clone();
        let old_higgs = self.higgs.clone();

        // Random momenta: real scalars for link angles, complex for Higgs
        let mut pi_links: Vec<f64> = (0..vol * 2).map(|_| lcg_gaussian(seed)).collect();
        let mut pi_higgs: Vec<Complex64> = (0..vol)
            .map(|_| Complex64::new(lcg_gaussian(seed), lcg_gaussian(seed)))
            .collect();

        // Initial Hamiltonian
        let ke_init = kinetic_energy(&pi_links, &pi_higgs);
        let s_init = self.total_action();
        let h_init = ke_init + s_init;

        // Leapfrog integration
        // Half-step on momenta
        self.update_momenta(&mut pi_links, &mut pi_higgs, dt / 2.0);

        for step in 0..n_md_steps {
            // Full step on fields
            self.update_links(&pi_links, dt);
            self.update_higgs(&pi_higgs, dt);

            // Full step on momenta (except last)
            if step < n_md_steps - 1 {
                self.update_momenta(&mut pi_links, &mut pi_higgs, dt);
            }
        }

        // Final half-step on momenta
        self.update_momenta(&mut pi_links, &mut pi_higgs, dt / 2.0);

        // Final Hamiltonian
        let ke_final = kinetic_energy(&pi_links, &pi_higgs);
        let s_final = self.total_action();
        let h_final = ke_final + s_final;

        let delta_h = h_final - h_init;

        // Metropolis accept/reject
        let accepted = if delta_h <= 0.0 {
            true
        } else {
            let r = lcg_uniform_f64(seed);
            r < (-delta_h).exp()
        };

        if !accepted {
            self.links = old_links;
            self.higgs = old_higgs;
        }

        HmcResult {
            accepted,
            delta_h,
            action_before: s_init,
            action_after: if accepted { s_final } else { s_init },
            plaquette: self.average_plaquette(),
            higgs_sq: self.average_higgs_sq(),
        }
    }

    /// Update link angles: θ → θ + dt × π.
    fn update_links(&mut self, pi_links: &[f64], dt: f64) {
        for (l, &pi) in self.links.iter_mut().zip(pi_links.iter()) {
            *l += dt * pi;
        }
    }

    /// Update Higgs fields: φ → φ + dt × p.
    fn update_higgs(&mut self, pi_higgs: &[Complex64], dt: f64) {
        for (h, pi) in self.higgs.iter_mut().zip(pi_higgs.iter()) {
            *h += pi.scale(dt);
        }
    }

    /// Update momenta: p → p + dt × (−dS/dq).
    fn update_momenta(&self, pi_links: &mut [f64], pi_higgs: &mut [Complex64], dt: f64) {
        for t in 0..self.nt {
            for x in 0..self.ns {
                for mu in 0..2 {
                    let idx = self.site_index(t, x) * 2 + mu;
                    let f_gauge = self.gauge_force(t, x, mu);
                    let f_higgs = self.higgs_link_force(t, x, mu);
                    pi_links[idx] += dt * (f_gauge + f_higgs);
                }
                let idx = self.site_index(t, x);
                let f = self.higgs_field_force(t, x);
                pi_higgs[idx] += f.scale(dt);
            }
        }
    }

    /// Run multiple HMC trajectories with thermalization.
    pub fn run_hmc(
        &mut self,
        n_therm: usize,
        n_traj: usize,
        n_md_steps: usize,
        dt: f64,
        seed: &mut u64,
    ) -> HmcStatistics {
        // Thermalization
        for _ in 0..n_therm {
            self.hmc_trajectory(n_md_steps, dt, seed);
        }

        let mut results = Vec::with_capacity(n_traj);
        for _ in 0..n_traj {
            results.push(self.hmc_trajectory(n_md_steps, dt, seed));
        }

        let n = results.len() as f64;
        let acceptance = results.iter().filter(|r| r.accepted).count() as f64 / n;
        let avg_plaq = results.iter().map(|r| r.plaquette).sum::<f64>() / n;
        let avg_higgs_sq = results.iter().map(|r| r.higgs_sq).sum::<f64>() / n;
        let avg_delta_h = results.iter().map(|r| r.delta_h.abs()).sum::<f64>() / n;

        HmcStatistics {
            acceptance_rate: acceptance,
            avg_plaquette: avg_plaq,
            avg_higgs_sq,
            avg_abs_delta_h: avg_delta_h,
            trajectories: results,
        }
    }
}

/// Kinetic energy: T = ½ Σ π² + ½ Σ |`p_φ`|².
fn kinetic_energy(pi_links: &[f64], pi_higgs: &[Complex64]) -> f64 {
    let ke_links: f64 = pi_links.iter().map(|p| p * p).sum();
    let ke_higgs: f64 = pi_higgs.iter().map(|p| p.abs_sq()).sum();
    0.5 * (ke_links + ke_higgs)
}

/// Result of a single HMC trajectory.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct HmcResult {
    pub accepted: bool,
    pub delta_h: f64,
    pub action_before: f64,
    pub action_after: f64,
    pub plaquette: f64,
    pub higgs_sq: f64,
}

/// Statistics from a sequence of HMC trajectories.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct HmcStatistics {
    pub acceptance_rate: f64,
    pub avg_plaquette: f64,
    pub avg_higgs_sq: f64,
    pub avg_abs_delta_h: f64,
    pub trajectories: Vec<HmcResult>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_start_plaquette_is_one() {
        let params = AbelianHiggsParams::new(1.0, 0.1, 1.0);
        let lat = U1HiggsLattice::cold_start(4, 4, params);
        let plaq = lat.average_plaquette();
        assert!((plaq - 1.0).abs() < 1e-14, "cold plaquette = {plaq}");
    }

    #[test]
    fn cold_start_gauge_action_is_zero() {
        let params = AbelianHiggsParams::new(2.0, 0.0, 0.0);
        let lat = U1HiggsLattice::cold_start(4, 4, params);
        let s = lat.gauge_action();
        assert!(s.abs() < 1e-14, "cold gauge action = {s}");
    }

    #[test]
    fn cold_start_higgs_sq_is_one() {
        let params = AbelianHiggsParams::new(1.0, 0.1, 1.0);
        let lat = U1HiggsLattice::cold_start(4, 4, params);
        assert!((lat.average_higgs_sq() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn cold_start_polyakov_loop_is_one() {
        let params = AbelianHiggsParams::new(1.0, 0.1, 1.0);
        let lat = U1HiggsLattice::cold_start(4, 4, params);
        assert!((lat.average_polyakov_loop() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn hot_start_plaquette_in_range() {
        let params = AbelianHiggsParams::new(1.0, 0.1, 1.0);
        let mut seed = 42u64;
        let lat = U1HiggsLattice::hot_start(8, 8, params, &mut seed);
        let plaq = lat.average_plaquette();
        assert!((-1.0..=1.0).contains(&plaq), "hot plaquette = {plaq}");
    }

    #[test]
    fn gauge_action_non_negative() {
        let params = AbelianHiggsParams::new(2.0, 0.0, 0.0);
        let mut seed = 42u64;
        let lat = U1HiggsLattice::hot_start(4, 4, params, &mut seed);
        let s = lat.gauge_action();
        assert!(s >= 0.0, "gauge action must be ≥ 0, got {s}");
    }

    #[test]
    fn higgs_potential_minimum_at_unit_modulus() {
        let params = AbelianHiggsParams::new(0.0, 0.0, 5.0);
        let lat = U1HiggsLattice::cold_start(4, 4, params);
        let s = lat.higgs_action();
        // At |φ|=1: V = 1 + λ(1−1)² = 1 per site, kinetic=0 (κ=0)
        let expected = lat.volume() as f64;
        assert!(
            (s - expected).abs() < 1e-12,
            "higgs action = {s}, expected {expected}"
        );
    }

    #[test]
    fn site_index_roundtrip() {
        let params = AbelianHiggsParams::new(1.0, 0.1, 1.0);
        let lat = U1HiggsLattice::cold_start(6, 8, params);
        for t in 0..6 {
            for x in 0..8 {
                let idx = lat.site_index(t, x);
                assert!(idx < lat.volume());
            }
        }
    }

    #[test]
    fn neighbor_periodicity() {
        let params = AbelianHiggsParams::new(1.0, 0.1, 1.0);
        let lat = U1HiggsLattice::cold_start(4, 4, params);
        assert_eq!(lat.neighbor_fwd(3, 2, 0), (0, 2));
        assert_eq!(lat.neighbor_fwd(2, 3, 1), (2, 0));
        assert_eq!(lat.neighbor_bwd(0, 2, 0), (3, 2));
        assert_eq!(lat.neighbor_bwd(2, 0, 1), (2, 3));
    }

    #[test]
    fn hmc_cold_start_small_delta_h() {
        let params = AbelianHiggsParams::new(2.0, 0.5, 1.0);
        let mut lat = U1HiggsLattice::cold_start(4, 4, params);
        let mut seed = 42u64;
        let result = lat.hmc_trajectory(10, 0.05, &mut seed);
        assert!(
            result.delta_h.abs() < 5.0,
            "|ΔH| = {} too large",
            result.delta_h.abs()
        );
    }

    #[test]
    fn hmc_delta_h_decreases_with_smaller_dt() {
        let params = AbelianHiggsParams::new(2.0, 0.5, 1.0);
        let mut lat1 = U1HiggsLattice::cold_start(4, 4, params.clone());
        let mut lat2 = U1HiggsLattice::cold_start(4, 4, params);
        let mut seed1 = 42u64;
        let mut seed2 = 42u64;
        let r1 = lat1.hmc_trajectory(10, 0.1, &mut seed1);
        let r2 = lat2.hmc_trajectory(20, 0.05, &mut seed2);
        assert!(
            r2.delta_h.abs() <= r1.delta_h.abs() * 2.0,
            "smaller dt should give smaller |ΔH|: {:.4} vs {:.4}",
            r2.delta_h.abs(),
            r1.delta_h.abs()
        );
    }

    #[test]
    fn hmc_thermalization_produces_physical_plaquette() {
        let params = AbelianHiggsParams::new(4.0, 0.5, 1.0);
        let mut lat = U1HiggsLattice::hot_start(8, 8, params, &mut 42u64);
        let stats = lat.run_hmc(20, 50, 10, 0.1, &mut 123u64);
        assert!(
            stats.acceptance_rate > 0.1,
            "acceptance too low: {:.2}%",
            stats.acceptance_rate * 100.0
        );
        assert!(
            stats.avg_plaquette > 0.0 && stats.avg_plaquette < 1.0,
            "plaquette = {:.4}",
            stats.avg_plaquette
        );
    }

    #[test]
    fn strong_coupling_low_plaquette() {
        let params = AbelianHiggsParams::new(0.5, 0.1, 1.0);
        let mut lat = U1HiggsLattice::hot_start(8, 8, params, &mut 42u64);
        let stats = lat.run_hmc(20, 30, 10, 0.1, &mut 99u64);
        assert!(
            stats.avg_plaquette < 0.8,
            "strong coupling should have low plaquette: {:.4}",
            stats.avg_plaquette
        );
    }

    #[test]
    fn weak_coupling_high_plaquette() {
        let params = AbelianHiggsParams::new(8.0, 0.1, 1.0);
        let mut lat = U1HiggsLattice::cold_start(8, 8, params);
        let stats = lat.run_hmc(20, 30, 10, 0.1, &mut 99u64);
        assert!(
            stats.avg_plaquette > 0.5,
            "weak coupling should have high plaquette: {:.4}",
            stats.avg_plaquette
        );
    }

    #[test]
    fn large_kappa_higgs_condensation() {
        let params = AbelianHiggsParams::new(2.0, 2.0, 1.0);
        let mut lat = U1HiggsLattice::hot_start(8, 8, params, &mut 42u64);
        let stats = lat.run_hmc(30, 50, 10, 0.05, &mut 77u64);
        assert!(
            stats.avg_higgs_sq > 0.3,
            "large κ should promote Higgs condensation: ⟨|φ|²⟩ = {:.4}",
            stats.avg_higgs_sq
        );
    }

    #[test]
    fn chemical_potential_zero_symmetric() {
        let params = AbelianHiggsParams::new(2.0, 0.5, 1.0);
        let mut lat = U1HiggsLattice::cold_start(4, 4, params);
        let s_zero = lat.total_action();
        lat.params.mu = 0.0;
        let s_still_zero = lat.total_action();
        assert!((s_zero - s_still_zero).abs() < 1e-14);
    }
}
