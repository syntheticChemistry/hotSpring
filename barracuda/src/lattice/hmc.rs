// SPDX-License-Identifier: AGPL-3.0-only

//! Hybrid Monte Carlo (HMC) integrator for pure gauge SU(3).
//!
//! HMC is the standard algorithm for lattice QCD gauge field generation.
//! It combines molecular dynamics evolution with a Metropolis accept/reject
//! step to guarantee detailed balance.
//!
//! The algorithm:
//!   1. Draw random momenta P from Gaussian distribution
//!   2. Compute `H_old` = T(P) + S(U)
//!   3. Leapfrog integrate (U, P) for `N_md` steps
//!   4. Compute `H_new` = T(P') + S(U')
//!   5. Accept with probability min(1, exp(`H_old` - `H_new`))
//!
//! The kinetic energy T(P) = -Tr(P²)/2 summed over all links.
//!
//! # References
//!
//! - Duane et al., PLB 195, 216 (1987) — original HMC
//! - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 8

use super::complex_f64::Complex64;
use super::su3::Su3Matrix;
use super::wilson::Lattice;
use crate::tolerances::OMELYAN_LAMBDA;

/// HMC integrator type.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum IntegratorType {
    /// Standard leapfrog: O(dt²) shadow Hamiltonian error.
    #[default]
    Leapfrog,
    /// Omelyan 2MN: O(dt⁴) shadow Hamiltonian error (same force count per step).
    Omelyan,
}

/// HMC configuration parameters.
#[derive(Clone, Debug)]
pub struct HmcConfig {
    /// Number of MD steps per trajectory
    pub n_md_steps: usize,
    /// MD step size (dt)
    pub dt: f64,
    /// Random seed for momentum refresh
    pub seed: u64,
    /// Integrator type (default: Leapfrog for backward compat)
    pub integrator: IntegratorType,
}

impl Default for HmcConfig {
    fn default() -> Self {
        Self {
            n_md_steps: 20,
            dt: 0.05,
            seed: 42,
            integrator: IntegratorType::Leapfrog,
        }
    }
}

/// HMC trajectory result.
#[derive(Clone, Debug)]
pub struct HmcResult {
    pub accepted: bool,
    pub delta_h: f64,
    pub action_before: f64,
    pub action_after: f64,
    pub plaquette: f64,
}

/// Run one HMC trajectory.
///
/// Modifies `lattice` in place if the trajectory is accepted.
/// Returns trajectory diagnostics.
pub fn hmc_trajectory(lattice: &mut Lattice, config: &mut HmcConfig) -> HmcResult {
    let vol = lattice.volume();

    // Save old configuration
    let old_links = lattice.links.clone();
    let action_before = lattice.wilson_action();

    // Generate random momenta (su(3) Lie algebra elements)
    let mut momenta = vec![Su3Matrix::ZERO; vol * 4];
    for p in &mut momenta {
        *p = Su3Matrix::random_algebra(&mut config.seed);
    }

    let kinetic_before = kinetic_energy(&momenta);
    let h_old = action_before + kinetic_before;

    // MD integration
    match config.integrator {
        IntegratorType::Leapfrog => leapfrog(lattice, &mut momenta, config.n_md_steps, config.dt),
        IntegratorType::Omelyan => omelyan(lattice, &mut momenta, config.n_md_steps, config.dt),
    }

    let action_after = lattice.wilson_action();
    let kinetic_after = kinetic_energy(&momenta);
    let h_new = action_after + kinetic_after;

    let delta_h = h_new - h_old;

    // Metropolis accept/reject
    let accept = if delta_h <= 0.0 {
        true
    } else {
        let r = super::constants::lcg_uniform_f64(&mut config.seed);
        r < (-delta_h).exp()
    };

    if !accept {
        lattice.links = old_links;
    }

    let plaquette = lattice.average_plaquette();

    HmcResult {
        accepted: accept,
        delta_h,
        action_before,
        action_after,
        plaquette,
    }
}

/// Omelyan (2MN) integration for (U, P) on the SU(3) manifold.
///
/// Per step: P→P+λ·dt·F, U→U+dt/2·P, P→P+(1-2λ)·dt·F, U→U+dt/2·P, P→P+λ·dt·F.
/// Achieves O(dt⁴) shadow Hamiltonian error vs O(dt²) for leapfrog.
fn omelyan(lattice: &mut Lattice, momenta: &mut [Su3Matrix], n_steps: usize, dt: f64) {
    let vol = lattice.volume();
    let lam = OMELYAN_LAMBDA;
    let half_dt = 0.5 * dt;

    for _step in 0..n_steps {
        // P -> P + λ·dt·F(U)
        update_momenta(lattice, momenta, lam * dt);

        // U -> U + dt/2·P
        for idx in 0..vol {
            let x = lattice.site_coords(idx);
            for mu in 0..4 {
                let p = momenta[idx * 4 + mu];
                let u = lattice.link(x, mu);
                let exp_p = exp_su3_cayley(&p, half_dt);
                let new_u = (exp_p * u).reunitarize();
                lattice.set_link(x, mu, new_u);
            }
        }

        // P -> P + (1-2λ)·dt·F(U)
        update_momenta(lattice, momenta, (1.0 - 2.0 * lam) * dt);

        // U -> U + dt/2·P
        for idx in 0..vol {
            let x = lattice.site_coords(idx);
            for mu in 0..4 {
                let p = momenta[idx * 4 + mu];
                let u = lattice.link(x, mu);
                let exp_p = exp_su3_cayley(&p, half_dt);
                let new_u = (exp_p * u).reunitarize();
                lattice.set_link(x, mu, new_u);
            }
        }

        // P -> P + λ·dt·F(U)
        update_momenta(lattice, momenta, lam * dt);
    }
}

/// Leapfrog integration for (U, P) on the SU(3) manifold.
///
/// Half-step P, full-step U, half-step P.
fn leapfrog(lattice: &mut Lattice, momenta: &mut [Su3Matrix], n_steps: usize, dt: f64) {
    let vol = lattice.volume();
    let half_dt = 0.5 * dt;

    // Initial half-step for momenta
    update_momenta(lattice, momenta, half_dt);

    for step in 0..n_steps {
        // Full step for links: U' = exp(i dt P) U
        for idx in 0..vol {
            let x = lattice.site_coords(idx);
            for mu in 0..4 {
                let p = momenta[idx * 4 + mu];
                let u = lattice.link(x, mu);
                let exp_p = exp_su3_cayley(&p, dt);
                let new_u = (exp_p * u).reunitarize();
                lattice.set_link(x, mu, new_u);
            }
        }

        // Full step for momenta (except last step → half-step)
        let p_dt = if step < n_steps - 1 { dt } else { half_dt };
        update_momenta(lattice, momenta, p_dt);
    }
}

/// Update momenta by the gauge force: P' = P + dt × F
fn update_momenta(lattice: &Lattice, momenta: &mut [Su3Matrix], dt: f64) {
    let vol = lattice.volume();
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        for mu in 0..4 {
            let force = lattice.gauge_force(x, mu);
            momenta[idx * 4 + mu] = momenta[idx * 4 + mu] + force.scale(dt);
        }
    }
}

/// Compute exp(dt × P) for a su(3) algebra element P via Cayley transform.
///
/// Uses (I + X/2)(I - X/2)^{-1} which is exactly unitary when X is
/// anti-Hermitian, eliminating the unitarity drift of Taylor approximations.
fn exp_su3_cayley(p: &Su3Matrix, dt: f64) -> Su3Matrix {
    let half = p.scale(dt * 0.5);
    let plus = Su3Matrix::IDENTITY + half;
    let minus = Su3Matrix::IDENTITY - half;
    let inv = su3_inverse(&minus);
    (plus * inv).reunitarize()
}

/// Exact inverse of a 3×3 complex matrix via cofactor expansion.
fn su3_inverse(a: &Su3Matrix) -> Su3Matrix {
    let m = &a.m;

    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    let inv_det = det.inv();

    let c10 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = m[0][1] * m[2][0] - m[0][0] * m[2][1];

    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = m[0][2] * m[1][0] - m[0][0] * m[1][2];
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];

    let zero = Complex64::ZERO;
    let mut result = Su3Matrix { m: [[zero; 3]; 3] };
    result.m[0][0] = c00 * inv_det;
    result.m[0][1] = c10 * inv_det;
    result.m[0][2] = c20 * inv_det;
    result.m[1][0] = c01 * inv_det;
    result.m[1][1] = c11 * inv_det;
    result.m[1][2] = c21 * inv_det;
    result.m[2][0] = c02 * inv_det;
    result.m[2][1] = c12 * inv_det;
    result.m[2][2] = c22 * inv_det;
    result
}

/// Kinetic energy T(P) = -(1/2) Σ Tr(P²)
fn kinetic_energy(momenta: &[Su3Matrix]) -> f64 {
    let mut t = 0.0;
    for p in momenta {
        let p2 = *p * *p;
        t -= 0.5 * p2.re_trace();
    }
    t
}

/// Run multiple HMC trajectories and collect statistics.
pub fn run_hmc(
    lattice: &mut Lattice,
    n_trajectories: usize,
    n_thermalization: usize,
    config: &mut HmcConfig,
) -> HmcStatistics {
    let mut plaquettes = Vec::new();
    let mut acceptance_count = 0usize;
    let mut delta_h_values = Vec::new();

    for traj in 0..(n_thermalization + n_trajectories) {
        let result = hmc_trajectory(lattice, config);

        if traj >= n_thermalization {
            plaquettes.push(result.plaquette);
            delta_h_values.push(result.delta_h);
            if result.accepted {
                acceptance_count += 1;
            }
        }

        if traj % 10 == 0 || traj == n_thermalization + n_trajectories - 1 {
            println!(
                "    traj {traj}: plaq={:.6}, ΔH={:.4e}, {}",
                result.plaquette,
                result.delta_h,
                if result.accepted { "ACC" } else { "REJ" }
            );
        }
    }

    let n = plaquettes.len() as f64;
    let mean_plaq = plaquettes.iter().sum::<f64>() / n.max(1.0);
    let var_plaq = plaquettes
        .iter()
        .map(|&p| (p - mean_plaq).powi(2))
        .sum::<f64>()
        / (n - 1.0).max(1.0);

    let mean_dh = delta_h_values.iter().sum::<f64>() / n.max(1.0);
    let acceptance_rate = acceptance_count as f64 / n_trajectories as f64;

    HmcStatistics {
        mean_plaquette: mean_plaq,
        std_plaquette: var_plaq.sqrt(),
        acceptance_rate,
        mean_delta_h: mean_dh,
        n_trajectories,
        plaquette_history: plaquettes,
    }
}

/// Statistics from an HMC run.
#[derive(Clone, Debug)]
pub struct HmcStatistics {
    pub mean_plaquette: f64,
    pub std_plaquette: f64,
    pub acceptance_rate: f64,
    pub mean_delta_h: f64,
    pub n_trajectories: usize,
    pub plaquette_history: Vec<f64>,
}

/// Public access to the Cayley matrix exponential for use by `pseudofermion`.
pub fn exp_su3_cayley_pub(p: &Su3Matrix, dt: f64) -> Su3Matrix {
    exp_su3_cayley(p, dt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmc_cold_start_preserves_order() {
        let mut lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let mut config = HmcConfig {
            n_md_steps: 10,
            dt: 0.01,
            seed: 42,
            ..Default::default()
        };

        let result = hmc_trajectory(&mut lat, &mut config);
        assert!(
            result.plaquette > 0.9,
            "cold start with small dt should stay near ordered: plaq={}",
            result.plaquette
        );
    }

    #[test]
    fn hmc_delta_h_decreases_with_smaller_dt() {
        // Verify |ΔH| ~ O(dt²) for the leapfrog integrator.
        // Same total trajectory τ=0.1: coarse dt=0.02 (5 steps) vs fine dt=0.01 (10 steps).
        // Hot start ensures nontrivial forces.
        let big_dt_result = {
            let mut lat = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
            let mut config = HmcConfig {
                n_md_steps: 5,
                dt: 0.02,
                seed: 42,
                ..Default::default()
            };
            hmc_trajectory(&mut lat, &mut config)
        };

        let small_dt_result = {
            let mut lat = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
            let mut config = HmcConfig {
                n_md_steps: 10,
                dt: 0.01,
                seed: 42,
                ..Default::default()
            };
            hmc_trajectory(&mut lat, &mut config)
        };

        assert!(
            small_dt_result.delta_h.abs() < big_dt_result.delta_h.abs(),
            "smaller dt should give smaller |ΔH|: {:.4e} vs {:.4e}",
            small_dt_result.delta_h.abs(),
            big_dt_result.delta_h.abs()
        );
    }

    #[test]
    fn exp_su3_cayley_identity() {
        let p = Su3Matrix::ZERO;
        let e = exp_su3_cayley(&p, 1.0);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (e.m[i][j].re - expected).abs() < 1e-14,
                    "exp(0) should be identity"
                );
            }
        }
    }

    #[test]
    fn kinetic_energy_zero_for_zero_momenta() {
        let momenta = vec![Su3Matrix::ZERO; 100];
        let t = kinetic_energy(&momenta);
        assert!(t.abs() < 1e-14, "T(0) should be 0");
    }

    #[test]
    fn omelyan_vs_leapfrog_delta_h() {
        // Same trajectory time τ: verify |ΔH_omelyan| < |ΔH_leapfrog|
        // Omelyan has O(dt⁴) shadow Hamiltonian error vs O(dt²) for leapfrog.
        let tau = 0.2;
        let n_steps = 10;
        let dt = tau / (n_steps as f64);

        let leapfrog_result = {
            let mut lat = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
            let mut config = HmcConfig {
                n_md_steps: n_steps,
                dt,
                seed: 42,
                integrator: IntegratorType::Leapfrog,
                ..Default::default()
            };
            hmc_trajectory(&mut lat, &mut config)
        };

        let omelyan_result = {
            let mut lat = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
            let mut config = HmcConfig {
                n_md_steps: n_steps,
                dt,
                seed: 42,
                integrator: IntegratorType::Omelyan,
                ..Default::default()
            };
            hmc_trajectory(&mut lat, &mut config)
        };

        assert!(
            omelyan_result.delta_h.abs() < leapfrog_result.delta_h.abs(),
            "|ΔH| Omelyan ({:.4e}) should be smaller than leapfrog ({:.4e})",
            omelyan_result.delta_h.abs(),
            leapfrog_result.delta_h.abs()
        );
    }

    #[test]
    fn omelyan_reversibility() {
        // Forward N steps + flip momenta + forward N steps + flip ≈ identity.
        // Verifies the Omelyan integrator is time-reversible.
        let mut lat = Lattice::hot_start([4, 4, 4, 4], 6.0, 77);
        let vol = lat.volume();
        let n_steps = 5;
        let dt = 0.02;

        let links_initial = lat.links.clone();
        let mut momenta = vec![Su3Matrix::ZERO; vol * 4];
        for p in &mut momenta {
            *p = Su3Matrix::random_algebra(&mut 77u64);
        }
        let momenta_initial: Vec<Su3Matrix> = momenta.clone();

        // Forward
        omelyan(&mut lat, &mut momenta, n_steps, dt);

        // Flip momenta (time reversal)
        for p in &mut momenta {
            *p = p.scale(-1.0);
        }

        // Forward again (equivalent to backward evolution)
        omelyan(&mut lat, &mut momenta, n_steps, dt);

        // Flip back
        for p in &mut momenta {
            *p = p.scale(-1.0);
        }

        // Compare links
        let mut max_link_diff = 0.0_f64;
        for (a, b) in lat.links.iter().zip(links_initial.iter()) {
            for i in 0..3 {
                for j in 0..3 {
                    let d = (a.m[i][j] - b.m[i][j]).abs_sq();
                    if d > max_link_diff {
                        max_link_diff = d;
                    }
                }
            }
        }
        assert!(
            max_link_diff < 1e-10,
            "links should return to initial (max diff²={max_link_diff:.4e})"
        );

        // Compare momenta
        let mut max_p_diff = 0.0_f64;
        for (a, b) in momenta.iter().zip(momenta_initial.iter()) {
            for i in 0..3 {
                for j in 0..3 {
                    let d = (a.m[i][j] - b.m[i][j]).abs_sq();
                    if d > max_p_diff {
                        max_p_diff = d;
                    }
                }
            }
        }
        assert!(
            max_p_diff < 1e-10,
            "momenta should return to initial (max diff²={max_p_diff:.4e})"
        );
    }

    #[test]
    fn omelyan_acceptance_rate() {
        // 30 trajectories on 4⁴, verify acceptance > 70%.
        let mut lat = Lattice::hot_start([4, 4, 4, 4], 6.0, 123);
        let mut config = HmcConfig {
            n_md_steps: 10,
            dt: 0.05,
            seed: 456,
            integrator: IntegratorType::Omelyan,
            ..Default::default()
        };

        let mut accepted = 0usize;
        for _ in 0..30 {
            let result = hmc_trajectory(&mut lat, &mut config);
            if result.accepted {
                accepted += 1;
            }
        }
        let rate = accepted as f64 / 30.0;
        assert!(
            rate > 0.70,
            "Omelyan acceptance rate should be > 70%, got {:.1}%",
            rate * 100.0
        );
    }

    #[test]
    fn hmc_trajectory_determinism() {
        let results: Vec<_> = (0..2)
            .map(|_| {
                let mut lat = Lattice::hot_start([4, 4, 4, 4], 5.5, 42);
                let mut cfg = HmcConfig {
                    n_md_steps: 10,
                    dt: 0.02,
                    seed: 42,
                    ..Default::default()
                };
                let r = hmc_trajectory(&mut lat, &mut cfg);
                (
                    r.plaquette,
                    r.delta_h,
                    r.accepted,
                    lat.average_polyakov_loop(),
                )
            })
            .collect();
        assert!(
            (results[0].0 - results[1].0).abs() < f64::EPSILON,
            "plaquette must be identical across runs"
        );
        assert!(
            (results[0].1 - results[1].1).abs() < f64::EPSILON,
            "delta_h must be identical across runs"
        );
        assert_eq!(results[0].2, results[1].2, "acceptance must match");
        assert!(
            (results[0].3 - results[1].3).abs() < f64::EPSILON,
            "Polyakov loop must be identical"
        );
    }
}
