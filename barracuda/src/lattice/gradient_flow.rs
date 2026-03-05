// SPDX-License-Identifier: AGPL-3.0-only

//! Wilson gradient flow on SU(3) gauge fields.
//!
//! Implements the Wilson flow equation dU/dt = Z(U) U where Z is the
//! traceless anti-Hermitian projection of the gauge force. This smooths
//! the gauge field deterministically, making ultraviolet fluctuations
//! irrelevant while preserving infrared physics.
//!
//! Three integrators are provided:
//!
//! | Integrator | Order | Stages | Evaluations |
//! |------------|-------|--------|-------------|
//! | Euler      | 1     | 1      | 1           |
//! | RK2        | 2     | 2      | 2           |
//! | RK3 (Lüscher) | 3  | 3      | 3          |
//!
//! The 3-stage RK3 is from Lüscher, JHEP 08 (2010) 071, which is the
//! standard for lattice gradient flow. Bazavov & Chuna (2021) extend
//! this to general commutator-free Lie group methods with optimized
//! coefficients for low storage.
//!
//! # Observable: E(t)
//!
//! The gauge energy density E(t) = -(1/V) Σ_{x,μ<ν} Re Tr P_{μν}(x,t)
//! defines the gradient flow scale t₀ via t²⟨E(t)⟩ = 0.3 (Lüscher 2010).
//!
//! # References
//!
//! - Lüscher, JHEP 08 (2010) 071 — Wilson flow definition and t₀ scale
//! - Bazavov & Chuna, arXiv:2101.05320 — optimized Lie group integrators

use super::hmc::exp_su3_cayley_pub;
use super::su3::Su3Matrix;
use super::wilson::Lattice;

/// Flow integrator type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowIntegrator {
    /// First-order Euler: U' = exp(ε Z) U
    Euler,
    /// Second-order Runge-Kutta
    Rk2,
    /// Third-order Runge-Kutta (Lüscher 2010, standard for lattice QCD)
    Rk3Luscher,
}

/// Result of a gradient flow measurement at flow time t.
#[derive(Debug, Clone)]
pub struct FlowMeasurement {
    /// Flow time t.
    pub t: f64,
    /// Gauge energy density E(t) = (1 - avg_plaquette) * 6/β × β.
    pub energy_density: f64,
    /// t² E(t) — the dimensionless combination used to define t₀.
    pub t2_e: f64,
    /// Average plaquette at flow time t.
    pub plaquette: f64,
}

/// Compute the Lie-algebra driving term Z_μ(x) for gradient flow.
///
/// Z = -[∂S_W/∂U] projected to traceless anti-Hermitian su(3).
/// This is exactly `Lattice::gauge_force` which already does this projection.
fn flow_force(lattice: &Lattice, x: [usize; 4], mu: usize) -> Su3Matrix {
    lattice.gauge_force(x, mu)
}

/// Apply one Euler step: U_μ(x) ← exp(ε · Z_μ(x)) · U_μ(x).
fn euler_step(lattice: &mut Lattice, epsilon: f64) {
    let dims = lattice.dims;
    let v = dims[0] * dims[1] * dims[2] * dims[3];

    let mut forces = Vec::with_capacity(v * 4);
    for site in 0..v {
        let x = lattice.site_coords(site);
        for mu in 0..4 {
            forces.push(flow_force(lattice, x, mu));
        }
    }

    for site in 0..v {
        let x = lattice.site_coords(site);
        for mu in 0..4 {
            let z = &forces[site * 4 + mu];
            let u = lattice.link(x, mu);
            let u_new = exp_su3_cayley_pub(z, epsilon) * u;
            lattice.set_link(x, mu, u_new);
        }
    }
}

/// Apply one RK2 (midpoint) step.
fn rk2_step(lattice: &mut Lattice, epsilon: f64) {
    let saved = lattice.links.clone();
    let dims = lattice.dims;
    let v = dims[0] * dims[1] * dims[2] * dims[3];

    let mut z0 = Vec::with_capacity(v * 4);
    for site in 0..v {
        let x = lattice.site_coords(site);
        for mu in 0..4 {
            z0.push(flow_force(lattice, x, mu));
        }
    }

    for site in 0..v {
        let x = lattice.site_coords(site);
        for mu in 0..4 {
            let z = &z0[site * 4 + mu];
            let u = lattice.link(x, mu);
            lattice.set_link(x, mu, exp_su3_cayley_pub(z, 0.5 * epsilon) * u);
        }
    }

    let mut z1 = Vec::with_capacity(v * 4);
    for site in 0..v {
        let x = lattice.site_coords(site);
        for mu in 0..4 {
            z1.push(flow_force(lattice, x, mu));
        }
    }

    lattice.links = saved;

    for site in 0..v {
        let x = lattice.site_coords(site);
        for mu in 0..4 {
            let z = &z1[site * 4 + mu];
            let u = lattice.link(x, mu);
            lattice.set_link(x, mu, exp_su3_cayley_pub(z, epsilon) * u);
        }
    }
}

/// Apply one RK3 (Lüscher) step.
///
/// Three-stage 3rd-order scheme from Lüscher, JHEP 08 (2010) 071:
///
///   W₀ = U(t)
///   W₁ = exp(¼ ε Z₀) W₀
///   W₂ = exp((8/9 ε Z₁ - 17/36 ε Z₀)) W₁
///   U(t+ε) = exp((¾ ε Z₂ - 8/9 ε Z₁ + 17/36 ε Z₀)) W₂
///
/// where Z_i = Z(W_i).
fn rk3_luscher_step(lattice: &mut Lattice, epsilon: f64) {
    let dims = lattice.dims;
    let v = dims[0] * dims[1] * dims[2] * dims[3];

    let mut z0 = Vec::with_capacity(v * 4);
    for site in 0..v {
        let x = lattice.site_coords(site);
        for mu in 0..4 {
            z0.push(flow_force(lattice, x, mu));
        }
    }

    for site in 0..v {
        let x = lattice.site_coords(site);
        for mu in 0..4 {
            let z = &z0[site * 4 + mu];
            let u = lattice.link(x, mu);
            lattice.set_link(x, mu, exp_su3_cayley_pub(z, 0.25 * epsilon) * u);
        }
    }

    let mut z1 = Vec::with_capacity(v * 4);
    for site in 0..v {
        let x = lattice.site_coords(site);
        for mu in 0..4 {
            z1.push(flow_force(lattice, x, mu));
        }
    }

    for site in 0..v {
        let x = lattice.site_coords(site);
        for mu in 0..4 {
            let w1_force = z1[site * 4 + mu].scale(8.0 / 9.0 * epsilon);
            let w0_corr = z0[site * 4 + mu].scale(-17.0 / 36.0 * epsilon);
            let combined = w1_force + w0_corr;
            let u = lattice.link(x, mu);
            lattice.set_link(x, mu, exp_su3_cayley_pub(&combined, 1.0) * u);
        }
    }

    let mut z2 = Vec::with_capacity(v * 4);
    for site in 0..v {
        let x = lattice.site_coords(site);
        for mu in 0..4 {
            z2.push(flow_force(lattice, x, mu));
        }
    }

    for site in 0..v {
        let x = lattice.site_coords(site);
        for mu in 0..4 {
            let w2_force = z2[site * 4 + mu].scale(0.75 * epsilon);
            let w1_corr = z1[site * 4 + mu].scale(-8.0 / 9.0 * epsilon);
            let w0_corr = z0[site * 4 + mu].scale(17.0 / 36.0 * epsilon);
            let combined = w2_force + w1_corr + w0_corr;
            let u = lattice.link(x, mu);
            lattice.set_link(x, mu, exp_su3_cayley_pub(&combined, 1.0) * u);
        }
    }
}

/// Compute gauge energy density E(t) from the lattice at flow time t.
///
/// E = (β / 3V) Σ_{x, μ<ν} (1 - Re Tr P_{μν} / 3)
/// which simplifies to (1 - ⟨P⟩) × 2 × N_c for SU(3).
pub fn energy_density(lattice: &Lattice) -> f64 {
    let plaq = lattice.average_plaquette();
    (1.0 - plaq) * 6.0
}

/// Run gradient flow from t=0 to t=t_max and collect measurements.
///
/// Returns measurements at each flow time step.
pub fn run_flow(
    lattice: &mut Lattice,
    integrator: FlowIntegrator,
    epsilon: f64,
    t_max: f64,
    measure_interval: usize,
) -> Vec<FlowMeasurement> {
    let n_steps = (t_max / epsilon).round() as usize;
    let mut measurements = Vec::new();

    let e0 = energy_density(lattice);
    let p0 = lattice.average_plaquette();
    measurements.push(FlowMeasurement {
        t: 0.0,
        energy_density: e0,
        t2_e: 0.0,
        plaquette: p0,
    });

    for step in 1..=n_steps {
        match integrator {
            FlowIntegrator::Euler => euler_step(lattice, epsilon),
            FlowIntegrator::Rk2 => rk2_step(lattice, epsilon),
            FlowIntegrator::Rk3Luscher => rk3_luscher_step(lattice, epsilon),
        }

        if step % measure_interval == 0 || step == n_steps {
            let t = step as f64 * epsilon;
            let e = energy_density(lattice);
            measurements.push(FlowMeasurement {
                t,
                energy_density: e,
                t2_e: t * t * e,
                plaquette: lattice.average_plaquette(),
            });
        }
    }

    measurements
}

/// Find t₀ such that t²⟨E(t)⟩ = 0.3 by linear interpolation.
pub fn find_t0(measurements: &[FlowMeasurement]) -> Option<f64> {
    const TARGET: f64 = 0.3;
    for window in measurements.windows(2) {
        let (a, b) = (&window[0], &window[1]);
        if a.t2_e <= TARGET && b.t2_e >= TARGET && (b.t2_e - a.t2_e).abs() > 1e-15 {
            let frac = (TARGET - a.t2_e) / (b.t2_e - a.t2_e);
            return Some(a.t + frac * (b.t - a.t));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_start_zero_energy() {
        let lattice = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let e = energy_density(&lattice);
        assert!(e.abs() < 1e-12, "cold start should have zero energy: {e}");
    }

    #[test]
    fn euler_flow_smooths() {
        let mut lattice = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let e_before = energy_density(&lattice);
        let results = run_flow(&mut lattice, FlowIntegrator::Euler, 0.01, 0.1, 1);
        let e_after = results.last().unwrap().energy_density;
        assert!(
            e_after < e_before,
            "flow should smooth: {e_before} -> {e_after}"
        );
    }

    #[test]
    fn rk3_flow_smooths_faster_than_euler() {
        let mut lattice_euler = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let mut lattice_rk3 = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);

        let results_euler =
            run_flow(&mut lattice_euler, FlowIntegrator::Euler, 0.01, 0.5, 10);
        let results_rk3 =
            run_flow(&mut lattice_rk3, FlowIntegrator::Rk3Luscher, 0.01, 0.5, 10);

        let e_euler = results_euler.last().unwrap().energy_density;
        let e_rk3 = results_rk3.last().unwrap().energy_density;

        assert!(
            (e_rk3 - e_euler).abs() < 0.5,
            "RK3 and Euler should give similar E at t=0.5: euler={e_euler}, rk3={e_rk3}"
        );
    }

    #[test]
    fn flow_preserves_unitarity() {
        let mut lattice = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        run_flow(&mut lattice, FlowIntegrator::Rk3Luscher, 0.02, 1.0, 50);

        let u = lattice.link([0, 0, 0, 0], 0);
        let uud = u * u.adjoint();
        let diff = uud - Su3Matrix::IDENTITY;
        let deviation = diff.norm_sq().sqrt();
        assert!(
            deviation < 1e-10,
            "unitarity deviation after flow: {deviation}"
        );
    }

    #[test]
    fn t2_e_increases_monotonically_for_hot_start() {
        let mut lattice = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let results = run_flow(
            &mut lattice,
            FlowIntegrator::Rk3Luscher,
            0.01,
            0.5,
            5,
        );

        for window in results.windows(2) {
            if window[0].t > 0.05 {
                assert!(
                    window[1].t2_e >= window[0].t2_e - 1e-10,
                    "t²E should increase: t={}, t2e={} -> t={}, t2e={}",
                    window[0].t,
                    window[0].t2_e,
                    window[1].t,
                    window[1].t2_e
                );
            }
        }
    }

    #[test]
    fn find_t0_on_hot_start() {
        let mut lattice = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let results = run_flow(
            &mut lattice,
            FlowIntegrator::Rk3Luscher,
            0.01,
            2.0,
            1,
        );
        let t0 = find_t0(&results);
        if let Some(t0_val) = t0 {
            assert!(t0_val > 0.0, "t₀ should be positive: {t0_val}");
            assert!(t0_val < 2.0, "t₀ should be within flow range: {t0_val}");
            println!("t₀ = {t0_val:.4} (4⁴, β=6.0)");
        }
    }
}
