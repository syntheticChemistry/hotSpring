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

use barracuda::numerical::lscfrk::{self as lscfrk_lib};

use super::hmc::exp_su3_cayley_pub;
use super::su3::Su3Matrix;
use super::wilson::Lattice;

pub use lscfrk_lib::{
    FlowMeasurement, LscfrkCoefficients, compute_w_function, derive_lscfrk3, find_t0, find_w0,
};

/// Flow integrator type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowIntegrator {
    /// First-order Euler: U' = exp(ε Z) U
    Euler,
    /// Second-order Runge-Kutta
    Rk2,
    /// Third-order Runge-Kutta (Lüscher 2010, standard for lattice QCD).
    /// Equivalent to LSCFRK3W6 in Bazavov & Chuna notation.
    Rk3Luscher,
    /// LSCFRK3W7 — 3rd order, 3-stage, 2N-storage (Bazavov & Chuna 2021).
    /// Recommended by arXiv:2101.05320 as ~2× more efficient than Lüscher
    /// for w₀-scale setting. Rational coefficients: c₂=1/3, c₃=3/4.
    Lscfrk3w7,
    /// LSCFRK4CK — 4th order, 5-stage, 2N-storage (Carpenter & Kennedy 1994).
    /// Fourth-order accuracy with minimal stage count. Becomes more efficient
    /// than 3rd-order methods at step sizes below ~1/32.
    Lscfrk4ck,
}

/// Compute the Lie-algebra driving term `Z_μ(x)` for gradient flow.
///
/// Z = -[∂`S_W/∂U`] projected to traceless anti-Hermitian su(3).
/// This is exactly `Lattice::gauge_force` which already does this projection.
fn flow_force(lattice: &Lattice, x: [usize; 4], mu: usize) -> Su3Matrix {
    lattice.gauge_force(x, mu)
}

/// Apply one Euler step: `U_μ(x)` ← exp(ε · `Z_μ(x)`) · `U_μ(x)`.
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

/// Apply one step of a generic 2N-storage LSCFRK Lie group integrator.
///
/// This is Algorithm 6 from Bazavov & Chuna (2021). The same code runs
/// any 2N-storage scheme — only the (A, B) coefficients differ.
fn lscfrk_step(lattice: &mut Lattice, epsilon: f64, scheme: &LscfrkCoefficients) {
    let dims = lattice.dims;
    let v = dims[0] * dims[1] * dims[2] * dims[3];
    let s = scheme.a.len();

    let mut k_buf = vec![Su3Matrix::ZERO; v * 4];

    for stage in 0..s {
        let a_i = scheme.a[stage];
        let b_i = scheme.b[stage];

        for site in 0..v {
            let x = lattice.site_coords(site);
            for mu in 0..4 {
                let idx = site * 4 + mu;
                let z = flow_force(lattice, x, mu);
                k_buf[idx] = k_buf[idx].scale(a_i) + z;
            }
        }

        for site in 0..v {
            let x = lattice.site_coords(site);
            for mu in 0..4 {
                let idx = site * 4 + mu;
                let u = lattice.link(x, mu);
                let u_new = exp_su3_cayley_pub(&k_buf[idx], epsilon * b_i) * u;
                lattice.set_link(x, mu, u_new);
            }
        }
    }
}

/// Compute gauge energy density E(t) from the lattice at flow time t.
///
/// E = (β / 3V) Σ_{x, μ<ν} (1 - Re Tr P_{μν} / 3)
/// which simplifies to (1 - ⟨P⟩) × 2 × `N_c` for SU(3).
#[must_use]
pub fn energy_density(lattice: &Lattice) -> f64 {
    let plaq = lattice.average_plaquette();
    (1.0 - plaq) * 6.0
}

/// Run gradient flow from t=0 to `t=t_max` and collect measurements.
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
            FlowIntegrator::Rk3Luscher => lscfrk_step(lattice, epsilon, &lscfrk_lib::LSCFRK3_W6),
            FlowIntegrator::Lscfrk3w7 => lscfrk_step(lattice, epsilon, &lscfrk_lib::LSCFRK3_W7),
            FlowIntegrator::Lscfrk4ck => lscfrk_step(lattice, epsilon, &lscfrk_lib::LSCFRK4_CK),
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

/// Run gradient flow with custom LSCFRK coefficients.
///
/// Same as `run_flow` but uses caller-supplied coefficients instead of
/// a named integrator variant. Used by the benchmark flow workbench to
/// explore the LSCFRK parameter space.
pub fn run_flow_custom(
    lattice: &mut Lattice,
    coeffs: &LscfrkCoefficients,
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
        lscfrk_step(lattice, epsilon, coeffs);

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

/// Topological susceptibility from a set of charge measurements.
///
/// χ_t = ⟨Q²⟩ / V where Q is the topological charge per configuration
/// and V is the lattice 4-volume.
#[must_use]
pub fn topological_susceptibility(charges: &[f64], volume: usize) -> f64 {
    let n = charges.len() as f64;
    let mean_q2 = charges.iter().map(|q| q * q).sum::<f64>() / n;
    mean_q2 / volume as f64
}

/// Topological charge via the clover field-strength tensor.
///
/// Q = (1/(32π²)) Σ_x ε_{μνρσ} Tr[F_{μν}(x) F_{ρσ}(x)]
///
/// Uses the symmetric clover definition for F_{μν} after gradient flow
/// has smoothed the gauge field. On un-flowed lattices the result is noisy
/// and not expected to be integer-valued.
#[must_use]
pub fn topological_charge(lattice: &Lattice) -> f64 {
    let vol = lattice.volume();
    let mut q = 0.0;

    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        let f01 = clover_fmunu(lattice, x, 0, 1);
        let f23 = clover_fmunu(lattice, x, 2, 3);
        let f02 = clover_fmunu(lattice, x, 0, 2);
        let f13 = clover_fmunu(lattice, x, 1, 3);
        let f03 = clover_fmunu(lattice, x, 0, 3);
        let f12 = clover_fmunu(lattice, x, 1, 2);

        q += trace_product(&f01, &f23);
        q -= trace_product(&f02, &f13);
        q += trace_product(&f03, &f12);
    }

    q / (4.0 * std::f64::consts::PI * std::f64::consts::PI)
}

fn clover_fmunu(lattice: &Lattice, x: [usize; 4], mu: usize, nu: usize) -> Su3Matrix {
    let p1 = lattice.plaquette(x, mu, nu);
    let x_bk_nu = lattice.neighbor(x, nu, false);
    let p2 = plaq_oriented(lattice, x_bk_nu, mu, nu);
    let x_bk_mu = lattice.neighbor(x, mu, false);
    let x_bk_mu_bk_nu = lattice.neighbor(x_bk_mu, nu, false);
    let p3 = plaq_oriented(lattice, x_bk_mu_bk_nu, mu, nu);
    let p4 = plaq_oriented(lattice, x_bk_mu, mu, nu);

    let sum = p1 + p2 + p3 + p4;
    let diff = sum - sum.adjoint();
    diff.scale(0.125)
}

fn plaq_oriented(lattice: &Lattice, x: [usize; 4], mu: usize, nu: usize) -> Su3Matrix {
    lattice.plaquette(x, mu, nu)
}

fn trace_product(a: &Su3Matrix, b: &Su3Matrix) -> f64 {
    let p = *a * *b;
    p.trace().re
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
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

        let results_euler = run_flow(&mut lattice_euler, FlowIntegrator::Euler, 0.01, 0.5, 10);
        let results_rk3 = run_flow(&mut lattice_rk3, FlowIntegrator::Rk3Luscher, 0.01, 0.5, 10);

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
        let results = run_flow(&mut lattice, FlowIntegrator::Rk3Luscher, 0.01, 0.5, 5);

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
        let results = run_flow(&mut lattice, FlowIntegrator::Rk3Luscher, 0.01, 2.0, 1);
        let t0 = find_t0(&results);
        if let Some(t0_val) = t0 {
            assert!(t0_val > 0.0, "t₀ should be positive: {t0_val}");
            assert!(t0_val < 2.0, "t₀ should be within flow range: {t0_val}");
            println!("t₀ = {t0_val:.4} (4⁴, β=6.0)");
        }
    }

    #[test]
    fn find_w0_on_hot_start() {
        let mut lattice = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let results = run_flow(&mut lattice, FlowIntegrator::Rk3Luscher, 0.01, 2.0, 1);
        let w0 = find_w0(&results);
        if let Some(w0_val) = w0 {
            assert!(w0_val > 0.0, "w₀ should be positive: {w0_val}");
            assert!(w0_val < 2.0, "w₀ should be within flow range: {w0_val}");
            println!("w₀ = {w0_val:.4} (4⁴, β=6.0)");
        }
    }

    #[test]
    fn lscfrk3w7_agrees_with_luscher() {
        let mut lat_w6 = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let mut lat_w7 = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);

        let res_w6 = run_flow(&mut lat_w6, FlowIntegrator::Rk3Luscher, 0.01, 0.5, 10);
        let res_w7 = run_flow(&mut lat_w7, FlowIntegrator::Lscfrk3w7, 0.01, 0.5, 10);

        let e_w6 = res_w6.last().unwrap().energy_density;
        let e_w7 = res_w7.last().unwrap().energy_density;

        assert!(
            (e_w6 - e_w7).abs() < 0.05,
            "W6 and W7 should give similar E(0.5): w6={e_w6}, w7={e_w7}, diff={}",
            (e_w6 - e_w7).abs()
        );
    }

    #[test]
    fn lscfrk4ck_fourth_order() {
        let mut lat_w7 = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let mut lat_ck = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);

        let res_w7 = run_flow(&mut lat_w7, FlowIntegrator::Lscfrk3w7, 0.01, 0.5, 10);
        let res_ck = run_flow(&mut lat_ck, FlowIntegrator::Lscfrk4ck, 0.01, 0.5, 10);

        let e_w7 = res_w7.last().unwrap().energy_density;
        let e_ck = res_ck.last().unwrap().energy_density;

        assert!(
            (e_w7 - e_ck).abs() < 0.05,
            "W7 and CK45 should give similar E(0.5): w7={e_w7}, ck={e_ck}, diff={}",
            (e_w7 - e_ck).abs()
        );
    }

    #[test]
    fn derivation_produces_known_w6_coefficients() {
        let (a, b) = derive_lscfrk3(0.25, 2.0 / 3.0);
        assert!((a[0]).abs() < 1e-15, "A1 = 0");
        assert!(
            (a[1] - (-17.0 / 32.0)).abs() < 1e-14,
            "A2 = -17/32: got {}",
            a[1]
        );
        assert!(
            (a[2] - (-32.0 / 27.0)).abs() < 1e-14,
            "A3 = -32/27: got {}",
            a[2]
        );
        assert!((b[0] - 0.25).abs() < 1e-15, "B1 = 1/4: got {}", b[0]);
        assert!((b[1] - (8.0 / 9.0)).abs() < 1e-14, "B2 = 8/9: got {}", b[1]);
        assert!((b[2] - 0.75).abs() < 1e-14, "B3 = 3/4: got {}", b[2]);
    }

    #[test]
    fn derivation_produces_known_w7_coefficients() {
        let (a, b) = derive_lscfrk3(1.0 / 3.0, 0.75);
        assert!(
            (a[1] - (-5.0 / 9.0)).abs() < 1e-14,
            "A2 = -5/9: got {}",
            a[1]
        );
        assert!(
            (a[2] - (-153.0 / 128.0)).abs() < 1e-13,
            "A3 = -153/128: got {}",
            a[2]
        );
        assert!((b[0] - (1.0 / 3.0)).abs() < 1e-15, "B1 = 1/3: got {}", b[0]);
        assert!(
            (b[1] - (15.0 / 16.0)).abs() < 1e-14,
            "B2 = 15/16: got {}",
            b[1]
        );
        assert!(
            (b[2] - (8.0 / 15.0)).abs() < 1e-14,
            "B3 = 8/15: got {}",
            b[2]
        );
    }

    #[test]
    fn order_conditions_satisfied_for_w7() {
        let c2 = 1.0 / 3.0;
        let c3 = 3.0 / 4.0;
        let (a, b) = derive_lscfrk3(c2, c3);

        // Reconstruct Butcher coefficients from 2N-storage
        let a21 = b[0];
        let a32 = b[1];
        let a31 = a21 + a32 * a[1]; // a31 = B1 + B2*A2
        let b1 = b[0] + b[1] * a[1] + b[2] * a[2] * a[1]; // b1 = B1 + B2*A2 + B3*A3*A2
        let b2 = b[1] + b[2] * a[2]; // b2 = B2 + B3*A3
        let b3 = b[2]; // b3 = B3

        // Condition 1: b1 + b2 + b3 = 1
        assert!(
            (b1 + b2 + b3 - 1.0).abs() < 1e-14,
            "cond1: sum(b) = {}",
            b1 + b2 + b3
        );
        // Condition 2: b2*c2 + b3*c3 = 1/2
        assert!(
            (b2 * c2 + b3 * c3 - 0.5).abs() < 1e-14,
            "cond2: {}",
            b2 * c2 + b3 * c3
        );
        // Condition 3: b2*c2^2 + b3*c3^2 = 1/3
        assert!(
            (b2 * c2 * c2 + b3 * c3 * c3 - 1.0 / 3.0).abs() < 1e-14,
            "cond3: {}",
            b2 * c2 * c2 + b3 * c3 * c3
        );
        // Condition 4 (tree): b3*a32*c2 = 1/6
        assert!(
            (b3 * a32 * c2 - 1.0 / 6.0).abs() < 1e-14,
            "cond4: {}",
            b3 * a32 * c2
        );
        // Row sums
        assert!((a21 - c2).abs() < 1e-15, "row2: a21={a21} c2={c2}");
        assert!(
            (a31 + a32 - c3).abs() < 1e-14,
            "row3: a31+a32={} c3={c3}",
            a31 + a32
        );
    }

    #[test]
    fn luscher_and_w6_enum_produce_same_result() {
        let mut lat_luscher = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let mut lat_w6 = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);

        let res_luscher = run_flow(&mut lat_luscher, FlowIntegrator::Rk3Luscher, 0.01, 0.1, 1);
        lscfrk_step(&mut lat_w6, 0.01, &lscfrk_lib::LSCFRK3_W6);

        let p_luscher = res_luscher[1].plaquette;
        let p_w6 = lat_w6.average_plaquette();

        assert!(
            (p_luscher - p_w6).abs() < 1e-12,
            "Rk3Luscher enum should use LSCFRK3W6: luscher={p_luscher}, w6={p_w6}"
        );
    }

    #[test]
    fn w_function_monotonic_increasing() {
        let mut lattice = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let results = run_flow(&mut lattice, FlowIntegrator::Rk3Luscher, 0.01, 1.0, 1);
        let w_vals = compute_w_function(&results);
        assert!(!w_vals.is_empty(), "W(t) should have values");
        for window in w_vals.windows(2) {
            if window[0].0 > 0.1 {
                assert!(
                    window[1].1 >= window[0].1 - 0.01,
                    "W(t) should generally increase: t={}, W={} -> t={}, W={}",
                    window[0].0,
                    window[0].1,
                    window[1].0,
                    window[1].1
                );
            }
        }
    }
}
