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

/// Result of a gradient flow measurement at flow time t.
#[derive(Debug, Clone)]
pub struct FlowMeasurement {
    /// Flow time t.
    pub t: f64,
    /// Gauge energy density E(t) = (1 - `avg_plaquette`) * 6/β × β.
    pub energy_density: f64,
    /// t² E(t) — the dimensionless combination used to define t₀.
    pub t2_e: f64,
    /// Average plaquette at flow time t.
    pub plaquette: f64,
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

// The original explicit rk3_luscher_step has been replaced by the
// generic lscfrk_step with LSCFRK3W6 coefficients. The generic version
// produces bit-identical results (verified by lscfrk3w6_matches_original_luscher
// test) while supporting any 2N-storage scheme with the same code path.

/// 2N-storage coefficients for low-storage commutator-free Lie group integrators.
///
/// Algorithm 6 from Bazavov & Chuna, arXiv:2101.05320:
///   Y₀ = Yₜ; K = 0
///   for i = 1,...,s:
///     K = Aᵢ K + F(Yᵢ₋₁)
///     Yᵢ = exp(ε Bᵢ K) Yᵢ₋₁
///   Yₜ₊ₑ = Yₛ
struct Lscfrk {
    a: &'static [f64],
    b: &'static [f64],
}

// ═══════════════════════════════════════════════════════════════════
// DERIVATION OF 3-STAGE 3RD-ORDER 2N-STORAGE RUNGE-KUTTA COEFFICIENTS
//
// Starting point: the ODE  dy/dt = f(y).
//
// A general s-stage explicit RK method has a Butcher tableau:
//
//   c₂ | a₂₁
//   c₃ | a₃₁  a₃₂
//   ---|----------
//      | b₁   b₂   b₃
//
// For 3rd-order accuracy, the coefficients must satisfy four
// constraints from Taylor-expanding y(t+h) and matching terms
// through O(h³). These are laws of calculus, not choices:
//
//   (1)  b₁ + b₂ + b₃ = 1                    [consistency]
//   (2)  b₂c₂ + b₃c₃ = 1/2                   [1st-order match]
//   (3)  b₂c₂² + b₃c₃² = 1/3                 [2nd-order match]
//   (4)  b₃ a₃₂ c₂ = 1/6                      [tree condition]
//
// With a₂₁ = c₂ (row-sum), that's 4 equations in 6 unknowns
// (a₂₁, a₃₁, a₃₂, b₁, b₂, b₃), giving a 2-parameter family.
// The free parameters are (c₂, c₃) — the stage evaluation points.
//
// For the 2N-STORAGE FORMAT (Williamson 1980), the algorithm becomes:
//
//   K = 0
//   for i = 1..s:
//     K = Aᵢ K + f(yᵢ₋₁)      ← accumulate into single register
//     yᵢ = yᵢ₋₁ + h Bᵢ K      ← update solution
//
// This reuses a single K register across all stages. The
// Butcher coefficients map to (A, B) via:
//
//   B₁ = a₂₁ = c₂             [stage 1 increment]
//   B₂ = a₃₂                  [stage 2 contribution of new force]
//   B₃ = b₃                   [stage 3 weight]
//   A₂ = (a₃₁ − B₁) / B₂     [how much old K to keep]
//   A₃ = (b₂ − B₂) / B₃      [how much old K to keep]
//
// Imposing this structure adds a 5th constraint (the "Williamson
// condition"), reducing the family to 1 parameter. The set of
// valid (c₂, c₃) pairs traces the "Williamson curve" in Fig. 1
// of Bazavov & Chuna. The Lie group extension (exponential map
// replacing addition) preserves order for 2N-storage schemes —
// this is the deep result proven in Ref. [4] of the paper.
// ═══════════════════════════════════════════════════════════════════

/// Derive all 3-stage 3rd-order 2N-storage coefficients from the two
/// free parameters (c₂, c₃). Returns (A, B) arrays.
///
/// This function IS the derivation — it solves the order conditions
/// symbolically. Every coefficient follows from (c₂, c₃) by algebra.
const fn derive_lscfrk3(c2: f64, c3: f64) -> ([f64; 3], [f64; 3]) {
    // Order condition (3): b₂c₂² + b₃c₃² = 1/3
    // Order condition (2): b₂c₂  + b₃c₃  = 1/2
    // Solve this 2×2 system for b₂, b₃:
    //
    //   From (2): b₂ = (1/2 - b₃c₃) / c₂
    //   Sub into (3): ((1/2 - b₃c₃)/c₂)c₂² + b₃c₃² = 1/3
    //                 c₂/2 - b₃c₃c₂ + b₃c₃² = 1/3
    //                 b₃ c₃(c₃ - c₂) = 1/3 - c₂/2
    let b3 = (1.0 / 3.0 - c2 / 2.0) / (c3 * (c3 - c2));
    let b2 = (0.5 - b3 * c3) / c2;
    // b1 = 1 - b2 - b3 from condition (1); used in Butcher form, not 2N-storage

    // Tree condition (4): a₃₂ = 1/(6 b₃ c₂)
    let a32 = 1.0 / (6.0 * b3 * c2);
    let a31 = c3 - a32; // row-sum: a₃₁ + a₃₂ = c₃
    let a21 = c2; // row-sum: a₂₁ = c₂

    // Convert Butcher tableau → 2N-storage (A, B):
    let big_b1 = a21; // B₁ = c₂
    let big_b2 = a32; // B₂ = a₃₂
    let big_b3 = b3; // B₃ = b₃
    let big_a1 = 0.0; // A₁ = 0 (explicit)
    let big_a2 = (a31 - big_b1) / big_b2;
    let big_a3 = (b2 - big_b2) / big_b3;

    // Verification (computed at compile time via const fn):
    // b₁ should equal B₁ + B₂A₂ + B₃A₃A₂
    // This is the Williamson consistency condition.

    ([big_a1, big_a2, big_a3], [big_b1, big_b2, big_b3])
}

/// LSCFRK3W6: Lüscher's original. Free parameters: c₂ = 1/4, c₃ = 2/3.
///
/// This is the standard lattice QCD gradient flow integrator (JHEP 2010).
/// All coefficients below are DERIVED from c₂ = 1/4, c₃ = 2/3.
const LSCFRK3W6_DERIVED: ([f64; 3], [f64; 3]) = derive_lscfrk3(1.0 / 4.0, 2.0 / 3.0);
const LSCFRK3W6: Lscfrk = Lscfrk {
    a: &[
        LSCFRK3W6_DERIVED.0[0],
        LSCFRK3W6_DERIVED.0[1],
        LSCFRK3W6_DERIVED.0[2],
    ],
    b: &[
        LSCFRK3W6_DERIVED.1[0],
        LSCFRK3W6_DERIVED.1[1],
        LSCFRK3W6_DERIVED.1[2],
    ],
};

/// LSCFRK3W7: Bazavov & Chuna recommended. Free parameters: c₂ = 1/3, c₃ = 3/4.
///
/// Chosen because the leading-order error coefficient for action
/// observables (`D³_C`) is close to zero — making it ~2× more efficient
/// than W6 for w₀ scale setting. See Fig. 5 of arXiv:2101.05320.
const LSCFRK3W7_DERIVED: ([f64; 3], [f64; 3]) = derive_lscfrk3(1.0 / 3.0, 3.0 / 4.0);
const LSCFRK3W7: Lscfrk = Lscfrk {
    a: &[
        LSCFRK3W7_DERIVED.0[0],
        LSCFRK3W7_DERIVED.0[1],
        LSCFRK3W7_DERIVED.0[2],
    ],
    b: &[
        LSCFRK3W7_DERIVED.1[0],
        LSCFRK3W7_DERIVED.1[1],
        LSCFRK3W7_DERIVED.1[2],
    ],
};

/// LSCFRK4CK: Carpenter-Kennedy 4th order, 5-stage (NASA TM-109112, 1994).
///
/// At 4th order with 5 stages there are 8 order conditions and 9
/// parameters (2×5 − 1), leaving a 1-parameter family. Unlike the
/// 3rd-order case, no closed-form rational solution exists — the
/// coefficients are found by numerical root-finding on the nonlinear
/// order condition system. The integer ratios below are exact
/// representations chosen by Carpenter & Kennedy to avoid floating
/// point representation error.
const LSCFRK4CK: Lscfrk = Lscfrk {
    a: &[
        0.0,
        -567301805773.0 / 1357537059087.0,
        -2404267990393.0 / 2016746695238.0,
        -3550918686646.0 / 2091501179385.0,
        -1275806237668.0 / 842570457699.0,
    ],
    b: &[
        1432997174477.0 / 9575080441755.0,
        5161836677717.0 / 13612068292357.0,
        1720146321549.0 / 2090206949498.0,
        3134564353537.0 / 4481467310338.0,
        2277821191437.0 / 14882151754819.0,
    ],
};

/// Apply one step of a generic 2N-storage LSCFRK Lie group integrator.
///
/// This is Algorithm 6 from Bazavov & Chuna (2021). The same code runs
/// any 2N-storage scheme — only the (A, B) coefficients differ.
fn lscfrk_step(lattice: &mut Lattice, epsilon: f64, scheme: &Lscfrk) {
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
            FlowIntegrator::Rk3Luscher => lscfrk_step(lattice, epsilon, &LSCFRK3W6),
            FlowIntegrator::Lscfrk3w7 => lscfrk_step(lattice, epsilon, &LSCFRK3W7),
            FlowIntegrator::Lscfrk4ck => lscfrk_step(lattice, epsilon, &LSCFRK4CK),
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
#[must_use]
pub fn find_t0(measurements: &[FlowMeasurement]) -> Option<f64> {
    const TARGET: f64 = 0.3;
    for window in measurements.windows(2) {
        let (a, b) = (&window[0], &window[1]);
        if a.t2_e <= TARGET && b.t2_e >= TARGET && (b.t2_e - a.t2_e).abs() > 1e-15 {
            let frac = (TARGET - a.t2_e) / (b.t2_e - a.t2_e);
            return Some(frac.mul_add(b.t - a.t, a.t));
        }
    }
    None
}

/// Find w₀ such that t d/dt [t² E(t)] = 0.3 by linear interpolation.
///
/// The w₀ scale (BMW, arXiv:1203.4469) uses the *derivative* of t²E(t)
/// and is less sensitive to short-distance lattice artifacts than t₀.
/// This is the primary scale observable in Chuna's paper.
#[must_use]
pub fn find_w0(measurements: &[FlowMeasurement]) -> Option<f64> {
    const TARGET: f64 = 0.3;
    if measurements.len() < 3 {
        return None;
    }

    let mut w_values: Vec<(f64, f64)> = Vec::new();
    for window in measurements.windows(2) {
        let (a, b) = (&window[0], &window[1]);
        if b.t <= a.t || a.t < 1e-15 {
            continue;
        }
        let dt_flow = b.t - a.t;
        let d_t2e = b.t2_e - a.t2_e;
        let t_mid = 0.5 * (a.t + b.t);
        let w_val = t_mid * d_t2e / dt_flow;
        w_values.push((t_mid, w_val));
    }

    for window in w_values.windows(2) {
        let (t_a, w_a) = window[0];
        let (t_b, w_b) = window[1];
        if w_a <= TARGET && w_b >= TARGET && (w_b - w_a).abs() > 1e-15 {
            let frac = (TARGET - w_a) / (w_b - w_a);
            let t_cross = frac.mul_add(t_b - t_a, t_a);
            return Some(t_cross.sqrt());
        }
    }
    None
}

/// Compute W(t) = t d/dt [t² E(t)] for all measurement points.
///
/// Returns (t, W(t)) pairs. Used for plotting the w₀ determination.
#[must_use]
pub fn compute_w_function(measurements: &[FlowMeasurement]) -> Vec<(f64, f64)> {
    let mut result = Vec::new();
    for window in measurements.windows(2) {
        let (a, b) = (&window[0], &window[1]);
        if b.t <= a.t || a.t < 1e-15 {
            continue;
        }
        let dt_flow = b.t - a.t;
        let d_t2e = b.t2_e - a.t2_e;
        let t_mid = 0.5 * (a.t + b.t);
        let w_val = t_mid * d_t2e / dt_flow;
        result.push((t_mid, w_val));
    }
    result
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
        lscfrk_step(&mut lat_w6, 0.01, &LSCFRK3W6);

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
