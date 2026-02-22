// SPDX-License-Identifier: AGPL-3.0-only

//! Screened Coulomb (Yukawa) bound-state eigenvalue solver.
//!
//! Reproduces the atomic-structure content of Murillo & Weisheit (1998),
//! "Dense plasmas, screened interactions, and atomic ionization",
//! Physics Reports 302, 1–65 (Paper 6 in the hotSpring review queue).
//!
//! The screened Coulomb potential for a hydrogen-like atom (nuclear charge Z)
//! embedded in a plasma with screening parameter `κ`:
//!
//!   V(r) = −Z exp(−κr) / r   [Hartree atomic units]
//!
//! At κ = 0 this recovers unscreened hydrogen: `E_n` = −Z²/(2n²).
//! As κ increases, bound states become shallower and eventually vanish.
//! The critical screening parameter `κ_c`(n,l) is the value at which
//! the (n,l) state ceases to be bound.
//!
//! This connects directly to the Yukawa OCP MD (Papers 1, 5): the same
//! exp(−κr)/r functional form describes both ion-ion interaction and
//! electron-ion atomic binding in dense plasmas.
//!
//! # Numerical method
//!
//! Discretize the radial Schrödinger equation for `u`(r) = r R(r) on a
//! uniform grid `r_i` = (i+1)h, h = r_max/(N+1), with finite differences:
//!
//!   H u = E u,   `H_{ii}` = 1/h² + l(l+1)/(2`r_i`²) − Z exp(−κ`r_i`)/`r_i`
//!                 `H_{i,i±1}` = −1/(2h²)
//!
//! The Hamiltonian is tridiagonal. Bound-state eigenvalues (E < 0) are
//! found via Sturm bisection — O(N) per eigenvalue, no full diagonalization.
//!
//! # References
//!
//! - Murillo & Weisheit, Physics Reports 302, 1–65 (1998)
//! - Lam & Varshni, Phys. Rev. A 4, 1875 (1971) — critical screening
//! - Rogers, Graboske & Harwood, Phys. Rev. A 1, 1577 (1970)
//! - Harris, J. Chem. Phys. 36, 1609 (1962) — screened eigenvalues

pub const DEFAULT_N_GRID: usize = 2000;
pub const DEFAULT_R_MAX: f64 = 100.0;

/// Literature critical screening parameters `κ_c` (a.u.) for hydrogen (Z=1).
///
/// At κ > `κ_c`(n,l) the (n,l) bound state ceases to exist. Values from
/// Lam & Varshni (1971) Phys. Rev. A 4, 1875 and Rogers et al. (1970).
///
/// Format: (n, l, `κ_c`)
pub const CRITICAL_SCREENING_REFERENCE: &[(u32, u32, f64)] = &[
    (1, 0, 1.19061), // 1s
    (2, 0, 0.31750), // 2s
    (2, 1, 0.21954), // 2p
    (3, 0, 0.14459), // 3s
    (3, 1, 0.10789), // 3p
    (3, 2, 0.09025), // 3d
];

/// Exact hydrogen eigenvalues for comparison at κ = 0.
///
/// `E_{n}` = −Z²/(2n²) in Hartree. n = 1, 2, 3 shown.
pub const HYDROGEN_EXACT: &[(u32, f64)] = &[
    (1, -0.500_000_000),
    (2, -0.125_000_000),
    (3, -0.055_555_556),
];

/// Screening models from Murillo & Weisheit (1998) §3.
///
/// Connect the abstract screening parameter `κ` to physical plasma
/// conditions (density n, temperature T, coupling Γ).
pub mod screening_models {
    /// Debye-Hückel screening in reduced units: `κ_DH` × `a_ws` = √(3Γ).
    ///
    /// Valid in the weak-coupling limit (Γ ≪ 1). At stronger coupling,
    /// ion correlations reduce screening below the Debye prediction.
    ///
    /// Derivation: `λ_D`² = `kT`/(4πne²), `a_ws` = (3/(4πn))^{1/3},
    /// κ = `a_ws`/`λ_D` = √(3Γ) where Γ = e²/(`a_ws` `kT`).
    #[must_use]
    pub fn debye_kappa_reduced(gamma: f64) -> f64 {
        (3.0 * gamma).sqrt()
    }

    /// Ion-sphere screening: `κ_IS` × `a_ws` ≈ √3 ≈ 1.73.
    ///
    /// Strong-coupling limit — the ion sphere of radius `a_ws` contains
    /// exactly Z electrons, providing complete screening beyond `a_ws`.
    #[must_use]
    pub fn ion_sphere_kappa_reduced() -> f64 {
        3.0_f64.sqrt()
    }

    /// Stewart-Pyatt interpolation: bridges Debye ↔ ion-sphere.
    ///
    /// IPD: Δ/(`kT`) = (3Γ/2) × [(1 + (`λ_D`/`a_ws`)³)^{2/3} − (`λ_D`/`a_ws`)²]
    ///
    /// Returns an effective κ × `a_ws` by matching IPD to Yukawa depth shift.
    /// Reduces to Debye at Γ→0 and ion-sphere at Γ→∞.
    #[must_use]
    pub fn stewart_pyatt_kappa_reduced(gamma: f64) -> f64 {
        let kd = debye_kappa_reduced(gamma);
        let ratio = 1.0 / kd; // λ_D / a_ws = 1/κ_D
        let r3 = ratio * ratio * ratio;
        let ipd_over_kt = 1.5 * gamma * ((1.0 + r3).powf(2.0 / 3.0) - ratio * ratio);
        // Effective κ from IPD: Δ ≈ Z²e²κ/2 → κ = 2Δ/(Z²e²) = 2Δ/(Γ a_ws kT)
        // In reduced units: κ a_ws = 2 Δ/(kT) / Γ = 2 ipd_over_kt / Γ
        // ... but this can exceed physical bounds. Cap at ion-sphere.
        let kappa_eff = (2.0 * ipd_over_kt / gamma).min(ion_sphere_kappa_reduced());
        kappa_eff.max(0.0)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Sturm bisection for tridiagonal eigenvalues
// ═══════════════════════════════════════════════════════════════════

/// Count eigenvalues of a symmetric tridiagonal matrix below `lambda`.
///
/// Uses the Sturm sequence property: the number of sign changes in the
/// sequence `d_0`, `d_1`, ..., `d_{n-1}` equals the number of eigenvalues &lt; λ,
/// where `d_i` = (a_i - λ) - b_{i-1}²/`d_{i-1}`.
fn sturm_count(diag: &[f64], off_diag: &[f64], lambda: f64) -> usize {
    let n = diag.len();
    let mut count = 0;
    let mut d = diag[0] - lambda;
    if d < 0.0 {
        count += 1;
    }

    for i in 1..n {
        let d_safe = if d.abs() < 1e-30 {
            1e-30_f64.copysign(if d >= 0.0 { 1.0 } else { -1.0 })
        } else {
            d
        };
        d = (diag[i] - lambda) - off_diag[i - 1] * off_diag[i - 1] / d_safe;
        if d < 0.0 {
            count += 1;
        }
    }

    count
}

/// Find all eigenvalues of a symmetric tridiagonal matrix below `threshold`.
///
/// Uses Sturm bisection: O(N) per eigenvalue, O(N × `n_bound`) total.
/// Much faster than full diagonalization for large N with few bound states.
fn eigenvalues_below_threshold(diag: &[f64], off_diag: &[f64], threshold: f64) -> Vec<f64> {
    let n_bound = sturm_count(diag, off_diag, threshold);
    if n_bound == 0 {
        return vec![];
    }

    // Gershgorin lower bound
    let n = diag.len();
    let mut lo_global = f64::INFINITY;
    for i in 0..n {
        let r = if i > 0 { off_diag[i - 1].abs() } else { 0.0 }
            + if i < n - 1 { off_diag[i].abs() } else { 0.0 };
        lo_global = lo_global.min(diag[i] - r);
    }
    lo_global -= 1.0;

    let mut results = Vec::with_capacity(n_bound);
    for k in 0..n_bound {
        let mut lo = lo_global;
        let mut hi = threshold;

        // Bisect: find λ where sturm_count transitions from k to k+1
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if sturm_count(diag, off_diag, mid) <= k {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        results.push(0.5 * (lo + hi));
    }

    results
}

// ═══════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════

/// Bound-state eigenvalues of the screened Coulomb potential.
///
/// Returns only the bound states (E < 0), sorted ascending (most bound first).
///
/// # Arguments
/// * `z` — Nuclear charge (Z=1 for hydrogen)
/// * `kappa` — Screening parameter (a.u.⁻¹); 0 = unscreened
/// * `l` — Orbital angular momentum quantum number
/// * `n_grid` — Number of interior grid points
/// * `r_max` — Maximum radius (a.u.)
#[must_use]
pub fn eigenvalues(z: f64, kappa: f64, l: u32, n_grid: usize, r_max: f64) -> Vec<f64> {
    let n = n_grid;
    let h = r_max / (n as f64 + 1.0);
    let inv_h2 = 1.0 / (h * h);
    let l_f = f64::from(l);
    let centrifugal = l_f * (l_f + 1.0) / 2.0;

    let mut diag = vec![0.0_f64; n];
    let off_diag = vec![-0.5 * inv_h2; n - 1];

    for (i, d) in diag.iter_mut().enumerate().take(n) {
        let r = (i as f64 + 1.0) * h;
        let v_yukawa = -z * (-kappa * r).exp() / r;
        let v_cent = centrifugal / (r * r);
        *d = inv_h2 + v_cent + v_yukawa;
    }

    eigenvalues_below_threshold(&diag, &off_diag, 0.0)
}

/// Number of bound states for given parameters.
#[must_use]
pub fn bound_state_count(z: f64, kappa: f64, l: u32, n_grid: usize, r_max: f64) -> usize {
    let n = n_grid;
    let h = r_max / (n as f64 + 1.0);
    let inv_h2 = 1.0 / (h * h);
    let l_f = f64::from(l);
    let centrifugal = l_f * (l_f + 1.0) / 2.0;

    let mut diag = vec![0.0_f64; n];
    let off_diag = vec![-0.5 * inv_h2; n - 1];

    for (i, d) in diag.iter_mut().enumerate().take(n) {
        let r = (i as f64 + 1.0) * h;
        *d = inv_h2 + centrifugal / (r * r) - z * (-kappa * r).exp() / r;
    }

    sturm_count(&diag, &off_diag, 0.0)
}

/// Critical screening parameter κ_c where the (n,l) state becomes unbound.
///
/// Uses bisection on the bound-state count from the Sturm sequence.
///
/// # Panics
/// Panics if n ≤ l (invalid quantum numbers).
#[must_use]
pub fn critical_screening(z: f64, n: u32, l: u32, n_grid: usize, r_max: f64) -> f64 {
    assert!(n > l, "n must be > l for a valid state");
    let target = (n - l) as usize;

    let has_state =
        |kappa: f64| -> bool { bound_state_count(z, kappa, l, n_grid, r_max) >= target };

    let mut hi = z * 2.0;
    while has_state(hi) {
        hi *= 2.0;
    }
    let mut lo = 0.0;

    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if has_state(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    0.5 * (lo + hi)
}

/// Ground-state energy for angular momentum l at given screening.
///
/// Returns `None` if no bound state exists.
#[must_use]
pub fn ground_state_energy(z: f64, kappa: f64, l: u32, n_grid: usize, r_max: f64) -> Option<f64> {
    let evals = eigenvalues(z, kappa, l, n_grid, r_max);
    evals.first().copied()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ── Hydrogen limit (κ = 0): eigenvalues match exact E_n = −Z²/(2n²) ──

    #[test]
    fn hydrogen_1s_eigenvalue() {
        let evals = eigenvalues(1.0, 0.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX);
        assert!(!evals.is_empty(), "must have at least one bound state");
        let e1s = evals[0];
        let exact = -0.5;
        let rel_err = ((e1s - exact) / exact).abs();
        assert!(
            rel_err < 0.005,
            "1s energy {e1s:.6} should be within 0.5% of {exact}, rel_err={rel_err:.4e}"
        );
    }

    #[test]
    fn hydrogen_2s_eigenvalue() {
        let evals = eigenvalues(1.0, 0.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX);
        assert!(evals.len() >= 2, "need at least 2 l=0 bound states");
        let e2s = evals[1];
        let exact = -0.125;
        let rel_err = ((e2s - exact) / exact).abs();
        assert!(
            rel_err < 0.01,
            "2s energy {e2s:.6} should be within 1% of {exact}, rel_err={rel_err:.4e}"
        );
    }

    #[test]
    fn hydrogen_2p_eigenvalue() {
        let evals = eigenvalues(1.0, 0.0, 1, DEFAULT_N_GRID, DEFAULT_R_MAX);
        assert!(!evals.is_empty(), "need at least 1 l=1 bound state");
        let e2p = evals[0];
        let exact = -0.125;
        let rel_err = ((e2p - exact) / exact).abs();
        assert!(
            rel_err < 0.01,
            "2p energy {e2p:.6} should be within 1% of {exact}, rel_err={rel_err:.4e}"
        );
    }

    #[test]
    fn hydrogen_degeneracy_2s_2p() {
        let evals_s = eigenvalues(1.0, 0.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX);
        let evals_p = eigenvalues(1.0, 0.0, 1, DEFAULT_N_GRID, DEFAULT_R_MAX);
        let e2s = evals_s[1];
        let e2p = evals_p[0];
        let split = (e2s - e2p).abs();
        assert!(
            split < 0.005,
            "2s/2p degeneracy should be <5 mHa at κ=0, got {split:.6}"
        );
    }

    // ── Screening physics ──

    #[test]
    fn screening_reduces_binding() {
        let e_0 = eigenvalues(1.0, 0.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX)[0];
        let e_05 = eigenvalues(1.0, 0.5, 0, DEFAULT_N_GRID, DEFAULT_R_MAX)[0];
        assert!(
            e_05 > e_0,
            "screening should weaken binding: E(κ=0.5)={e_05:.6} > E(κ=0)={e_0:.6}"
        );
    }

    #[test]
    fn screening_breaks_degeneracy() {
        let kappa = 0.1;
        let evals_s = eigenvalues(1.0, kappa, 0, DEFAULT_N_GRID, DEFAULT_R_MAX);
        let evals_p = eigenvalues(1.0, kappa, 1, DEFAULT_N_GRID, DEFAULT_R_MAX);
        let e2s = evals_s[1];
        let e2p = evals_p[0];
        assert!(
            e2s < e2p,
            "at κ={kappa}, 2s ({e2s:.6}) should be deeper than 2p ({e2p:.6})"
        );
    }

    #[test]
    fn no_bound_states_at_large_kappa() {
        let count = bound_state_count(1.0, 5.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX);
        assert!(
            count == 0,
            "no bound states at κ=5 for hydrogen, found {count}"
        );
    }

    #[test]
    fn bound_state_count_decreases_with_kappa() {
        let n0 = bound_state_count(1.0, 0.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX);
        let n05 = bound_state_count(1.0, 0.5, 0, DEFAULT_N_GRID, DEFAULT_R_MAX);
        let n10 = bound_state_count(1.0, 1.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX);
        assert!(
            n0 >= n05 && n05 >= n10,
            "bound-state count should decrease: N(0)={n0}, N(0.5)={n05}, N(1.0)={n10}"
        );
    }

    // ── Critical screening ──

    #[test]
    fn critical_screening_1s() {
        let kc = critical_screening(1.0, 1, 0, DEFAULT_N_GRID, DEFAULT_R_MAX);
        let lit = 1.19061;
        let rel_err = ((kc - lit) / lit).abs();
        assert!(
            rel_err < 0.02,
            "κ_c(1s) = {kc:.5} should be within 2% of literature {lit}, rel_err={rel_err:.4e}"
        );
    }

    #[test]
    fn critical_screening_2s() {
        let kc = critical_screening(1.0, 2, 0, DEFAULT_N_GRID, DEFAULT_R_MAX);
        let lit = 0.31750;
        let rel_err = ((kc - lit) / lit).abs();
        assert!(
            rel_err < crate::tolerances::SCREENED_CRITICAL_VS_LITERATURE,
            "κ_c(2s) = {kc:.5} should be within 5% of literature {lit}, rel_err={rel_err:.4e}"
        );
    }

    #[test]
    fn critical_screening_2p() {
        let kc = critical_screening(1.0, 2, 1, DEFAULT_N_GRID, DEFAULT_R_MAX);
        let lit = 0.21954;
        let rel_err = ((kc - lit) / lit).abs();
        assert!(
            rel_err < crate::tolerances::SCREENED_CRITICAL_VS_LITERATURE,
            "κ_c(2p) = {kc:.5} should be within 5% of literature {lit}, rel_err={rel_err:.4e}"
        );
    }

    #[test]
    fn critical_screening_ordering() {
        let kc_1s = critical_screening(1.0, 1, 0, DEFAULT_N_GRID, DEFAULT_R_MAX);
        let kc_2s = critical_screening(1.0, 2, 0, DEFAULT_N_GRID, DEFAULT_R_MAX);
        let kc_2p = critical_screening(1.0, 2, 1, DEFAULT_N_GRID, DEFAULT_R_MAX);
        assert!(
            kc_1s > kc_2s && kc_2s > kc_2p,
            "κ_c ordering: 1s({kc_1s:.4}) > 2s({kc_2s:.4}) > 2p({kc_2p:.4})"
        );
    }

    // ── Screening models ──

    #[test]
    fn debye_weak_coupling() {
        let kd = screening_models::debye_kappa_reduced(0.01);
        let expected = (3.0 * 0.01_f64).sqrt();
        assert!(
            (kd - expected).abs() < 1e-10,
            "Debye κ at Γ=0.01: {kd:.6} vs {expected:.6}"
        );
    }

    #[test]
    fn debye_increases_with_coupling() {
        let k1 = screening_models::debye_kappa_reduced(0.1);
        let k2 = screening_models::debye_kappa_reduced(1.0);
        let k3 = screening_models::debye_kappa_reduced(10.0);
        assert!(
            k1 < k2 && k2 < k3,
            "Debye κ increases with Γ: {k1:.4} < {k2:.4} < {k3:.4}"
        );
    }

    #[test]
    fn ion_sphere_value() {
        let is_val = screening_models::ion_sphere_kappa_reduced();
        let expected = 3.0_f64.sqrt();
        assert!((is_val - expected).abs() < 1e-14);
    }

    #[test]
    fn stewart_pyatt_strong_coupling_limit() {
        let sp = screening_models::stewart_pyatt_kappa_reduced(1000.0);
        let is_val = screening_models::ion_sphere_kappa_reduced();
        let rel = ((sp - is_val) / is_val).abs();
        assert!(
            rel < 0.05,
            "SP → IS at strong coupling: {sp:.4} vs {is_val:.4}, rel_err={rel:.4e}"
        );
    }

    // ── Z > 1 scaling ──

    #[test]
    fn helium_ion_deeper_binding() {
        let e_h = eigenvalues(1.0, 0.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX)[0];
        let e_he = eigenvalues(2.0, 0.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX)[0];
        assert!(e_he < e_h * 3.5, "He+ 1s should be ~4× deeper than H 1s");
    }

    #[test]
    fn helium_ion_ground_state() {
        let e = eigenvalues(2.0, 0.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX)[0];
        let exact = -2.0; // −Z²/2
        let rel_err = ((e - exact) / exact).abs();
        assert!(
            rel_err < 0.005,
            "He+ 1s = {e:.6} should be within 0.5% of {exact}"
        );
    }

    // ── Sturm sequence correctness ──

    #[test]
    fn sturm_count_simple_2x2() {
        let diag = vec![2.0, 2.0];
        let off = vec![-1.0];
        // Eigenvalues of [[2,-1],[-1,2]] are 1 and 3
        assert_eq!(sturm_count(&diag, &off, 0.5), 0);
        assert_eq!(sturm_count(&diag, &off, 1.5), 1);
        assert_eq!(sturm_count(&diag, &off, 3.5), 2);
    }

    #[test]
    fn eigenvalues_below_threshold_simple() {
        let diag = vec![2.0, 2.0, 2.0];
        let off = vec![-1.0, -1.0];
        // Eigenvalues: 2-√2, 2, 2+√2 ≈ 0.586, 2.0, 3.414
        let below_1 = eigenvalues_below_threshold(&diag, &off, 1.0);
        assert_eq!(below_1.len(), 1);
        let expected = 2.0 - 2.0_f64.sqrt();
        assert!(
            (below_1[0] - expected).abs() < 1e-10,
            "got {}, expected {expected}",
            below_1[0]
        );
    }

    // ── Uncovered regions: ground_state_energy None, sturm d_safe, n_bound=0 ──

    #[test]
    fn ground_state_energy_none_when_no_bound_state() {
        let e = ground_state_energy(1.0, 10.0, 0, DEFAULT_N_GRID, DEFAULT_R_MAX);
        assert!(e.is_none(), "κ=10: no bound state for hydrogen");
    }

    #[test]
    fn ground_state_energy_some_for_weak_screening() {
        let e = ground_state_energy(1.0, 0.01, 0, DEFAULT_N_GRID, DEFAULT_R_MAX);
        assert!(e.is_some());
        let ev = e.unwrap();
        assert!(ev < 0.0, "bound state energy should be negative");
        assert!(ev > -0.6, "1s at κ=0.01 should be close to −0.5");
    }

    #[test]
    fn eigenvalues_below_threshold_empty_when_none_below() {
        let diag = vec![5.0, 5.0, 5.0];
        let off = vec![0.0, 0.0];
        let below = eigenvalues_below_threshold(&diag, &off, 0.0);
        assert!(below.is_empty());
    }

    #[test]
    fn stewart_pyatt_weak_coupling_positive() {
        let sp = screening_models::stewart_pyatt_kappa_reduced(0.01);
        assert!(
            sp >= 0.0 && sp.is_finite(),
            "SP should be non-negative and finite"
        );
    }
}
