//! Nuclear matter properties from Skyrme parameters (analytic)
//!
//! Port of: `control/surrogate/nuclear-eos/wrapper/skyrme_hf.py`
//! Uses: `barracuda::optimize::bisect` for saturation density

use super::constants::*;
use barracuda::optimize::bisect;
use std::f64::consts::PI;

/// Nuclear matter property results
#[derive(Debug, Clone)]
pub struct NuclearMatterProps {
    pub rho0_fm3: f64,      // saturation density
    pub e_a_mev: f64,       // energy per nucleon at saturation
    pub k_inf_mev: f64,     // incompressibility
    pub m_eff_ratio: f64,   // effective mass ratio m*/m
    pub j_mev: f64,         // symmetry energy
}

/// Energy per nucleon in symmetric nuclear matter
fn energy_per_nucleon_snm(rho: f64, p: &[f64; 10]) -> f64 {
    if rho <= 0.0 {
        return 0.0;
    }

    let (t0, t1, t2, t3) = (p[0], p[1], p[2], p[3]);
    let (_x0, _x1, x2, _x3) = (p[4], p[5], p[6], p[7]);
    let alpha = p[8];

    let kf = (3.0 * PI * PI * rho / 2.0).powf(1.0 / 3.0);
    let tau = (3.0 / 5.0) * kf * kf * rho;

    let e_kin = HBAR2_2M * (3.0 / 5.0) * kf * kf;
    let e_t0 = (3.0 / 8.0) * t0 * rho;
    let e_t3 = (1.0 / 16.0) * t3 * rho.powf(alpha + 1.0);

    let theta = 3.0 * t1 + t2 * (5.0 + 4.0 * x2);
    let e_t1t2 = (1.0 / 16.0) * theta * tau;

    e_kin + e_t0 + e_t3 + e_t1t2
}

/// Compute nuclear matter properties from Skyrme parameters.
///
/// Uses `barracuda::optimize::bisect` for saturation density (replacing scipy.optimize.brentq).
pub fn nuclear_matter_properties(params: &[f64]) -> Option<NuclearMatterProps> {
    if params.len() != 10 {
        return None;
    }

    let p: [f64; 10] = params.try_into().ok()?;
    let (t0, t1, t2, t3) = (p[0], p[1], p[2], p[3]);
    let (x0, x1, x2, x3) = (p[4], p[5], p[6], p[7]);
    let alpha = p[8];

    // Find saturation density: d(E/A)/dρ = 0
    // Using barracuda::optimize::bisect (equivalent to scipy.optimize.brentq)
    let de_drho = |rho: f64| {
        let dr = (rho * 1e-6_f64).max(1e-10);
        (energy_per_nucleon_snm(rho + dr, &p) - energy_per_nucleon_snm(rho - dr, &p)) / (2.0 * dr)
    };

    let rho0 = bisect(de_drho, 0.05, 0.30, 1e-10, 100).unwrap_or(0.16);

    let e_a = energy_per_nucleon_snm(rho0, &p);

    // Incompressibility
    let dr = rho0 * 1e-4;
    let d2e = (energy_per_nucleon_snm(rho0 + dr, &p)
        - 2.0 * energy_per_nucleon_snm(rho0, &p)
        + energy_per_nucleon_snm(rho0 - dr, &p))
        / (dr * dr);
    let k_inf = 9.0 * rho0 * rho0 * d2e;

    // Effective mass
    let theta = 3.0 * t1 + t2 * (5.0 + 4.0 * x2);
    let m_eff = 1.0 / (1.0 + (M_NUCLEON / (4.0 * HBAR_C * HBAR_C)) * theta * rho0);

    // Symmetry energy
    let kf0 = (3.0 * PI * PI * rho0 / 2.0).powf(1.0 / 3.0);
    let j_kin = HBAR2_2M * kf0 * kf0 / (3.0 * m_eff);
    let j_t0 = -(t0 / 4.0) * (2.0 * x0 + 1.0) * rho0;
    let j_t3 = -(t3 / 24.0) * (2.0 * x3 + 1.0) * rho0.powf(alpha + 1.0);
    let theta_s = t2 * (4.0 + 5.0 * x2) - 3.0 * t1 * x1;
    let tau0 = (3.0 / 5.0) * kf0 * kf0 * rho0;
    let j_t1t2 = -(1.0 / 24.0) * theta_s * tau0;
    let j = j_kin + j_t0 + j_t3 + j_t1t2;

    Some(NuclearMatterProps {
        rho0_fm3: rho0,
        e_a_mev: e_a,
        k_inf_mev: k_inf,
        m_eff_ratio: m_eff,
        j_mev: j,
    })
}

