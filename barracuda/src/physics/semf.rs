//! Semi-Empirical Mass Formula with Skyrme-derived coefficients
//!
//! Port of: `control/surrogate/nuclear-eos/wrapper/skyrme_hf.py::semf_binding_energy`

use super::constants::*;
use super::nuclear_matter::nuclear_matter_properties;
use std::f64::consts::PI;

/// Binding energy via SEMF with Skyrme-derived coefficients
pub fn semf_binding_energy(z: usize, n: usize, params: &[f64]) -> f64 {
    let a = z + n;
    if a == 0 {
        return 0.0;
    }
    let af = a as f64;
    let zf = z as f64;
    let nf = n as f64;

    // Derive SEMF coefficients from nuclear matter properties
    let nmp = match nuclear_matter_properties(params) {
        Some(nmp) => nmp,
        None => return 0.0,
    };

    let a_v = nmp.e_a_mev.abs();
    let r0 = (3.0 / (4.0 * PI * nmp.rho0_fm3)).powf(1.0 / 3.0);
    let a_s = a_v * 1.1;
    let a_c = 3.0 * E2 / (5.0 * r0);
    let a_a = nmp.j_mev;
    let a_p = 12.0 / af.max(1.0).sqrt();

    // Bethe-Weizsäcker
    let mut b = a_v * af;
    b -= a_s * af.powf(2.0 / 3.0);
    b -= a_c * zf * (zf - 1.0) / af.powf(1.0 / 3.0);
    b -= a_a * (nf - zf).powi(2) / af;

    // Pairing
    if z % 2 == 0 && n % 2 == 0 {
        b += a_p;
    } else if z % 2 == 1 && n % 2 == 1 {
        b -= a_p;
    }

    b.max(0.0)
}

