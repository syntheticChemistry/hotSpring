// SPDX-License-Identifier: AGPL-3.0-or-later

use super::super::{
    complex_f64::Complex64,
    su3::Su3Matrix,
    wilson::Lattice,
};

/// Flatten lattice links to f64 array (same layout as `DiracGpuLayout`).
#[must_use]
pub fn flatten_links(lattice: &Lattice) -> Vec<f64> {
    let vol = lattice.volume();
    let mut flat = vec![0.0_f64; vol * 4 * 18];
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        for mu in 0..4 {
            let u = lattice.link(x, mu);
            let base = (idx * 4 + mu) * 18;
            for row in 0..3 {
                for col in 0..3 {
                    flat[base + row * 6 + col * 2] = u.m[row][col].re;
                    flat[base + row * 6 + col * 2 + 1] = u.m[row][col].im;
                }
            }
        }
    }
    flat
}

/// Build neighbor table (same layout as `DiracGpuLayout`).
#[must_use]
pub fn build_neighbors(lattice: &Lattice) -> Vec<u32> {
    let vol = lattice.volume();
    let mut neighbors = vec![0_u32; vol * 8];
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        for mu in 0..4 {
            let fwd = lattice.site_index(lattice.neighbor(x, mu, true));
            let bwd = lattice.site_index(lattice.neighbor(x, mu, false));
            neighbors[idx * 8 + mu * 2] = fwd as u32;
            neighbors[idx * 8 + mu * 2 + 1] = bwd as u32;
        }
    }
    neighbors
}

/// Flatten SU(3) momenta to f64 array.
#[must_use]
pub fn flatten_momenta(momenta: &[Su3Matrix]) -> Vec<f64> {
    let mut flat = vec![0.0_f64; momenta.len() * 18];
    for (i, p) in momenta.iter().enumerate() {
        let base = i * 18;
        for row in 0..3 {
            for col in 0..3 {
                flat[base + row * 6 + col * 2] = p.m[row][col].re;
                flat[base + row * 6 + col * 2 + 1] = p.m[row][col].im;
            }
        }
    }
    flat
}

/// Unflatten f64 array back to SU(3) link matrices and update lattice.
pub fn unflatten_links_into(lattice: &mut Lattice, flat: &[f64]) {
    let vol = lattice.volume();
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        for mu in 0..4 {
            let base = (idx * 4 + mu) * 18;
            let mut m = Su3Matrix::ZERO;
            for row in 0..3 {
                for col in 0..3 {
                    m.m[row][col] =
                        Complex64::new(flat[base + row * 6 + col * 2], flat[base + row * 6 + col * 2 + 1]);
                }
            }
            lattice.set_link(x, mu, m);
        }
    }
}
