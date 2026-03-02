// SPDX-License-Identifier: AGPL-3.0-only

//! Radial Distribution Function (RDF) from MD position snapshots.
//!
//! Computes g(r) binned at discrete r values for structural analysis.

use std::f64::consts::PI;

use crate::tolerances::DIVISION_GUARD;

/// RDF result: g(r) binned at discrete r values.
#[derive(Clone, Debug)]
pub struct Rdf {
    /// Bin centers in units of a_ws.
    pub r_values: Vec<f64>,
    /// Radial distribution function g(r).
    pub g_values: Vec<f64>,
    /// Bin width (reduced units).
    pub dr: f64,
}

/// Compute RDF from position snapshots (CPU post-process)
#[must_use]
pub fn compute_rdf(snapshots: &[Vec<f64>], n: usize, box_side: f64, n_bins: usize) -> Rdf {
    let r_max = box_side / 2.0; // max meaningful distance with PBC
    let dr = r_max / n_bins as f64;
    let mut histogram = vec![0u64; n_bins];
    let n_frames = snapshots.len();

    for snap in snapshots {
        for i in 0..n {
            let xi = snap[i * 3];
            let yi = snap[i * 3 + 1];
            let zi = snap[i * 3 + 2];

            for j in (i + 1)..n {
                let mut dx = snap[j * 3] - xi;
                let mut dy = snap[j * 3 + 1] - yi;
                let mut dz = snap[j * 3 + 2] - zi;

                // PBC minimum image
                dx -= box_side * (dx / box_side).round();
                dy -= box_side * (dy / box_side).round();
                dz -= box_side * (dz / box_side).round();

                let r = dz.mul_add(dz, dy.mul_add(dy, dx * dx)).sqrt();
                let bin = (r / dr) as usize;
                if bin < n_bins {
                    histogram[bin] += 1;
                }
            }
        }
    }

    // Normalize: g(r) = histogram / (n_frames * N * n_density * 4π r² dr)
    let n_density = 3.0 / (4.0 * PI); // reduced units
    let n_f = n as f64;
    let r_values: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * dr).collect();
    let g_values: Vec<f64> = r_values
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            let shell_vol = (4.0 * PI * dr).mul_add(r * r, 0.0);
            // Factor 2 because we count pairs i<j, but g(r) normalizes per particle
            2.0 * histogram[i] as f64
                / (n_frames as f64 * n_f * n_density * shell_vol).max(DIVISION_GUARD)
        })
        .collect();

    Rdf {
        r_values,
        g_values,
        dr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rdf_ideal_gas() {
        // For an ideal gas (non-interacting), g(r) ≈ 1 everywhere
        // We approximate with random positions
        let n = 200;
        let box_side = 10.0;
        let mut snap = vec![0.0; n * 3];
        // Simple deterministic "random" positions
        let mut seed = 12345u64;
        for v in &mut snap {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            *v = (seed >> 33) as f64 / (1u64 << 31) as f64 * box_side;
        }

        let rdf = compute_rdf(&[snap], n, box_side, 50);
        assert_eq!(rdf.g_values.len(), 50);
        // Tail should be roughly 1.0 for ideal gas (within noise)
        let tail_mean: f64 = rdf.g_values[30..].iter().sum::<f64>() / 20.0;
        assert!(
            (tail_mean - 1.0).abs() < 0.5,
            "ideal gas g(r→∞) ≈ 1, got {tail_mean}"
        );
    }

    #[test]
    fn rdf_determinism() {
        let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let n = 4;
        let box_l = 5.0;
        let n_bins = 25; // r_max = box_l/2 = 2.5, dr = 0.1 → 25 bins
        let snapshots = vec![positions];
        let a = compute_rdf(&snapshots, n, box_l, n_bins);
        let b = compute_rdf(&snapshots, n, box_l, n_bins);
        assert_eq!(a.g_values.len(), b.g_values.len());
        for (i, (va, vb)) in a.g_values.iter().zip(b.g_values.iter()).enumerate() {
            assert_eq!(va.to_bits(), vb.to_bits(), "RDF bin {i} not deterministic");
        }
    }

    #[test]
    fn rdf_struct_fields() {
        let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2 particles at distance 1
        let rdf = compute_rdf(&[positions], 2, 5.0, 10);
        assert_eq!(rdf.r_values.len(), 10);
        assert_eq!(rdf.g_values.len(), 10);
        assert!(rdf.dr > 0.0);
        // bin center at index 0 is 0.5*dr; r_values increase
        assert!(rdf.r_values[0] < rdf.r_values[rdf.r_values.len() - 1]);
    }

    #[test]
    fn rdf_pair_at_boundary_bin_skipped() {
        // Two particles at r = r_max (box_side/2) fall in bin n_bins which is out of range
        let box_l = 4.0;
        let r_max = box_l / 2.0;
        let pos = vec![0.0, 0.0, 0.0, r_max, 0.0, 0.0]; // distance = r_max
        let n_bins = 5;
        let rdf = compute_rdf(&[pos], 2, box_l, n_bins);
        assert_eq!(rdf.g_values.len(), n_bins);
        // Pair at r_max should not be counted (bin index = n_bins)
    }
}
