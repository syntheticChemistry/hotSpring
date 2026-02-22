// SPDX-License-Identifier: AGPL-3.0-only

//! Level spacing statistics and band detection.
//!
//! Mean level spacing ratio ⟨r⟩ distinguishes Poisson (localized) from GOE
//! (extended) statistics. Band detection groups eigenvalues by gaps.

/// Compute the mean level spacing ratio ⟨r⟩ from sorted eigenvalues.
///
/// `r_i` = `min(s_i`, s_{i+1}) / `max(s_i`, s_{i+1})
/// where `s_i` = λ_{i+1} − `λ_i`.
///
/// Known values:
/// - Poisson (localized): ⟨r⟩ = 2 ln 2 − 1 ≈ 0.3863
/// - GOE (extended + time-reversal): ⟨r⟩ ≈ 0.5307
///
/// # Provenance
/// Oganesyan & Huse (2007), Phys. Rev. B 75, 155111
/// Atas et al. (2013), Phys. Rev. Lett. 110, 084101
#[must_use]
pub fn level_spacing_ratio(eigenvalues: &[f64]) -> f64 {
    let n = eigenvalues.len();
    if n < 3 {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..n - 2 {
        let s1 = eigenvalues[i + 1] - eigenvalues[i];
        let s2 = eigenvalues[i + 2] - eigenvalues[i + 1];
        if s1 > 0.0 && s2 > 0.0 {
            let r = s1.min(s2) / s1.max(s2);
            sum += r;
            count += 1;
        }
    }

    if count > 0 {
        sum / f64::from(count)
    } else {
        0.0
    }
}

/// Poisson level spacing ratio (localized states).
pub const POISSON_R: f64 = 0.386_294_361_119_890_6; // 2 ln 2 - 1

/// GOE level spacing ratio (extended states with time-reversal symmetry).
pub const GOE_R: f64 = 0.5307;

/// Detect spectral bands from sorted eigenvalues.
///
/// Groups eigenvalues into bands separated by gaps. A "gap" is defined as a
/// spacing exceeding `gap_factor` times the median spacing. Returns a vector
/// of (`band_min`, `band_max`) pairs.
pub fn detect_bands(eigenvalues: &[f64], gap_factor: f64) -> Vec<(f64, f64)> {
    if eigenvalues.len() < 2 {
        if eigenvalues.len() == 1 {
            return vec![(eigenvalues[0], eigenvalues[0])];
        }
        return Vec::new();
    }

    let mut spacings: Vec<f64> = eigenvalues.windows(2).map(|w| w[1] - w[0]).collect();
    spacings.sort_by(f64::total_cmp);
    let median = spacings[spacings.len() / 2];

    let threshold = median * gap_factor;
    let mut bands = Vec::new();
    let mut band_start = eigenvalues[0];

    for w in eigenvalues.windows(2) {
        if w[1] - w[0] > threshold {
            bands.push((band_start, w[0]));
            band_start = w[1];
        }
    }
    if let Some(&last_ev) = eigenvalues.last() {
        bands.push((band_start, last_ev));
    }

    bands
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::{anderson_hamiltonian, find_all_eigenvalues};

    #[test]
    fn level_spacing_poisson() {
        // Strong Anderson disorder → Poisson statistics
        let (d, e) = anderson_hamiltonian(1000, 8.0, 42);
        let evals = find_all_eigenvalues(&d, &e);
        let r = level_spacing_ratio(&evals);
        assert!(
            (r - POISSON_R).abs() < 0.05,
            "Strong disorder: r={r:.4}, expected Poisson={POISSON_R:.4}"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact known value (0.0)
    fn level_spacing_ratio_two_eigenvalues() {
        let evals = vec![0.0, 1.0];
        let r = level_spacing_ratio(&evals);
        assert_eq!(r, 0.0, "n<3 should return 0");
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact known value (0.0)
    fn level_spacing_ratio_identical_eigenvalues() {
        let evals = vec![1.0, 1.0, 1.0, 1.0];
        let r = level_spacing_ratio(&evals);
        assert_eq!(
            r, 0.0,
            "all identical spacing s1=0 or s2=0 should give count=0"
        );
    }

    #[test]
    fn level_spacing_ratio_equally_spaced() {
        let evals: Vec<f64> = (0..10).map(f64::from).collect();
        let r = level_spacing_ratio(&evals);
        assert!((r - 1.0).abs() < 1e-10, "equal spacing gives r=1");
    }

    #[test]
    fn poisson_constant_value() {
        use std::f64::consts::LN_2;
        let expected = 2.0 * LN_2 - 1.0;
        assert!((POISSON_R - expected).abs() < 1e-12);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // constants sanity check
    fn goe_constant_in_range() {
        assert!(GOE_R > POISSON_R);
        assert!(GOE_R < 1.0);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // empty input → empty output
    fn detect_bands_empty() {
        let evals: Vec<f64> = vec![];
        let bands = detect_bands(&evals, 2.0);
        assert!(bands.is_empty());
    }

    #[test]
    #[allow(clippy::float_cmp, clippy::assertions_on_constants)] // exact known values
    fn detect_bands_single_eigenvalue() {
        let evals = vec![2.5];
        let bands = detect_bands(&evals, 2.0);
        assert_eq!(bands.len(), 1);
        assert_eq!(bands[0], (2.5, 2.5));
    }

    #[test]
    fn detect_bands_no_gap() {
        let evals: Vec<f64> = (0..20).map(f64::from).collect();
        let bands = detect_bands(&evals, 10.0);
        assert_eq!(bands.len(), 1);
        assert_eq!(bands[0], (0.0, 19.0));
    }

    #[test]
    fn detect_bands_large_gap() {
        let mut evals: Vec<f64> = (0..5).map(f64::from).collect();
        evals.extend((100..105).map(f64::from));
        let bands = detect_bands(&evals, 2.0);
        assert!(bands.len() >= 2);
        assert!((bands[0].1 - bands[0].0).abs() < 5.0);
        assert!((bands[1].1 - bands[1].0).abs() < 5.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn level_spacing_ratio_empty_returns_zero() {
        let evals: Vec<f64> = vec![];
        assert_eq!(level_spacing_ratio(&evals), 0.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn level_spacing_ratio_single_element_returns_zero() {
        let evals = vec![2.5];
        assert_eq!(level_spacing_ratio(&evals), 0.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn level_spacing_ratio_alternating_zeros_count_zero() {
        let evals = vec![0.0, 1.0, 1.0, 2.0];
        let r = level_spacing_ratio(&evals);
        assert_eq!(
            r, 0.0,
            "s1=0 or s2=0 skips that pair; this pattern may yield count=0"
        );
    }

    #[test]
    fn detect_bands_two_eigenvalues_one_band() {
        let evals = vec![1.0, 2.0];
        let bands = detect_bands(&evals, 2.0);
        assert_eq!(bands.len(), 1);
        assert_eq!(bands[0], (1.0, 2.0));
    }
}
