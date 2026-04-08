// SPDX-License-Identifier: AGPL-3.0-or-later

//! Small statistical helpers shared by production binaries.
//!
//! Descriptive statistics delegate to [`barracuda::stats`] (barraCuda).

/// Arithmetic mean of a slice; empty slices yield `0.0`.
pub fn mean(v: &[f64]) -> f64 {
    barracuda::stats::mean(v)
}

/// Sample standard deviation (Bessel correction); fewer than two points yield `0.0`.
pub fn std_dev(v: &[f64]) -> f64 {
    barracuda::stats::correlation::std_dev(v).unwrap_or(0.0)
}
