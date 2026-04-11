// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared substrate state for overnight Chuna validation.

/// Key observables collected per substrate for cross-GPU comparison.
#[derive(Debug, Default)]
pub struct SubstrateResults {
    pub adapter_name: String,
    pub plaquettes: Vec<(String, f64)>,
    pub w0: Option<f64>,
    pub t0: Option<f64>,
    pub wall_seconds: f64,
}

/// Determine max lattice L for SU(3) based on available VRAM.
///
/// Memory per config: links + momenta + forces ~ 3 * 4 * L^4 * 4 * 18 * 8 bytes
/// (4D, 4 directions, 3x3 complex matrix = 18 f64, times 3 for link+mom+force).
pub fn max_lattice_l(max_buffer_bytes: u64) -> usize {
    let safety_margin = 4;
    let bytes_per_site: u64 = 4 * 18 * 8 * safety_margin;
    let max_sites = max_buffer_bytes / bytes_per_site;
    let l = (max_sites as f64).powf(0.25).floor() as usize;
    l.clamp(8, 64)
}
