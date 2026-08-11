// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lattice geometry via barraCuda's `DiracGpuLayout`.
//!
//! Canonical 4D periodic topology — no local neighbor builder, no local
//! dimension conventions. Uses barraCuda's tested `[nt, nx, ny, nz]` ordering.

use barracuda::ops::lattice::dirac::DiracGpuLayout;
use barracuda::ops::lattice::gpu_hmc_types::GpuHmcConfig;

/// Geometry for a 4D periodic lattice, backed by barraCuda's conventions.
pub struct LatticeGeometry {
    pub dims: [usize; 4],
    pub volume: usize,
    pub n_links: usize,
}

impl LatticeGeometry {
    /// Build geometry from a `GpuHmcConfig`.
    #[must_use]
    pub fn new(config: &GpuHmcConfig) -> Self {
        let dims = [
            config.nt as usize,
            config.nx as usize,
            config.ny as usize,
            config.nz as usize,
        ];
        let volume = dims.iter().product::<usize>();
        let n_links = volume * 4;
        Self { dims, volume, n_links }
    }

    /// Produce a `DiracGpuLayout` with cold-start (identity) links.
    ///
    /// Identity SU(3): diag(1,1,1) in row-major complex form = 18 f64 per link.
    #[must_use]
    pub fn to_dirac_layout_cold(&self) -> DiracGpuLayout {
        let mut links = vec![0.0_f64; self.n_links * 18];
        for link_idx in 0..self.n_links {
            let base = link_idx * 18;
            links[base] = 1.0;      // Re(U[0][0])
            links[base + 8] = 1.0;  // Re(U[1][1])
            links[base + 16] = 1.0; // Re(U[2][2])
        }
        DiracGpuLayout::new(
            [self.dims[0], self.dims[1], self.dims[2], self.dims[3]],
            links,
        )
    }
}
