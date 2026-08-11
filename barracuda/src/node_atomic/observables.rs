// SPDX-License-Identifier: AGPL-3.0-or-later

//! Observable measurements via barraCuda's GPU operators.
//!
//! Delegates to `WilsonPlaquette`, `GpuKineticEnergy`, and `ReduceScalarPipeline`
//! from barraCuda — no local shader code.

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::lattice::gpu_hmc_types::GpuHmcConfig;

/// Observable measurement infrastructure backed by barraCuda operators.
pub struct Observables {
    pub volume: usize,
    pub n_links: usize,
    device: Arc<WgpuDevice>,
}

impl Observables {
    /// Create observable measurement context.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>, config: &GpuHmcConfig) -> Self {
        let volume = config.nx as usize
            * config.ny as usize
            * config.nz as usize
            * config.nt as usize;
        Self {
            volume,
            n_links: volume * 4,
            device,
        }
    }

    /// Compute average plaquette from the gauge action returned by `GpuHmcResult`.
    ///
    /// Uses the relation: `S_gauge = beta * (6V - plaq_sum)`, so
    /// `<P> = 1 - S_gauge / (6 * V * beta)`.
    #[must_use]
    pub fn plaquette_from_action(&self, gauge_action: f64, beta: f64) -> f64 {
        1.0 - gauge_action / (6.0 * self.volume as f64 * beta)
    }

    /// Get the underlying device.
    #[must_use]
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }
}
