// SPDX-License-Identifier: AGPL-3.0-or-later

//! Node-atomic composition layer for lattice QCD.
//!
//! Replaces the fossil's 37-module `gpu_hmc/` tree with thin wrappers around
//! upstream primal capabilities:
//!
//! - **geometry** — `barraCuda::ops::lattice::DiracGpuLayout` for topology
//! - **dispatch** — `barraCuda::device::WgpuDevice` or toadStool IPC
//! - **observables** — Plaquette, KE, Polyakov via barraCuda reduce
//! - **trajectory** — HMC orchestration consuming `GpuHmcTrajectory`

pub mod dispatch;
pub mod geometry;
pub mod observables;
pub mod trajectory;

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::lattice::gpu_hmc_trajectory::GpuHmcTrajectory;
use barracuda::ops::lattice::gpu_hmc_types::{GpuHmcBuffers, GpuHmcConfig, GpuHmcResult};

pub use geometry::LatticeGeometry;
pub use observables::Observables;
pub use trajectory::TrajectoryRunner;

/// Node-atomic QCD state: thin orchestration over barraCuda's HMC infrastructure.
///
/// Owns the device, config, buffers, and trajectory pipeline. All GPU work is
/// delegated to barraCuda — no local shaders, no local dispatch splitting.
pub struct NodeAtomicQcd {
    pub device: Arc<WgpuDevice>,
    pub config: GpuHmcConfig,
    pub buffers: GpuHmcBuffers,
    pub trajectory: GpuHmcTrajectory,
    pub geometry: LatticeGeometry,
}

impl NodeAtomicQcd {
    /// Create a new node-atomic QCD state for a given lattice configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if device creation, buffer allocation, or pipeline
    /// compilation fails.
    pub fn new(config: GpuHmcConfig, seed: u64) -> Result<Self, barracuda::error::BarracudaError> {
        let device = Arc::new(
            crate::block_on::block_on(barracuda::device::WgpuDevice::new_f64_capable())?
        );
        let geometry = LatticeGeometry::new(&config);
        let trajectory = GpuHmcTrajectory::with_seed(device.clone(), config.clone(), seed)?;
        let buffers = GpuHmcBuffers::new(&device, &config)?;

        Ok(Self {
            device,
            config,
            buffers,
            trajectory,
            geometry,
        })
    }

    /// Create with an existing device (for shared-device scenarios).
    pub fn with_device(
        device: Arc<WgpuDevice>,
        config: GpuHmcConfig,
        seed: u64,
    ) -> Result<Self, barracuda::error::BarracudaError> {
        let geometry = LatticeGeometry::new(&config);
        let trajectory = GpuHmcTrajectory::with_seed(device.clone(), config.clone(), seed)?;
        let buffers = GpuHmcBuffers::new(&device, &config)?;

        Ok(Self {
            device,
            config,
            buffers,
            trajectory,
            geometry,
        })
    }

    /// Initialize topology (neighbors + staggered phases) on the GPU.
    pub fn upload_topology(&self) {
        let layout = self.geometry.to_dirac_layout_cold();
        self.trajectory.upload_topology(&layout, &self.buffers);
    }

    /// Seed the GPU RNG state.
    pub fn seed_rng(&self, seed: u32) {
        self.trajectory.seed_rng(seed, &self.buffers);
    }

    /// Run one HMC trajectory. Returns accept/reject + observables.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch or readback fails.
    pub fn run_trajectory(&self) -> Result<GpuHmcResult, barracuda::error::BarracudaError> {
        self.trajectory.run(&self.buffers)
    }

    /// Number of lattice sites (Nx × Ny × Nz × Nt).
    #[must_use]
    pub fn volume(&self) -> usize {
        self.config.nx as usize
            * self.config.ny as usize
            * self.config.nz as usize
            * self.config.nt as usize
    }
}
