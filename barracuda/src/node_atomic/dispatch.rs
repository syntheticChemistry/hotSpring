// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dispatch routing: barraCuda `WgpuDevice` (local) or toadStool IPC (sovereign).
//!
//! The node-atomic composition uses barraCuda's `GpuHmcTrajectory` which handles
//! its own dispatch internally. This module provides the device acquisition path
//! and feature-flag routing for future toadStool sovereign dispatch migration.

use std::sync::Arc;
use barracuda::device::WgpuDevice;

/// Dispatch mode for compute operations.
pub enum DispatchMode {
    /// Direct local GPU via barraCuda's WgpuDevice.
    Local(Arc<WgpuDevice>),
    /// Future: toadStool IPC `compute.dispatch` (sovereign path).
    #[cfg(feature = "toadstool-dispatch")]
    Sovereign { socket_path: String },
}

impl DispatchMode {
    /// Create a local dispatch mode with a new f64-capable device.
    ///
    /// # Errors
    ///
    /// Returns an error if no f64-capable GPU adapter is found.
    pub fn local_auto() -> Result<Self, barracuda::error::BarracudaError> {
        let device = crate::block_on::block_on(WgpuDevice::new_f64_capable())?;
        Ok(Self::Local(Arc::new(device)))
    }

    /// Create local dispatch mode from an existing device.
    #[must_use]
    pub fn local(device: Arc<WgpuDevice>) -> Self {
        Self::Local(device)
    }

    /// Get the underlying WgpuDevice (panics if sovereign mode).
    #[must_use]
    pub fn device(&self) -> &Arc<WgpuDevice> {
        match self {
            Self::Local(d) => d,
            #[cfg(feature = "toadstool-dispatch")]
            Self::Sovereign { .. } => panic!("Cannot get local device in sovereign mode"),
        }
    }
}
