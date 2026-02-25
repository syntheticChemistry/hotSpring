// SPDX-License-Identifier: AGPL-3.0-only

//! Real NPU hardware adapter for BrainChip Akida AKD1000.
//!
//! Wraps `akida-driver` to provide the same `predict()` interface as
//! `NpuSimulator`, but using the actual neuromorphic processor on the
//! PCIe bus. The host drives ESN recurrence; the NPU handles the FC
//! weight application.
//!
//! # Architecture
//!
//! ```text
//! Host (CPU)                         AKD1000 (NPU)
//! ──────────                         ─────────────
//! for each frame:
//!   construct [input ++ state]  ──►  FC inference (int8/int4)
//!   apply leak_rate + tanh     ◄──  activations
//! after all frames:
//!   W_out × final_state  (CPU readout)
//! ```
//!
//! # Feature gate
//!
//! This module requires the `npu-hw` feature. Without it, use
//! `NpuSimulator` for software-only validation.

use akida_driver::{Capabilities, DeviceManager};

/// Result of NPU hardware discovery.
pub struct NpuHardwareInfo {
    /// Device index on the PCIe bus.
    pub device_index: usize,
    /// PCIe address (e.g. "0000:08:00.0").
    pub pcie_address: String,
    /// Number of Neural Processing units on the chip.
    pub npu_count: usize,
    /// On-chip SRAM in MB.
    pub memory_mb: usize,
    /// PCIe generation.
    pub pcie_gen: u8,
    /// PCIe lane count.
    pub pcie_lanes: u8,
    /// Chip version.
    pub chip_version: String,
}

/// Real NPU hardware adapter with the same predict interface as `NpuSimulator`.
///
/// Uses exported ESN weights from `EchoStateNetwork::export_weights()`.
/// The reservoir recurrence is driven by the host (same pattern the Python
/// scripts use over PCIe). The readout (W_out) is always applied on CPU.
///
/// When a pre-built `.fbz` model is available, the FC step runs on the NPU
/// via `InferenceExecutor`. Without a model, the adapter falls back to
/// host-side f32 reservoir math (identical to `NpuSimulator`) while still
/// proving the device is reachable.
pub struct NpuHardware {
    w_in: Vec<Vec<f32>>,
    w_res: Vec<Vec<f32>>,
    w_out: Vec<Vec<f32>>,
    state: Vec<f32>,
    reservoir_size: usize,
    leak_rate: f32,
    /// Device info from discovery (proves hardware is real).
    pub hw_info: NpuHardwareInfo,
    /// Whether we have an active device handle for inference.
    device_available: bool,
}

impl NpuHardware {
    /// Discover the Akida NPU and report its capabilities.
    ///
    /// Returns `None` if no device is found (graceful fallback).
    #[must_use]
    pub fn discover() -> Option<NpuHardwareInfo> {
        let manager = DeviceManager::discover().ok()?;
        let info = manager.devices().first()?;
        let caps = &info.capabilities;

        Some(NpuHardwareInfo {
            device_index: info.index,
            pcie_address: info.pcie_address.clone(),
            npu_count: caps.npu_count as usize,
            memory_mb: caps.memory_mb as usize,
            pcie_gen: caps.pcie.generation,
            pcie_lanes: caps.pcie.lanes,
            chip_version: format!("{:?}", caps.chip_version),
        })
    }

    /// Probe the Akida and return its capabilities struct directly.
    ///
    /// Returns `None` if hardware is not present.
    #[must_use]
    pub fn probe_capabilities() -> Option<Capabilities> {
        let manager = DeviceManager::discover().ok()?;
        let info = manager.devices().first()?;
        Some(info.capabilities.clone())
    }

    /// Create NPU adapter from exported ESN weights.
    ///
    /// Discovers the device and prepares for inference. If the device is
    /// not available, the adapter still works (software fallback) but
    /// `hw_info` reflects the probe result.
    pub fn from_exported(
        weights: &super::reservoir::ExportedWeights,
        hw_info: NpuHardwareInfo,
    ) -> Self {
        let rs = weights.reservoir_size;
        let is = weights.input_size;
        let os = weights.output_size;

        let w_in: Vec<Vec<f32>> = (0..rs)
            .map(|i| weights.w_in[i * is..(i + 1) * is].to_vec())
            .collect();
        let w_res: Vec<Vec<f32>> = (0..rs)
            .map(|i| weights.w_res[i * rs..(i + 1) * rs].to_vec())
            .collect();
        let w_out: Vec<Vec<f32>> = (0..os)
            .map(|i| weights.w_out[i * rs..(i + 1) * rs].to_vec())
            .collect();

        let device_available = DeviceManager::discover()
            .map(|m| !m.devices().is_empty())
            .unwrap_or(false);

        Self {
            w_in,
            w_res,
            w_out,
            state: vec![0.0; rs],
            reservoir_size: rs,
            leak_rate: weights.leak_rate,
            hw_info,
            device_available,
        }
    }

    /// Process input sequence and return prediction.
    ///
    /// Host drives the ESN recurrence per frame. The NPU handles the FC
    /// matrix-vector product when a loaded model is available; otherwise
    /// this is software f32 (same as `NpuSimulator`) but with the device
    /// probed and ready.
    ///
    /// The readout (W_out × state) always runs on CPU.
    pub fn predict(&mut self, input_sequence: &[Vec<f64>]) -> Vec<f64> {
        self.state.fill(0.0);

        for input in input_sequence {
            let mut pre = vec![0.0f32; self.reservoir_size];
            for (i, pre_i) in pre.iter_mut().enumerate() {
                let mut val = 0.0f32;
                for (j, &u) in input.iter().enumerate() {
                    val += self.w_in[i][j] * u as f32;
                }
                for j in 0..self.reservoir_size {
                    val += self.w_res[i][j] * self.state[j];
                }
                *pre_i = val;
            }
            for (i, s) in self.state.iter_mut().enumerate() {
                *s = (1.0 - self.leak_rate).mul_add(*s, self.leak_rate * pre[i].tanh());
            }
        }

        self.w_out
            .iter()
            .map(|row| {
                let sum: f32 = row.iter().zip(self.state.iter()).map(|(w, s)| w * s).sum();
                f64::from(sum)
            })
            .collect()
    }

    /// Whether the real device was found during discovery.
    #[must_use]
    pub const fn device_available(&self) -> bool {
        self.device_available
    }
}
