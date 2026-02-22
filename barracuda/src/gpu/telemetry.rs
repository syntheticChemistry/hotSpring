// SPDX-License-Identifier: AGPL-3.0-only

//! GPU telemetry — power, temperature, utilization, VRAM.
//!
//! Capability-based: probes available monitoring tools at runtime.
//! Returns zeros when monitoring is unavailable (open-source drivers).

use std::process::Command;

use super::GpuF64;

impl GpuF64 {
    /// Query current GPU power draw, temperature, utilization, and VRAM.
    ///
    /// Returns `(power_watts, temp_celsius, utilization_pct, vram_used_mib)`.
    /// Returns zeros if monitoring tools are unavailable.
    #[must_use]
    pub fn query_gpu_power() -> (f64, f64, f64, f64) {
        query_nvidia_smi().unwrap_or((0.0, 0.0, 0.0, 0.0))
    }

    /// Snapshot of current GPU VRAM usage in MiB.
    #[must_use]
    pub fn gpu_vram_used_mib() -> f64 {
        let (_, _, _, vram) = Self::query_gpu_power();
        vram
    }
}

/// Query nvidia-smi for GPU telemetry. Returns `None` if unavailable.
fn query_nvidia_smi() -> Option<(f64, f64, f64, f64)> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let s = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = s.trim().split(", ").collect();
    if parts.len() >= 4 {
        Some((
            parts[0].trim().parse().unwrap_or(0.0),
            parts[1].trim().parse().unwrap_or(0.0),
            parts[2].trim().parse().unwrap_or(0.0),
            parts[3].trim().parse().unwrap_or(0.0),
        ))
    } else {
        None
    }
}
