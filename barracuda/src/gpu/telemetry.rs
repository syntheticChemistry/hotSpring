// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU telemetry — power, temperature, utilization, VRAM.
//!
//! Delegates to toadStool `gpu.query_telemetry` via NUCLEUS when available.
//! Returns zeros when monitoring is unavailable (headless / no ember).

use super::GpuF64;

impl GpuF64 {
    /// Query current GPU power draw, temperature, utilization, and VRAM.
    ///
    /// Returns `(power_watts, temp_celsius, utilization_pct, vram_used_mib)`.
    /// Returns zeros if toadStool telemetry is unavailable.
    #[must_use]
    pub fn query_gpu_power() -> (f64, f64, f64, f64) {
        query_via_toadstool().unwrap_or((0.0, 0.0, 0.0, 0.0))
    }

    /// Snapshot of current GPU VRAM usage in MiB.
    #[must_use]
    pub fn gpu_vram_used_mib() -> f64 {
        let (_, _, _, vram) = Self::query_gpu_power();
        vram
    }
}

/// Query GPU telemetry via toadStool `gpu.query_telemetry` (NUCLEUS-routed).
fn query_via_toadstool() -> Option<(f64, f64, f64, f64)> {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        log::debug!(
            "GpuF64::query_gpu_power uses toadStool gpu.query_telemetry; \
             local nvidia-smi parsing removed"
        );
    });

    let ctx = crate::primal_bridge::NucleusContext::detect();
    let resp = ctx
        .call_by_capability("compute", "gpu.query_telemetry", serde_json::json!({}))
        .ok()?;

    let gpu = resp.get("gpus")?.as_array()?.first()?;
    let power = json_f64(gpu.get("power_watts")?)?;
    let temp = json_f64(gpu.get("temperature_celsius")?)?;
    let util = json_f64(gpu.get("utilization_percent")?)?;
    let vram_bytes = json_f64(gpu.get("vram_used_bytes")?).unwrap_or(0.0);
    let vram_mib = vram_bytes / (1024.0 * 1024.0);
    Some((power, temp, util, vram_mib))
}

fn json_f64(v: &serde_json::Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_i64().map(|n| n as f64))
}
