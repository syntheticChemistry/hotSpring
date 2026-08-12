// SPDX-License-Identifier: AGPL-3.0-or-later

use barracuda::device::driver_profile::Fp64Strategy;

use crate::gpu::GpuF64;

/// **DEPRECATED** — Substrate-aware FP64 strategy (adapter-name heuristic).
///
/// This function has been superseded by upstream barraCuda's
/// `DeviceCapabilities::fp64_strategy()` which now uses measured
/// `probe_f64_throughput_ratio` to select `Fp64Strategy::Concurrent`
/// when ratio > 8x. The adapter-name classification here was the
/// prototype — the upstream version is measurement-driven.
///
/// Production code should rely on `DeviceCapabilities::fp64_strategy()`
/// after calling `with_f64_throughput_ratio()` during device setup.
///
/// Retained for local validation binaries that still use the hotSpring
/// `gpu_hmc/` pipeline directly.
#[deprecated(
    since = "0.7.0",
    note = "Use barraCuda DeviceCapabilities::fp64_strategy() with measured throughput ratio"
)]
pub fn substrate_fp64_strategy(gpu: &GpuF64) -> Fp64Strategy {
    if let Ok(override_val) = std::env::var("HOTSPRING_FP64_STRATEGY") {
        match override_val.to_lowercase().as_str() {
            "native" => return Fp64Strategy::Native,
            "hybrid" => return Fp64Strategy::Hybrid,
            "concurrent" => return Fp64Strategy::Concurrent,
            _ => {}
        }
    }

    if gpu.full_df64_mode {
        return Fp64Strategy::Hybrid;
    }

    if !gpu.has_f64 {
        return Fp64Strategy::Hybrid;
    }

    // Rate-aware routing: FP64 is premium silicon, not the default path.
    // Full/Half-rate hardware (A100, V100, MI250) has enough f64 units to
    // saturate — use Native. Narrow-rate hardware (RX 6950 XT 1:16, RTX 3090
    // 1:64) uses Concurrent: bulk throughput on FP32 cores via DF64, precision-
    // critical reductions on the scarce FP64 units. Both core populations busy.
    //
    // The WG64 dispatch-overflow bug that previously forced Native is resolved:
    // all DF64 lattice shaders now use @workgroup_size(128), staying under the
    // 65535 workgroup-per-dimension limit at volumes up to 48⁴.
    match classify_fp64_rate_from_adapter(&gpu.adapter_name) {
        Fp64RateLocal::Full => Fp64Strategy::Native,
        Fp64RateLocal::Half => Fp64Strategy::Native,
        Fp64RateLocal::Narrow => Fp64Strategy::Concurrent,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Fp64RateLocal {
    /// 1:1 FP64:FP32 (datacenter: A100, H100).
    Full,
    /// 1:2 FP64:FP32 (Titan V / Volta, some Turing).
    Half,
    /// 1:32 or 1:64 FP64:FP32 (consumer Ampere, Ada, Turing).
    Narrow,
}

/// Classify FP64:FP32 rate from adapter name (mirrors metalForge probe logic).
pub(super) fn classify_fp64_rate_from_adapter(name: &str) -> Fp64RateLocal {
    let name_lower = name.to_lowercase();
    if name_lower.contains("a100") || name_lower.contains("h100") {
        Fp64RateLocal::Full
    } else if name_lower.contains("titan v")
        || name_lower.contains("v100")
        || name_lower.contains("gv100")
        || name_lower.contains("mi50")
        || name_lower.contains("mi100")
        || name_lower.contains("mi250")
    {
        Fp64RateLocal::Half
    } else {
        Fp64RateLocal::Narrow
    }
}

/// Silicon composition validation for Concurrent strategy.
///
/// Attempts to load the silicon profile for this adapter and check if the
/// FP32+FP64 composition multiplier confirms genuine parallel execution.
/// Returns `(confirmed, multiplier)` where confirmed=true means the
/// silicon profile proves >1.0x overlap between FP32 ALU and FP64 ALU.
///
/// When no profile exists (first run), returns `(false, 1.0)` — Concurrent
/// mode is still used but the speedup is unvalidated until profiling runs.
pub fn validate_concurrent_composition(adapter_name: &str) -> (bool, f64) {
    use crate::bench::silicon_profile::SiliconUnit;

    let workspace = std::env::var("HOTSPRING_ROOT").unwrap_or_else(|_| ".".to_string());
    let profile_dir = std::path::PathBuf::from(workspace)
        .join("profiles")
        .join("silicon");
    let safe_name = adapter_name
        .replace(['/', '\\', ' '], "_")
        .to_lowercase();
    let profile_path = profile_dir.join(format!("{safe_name}.json"));

    let json = match std::fs::read_to_string(&profile_path) {
        Ok(s) => s,
        Err(_) => return (false, 1.0),
    };

    let profile: crate::bench::silicon_profile::SiliconProfile =
        match serde_json::from_str(&json) {
            Ok(p) => p,
            Err(_) => return (false, 1.0),
        };

    let multiplier = profile
        .compositions
        .iter()
        .filter(|c| {
            (c.unit_a == SiliconUnit::Fp32Alu && c.unit_b == SiliconUnit::Fp64Alu)
                || (c.unit_a == SiliconUnit::Fp64Alu && c.unit_b == SiliconUnit::Fp32Alu)
        })
        .map(|c| c.multiplier)
        .fold(1.0f64, f64::max);

    (multiplier > 1.05, multiplier)
}
