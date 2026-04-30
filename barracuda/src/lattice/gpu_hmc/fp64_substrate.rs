// SPDX-License-Identifier: AGPL-3.0-or-later

use barracuda::device::driver_profile::Fp64Strategy;

use crate::gpu::GpuF64;

/// Substrate-aware FP64 strategy using adapter-name hardware classification.
///
/// barraCuda's `DeviceCapabilities::fp64_strategy()` only distinguishes Native
/// vs Hybrid based on f64 probe results. It never returns Concurrent because
/// it doesn't know the FP64:FP32 hardware ratio. This function classifies
/// the rate from the adapter name (mirroring metalForge's substrate probe).
///
/// On narrow-rate GPUs (RTX 3090 1:64, RX 6950 XT 1:16), this returns
/// `Concurrent`: saturate native f64 AND use DF64 on FP32 cores for bulk
/// computation. This activates the hand-written df64 force/plaquette/KE shaders,
/// unlocking 4-6x more effective f64 TFLOPS from the FP32 silicon.
pub fn substrate_fp64_strategy(gpu: &GpuF64) -> Fp64Strategy {
    if gpu.full_df64_mode {
        return Fp64Strategy::Hybrid;
    }

    let rate = classify_fp64_rate_from_adapter(&gpu.adapter_name);
    match rate {
        Fp64RateLocal::Full | Fp64RateLocal::Half => Fp64Strategy::Native,
        Fp64RateLocal::Narrow if gpu.has_f64 => {
            Fp64Strategy::Concurrent
        }
        Fp64RateLocal::Narrow => Fp64Strategy::Hybrid,
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
