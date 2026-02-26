// SPDX-License-Identifier: AGPL-3.0-only

//! Bridge between forge substrates and barracuda device creation.
//!
//! Connects the capability-based substrate model (forge) to barracuda's
//! GPU device management (`WgpuDevice`). When toadstool absorbs forge,
//! this bridge becomes the integration point — substrate discovery feeds
//! directly into device creation.
//!
//! # Absorption path
//!
//! Forge `Substrate` → barracuda `WgpuDevice`. The capability model
//! (`F64Compute`, `ShaderDispatch`, etc.) maps directly to barracuda's
//! feature requirements. The bridge provides:
//!
//! 1. `create_device` — forge substrate → barracuda `WgpuDevice`
//! 2. `best_f64_gpu` — find the best f64-capable GPU from inventory
//! 3. `substrate_from_device` — wrap existing barracuda device as substrate

use crate::substrate::{Capability, Identity, Properties, Substrate, SubstrateKind};
use barracuda::device::WgpuDevice;

/// Create a barracuda `WgpuDevice` from a forge GPU substrate.
///
/// Uses the adapter index from the substrate identity to select the
/// correct wgpu adapter via barracuda's standard creation path.
///
/// # Errors
///
/// Returns `None` if the substrate is not a GPU, has no adapter index,
/// or if device creation fails.
#[must_use]
pub fn create_device(substrate: &Substrate) -> Option<WgpuDevice> {
    if substrate.kind != SubstrateKind::Gpu {
        return None;
    }

    let adapter_index = substrate.identity.adapter_index?;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;

    rt.block_on(async { WgpuDevice::from_adapter_index(adapter_index).await.ok() })
}

/// Find the best f64-capable GPU from a substrate inventory.
///
/// Prefers discrete GPUs. Among multiple f64-capable GPUs, picks the first.
#[must_use]
pub fn best_f64_gpu(substrates: &[Substrate]) -> Option<&Substrate> {
    substrates
        .iter()
        .find(|s| s.kind == SubstrateKind::Gpu && s.has(&Capability::F64Compute))
}

/// Wrap an existing barracuda `WgpuDevice` as a forge substrate.
///
/// Enables code that already has a barracuda device to participate
/// in forge's capability-based dispatch. The adapter info populates
/// identity and capabilities.
#[must_use]
pub fn substrate_from_device(device: &WgpuDevice) -> Substrate {
    let info = device.adapter_info();
    let has_f64 = device.has_f64_shaders();

    let mut capabilities = vec![
        Capability::F32Compute,
        Capability::ShaderDispatch,
        Capability::PcieTransfer,
        Capability::StreamingStage,
    ];
    if has_f64 {
        capabilities.push(Capability::F64Compute);
        capabilities.push(Capability::ScalarReduce);
        capabilities.push(Capability::SparseSpMV);
        capabilities.push(Capability::Eigensolve);
        capabilities.push(Capability::ConjugateGradient);
    }
    capabilities.push(Capability::DF64Compute);

    Substrate {
        kind: SubstrateKind::Gpu,
        identity: Identity {
            name: info.name.clone(),
            driver: Some(format!("{} ({})", info.driver, info.driver_info)),
            backend: Some(format!("{:?}", info.backend)),
            adapter_index: None,
            device_node: None,
            pci_id: None,
        },
        properties: Properties {
            has_f64,
            has_timestamps: false,
            has_df64: true,
            ..Properties::default()
        },
        capabilities,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::inventory;

    #[test]
    fn best_f64_gpu_returns_gpu_if_available() {
        let subs = inventory::discover();
        let gpu = best_f64_gpu(&subs);
        if let Some(g) = gpu {
            assert_eq!(g.kind, SubstrateKind::Gpu);
            assert!(g.has(&Capability::F64Compute));
        }
    }

    #[test]
    fn best_f64_gpu_returns_none_for_cpu_only() {
        let subs = vec![crate::probe::probe_cpu()];
        assert!(best_f64_gpu(&subs).is_none());
    }

    #[test]
    fn best_f64_finds_first_capable() {
        let gpu_f32 = Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity::named("f32-only GPU"),
            properties: Properties::default(),
            capabilities: vec![Capability::F32Compute, Capability::ShaderDispatch],
        };
        let gpu_f64 = Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity::named("f64 GPU"),
            properties: Properties {
                has_f64: true,
                ..Properties::default()
            },
            capabilities: vec![
                Capability::F64Compute,
                Capability::F32Compute,
                Capability::ShaderDispatch,
            ],
        };
        let cpu = crate::probe::probe_cpu();

        let subs = vec![gpu_f32, gpu_f64, cpu];
        let best = best_f64_gpu(&subs).expect("should find f64 GPU");
        assert_eq!(best.identity.name, "f64 GPU");
    }
}
