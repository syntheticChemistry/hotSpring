// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU adapter discovery and selection.
//!
//! hotSpring-specific env overrides (`HOTSPRING_GPU_*`) wrap barraCuda's
//! [`WgpuDevice`] adapter selection (`BARRACUDA_GPU_ADAPTER`, registry scoring).

use barracuda::device::WgpuDevice;
use log::warn;

/// Summary of a discovered GPU adapter.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    /// Enumeration index (stable within a single run).
    pub index: usize,
    /// Adapter name as reported by the driver.
    pub name: String,
    /// Vulkan driver name (e.g. `"NVIDIA"`, `"NVK"`, `"radv"`).
    pub driver: String,
    /// Whether `SHADER_F64` is supported.
    pub has_f64: bool,
    /// Whether `TIMESTAMP_QUERY` is supported.
    pub has_timestamps: bool,
    /// Adapter device type (discrete, integrated, software, etc.).
    pub device_type: wgpu::DeviceType,
    /// Maximum buffer size in bytes (from adapter limits; proxy for VRAM).
    pub memory_bytes: u64,
}

impl std::fmt::Display for AdapterInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let f64_tag = if self.has_f64 { "f64" } else { "f32" };
        let kind = match self.device_type {
            wgpu::DeviceType::DiscreteGpu => "discrete",
            wgpu::DeviceType::IntegratedGpu => "integrated",
            wgpu::DeviceType::VirtualGpu => "virtual",
            wgpu::DeviceType::Cpu => "cpu",
            wgpu::DeviceType::Other => "other",
        };
        write!(
            f,
            "[{}] {} ({}, {}, {})",
            self.index, self.name, self.driver, kind, f64_tag
        )
    }
}

/// Create a wgpu instance with the backend configured via `HOTSPRING_WGPU_BACKEND`.
///
/// hotSpring-only; barraCuda device creation uses `Backends::all()`.
pub fn create_instance() -> wgpu::Instance {
    let backends = match std::env::var("HOTSPRING_WGPU_BACKEND").as_deref() {
        Ok("vulkan") => wgpu::Backends::VULKAN,
        Ok("metal") => wgpu::Backends::METAL,
        Ok("dx12") => wgpu::Backends::DX12,
        _ => wgpu::Backends::all(),
    };
    wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    })
}

/// Enumerate all available GPU adapters.
///
/// Returns a summary for each adapter including name, driver, and
/// `SHADER_F64` support. Use the `index` field with
/// `HOTSPRING_GPU_ADAPTER=<index>` to target a specific GPU.
///
/// Set `HOTSPRING_NO_GPU=1` to skip enumeration entirely (returns empty).
#[must_use]
pub fn enumerate_adapters() -> Vec<AdapterInfo> {
    if std::env::var("HOTSPRING_NO_GPU").is_ok() {
        return Vec::new();
    }

    let result = std::panic::catch_unwind(|| {
        let instance = create_instance();
        crate::block_on::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .enumerate()
            .map(|(i, adapter): (usize, wgpu::Adapter)| {
                let info = adapter.get_info();
                let features = adapter.features();
                let limits = adapter.limits();
                AdapterInfo {
                    index: i,
                    name: info.name.clone(),
                    driver: info.driver.clone(),
                    has_f64: features.contains(wgpu::Features::SHADER_F64),
                    has_timestamps: features.contains(wgpu::Features::TIMESTAMP_QUERY),
                    device_type: info.device_type,
                    memory_bytes: limits.max_buffer_size,
                }
            })
            .collect()
    });

    if let Ok(adapters) = result {
        adapters
    } else {
        warn!("GPU adapter enumeration failed (broken Vulkan/ICD?). Continuing CPU-only.");
        Vec::new()
    }
}

fn sort_f64_adapters(adapters: &mut [AdapterInfo]) {
    adapters.sort_by(|a, b| {
        let mem_cmp = b.memory_bytes.cmp(&a.memory_bytes);
        if mem_cmp != std::cmp::Ordering::Equal {
            return mem_cmp;
        }
        let discrete = |x: &AdapterInfo| x.device_type == wgpu::DeviceType::DiscreteGpu;
        discrete(b).cmp(&discrete(a))
    });
}

/// Discover the best available GPU adapter by memory/capability.
///
/// Returns the adapter identifier (index as string) suitable for
/// `HOTSPRING_GPU_ADAPTER`, or `None` if no compatible adapter exists.
#[must_use]
pub fn discover_best_adapter() -> Option<String> {
    let mut adapters = enumerate_adapters();
    adapters.retain(|a| a.has_f64);
    if adapters.is_empty() {
        return None;
    }
    sort_f64_adapters(&mut adapters);
    Some(adapters[0].index.to_string())
}

/// Discover primary and secondary GPU adapters by memory/capability.
///
/// Env var override: if `HOTSPRING_GPU_PRIMARY` or `HOTSPRING_GPU_SECONDARY`
/// are set, those values are used instead of discovery for that slot.
#[must_use]
pub fn discover_primary_and_secondary_adapters() -> (Option<String>, Option<String>) {
    let primary_override = std::env::var("HOTSPRING_GPU_PRIMARY").ok();
    let secondary_override = std::env::var("HOTSPRING_GPU_SECONDARY").ok();

    if primary_override.is_some() && secondary_override.is_some() {
        return (primary_override, secondary_override);
    }

    let mut adapters = enumerate_adapters();
    adapters.retain(|a| a.has_f64);
    sort_f64_adapters(&mut adapters);

    let primary = primary_override.or_else(|| adapters.first().map(|a| a.index.to_string()));
    let secondary = secondary_override.or_else(|| {
        adapters
            .iter()
            .skip(1)
            .find(|a| {
                primary
                    .as_ref()
                    .is_none_or(|p| p != &a.index.to_string() && p != &a.name)
            })
            .map(|a| a.index.to_string())
    });

    (primary, secondary)
}

/// Resolve adapter selector from hotSpring / barraCuda env vars.
///
/// Supports comma-separated priority lists: `"3090,titan,auto"`.
fn adapter_selector_tokens() -> Vec<String> {
    let raw = std::env::var("HOTSPRING_GPU_ADAPTER")
        .or_else(|_| std::env::var("BARRACUDA_GPU_ADAPTER"))
        .unwrap_or_default();
    let tokens: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if tokens.is_empty() {
        vec!["auto".to_string()]
    } else {
        tokens
    }
}

/// Create a barraCuda [`WgpuDevice`] using env-based adapter selection.
///
/// Tries each token from `HOTSPRING_GPU_ADAPTER` / `BARRACUDA_GPU_ADAPTER`
/// (comma-separated) via [`WgpuDevice::with_adapter_selector`].
pub(super) async fn create_wgpu_from_env()
-> Result<barracuda::device::WgpuDevice, crate::error::HotSpringError> {
    for token in adapter_selector_tokens() {
        match WgpuDevice::with_adapter_selector(&token).await {
            Ok(dev) => return Ok(dev),
            Err(e) => {
                log::debug!("adapter selector '{token}' failed: {e}");
            }
        }
    }
    Err(crate::error::HotSpringError::NoAdapter)
}

/// Create a barraCuda [`WgpuDevice`] from an explicit selector hint.
pub(super) async fn create_wgpu_from_hint(
    hint: &str,
) -> Result<barracuda::device::WgpuDevice, crate::error::HotSpringError> {
    WgpuDevice::with_adapter_selector(hint.trim())
        .await
        .map_err(Into::into)
}

/// Create a barraCuda [`WgpuDevice`] from a pre-selected wgpu adapter.
///
/// Matches the adapter against enumeration order and delegates to
/// [`WgpuDevice::from_adapter_index`].
pub(super) async fn create_wgpu_from_adapter(
    selected: wgpu::Adapter,
) -> Result<barracuda::device::WgpuDevice, crate::error::HotSpringError> {
    let target = selected.get_info();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> =
        crate::block_on::block_on(instance.enumerate_adapters(wgpu::Backends::all()));

    let idx = adapters.iter().position(|a| {
        let info = a.get_info();
        info.name == target.name && info.vendor == target.vendor && info.device == target.device
    });

    match idx {
        Some(index) => WgpuDevice::from_adapter_index(index)
            .await
            .map_err(Into::into),
        None => Err(crate::error::HotSpringError::NoAdapter),
    }
}
