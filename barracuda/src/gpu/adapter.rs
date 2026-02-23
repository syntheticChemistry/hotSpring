// SPDX-License-Identifier: AGPL-3.0-only

//! GPU adapter discovery and selection.
//!
//! Runtime capability probing — no hardcoded GPU assumptions. The adapter
//! is selected by environment variable or auto-detected based on `SHADER_F64`
//! support.

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
pub fn create_instance() -> wgpu::Instance {
    let backends = match std::env::var("HOTSPRING_WGPU_BACKEND").as_deref() {
        Ok("vulkan") => wgpu::Backends::VULKAN,
        Ok("metal") => wgpu::Backends::METAL,
        Ok("dx12") => wgpu::Backends::DX12,
        _ => wgpu::Backends::all(),
    };
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    })
}

/// Enumerate all available GPU adapters.
///
/// Returns a summary for each adapter including name, driver, and
/// `SHADER_F64` support. Use the `index` field with
/// `HOTSPRING_GPU_ADAPTER=<index>` to target a specific GPU.
#[must_use]
pub fn enumerate_adapters() -> Vec<AdapterInfo> {
    let instance = create_instance();
    instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .enumerate()
        .map(|(i, adapter)| {
            let info = adapter.get_info();
            let features = adapter.features();
            AdapterInfo {
                index: i,
                name: info.name.clone(),
                driver: info.driver.clone(),
                has_f64: features.contains(wgpu::Features::SHADER_F64),
                has_timestamps: features.contains(wgpu::Features::TIMESTAMP_QUERY),
                device_type: info.device_type,
            }
        })
        .collect()
}

/// Select an adapter based on the `HOTSPRING_GPU_ADAPTER` / `BARRACUDA_GPU_ADAPTER`
/// environment variables. Falls back to auto-detection (discrete + `SHADER_F64` first).
///
/// # Errors
///
/// Returns [`crate::error::HotSpringError`] if no compatible adapter is found.
pub fn select_adapter() -> Result<wgpu::Adapter, crate::error::HotSpringError> {
    let selector = std::env::var("HOTSPRING_GPU_ADAPTER")
        .or_else(|_| std::env::var("BARRACUDA_GPU_ADAPTER"))
        .unwrap_or_default()
        .trim()
        .to_lowercase();

    let instance = create_instance();
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all());
    if adapters.is_empty() {
        return Err(crate::error::HotSpringError::NoAdapter);
    }

    if selector.is_empty() || selector == "auto" {
        auto_select(adapters)
    } else if let Ok(idx) = selector.parse::<usize>() {
        select_by_index_or_name(adapters, idx, &selector)
    } else {
        select_by_name(adapters, &selector)
    }
}

fn auto_select(
    adapters: Vec<wgpu::Adapter>,
) -> Result<wgpu::Adapter, crate::error::HotSpringError> {
    let mut chosen: Option<wgpu::Adapter> = None;
    let mut fallback: Option<wgpu::Adapter> = None;
    for a in adapters {
        if a.features().contains(wgpu::Features::SHADER_F64) {
            if a.get_info().device_type == wgpu::DeviceType::DiscreteGpu && chosen.is_none() {
                chosen = Some(a);
            } else if fallback.is_none() {
                fallback = Some(a);
            }
        }
    }
    chosen
        .or(fallback)
        .ok_or(crate::error::HotSpringError::NoAdapter)
}

fn select_by_index_or_name(
    adapters: Vec<wgpu::Adapter>,
    idx: usize,
    selector: &str,
) -> Result<wgpu::Adapter, crate::error::HotSpringError> {
    if idx < adapters.len() {
        adapters
            .into_iter()
            .nth(idx)
            .ok_or(crate::error::HotSpringError::NoAdapter)
    } else {
        adapters
            .into_iter()
            .find(|a| a.get_info().name.to_ascii_lowercase().contains(selector))
            .ok_or_else(|| {
                crate::error::HotSpringError::DeviceCreation(format!(
                    "No adapter matching '{selector}' (tried as index {idx} and name)"
                ))
            })
    }
}

fn select_by_name(
    adapters: Vec<wgpu::Adapter>,
    selector: &str,
) -> Result<wgpu::Adapter, crate::error::HotSpringError> {
    adapters
        .into_iter()
        .find(|a| a.get_info().name.to_ascii_lowercase().contains(selector))
        .ok_or_else(|| {
            crate::error::HotSpringError::DeviceCreation(format!(
                "No adapter matching '{selector}'"
            ))
        })
}
