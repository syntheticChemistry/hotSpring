// SPDX-License-Identifier: AGPL-3.0-or-later

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
/// On broken Vulkan stacks (common on headless HPC nodes), panics from
/// the native ICD loader are caught and an empty list is returned.
#[must_use]
pub fn enumerate_adapters() -> Vec<AdapterInfo> {
    if std::env::var("HOTSPRING_NO_GPU").is_ok() {
        return Vec::new();
    }

    let result = std::panic::catch_unwind(|| {
        let instance = create_instance();
        pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
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
        eprintln!(
            "warning: GPU adapter enumeration failed (broken Vulkan/ICD?). Continuing CPU-only."
        );
        Vec::new()
    }
}

/// Discover the best available GPU adapter by memory/capability.
///
/// Enumerates all adapters, filters to those with `SHADER_F64`, and selects
/// the one with the largest `max_buffer_size` (proxy for VRAM). Prefers
/// discrete GPUs over integrated when memory is equal.
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
    adapters.sort_by(|a, b| {
        let mem_cmp = b.memory_bytes.cmp(&a.memory_bytes);
        if mem_cmp != std::cmp::Ordering::Equal {
            return mem_cmp;
        }
        // Prefer discrete over integrated when memory equal
        let discrete = |x: &AdapterInfo| x.device_type == wgpu::DeviceType::DiscreteGpu;
        discrete(b).cmp(&discrete(a))
    });
    Some(adapters[0].index.to_string())
}

/// Discover primary and secondary GPU adapters by memory/capability.
///
/// Returns (primary, secondary) identifiers for multi-GPU setups. Primary is
/// the adapter with most memory; secondary is the next-best different adapter.
/// Either may be `None` if not enough compatible adapters exist.
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
    adapters.sort_by(|a, b| {
        let mem_cmp = b.memory_bytes.cmp(&a.memory_bytes);
        if mem_cmp != std::cmp::Ordering::Equal {
            return mem_cmp;
        }
        let discrete = |x: &AdapterInfo| x.device_type == wgpu::DeviceType::DiscreteGpu;
        discrete(b).cmp(&discrete(a))
    });

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

/// Select an adapter from a single hint string (same rules as one token in
/// `HOTSPRING_GPU_ADAPTER`: `"auto"`, a numeric index, or a case-insensitive name substring).
///
/// Does not read the environment — use [`select_adapter`] for env-based selection.
///
/// # Errors
///
/// Returns [`crate::error::HotSpringError`] if the hint is empty or no adapter matches.
pub fn select_adapter_hint(hint: &str) -> Result<wgpu::Adapter, crate::error::HotSpringError> {
    let token = hint.trim();
    if token.is_empty() {
        return Err(crate::error::HotSpringError::DeviceCreation(
            "empty GPU adapter hint".into(),
        ));
    }

    let selector = token.to_lowercase();
    if selector == "auto" {
        let fresh: Vec<wgpu::Adapter> =
            pollster::block_on(create_instance().enumerate_adapters(wgpu::Backends::all()));
        auto_select(fresh)
    } else if let Ok(idx) = selector.parse::<usize>() {
        select_by_index_or_name(
            pollster::block_on(create_instance().enumerate_adapters(wgpu::Backends::all())),
            idx,
            &selector,
        )
    } else {
        select_by_name(
            pollster::block_on(create_instance().enumerate_adapters(wgpu::Backends::all())),
            &selector,
        )
    }
}

/// Select an adapter based on the `HOTSPRING_GPU_ADAPTER` / `BARRACUDA_GPU_ADAPTER`
/// environment variables. Falls back to auto-detection (discrete + `SHADER_F64` first).
///
/// The value may be a comma-separated priority list, e.g. `"3090,titan,auto"`.
/// Each token is tried in order; the first match wins. `"auto"` at any position
/// triggers the automatic discrete-GPU-first heuristic. If no token matches,
/// returns an error.
///
/// # Errors
///
/// Returns [`crate::error::HotSpringError`] if no compatible adapter is found.
pub fn select_adapter() -> Result<wgpu::Adapter, crate::error::HotSpringError> {
    let raw = std::env::var("HOTSPRING_GPU_ADAPTER")
        .or_else(|_| std::env::var("BARRACUDA_GPU_ADAPTER"))
        .unwrap_or_default();

    let instance = create_instance();
    let adapters: Vec<wgpu::Adapter> =
        pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    if adapters.is_empty() {
        return Err(crate::error::HotSpringError::NoAdapter);
    }

    // Support comma-separated priority lists: "3090,titan,auto"
    let tokens: Vec<&str> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    let tokens = if tokens.is_empty() {
        vec!["auto"]
    } else {
        tokens
    };

    for token in &tokens {
        if let Ok(a) = select_adapter_hint(token) {
            return Ok(a);
        }
    }

    Err(crate::error::HotSpringError::DeviceCreation(format!(
        "No adapter matched any of {:?}. Available: {:?}",
        tokens,
        adapters
            .iter()
            .map(|a: &wgpu::Adapter| a.get_info().name)
            .collect::<Vec<_>>(),
    )))
}

fn auto_select(
    adapters: Vec<wgpu::Adapter>,
) -> Result<wgpu::Adapter, crate::error::HotSpringError> {
    let mut chosen: Option<wgpu::Adapter> = None;
    let mut fallback: Option<wgpu::Adapter> = None;
    for a in adapters {
        let a: wgpu::Adapter = a;
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
            .find(|a: &wgpu::Adapter| a.get_info().name.to_ascii_lowercase().contains(selector))
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
        .find(|a: &wgpu::Adapter| a.get_info().name.to_ascii_lowercase().contains(selector))
        .ok_or_else(|| {
            crate::error::HotSpringError::DeviceCreation(format!(
                "No adapter matching '{selector}'"
            ))
        })
}

// ── Device construction helpers ───────────────────────────────────────────────

use barracuda::device::{DeviceCapabilities, TensorContext, WgpuDevice};
use log::{info, warn};
use std::sync::Arc;

/// Negotiate wgpu features, requesting all science-relevant capabilities.
///
/// Silently drops features the adapter does not advertise — no panics.
pub(super) fn negotiate_features(adapter_features: wgpu::Features) -> wgpu::Features {
    let mut f = wgpu::Features::empty();
    for feat in [
        wgpu::Features::SHADER_F64,
        wgpu::Features::SHADER_F16,
        wgpu::Features::SUBGROUP,
        wgpu::Features::TIMESTAMP_QUERY,
        wgpu::Features::PIPELINE_CACHE,
    ] {
        if adapter_features.contains(feat) {
            f |= feat;
        }
    }
    f
}

/// Heuristic NVK detection from adapter info (before device creation).
pub(super) fn adapter_is_nvk(info: &wgpu::AdapterInfo) -> bool {
    let d = info.driver.to_lowercase();
    let di = info.driver_info.to_lowercase();
    d.contains("nvk")
        || d.contains("nouveau")
        || di.contains("nvk")
        || di.contains("nouveau")
        || (d.contains("mesa") && info.name.contains("NV"))
}

/// Shared post-creation logic: probe f64 builtins, detect `full_df64_mode`.
pub(super) async fn open_from_adapter_inner(
    selected: wgpu::Adapter,
    device_label: &'static str,
) -> Result<super::GpuF64, crate::error::HotSpringError> {
    let adapter_info = selected.get_info();
    let adapter_features = selected.features();
    let is_nvk = adapter_is_nvk(&adapter_info);
    let required_features = negotiate_features(adapter_features);

    let adapter_limits = selected.limits();
    let required_limits = wgpu::Limits {
        max_storage_buffer_binding_size: adapter_limits
            .max_storage_buffer_binding_size
            .min(2 * 1024 * 1024 * 1024),
        max_buffer_size: adapter_limits.max_buffer_size.min(4 * 1024 * 1024 * 1024),
        max_storage_buffers_per_shader_stage: 12,
        ..wgpu::Limits::default()
    };

    let (device, queue) = selected
        .request_device(&wgpu::DeviceDescriptor {
            label: Some(device_label),
            required_features,
            required_limits,
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::default(),
        })
        .await
        .map_err(|e| crate::error::HotSpringError::DeviceCreation(e.to_string()))?;

    let adapter_name = adapter_info.name.clone();
    let advertised_f64 = required_features.contains(wgpu::Features::SHADER_F64);
    let has_timestamps = required_features.contains(wgpu::Features::TIMESTAMP_QUERY);
    let has_subgroups = required_features.contains(wgpu::Features::SUBGROUP);
    let has_f16 = required_features.contains(wgpu::Features::SHADER_F16);

    let wgpu_device = Arc::new(WgpuDevice::from_existing(device, queue, adapter_info));

    Ok(finalize_device(
        adapter_name,
        advertised_f64,
        has_timestamps,
        has_subgroups,
        has_f16,
        is_nvk,
        wgpu_device,
    )
    .await)
}

/// Final device configuration: f64 probe + capabilities detection.
pub(super) async fn finalize_device(
    adapter_name: String,
    advertised_f64: bool,
    has_timestamps: bool,
    has_subgroups: bool,
    has_f16: bool,
    is_nvk: bool,
    wgpu_device: Arc<WgpuDevice>,
) -> super::GpuF64 {
    let tensor_ctx = Arc::new(TensorContext::new(Arc::clone(&wgpu_device)));
    let capabilities = DeviceCapabilities::from_device(&wgpu_device);

    let (has_f64, full_df64_mode) = if advertised_f64 {
        let caps = barracuda::device::probe::probe_f64_builtins(&wgpu_device).await;
        if caps.can_compile_f64() {
            if is_nvk {
                info!(
                    "f64 probe: PASS on NVK ({}/{} builtins) — using barraCuda polyfill pipeline",
                    caps.native_count(),
                    9
                );
            } else {
                info!(
                    "f64 probe: PASS — native f64 shaders available ({}/{} builtins)",
                    caps.native_count(),
                    9
                );
            }
            (true, false)
        } else {
            warn!("f64 probe: FAIL — switching to full DF64 data plane");
            (false, true)
        }
    } else {
        warn!("SHADER_F64 not advertised — using full DF64 data plane");
        (false, true)
    };

    super::GpuF64 {
        adapter_name,
        has_f64,
        has_timestamps,
        has_subgroups,
        has_f16,
        full_df64_mode,
        wgpu_device,
        tensor_ctx,
        capabilities,
    }
}
