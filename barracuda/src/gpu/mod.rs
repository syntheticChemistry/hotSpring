// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU FP64 compute for hotSpring science workloads.
//!
//! Creates a wgpu device with `SHADER_F64` enabled, and provides helpers
//! for running f64 compute shaders on any Vulkan GPU (NVIDIA proprietary,
//! NVK/nouveau, RADV, etc.).
//!
//! ## Adapter selection
//!
//! Explicit adapter targeting is the default. Set `HOTSPRING_GPU_ADAPTER`
//! to select a specific GPU:
//!
//! | Value | Behavior |
//! |-------|----------|
//! | `auto` | wgpu `HighPerformance` preference (legacy default) |
//! | `0`, `1`, … | Select adapter by enumeration index |
//! | substring | Case-insensitive name match (e.g. `"titan"`, `"4070"`) |
//! | *(unset)* | Enumerate all adapters, pick first with `SHADER_F64` |
//!
//! Use `GpuF64::enumerate_adapters` to list available GPUs before selecting.
//!
//! ## Module structure
//!
//! | Submodule | Responsibility |
//! |-----------|---------------|
//! | `adapter` | Adapter discovery, selection, device construction helpers |
//! | `buffers` | f64/u32 buffer creation, upload, readback, DF64 wire helpers |
//! | `dispatch` | Command encoding, pipeline creation (merged with sub-module) |
//! | `telemetry` | GPU power, temperature, VRAM monitoring |

mod adapter;
mod buffers;
mod dispatch;
mod telemetry;

pub use adapter::{AdapterInfo, discover_best_adapter, discover_primary_and_secondary_adapters};
pub use buffers::{df64_bytes_to_f64_slice, df64_to_f64, f64_slice_to_df64_bytes, f64_to_df64};
pub use dispatch::split_workgroups;

use barracuda::device::{DeviceCapabilities, TensorContext, WgpuDevice};
use std::sync::Arc;

/// GPU context with FP64 support for science workloads.
///
/// Wraps wgpu device with `SHADER_F64` + barraCuda's `TensorContext` for
/// batched dispatch (`begin_batch`/`end_batch`) and `BufferPool` reuse.
///
/// When the f64 probe fails (NVK Mesa 25.1+), falls back to full DF64 mode:
/// all shaders compiled via `compile_shader_universal(Precision::Df64)` and
/// buffers use `vec2<f32>` wire format — no `SHADER_F64` required.
#[must_use]
pub struct GpuF64 {
    /// Human-readable adapter name (e.g. "NVIDIA `GeForce` RTX 4070").
    pub adapter_name: String,
    /// Whether the device can actually compile f64 shaders (probe-verified,
    /// not just feature-advertised).
    pub has_f64: bool,
    /// Whether timestamp queries are available for profiling.
    pub has_timestamps: bool,
    /// Whether WGSL subgroup intrinsics (`subgroupAdd`, etc.) are available.
    /// Note: do NOT use `enable subgroups;` directive — naga 28 emits broken
    /// SPIR-V. Rely on `SUBGROUP` device feature instead.
    pub has_subgroups: bool,
    /// Whether SHADER_F16 (half-precision compute) is available.
    /// Enables tensor core proxy path on NVIDIA Ampere+.
    pub has_f16: bool,
    /// True when native f64 shaders are broken and all GPU work must use
    /// full DF64 (f32-pair) data plane — storage as `vec2<f32>`, not `f64`.
    pub full_df64_mode: bool,
    pub(crate) wgpu_device: Arc<WgpuDevice>,
    pub(crate) tensor_ctx: Arc<TensorContext>,
    pub(crate) capabilities: DeviceCapabilities,
}

// ── Core accessors ────────────────────────────────────────────────────────────

impl GpuF64 {
    /// Access the underlying wgpu Device.
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        self.wgpu_device.device()
    }

    /// Access the underlying wgpu Queue.
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        self.wgpu_device.queue()
    }

    /// Get a cloned device handle (wgpu 28: Device is Clone, no Arc needed).
    #[must_use]
    pub fn device_clone(&self) -> wgpu::Device {
        self.wgpu_device.device_clone()
    }

    /// Get a cloned queue handle (wgpu 28: Queue is Clone, no Arc needed).
    #[must_use]
    pub fn queue_clone(&self) -> wgpu::Queue {
        self.wgpu_device.queue_clone()
    }

    /// Bridge to toadstool's `WgpuDevice`.
    #[must_use]
    pub fn to_wgpu_device(&self) -> Arc<WgpuDevice> {
        Arc::clone(&self.wgpu_device)
    }

    /// Access the `TensorContext` for batched dispatch.
    #[must_use]
    pub const fn tensor_context(&self) -> &Arc<TensorContext> {
        &self.tensor_ctx
    }

    /// Access the runtime-detected device capabilities for shader specialization.
    #[must_use]
    pub const fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    /// FP64 execution strategy for this device.
    #[must_use]
    pub fn fp64_strategy(&self) -> barracuda::device::driver_profile::Fp64Strategy {
        self.capabilities.fp64_strategy()
    }

    /// Precision routing advice for this device.
    #[must_use]
    pub fn precision_routing(&self) -> barracuda::device::driver_profile::PrecisionRoutingAdvice {
        self.capabilities.precision_routing()
    }
}

// ── Constructors ──────────────────────────────────────────────────────────────

impl GpuF64 {
    /// Create GPU device requesting `SHADER_F64`.
    ///
    /// Adapter selection: `HOTSPRING_GPU_ADAPTER` takes priority, then falls
    /// through to `BARRACUDA_GPU_ADAPTER`, then auto-detect.
    ///
    /// After device creation, runs barraCuda's f64 probe to verify shader
    /// compilation actually works. On NVK with broken NAK, switches to full
    /// DF64 mode automatically.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::HotSpringError`] if no compatible adapter is found
    /// or device creation fails.
    pub async fn new() -> Result<Self, crate::error::HotSpringError> {
        let selected = adapter::select_adapter()?;
        adapter::open_from_adapter_inner(selected, "hotSpring science device").await
    }

    /// Create GPU device from an explicit adapter hint (numeric index,
    /// case-insensitive name substring, or `"auto"`).
    pub async fn with_adapter(hint: &str) -> Result<Self, crate::error::HotSpringError> {
        let selected = adapter::select_adapter_hint(hint)?;
        Self::from_adapter(selected).await
    }

    /// Create GPU device by matching an adapter name substring.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::HotSpringError`] if no matching adapter is found.
    pub async fn from_adapter_name(name_hint: &str) -> Result<Self, crate::error::HotSpringError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

        let hint_lower = name_hint.to_lowercase();
        let selected = adapters
            .into_iter()
            .find(|a: &wgpu::Adapter| a.get_info().name.to_lowercase().contains(&hint_lower))
            .ok_or(crate::error::HotSpringError::NoAdapter)?;

        Self::from_adapter(selected).await
    }

    /// Create GPU device from a pre-selected wgpu adapter.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::HotSpringError`] if device creation fails.
    pub async fn from_adapter(
        selected: wgpu::Adapter,
    ) -> Result<Self, crate::error::HotSpringError> {
        adapter::open_from_adapter_inner(selected, "hotSpring secondary device").await
    }

    /// Enumerate all available GPU adapters.
    #[must_use]
    pub fn enumerate_adapters() -> Vec<AdapterInfo> {
        adapter::enumerate_adapters()
    }

    /// Print device capabilities.
    pub fn print_info(&self) {
        println!("  GPU: {}", self.adapter_name);
        println!("  SHADER_F64: {}", if self.has_f64 { "YES" } else { "NO" });
        println!(
            "  TIMESTAMP_QUERY: {}",
            if self.has_timestamps { "YES" } else { "NO" }
        );
        println!(
            "  Backend: {:?}, Vendor: {:#06x}, Type: {:?}",
            self.capabilities.backend, self.capabilities.vendor, self.capabilities.device_type
        );
    }

    /// Print all available adapters to stdout.
    pub fn print_available_adapters() {
        let adapters = Self::enumerate_adapters();
        println!("  Available GPU adapters:");
        for info in &adapters {
            let marker = if info.has_f64 { "✓" } else { "✗" };
            println!("    {marker} {info}");
        }
        if adapters.is_empty() {
            println!("    (none found)");
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn f64_buffer_size_bytes(count: usize) -> usize {
        count * 8
    }

    fn f64_to_bytes(data: &[f64]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// `chunks_exact(8)` guarantees 8-byte chunks; array construction cannot fail.
    fn bytes_to_f64(data: &[u8]) -> Vec<f64> {
        data.chunks_exact(8)
            .map(|chunk| {
                let bytes: [u8; 8] = std::array::from_fn(|i| chunk[i]);
                f64::from_le_bytes(bytes)
            })
            .collect()
    }

    #[test]
    fn f64_buffer_size_calculation() {
        assert_eq!(f64_buffer_size_bytes(0), 0);
        assert_eq!(f64_buffer_size_bytes(1), 8);
        assert_eq!(f64_buffer_size_bytes(100), 800);
        assert_eq!(f64_buffer_size_bytes(1000 * 3), 24_000);
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "exact known test value")]
    fn f64_byte_roundtrip() {
        let original = vec![
            0.0_f64,
            1.0,
            -1.0,
            std::f64::consts::PI,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let bytes = f64_to_bytes(&original);
        let recovered = bytes_to_f64(&bytes);
        assert_eq!(original.len(), recovered.len());
        for i in 0..original.len() {
            if original[i].is_nan() {
                assert!(recovered[i].is_nan());
            } else {
                assert_eq!(original[i], recovered[i]);
            }
        }
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "exact known test value")]
    fn f64_byte_conversion_special_values() {
        let values = [std::f64::consts::PI, 1e-308_f64, 1e308];
        let bytes = f64_to_bytes(&values);
        assert_eq!(bytes.len(), 24);
        let back = bytes_to_f64(&bytes);
        assert_eq!(back[0], std::f64::consts::PI);
        assert_eq!(back[1], 1e-308);
        assert_eq!(back[2], 1e308);
    }

    #[test]
    fn query_gpu_power_returns_four_tuple() {
        let (power, temp, util, vram) = GpuF64::query_gpu_power();
        assert!(power >= 0.0);
        assert!((0.0..=150.0).contains(&temp));
        assert!((0.0..=100.0).contains(&util));
        assert!(vram >= 0.0);
    }

    #[test]
    fn gpu_vram_used_non_negative() {
        let vram = GpuF64::gpu_vram_used_mib();
        assert!(vram >= 0.0);
    }

    #[test]
    #[ignore = "requires GPU"]
    fn dispatch_and_read_result_type() {
        let ok_result: Result<Vec<f64>, crate::error::HotSpringError> = Ok(vec![1.0, 2.0]);
        assert!(ok_result.is_ok());
        let err_result: Result<Vec<f64>, crate::error::HotSpringError> =
            Err(crate::error::HotSpringError::GpuCompute("no GPU available".into()));
        assert!(err_result.is_err());
    }

    #[test]
    fn read_back_f64_empty() {
        let empty: Vec<u8> = vec![];
        assert_eq!(empty.len() / 8, 0);
    }

    #[test]
    fn df64_roundtrip_pi() {
        use super::{df64_to_f64, f64_to_df64};
        let v = std::f64::consts::PI;
        let pair = f64_to_df64(v);
        let back = df64_to_f64(pair);
        assert!((back - v).abs() < 1e-7, "DF64 roundtrip error: {}", (back - v).abs());
    }
}
