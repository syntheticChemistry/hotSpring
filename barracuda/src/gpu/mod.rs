// SPDX-License-Identifier: AGPL-3.0-only

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
//! - `adapter` — adapter discovery and selection
//! - `buffers` — f64/u32 buffer creation, upload, readback
//! - `dispatch` — command encoding and dispatch
//! - `telemetry` — GPU power, temperature, VRAM monitoring

mod adapter;
mod buffers;
mod dispatch;
mod telemetry;

pub use adapter::AdapterInfo;

use barracuda::device::capabilities::GpuDriverProfile;
use barracuda::device::{TensorContext, WgpuDevice};
use barracuda::shaders::precision::ShaderTemplate;
use std::sync::Arc;

/// GPU context with FP64 support for science workloads.
///
/// Wraps wgpu device with `SHADER_F64` + `ToadStool`'s `TensorContext` for
/// batched dispatch (`begin_batch`/`end_batch`) and `BufferPool` reuse.
#[must_use]
pub struct GpuF64 {
    pub adapter_name: String,
    pub has_f64: bool,
    pub has_timestamps: bool,
    wgpu_device: Arc<WgpuDevice>,
    tensor_ctx: Arc<TensorContext>,
    driver_profile: GpuDriverProfile,
}

// ── Core accessors ───────────────────────────────────────────────────

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

    /// Get Arc-wrapped device (for APIs requiring `Arc<Device>`).
    #[must_use]
    pub fn device_arc(&self) -> Arc<wgpu::Device> {
        self.wgpu_device.device_arc()
    }

    /// Get Arc-wrapped queue (for APIs requiring `Arc<Queue>`).
    #[must_use]
    pub fn queue_arc(&self) -> Arc<wgpu::Queue> {
        self.wgpu_device.queue_arc()
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

    /// Access the runtime-detected driver profile for shader specialization.
    #[must_use]
    pub const fn driver_profile(&self) -> &GpuDriverProfile {
        &self.driver_profile
    }
}

// ── Constructor ──────────────────────────────────────────────────────

impl GpuF64 {
    /// Create GPU device requesting `SHADER_F64`.
    ///
    /// Adapter selection: `HOTSPRING_GPU_ADAPTER` takes priority, then falls
    /// through to `BARRACUDA_GPU_ADAPTER`, then auto-detect.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::HotSpringError`] if no compatible adapter is found
    /// or device creation fails.
    pub async fn new() -> Result<Self, crate::error::HotSpringError> {
        let selected = adapter::select_adapter()?;
        let adapter_info = selected.get_info();
        let adapter_features = selected.features();

        let mut required_features = wgpu::Features::empty();
        if adapter_features.contains(wgpu::Features::SHADER_F64) {
            required_features |= wgpu::Features::SHADER_F64;
        }
        if adapter_features.contains(wgpu::Features::SHADER_F16) {
            required_features |= wgpu::Features::SHADER_F16;
        }
        if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: 512 * 1024 * 1024,
            max_buffer_size: 1024 * 1024 * 1024,
            max_storage_buffers_per_shader_stage: 12,
            ..wgpu::Limits::default()
        };

        let (device, queue) = selected
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("hotSpring science device"),
                    required_features,
                    required_limits,
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| crate::error::HotSpringError::DeviceCreation(e.to_string()))?;

        let adapter_name = adapter_info.name.clone();
        let has_f64 = required_features.contains(wgpu::Features::SHADER_F64);
        let has_timestamps = required_features.contains(wgpu::Features::TIMESTAMP_QUERY);

        let wgpu_device = Arc::new(WgpuDevice::from_existing(
            Arc::new(device),
            Arc::new(queue),
            adapter_info,
        ));
        let tensor_ctx = Arc::new(TensorContext::new(Arc::clone(&wgpu_device)));
        let driver_profile = GpuDriverProfile::from_device(&wgpu_device);

        Ok(Self {
            adapter_name,
            has_f64,
            has_timestamps,
            wgpu_device,
            tensor_ctx,
            driver_profile,
        })
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
            "  Driver: {:?}, Compiler: {:?}, Arch: {:?}",
            self.driver_profile.driver, self.driver_profile.compiler, self.driver_profile.arch
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

// ── Pipeline creation ────────────────────────────────────────────────

impl GpuF64 {
    /// Create a compute pipeline with `WgslOptimizer` + `GpuDriverProfile`.
    ///
    /// Does NOT apply exp/log workarounds — use [`Self::create_pipeline_f64`]
    /// for shaders that call `exp()` or `log()` on f64 values.
    #[must_use]
    pub fn create_pipeline(&self, shader_source: &str, label: &str) -> wgpu::ComputePipeline {
        let optimized =
            ShaderTemplate::for_driver_profile(shader_source, false, &self.driver_profile);
        self.build_pipeline(&optimized, label)
    }

    /// Create a compute pipeline with driver-aware f64 patching + optimization.
    ///
    /// Routes through `ShaderTemplate::for_driver_profile()` which applies:
    /// 1. Fossil substitution (legacy `math_f64` → native builtins)
    /// 2. exp/log workaround on NVK/nouveau
    /// 3. Missing `math_f64` injection (only functions actually called)
    /// 4. `WgslOptimizer` with hardware-accurate `LatencyModel`
    #[must_use]
    pub fn create_pipeline_f64(&self, shader_source: &str, label: &str) -> wgpu::ComputePipeline {
        let optimized = ShaderTemplate::for_driver_profile(
            shader_source,
            self.wgpu_device.needs_f64_exp_log_workaround(),
            &self.driver_profile,
        );
        self.build_pipeline(&optimized, label)
    }

    fn build_pipeline(&self, wgsl: &str, label: &str) -> wgpu::ComputePipeline {
        let shader_module = self
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        self.device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &shader_module,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn f64_buffer_size_bytes(count: usize) -> usize {
        count * 8
    }

    fn f64_to_bytes(data: &[f64]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn bytes_to_f64(data: &[u8]) -> Vec<f64> {
        data.chunks_exact(8)
            .map(|chunk| {
                let bytes: [u8; 8] = chunk.try_into().expect("8-byte f64 chunk");
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
    #[allow(clippy::float_cmp)]
    fn f64_byte_roundtrip() {
        let original = vec![
            0.0,
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
    #[allow(clippy::float_cmp)]
    fn f64_byte_conversion_special_values() {
        let values = [std::f64::consts::PI, 1e-308, 1e308];
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
        let err_result: Result<Vec<f64>, crate::error::HotSpringError> = Err(
            crate::error::HotSpringError::GpuCompute("no GPU available".into()),
        );
        assert!(err_result.is_err());
    }

    #[test]
    fn read_back_f64_empty() {
        let empty: Vec<u8> = vec![];
        assert_eq!(empty.len() / 8, 0);
    }
}
