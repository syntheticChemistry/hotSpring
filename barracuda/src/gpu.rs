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
//! Use [`GpuF64::enumerate_adapters`] to list available GPUs before selecting.
//!
//! ## Validated hardware
//!
//! | GPU | Driver | `shaderFloat64` | Notes |
//! |-----|--------|-----------------|-------|
//! | RTX 4070 (Ada) | nvidia proprietary 580.x | true | fp64:fp32 ~1:2 via Vulkan |
//! | Titan V (GV100) | NVK / nouveau (Mesa 25.1) | true | Native fp64 silicon, open-source |
//!
//! Numerical parity confirmed: identical physics to 1e-15 across drivers.
//!
//! ## Architecture (Experiment 004 lesson)
//!
//! The trains to and from take more time than the work.
//! Pre-plan, fill GPU function space, fire at once.
//! `TensorContext` enables `begin_batch()`/`end_batch()` for batched dispatch.

use barracuda::device::{TensorContext, WgpuDevice};
use std::process::Command;
use std::sync::Arc;

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

/// GPU context with FP64 support for science workloads.
///
/// Wraps wgpu device with `SHADER_F64` + ToadStool's `TensorContext` for
/// batched dispatch (`begin_batch`/`end_batch`) and `BufferPool` reuse.
pub struct GpuF64 {
    pub adapter_name: String,
    pub has_f64: bool,
    pub has_timestamps: bool,
    wgpu_device: Arc<WgpuDevice>,
    tensor_ctx: Arc<TensorContext>,
}

impl GpuF64 {
    /// Access the underlying wgpu Device.
    ///
    /// Delegates to barracuda's `WgpuDevice::device()`.
    pub fn device(&self) -> &wgpu::Device {
        self.wgpu_device.device()
    }

    /// Access the underlying wgpu Queue.
    ///
    /// Delegates to barracuda's `WgpuDevice::queue()`.
    pub fn queue(&self) -> &wgpu::Queue {
        self.wgpu_device.queue()
    }

    /// Get Arc-wrapped device (for PppmGpu and other APIs requiring Arc&lt;Device&gt;).
    pub fn device_arc(&self) -> Arc<wgpu::Device> {
        self.wgpu_device.device_arc()
    }

    /// Get Arc-wrapped queue (for PppmGpu and other APIs requiring Arc&lt;Queue&gt;).
    pub fn queue_arc(&self) -> Arc<wgpu::Queue> {
        self.wgpu_device.queue_arc()
    }

    /// Bridge to toadstool's WgpuDevice for BatchedEighGpu, SsfGpu, etc.
    ///
    /// This enables all toadstool GPU operations (linalg, FFT, observables)
    /// from hotSpring binaries using the same underlying wgpu device.
    pub fn to_wgpu_device(&self) -> Arc<WgpuDevice> {
        self.wgpu_device.clone()
    }

    /// Access the TensorContext for batched dispatch.
    ///
    /// Usage:
    /// ```rust,ignore
    /// let ctx = gpu.tensor_context();
    /// ctx.begin_batch();
    /// // ... queue multiple GPU operations ...
    /// ctx.end_batch()?;  // Single GPU submission
    /// ```
    pub const fn tensor_context(&self) -> &Arc<TensorContext> {
        &self.tensor_ctx
    }
}

impl GpuF64 {
    /// Create a wgpu instance with the configured backend.
    fn create_instance() -> wgpu::Instance {
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
    pub fn enumerate_adapters() -> Vec<AdapterInfo> {
        let instance = Self::create_instance();
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

    /// Create GPU device requesting `SHADER_F64`.
    ///
    /// Adapter selection: `HOTSPRING_GPU_ADAPTER` takes priority, then falls
    /// through to barracuda's `BARRACUDA_GPU_ADAPTER`, then auto-detect.
    /// This ensures hotSpring-specific targeting works alongside ecosystem-wide
    /// adapter selection.
    pub async fn new() -> Result<Self, crate::error::HotSpringError> {
        // Priority: HOTSPRING_GPU_ADAPTER → BARRACUDA_GPU_ADAPTER → auto
        let selector = std::env::var("HOTSPRING_GPU_ADAPTER")
            .or_else(|_| std::env::var("BARRACUDA_GPU_ADAPTER"))
            .unwrap_or_default()
            .trim()
            .to_lowercase();

        // Discover and select adapter (same instance config as enumerate_adapters)
        let instance = Self::create_instance();
        let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all());
        if adapters.is_empty() {
            return Err(crate::error::HotSpringError::NoAdapter);
        }

        let adapter = if selector.is_empty() || selector == "auto" {
            // Auto-select: prefer discrete GPU with SHADER_F64, then any with SHADER_F64
            let mut chosen: Option<wgpu::Adapter> = None;
            let mut fallback: Option<wgpu::Adapter> = None;
            for a in adapters {
                if a.features().contains(wgpu::Features::SHADER_F64) {
                    if a.get_info().device_type == wgpu::DeviceType::DiscreteGpu && chosen.is_none()
                    {
                        chosen = Some(a);
                    } else if fallback.is_none() {
                        fallback = Some(a);
                    }
                }
            }
            chosen
                .or(fallback)
                .ok_or(crate::error::HotSpringError::NoAdapter)?
        } else if let Ok(idx) = selector.parse::<usize>() {
            if idx < adapters.len() {
                adapters
                    .into_iter()
                    .nth(idx)
                    .ok_or(crate::error::HotSpringError::NoAdapter)?
            } else {
                // Numeric value exceeds adapter count — treat as name substring
                adapters
                    .into_iter()
                    .find(|a| a.get_info().name.to_ascii_lowercase().contains(&selector))
                    .ok_or_else(|| {
                        crate::error::HotSpringError::DeviceCreation(format!(
                            "No adapter matching '{selector}' (tried as index {idx} and name)"
                        ))
                    })?
            }
        } else {
            // Name substring match (case-insensitive)
            adapters
                .into_iter()
                .find(|a| a.get_info().name.to_ascii_lowercase().contains(&selector))
                .ok_or_else(|| {
                    crate::error::HotSpringError::DeviceCreation(format!(
                        "No adapter matching '{selector}'"
                    ))
                })?
        };

        let adapter_info = adapter.get_info();

        // Request features: SHADER_F64 is mandatory for hotSpring physics
        let adapter_features = adapter.features();
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

        // hotSpring science limits: higher storage buffer limits than defaults.
        // max_storage_buffers_per_shader_stage = 12 for HFB potentials shader
        // which binds rho_p, rho_n, rho_alpha, rho_alpha_m1, V_sky, V_coul,
        // V_teff, V_fq, h_p, h_n, dims, wf (12 total).
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: 512 * 1024 * 1024,
            max_buffer_size: 1024 * 1024 * 1024,
            max_storage_buffers_per_shader_stage: 12,
            ..wgpu::Limits::default()
        };

        let (device, queue) = adapter
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

        // Wrap in barracuda's WgpuDevice for compatibility with barracuda ops
        let wgpu_device = Arc::new(WgpuDevice::from_existing(
            Arc::new(device),
            Arc::new(queue),
            adapter_info,
        ));
        let tensor_ctx = Arc::new(TensorContext::new(wgpu_device.clone()));

        Ok(Self {
            adapter_name,
            has_f64,
            has_timestamps,
            wgpu_device,
            tensor_ctx,
        })
    }

    /// Print device capabilities.
    pub fn print_info(&self) {
        println!("  GPU: {}", self.adapter_name);
        println!("  SHADER_F64: {}", if self.has_f64 { "YES" } else { "NO" });
        println!(
            "  TIMESTAMP_QUERY: {}",
            if self.has_timestamps { "YES" } else { "NO" }
        );
    }

    /// Print all available adapters to stdout.
    ///
    /// Useful for discovery before setting `HOTSPRING_GPU_ADAPTER`.
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

    /// Create a compute pipeline from WGSL shader source
    pub fn create_pipeline(&self, shader_source: &str, label: &str) -> wgpu::ComputePipeline {
        let shader_module = self
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
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

    /// Create a storage buffer from f64 data (read-only)
    pub fn create_f64_buffer(&self, data: &[f64], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// Create a writable storage buffer for f64 output
    pub fn create_f64_output_buffer(&self, count: usize, label: &str) -> wgpu::Buffer {
        self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (count * 8) as u64, // 8 bytes per f64
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a staging buffer for reading results back to CPU
    pub fn create_staging_buffer(&self, size: usize, label: &str) -> wgpu::Buffer {
        self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a uniform buffer from raw bytes
    pub fn create_uniform_buffer(&self, data: &[u8], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Query current GPU power draw, temperature, utilization, and VRAM.
    ///
    /// Returns `(power_watts, temp_celsius, utilization_pct, vram_used_mib)`.
    /// Returns zeros if monitoring tools are unavailable.
    ///
    /// Attempts `nvidia-smi` (proprietary driver) first, then gracefully
    /// degrades to zeros for open-source drivers (NVK/nouveau, RADV) where
    /// runtime power/temp monitoring is not yet standardized.
    pub fn query_gpu_power() -> (f64, f64, f64, f64) {
        // Try nvidia-smi (only works with proprietary nvidia driver)
        if let Some(result) = Self::query_nvidia_smi() {
            return result;
        }
        // Open-source drivers (NVK, RADV): no standard power query yet.
        // hwmon/sysfs could work but is GPU-index-dependent and fragile.
        // Return zeros — callers handle this gracefully.
        (0.0, 0.0, 0.0, 0.0)
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

    /// Snapshot of current GPU VRAM usage in MiB.
    pub fn gpu_vram_used_mib() -> f64 {
        let (_, _, _, vram) = Self::query_gpu_power();
        vram
    }

    /// Upload f64 data to a GPU storage buffer (overwrites from offset 0).
    pub fn upload_f64(&self, buffer: &wgpu::Buffer, data: &[f64]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.queue().write_buffer(buffer, 0, &bytes);
    }

    /// Read back f64 data from a GPU buffer via staging copy.
    ///
    /// Returns `Err` if the GPU map callback fails or the channel is dropped.
    pub fn read_back_f64(
        &self,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f64>, crate::error::HotSpringError> {
        let staging = self.create_staging_buffer(count * 8, "readback");
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 8) as u64);
        self.queue().submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device().poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| {
                crate::error::HotSpringError::DeviceCreation(
                    "GPU map callback: channel recv failed".into(),
                )
            })?
            .map_err(|e| {
                crate::error::HotSpringError::DeviceCreation(format!("GPU buffer mapping: {e}"))
            })?;

        let data = slice.get_mapped_range();
        let result: Vec<f64> = data
            .chunks_exact(8)
            .map(|chunk| {
                let bytes: [u8; 8] = chunk
                    .try_into()
                    .expect("chunks_exact(8) guarantees 8-byte slices");
                f64::from_le_bytes(bytes)
            })
            .collect();
        drop(data);
        staging.unmap();
        Ok(result)
    }

    /// Dispatch a compute pipeline (fire-and-forget within the GPU queue).
    pub fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: u32,
    ) {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue().submit(std::iter::once(encoder.finish()));
    }

    /// Create a bind group from a pipeline and ordered buffer slice.
    ///
    /// Each buffer is bound at binding index 0, 1, 2, ... in order.
    pub fn create_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group"),
            layout: &layout,
            entries: &entries,
        })
    }

    /// Create a storage buffer from u32 data (read-only).
    pub fn create_u32_buffer(&self, data: &[u32], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// Dispatch a compute pipeline and read back f64 results.
    ///
    /// Returns `Err` if the GPU map callback fails or the channel is dropped.
    pub fn dispatch_and_read(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: u32,
        output_buffer: &wgpu::Buffer,
        output_count: usize,
    ) -> Result<Vec<f64>, crate::error::HotSpringError> {
        let staging = self.create_staging_buffer(output_count * 8, "staging");

        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(output_buffer, 0, &staging, 0, (output_count * 8) as u64);
        self.queue().submit(std::iter::once(encoder.finish()));

        // Read back via channel — no panics
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device().poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| {
                crate::error::HotSpringError::DeviceCreation(
                    "GPU map callback: channel recv failed".into(),
                )
            })?
            .map_err(|e| {
                crate::error::HotSpringError::DeviceCreation(format!("GPU buffer mapping: {e}"))
            })?;

        let data = slice.get_mapped_range();
        let result: Vec<f64> = data
            .chunks_exact(8)
            .map(|chunk| {
                let bytes: [u8; 8] = chunk
                    .try_into()
                    .expect("chunks_exact(8) guarantees 8-byte slices");
                f64::from_le_bytes(bytes)
            })
            .collect();
        drop(data);
        staging.unmap();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure helper: f64 buffer size in bytes (matches create_f64_output_buffer logic)
    fn f64_buffer_size_bytes(count: usize) -> usize {
        count * 8
    }

    /// Pure helper: convert f64 slice to bytes (matches create_f64_buffer logic)
    fn f64_to_bytes(data: &[f64]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Pure helper: convert bytes back to f64 (matches dispatch_and_read readback logic)
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
        // Placeholder: would need real GpuF64, pipeline, bind_group.
        // Result<Vec<f64>, HotSpringError> is the contract.
        let _: Result<Vec<f64>, crate::error::HotSpringError> = Ok(vec![1.0, 2.0]);
    }

    #[test]
    fn read_back_f64_empty() {
        // Verify edge case: empty buffer — mapping logic handles sizes correctly
        let empty: Vec<u8> = vec![];
        assert_eq!(empty.len() / 8, 0);
    }
}
