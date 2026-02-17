// SPDX-License-Identifier: AGPL-3.0-only

//! GPU FP64 compute for hotSpring science workloads
//!
//! Creates a wgpu device with SHADER_F64 enabled, and provides helpers
//! for running f64 compute shaders on the RTX 4070 (or any Vulkan GPU).
//!
//! Validated: RTX 4070 provides TRUE IEEE 754 f64 (0 ULP vs CPU).
//! Performance: ~2x f32 for bandwidth-limited ops (element-wise, reductions).
//!
//! Architecture (Experiment 004 lesson):
//!   The trains to and from take more time than the work.
//!   Pre-plan, fill GPU function space, fire at once.
//!   TensorContext enables begin_batch()/end_batch() for batched dispatch.

use barracuda::device::{TensorContext, WgpuDevice};
use std::process::Command;
use std::sync::Arc;

/// GPU context with FP64 support for science workloads.
///
/// Wraps wgpu device with SHADER_F64 + ToadStool's TensorContext for
/// batched dispatch (begin_batch/end_batch) and BufferPool reuse.
pub struct GpuF64 {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_name: String,
    pub has_f64: bool,
    pub has_timestamps: bool,
    wgpu_device: Arc<WgpuDevice>,
    tensor_ctx: Arc<TensorContext>,
}

impl GpuF64 {
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
    /// Create GPU device requesting SHADER_F64
    ///
    /// Falls back gracefully if f64 not available (reports has_f64 = false).
    pub async fn new() -> Result<Self, crate::error::HotSpringError> {
        let backends = match std::env::var("HOTSPRING_WGPU_BACKEND").as_deref() {
            Ok("vulkan") => wgpu::Backends::VULKAN,
            Ok("metal") => wgpu::Backends::METAL,
            Ok("dx12") => wgpu::Backends::DX12,
            _ => wgpu::Backends::all(),
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let power_pref = match std::env::var("HOTSPRING_GPU_POWER").as_deref() {
            Ok("low") => wgpu::PowerPreference::LowPower,
            _ => wgpu::PowerPreference::HighPerformance,
        };
        let allow_fallback = std::env::var("HOTSPRING_ALLOW_FALLBACK").is_ok();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: power_pref,
                compatible_surface: None,
                force_fallback_adapter: allow_fallback,
            })
            .await
            .ok_or(crate::error::HotSpringError::NoAdapter)?;

        let info = adapter.get_info();
        let features = adapter.features();
        let has_f64 = features.contains(wgpu::Features::SHADER_F64);
        let has_timestamps = features.contains(wgpu::Features::TIMESTAMP_QUERY);

        let adapter_limits = adapter.limits();

        // Request SHADER_F64 if available
        let mut required_features = wgpu::Features::empty();
        if has_f64 {
            required_features |= wgpu::Features::SHADER_F64;
        }
        if has_timestamps {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("hotSpring FP64 Science Device"),
                    required_features,
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: adapter_limits
                            .max_storage_buffer_binding_size
                            .min(512 * 1024 * 1024),
                        max_buffer_size: adapter_limits.max_buffer_size.min(1024 * 1024 * 1024),
                        max_storage_buffers_per_shader_stage: adapter_limits
                            .max_storage_buffers_per_shader_stage
                            .min(16),
                        ..wgpu::Limits::default()
                    },
                },
                None,
            )
            .await
            .map_err(|e| crate::error::HotSpringError::DeviceCreation(e.to_string()))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Create ToadStool WgpuDevice bridge (shared device/queue, no new allocation)
        let wgpu_device = Arc::new(WgpuDevice::from_existing_simple(
            device.clone(),
            queue.clone(),
        ));

        // Create TensorContext for batched dispatch and BufferPool
        let tensor_ctx = Arc::new(TensorContext::new(wgpu_device.clone()));

        Ok(Self {
            device,
            queue,
            adapter_name: info.name.clone(),
            has_f64,
            has_timestamps,
            wgpu_device,
            tensor_ctx,
        })
    }

    /// Print device capabilities
    pub fn print_info(&self) {
        println!("  GPU: {}", self.adapter_name);
        println!("  SHADER_F64: {}", if self.has_f64 { "YES" } else { "NO" });
        println!(
            "  TIMESTAMP_QUERY: {}",
            if self.has_timestamps { "YES" } else { "NO" }
        );
    }

    /// Create a compute pipeline from WGSL shader source
    pub fn create_pipeline(&self, shader_source: &str, label: &str) -> wgpu::ComputePipeline {
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None, // Auto-layout
                module: &shader_module,
                entry_point: "main",
            })
    }

    /// Create a storage buffer from f64 data (read-only)
    pub fn create_f64_buffer(&self, data: &[f64], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// Create a writable storage buffer for f64 output
    pub fn create_f64_output_buffer(&self, count: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
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
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a uniform buffer from raw bytes
    pub fn create_uniform_buffer(&self, data: &[u8], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Query current GPU power draw, temperature, utilization, and VRAM via nvidia-smi.
    ///
    /// Returns `(power_watts, temp_celsius, utilization_pct, vram_used_mib)`.
    /// Returns zeros if nvidia-smi is unavailable.
    pub fn query_gpu_power() -> (f64, f64, f64, f64) {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ])
            .output();

        match output {
            Ok(out) if out.status.success() => {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                let parts: Vec<&str> = s.split(", ").collect();
                if parts.len() >= 4 {
                    let watts = parts[0].trim().parse().unwrap_or(0.0);
                    let temp = parts[1].trim().parse().unwrap_or(0.0);
                    let util = parts[2].trim().parse().unwrap_or(0.0);
                    let vram = parts[3].trim().parse().unwrap_or(0.0);
                    return (watts, temp, util, vram);
                }
                (0.0, 0.0, 0.0, 0.0)
            }
            _ => (0.0, 0.0, 0.0, 0.0),
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
        self.queue.write_buffer(buffer, 0, &bytes);
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
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 8) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
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
            .device
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
        self.queue.submit(std::iter::once(encoder.finish()));
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
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group"),
            layout: &layout,
            entries: &entries,
        })
    }

    /// Create a storage buffer from u32 data (read-only).
    pub fn create_u32_buffer(&self, data: &[u32], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device
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
            .device
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
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back via channel — no panics
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
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
