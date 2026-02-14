//! GPU FP64 compute for hotSpring science workloads
//!
//! Creates a wgpu device with SHADER_F64 enabled, and provides helpers
//! for running f64 compute shaders on the RTX 4070 (or any Vulkan GPU).
//!
//! Validated: RTX 4070 provides TRUE IEEE 754 f64 (0 ULP vs CPU).
//! Performance: ~2x f32 for bandwidth-limited ops (element-wise, reductions).

use barracuda::device::WgpuDevice;
use std::process::Command;
use std::sync::Arc;

/// GPU context with FP64 support for science workloads
pub struct GpuF64 {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_name: String,
    pub has_f64: bool,
    pub has_timestamps: bool,
}

impl GpuF64 {
    /// Bridge to toadstool's WgpuDevice for BatchedEighGpu, SsfGpu, etc.
    ///
    /// This enables all toadstool GPU operations (linalg, FFT, observables)
    /// from hotSpring binaries using the same underlying wgpu device.
    pub fn to_wgpu_device(&self) -> Arc<WgpuDevice> {
        Arc::new(WgpuDevice::from_existing_simple(
            self.device.clone(),
            self.queue.clone(),
        ))
    }
}

impl GpuF64 {
    /// Create GPU device requesting SHADER_F64
    ///
    /// Falls back gracefully if f64 not available (reports has_f64 = false).
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "No GPU adapter found".to_string())?;

        let info = adapter.get_info();
        let features = adapter.features();
        let has_f64 = features.contains(wgpu::Features::SHADER_F64);
        let has_timestamps = features.contains(wgpu::Features::TIMESTAMP_QUERY);

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
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {e}"))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_name: info.name.clone(),
            has_f64,
            has_timestamps,
        })
    }

    /// Print device capabilities
    pub fn print_info(&self) {
        println!("  GPU: {}", self.adapter_name);
        println!("  SHADER_F64: {}", if self.has_f64 { "YES" } else { "NO" });
        println!("  TIMESTAMP_QUERY: {}", if self.has_timestamps { "YES" } else { "NO" });
    }

    /// Create a compute pipeline from WGSL shader source
    pub fn create_pipeline(&self, shader_source: &str, label: &str) -> wgpu::ComputePipeline {
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
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
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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

    /// Dispatch a compute pipeline and read back f64 results
    pub fn dispatch_and_read(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: u32,
        output_buffer: &wgpu::Buffer,
        output_count: usize,
    ) -> Vec<f64> {
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

        // Read back
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f64> = data
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        drop(data);
        staging.unmap();

        result
    }
}
