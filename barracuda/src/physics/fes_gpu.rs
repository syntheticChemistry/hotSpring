// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU FES Reconstruction — Metadynamics Gaussian Summation
//!
//! Reconstructs Free Energy Surfaces from HILLS data on GPU using native f64
//! WGSL shaders. Each thread computes one grid point by summing all Gaussians:
//!
//!   bias(x,y) = Σ_g h_g · exp(-(x-cx_g)²/(2σx²)) · exp(-(y-cy_g)²/(2σy²))
//!   FES(x,y) = -bias(x,y) + min_shift
//!
//! Mirrors the CPU implementation in `cazyme-fel` for Tier 2→3 parity.

use crate::gpu::GpuF64;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const FES_SHADER: &str = include_str!("shaders/fes_gaussian_sum_f64.wgsl");

/// Uniform params matching the WGSL `FesParams` struct.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FesParams {
    n_gaussians: u32,
    nbins_x: u32,
    nbins_y: u32,
    _pad: u32,
    grid_min_x_lo: u32,
    grid_min_x_hi: u32,
    grid_max_x_lo: u32,
    grid_max_x_hi: u32,
    grid_min_y_lo: u32,
    grid_min_y_hi: u32,
    grid_max_y_lo: u32,
    grid_max_y_hi: u32,
}

impl FesParams {
    fn new(
        n_gaussians: u32,
        nbins_x: u32,
        nbins_y: u32,
        grid_min_x: f64,
        grid_max_x: f64,
        grid_min_y: f64,
        grid_max_y: f64,
    ) -> Self {
        Self {
            n_gaussians,
            nbins_x,
            nbins_y,
            _pad: 0,
            grid_min_x_lo: grid_min_x.to_bits() as u32,
            grid_min_x_hi: (grid_min_x.to_bits() >> 32) as u32,
            grid_max_x_lo: grid_max_x.to_bits() as u32,
            grid_max_x_hi: (grid_max_x.to_bits() >> 32) as u32,
            grid_min_y_lo: grid_min_y.to_bits() as u32,
            grid_min_y_hi: (grid_min_y.to_bits() >> 32) as u32,
            grid_max_y_lo: grid_max_y.to_bits() as u32,
            grid_max_y_hi: (grid_max_y.to_bits() >> 32) as u32,
        }
    }
}

/// GPU FES reconstruction result.
pub struct FesGpuResult {
    /// Flat FES values (nbins_x × nbins_y), min-shifted to zero.
    pub fes: Vec<f64>,
    /// Grid bounds used.
    pub grid_min_x: f64,
    pub grid_max_x: f64,
    pub grid_min_y: f64,
    pub grid_max_y: f64,
    pub nbins_x: usize,
    pub nbins_y: usize,
    /// GPU wall time in seconds.
    pub gpu_secs: f64,
}

/// Cached GPU FES pipeline.
pub struct FesGaussianSumGpu<'a> {
    gpu: &'a GpuF64,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl<'a> FesGaussianSumGpu<'a> {
    /// Compile the FES shader and cache the pipeline.
    #[must_use]
    pub fn new(gpu: &'a GpuF64) -> Self {
        let device = gpu.device();

        let shader = gpu
            .to_wgpu_device()
            .compile_shader_f64(FES_SHADER, Some("hotSpring FES Gaussian Sum f64"));

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FES BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FES PL"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fes_gaussian_sum"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self { gpu, pipeline, bgl }
    }

    /// Reconstruct a 2D FES from packed HILLS data on GPU.
    ///
    /// `hills_packed`: flat array of [cx, cy, sx, sy, h] × n_gaussians (5 f64 per Gaussian).
    ///
    /// Returns the FES grid (flat, row-major: fes[iy * nbins_x + ix]) shifted to min=0.
    pub fn reconstruct_2d(
        &self,
        hills_packed: &[f64],
        n_gaussians: usize,
        grid_min_x: f64,
        grid_max_x: f64,
        grid_min_y: f64,
        grid_max_y: f64,
        nbins_x: usize,
        nbins_y: usize,
    ) -> FesGpuResult {
        let device = self.gpu.device();
        let queue = self.gpu.queue();
        let total_bins = nbins_x * nbins_y;
        let start = std::time::Instant::now();

        let hills_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FES hills"),
            contents: bytemuck::cast_slice(hills_packed),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FES output"),
            size: (total_bins * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = FesParams::new(
            n_gaussians as u32,
            nbins_x as u32,
            nbins_y as u32,
            grid_min_x,
            grid_max_x,
            grid_min_y,
            grid_max_y,
        );

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FES params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FES BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: hills_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        let workgroups = (total_bins as u32 + 63) / 64;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FES encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FES pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FES readback"),
            size: (total_bins * 8) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, (total_bins * 8) as u64);

        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let data = slice.get_mapped_range();
        let raw: &[f64] = bytemuck::cast_slice(&data);
        let mut fes: Vec<f64> = raw.to_vec();
        drop(data);
        readback_buf.unmap();

        let min_val = fes.iter().cloned().fold(f64::INFINITY, f64::min);
        for f in &mut fes {
            *f -= min_val;
        }

        let gpu_secs = start.elapsed().as_secs_f64();

        FesGpuResult {
            fes,
            grid_min_x,
            grid_max_x,
            grid_min_y,
            grid_max_y,
            nbins_x,
            nbins_y,
            gpu_secs,
        }
    }
}
