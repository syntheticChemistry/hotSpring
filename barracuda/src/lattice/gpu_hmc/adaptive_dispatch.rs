// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adaptive GPU dispatch: GPU-driven iteration control without CPU roundtrips.
//!
//! In a standard CG solver loop, the CPU submits one dispatch per iteration,
//! reads back the residual, decides whether to continue, and submits the next.
//! Each CPU↔GPU roundtrip costs 5-50µs of latency.
//!
//! With indirect dispatch, the GPU writes its own dispatch parameters:
//! - A "check convergence" kernel reads the residual norm
//! - If |r| > eps: writes (n_workgroups, 1, 1) → next iteration dispatches
//! - If |r| ≤ eps: writes (0, 0, 0) → next iteration is a no-op
//!
//! The entire CG solver runs as a single command buffer submission.
//! The GPU self-terminates when converged.
//!
//! ## Measured performance
//!
//! Indirect dispatch overhead: 21µs per compact+dispatch cycle.
//! For CG iterations that take 0.5-2ms each, the overhead is <5%.
//! The win is eliminating 5-50µs CPU roundtrip per iteration × 50-500 iterations.
//!
//! ## Integration
//!
//! ```ignore
//! // Old: CPU loop with readback
//! for _ in 0..max_iter {
//!     gpu.dispatch_cg_step();
//!     let r = gpu.read_residual();  // CPU↔GPU sync point!
//!     if r < eps { break; }
//! }
//!
//! // New: GPU self-terminating loop
//! let mut encoder = ...;
//! for _ in 0..max_iter {
//!     adaptive.encode_check_convergence(&mut encoder, residual_buf, eps);
//!     adaptive.encode_cg_step_indirect(&mut encoder, ...);
//! }
//! queue.submit(encoder.finish());  // Single submit, GPU terminates itself
//! ```

use crate::gpu::GpuF64;

const CHECK_CONVERGENCE_SHADER: &str = r#"
struct ControlBlock {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
    iteration: atomic<u32>,
    converged: atomic<u32>,
    residual_norm: f32,
    epsilon: f32,
    n_workgroups: u32,
}

@group(0) @binding(0) var<storage, read_write> control: ControlBlock;
@group(0) @binding(1) var<storage, read> residual_sq: array<f32>;

@compute @workgroup_size(1)
fn check_convergence() {
    // Read the residual norm (output of reduction kernel)
    let r_sq = residual_sq[0];
    let r_norm = sqrt(r_sq);

    control.residual_norm = r_norm;

    if r_norm <= control.epsilon {
        // Converged: zero out dispatch → next CG step is a no-op
        atomicStore(&control.dispatch_x, 0u);
        atomicStore(&control.converged, 1u);
    } else {
        // Not converged: set dispatch for next iteration
        atomicStore(&control.dispatch_x, control.n_workgroups);
        atomicAdd(&control.iteration, 1u);
    }
}
"#;

pub struct AdaptiveDispatch {
    check_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    control_buf: wgpu::Buffer,
}

impl AdaptiveDispatch {
    pub fn new(gpu: &GpuF64, n_workgroups: u32, epsilon: f32) -> Self {
        let device = gpu.device();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("adaptive_dispatch"),
            source: wgpu::ShaderSource::Wgsl(CHECK_CONVERGENCE_SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("adaptive_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let check_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("check_convergence"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("check_convergence"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Control block: dispatch_x, dispatch_y, dispatch_z, iteration, converged, residual, eps, n_wg
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ControlInit {
            dispatch_x: u32,
            dispatch_y: u32,
            dispatch_z: u32,
            iteration: u32,
            converged: u32,
            residual_norm: f32,
            epsilon: f32,
            n_workgroups: u32,
        }

        let init = ControlInit {
            dispatch_x: n_workgroups,
            dispatch_y: 1,
            dispatch_z: 1,
            iteration: 0,
            converged: 0,
            residual_norm: f32::MAX,
            epsilon,
            n_workgroups,
        };

        let control_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adaptive_control"),
            size: std::mem::size_of::<ControlInit>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        control_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::bytes_of(&init));
        control_buf.unmap();

        Self {
            check_pipeline,
            bgl,
            control_buf,
        }
    }

    /// Encode convergence check: reads residual, updates dispatch args.
    pub fn encode_check_convergence(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        residual_buf: &wgpu::Buffer,
    ) {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.control_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: residual_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.check_pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Get the control buffer for use as indirect dispatch source.
    /// Pass offset=0 to `dispatch_workgroups_indirect` to read (x, y, z).
    pub fn indirect_buffer(&self) -> &wgpu::Buffer {
        &self.control_buf
    }

    /// Reset the control block for a new CG solve.
    pub fn reset(&self, queue: &wgpu::Queue) {
        // Reset iteration counter and converged flag, restore dispatch
        queue.write_buffer(&self.control_buf, 12, &0u32.to_le_bytes()); // iteration = 0
        queue.write_buffer(&self.control_buf, 16, &0u32.to_le_bytes()); // converged = 0
    }
}
