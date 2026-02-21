// SPDX-License-Identifier: AGPL-3.0-only

//! GPU BCS Bisection — hotSpring Nuclear HFB Implementation
//!
//! Solves batched BCS chemical-potential problems on GPU:
//!   Find μ such that Σ_k deg_k · v²_k(μ) = N
//!
//! This is hotSpring's domain-specific BCS solver with nuclear HFB features:
//!   - `solve_bcs_with_degeneracy()` for shell-model level degeneracies (2j+1)
//!   - Custom buffer layout matching nuclear physics conventions
//!
//! **Status (Feb 16 2026)**: The `target` → `target_val` WGSL keyword fix has
//! been absorbed by ToadStool (commit `0c477306`). This local shader was always
//! correct (uses `target_n` for particle number). We retain the local copy for
//! the domain-specific `use_degeneracy` feature that ToadStool's generic
//! `BatchedBisectionGpu` does not provide.
//!
//! Uses the same buffer layout and uniform struct as BarraCUDA's
//! `BatchedBisectionGpu`, so results are directly comparable.

use crate::gpu::GpuF64;
use barracuda::shaders::precision::ShaderTemplate;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Uniform params for the bisection shader.
/// Must match `BisectionParams` in `bcs_bisection_f64.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BisectionParams {
    batch_size: u32,
    max_iterations: u32,
    n_levels: u32,
    use_degeneracy: u32,
    // f64 tolerance encoded as two u32s for uniform buffer alignment
    tolerance_lo: u32,
    tolerance_hi: u32,
}

impl BisectionParams {
    fn new(batch_size: u32, max_iterations: u32, n_levels: u32, use_deg: bool, tol: f64) -> Self {
        let bits = tol.to_bits();
        Self {
            batch_size,
            max_iterations,
            n_levels,
            use_degeneracy: u32::from(use_deg),
            tolerance_lo: bits as u32,
            tolerance_hi: (bits >> 32) as u32,
        }
    }
}

/// Result of batched BCS bisection.
pub struct BcsResult {
    /// Chemical potentials (μ) for each problem.
    pub roots: Vec<f64>,
    /// Number of bisection iterations used per problem.
    pub iterations: Vec<u32>,
}

/// Local GPU BCS bisection solver.
///
/// Compiles the WGSL shader once at creation and caches the pipeline
/// for reuse across multiple `solve_bcs` / `solve_bcs_with_degeneracy` calls.
pub struct BcsBisectionGpu<'a> {
    gpu: &'a GpuF64,
    max_iterations: u32,
    tolerance: f64,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl<'a> BcsBisectionGpu<'a> {
    /// Create a new BCS bisection solver.
    ///
    /// Compiles the BCS WGSL shader and caches the compute pipeline.
    ///
    /// # Arguments
    /// * `gpu` - hotSpring's GpuF64 device (has SHADER_F64)
    /// * `max_iterations` - Max bisection iterations per problem (50–100 typical)
    /// * `tolerance` - Convergence tolerance (1e-10 to 1e-14)
    pub fn new(gpu: &'a GpuF64, max_iterations: u32, tolerance: f64) -> Self {
        let device = gpu.device();
        let shader_body = include_str!("shaders/bcs_bisection_f64.wgsl");
        let shader_src =
            ShaderTemplate::for_device_auto(shader_body, gpu.to_wgpu_device().as_ref());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hotSpring BCS Bisection f64"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BCS BGL"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry_u32(4, false),
                uniform_entry(5),
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BCS PL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bcs_bisection"),
            layout: Some(&pl),
            module: &shader,
            entry_point: "bcs_bisection",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            gpu,
            max_iterations,
            tolerance,
            pipeline,
            bgl,
        }
    }

    /// Solve BCS pairing: find μ where Σ v²_k(μ) = N.
    ///
    /// # Arguments
    /// * `lower` - Lower bounds for μ \[batch_size\]
    /// * `upper` - Upper bounds for μ \[batch_size\]
    /// * `eigenvalues` - Single-particle energies \[batch_size × n_levels\]
    /// * `delta` - Pairing gap per problem \[batch_size\]
    /// * `target_n` - Target particle number per problem \[batch_size\]
    pub fn solve_bcs(
        &self,
        lower: &[f64],
        upper: &[f64],
        eigenvalues: &[f64],
        delta: &[f64],
        target_n: &[f64],
    ) -> Result<BcsResult, String> {
        let batch_size = lower.len();
        if batch_size == 0 {
            return Ok(BcsResult {
                roots: vec![],
                iterations: vec![],
            });
        }
        let n_levels = eigenvalues.len() / batch_size;

        // Pack params: [ε_0..ε_{n-1}, Δ, N] per problem
        let mut params = Vec::with_capacity(batch_size * (n_levels + 2));
        for i in 0..batch_size {
            for j in 0..n_levels {
                params.push(eigenvalues[i * n_levels + j]);
            }
            params.push(delta[i]);
            params.push(target_n[i]);
        }

        self.dispatch(
            lower,
            upper,
            &params,
            n_levels as u32,
            false,
            "bcs_bisection",
        )
    }

    /// Solve BCS with level degeneracy (nuclear HFB: deg_k = 2j+1).
    ///
    /// # Arguments
    /// * `lower` - Lower bounds for μ \[batch_size\]
    /// * `upper` - Upper bounds for μ \[batch_size\]
    /// * `eigenvalues` - Energy levels \[batch_size × n_levels\]
    /// * `degeneracies` - Degeneracy per level \[batch_size × n_levels\]
    /// * `delta` - Pairing gap per problem \[batch_size\]
    /// * `target_n` - Target particle number per problem \[batch_size\]
    pub fn solve_bcs_with_degeneracy(
        &self,
        lower: &[f64],
        upper: &[f64],
        eigenvalues: &[f64],
        degeneracies: &[f64],
        delta: &[f64],
        target_n: &[f64],
    ) -> Result<BcsResult, String> {
        let batch_size = lower.len();
        if batch_size == 0 {
            return Ok(BcsResult {
                roots: vec![],
                iterations: vec![],
            });
        }
        let n_levels = eigenvalues.len() / batch_size;

        // Pack params: [ε_0..ε_{n-1}, deg_0..deg_{n-1}, Δ, N] per problem
        let mut params = Vec::with_capacity(batch_size * (n_levels * 2 + 2));
        for i in 0..batch_size {
            for j in 0..n_levels {
                params.push(eigenvalues[i * n_levels + j]);
            }
            for j in 0..n_levels {
                params.push(degeneracies[i * n_levels + j]);
            }
            params.push(delta[i]);
            params.push(target_n[i]);
        }

        self.dispatch(
            lower,
            upper,
            &params,
            n_levels as u32,
            true,
            "bcs_bisection",
        )
    }

    fn dispatch(
        &self,
        lower: &[f64],
        upper: &[f64],
        params: &[f64],
        n_levels: u32,
        use_deg: bool,
        _entry_point: &str,
    ) -> Result<BcsResult, String> {
        let batch_size = lower.len();
        let device = self.gpu.device();
        let queue = self.gpu.queue();

        // Upload buffers
        let lower_buf = make_f64_buf(device, "lower", lower);
        let upper_buf = make_f64_buf(device, "upper", upper);
        let params_buf = make_f64_buf(device, "params", params);
        let roots_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roots"),
            size: (batch_size * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let iter_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iterations"),
            size: (batch_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let config = BisectionParams::new(
            batch_size as u32,
            self.max_iterations,
            n_levels,
            use_deg,
            self.tolerance,
        );
        let config_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("config"),
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BCS BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lower_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: upper_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: roots_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: iter_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: config_buf.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let n_wg = batch_size.div_ceil(64);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("BCS encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BCS pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_wg as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));

        // Read back
        let roots = read_f64(device, queue, &roots_buf, batch_size)?;
        let iterations = read_u32(device, queue, &iter_buf, batch_size)?;

        Ok(BcsResult { roots, iterations })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

const fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const fn storage_entry_u32(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    // Same as storage_entry — u32 vs f64 is a shader-side concern
    storage_entry(binding, read_only)
}

const fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_f64_buf(device: &wgpu::Device, label: &str, data: &[f64]) -> wgpu::Buffer {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &bytes,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn read_f64(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<f64>, String> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("f64 staging"),
        size: (count * 8) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("f64 readback"),
    });
    enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 8) as u64);
    queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("channel: {e}"))?
        .map_err(|e| format!("map: {e}"))?;

    let data = slice.get_mapped_range();
    let result: Vec<f64> = data
        .chunks_exact(8)
        .map(|c| {
            // SAFETY: chunks_exact(8) guarantees 8-byte slices
            #[allow(clippy::expect_used)]
            f64::from_le_bytes(c.try_into().expect("8-byte chunk"))
        })
        .collect();
    drop(data);
    staging.unmap();
    Ok(result)
}

fn read_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<u32>, String> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("u32 staging"),
        size: (count * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("u32 readback"),
    });
    enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 4) as u64);
    queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("channel: {e}"))?
        .map_err(|e| format!("map: {e}"))?;

    let data = slice.get_mapped_range();
    let result: Vec<u32> = data
        .chunks_exact(4)
        .map(|c| {
            // SAFETY: chunks_exact(4) guarantees 4-byte slices
            #[allow(clippy::expect_used)]
            u32::from_le_bytes(c.try_into().expect("4-byte chunk"))
        })
        .collect();
    drop(data);
    staging.unmap();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)] // exact known constant roundtrip
    fn bisection_params_tolerance_roundtrip() {
        let tol = 1e-12_f64;
        let params = BisectionParams::new(4, 100, 10, true, tol);
        let reconstructed =
            f64::from_bits(u64::from(params.tolerance_lo) | (u64::from(params.tolerance_hi) << 32));
        assert_eq!(
            reconstructed, tol,
            "tolerance should round-trip through u32 pair"
        );
    }

    #[test]
    fn bisection_params_degeneracy_flag() {
        let with_deg = BisectionParams::new(1, 50, 5, true, 1e-10);
        assert_eq!(with_deg.use_degeneracy, 1);

        let no_deg = BisectionParams::new(1, 50, 5, false, 1e-10);
        assert_eq!(no_deg.use_degeneracy, 0);
    }

    #[test]
    fn bisection_params_layout_size() {
        let params = BisectionParams::new(1, 1, 1, false, 1e-10);
        let bytes = bytemuck::bytes_of(&params);
        // 6 × u32 = 24 bytes
        assert_eq!(bytes.len(), 24, "BisectionParams should be 24 bytes");
    }

    #[test]
    fn bcs_param_packing_no_degeneracy() {
        let batch_size = 2;
        let n_levels = 3;
        let eigenvalues = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 × 3
        let delta = [0.5, 0.6];
        let target_n = [10.0, 12.0];

        let mut params = Vec::with_capacity(batch_size * (n_levels + 2));
        for i in 0..batch_size {
            for j in 0..n_levels {
                params.push(eigenvalues[i * n_levels + j]);
            }
            params.push(delta[i]);
            params.push(target_n[i]);
        }

        // Problem 0: [1, 2, 3, 0.5, 10]
        assert_eq!(params[0..5], [1.0, 2.0, 3.0, 0.5, 10.0]);
        // Problem 1: [4, 5, 6, 0.6, 12]
        assert_eq!(params[5..10], [4.0, 5.0, 6.0, 0.6, 12.0]);
    }

    #[test]
    fn bcs_param_packing_with_degeneracy() {
        let batch_size = 1;
        let n_levels = 2;
        let eigenvalues = [1.0, 2.0];
        let degeneracies = [6.0, 10.0]; // 2j+1 for j=5/2, 9/2
        let delta = [0.5];
        let target_n = [16.0];

        let mut params = Vec::with_capacity(batch_size * (n_levels * 2 + 2));
        for i in 0..batch_size {
            for j in 0..n_levels {
                params.push(eigenvalues[i * n_levels + j]);
            }
            for j in 0..n_levels {
                params.push(degeneracies[i * n_levels + j]);
            }
            params.push(delta[i]);
            params.push(target_n[i]);
        }

        // [ε₀, ε₁, deg₀, deg₁, Δ, N]
        assert_eq!(params, vec![1.0, 2.0, 6.0, 10.0, 0.5, 16.0]);
    }

    #[test]
    fn bcs_result_fields() {
        let result = BcsResult {
            roots: vec![-5.0, -3.2],
            iterations: vec![42, 38],
        };
        assert_eq!(result.roots.len(), 2);
        assert_eq!(result.iterations[0], 42);
    }
}
