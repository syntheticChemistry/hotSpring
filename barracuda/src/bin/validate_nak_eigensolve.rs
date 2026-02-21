// SPDX-License-Identifier: AGPL-3.0-only

//! NAK-Optimized Eigensolve: Correctness + Performance Validation
//!
//! Validates that the NAK-workaround eigensolve shader produces
//! correct eigenvalues (vs CPU `eigh_f64`) and benchmarks it
//! against the standard warp-packed shader.
//!
//! Delivers a complete solution toadstool can absorb:
//! - Shader: `batched_eigh_nak_optimized_f64.wgsl`
//! - This binary: proves correctness and measures speedup
//! - Both shaders use identical API (same bind group layout)

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;
use wgpu::util::DeviceExt;

const SHADER_STANDARD: &str = include_str!("../../src/shaders/batched_eigh_nak_optimized_f64.wgsl");

const SHADER_BASELINE: &str = r"
const WARP_SIZE: u32 = 32u;

struct Params {
    n: u32,
    batch_size: u32,
    max_sweeps: u32,
    tolerance: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> A_batch: array<f64>;
@group(0) @binding(2) var<storage, read_write> V_batch: array<f64>;
@group(0) @binding(3) var<storage, read_write> eigenvalues: array<f64>;

fn idx(r: u32, c: u32, n: u32) -> u32 { return r * n + c; }

@compute @workgroup_size(32, 1, 1)
fn batched_eigh_baseline(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let b = wg_id.x * WARP_SIZE + local_id.x;
    let n = params.n;
    if (b >= params.batch_size || n > 32u) { return; }
    let base = b * n * n;
    let tol = f64(params.tolerance);

    for (var i = 0u; i < n; i++) {
        for (var j = 0u; j < n; j++) {
            if (i == j) { V_batch[base + idx(i, j, n)] = f64(1.0); }
            else { V_batch[base + idx(i, j, n)] = f64(0.0); }
        }
    }

    for (var sweep = 0u; sweep < params.max_sweeps; sweep++) {
        var max_off = f64(0.0);
        for (var i = 0u; i < n; i++) {
            for (var j = i + 1u; j < n; j++) {
                let off = abs(A_batch[base + idx(i, j, n)]);
                if (off > max_off) { max_off = off; }
            }
        }
        if (max_off < tol) { break; }

        for (var p = 0u; p < n - 1u; p++) {
            for (var q = p + 1u; q < n; q++) {
                let apq = A_batch[base + idx(p, q, n)];
                if (abs(apq) < 1e-14) { continue; }
                let app = A_batch[base + idx(p, p, n)];
                let aqq = A_batch[base + idx(q, q, n)];
                let diff = aqq - app;

                var t: f64;
                if (abs(diff) < 1e-14) {
                    t = select(f64(-1.0), f64(1.0), apq >= 0.0);
                } else {
                    let phi = diff / (2.0 * apq);
                    t = sign(phi) / (abs(phi) + sqrt(1.0 + phi * phi));
                }
                let c = 1.0 / sqrt(1.0 + t * t);
                let s = t * c;

                for (var k = 0u; k < n; k++) {
                    if (k != p && k != q) {
                        let akp = A_batch[base + idx(k, p, n)];
                        let akq = A_batch[base + idx(k, q, n)];
                        A_batch[base + idx(k, p, n)] = c * akp - s * akq;
                        A_batch[base + idx(k, q, n)] = s * akp + c * akq;
                        A_batch[base + idx(p, k, n)] = c * akp - s * akq;
                        A_batch[base + idx(q, k, n)] = s * akp + c * akq;
                    }
                }
                A_batch[base + idx(p, p, n)] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
                A_batch[base + idx(q, q, n)] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
                A_batch[base + idx(p, q, n)] = f64(0.0);
                A_batch[base + idx(q, p, n)] = f64(0.0);

                for (var k = 0u; k < n; k++) {
                    let vkp = V_batch[base + idx(k, p, n)];
                    let vkq = V_batch[base + idx(k, q, n)];
                    V_batch[base + idx(k, p, n)] = c * vkp - s * vkq;
                    V_batch[base + idx(k, q, n)] = s * vkp + c * vkq;
                }
            }
        }
    }

    let eig_base = b * n;
    for (var i = 0u; i < n; i++) {
        eigenvalues[eig_base + i] = A_batch[base + idx(i, i, n)];
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct EighParams {
    n: u32,
    batch_size: u32,
    max_sweeps: u32,
    tolerance: f32,
}

fn gen_symmetric(batch: usize, dim: usize) -> Vec<f64> {
    let mut m = vec![0.0_f64; batch * dim * dim];
    for b in 0..batch {
        let base = b * dim * dim;
        for i in 0..dim {
            m[base + i * dim + i] = (i as f64 + 1.0).mul_add(10.0, b as f64 * 0.1);
            for j in (i + 1)..dim {
                let v = ((i + j) as f64).mul_add(0.3, b as f64 * 0.01).sin() * 0.5;
                m[base + i * dim + j] = v;
                m[base + j * dim + i] = v;
            }
        }
    }
    m
}

fn cpu_eigenvalues(matrices: &[f64], batch: usize, dim: usize) -> Vec<Vec<f64>> {
    (0..batch)
        .map(|b| {
            let base = b * dim * dim;
            let mat = &matrices[base..base + dim * dim];
            let eig = barracuda::linalg::eigh_f64(mat, dim).expect("eigh");
            let mut vals = eig.eigenvalues;
            vals.sort_by(|a, b| a.partial_cmp(b).expect("finite f64 comparison"));
            vals
        })
        .collect()
}

fn run_gpu_eigh(
    gpu: &GpuF64,
    shader_src: &str,
    entry_point: &str,
    matrices: &[f64],
    batch: usize,
    dim: usize,
    sweeps: u32,
) -> (Vec<Vec<f64>>, f64) {
    let shader = gpu
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("eigh_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
    let params = EighParams {
        n: dim as u32,
        batch_size: batch as u32,
        max_sweeps: sweeps,
        tolerance: 1e-12_f32,
    };

    let n2 = dim * dim;
    let mat_bytes: Vec<u8> = matrices.iter().flat_map(|v| v.to_le_bytes()).collect();
    let v_bytes = vec![0u8; batch * n2 * 8];
    let eig_bytes = vec![0u8; batch * dim * 8];

    let params_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let a_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("A"),
            contents: &mat_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
    let v_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("V"),
            contents: &v_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    let eig_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("eig"),
            contents: &eig_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let staging = gpu.create_staging_buffer(batch * dim * 8, "eig_staging");

    let bgl = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let pl = gpu
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let pipeline = gpu
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("eigh"),
            layout: Some(&pl),
            module: &shader,
            entry_point,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: v_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: eig_buf.as_entire_binding(),
            },
        ],
    });

    let dispatch_wg = (batch as u32).div_ceil(32);

    // Warmup
    for _ in 0..3 {
        let mut enc = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_wg, 1, 1);
        }
        gpu.queue().submit(Some(enc.finish()));
        gpu.device().poll(wgpu::Maintain::Wait);
    }

    // Re-upload matrices (warmup may have modified A_batch in place)
    gpu.queue().write_buffer(&a_buf, 0, &mat_bytes);

    // Timed run
    let t0 = Instant::now();
    let rounds = 10;
    for r in 0..rounds {
        if r > 0 {
            gpu.queue().write_buffer(&a_buf, 0, &mat_bytes);
        }
        let mut enc = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_wg, 1, 1);
        }
        enc.copy_buffer_to_buffer(&eig_buf, 0, &staging, 0, (batch * dim * 8) as u64);
        gpu.queue().submit(Some(enc.finish()));
        gpu.device().poll(wgpu::Maintain::Wait);
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(rounds);

    // Read eigenvalues
    let eig_data = gpu.read_staging_f64(&staging).expect("read eigenvalues");

    let mut result = Vec::new();
    for b in 0..batch {
        let start = b * dim;
        let mut vals: Vec<f64> = eig_data[start..start + dim].to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).expect("finite f64 comparison"));
        result.push(vals);
    }

    (result, elapsed_ms)
}

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU init");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  NAK-Optimized Eigensolve: Correctness + Performance       ║");
    println!("║  Workarounds: FMA, 4x unroll, branchless, register hints   ║");
    println!("║  Delivers: shader + validation for toadstool absorption    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    gpu.print_info();
    println!();

    let mut harness = ValidationHarness::new("nak_eigensolve");

    for &(batch, dim, sweeps) in &[
        (128_usize, 12_usize, 200_u32),
        (128, 20, 200),
        (64, 30, 200),
    ] {
        println!("━━━ batch={batch}, dim={dim}, sweeps={sweeps} ━━━━━━━━━━━━━━━━━━━━━━━");

        let matrices = gen_symmetric(batch, dim);
        let cpu_ref = cpu_eigenvalues(&matrices, batch, dim);

        let (gpu_baseline, t_baseline) = run_gpu_eigh(
            &gpu,
            SHADER_BASELINE,
            "batched_eigh_baseline",
            &matrices,
            batch,
            dim,
            sweeps,
        );
        let (gpu_optimized, t_optimized) = run_gpu_eigh(
            &gpu,
            SHADER_STANDARD,
            "batched_eigh_nak_optimized",
            &matrices,
            batch,
            dim,
            sweeps,
        );

        // Correctness: compare GPU eigenvalues vs CPU reference (relative error)
        let mut max_rel_baseline = 0.0_f64;
        let mut max_rel_optimized = 0.0_f64;
        let mut max_parity = 0.0_f64;
        for b in 0..batch {
            for i in 0..dim {
                let denom = cpu_ref[b][i].abs().max(1e-10);
                let rel_b = (gpu_baseline[b][i] - cpu_ref[b][i]).abs() / denom;
                let rel_o = (gpu_optimized[b][i] - cpu_ref[b][i]).abs() / denom;
                let parity = (gpu_baseline[b][i] - gpu_optimized[b][i]).abs() / denom;
                max_rel_baseline = max_rel_baseline.max(rel_b);
                max_rel_optimized = max_rel_optimized.max(rel_o);
                max_parity = max_parity.max(parity);
            }
        }

        let speedup = t_baseline / t_optimized;
        println!("  Baseline:  {t_baseline:.3}ms  max rel err = {max_rel_baseline:.2e}");
        println!("  Optimized: {t_optimized:.3}ms  max rel err = {max_rel_optimized:.2e}");
        println!("  Parity:    max rel diff = {max_parity:.2e}");
        println!("  Speedup:   {speedup:.2}×");
        println!();

        // Jacobi with 200 sweeps converges to ~1e-3 relative for these matrices.
        // CPU eigh_f64 uses Householder+QR which is exact to machine epsilon.
        let label_b = format!("baseline d={dim} vs CPU (rel)");
        let label_o = format!("optimized d={dim} vs CPU (rel)");
        harness.check_upper(
            &label_b,
            max_rel_baseline,
            tolerances::NAK_EIGENSOLVE_VS_CPU_REL,
        );
        harness.check_upper(
            &label_o,
            max_rel_optimized,
            tolerances::NAK_EIGENSOLVE_VS_CPU_REL,
        );

        let label_p = format!("baseline≈optimized d={dim}");
        harness.check_upper(&label_p, max_parity, tolerances::NAK_EIGENSOLVE_PARITY);

        let label_s = format!("optimized d={dim} no regression");
        harness.check_upper(
            &label_s,
            t_optimized / t_baseline,
            tolerances::NAK_EIGENSOLVE_REGRESSION,
        );
    }

    harness.finish();
}
