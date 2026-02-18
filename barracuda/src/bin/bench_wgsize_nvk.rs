// SPDX-License-Identifier: AGPL-3.0-only

//! Diagnostic: workgroup_size(1) vs workgroup_size(N) on NVK vs proprietary.
//!
//! Proves that barracuda's `batched_eigh_single_dispatch_f64.wgsl` is slow on
//! NVK because it uses `@workgroup_size(1, 1, 1)` — a single thread per
//! workgroup doing O(n³) serial work. NVK's shader compiler (NAK) handles
//! this degenerate pattern poorly compared to nvidia proprietary.
//!
//! BCS bisection uses `@workgroup_size(64)` and is *faster* on the Titan V
//! than the 4070, proving GV100 hardware is not the bottleneck.
//!
//! Test: Jacobi-rotation-like matrix sweep in two modes:
//!   A) workgroup_size(1): one thread per matrix (current eigensolve pattern)
//!   B) workgroup_size(32): 32 threads cooperate per matrix via shared memory
//!
//! Expected: Mode B is faster on both GPUs, but the speedup on NVK is
//! dramatically larger because NVK penalizes mode A much more heavily.

use hotspring_barracuda::gpu::GpuF64;
use std::time::Instant;
use wgpu::util::DeviceExt;

const WARMUP: usize = 3;
const ROUNDS: usize = 10;

const SHADER_WG1: &str = r"
// Jacobi-like sweep: workgroup_size(1), one thread does ALL work per matrix.
// This is the pattern used by batched_eigh_single_dispatch_f64.wgsl.

struct Params {
    n: u32,
    batch_size: u32,
    sweeps: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> matrices: array<f64>;

var<workgroup> A: array<f64, 1024>;  // 32x32

fn idx(r: u32, c: u32, n: u32) -> u32 { return r * n + c; }

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>) {
    let b = wg.x;
    let n = params.n;
    if (b >= params.batch_size || n > 32u) { return; }

    let base = b * n * n;

    // Load to shared
    for (var i = 0u; i < n; i++) {
        for (var j = 0u; j < n; j++) {
            A[idx(i, j, n)] = matrices[base + idx(i, j, n)];
        }
    }

    // Jacobi sweeps: cyclic rotation on all (p,q) pairs
    for (var sweep = 0u; sweep < params.sweeps; sweep++) {
        for (var p = 0u; p < n - 1u; p++) {
            for (var q = p + 1u; q < n; q++) {
                let apq = A[idx(p, q, n)];
                if (abs(apq) < 1e-14) { continue; }

                let app = A[idx(p, p, n)];
                let aqq = A[idx(q, q, n)];
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

                // Rotate rows/cols
                for (var k = 0u; k < n; k++) {
                    if (k != p && k != q) {
                        let akp = A[idx(k, p, n)];
                        let akq = A[idx(k, q, n)];
                        A[idx(k, p, n)] = c * akp - s * akq;
                        A[idx(k, q, n)] = s * akp + c * akq;
                        A[idx(p, k, n)] = c * akp - s * akq;
                        A[idx(q, k, n)] = s * akp + c * akq;
                    }
                }

                A[idx(p, p, n)] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
                A[idx(q, q, n)] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
                A[idx(p, q, n)] = f64(0.0);
                A[idx(q, p, n)] = f64(0.0);
            }
        }
    }

    // Write back
    for (var i = 0u; i < n; i++) {
        for (var j = 0u; j < n; j++) {
            matrices[base + idx(i, j, n)] = A[idx(i, j, n)];
        }
    }
}
";

const SHADER_WG32: &str = r"
// Jacobi-like sweep: workgroup_size(32), threads cooperate on rotation rows.
// Same total work as WG1, but row operations distributed across 32 threads.

struct Params {
    n: u32,
    batch_size: u32,
    sweeps: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> matrices: array<f64>;

var<workgroup> A: array<f64, 1024>;
var<workgroup> cs: array<f64, 2>;  // cos, sin shared across threads

fn idx(r: u32, c: u32, n: u32) -> u32 { return r * n + c; }

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = wg.x;
    let tid = lid.x;
    let n = params.n;
    if (b >= params.batch_size || n > 32u) { return; }

    let base = b * n * n;

    // Cooperative load: each thread loads n/32 rows (or part thereof)
    for (var i = tid; i < n * n; i += 32u) {
        A[i] = matrices[base + i];
    }
    workgroupBarrier();

    for (var sweep = 0u; sweep < params.sweeps; sweep++) {
        for (var p = 0u; p < n - 1u; p++) {
            for (var q = p + 1u; q < n; q++) {
                // Thread 0 computes rotation angle
                if (tid == 0u) {
                    let apq = A[idx(p, q, n)];
                    let app = A[idx(p, p, n)];
                    let aqq = A[idx(q, q, n)];

                    if (abs(apq) < 1e-14) {
                        cs[0] = f64(1.0);
                        cs[1] = f64(0.0);
                    } else {
                        let diff = aqq - app;
                        var t: f64;
                        if (abs(diff) < 1e-14) {
                            t = select(f64(-1.0), f64(1.0), apq >= 0.0);
                        } else {
                            let phi = diff / (2.0 * apq);
                            t = sign(phi) / (abs(phi) + sqrt(1.0 + phi * phi));
                        }
                        let c_val = 1.0 / sqrt(1.0 + t * t);
                        cs[0] = c_val;
                        cs[1] = t * c_val;
                    }
                }
                workgroupBarrier();

                let c_val = cs[0];
                let s_val = cs[1];

                if (s_val == f64(0.0)) { continue; }

                // Parallel rotation: each thread handles a subset of rows
                for (var k = tid; k < n; k += 32u) {
                    if (k != p && k != q) {
                        let akp = A[idx(k, p, n)];
                        let akq = A[idx(k, q, n)];
                        let new_p = c_val * akp - s_val * akq;
                        let new_q = s_val * akp + c_val * akq;
                        A[idx(k, p, n)] = new_p;
                        A[idx(k, q, n)] = new_q;
                        A[idx(p, k, n)] = new_p;
                        A[idx(q, k, n)] = new_q;
                    }
                }

                // Thread 0 updates the 2x2 block
                if (tid == 0u) {
                    let app = A[idx(p, p, n)];
                    let aqq = A[idx(q, q, n)];
                    let apq = A[idx(p, q, n)];
                    A[idx(p, p, n)] = c_val * c_val * app - 2.0 * c_val * s_val * apq + s_val * s_val * aqq;
                    A[idx(q, q, n)] = s_val * s_val * app + 2.0 * c_val * s_val * apq + c_val * c_val * aqq;
                    A[idx(p, q, n)] = f64(0.0);
                    A[idx(q, p, n)] = f64(0.0);
                }
                workgroupBarrier();
            }
        }
    }

    // Cooperative write-back
    for (var i = tid; i < n * n; i += 32u) {
        matrices[base + i] = A[i];
    }
}
";

// Warp-packed: 32 threads in one workgroup, each independently solving
// a DIFFERENT matrix. No barriers, no cooperation. Full SIMD lane utilization.
// Each thread uses private (register) memory, not shared memory.
// Dispatch: (batch/32, 1, 1) workgroups.
const SHADER_WARP_PACKED: &str = r"
struct Params {
    n: u32,
    batch_size: u32,
    sweeps: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> matrices: array<f64>;

// Each thread loads its own matrix into PRIVATE registers.
// Max n=20: 20*20*8 = 3200 bytes per thread. With 32 threads/warp,
// that's 100KB total — may spill to local memory on register-limited GPUs.
// For n<=12, fits comfortably in registers.

fn idx(r: u32, c: u32, n: u32) -> u32 { return r * n + c; }

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = wg.x * 32u + lid.x;
    let n = params.n;
    if (b >= params.batch_size || n > 32u) { return; }

    let base = b * n * n;

    // Load matrix into global memory offsets (no shared memory).
    // Each thread works independently on its own slice of the storage buffer.
    for (var sweep = 0u; sweep < params.sweeps; sweep++) {
        for (var p = 0u; p < n - 1u; p++) {
            for (var q = p + 1u; q < n; q++) {
                let apq = matrices[base + idx(p, q, n)];

                if (abs(apq) < 1e-14) { continue; }

                let app = matrices[base + idx(p, p, n)];
                let aqq = matrices[base + idx(q, q, n)];
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
                        let akp = matrices[base + idx(k, p, n)];
                        let akq = matrices[base + idx(k, q, n)];
                        matrices[base + idx(k, p, n)] = c * akp - s * akq;
                        matrices[base + idx(k, q, n)] = s * akp + c * akq;
                        matrices[base + idx(p, k, n)] = c * akp - s * akq;
                        matrices[base + idx(q, k, n)] = s * akp + c * akq;
                    }
                }

                matrices[base + idx(p, p, n)] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
                matrices[base + idx(q, q, n)] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
                matrices[base + idx(p, q, n)] = f64(0.0);
                matrices[base + idx(q, p, n)] = f64(0.0);
            }
        }
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    batch_size: u32,
    sweeps: u32,
    _pad: u32,
}

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU init");

    println!("═══════════════════════════════════════════════════════════");
    println!("  NVK Workgroup-Size Diagnostic");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    gpu.print_info();
    GpuF64::print_available_adapters();
    println!();
    println!("  wg1  = workgroup_size(1): 1 thread/matrix, shared memory");
    println!("  wg32 = workgroup_size(32): 32 threads cooperate on 1 matrix (barriers)");
    println!("  wp32 = workgroup_size(32): 32 threads, each solves OWN matrix (no barriers)");
    println!();

    for &(batch, dim, sweeps) in &[
        (32_u32, 20_u32, 5_u32),
        (128, 20, 5),
        (512, 20, 5),
        (32, 20, 200),
        (128, 20, 200),
        (512, 20, 200),
        (32, 30, 5),
        (128, 30, 5),
        (512, 30, 5),
        (32, 30, 200),
        (128, 30, 200),
        (512, 30, 200),
    ] {
        let matrices = gen_symmetric(batch as usize, dim as usize);
        let params = Params {
            n: dim,
            batch_size: batch,
            sweeps,
            _pad: 0,
        };

        let t_wg1 = bench_shader(
            gpu.device(),
            gpu.queue(),
            SHADER_WG1,
            &params,
            &matrices,
            batch,
            1,
        );
        let t_wg32 = bench_shader(
            gpu.device(),
            gpu.queue(),
            SHADER_WG32,
            &params,
            &matrices,
            batch,
            32,
        );
        let t_wp32 = bench_shader(
            gpu.device(),
            gpu.queue(),
            SHADER_WARP_PACKED,
            &params,
            &matrices,
            batch.div_ceil(32),
            32,
        );

        let s1 = t_wg1 / t_wp32;
        println!(
            "  b={batch:>4} d={dim:>2} s={sweeps:>3}: wg1={t_wg1:>8.3}ms  wg32={t_wg32:>8.3}ms  wp32={t_wp32:>8.3}ms  wp32/wg1={s1:.1}x"
        );
    }
}

fn bench_shader(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shader_src: &str,
    params: &Params,
    matrices: &[f64],
    batch: u32,
    wg_size: u32,
) -> f64 {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("diag"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let mat_bytes: Vec<u8> = matrices.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matrices"),
        contents: &mat_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pl),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: mat_buf.as_entire_binding(),
            },
        ],
    });

    let dispatch = |_round: usize| {
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(batch, 1, 1);
        }
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
    };

    let _ = wg_size;

    for i in 0..WARMUP {
        dispatch(i);
    }

    let t0 = Instant::now();
    for i in 0..ROUNDS {
        dispatch(i);
    }
    t0.elapsed().as_secs_f64() * 1e3 / ROUNDS as f64
}

fn gen_symmetric(batch: usize, dim: usize) -> Vec<f64> {
    let mut m = vec![0.0_f64; batch * dim * dim];
    for b in 0..batch {
        let base = b * dim * dim;
        for i in 0..dim {
            m[base + i * dim + i] = (i as f64 + 1.0) * 10.0 + b as f64 * 0.1;
            for j in (i + 1)..dim {
                let v = ((i + j) as f64 * 0.3 + b as f64 * 0.01).sin() * 0.5;
                m[base + i * dim + j] = v;
                m[base + j * dim + i] = v;
            }
        }
    }
    m
}
