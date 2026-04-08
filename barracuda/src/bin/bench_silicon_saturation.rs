// SPDX-License-Identifier: AGPL-3.0-or-later

//! Silicon saturation micro-experiments: find actual peak of each silicon unit.
//!
//! Six targeted experiments, each designed to saturate exactly one silicon unit
//! and measure achieved throughput vs theoretical peak. The ratio gives the
//! "efficiency floor" — any QCD kernel scoring below this ratio has headroom.
//!
//! ## Experiments
//!
//! 1. **Pure FMA chain** — shader ALU peak (FP32 and DF64)
//! 2. **Bandwidth sweep** — memory controller peak (sequential, strided)
//! 3. **Cache hierarchy** — L2/Infinity Cache boundary detection
//! 4. **TMU saturation** — texture unit peak (textureLoad throughput)
//! 5. **Workgroup reduce** — shared memory / LDS bandwidth
//! 6. **Atomic contention** — global atomicAdd throughput

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::toadstool_report::{self, PerformanceMeasurement};

use std::time::Instant;

// ── Experiment 1: Pure FMA chain (shader ALU saturation) ─────────────────────

const SHADER_FMA_CHAIN_FP32: &str = r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var a: f32 = f32(gid.x + 1u) * 1.0001;
    var b: f32 = 0.9999;
    for (var i = 0u; i < 512u; i = i + 1u) {
        a = fma(a, b, a);
        a = fma(a, b, a);
        a = fma(a, b, a);
        a = fma(a, b, a);
    }
    out[gid.x] = a;
}
";

const SHADER_FMA_CHAIN_DF64: &str = r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var hi: f32 = f32(gid.x + 1u) * 1.0001;
    var lo: f32 = 0.0;
    let b_hi: f32 = 0.9999;
    let b_lo: f32 = 0.00001;
    for (var i = 0u; i < 256u; i = i + 1u) {
        // Dekker two_prod: p = hi*b_hi, e = fma(hi, b_hi, -p)
        let p = hi * b_hi;
        let e = fma(hi, b_hi, -p);
        // Accumulate: hi = p, lo = e + lo*b_hi + hi*b_lo
        lo = fma(lo, b_hi, fma(hi, b_lo, e));
        hi = p;
        // two_sum for renormalization
        let s = hi + lo;
        lo = lo - (s - hi);
        hi = s;
    }
    out[gid.x * 2u] = hi;
    out[gid.x * 2u + 1u] = lo;
}
";

// ── Experiment 2: Bandwidth sweep (memory controller saturation) ─────────────

const SHADER_BANDWIDTH_SEQ: &str = r"
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let v = input[idx];
    out[idx] = v.x + v.y + v.z + v.w;
}
";

// ── Experiment 3: Cache hierarchy (L2 / Infinity Cache boundary) ─────────────

const SHADER_CACHE_PROBE: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.x;
    var sum: f32 = 0.0;
    var idx = gid.x;
    for (var i = 0u; i < 64u; i = i + 1u) {
        sum = sum + input[idx % size];
        idx = idx + 256u;
    }
    out[gid.x] = sum;
}
";

// ── Experiment 4: TMU saturation (texture unit throughput) ───────────────────

const SHADER_TMU_FLOOD: &str = r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var tex: texture_2d<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var sum: f32 = 0.0;
    let base_y = gid.x / 1024u;
    let base_x = gid.x % 1024u;
    for (var i = 0u; i < 64u; i = i + 1u) {
        let coord = vec2<i32>(i32((base_x + i * 16u) % 1024u), i32(base_y % 1024u));
        sum = sum + textureLoad(tex, coord, 0).r;
    }
    out[gid.x] = sum;
}
";

// ── Experiment 5: Workgroup reduce (shared memory / LDS throughput) ──────────

const SHADER_REDUCE_SWEEP: &str = r"
var<workgroup> wg_data: array<f32, 1024>;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1024)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    wg_data[lid.x] = f32(gid.x + 1u) * 0.001;
    workgroupBarrier();
    // Full tree reduction
    for (var s = 512u; s > 0u; s = s >> 1u) {
        if lid.x < s {
            wg_data[lid.x] = wg_data[lid.x] + wg_data[lid.x + s];
        }
        workgroupBarrier();
    }
    if lid.x == 0u { out[gid.x / 1024u] = wg_data[0]; }
}
";

// ── Experiment 6: Atomic contention (global atomicAdd throughput) ─────────────

const SHADER_ATOMIC_FLOOD: &str = r"
@group(0) @binding(0) var<storage, read_write> counters: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bucket = gid.x % 256u;
    for (var i = 0u; i < 64u; i = i + 1u) {
        atomicAdd(&counters[(bucket + i) % 256u], 1u);
    }
}
";

// ── Dispatch helpers ─────────────────────────────────────────────────────────

fn create_compute_pipeline(
    device: &wgpu::Device,
    wgsl: &str,
    bgl: &wgpu::BindGroupLayout,
    label: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn timed_dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    workgroups: u32,
    iterations: u32,
) -> std::time::Duration {
    // Warmup
    {
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(bg), &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(bg), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(Some(enc.finish()));
    }
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    start.elapsed()
}

// ── Experiment runners ───────────────────────────────────────────────────────

fn run_fma_experiment(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 1: Pure FMA Chain (shader ALU saturation) ──\n");

    let n_threads: u32 = 262_144; // 256K threads
    let iterations: u32 = 200;
    let wg_size: u32 = 256;
    let workgroups = n_threads / wg_size;

    // FP32: 512 iterations × 4 FMA × 2 FLOP = 4096 FLOP per thread
    let fp32_flops_per_thread: u64 = 512 * 4 * 2;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n_threads as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pipeline_fp32 = create_compute_pipeline(device, SHADER_FMA_CHAIN_FP32, &bgl, "fma_fp32");
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });

    let elapsed_fp32 = timed_dispatch(device, queue, &pipeline_fp32, &bg, workgroups, iterations);
    let total_fp32_flops = fp32_flops_per_thread * n_threads as u64 * iterations as u64;
    let fp32_tflops = total_fp32_flops as f64 / elapsed_fp32.as_secs_f64() / 1e12;

    println!(
        "  FP32 FMA chain:  {:.2} TFLOPS  ({:.1} ms, {} GFLOP)",
        fp32_tflops,
        elapsed_fp32.as_secs_f64() * 1000.0,
        total_fp32_flops / 1_000_000_000
    );

    measurements.push(PerformanceMeasurement {
        operation: "saturation.shader_core.fp32_fma".into(),
        silicon_unit: "shader_core".into(),
        precision_mode: "fp32".into(),
        throughput_gflops: fp32_tflops * 1000.0,
        tolerance_achieved: 0.0,
        gpu_model: gpu_name.into(),
        measured_by: "hotSpring/bench_silicon_saturation".into(),
        timestamp: ts,
    });

    // DF64: 256 iterations × ~10 FMA-equivalent ops (two_prod + two_sum + accumulate)
    let df64_flops_per_thread: u64 = 256 * 10 * 2;

    let out_buf_df64 = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n_threads as u64) * 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bg_df64 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf_df64.as_entire_binding(),
        }],
    });

    let pipeline_df64 = create_compute_pipeline(device, SHADER_FMA_CHAIN_DF64, &bgl, "fma_df64");
    let elapsed_df64 = timed_dispatch(
        device,
        queue,
        &pipeline_df64,
        &bg_df64,
        workgroups,
        iterations,
    );
    let total_df64_flops = df64_flops_per_thread * n_threads as u64 * iterations as u64;
    let df64_tflops = total_df64_flops as f64 / elapsed_df64.as_secs_f64() / 1e12;

    println!(
        "  DF64 Dekker chain: {:.2} TFLOPS  ({:.1} ms, {} GFLOP)",
        df64_tflops,
        elapsed_df64.as_secs_f64() * 1000.0,
        total_df64_flops / 1_000_000_000
    );

    measurements.push(PerformanceMeasurement {
        operation: "saturation.shader_core.df64_fma".into(),
        silicon_unit: "shader_core".into(),
        precision_mode: "df64".into(),
        throughput_gflops: df64_tflops * 1000.0,
        tolerance_achieved: 0.0,
        gpu_model: gpu_name.into(),
        measured_by: "hotSpring/bench_silicon_saturation".into(),
        timestamp: ts,
    });

    println!();
}

fn run_bandwidth_experiment(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 2: Bandwidth Sweep (memory controller saturation) ──\n");

    let iterations: u32 = 100;
    let sizes_mb: &[u64] = &[16, 64, 256, 512];

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
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
        ],
    });

    let pipeline = create_compute_pipeline(device, SHADER_BANDWIDTH_SEQ, &bgl, "bw_seq");

    println!("  {:<12} {:>10} {:>10}", "Size (MB)", "GB/s", "Efficiency");
    println!("  {}", "─".repeat(36));

    for size_mb in sizes_mb {
        let bytes = size_mb * 1024 * 1024;
        let n_vec4 = bytes / 16; // each vec4<f32> is 16 bytes
        let n_threads = n_vec4 as u32;
        let workgroups = n_threads.div_ceil(256);

        let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: n_threads as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let elapsed = timed_dispatch(device, queue, &pipeline, &bg, workgroups, iterations);
        let total_bytes_read = bytes * iterations as u64;
        let gbs = total_bytes_read as f64 / elapsed.as_secs_f64() / 1e9;

        println!("  {:<12} {:>8.1} {:>10}", size_mb, gbs, "—");

        measurements.push(PerformanceMeasurement {
            operation: format!("saturation.memory.bw_seq_{size_mb}mb"),
            silicon_unit: "memory".into(),
            precision_mode: "fp32_vec4".into(),
            throughput_gflops: gbs, // GB/s (not GFLOPS, but same field for simplicity)
            tolerance_achieved: 0.0,
            gpu_model: gpu_name.into(),
            measured_by: "hotSpring/bench_silicon_saturation".into(),
            timestamp: ts,
        });
    }
    println!();
}

fn run_cache_experiment(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 3: Cache Hierarchy (L2 / Infinity Cache boundary) ──\n");

    let iterations: u32 = 100;
    let n_threads: u32 = 65_536;
    let workgroups = n_threads / 256;
    // Sweep working-set sizes from 64 KB to 256 MB
    let sizes_kb: &[(u64, &str)] = &[
        (64, "64 KB"),
        (256, "256 KB"),
        (1024, "1 MB"),
        (4096, "4 MB"),
        (8192, "8 MB"),
        (32768, "32 MB"),
        (65536, "64 MB"),
        (131072, "128 MB"),
        (262144, "256 MB"),
    ];

    // Allocate the largest buffer once, then use params to limit the working set
    let max_bytes = 256 * 1024 * 1024u64;
    let max_elems = max_bytes / 4;

    let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: max_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n_threads as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
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

    let pipeline = create_compute_pipeline(device, SHADER_CACHE_PROBE, &bgl, "cache_probe");

    println!("  {:<12} {:>10} {:>10}", "Working set", "GB/s", "Note");
    println!("  {}", "─".repeat(36));

    let mut prev_gbs = 0.0f64;

    for (size_kb, label) in sizes_kb {
        let size_bytes = size_kb * 1024;
        let size_elems = (size_bytes / 4).min(max_elems) as u32;
        let params_data = [size_elems, 0u32, 0u32, 0u32];

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let elapsed = timed_dispatch(device, queue, &pipeline, &bg, workgroups, iterations);
        // Each thread reads 64 f32 values = 256 bytes
        let total_bytes_read = 256u64 * n_threads as u64 * iterations as u64;
        let gbs = total_bytes_read as f64 / elapsed.as_secs_f64() / 1e9;

        let note = if prev_gbs > 0.0 && gbs < prev_gbs * 0.7 {
            "← CACHE BOUNDARY"
        } else {
            ""
        };

        println!("  {label:<12} {gbs:>8.1} {note:>10}");
        prev_gbs = gbs;

        measurements.push(PerformanceMeasurement {
            operation: format!("saturation.cache.{size_kb}kb"),
            silicon_unit: "memory".into(),
            precision_mode: "cache_probe".into(),
            throughput_gflops: gbs,
            tolerance_achieved: 0.0,
            gpu_model: gpu_name.into(),
            measured_by: "hotSpring/bench_silicon_saturation".into(),
            timestamp: ts,
        });
    }
    println!();
}

fn run_tmu_experiment(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 4: TMU Saturation (texture unit throughput) ──\n");

    let n_threads: u32 = 262_144;
    let iterations: u32 = 200;
    let workgroups = n_threads / 256;
    let tex_dim: u32 = 1024;

    let tex_data: Vec<f32> = (0..tex_dim * tex_dim)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let tex_bytes: Vec<u8> = tex_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tmu_flood"),
        size: wgpu::Extent3d {
            width: tex_dim,
            height: tex_dim,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &tex_bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(tex_dim * 4),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: tex_dim,
            height: tex_dim,
            depth_or_array_layers: 1,
        },
    );
    let tex_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n_threads as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
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
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let pipeline = create_compute_pipeline(device, SHADER_TMU_FLOOD, &bgl, "tmu_flood");

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&tex_view),
            },
        ],
    });

    let elapsed = timed_dispatch(device, queue, &pipeline, &bg, workgroups, iterations);
    // Each thread does 64 textureLoads
    let total_texels = 64u64 * n_threads as u64 * iterations as u64;
    let gtexels = total_texels as f64 / elapsed.as_secs_f64() / 1e9;

    println!(
        "  TMU textureLoad:  {:.1} GT/s  ({:.1} ms, {}M texels)",
        gtexels,
        elapsed.as_secs_f64() * 1000.0,
        total_texels / 1_000_000
    );

    measurements.push(PerformanceMeasurement {
        operation: "saturation.tmu.textureload".into(),
        silicon_unit: "texture_unit".into(),
        precision_mode: "r32float".into(),
        throughput_gflops: gtexels,
        tolerance_achieved: 0.0,
        gpu_model: gpu_name.into(),
        measured_by: "hotSpring/bench_silicon_saturation".into(),
        timestamp: ts,
    });
    println!();
}

fn run_reduce_experiment(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 5: Workgroup Reduce (shared memory / LDS) ──\n");

    let n_threads: u32 = 262_144;
    let iterations: u32 = 200;
    let wg_size: u32 = 1024;
    let workgroups = n_threads / wg_size;

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (workgroups as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pipeline = create_compute_pipeline(device, SHADER_REDUCE_SWEEP, &bgl, "reduce_sweep");
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });

    let elapsed = timed_dispatch(device, queue, &pipeline, &bg, workgroups, iterations);
    // Tree reduction: log2(1024) = 10 steps, each step 1 add per active thread
    // Total adds ≈ 1024 per workgroup (geometric sum)
    let total_adds = 1023u64 * workgroups as u64 * iterations as u64;
    let gops = total_adds as f64 / elapsed.as_secs_f64() / 1e9;
    // Shared memory traffic: each step reads + writes, 4 bytes each
    let total_lds_bytes = 1023u64 * 2 * 4 * workgroups as u64 * iterations as u64;
    let lds_gbs = total_lds_bytes as f64 / elapsed.as_secs_f64() / 1e9;

    println!(
        "  Workgroup reduce:  {:.1} Gop/s  LDS: {:.1} GB/s  ({:.1} ms)",
        gops,
        lds_gbs,
        elapsed.as_secs_f64() * 1000.0,
    );

    measurements.push(PerformanceMeasurement {
        operation: "saturation.shared_mem.reduce".into(),
        silicon_unit: "shader_core".into(),
        precision_mode: "fp32_lds".into(),
        throughput_gflops: gops,
        tolerance_achieved: 0.0,
        gpu_model: gpu_name.into(),
        measured_by: "hotSpring/bench_silicon_saturation".into(),
        timestamp: ts,
    });
    println!();
}

fn run_atomic_experiment(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 6: Atomic Contention (global atomicAdd) ──\n");

    let n_threads: u32 = 262_144;
    let iterations: u32 = 200;
    let workgroups = n_threads / 256;

    let counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 4, // 256 atomic<u32> counters
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pipeline = create_compute_pipeline(device, SHADER_ATOMIC_FLOOD, &bgl, "atomic_flood");
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: counter_buf.as_entire_binding(),
        }],
    });

    let elapsed = timed_dispatch(device, queue, &pipeline, &bg, workgroups, iterations);
    // Each thread does 64 atomicAdds
    let total_atomics = 64u64 * n_threads as u64 * iterations as u64;
    let gatom = total_atomics as f64 / elapsed.as_secs_f64() / 1e9;

    println!(
        "  Atomic throughput: {:.1} Gatom/s  ({:.1} ms, {}M ops)",
        gatom,
        elapsed.as_secs_f64() * 1000.0,
        total_atomics / 1_000_000
    );

    measurements.push(PerformanceMeasurement {
        operation: "saturation.atomic.global_add".into(),
        silicon_unit: "shader_core".into(),
        precision_mode: "u32_atomic".into(),
        throughput_gflops: gatom,
        tolerance_achieved: 0.0,
        gpu_model: gpu_name.into(),
        measured_by: "hotSpring/bench_silicon_saturation".into(),
        timestamp: ts,
    });
    println!();
}

// ── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Silicon Saturation Micro-Experiments");
    println!("  Find actual peak of each silicon unit");
    println!("═══════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    if adapters.is_empty() {
        eprintln!("No GPU adapters found.");
        std::process::exit(1);
    }

    let mut measurements: Vec<PerformanceMeasurement> = Vec::new();
    let ts = toadstool_report::epoch_now();

    for adapter in adapters {
        let info = adapter.get_info();
        let gpu_name = info.name.clone();

        let gpu = match GpuF64::from_adapter(adapter).await {
            Ok(g) => g,
            Err(e) => {
                println!("  Skip {gpu_name}: {e}\n");
                continue;
            }
        };

        let device = gpu.device();
        let queue = gpu.queue();

        println!("━━━ {} ━━━\n", gpu.adapter_name);

        run_fma_experiment(device, queue, &gpu.adapter_name, &mut measurements, ts);
        run_bandwidth_experiment(device, queue, &gpu.adapter_name, &mut measurements, ts);
        run_cache_experiment(device, queue, &gpu.adapter_name, &mut measurements, ts);
        run_tmu_experiment(device, queue, &gpu.adapter_name, &mut measurements, ts);
        run_reduce_experiment(device, queue, &gpu.adapter_name, &mut measurements, ts);
        run_atomic_experiment(device, queue, &gpu.adapter_name, &mut measurements, ts);
    }

    println!(
        "── Reporting {} measurements to toadStool ──\n",
        measurements.len()
    );
    toadstool_report::report_to_toadstool(&measurements);

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Silicon Saturation Complete");
    println!("═══════════════════════════════════════════════════════════");
}
