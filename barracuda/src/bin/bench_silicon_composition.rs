// SPDX-License-Identifier: AGPL-3.0-only

//! Silicon composition experiments: measure compound throughput when multiple
//! GPU silicon units operate on different sub-problems simultaneously.
//!
//! The theoretical "compound effect" (50-100 TFLOPS equivalent) requires
//! different hardware blocks working in parallel. This binary quantifies
//! the actual multiplier for QCD-relevant workload combinations.
//!
//! ## Experiments
//!
//! 1. **ALU + TMU**: shader core compute + texture unit lookup in same dispatch
//! 2. **ALU + Bandwidth**: compute-heavy + memory-heavy kernels interleaved
//! 3. **ALU + Reduce**: matrix compute + scalar accumulation (CG pattern)
//!
//! ## Method
//!
//! For each combination: run sub-problems separately, then together. Compare
//! combined throughput vs each alone. The ratio is the "composition multiplier."

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::toadstool_report::{self, PerformanceMeasurement};

use std::time::Instant;

// ── Component shaders ────────────────────────────────────────────────────────

const SHADER_ALU_ONLY: &str = r"
// Pure ALU: SU(3) matmul chain (no memory access beyond output)
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var a: f32 = f32(gid.x + 1u) * 0.001;
    for (var i = 0u; i < 256u; i = i + 1u) {
        a = fma(a, 0.9999, a * 0.0001);
    }
    out[gid.x] = a;
}
";

const SHADER_TMU_ONLY: &str = r"
// Pure TMU: maximum texture fetch rate
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var tex: texture_2d<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var sum: f32 = 0.0;
    for (var i = 0u; i < 32u; i = i + 1u) {
        let coord = vec2<i32>(i32((gid.x + i * 7u) % 1024u), i32((gid.x / 1024u + i) % 1024u));
        sum = sum + textureLoad(tex, coord, 0).r;
    }
    out[gid.x] = sum;
}
";

const SHADER_ALU_PLUS_TMU: &str = r"
// ALU + TMU combined: compute on shader cores while texture unit fetches tables
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var tex: texture_2d<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var a: f32 = f32(gid.x + 1u) * 0.001;
    var sum: f32 = 0.0;
    for (var i = 0u; i < 32u; i = i + 1u) {
        // ALU work: FMA chain
        a = fma(a, 0.9999, a * 0.0001);
        a = fma(a, 0.9999, a * 0.0001);
        a = fma(a, 0.9999, a * 0.0001);
        a = fma(a, 0.9999, a * 0.0001);
        a = fma(a, 0.9999, a * 0.0001);
        a = fma(a, 0.9999, a * 0.0001);
        a = fma(a, 0.9999, a * 0.0001);
        a = fma(a, 0.9999, a * 0.0001);
        // TMU work: texture fetch (overlaps with ALU)
        let coord = vec2<i32>(i32((gid.x + i * 7u) % 1024u), i32((gid.x / 1024u + i) % 1024u));
        sum = sum + textureLoad(tex, coord, 0).r;
    }
    out[gid.x] = a + sum;
}
";

const SHADER_BW_ONLY: &str = r"
// Pure bandwidth: sequential read through large buffer
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = input[gid.x];
    out[gid.x] = v.x + v.y + v.z + v.w;
}
";

const SHADER_REDUCE_ONLY: &str = r"
// Pure reduce: workgroup shared memory tree reduction
var<workgroup> wg_data: array<f32, 256>;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    wg_data[lid.x] = f32(gid.x + 1u) * 0.001;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if lid.x < s { wg_data[lid.x] = wg_data[lid.x] + wg_data[lid.x + s]; }
        workgroupBarrier();
    }
    if lid.x == 0u { out[gid.x / 256u] = wg_data[0]; }
}
";

// ── Dispatch helpers ─────────────────────────────────────────────────────────

fn timed_dispatch_single_bgl(
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

fn make_storage_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    })
}

fn make_storage_tex_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    })
}

fn make_pipeline(
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

fn create_texture_1024(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let tex_data: Vec<f32> = (0..1024 * 1024).map(|i| (i as f32 * 0.001).sin()).collect();
    let tex_bytes: Vec<u8> = tex_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("composition_tex"),
        size: wgpu::Extent3d {
            width: 1024,
            height: 1024,
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
            bytes_per_row: Some(1024 * 4),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: 1024,
            height: 1024,
            depth_or_array_layers: 1,
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

// ── Experiment runners ───────────────────────────────────────────────────────

fn run_alu_tmu_composition(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 1: ALU + TMU Composition ──\n");
    println!("  QCD analog: SU(3) force compute + EOS table lookup\n");

    let n_threads: u32 = 262_144;
    let iterations: u32 = 200;
    let workgroups = n_threads / 256;

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n_threads as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let tex_view = create_texture_1024(device, queue);

    // ALU only
    let bgl_storage = make_storage_bgl(device);
    let pipeline_alu = make_pipeline(device, SHADER_ALU_ONLY, &bgl_storage, "alu_only");
    let bg_alu = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_storage,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });
    let elapsed_alu = timed_dispatch_single_bgl(
        device,
        queue,
        &pipeline_alu,
        &bg_alu,
        workgroups,
        iterations,
    );
    let alu_ops = 256u64 * 2 * n_threads as u64 * iterations as u64; // 256 iters × 2 FLOP per FMA
    let alu_tflops = alu_ops as f64 / elapsed_alu.as_secs_f64() / 1e12;

    // TMU only
    let bgl_tex = make_storage_tex_bgl(device);
    let pipeline_tmu = make_pipeline(device, SHADER_TMU_ONLY, &bgl_tex, "tmu_only");
    let bg_tmu = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_tex,
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
    let elapsed_tmu = timed_dispatch_single_bgl(
        device,
        queue,
        &pipeline_tmu,
        &bg_tmu,
        workgroups,
        iterations,
    );
    let tmu_fetches = 32u64 * n_threads as u64 * iterations as u64;
    let tmu_gtexels = tmu_fetches as f64 / elapsed_tmu.as_secs_f64() / 1e9;

    // Combined
    let pipeline_combined = make_pipeline(device, SHADER_ALU_PLUS_TMU, &bgl_tex, "alu_tmu");
    let bg_combined = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_tex,
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
    let elapsed_combined = timed_dispatch_single_bgl(
        device,
        queue,
        &pipeline_combined,
        &bg_combined,
        workgroups,
        iterations,
    );
    let combined_alu_ops = 32u64 * 8 * 2 * n_threads as u64 * iterations as u64;
    let combined_tmu_fetches = 32u64 * n_threads as u64 * iterations as u64;
    let combined_alu_tflops = combined_alu_ops as f64 / elapsed_combined.as_secs_f64() / 1e12;
    let combined_tmu_gtexels = combined_tmu_fetches as f64 / elapsed_combined.as_secs_f64() / 1e9;

    // The composition multiplier: is combined work > max(alu_alone, tmu_alone)?
    let alu_alone_time = elapsed_alu.as_secs_f64();
    let tmu_alone_time = elapsed_tmu.as_secs_f64();
    let combined_time = elapsed_combined.as_secs_f64();
    let sequential_time = alu_alone_time + tmu_alone_time;
    let composition_ratio = sequential_time / combined_time;

    println!(
        "  ALU only:     {:.2} TFLOPS  ({:.1} ms)",
        alu_tflops,
        alu_alone_time * 1000.0
    );
    println!(
        "  TMU only:     {:.1} GT/s  ({:.1} ms)",
        tmu_gtexels,
        tmu_alone_time * 1000.0
    );
    println!(
        "  Combined:     {:.2} TFLOPS + {:.1} GT/s  ({:.1} ms)",
        combined_alu_tflops,
        combined_tmu_gtexels,
        combined_time * 1000.0
    );
    println!(
        "  Composition:  {composition_ratio:.2}x (1.0 = fully sequential, 2.0 = fully parallel)"
    );
    println!();

    measurements.push(PerformanceMeasurement {
        operation: "composition.alu_tmu.multiplier".into(),
        silicon_unit: "compound".into(),
        precision_mode: "fp32".into(),
        throughput_gflops: composition_ratio * 1000.0,
        tolerance_achieved: 0.0,
        gpu_model: gpu_name.into(),
        measured_by: "hotSpring/bench_silicon_composition".into(),
        timestamp: ts,
    });
}

fn run_alu_bw_composition(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 2: ALU + Bandwidth Composition ──\n");
    println!("  QCD analog: force compute (ALU) overlapping with momentum update (BW)\n");

    let n_threads: u32 = 262_144;
    let iterations: u32 = 200;
    let workgroups = n_threads / 256;

    // ALU only
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n_threads as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bgl = make_storage_bgl(device);
    let pipeline_alu = make_pipeline(device, SHADER_ALU_ONLY, &bgl, "comp_alu");
    let bg_alu = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });
    let elapsed_alu = timed_dispatch_single_bgl(
        device,
        queue,
        &pipeline_alu,
        &bg_alu,
        workgroups,
        iterations,
    );

    // BW only
    let bgl_bw = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n_threads as u64) * 16, // vec4<f32> per thread
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let pipeline_bw = make_pipeline(device, SHADER_BW_ONLY, &bgl_bw, "comp_bw");
    let bg_bw = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_bw,
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
    let elapsed_bw =
        timed_dispatch_single_bgl(device, queue, &pipeline_bw, &bg_bw, workgroups, iterations);

    // Interleaved: ALU then BW in same encoder (tests pipeline switching overhead)
    let start = Instant::now();
    for _ in 0..iterations {
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline_alu);
            pass.set_bind_group(0, Some(&bg_alu), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
            pass.set_pipeline(&pipeline_bw);
            pass.set_bind_group(0, Some(&bg_bw), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(Some(enc.finish()));
    }
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let elapsed_interleaved = start.elapsed();

    let alu_time = elapsed_alu.as_secs_f64();
    let bw_time = elapsed_bw.as_secs_f64();
    let interleaved_time = elapsed_interleaved.as_secs_f64();
    let sequential_time = alu_time + bw_time;
    let overlap_ratio = sequential_time / interleaved_time;

    println!("  ALU only:       {:.1} ms", alu_time * 1000.0);
    println!("  BW only:        {:.1} ms", bw_time * 1000.0);
    println!(
        "  Interleaved:    {:.1} ms (both in same encoder)",
        interleaved_time * 1000.0
    );
    println!("  Overlap ratio:  {overlap_ratio:.2}x (1.0 = no overlap, 2.0 = full overlap)");
    println!();

    measurements.push(PerformanceMeasurement {
        operation: "composition.alu_bw.multiplier".into(),
        silicon_unit: "compound".into(),
        precision_mode: "fp32".into(),
        throughput_gflops: overlap_ratio * 1000.0,
        tolerance_achieved: 0.0,
        gpu_model: gpu_name.into(),
        measured_by: "hotSpring/bench_silicon_composition".into(),
        timestamp: ts,
    });
}

fn run_alu_reduce_composition(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 3: ALU + Reduce Composition ──\n");
    println!("  QCD analog: D†Dx compute (ALU) + dot product (reduce) per CG iteration\n");

    let n_threads: u32 = 262_144;
    let iterations: u32 = 200;
    let wg_alu = n_threads / 256;
    let wg_reduce = n_threads / 256;

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n_threads as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bgl = make_storage_bgl(device);
    let pipeline_alu = make_pipeline(device, SHADER_ALU_ONLY, &bgl, "cg_alu");
    let bg_alu = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });

    let reduce_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (wg_reduce as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let pipeline_reduce = make_pipeline(device, SHADER_REDUCE_ONLY, &bgl, "cg_reduce");
    let bg_reduce = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: reduce_out.as_entire_binding(),
        }],
    });

    // ALU only
    let elapsed_alu =
        timed_dispatch_single_bgl(device, queue, &pipeline_alu, &bg_alu, wg_alu, iterations);

    // Reduce only
    let elapsed_reduce = timed_dispatch_single_bgl(
        device,
        queue,
        &pipeline_reduce,
        &bg_reduce,
        wg_reduce,
        iterations,
    );

    // CG pattern: ALU dispatch → reduce dispatch (sequential dependency)
    let start = Instant::now();
    for _ in 0..iterations {
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            // Dirac apply (ALU-heavy)
            pass.set_pipeline(&pipeline_alu);
            pass.set_bind_group(0, Some(&bg_alu), &[]);
            pass.dispatch_workgroups(wg_alu, 1, 1);
            // Dot product (reduce)
            pass.set_pipeline(&pipeline_reduce);
            pass.set_bind_group(0, Some(&bg_reduce), &[]);
            pass.dispatch_workgroups(wg_reduce, 1, 1);
        }
        queue.submit(Some(enc.finish()));
    }
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let elapsed_cg = start.elapsed();

    let alu_time = elapsed_alu.as_secs_f64();
    let reduce_time = elapsed_reduce.as_secs_f64();
    let cg_time = elapsed_cg.as_secs_f64();
    let sequential_time = alu_time + reduce_time;
    let cg_overhead = cg_time / sequential_time;

    println!("  ALU (Dirac):    {:.1} ms", alu_time * 1000.0);
    println!("  Reduce (dot):   {:.1} ms", reduce_time * 1000.0);
    println!(
        "  CG pattern:     {:.1} ms (ALU → reduce per iter)",
        cg_time * 1000.0
    );
    println!(
        "  CG efficiency:  {:.0}% (100% = zero pipeline switch overhead)",
        100.0 / cg_overhead
    );
    println!();

    measurements.push(PerformanceMeasurement {
        operation: "composition.alu_reduce.cg_efficiency".into(),
        silicon_unit: "compound".into(),
        precision_mode: "fp32".into(),
        throughput_gflops: (100.0 / cg_overhead) * 10.0, // scaled for reporting
        tolerance_achieved: 0.0,
        gpu_model: gpu_name.into(),
        measured_by: "hotSpring/bench_silicon_composition".into(),
        timestamp: ts,
    });
}

// ── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Silicon Composition Experiments");
    println!("  Multi-unit parallel throughput measurement");
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

        run_alu_tmu_composition(device, queue, &gpu.adapter_name, &mut measurements, ts);
        run_alu_bw_composition(device, queue, &gpu.adapter_name, &mut measurements, ts);
        run_alu_reduce_composition(device, queue, &gpu.adapter_name, &mut measurements, ts);

        // Summary
        println!("  ── Composition Summary ──\n");
        println!("  The compound effect is limited by:");
        println!("  - Pipeline switching overhead (dispatch → dispatch latency)");
        println!("  - Shared resource contention (register file, L1 cache, memory bus)");
        println!("  - Warp scheduler: shader cores, TMUs, and memory units share warps");
        println!();
        println!("  Ideal for QCD: PRNG (TMU) while gauge force (ALU) computes");
        println!("  Ideal for CG: Dirac (ALU) → dot (reduce) with minimal switch cost");
        println!();
    }

    println!(
        "── Reporting {} measurements to toadStool ──\n",
        measurements.len()
    );
    toadstool_report::report_to_toadstool(&measurements);

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Silicon Composition Complete");
    println!("═══════════════════════════════════════════════════════════");
}
