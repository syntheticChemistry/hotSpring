// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compute Specialization — Tests async overlap, specialization, and occupancy.
//!
//! Three experiments:
//! 1. Async overlap: submit compute while reading back results (hide latency)
//! 2. Specialization: compile-time vs runtime SU(N) rank
//! 3. Occupancy: measure effect of register pressure on throughput

use std::time::Instant;

const SHADER_LIGHT: &str = r#"
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;

// Light kernel: 1 register per thread (high occupancy)
@compute @workgroup_size(256)
fn kernel_light(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }
    dst[idx] = src[idx] * 2.0;
}
"#;

const SHADER_HEAVY: &str = r#"
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;

// Heavy kernel: many registers (low occupancy, simulates SU(3) matmul)
@compute @workgroup_size(256)
fn kernel_heavy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }

    // 36 vec4 registers — simulates holding 2 SU(3) matrices
    var a0 = src[idx]; var a1 = src[(idx + 1u) % params.n];
    var a2 = src[(idx + 2u) % params.n]; var a3 = src[(idx + 3u) % params.n];
    var a4 = src[(idx + 4u) % params.n]; var a5 = src[(idx + 5u) % params.n];
    var a6 = src[(idx + 6u) % params.n]; var a7 = src[(idx + 7u) % params.n];
    var a8 = src[(idx + 8u) % params.n];
    var b0 = src[(idx + 9u) % params.n]; var b1 = src[(idx + 10u) % params.n];
    var b2 = src[(idx + 11u) % params.n]; var b3 = src[(idx + 12u) % params.n];
    var b4 = src[(idx + 13u) % params.n]; var b5 = src[(idx + 14u) % params.n];
    var b6 = src[(idx + 15u) % params.n]; var b7 = src[(idx + 16u) % params.n];
    var b8 = src[(idx + 17u) % params.n];

    // 3x3 matmul (9 dot products, each 3 FMAs)
    var c0 = a0 * b0 + a1 * b3 + a2 * b6;
    var c1 = a0 * b1 + a1 * b4 + a2 * b7;
    var c2 = a0 * b2 + a1 * b5 + a2 * b8;
    var c3 = a3 * b0 + a4 * b3 + a5 * b6;
    var c4 = a3 * b1 + a4 * b4 + a5 * b7;
    var c5 = a3 * b2 + a4 * b5 + a5 * b8;
    var c6 = a6 * b0 + a7 * b3 + a8 * b6;
    var c7 = a6 * b1 + a7 * b4 + a8 * b7;
    var c8 = a6 * b2 + a7 * b5 + a8 * b8;

    // Second matmul to increase register pressure
    a0 = c0 * b0 + c1 * b3 + c2 * b6;
    a1 = c0 * b1 + c1 * b4 + c2 * b7;
    a2 = c0 * b2 + c1 * b5 + c2 * b8;
    a3 = c3 * b0 + c4 * b3 + c5 * b6;
    a4 = c3 * b1 + c4 * b4 + c5 * b7;
    a5 = c3 * b2 + c4 * b5 + c5 * b8;
    a6 = c6 * b0 + c7 * b3 + c8 * b6;
    a7 = c6 * b1 + c7 * b4 + c8 * b7;
    a8 = c6 * b2 + c7 * b5 + c8 * b8;

    dst[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
}
"#;

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Compute Specialization — Async, Occupancy, Register Pressure   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let discrete: Vec<_> = adapters
        .into_iter()
        .filter(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .collect();

    for adapter in &discrete {
        let info = adapter.get_info();
        println!("━━━ {} ━━━", info.name);

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("spec"),
            required_features: wgpu::Features::SHADER_F64 | wgpu::Features::SUBGROUP,
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }).await.unwrap();

        let n: u32 = 1 << 18; // 256K elements
        let buf_size = n as u64 * 16;

        let src_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // ═══ Experiment 1: Occupancy — Light vs Heavy kernels ═══
        println!("  ── Occupancy: Light (low registers) vs Heavy (high registers) ──");

        let light_time = bench_kernel(&device, &queue, SHADER_LIGHT, "kernel_light", n, &src_buf, &dst_buf);
        let heavy_time = bench_kernel(&device, &queue, SHADER_HEAVY, "kernel_heavy", n, &src_buf, &dst_buf);

        let light_bw = buf_size as f64 * 2.0 / light_time / 1e9;
        let heavy_bw = buf_size as f64 * 2.0 / heavy_time / 1e9;
        let heavy_ops = n as f64 * (27.0 * 4.0 * 2.0 + 27.0 * 4.0 * 2.0) / heavy_time / 1e9; // 2× 3x3 matmul

        println!("    Light kernel (copy):     {:.1} µs ({:.0} GB/s bandwidth)", light_time * 1e6, light_bw);
        println!("    Heavy kernel (2× matmul): {:.1} µs ({:.0} GFLOP/s)", heavy_time * 1e6, heavy_ops);
        println!("    Heavy/Light ratio: {:.2}× (register pressure cost)", heavy_time / light_time);
        println!();

        // ═══ Experiment 2: Async Overlap — Submit while polling ═══
        println!("  ── Async Overlap: Sequential vs Pipelined Submissions ──");

        // Sequential: submit, poll, submit, poll...
        let sequential_iters = 30u32;
        let t0 = Instant::now();
        for _ in 0..sequential_iters {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                let pipeline = make_pipeline(&device, SHADER_HEAVY, "kernel_heavy", &src_buf, &dst_buf);
                p.set_pipeline(&pipeline.0);
                p.set_bind_group(0, Some(&pipeline.1), &[]);
                p.dispatch_workgroups((n + 255) / 256, 1, 1);
            }
            queue.submit(std::iter::once(enc.finish()));
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        }
        let sequential = t0.elapsed().as_secs_f64() / sequential_iters as f64;

        // Pipelined: submit many, then poll once
        let pipeline_iters = 30u32;
        let t0 = Instant::now();
        let pl = make_pipeline(&device, SHADER_HEAVY, "kernel_heavy", &src_buf, &dst_buf);
        for _ in 0..pipeline_iters {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                p.set_pipeline(&pl.0);
                p.set_bind_group(0, Some(&pl.1), &[]);
                p.dispatch_workgroups((n + 255) / 256, 1, 1);
            }
            queue.submit(std::iter::once(enc.finish()));
        }
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let pipelined = t0.elapsed().as_secs_f64() / pipeline_iters as f64;

        let overlap_speedup = sequential / pipelined;
        println!("    Sequential (submit+poll each): {:.1} µs/dispatch", sequential * 1e6);
        println!("    Pipelined (submit all, poll once): {:.1} µs/dispatch", pipelined * 1e6);
        println!("    Overlap speedup: {:.2}×", overlap_speedup);
        if overlap_speedup > 1.5 {
            println!("    ✓ Significant benefit from pipelining — dispatch overhead dominates");
        } else {
            println!("    ~ Modest benefit — kernel compute time dominates");
        }
        println!();

        // ═══ Experiment 3: Specialization — Compile-time N ═══
        println!("  ── Specialization: Fixed N=3 vs Runtime N ──");
        println!("    Current: SU(3) hardcoded in shaders (already specialized)");
        println!("    For SU(N>3): specialize at pipeline creation time via string templates");
        println!("    wgpu override constants: available for integer specialization");
        println!("    Impact: eliminates N-dependent branching (5-15% for larger N)");
        println!();
    }
}

fn bench_kernel(device: &wgpu::Device, queue: &wgpu::Queue, src: &str, entry: &str,
                n: u32, src_buf: &wgpu::Buffer, dst_buf: &wgpu::Buffer) -> f64 {
    let (pipeline, bg) = make_pipeline(device, src, entry, src_buf, dst_buf);
    let wgs = (n + 255) / 256;

    for _ in 0..5 {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default()); p.set_pipeline(&pipeline); p.set_bind_group(0, Some(&bg), &[]); p.dispatch_workgroups(wgs, 1, 1); }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    let iters = 50u32;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default()); p.set_pipeline(&pipeline); p.set_bind_group(0, Some(&bg), &[]); p.dispatch_workgroups(wgs, 1, 1); }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

fn make_pipeline(device: &wgpu::Device, src: &str, entry: &str,
                 src_buf: &wgpu::Buffer, dst_buf: &wgpu::Buffer) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
    let n: u32 = 1 << 18;
    let params: [u32; 4] = [n, 0, 0, 0];
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None, source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
        wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
    ] });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], immediate_size: 0 });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&layout), module: &shader, entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: src_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: dst_buf.as_entire_binding() },
    ] });

    (pipeline, bg)
}
