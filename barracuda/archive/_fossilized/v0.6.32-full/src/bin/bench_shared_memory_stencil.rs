// SPDX-License-Identifier: AGPL-3.0-or-later
//! Workgroup Shared Memory — The biggest untapped optimization.
//!
//! Compares: global memory stencil vs shared memory stencil.
//! Also tests workgroup sizes: 64, 128, 256, 512.
//!
//! Expected: 3-10× speedup from shared memory for stencil operations.
//! This is how QUDA gets its performance — we haven't touched it yet.

use std::time::Instant;

const SHADER: &str = include_str!("shaders/silicon_genealogy/shared_memory_stencil.wgsl");

// Separate shader for workgroup size tuning (global memory, varying WG size)
fn wgsize_shader(wg_size: u32) -> String {
    format!(r#"
struct Params {{
    volume: u32,
    tile_l: u32,
    lattice_l: u32,
    pad0: u32,
}}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>>;

fn global_idx(x: i32, y: i32, z: i32, t: i32, l: i32) -> u32 {{
    let xp = ((x % l) + l) % l;
    let yp = ((y % l) + l) % l;
    let zp = ((z % l) + l) % l;
    let tp = ((t % l) + l) % l;
    return u32(xp + l * (yp + l * (zp + l * tp)));
}}

@compute @workgroup_size({wg_size})
fn stencil_wg(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let site = gid.x;
    if site >= params.volume {{ return; }}

    let ll = params.lattice_l;
    let l = i32(ll);
    let x = i32(site % ll);
    let y = i32((site / ll) % ll);
    let z = i32((site / (ll * ll)) % ll);
    let t = i32(site / (ll * ll * ll));

    var acc = vec4<f32>(0.0);
    acc += field[global_idx(x - 1, y, z, t, l)];
    acc += field[global_idx(x + 1, y, z, t, l)];
    acc += field[global_idx(x, y - 1, z, t, l)];
    acc += field[global_idx(x, y + 1, z, t, l)];
    acc += field[global_idx(x, y, z - 1, t, l)];
    acc += field[global_idx(x, y, z + 1, t, l)];
    acc += field[global_idx(x, y, z, t - 1, l)];
    acc += field[global_idx(x, y, z, t + 1, l)];

    let center = field[site];
    result[site] = acc * 0.125 - center;
}}
"#, wg_size = wg_size)
}

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Workgroup Shared Memory + WG Size Tuning                       ║");
    println!("║  The BIGGEST untapped optimization for QCD stencils             ║");
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
        let limits = adapter.limits();
        println!("━━━ {} ━━━", info.name);
        println!("  max_compute_workgroup_size_x: {}", limits.max_compute_workgroup_size_x);
        println!("  max_compute_workgroups_per_dimension: {}", limits.max_compute_workgroups_per_dimension);
        println!("  max_compute_workgroup_storage_size: {} bytes", limits.max_compute_workgroup_storage_size);
        println!();

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("shared_mem"),
            required_features: wgpu::Features::SHADER_F64 | wgpu::Features::SUBGROUP,
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }).await.unwrap();

        // ═══ PART 1: Workgroup Size Tuning ═══
        println!("  ── Part 1: Workgroup Size Tuning (global memory, 16⁴ lattice) ──");
        let lattice_l: u32 = 16;
        let volume = lattice_l.pow(4);

        let field_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: volume as u64 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: volume as u64 * 16,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });

        let wg_sizes: &[u32] = &[32, 64, 128, 256, 512, 1024];
        println!("    {:>6}  {:>10}  {:>10}  {:>10}", "WG Size", "µs/stencil", "Gsite/s", "vs 256");

        let mut baseline_time = 0.0f64;

        for &wgs in wg_sizes {
            if wgs > limits.max_compute_workgroup_size_x { continue; }

            let src = wgsize_shader(wgs);
            let time = bench_stencil_raw(&device, &queue, &src, "stencil_wg",
                                         volume, lattice_l, wgs, &field_buf, &result_buf);
            let gsites = volume as f64 / time / 1e9;

            if wgs == 256 { baseline_time = time; }
            let vs_256 = if baseline_time > 0.0 { baseline_time / time } else { 1.0 };

            let marker = if wgs == 32 && info.name.contains("NVIDIA") { " ← warp" }
                else if wgs == 64 && info.name.contains("AMD") { " ← wavefront" }
                else { "" };

            println!("    {:>6}  {:>8.1}  {:>8.3}  {:>8.2}×{}",
                wgs, time * 1e6, gsites, vs_256, marker);
        }
        println!();

        // ═══ PART 2: Shared Memory vs Global ═══
        println!("  ── Part 2: Shared Memory vs Global (16⁴ lattice, WG=256) ──");

        // Global memory baseline
        let global_time = bench_stencil_raw(&device, &queue, SHADER, "stencil_global",
                                           volume, lattice_l, 256, &field_buf, &result_buf);

        // Shared memory (tile_l=4, workgroup = 4⁴ = 256 threads)
        let shared_time = bench_stencil_shared(&device, &queue, SHADER,
                                              volume, lattice_l, 4, &field_buf, &result_buf);

        let speedup = global_time / shared_time;
        println!("    Global memory:  {:.1} µs ({:.3} Gsite/s)", global_time * 1e6, volume as f64 / global_time / 1e9);
        println!("    Shared memory:  {:.1} µs ({:.3} Gsite/s)", shared_time * 1e6, volume as f64 / shared_time / 1e9);
        println!("    Speedup: {:.2}×", speedup);
        println!();

        if speedup > 1.5 {
            println!("    ✓ Shared memory gives {:.0}% speedup — SIGNIFICANT", (speedup - 1.0) * 100.0);
            println!("    → Implement for SU(3) staple computation (18 matmuls share neighbors)");
        } else if speedup > 1.0 {
            println!("    ~ Shared memory gives modest {:.0}% speedup", (speedup - 1.0) * 100.0);
            println!("    → Benefit depends on memory-bound vs compute-bound balance");
        } else {
            println!("    ✗ No benefit from shared memory for this stencil");
            println!("    → Likely already L2/IC-cached at this lattice size");
        }
        println!();
    }
}

fn bench_stencil_raw(device: &wgpu::Device, queue: &wgpu::Queue, src: &str, entry: &str,
                     volume: u32, lattice_l: u32, wg_size: u32,
                     field_buf: &wgpu::Buffer, result_buf: &wgpu::Buffer) -> f64 {
    let params: [u32; 4] = [volume, 4, lattice_l, 0];
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

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
        wgpu::BindGroupEntry { binding: 1, resource: field_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: result_buf.as_entire_binding() },
    ] });

    let wgs = (volume + wg_size - 1) / wg_size;

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

fn bench_stencil_shared(device: &wgpu::Device, queue: &wgpu::Queue, src: &str,
                        volume: u32, lattice_l: u32, tile_l: u32,
                        field_buf: &wgpu::Buffer, result_buf: &wgpu::Buffer) -> f64 {
    let params: [u32; 4] = [volume, tile_l, lattice_l, 0];
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

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
        label: None, layout: Some(&layout), module: &shader, entry_point: Some("stencil_shared"),
        compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: field_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: result_buf.as_entire_binding() },
    ] });

    // For shared memory: one workgroup per tile (tile_l⁴ threads)
    let tiles_per_dim = lattice_l / tile_l;
    let total_tiles = tiles_per_dim.pow(4);

    for _ in 0..5 {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default()); p.set_pipeline(&pipeline); p.set_bind_group(0, Some(&bg), &[]); p.dispatch_workgroups(total_tiles, 1, 1); }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    let iters = 50u32;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default()); p.set_pipeline(&pipeline); p.set_bind_group(0, Some(&bg), &[]); p.dispatch_workgroups(total_tiles, 1, 1); }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    t0.elapsed().as_secs_f64() / iters as f64
}
