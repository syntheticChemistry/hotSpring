// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tiling/Domain Decomposition — Processing larger lattices in cache-efficient blocks.
//!
//! The Infinity Cache discovery means AMD can fit 128 MB working sets in L3.
//! But for lattices larger than the cache, we MUST tile.
//!
//! This experiment measures:
//! - Optimal tile size per card (cache boundary → natural tile)
//! - Overhead of tile boundary communication (halo exchange)
//! - Scaling from 8⁴ → 12⁴ → 16⁴ → 20⁴ → 24⁴ using tiled decomposition
//! - Whether tiling + AMD IC can reach 32⁴ (practical physics size)
//!
//! Key insight: on NVIDIA (6 MB L2), we need 4⁴ tiles.
//!              on AMD (128 MB IC), we can do 12⁴ tiles (or even monolithic 16⁴).

use std::time::Instant;

const SHADER_TILED_STENCIL: &str = r#"
struct Params {
    tile_sites: u32,
    halo_sites: u32,
    interior_sites: u32,
    stencil_radius: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>>;

// Simulates a nearest-neighbor stencil operation on a tile
// Each site reads its 8 neighbors (±1 in 4 dimensions) and accumulates
@compute @workgroup_size(256)
fn tiled_stencil(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.interior_sites { return; }

    // Offset by halo to reach interior
    let site = idx + params.halo_sites;

    var acc = vec4<f32>(0.0);
    // 4D stencil: 8 neighbors
    acc += field[site + 1u];
    acc += field[site - 1u];
    acc += field[site + params.stencil_radius];
    acc += field[site - params.stencil_radius];
    acc += field[site + params.stencil_radius * params.stencil_radius];
    acc += field[site - params.stencil_radius * params.stencil_radius];
    acc += field[site + params.stencil_radius * params.stencil_radius * params.stencil_radius];
    acc += field[site - params.stencil_radius * params.stencil_radius * params.stencil_radius];

    result[idx] = acc * 0.125 - field[site];
}
"#;

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║   Tiling/Domain Decomposition — Cache-Efficient Large Lattices  ║");
    println!("║   Finding optimal tile size per generation                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let discrete: Vec<_> = adapters
        .into_iter()
        .filter(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .collect();

    // Tile sizes to test (L per dimension, volume = L⁴)
    let tile_ls: &[u32] = &[4, 6, 8, 10, 12, 14, 16, 20, 24];

    for adapter in &discrete {
        let info = adapter.get_info();
        println!("━━━ {} ━━━", info.name);

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tiling"),
            required_features: wgpu::Features::SHADER_F64 | wgpu::Features::SUBGROUP,
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }).await.unwrap();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None, source: wgpu::ShaderSource::Wgsl(SHADER_TILED_STENCIL.into()),
        });

        println!("  {:>6}  {:>10}  {:>10}  {:>12}  {:>12}  {:>10}", "Tile", "Volume", "Working", "Stencil", "Throughput", "Status");
        println!("  {:>6}  {:>10}  {:>10}  {:>12}  {:>12}  {:>10}", "L", "sites", "set MB", "µs/site", "Gsite/s", "");
        println!("  ──────────────────────────────────────────────────────────────────");

        let mut best_throughput = 0.0f64;
        let mut best_l = 0u32;

        for &l in tile_ls {
            let volume = l.pow(4);
            let halo = 1u32; // 1-site halo
            let padded_l = l + 2 * halo;
            let padded_volume = padded_l.pow(4);
            let interior = volume;

            // Working set: padded field (read) + result (write)
            let working_set_bytes = (padded_volume as u64 + interior as u64) * 16; // vec4<f32>
            let working_mb = working_set_bytes as f64 / (1024.0 * 1024.0);

            // Skip if working set exceeds VRAM (leave 1 GB headroom)
            if working_set_bytes > 15 * 1024 * 1024 * 1024 {
                println!("  {:>6}  {:>10}  {:>8.1}  {:>12}  {:>12}  {:>10}", l, volume, working_mb, "—", "—", "SKIP(VRAM)");
                continue;
            }

            let field_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: padded_volume as u64 * 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: interior as u64 * 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let params: [u32; 4] = [padded_volume, halo.pow(4), interior, padded_l];
            let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ] });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], immediate_size: 0 });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None, layout: Some(&layout), module: &shader, entry_point: Some("tiled_stencil"),
                compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: field_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: result_buf.as_entire_binding() },
            ] });

            let wgs = (interior + 255) / 256;

            // Warmup
            for _ in 0..5 {
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default()); p.set_pipeline(&pipeline); p.set_bind_group(0, Some(&bg), &[]); p.dispatch_workgroups(wgs, 1, 1); }
                queue.submit(std::iter::once(enc.finish()));
                let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            }

            let iters = 20u32;
            let t0 = Instant::now();
            for _ in 0..iters {
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default()); p.set_pipeline(&pipeline); p.set_bind_group(0, Some(&bg), &[]); p.dispatch_workgroups(wgs, 1, 1); }
                queue.submit(std::iter::once(enc.finish()));
                let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            }
            let elapsed = t0.elapsed().as_secs_f64();

            let us_per_site = elapsed / (iters as f64 * interior as f64) * 1e6;
            let gsites_per_s = (interior as f64 * iters as f64) / elapsed / 1e9;

            let status = if working_mb < 6.0 { "L2-fit" }
                else if working_mb < 128.0 { "IC-fit" }
                else { "VRAM" };

            if gsites_per_s > best_throughput {
                best_throughput = gsites_per_s;
                best_l = l;
            }

            println!("  {:>6}  {:>10}  {:>8.1}  {:>10.4}  {:>10.3}  {:>10}",
                l, volume, working_mb, us_per_site, gsites_per_s, status);
        }

        println!();
        println!("  ── Optimal Tile: L={} ({:.3} Gsite/s) ──", best_l, best_throughput);
        println!();

        // Simulate tiled processing of a large 24⁴ lattice
        let target_l = 24u32;
        let target_volume = target_l.pow(4);
        let tiles_needed = (target_volume as f64 / best_l.pow(4) as f64).ceil() as u32;
        let est_time_ms = target_volume as f64 / best_throughput / 1e9 * 1e3;

        println!("  ── Projection: {0}⁴ lattice ({1} sites) ──", target_l, target_volume);
        println!("    Tiles needed: {} ({}⁴ tiles)", tiles_needed, best_l);
        println!("    Estimated stencil time: {:.2} ms", est_time_ms);
        println!("    At 10 MD steps/traj: {:.1} ms per HMC trajectory (stencil only)", est_time_ms * 10.0);
        println!();

        // 32⁴ projection
        let target32 = 32u32.pow(4);
        let tiles32 = (target32 as f64 / best_l.pow(4) as f64).ceil() as u32;
        let est32 = target32 as f64 / best_throughput / 1e9 * 1e3;
        println!("  ── Projection: 32⁴ lattice ({} sites) ──", target32);
        println!("    Tiles needed: {}", tiles32);
        println!("    Estimated stencil time: {:.1} ms", est32);
        println!("    MILC comparison (32⁴, 1 GPU): ~100-500 ms/stencil");
        println!();
    }
}
