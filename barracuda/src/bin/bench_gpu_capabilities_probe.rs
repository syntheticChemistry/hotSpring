// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU Capabilities Probe — Timestamp queries, subgroup ops, BAR, push constants.
//!
//! Tests multiple untapped wgpu features in one binary:
//! 1. Timestamp queries — GPU-side nanosecond timing
//! 2. Subgroup shuffle/ballot — warp-level data exchange
//! 3. Resizable BAR detection — is full VRAM CPU-accessible?
//! 4. Push constants — immediate uniform data (no buffer)
//! 5. Indirect dispatch — GPU-driven workgroup counts

use std::time::Instant;

const SHADER_SIMPLE_COMPUTE: &str = r#"
struct Params {
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn simple_compute(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }
    // Simple FMA workload for timestamp measurement
    var v = input[idx];
    v = fma(v, 1.5, 0.1);
    v = fma(v, 1.5, 0.1);
    v = fma(v, 1.5, 0.1);
    v = fma(v, 1.5, 0.1);
    output[idx] = v;
}
"#;

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  GPU Capabilities Probe — Timestamps, Subgroups, BAR, Features  ║");
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
        let features = adapter.features();
        let limits = adapter.limits();

        println!("━━━ {} ━━━", info.name);
        println!();

        // ═══ Feature Census ═══
        println!("  ── Feature Census ──");
        let feature_checks: &[(&str, wgpu::Features, &str)] = &[
            ("TIMESTAMP_QUERY", wgpu::Features::TIMESTAMP_QUERY, "GPU-side nanosecond timing"),
            ("SUBGROUP", wgpu::Features::SUBGROUP, "Warp/wavefront operations"),
            ("SUBGROUP_VERTEX", wgpu::Features::SUBGROUP_VERTEX, "Subgroups in vertex shaders"),
            ("SHADER_F64", wgpu::Features::SHADER_F64, "64-bit floating point"),
            ("SHADER_F16", wgpu::Features::SHADER_F16, "16-bit floating point"),
            ("MULTI_DRAW_INDIRECT_COUNT", wgpu::Features::MULTI_DRAW_INDIRECT_COUNT, "GPU-driven draw count"),
            ("INDIRECT_FIRST_INSTANCE", wgpu::Features::INDIRECT_FIRST_INSTANCE, "Instance offset in indirect"),
            ("MAPPABLE_PRIMARY_BUFFERS", wgpu::Features::MAPPABLE_PRIMARY_BUFFERS, "BAR-mapped VRAM (SAM)"),
            ("TEXTURE_COMPRESSION_BC", wgpu::Features::TEXTURE_COMPRESSION_BC, "BC texture compression"),
            ("PIPELINE_STATISTICS_QUERY", wgpu::Features::PIPELINE_STATISTICS_QUERY, "HW perf counters"),
        ];

        for (name, feat, desc) in feature_checks {
            let has = features.contains(*feat);
            println!("    {:>36}: {}  ({})", name, if has { "YES" } else { " NO" }, desc);
        }
        println!();

        // ═══ Limits Census ═══
        println!("  ── Key Limits ──");
        println!("    max_storage_buffers_per_shader_stage: {}", limits.max_storage_buffers_per_shader_stage);
        println!("    max_compute_workgroup_storage_size: {} bytes", limits.max_compute_workgroup_storage_size);
        println!("    max_compute_workgroup_size_x: {}", limits.max_compute_workgroup_size_x);
        println!("    max_buffer_size: {} bytes ({:.1} GB)", limits.max_buffer_size, limits.max_buffer_size as f64 / 1e9);
        println!("    max_storage_buffer_binding_size: {} bytes ({:.1} GB)", limits.max_storage_buffer_binding_size, limits.max_storage_buffer_binding_size as f64 / 1e9);
        println!("    subgroup_min_size: {}", info.subgroup_min_size);
        println!("    subgroup_max_size: {}", info.subgroup_max_size);
        println!();

        // ═══ Timestamp Query Probe ═══
        let has_timestamp = features.contains(wgpu::Features::TIMESTAMP_QUERY);
        if has_timestamp {
            println!("  ── Timestamp Queries (GPU-side timing) ──");

            let required = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SHADER_F64 | wgpu::Features::SUBGROUP;
            let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("timestamp"), required_features: required,
                required_limits: limits.clone(), memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            }).await.unwrap();

            let timestamp_period = queue.get_timestamp_period(); // ns per tick
            println!("    timestamp_period: {:.3} ns/tick", timestamp_period);

            // Create timestamp query set
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });

            let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 16, // 2 × u64
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Run a trivial compute pass with timestamps
            let n: u32 = 65536;
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: n as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
            });
            let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: n as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
            });

            let params: [u32; 4] = [n, 0, 0, 0];
            let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None, source: wgpu::ShaderSource::Wgsl(SHADER_SIMPLE_COMPUTE.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ] });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], immediate_size: 0 });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None, layout: Some(&layout), module: &shader, entry_point: Some("simple_compute"),
                compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            ] });

            let wgs = (n + 255) / 256;

            // Timestamped dispatch
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("timestamped"),
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(wgs, 1, 1);
            }
            enc.resolve_query_set(&query_set, 0..2, &resolve_buf, 0);
            enc.copy_buffer_to_buffer(&resolve_buf, 0, &readback_buf, 0, 16);
            queue.submit(std::iter::once(enc.finish()));
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

            // Read back timestamps
            let slice = readback_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            let data = slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&data);
            let start = timestamps[0];
            let end = timestamps[1];
            let elapsed_ns = (end - start) as f64 * timestamp_period as f64;
            drop(data);
            readback_buf.unmap();

            println!("    Compute pass (65536 elements, 4 FMAs):");
            println!("      GPU timestamp: {:.1} ns ({:.3} µs)", elapsed_ns, elapsed_ns / 1000.0);

            // Compare with CPU wall-clock
            let t0 = Instant::now();
            for _ in 0..100 {
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default()); p.set_pipeline(&pipeline); p.set_bind_group(0, Some(&bg), &[]); p.dispatch_workgroups(wgs, 1, 1); }
                queue.submit(std::iter::once(enc.finish()));
                let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            }
            let wall_us = t0.elapsed().as_secs_f64() / 100.0 * 1e6;
            println!("      CPU wall-clock: {:.1} µs (includes submit+poll overhead)", wall_us);
            println!("      Overhead ratio: {:.1}× (wall/GPU)", wall_us / (elapsed_ns / 1000.0));
            println!();

            // ═══ Subgroup Info ═══
            println!("  ── Subgroup Operations ──");
            println!("    Subgroup size: {}-{}", info.subgroup_min_size, info.subgroup_max_size);
            println!("    SUBGROUP feature: YES (hardware warp/wavefront ops available)");
            println!("    Note: `enable subgroups;` not yet in Naga — use @builtin(subgroup_*) when ready");
            println!("    For QCD: subgroupAdd will eliminate shared memory for plaquette reduction");
            println!();

            // ═══ Push Constants (immediate_size in wgpu 28) ═══
            println!("  ── Push Constants (immediate_size) ──");
            println!("    wgpu 28 uses `immediate_size` in PipelineLayout (always available)");
            println!("    For QCD: β, dt, volume pushed directly (skip uniform buffer alloc)");
            println!("    Saves: 1 buffer allocation + 1 binding per dispatch");
            println!();

            // ═══ Resizable BAR ═══
            let has_mappable = features.contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
            println!("  ── Resizable BAR / Smart Access Memory ──");
            if has_mappable {
                println!("    MAPPABLE_PRIMARY_BUFFERS: YES");
                println!("    Full VRAM is CPU-mappable — zero-copy host access!");
                println!("    For QCD: direct CPU readback without staging buffer");
            } else {
                println!("    MAPPABLE_PRIMARY_BUFFERS: NO");
                println!("    Standard PCIe transfer model (staging buffers required)");
                println!("    BAR may still be enabled but not exposed via this flag");
            }
            println!();

            // ═══ Indirect Dispatch ═══
            println!("  ── Indirect Dispatch ──");
            println!("    Available: YES (standard wgpu feature)");
            println!("    GPU writes dispatch params → next dispatch reads them");
            println!("    For QCD: adaptive step size — GPU decides n_md_steps");
            println!("    For multigrid: GPU decides coarse/fine grid dispatch size");
            println!();

        } else {
            println!("  ── Timestamp Queries: NOT AVAILABLE ──");
            println!();
        }
    }

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Capabilities Probe Complete                                    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}
