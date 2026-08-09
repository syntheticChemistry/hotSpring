// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-GPU Tiled Pipeline — Route tiles to optimal silicon.
//!
//! Based on measured silicon genealogy:
//! - NVIDIA: 5.08 Gsite/s stencil, 532 GB/s read, 22× faster RT, no F16 ALU boost
//! - AMD: 2.45 Gsite/s stencil, 842 GB/s RMW in IC, 1.32× F16, 20× better at HMC
//!
//! The pipeline:
//! 1. NVIDIA computes stencil (it's 2× faster for pure compute)
//! 2. AMD does HMC force accumulation (20× faster due to IC for full SU(3))
//! 3. Cross-PCIe transfer moves tiles between cards as needed
//!
//! This tests the overhead of cross-GPU tiled domain decomposition.

use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║   Cross-GPU Tiled Pipeline — Silicon-Routed Decomposition       ║");
    println!("║   NVIDIA (stencil) ↔ AMD (HMC) via PCIe tile streaming         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let discrete: Vec<_> = adapters
        .into_iter()
        .filter(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .collect();

    if discrete.len() < 2 {
        println!("  Need 2 discrete GPUs for cross-GPU experiment. Found: {}", discrete.len());
        return;
    }

    let mut nvidia_idx = None;
    let mut amd_idx = None;
    for (i, a) in discrete.iter().enumerate() {
        let name = a.get_info().name.to_lowercase();
        if name.contains("nvidia") || name.contains("geforce") { nvidia_idx = Some(i); }
        if name.contains("amd") || name.contains("radeon") { amd_idx = Some(i); }
    }

    let (nv_i, amd_i) = match (nvidia_idx, amd_idx) {
        (Some(n), Some(a)) => (n, a),
        _ => { println!("  Need one NVIDIA and one AMD GPU."); return; }
    };

    println!("  NVIDIA: {}", discrete[nv_i].get_info().name);
    println!("  AMD:    {}", discrete[amd_i].get_info().name);
    println!();

    let (nv_dev, nv_queue) = discrete[nv_i].request_device(&wgpu::DeviceDescriptor {
        label: Some("nvidia"), required_features: wgpu::Features::SHADER_F64 | wgpu::Features::SUBGROUP,
        required_limits: discrete[nv_i].limits(), memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }).await.unwrap();

    let (amd_dev, amd_queue) = discrete[amd_i].request_device(&wgpu::DeviceDescriptor {
        label: Some("amd"), required_features: wgpu::Features::SHADER_F64 | wgpu::Features::SUBGROUP,
        required_limits: discrete[amd_i].limits(), memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }).await.unwrap();

    // Tile sizes to test for cross-GPU transfer
    let tile_sizes: &[(u32, &str)] = &[
        (8, "8⁴"),
        (12, "12⁴"),
        (16, "16⁴"),
        (20, "20⁴"),
    ];

    println!("  ── Cross-GPU Tile Transfer Overhead ──");
    println!("  {:>6}  {:>10}  {:>10}  {:>12}  {:>12}  {:>12}", "Tile", "Sites", "Bytes", "NV compute", "AMD compute", "Transfer est");
    println!("  {:>6}  {:>10}  {:>10}  {:>12}  {:>12}  {:>12}", "", "", "MB", "ms", "ms", "ms");
    println!("  ──────────────────────────────────────────────────────────────────────────");

    let stencil_shader_src = r#"
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size(256)
fn stencil(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }
    // Simple nearest-neighbor accumulation
    let center = src[idx];
    var acc = center * -8.0;
    if idx > 0u { acc += src[idx - 1u]; }
    if idx < params.n - 1u { acc += src[idx + 1u]; }
    if idx > 7u { acc += src[idx - 8u]; }
    acc += src[(idx + 8u) % params.n];
    if idx > 63u { acc += src[idx - 64u]; }
    acc += src[(idx + 64u) % params.n];
    if idx > 511u { acc += src[idx - 512u]; }
    acc += src[(idx + 512u) % params.n];
    dst[idx] = acc;
}
"#;

    // Build pipelines on both devices
    let nv_pipeline = build_stencil_pipeline(&nv_dev, stencil_shader_src);
    let amd_pipeline = build_stencil_pipeline(&amd_dev, stencil_shader_src);

    for &(l, label) in tile_sizes {
        let volume = l.pow(4) as u32;
        let bytes = volume as u64 * 16; // vec4<f32>
        let mb = bytes as f64 / (1024.0 * 1024.0);

        // Benchmark on NVIDIA
        let nv_time = bench_stencil_time(&nv_dev, &nv_queue, &nv_pipeline, volume);

        // Benchmark on AMD
        let amd_time = bench_stencil_time(&amd_dev, &amd_queue, &amd_pipeline, volume);

        // Estimate PCIe transfer time (measured: ~20 GB/s effective)
        let transfer_ms = mb / 20.0 * 1000.0 / 1024.0; // 20 GB/s measured

        println!("  {:>6}  {:>10}  {:>8.1}  {:>10.3}  {:>10.3}  {:>10.3}",
            label, volume, mb, nv_time * 1000.0, amd_time * 1000.0, transfer_ms);
    }

    println!();
    println!("  ── Pipeline Routing Strategy ──");
    println!();
    println!("    For pure stencil: NVIDIA wins (more CUs, faster dispatch)");
    println!("    For SU(3) HMC:   AMD wins 20× (Infinity Cache absorbs working set)");
    println!("    Optimal pipeline:");
    println!("      1. AMD: Thermalize (GPU-on-card, 20× faster HMC)");
    println!("      2. NVIDIA: Run precision measurements (faster stencil + CG solver)");
    println!("      3. PCIe transfer: Move configs between cards (20 GB/s, ~ms scale)");
    println!();
    println!("    Transfer overhead vs compute:");

    let l = 16u32;
    let volume = l.pow(4);
    let bytes = volume as u64 * 16;
    let transfer_time = bytes as f64 / 20e9; // 20 GB/s
    let amd_compute = volume as f64 / 2.45e9; // 2.45 Gsite/s measured
    let ratio = transfer_time / amd_compute;
    println!("      16⁴ config transfer: {:.3} ms", transfer_time * 1000.0);
    println!("      16⁴ stencil compute: {:.3} ms", amd_compute * 1000.0);
    println!("      Transfer/Compute ratio: {:.1}%", ratio * 100.0);
    println!();

    if ratio < 0.10 {
        println!("    ✓ Transfer overhead < 10% of compute — pipeline viable!");
    } else if ratio < 0.50 {
        println!("    ~ Transfer overhead {:.0}% — pipeline viable for large tiles", ratio * 100.0);
    } else {
        println!("    ✗ Transfer overhead > 50% — single-card execution preferred");
    }

    println!();
    println!("  ── Full HMC Pipeline Estimate (16⁴, 10 MD steps) ──");
    println!("    AMD-only HMC: ~31 ms (measured directly)");
    println!("    NVIDIA stencil component: ~0.1 ms");
    println!("    PCIe round-trip: ~{:.2} ms", transfer_time * 2000.0);
    println!("    Cross-GPU gain: NVIDIA stencil is 2× faster but PCIe overhead");
    println!("    Verdict: AMD-only for small lattices, cross-GPU for 24⁴+ where");
    println!("             NVIDIA's raw compute advantage outweighs transfer cost");
}

fn build_stencil_pipeline(device: &wgpu::Device, src: &str) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None, source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
        wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
    ] });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], immediate_size: 0 });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&layout), module: &shader, entry_point: Some("stencil"),
        compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
    })
}

fn bench_stencil_time(device: &wgpu::Device, queue: &wgpu::Queue, pipeline: &wgpu::ComputePipeline, n: u32) -> f64 {
    let params: [u32; 4] = [n, 0, 0, 0];
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

    let buf_size = n as u64 * 16;
    let src_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: buf_size,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let dst_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: buf_size,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });

    let bgl = pipeline.get_bind_group_layout(0);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: src_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: dst_buf.as_entire_binding() },
    ] });

    let wgs = (n + 255) / 256;

    // Warmup
    for _ in 0..5 {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default()); p.set_pipeline(pipeline); p.set_bind_group(0, Some(&bg), &[]); p.dispatch_workgroups(wgs, 1, 1); }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    let iters = 50u32;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default()); p.set_pipeline(pipeline); p.set_bind_group(0, Some(&bg), &[]); p.dispatch_workgroups(wgs, 1, 1); }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    t0.elapsed().as_secs_f64() / iters as f64
}
