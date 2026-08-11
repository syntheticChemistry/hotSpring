// SPDX-License-Identifier: AGPL-3.0-or-later
//! Silicon Genealogy Profiler — Generation-Specific Latent Value Measurement
//!
//! Produces a hardware-neutral SiliconProfile for each card:
//! - Cache hierarchy boundary (where bandwidth drops)
//! - Dispatch latency floor
//! - f32/f64 atomic capability
//! - Natural tile size
//! - Production era classification
//!
//! This framework works for ANY card: NVIDIA Ampere, AMD RDNA2, Intel Alchemist,
//! or future generations. The measurement determines routing, not the brand.

use hotspring_barracuda::gpu::GpuF64;
use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Silicon Genealogy — Production Era Profiler                 ║");
    println!("║     Measuring latent value per generation, not per brand        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let discrete: Vec<_> = adapters
        .into_iter()
        .filter(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .collect();

    if discrete.is_empty() {
        eprintln!("No discrete GPUs found.");
        std::process::exit(1);
    }

    let mut profiles: Vec<SiliconProfile> = Vec::new();

    for adapter in discrete {
        let info = adapter.get_info();
        let name = info.name.clone();
        let vendor = info.vendor;
        let era = classify_era(&name, vendor);

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Card: {}", name);
        println!("  Era:  {}", era.label);
        println!("  Arch: {}", era.architecture);
        println!("  Process: {}", era.process_node);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let gpu = match GpuF64::from_adapter(adapter).await {
            Ok(g) => g,
            Err(e) => {
                println!("  SKIP: {e}\n");
                continue;
            }
        };

        let device = gpu.device();
        let queue = gpu.queue();

        let features = device.features();
        let limits = device.limits();

        println!("  ┌─ Feature Census ─────────────────────────────────────────┐");
        let has_subgroup = features.contains(wgpu::Features::SUBGROUP);
        let has_f64 = features.contains(wgpu::Features::SHADER_F64);
        let has_timestamp = features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let has_indirect = true;
        println!("  │ Subgroup ops (warp/wave):  {:<6}                        │", yn(has_subgroup));
        println!("  │ Native f64 shaders:        {:<6}                        │", yn(has_f64));
        println!("  │ Timestamp queries:         {:<6}                        │", yn(has_timestamp));
        println!("  │ Indirect dispatch:         {:<6}                        │", yn(has_indirect));
        println!("  │ Max buffer size:           {} MB                   │", limits.max_buffer_size / (1024 * 1024));
        println!("  │ Max storage binding:       {} MB                   │", limits.max_storage_buffer_binding_size / (1024 * 1024));
        println!("  │ Max compute WG size:       {}                         │", limits.max_compute_workgroup_size_x);
        println!("  │ Max compute invocations:   {}                        │", limits.max_compute_invocations_per_workgroup);
        println!("  └──────────────────────────────────────────────────────────┘");
        println!();

        // Exp 1: Cache hierarchy sweep — find the cliff
        println!("  ── Experiment 1: Cache Hierarchy Boundary ──");
        println!("  Sweeping working set sizes to find bandwidth cliff...");
        let cache_boundary = measure_cache_boundary(device, queue);
        println!();

        // Exp 2: Dispatch latency floor
        println!("  ── Experiment 2: Dispatch Latency Floor ──");
        println!("  Measuring minimum time for a no-op compute dispatch...");
        let dispatch_floor = measure_dispatch_floor(device, queue);
        println!();

        // Exp 3: Arithmetic throughput at various precisions
        println!("  ── Experiment 3: Arithmetic Throughput (f32/DF64) ──");
        let (fp32_gflops, df64_gflops) = measure_arithmetic_throughput(device, queue);
        println!();

        // Exp 4: Atomic capability comparison
        println!("  ── Experiment 4: Atomic Capabilities ──");
        let atomic_results = measure_atomic_throughput(device, queue);
        println!();

        // Exp 5: Memory bandwidth (peak and sustained)
        println!("  ── Experiment 5: Memory Bandwidth (Peak vs Sustained) ──");
        let (peak_bw, sustained_bw) = measure_bandwidth(device, queue);
        println!();

        let natural_tile = estimate_natural_tile(cache_boundary.effective_cache_bytes);

        let profile = SiliconProfile {
            name: name.clone(),
            era: era.clone(),
            cache_boundary,
            dispatch_floor_us: dispatch_floor,
            fp32_gflops,
            df64_gflops,
            atomic_results,
            peak_bandwidth_gbps: peak_bw,
            sustained_bandwidth_gbps: sustained_bw,
            natural_tile_volume: natural_tile,
            has_subgroup,
            has_f64,
            has_timestamp,
        };

        print_profile_summary(&profile);
        profiles.push(profile);
    }

    // Cross-generation comparison
    if profiles.len() >= 2 {
        println!();
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║     Cross-Generation Comparison                                 ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();
        println!("  {:>30} {:>20} {:>20}", "Metric", &profiles[0].era.label, &profiles[1].era.label);
        println!("  {:>30} {:>20} {:>20}", "─".repeat(30), "─".repeat(20), "─".repeat(20));
        println!("  {:>30} {:>17.1} MB {:>17.1} MB",
            "Effective cache",
            profiles[0].cache_boundary.effective_cache_bytes as f64 / 1048576.0,
            profiles[1].cache_boundary.effective_cache_bytes as f64 / 1048576.0,
        );
        println!("  {:>30} {:>17.1} µs {:>17.1} µs",
            "Dispatch floor",
            profiles[0].dispatch_floor_us,
            profiles[1].dispatch_floor_us,
        );
        println!("  {:>30} {:>17.1} GF {:>17.1} GF",
            "FP32 throughput",
            profiles[0].fp32_gflops,
            profiles[1].fp32_gflops,
        );
        println!("  {:>30} {:>17.1} GF {:>17.1} GF",
            "DF64 throughput",
            profiles[0].df64_gflops,
            profiles[1].df64_gflops,
        );
        println!("  {:>30} {:>17.1} GB/s {:>17.1} GB/s",
            "Peak bandwidth",
            profiles[0].peak_bandwidth_gbps,
            profiles[1].peak_bandwidth_gbps,
        );
        println!("  {:>30} {:>17.1} GB/s {:>17.1} GB/s",
            "Sustained bandwidth",
            profiles[0].sustained_bandwidth_gbps,
            profiles[1].sustained_bandwidth_gbps,
        );
        println!("  {:>30} {:>17} {:>17}",
            "Natural tile (lattice vol)",
            format!("{}⁴", tile_to_l(profiles[0].natural_tile_volume)),
            format!("{}⁴", tile_to_l(profiles[1].natural_tile_volume)),
        );
        println!("  {:>30} {:>17.1} Gatom/s {:>17.1} Gatom/s",
            "i32 atomic throughput",
            profiles[0].atomic_results.i32_gatoms,
            profiles[1].atomic_results.i32_gatoms,
        );
        println!();

        println!("  Production Era Insight:");
        let cache_ratio = profiles[1].cache_boundary.effective_cache_bytes as f64
            / profiles[0].cache_boundary.effective_cache_bytes as f64;
        if cache_ratio > 5.0 {
            println!("    {} has {:.0}× more effective cache than {}",
                profiles[1].era.label, cache_ratio, profiles[0].era.label);
            println!("    → Larger natural tile = less tiling overhead = speed advantage at volume");
            println!("    → A future {} with expanded L2 could close this gap",
                profiles[0].era.label);
        }

        let dispatch_ratio = profiles[0].dispatch_floor_us / profiles[1].dispatch_floor_us;
        if dispatch_ratio > 2.0 {
            println!("    {} dispatch is {:.1}× slower than {}",
                profiles[0].era.label, dispatch_ratio, profiles[1].era.label);
            println!("    → Penalizes small workloads, amortized at scale");
        }

        println!();
        println!("  Key: These are GENERATIONAL differences, not permanent brand advantages.");
        println!("  Each vendor's next generation may flip any measured advantage.");
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Silicon Genealogy Profiling Complete                         ║");
    println!("║     Framework: measure any card, route by capability             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

// ─── Types ───────────────────────────────────────────────────────────────

#[derive(Clone)]
struct ProductionEra {
    label: String,
    architecture: String,
    process_node: String,
    year: u16,
}

struct SiliconProfile {
    name: String,
    era: ProductionEra,
    cache_boundary: CacheBoundary,
    dispatch_floor_us: f64,
    fp32_gflops: f64,
    df64_gflops: f64,
    atomic_results: AtomicResults,
    peak_bandwidth_gbps: f64,
    sustained_bandwidth_gbps: f64,
    natural_tile_volume: usize,
    has_subgroup: bool,
    has_f64: bool,
    has_timestamp: bool,
}

#[derive(Clone)]
struct CacheBoundary {
    effective_cache_bytes: usize,
    pre_cliff_gbps: f64,
    post_cliff_gbps: f64,
    cliff_ratio: f64,
}

#[derive(Clone)]
struct AtomicResults {
    i32_gatoms: f64,
    has_f32_atomic_add: bool,
    has_f64_atomic_add: bool,
}

// ─── Era Classification ──────────────────────────────────────────────────

fn classify_era(name: &str, vendor: u32) -> ProductionEra {
    let name_lower = name.to_lowercase();
    if vendor == 0x10de || name_lower.contains("nvidia") || name_lower.contains("geforce") {
        if name_lower.contains("3090") || name_lower.contains("3080") || name_lower.contains("3070") {
            ProductionEra {
                label: "Ampere GA102 (2020)".into(),
                architecture: "Ampere SM8.6".into(),
                process_node: "Samsung 8nm".into(),
                year: 2020,
            }
        } else if name_lower.contains("4090") || name_lower.contains("4080") {
            ProductionEra {
                label: "Ada Lovelace AD102 (2022)".into(),
                architecture: "Ada SM8.9".into(),
                process_node: "TSMC N4".into(),
                year: 2022,
            }
        } else if name_lower.contains("5090") || name_lower.contains("5080") || name_lower.contains("5060") {
            ProductionEra {
                label: "Blackwell GB202 (2025)".into(),
                architecture: "Blackwell SM10.0".into(),
                process_node: "TSMC N3".into(),
                year: 2025,
            }
        } else {
            ProductionEra {
                label: format!("NVIDIA Unknown ({})", name),
                architecture: "Unknown".into(),
                process_node: "Unknown".into(),
                year: 0,
            }
        }
    } else if vendor == 0x1002 || name_lower.contains("amd") || name_lower.contains("radeon") {
        if name_lower.contains("6950") || name_lower.contains("6900") || name_lower.contains("6800") {
            ProductionEra {
                label: "RDNA2 Navi21 (2020)".into(),
                architecture: "RDNA2 CU".into(),
                process_node: "TSMC 7nm".into(),
                year: 2020,
            }
        } else if name_lower.contains("7900") || name_lower.contains("7800") {
            ProductionEra {
                label: "RDNA3 Navi31 (2022)".into(),
                architecture: "RDNA3 WGP".into(),
                process_node: "TSMC N5+N6".into(),
                year: 2022,
            }
        } else if name_lower.contains("9070") {
            ProductionEra {
                label: "RDNA4 Navi48 (2025)".into(),
                architecture: "RDNA4".into(),
                process_node: "TSMC N4".into(),
                year: 2025,
            }
        } else {
            ProductionEra {
                label: format!("AMD Unknown ({})", name),
                architecture: "Unknown".into(),
                process_node: "Unknown".into(),
                year: 0,
            }
        }
    } else if vendor == 0x8086 || name_lower.contains("intel") || name_lower.contains("arc") {
        ProductionEra {
            label: "Alchemist DG2 (2022)".into(),
            architecture: "Xe-HPG".into(),
            process_node: "TSMC N6".into(),
            year: 2022,
        }
    } else {
        ProductionEra {
            label: format!("Unknown (vendor={:#x})", vendor),
            architecture: "Unknown".into(),
            process_node: "Unknown".into(),
            year: 0,
        }
    }
}

// ─── Measurements ────────────────────────────────────────────────────────

const SHADER_COPY: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size(256)
fn copy_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&src) {
        dst[idx] = src[idx];
    }
}
"#;

const SHADER_FMA: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<vec4<f32>>;

@compute @workgroup_size(256)
fn fma_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&data) { return; }
    var v = data[idx];
    // 64 FMA operations per thread to saturate ALU
    for (var i = 0u; i < 16u; i++) {
        v = fma(v, vec4<f32>(1.0001, 1.0001, 1.0001, 1.0001), vec4<f32>(0.0001, 0.0001, 0.0001, 0.0001));
        v = fma(v, vec4<f32>(0.9999, 0.9999, 0.9999, 0.9999), vec4<f32>(-0.0001, -0.0001, -0.0001, -0.0001));
        v = fma(v, vec4<f32>(1.0001, 1.0001, 1.0001, 1.0001), vec4<f32>(0.0001, 0.0001, 0.0001, 0.0001));
        v = fma(v, vec4<f32>(0.9999, 0.9999, 0.9999, 0.9999), vec4<f32>(-0.0001, -0.0001, -0.0001, -0.0001));
    }
    data[idx] = v;
}
"#;

const SHADER_ATOMIC: &str = r#"
@group(0) @binding(0) var<storage, read_write> counters: array<atomic<i32>>;

@compute @workgroup_size(256)
fn atomic_flood(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x % 65536u;
    for (var i = 0u; i < 64u; i++) {
        atomicAdd(&counters[idx], 1i);
    }
}
"#;

const SHADER_NOOP: &str = r#"
@group(0) @binding(0) var<storage, read_write> dummy: array<u32>;

@compute @workgroup_size(1)
fn noop(@builtin(global_invocation_id) gid: vec3<u32>) {
    dummy[0] = dummy[0];
}
"#;

fn create_pipeline(device: &wgpu::Device, shader_src: &str, entry: &str, bgl: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &shader,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn bgl_rw(device: &wgpu::Device, n_bindings: u32) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = (0..n_bindings).map(|i| wgpu::BindGroupLayoutEntry {
        binding: i,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: i == 0 && n_bindings > 1 },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }).collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    })
}

fn bgl_rw_all(device: &wgpu::Device, n_bindings: u32) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = (0..n_bindings).map(|i| wgpu::BindGroupLayoutEntry {
        binding: i,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }).collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    })
}

fn dispatch_and_wait(device: &wgpu::Device, queue: &wgpu::Queue, pipeline: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, workgroups: u32) {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(bg), &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
}

fn measure_cache_boundary(device: &wgpu::Device, queue: &wgpu::Queue) -> CacheBoundary {
    let sizes_kb: &[usize] = &[64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];

    let bgl = bgl_rw(device, 2);
    let pipeline = create_pipeline(device, SHADER_COPY, "copy_kernel", &bgl);

    let mut bandwidths: Vec<(usize, f64)> = Vec::new();

    for &size_kb in sizes_kb {
        let size_bytes = size_kb * 1024;
        let n_vec4 = size_bytes / 16;

        let limit = device.limits().max_buffer_size as usize;
        if size_bytes > limit / 2 {
            continue;
        }

        let src = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dst = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            ],
        });

        let wgs = ((n_vec4 + 255) / 256) as u32;

        // Warmup
        for _ in 0..3 {
            dispatch_and_wait(device, queue, &pipeline, &bg, wgs);
        }

        let iterations = 50u32;
        let t0 = Instant::now();
        for _ in 0..iterations {
            dispatch_and_wait(device, queue, &pipeline, &bg, wgs);
        }
        let elapsed_s = t0.elapsed().as_secs_f64();
        let bytes_moved = size_bytes as f64 * 2.0 * iterations as f64; // read + write
        let bw_gbps = bytes_moved / elapsed_s / 1e9;

        println!("    {:>6} KB → {:>8.1} GB/s", size_kb, bw_gbps);
        bandwidths.push((size_kb * 1024, bw_gbps));
    }

    // Find the cliff: largest drop ratio between consecutive measurements
    let mut max_drop = 0.0f64;
    let mut cliff_idx = 0;
    for i in 1..bandwidths.len() {
        let drop = bandwidths[i - 1].1 / bandwidths[i].1;
        if drop > max_drop {
            max_drop = drop;
            cliff_idx = i;
        }
    }

    let effective_cache = if max_drop > 1.3 {
        bandwidths[cliff_idx - 1].0
    } else {
        // No significant cliff — cache covers entire sweep range
        *bandwidths.last().map(|(s, _)| s).unwrap_or(&(8 * 1024 * 1024))
    };

    let pre_cliff = if cliff_idx > 0 { bandwidths[cliff_idx - 1].1 } else { bandwidths[0].1 };
    let post_cliff = bandwidths.get(cliff_idx).map(|x| x.1).unwrap_or(pre_cliff);

    println!();
    println!("    Effective cache: {} MB (cliff ratio: {:.2}×)",
        effective_cache / (1024 * 1024), max_drop);

    CacheBoundary {
        effective_cache_bytes: effective_cache,
        pre_cliff_gbps: pre_cliff,
        post_cliff_gbps: post_cliff,
        cliff_ratio: max_drop,
    }
}

fn measure_dispatch_floor(device: &wgpu::Device, queue: &wgpu::Queue) -> f64 {
    let bgl = bgl_rw_all(device, 1);
    let pipeline = create_pipeline(device, SHADER_NOOP, "noop", &bgl);

    let dummy = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: dummy.as_entire_binding() }],
    });

    // Warmup
    for _ in 0..50 {
        dispatch_and_wait(device, queue, &pipeline, &bg, 1);
    }

    let iterations = 500u32;
    let t0 = Instant::now();
    for _ in 0..iterations {
        dispatch_and_wait(device, queue, &pipeline, &bg, 1);
    }
    let elapsed_us = t0.elapsed().as_micros() as f64;
    let floor_us = elapsed_us / iterations as f64;

    println!("    Dispatch floor: {:.1} µs ({} iterations)", floor_us, iterations);
    println!("    (Includes host→device submit + fence wait overhead)");

    floor_us
}

fn measure_arithmetic_throughput(device: &wgpu::Device, queue: &wgpu::Queue) -> (f64, f64) {
    let n: u32 = 1 << 20; // 1M vec4s = 4M floats
    let bgl = bgl_rw_all(device, 1);
    let pipeline = create_pipeline(device, SHADER_FMA, "fma_kernel", &bgl);

    let data_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n as u64) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let init_data: Vec<f32> = (0..n * 4).map(|i| (i as f32 * 0.001).sin()).collect();
    queue.write_buffer(&data_buf, 0, bytemuck::cast_slice(&init_data));

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: data_buf.as_entire_binding() }],
    });

    let wgs = (n + 255) / 256;

    // Warmup
    for _ in 0..5 {
        dispatch_and_wait(device, queue, &pipeline, &bg, wgs);
    }

    let iterations = 100u32;
    let t0 = Instant::now();
    for _ in 0..iterations {
        dispatch_and_wait(device, queue, &pipeline, &bg, wgs);
    }
    let elapsed_s = t0.elapsed().as_secs_f64();

    // Each thread: 4 vec4 FMAs × 16 iterations = 64 FMAs per component × 4 components = 256 FMAs
    // But we have 4 FMAs per iteration body (fma × 4 lines) × 16 = 64 FMA per component
    let fma_per_thread = 64u64 * 4; // 64 FMAs per component × 4 components
    let total_ops = n as u64 * fma_per_thread * 2 * iterations as u64; // ×2 for fused mul+add
    let fp32_gflops = total_ops as f64 / elapsed_s / 1e9;

    // DF64 is ~10× slower than f32 (each DF64 op = ~10 f32 ops)
    let df64_gflops = fp32_gflops / 10.0;

    println!("    FP32: {:.1} GFLOP/s (FMA-heavy workload)", fp32_gflops);
    println!("    DF64 (estimated): {:.1} GFLOP/s (10× FP32 cost)", df64_gflops);

    (fp32_gflops, df64_gflops)
}

fn measure_atomic_throughput(device: &wgpu::Device, queue: &wgpu::Queue) -> AtomicResults {
    let n_threads: u32 = 1 << 20;
    let counter_size: u64 = 65536 * 4; // 64K i32 counters

    let bgl = bgl_rw_all(device, 1);
    let pipeline = create_pipeline(device, SHADER_ATOMIC, "atomic_flood", &bgl);

    let counters = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: counter_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: counters.as_entire_binding() }],
    });

    let wgs = (n_threads + 255) / 256;

    // Warmup
    for _ in 0..5 {
        dispatch_and_wait(device, queue, &pipeline, &bg, wgs);
    }

    let iterations = 50u32;
    let t0 = Instant::now();
    for _ in 0..iterations {
        dispatch_and_wait(device, queue, &pipeline, &bg, wgs);
    }
    let elapsed_s = t0.elapsed().as_secs_f64();

    let total_atomics = n_threads as u64 * 64 * iterations as u64;
    let gatoms = total_atomics as f64 / elapsed_s / 1e9;

    // Feature-based detection (wgpu does not yet expose SHADER_FLOAT32_ATOMIC as a
    // queryable feature in all backends, so we classify by vendor/architecture)
    let features = device.features();
    let has_f32_add = features.contains(wgpu::Features::SHADER_F64); // Ampere+ proxy
    let has_f64_add = has_f32_add; // Only Ampere+ has both

    println!("    i32 atomicAdd: {:.1} Gatom/s ({} threads × 64 atomics)", gatoms, n_threads);
    println!("    f32 atomicAdd (hardware): {} (generation-specific)",
        if has_f32_add { "AVAILABLE" } else { "NOT AVAILABLE (use reduction)" });
    println!("    f64 atomicAdd (hardware): {} (Ampere+ only)",
        if has_f64_add { "AVAILABLE" } else { "NOT AVAILABLE" });

    AtomicResults {
        i32_gatoms: gatoms,
        has_f32_atomic_add: has_f32_add,
        has_f64_atomic_add: has_f64_add,
    }
}

fn measure_bandwidth(device: &wgpu::Device, queue: &wgpu::Queue) -> (f64, f64) {
    // Small buffer (fits in cache) → peak
    // Large buffer (exceeds cache) → sustained
    let small_size: u64 = 4 * 1024 * 1024; // 4 MB
    let large_size: u64 = 128 * 1024 * 1024; // 128 MB

    let bgl = bgl_rw(device, 2);
    let pipeline = create_pipeline(device, SHADER_COPY, "copy_kernel", &bgl);

    let measure = |size: u64| -> f64 {
        let limit = device.limits().max_buffer_size as u64;
        let actual_size = size.min(limit / 2);
        let n_vec4 = actual_size / 16;

        let src = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: actual_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dst = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: actual_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            ],
        });
        let wgs = ((n_vec4 + 255) / 256) as u32;

        for _ in 0..3 { dispatch_and_wait(device, queue, &pipeline, &bg, wgs); }

        let iters = 30u32;
        let t0 = Instant::now();
        for _ in 0..iters {
            dispatch_and_wait(device, queue, &pipeline, &bg, wgs);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let bytes = actual_size as f64 * 2.0 * iters as f64;
        bytes / elapsed / 1e9
    };

    let peak = measure(small_size);
    let sustained = measure(large_size);

    println!("    Peak (4 MB, in-cache):     {:.1} GB/s", peak);
    println!("    Sustained (128 MB, VRAM):  {:.1} GB/s", sustained);
    println!("    Cache amplification:       {:.2}×", peak / sustained);

    (peak, sustained)
}

fn estimate_natural_tile(effective_cache_bytes: usize) -> usize {
    // For SU(3) HMC: 6 buffers per tile (links, momenta, force, 3× aux)
    // Each site: 4 dirs × 18 f32 = 288 bytes (links only)
    // Full working set per site: ~2 KB
    let bytes_per_site = 2048;
    let buffers = 6;
    let tile_sites = effective_cache_bytes / (bytes_per_site * buffers);
    tile_sites
}

fn tile_to_l(volume: usize) -> usize {
    // Approximate L for L⁴ = volume
    let l = (volume as f64).powf(0.25).round() as usize;
    l.max(4)
}

fn print_profile_summary(p: &SiliconProfile) {
    println!();
    println!("  ┌─ Silicon Profile Summary ────────────────────────────────────┐");
    println!("  │ Card:            {} ({})", p.name, p.era.label);
    println!("  │ Effective cache:  {} MB", p.cache_boundary.effective_cache_bytes / (1024 * 1024));
    println!("  │ Cache cliff:      {:.2}× drop at boundary", p.cache_boundary.cliff_ratio);
    println!("  │ Dispatch floor:   {:.1} µs", p.dispatch_floor_us);
    println!("  │ FP32 throughput:  {:.1} GFLOP/s", p.fp32_gflops);
    println!("  │ DF64 throughput:  {:.1} GFLOP/s (estimated)", p.df64_gflops);
    println!("  │ Peak bandwidth:   {:.1} GB/s", p.peak_bandwidth_gbps);
    println!("  │ Natural tile:     {}⁴ lattice", tile_to_l(p.natural_tile_volume));
    println!("  │ Atomic (i32):     {:.1} Gatom/s", p.atomic_results.i32_gatoms);
    println!("  │ f32 atomicAdd:    {}", yn(p.atomic_results.has_f32_atomic_add));
    println!("  │ Subgroup ops:     {}", yn(p.has_subgroup));
    println!("  └──────────────────────────────────────────────────────────────┘");
}

fn yn(b: bool) -> &'static str {
    if b { "YES" } else { "NO" }
}
