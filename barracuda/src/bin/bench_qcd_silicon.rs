// SPDX-License-Identifier: AGPL-3.0-only

//! QCD silicon benchmark: per-kernel timing across GPUs at multiple lattice sizes.
//!
//! Profiles each HMC kernel (force, plaquette, momentum update, link update,
//! kinetic energy, CG dot, sum reduce) individually on every available GPU.
//! Identifies ALU-bound vs memory-bound kernels and maps to silicon unit
//! opportunities from the all-silicon pipeline.
//!
//! ## What this measures
//!
//! For each (kernel, lattice_size, GPU) triple:
//! - Dispatch throughput (dispatches/sec)
//! - Effective bandwidth (bytes read+written / time)
//! - Compute intensity (FLOPs/byte estimate)
//!
//! ## Silicon mapping
//!
//! - ALU-bound (high compute intensity): shader_core is optimal
//! - Memory-bound (low compute intensity): TMU spatial cache could help
//! - Reduction-bound: shader_core with workgroup shared memory
//! - Matmul-shaped: tensor_core candidate (future)

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::toadstool_report::{self, PerformanceMeasurement};

use std::time::Instant;

// ── Kernel shaders (representative QCD operations at f32 for universal testing) ──

const SHADER_SU3_MATMUL: &str = r"
// SU(3) matrix multiply proxy: 3x3 complex = 108 FMA ops per site
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let base = f32(idx + 1u) * 0.001;
    var sum: f32 = 0.0;
    for (var i = 0u; i < 18u; i = i + 1u) {
        for (var j = 0u; j < 6u; j = j + 1u) {
            let a = base + f32(i) * 0.01;
            let b = base + f32(j) * 0.01;
            sum = fma(a, b, sum);
        }
    }
    out[idx] = sum;
}
";

const SHADER_FORCE_PROXY: &str = r"
// Gauge force proxy: 4 neighbor reads + 6 staple products + TA projection
// ~432 FMA per site (4 staples × 108 muls)
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let base = f32(idx + 1u) * 0.001;
    var force: f32 = 0.0;
    for (var staple = 0u; staple < 4u; staple = staple + 1u) {
        var prod: f32 = 0.0;
        for (var i = 0u; i < 108u; i = i + 1u) {
            let a = base + f32(staple * 108u + i) * 0.0001;
            prod = fma(a, a, prod);
        }
        force = force + prod;
    }
    out[idx] = force;
}
";

const SHADER_PLAQUETTE_PROXY: &str = r"
// Plaquette proxy: 4 matmuls + trace for 6 oriented planes per site
// ~648 FMA per site (6 planes × 108 muls)
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let base = f32(idx + 1u) * 0.001;
    var plaq: f32 = 0.0;
    for (var plane = 0u; plane < 6u; plane = plane + 1u) {
        var trace: f32 = 0.0;
        for (var i = 0u; i < 108u; i = i + 1u) {
            let a = base + f32(plane * 108u + i) * 0.0001;
            trace = fma(a, a, trace);
        }
        plaq = plaq + trace;
    }
    out[idx] = plaq;
}
";

const SHADER_LINK_UPDATE_PROXY: &str = r"
// Link update proxy: Cayley exp(dt*P)*U = (1 + dt/2*P)/(1 - dt/2*P) * U
// ~200 FMA per site (matrix inverse + multiply + reunitarize)
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let base = f32(idx + 1u) * 0.001;
    var link: f32 = base;
    for (var i = 0u; i < 200u; i = i + 1u) {
        link = fma(link, 0.999, base * 0.001);
    }
    out[idx] = link;
}
";

const SHADER_CG_DOT_PROXY: &str = r"
// CG dot product proxy: complex dot of two lattice vectors
// 4 FMA per site (re*re + im*im for each pair) + workgroup reduce
var<workgroup> wg_data: array<f32, 256>;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let base = f32(gid.x + 1u) * 0.001;
    wg_data[lid.x] = fma(base, base, base * 0.5);
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if lid.x < s { wg_data[lid.x] = wg_data[lid.x] + wg_data[lid.x + s]; }
        workgroupBarrier();
    }
    if lid.x == 0u { out[gid.x / 256u] = wg_data[0]; }
}
";

const SHADER_MOMENTUM_UPDATE_PROXY: &str = r"
// Momentum update proxy: P += dt * F (18 f64 = 36 f32 adds per site)
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let base = f32(idx + 1u) * 0.001;
    var mom: f32 = base;
    for (var i = 0u; i < 36u; i = i + 1u) {
        mom = mom + base * 0.01;
    }
    out[idx] = mom;
}
";

// ── Dispatch helper ──────────────────────────────────────────────────────────

fn bench_kernel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    n_sites: u32,
    wg_size: u32,
    iterations: u32,
) -> std::time::Duration {
    let workgroups = (n_sites + wg_size - 1) / wg_size;
    let out_bytes = (n_sites as u64) * 4;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("qcd_bench"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_bytes,
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

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });

    // Warmup
    {
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
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

// ── Main ─────────────────────────────────────────────────────────────────────

struct KernelSpec {
    name: &'static str,
    op: &'static str,
    wgsl: &'static str,
    wg_size: u32,
    flops_per_site: u32,
    bytes_per_site: u32,
    silicon_note: &'static str,
}

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  QCD Silicon Benchmark");
    println!("  Per-kernel profiling × lattice size × GPU vendor");
    println!("═══════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    if adapters.is_empty() {
        eprintln!("No GPU adapters found.");
        std::process::exit(1);
    }

    let kernels = [
        KernelSpec {
            name: "gauge force",
            op: "qcd.force.su3",
            wgsl: SHADER_FORCE_PROXY,
            wg_size: 64,
            flops_per_site: 864,
            bytes_per_site: 4 * 18 * 5, // 4 neighbors + self, 18 f32 each
            silicon_note: "ALU-bound (shader_core) — 4 staple matmuls",
        },
        KernelSpec {
            name: "plaquette",
            op: "qcd.plaquette.wilson",
            wgsl: SHADER_PLAQUETTE_PROXY,
            wg_size: 64,
            flops_per_site: 1296,
            bytes_per_site: 4 * 18 * 4,
            silicon_note: "ALU-bound (shader_core) — 6 plane matmuls",
        },
        KernelSpec {
            name: "SU3 matmul",
            op: "qcd.matmul.su3",
            wgsl: SHADER_SU3_MATMUL,
            wg_size: 64,
            flops_per_site: 216,
            bytes_per_site: 4 * 18 * 2,
            silicon_note: "ALU-bound → tensor_core candidate (MMA-shaped)",
        },
        KernelSpec {
            name: "link update",
            op: "qcd.link_update.cayley",
            wgsl: SHADER_LINK_UPDATE_PROXY,
            wg_size: 64,
            flops_per_site: 400,
            bytes_per_site: 4 * 18 * 2,
            silicon_note: "ALU-bound (shader_core) — Cayley exp + reunitarize",
        },
        KernelSpec {
            name: "mom update",
            op: "qcd.momentum_update",
            wgsl: SHADER_MOMENTUM_UPDATE_PROXY,
            wg_size: 64,
            flops_per_site: 72,
            bytes_per_site: 4 * 18 * 3,
            silicon_note: "Memory-bound — simple P += dt*F, low arithmetic",
        },
        KernelSpec {
            name: "CG dot+reduce",
            op: "qcd.cg.dot_reduce",
            wgsl: SHADER_CG_DOT_PROXY,
            wg_size: 256,
            flops_per_site: 8,
            bytes_per_site: 4 * 2,
            silicon_note: "Shared-mem reduce (shader_core) — CG bottleneck",
        },
    ];

    // 4⁴ = 256, 8⁴ = 4096, 8³×16 = 8192, 16⁴ = 65536
    let lattice_sizes: &[(u32, &str)] = &[
        (256, "4^4"),
        (4096, "8^4"),
        (8192, "8^3x16"),
        (65536, "16^4"),
    ];

    let iterations = 200;
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

        for (n_sites, vol_name) in lattice_sizes {
            println!("  ── Volume {vol_name} ({n_sites} sites) ──\n");
            println!(
                "  {:<18} {:>10} {:>10} {:>10}  {}",
                "Kernel", "sites/s", "GFLOP/s", "GB/s", "Silicon note"
            );
            println!("  {}", "─".repeat(80));

            for k in &kernels {
                let elapsed = bench_kernel(
                    device, queue, k.wgsl, *n_sites, k.wg_size, iterations,
                );
                let sites_per_sec = f64::from(*n_sites) * f64::from(iterations)
                    / elapsed.as_secs_f64();
                let gflops = sites_per_sec * f64::from(k.flops_per_site) / 1e9;
                let gbps = sites_per_sec * f64::from(k.bytes_per_site) / 1e9;

                println!(
                    "  {:<18} {:>8.1}M {:>8.2} {:>8.2}  {}",
                    k.name,
                    sites_per_sec / 1e6,
                    gflops,
                    gbps,
                    k.silicon_note,
                );

                measurements.push(PerformanceMeasurement {
                    operation: format!("{}.v{n_sites}", k.op),
                    silicon_unit: "shader_core".into(),
                    precision_mode: "fp32_proxy".into(),
                    throughput_gflops: gflops,
                    tolerance_achieved: 0.0,
                    gpu_model: gpu.adapter_name.clone(),
                    measured_by: "hotSpring/bench_qcd_silicon".into(),
                    timestamp: ts,
                });
            }
            println!();
        }

        // Summary: compute intensity classification
        println!("  ── Silicon unit opportunity analysis ──\n");
        println!("  Kernel             Intensity    Bottleneck    Target Unit");
        println!("  {}", "─".repeat(60));
        for k in &kernels {
            let intensity = f64::from(k.flops_per_site) / f64::from(k.bytes_per_site);
            let bottleneck = if intensity > 3.0 { "compute" } else { "memory" };
            let target = if intensity > 3.0 {
                if k.name.contains("matmul") {
                    "tensor_core (MMA)"
                } else {
                    "shader_core (ALU)"
                }
            } else if k.name.contains("reduce") || k.name.contains("dot") {
                "shader_core (LDS)"
            } else {
                "shader_core (BW)"
            };
            println!(
                "  {:<18}  {intensity:>5.1} F/B    {bottleneck:<10}    {target}",
                k.name,
            );
        }
        println!();
    }

    // Report to toadStool
    println!("── Reporting {} measurements to toadStool ──\n", measurements.len());
    toadstool_report::report_to_toadstool(&measurements);
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("  QCD Silicon Benchmark Complete");
    println!("═══════════════════════════════════════════════════════════");
}
