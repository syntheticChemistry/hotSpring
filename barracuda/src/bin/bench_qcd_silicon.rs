// SPDX-License-Identifier: AGPL-3.0-only

//! QCD silicon benchmark: per-kernel profiling across GPUs, lattice sizes, and
//! precision tiers (FP32 proxy + DF64).
//!
//! Profiles QCD HMC kernels individually on every available GPU, from quenched
//! through dynamical (Dirac, pseudofermion force, PRNG) operations. Includes
//! 32^4 production scale and identifies silicon unit opportunities.
//!
//! ## What this measures
//!
//! For each (kernel, lattice_size, GPU) triple:
//! - Dispatch throughput (sites/sec)
//! - Effective GFLOP/s and GB/s
//! - Compute intensity (FLOP/byte)
//!
//! ## Kernel coverage
//!
//! Quenched: gauge force, plaquette, SU(3) matmul, link update, momentum update
//! Dynamical: Dirac stencil, CG dot+reduce, pseudofermion force, PRNG heat bath
//! Observables: Polyakov loop, gradient flow accumulate
//!
//! ## Scales
//!
//! 4^4 (256) through 32^4 (1M sites) — the full range from validation to production.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::toadstool_report::{self, PerformanceMeasurement};

use std::time::Instant;

// ── FP32 proxy kernels ───────────────────────────────────────────────────────

const SHADER_FORCE_FP32: &str = r"
// Gauge force: 4 staple matmuls (each 108 FMA) + TA projection
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = f32(gid.x + 1u) * 0.001;
    var force: f32 = 0.0;
    for (var s = 0u; s < 4u; s = s + 1u) {
        var prod: f32 = 0.0;
        for (var i = 0u; i < 108u; i = i + 1u) {
            prod = fma(base + f32(s * 108u + i) * 0.0001, base, prod);
        }
        force = force + prod;
    }
    out[gid.x] = force;
}
";

const SHADER_PLAQUETTE_FP32: &str = r"
// Plaquette: 6 oriented planes × 4 matmuls × trace
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = f32(gid.x + 1u) * 0.001;
    var plaq: f32 = 0.0;
    for (var p = 0u; p < 6u; p = p + 1u) {
        var trace: f32 = 0.0;
        for (var i = 0u; i < 108u; i = i + 1u) {
            trace = fma(base + f32(p * 108u + i) * 0.0001, base, trace);
        }
        plaq = plaq + trace;
    }
    out[gid.x] = plaq;
}
";

const SHADER_SU3_MATMUL_FP32: &str = r"
// SU(3) 3×3 complex matmul: 108 FMA (6 complex × 3 × 3 × 2 real ops)
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = f32(gid.x + 1u) * 0.001;
    var sum: f32 = 0.0;
    for (var i = 0u; i < 18u; i = i + 1u) {
        for (var j = 0u; j < 6u; j = j + 1u) {
            sum = fma(base + f32(i) * 0.01, base + f32(j) * 0.01, sum);
        }
    }
    out[gid.x] = sum;
}
";

const SHADER_LINK_UPDATE_FP32: &str = r"
// Link update: Cayley exp + reunitarize (~200 FMA + sqrt)
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = f32(gid.x + 1u) * 0.001;
    var link: f32 = base;
    for (var i = 0u; i < 200u; i = i + 1u) {
        link = fma(link, 0.999, base * 0.001);
    }
    link = link * inverseSqrt(link * link + 0.001);
    out[gid.x] = link;
}
";

const SHADER_MOM_UPDATE_FP32: &str = r"
// Momentum update: P += dt*F (36 f32 adds per link — memory-bound)
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = f32(gid.x + 1u) * 0.001;
    var mom: f32 = base;
    for (var i = 0u; i < 36u; i = i + 1u) {
        mom = mom + base * 0.01;
    }
    out[gid.x] = mom;
}
";

const SHADER_CG_DOT_FP32: &str = r"
// CG dot+reduce: complex dot product + workgroup tree reduction
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

// ── Dynamical fermion kernels ────────────────────────────────────────────────

const SHADER_DIRAC_STENCIL_FP32: &str = r"
// Staggered Dirac operator proxy: 4-direction stencil + U·ψ matvec per hop
// 8 hops × (18 real ops for U·ψ_3) ≈ 144 FMA + 8 memory reads
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = f32(gid.x + 1u) * 0.001;
    var result: f32 = 0.0;
    // 8 hops (±4 directions), each a 3×3 complex matvec
    for (var hop = 0u; hop < 8u; hop = hop + 1u) {
        var matvec: f32 = 0.0;
        for (var i = 0u; i < 18u; i = i + 1u) {
            let u = base + f32(hop * 18u + i) * 0.0001;
            let psi = base + f32(i) * 0.0002;
            matvec = fma(u, psi, matvec);
        }
        let phase = select(-1.0, 1.0, hop < 4u);
        result = result + phase * matvec;
    }
    out[gid.x] = result;
}
";

const SHADER_PSEUDOFERMION_FORCE_FP32: &str = r"
// Pseudofermion force: staples × X × Y† per link (~500 FMA)
// Most expensive per-link kernel in dynamical HMC
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = f32(gid.x + 1u) * 0.001;
    var force: f32 = 0.0;
    // 4 directions × staple(108 FMA) + X·Y†(18 FMA)
    for (var d = 0u; d < 4u; d = d + 1u) {
        var staple: f32 = 0.0;
        for (var i = 0u; i < 108u; i = i + 1u) {
            staple = fma(base + f32(d * 108u + i) * 0.0001, base, staple);
        }
        // outer product X × Y†
        for (var i = 0u; i < 18u; i = i + 1u) {
            force = fma(staple * 0.001, base + f32(i) * 0.01, force);
        }
    }
    out[gid.x] = force;
}
";

const SHADER_PRNG_BOXMULLER_FP32: &str = r"
// PRNG heat bath: PCG hash + Box-Muller (log + sin + cos)
// Transcendental-heavy — candidate for TMU table lookup acceleration
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var state = gid.x * 747796405u + 2891336453u;
    var sum: f32 = 0.0;
    for (var i = 0u; i < 18u; i = i + 1u) {
        // PCG step
        state = state * 747796405u + 2891336453u;
        let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        let u1 = f32((word >> 1u) ^ word) / 4294967296.0;
        state = state * 747796405u + 2891336453u;
        let word2 = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        let u2 = f32((word2 >> 1u) ^ word2) / 4294967296.0;
        // Box-Muller: sqrt(-2 ln u1) × cos(2π u2)
        let r = sqrt(-2.0 * log(max(u1, 0.0001)));
        sum = sum + r * cos(6.2831853 * u2);
    }
    out[gid.x] = sum;
}
";

const SHADER_POLYAKOV_FP32: &str = r"
// Polyakov loop: Nt products along temporal direction (serial chain)
// Latency-bound — long dependency chain of SU(3) matmuls
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = f32(gid.x + 1u) * 0.001;
    // Simulate Nt=16 temporal products
    var re: f32 = 1.0;
    var im: f32 = 0.0;
    for (var t = 0u; t < 16u; t = t + 1u) {
        let u_re = base + f32(t) * 0.01;
        let u_im = f32(t) * 0.001;
        // Complex multiply: (re + i*im) × (u_re + i*u_im)
        for (var c = 0u; c < 9u; c = c + 1u) {
            let new_re = fma(re, u_re, -(im * u_im));
            let new_im = fma(re, u_im, im * u_re);
            re = new_re;
            im = new_im;
        }
    }
    out[gid.x] = re;
}
";

const SHADER_GRADIENT_FLOW_FP32: &str = r"
// Gradient flow accumulate: V += epsilon * Z (algebra accumulation)
// Balanced compute/memory
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = f32(gid.x + 1u) * 0.001;
    var acc: f32 = base;
    for (var step = 0u; step < 8u; step = step + 1u) {
        for (var i = 0u; i < 18u; i = i + 1u) {
            acc = fma(0.01, base + f32(step * 18u + i) * 0.001, acc);
        }
    }
    out[gid.x] = acc;
}
";

// ── DF64 proxy kernels (Dekker double-single arithmetic on FP32 ALUs) ────────

const SHADER_FORCE_DF64: &str = r"
// Gauge force in DF64: ~4× FP32 ops for Dekker two_prod + two_sum
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base_hi = f32(gid.x + 1u) * 0.001;
    let base_lo: f32 = 0.0;
    var f_hi: f32 = 0.0;
    var f_lo: f32 = 0.0;
    for (var s = 0u; s < 4u; s = s + 1u) {
        for (var i = 0u; i < 54u; i = i + 1u) {
            let a = base_hi + f32(s * 54u + i) * 0.0001;
            // two_prod: p = a*a, e = fma(a,a,-p)
            let p = a * a;
            let e = fma(a, a, -p);
            // two_sum: s = f_hi + p
            let sum = f_hi + p;
            let v = sum - f_hi;
            f_lo = f_lo + ((f_hi - (sum - v)) + (p - v)) + e;
            f_hi = sum;
            // renormalize
            let s2 = f_hi + f_lo;
            f_lo = f_lo - (s2 - f_hi);
            f_hi = s2;
        }
    }
    out[gid.x * 2u] = f_hi;
    out[gid.x * 2u + 1u] = f_lo;
}
";

const SHADER_PLAQUETTE_DF64: &str = r"
// Plaquette in DF64: 6 planes × matmul chain with error tracking
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base_hi = f32(gid.x + 1u) * 0.001;
    var p_hi: f32 = 0.0;
    var p_lo: f32 = 0.0;
    for (var plane = 0u; plane < 6u; plane = plane + 1u) {
        var t_hi: f32 = 0.0;
        var t_lo: f32 = 0.0;
        for (var i = 0u; i < 54u; i = i + 1u) {
            let a = base_hi + f32(plane * 54u + i) * 0.0001;
            let prod = a * a;
            let err = fma(a, a, -prod);
            let s = t_hi + prod;
            let v = s - t_hi;
            t_lo = t_lo + ((t_hi - (s - v)) + (prod - v)) + err;
            t_hi = s;
        }
        let s = p_hi + t_hi;
        let v = s - p_hi;
        p_lo = p_lo + ((p_hi - (s - v)) + (t_hi - v)) + t_lo;
        p_hi = s;
    }
    out[gid.x * 2u] = p_hi;
    out[gid.x * 2u + 1u] = p_lo;
}
";

const SHADER_CG_DOT_DF64: &str = r"
// CG dot+reduce in DF64: error-tracked accumulation + workgroup reduce
var<workgroup> wg_hi: array<f32, 256>;
var<workgroup> wg_lo: array<f32, 256>;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let base = f32(gid.x + 1u) * 0.001;
    let prod = base * base;
    let err = fma(base, base, -prod);
    wg_hi[lid.x] = prod;
    wg_lo[lid.x] = err;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if lid.x < s {
            let sum = wg_hi[lid.x] + wg_hi[lid.x + s];
            let v = sum - wg_hi[lid.x];
            wg_lo[lid.x] = wg_lo[lid.x] + wg_lo[lid.x + s]
                + ((wg_hi[lid.x] - (sum - v)) + (wg_hi[lid.x + s] - v));
            wg_hi[lid.x] = sum;
        }
        workgroupBarrier();
    }
    if lid.x == 0u {
        out[(gid.x / 256u) * 2u] = wg_hi[0];
        out[(gid.x / 256u) * 2u + 1u] = wg_lo[0];
    }
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
    out_bytes_per_site: u32,
) -> std::time::Duration {
    let workgroups = n_sites.div_ceil(wg_size);
    let out_bytes = n_sites as u64 * out_bytes_per_site as u64;

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
    out_bytes_per_site: u32,
    precision: &'static str,
    phase: &'static str,
    silicon_note: &'static str,
}

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  QCD Silicon Benchmark v2");
    println!("  Quenched → Dynamical × FP32/DF64 × 4^4 → 32^4");
    println!("═══════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    if adapters.is_empty() {
        eprintln!("No GPU adapters found.");
        std::process::exit(1);
    }

    // -- FP32 kernels (quenched + dynamical + observables) --
    let fp32_kernels = [
        KernelSpec {
            name: "gauge force",
            op: "qcd.force.su3",
            wgsl: SHADER_FORCE_FP32,
            wg_size: 64,
            flops_per_site: 864,
            bytes_per_site: 4 * 18 * 5,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "quenched",
            silicon_note: "ALU-bound — 4 staple matmuls",
        },
        KernelSpec {
            name: "plaquette",
            op: "qcd.plaquette.wilson",
            wgsl: SHADER_PLAQUETTE_FP32,
            wg_size: 64,
            flops_per_site: 1296,
            bytes_per_site: 4 * 18 * 4,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "quenched",
            silicon_note: "ALU-bound — 6 plane matmuls",
        },
        KernelSpec {
            name: "SU3 matmul",
            op: "qcd.matmul.su3",
            wgsl: SHADER_SU3_MATMUL_FP32,
            wg_size: 64,
            flops_per_site: 216,
            bytes_per_site: 4 * 18 * 2,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "quenched",
            silicon_note: "ALU-bound → tensor_core candidate (MMA-shaped)",
        },
        KernelSpec {
            name: "link update",
            op: "qcd.link_update.cayley",
            wgsl: SHADER_LINK_UPDATE_FP32,
            wg_size: 64,
            flops_per_site: 400,
            bytes_per_site: 4 * 18 * 2,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "quenched",
            silicon_note: "ALU-bound — Cayley exp + reunitarize (sqrt)",
        },
        KernelSpec {
            name: "mom update",
            op: "qcd.momentum_update",
            wgsl: SHADER_MOM_UPDATE_FP32,
            wg_size: 64,
            flops_per_site: 72,
            bytes_per_site: 4 * 18 * 3,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "quenched",
            silicon_note: "Memory-bound — P += dt*F",
        },
        KernelSpec {
            name: "CG dot+reduce",
            op: "qcd.cg.dot_reduce",
            wgsl: SHADER_CG_DOT_FP32,
            wg_size: 256,
            flops_per_site: 8,
            bytes_per_site: 4 * 2,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "dynamical",
            silicon_note: "Shared-mem reduce — CG bottleneck",
        },
        KernelSpec {
            name: "Dirac stencil",
            op: "qcd.dirac.staggered",
            wgsl: SHADER_DIRAC_STENCIL_FP32,
            wg_size: 64,
            flops_per_site: 288,
            bytes_per_site: 4 * (18 * 8 + 6), // 8 link reads + 1 psi read
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "dynamical",
            silicon_note: "Balanced (stencil + matvec) — heart of CG",
        },
        KernelSpec {
            name: "pf force",
            op: "qcd.pseudofermion_force",
            wgsl: SHADER_PSEUDOFERMION_FORCE_FP32,
            wg_size: 64,
            flops_per_site: 1000,
            bytes_per_site: 4 * 18 * 6,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "dynamical",
            silicon_note: "ALU-bound — most expensive dynamical kernel",
        },
        KernelSpec {
            name: "PRNG heat bath",
            op: "qcd.prng.box_muller",
            wgsl: SHADER_PRNG_BOXMULLER_FP32,
            wg_size: 64,
            flops_per_site: 360,
            bytes_per_site: 4,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "dynamical",
            silicon_note: "Transcendental-heavy — TMU LUT candidate",
        },
        KernelSpec {
            name: "Polyakov loop",
            op: "qcd.polyakov.loop",
            wgsl: SHADER_POLYAKOV_FP32,
            wg_size: 64,
            flops_per_site: 576,
            bytes_per_site: 4 * 18 * 16,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "observable",
            silicon_note: "Latency-bound — serial Nt chain",
        },
        KernelSpec {
            name: "grad flow acc",
            op: "qcd.flow.accumulate",
            wgsl: SHADER_GRADIENT_FLOW_FP32,
            wg_size: 64,
            flops_per_site: 288,
            bytes_per_site: 4 * 18 * 2,
            out_bytes_per_site: 4,
            precision: "fp32",
            phase: "observable",
            silicon_note: "Balanced — algebra accumulation",
        },
    ];

    // -- DF64 kernels --
    let df64_kernels = [
        KernelSpec {
            name: "force (DF64)",
            op: "qcd.force.su3.df64",
            wgsl: SHADER_FORCE_DF64,
            wg_size: 64,
            flops_per_site: 864 * 4, // ~4× FP32 ops for Dekker
            bytes_per_site: 8 * 18 * 5,
            out_bytes_per_site: 8,
            precision: "df64",
            phase: "quenched",
            silicon_note: "DF64 ALU — AMD 1:16 advantage",
        },
        KernelSpec {
            name: "plaquette (DF64)",
            op: "qcd.plaquette.wilson.df64",
            wgsl: SHADER_PLAQUETTE_DF64,
            wg_size: 64,
            flops_per_site: 1296 * 4,
            bytes_per_site: 8 * 18 * 4,
            out_bytes_per_site: 8,
            precision: "df64",
            phase: "quenched",
            silicon_note: "DF64 ALU — higher precision trace",
        },
        KernelSpec {
            name: "CG dot (DF64)",
            op: "qcd.cg.dot_reduce.df64",
            wgsl: SHADER_CG_DOT_DF64,
            wg_size: 256,
            flops_per_site: 8 * 4,
            bytes_per_site: 8 * 2,
            out_bytes_per_site: 8,
            precision: "df64",
            phase: "dynamical",
            silicon_note: "DF64 reduce — error-compensated accumulation",
        },
    ];

    let lattice_sizes: &[(u32, &str)] = &[
        (256, "4^4"),
        (4096, "8^4"),
        (8192, "8^3x16"),
        (65536, "16^4"),
        (1_048_576, "32^4"),
    ];

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

        // Determine iteration count based on whether this is a software renderer
        let is_software = gpu.adapter_name.to_lowercase().contains("llvmpipe");

        for (n_sites, vol_name) in lattice_sizes {
            // Skip 32^4 on software renderer
            if is_software && *n_sites > 65536 {
                println!("  Skipping {vol_name} on software renderer\n");
                continue;
            }

            let iterations = if *n_sites >= 1_048_576 {
                20 // Fewer iterations for 32^4 (each dispatch is ~1M threads)
            } else if *n_sites >= 65536 {
                100
            } else {
                200
            };

            println!("  ── Volume {vol_name} ({n_sites} sites, {iterations} iters) ──\n");
            println!(
                "  {:<20} {:>10} {:>10} {:>10} {:>8}  Silicon",
                "Kernel", "sites/s", "GFLOP/s", "GB/s", "Phase"
            );
            println!("  {}", "─".repeat(90));

            // Run FP32 kernels
            for k in &fp32_kernels {
                let elapsed = bench_kernel(
                    device,
                    queue,
                    k.wgsl,
                    *n_sites,
                    k.wg_size,
                    iterations,
                    k.out_bytes_per_site,
                );
                let sites_per_sec =
                    f64::from(*n_sites) * f64::from(iterations) / elapsed.as_secs_f64();
                let gflops = sites_per_sec * f64::from(k.flops_per_site) / 1e9;
                let gbps = sites_per_sec * f64::from(k.bytes_per_site) / 1e9;

                println!(
                    "  {:<20} {:>8.1}M {:>8.2} {:>8.2} {:>8}  {}",
                    k.name,
                    sites_per_sec / 1e6,
                    gflops,
                    gbps,
                    k.phase,
                    k.silicon_note,
                );

                measurements.push(PerformanceMeasurement {
                    operation: format!("{}.v{n_sites}", k.op),
                    silicon_unit: "shader_core".into(),
                    precision_mode: k.precision.into(),
                    throughput_gflops: gflops,
                    tolerance_achieved: 0.0,
                    gpu_model: gpu.adapter_name.clone(),
                    measured_by: "hotSpring/bench_qcd_silicon".into(),
                    timestamp: ts,
                });
            }

            // Run DF64 kernels
            println!();
            for k in &df64_kernels {
                let elapsed = bench_kernel(
                    device,
                    queue,
                    k.wgsl,
                    *n_sites,
                    k.wg_size,
                    iterations,
                    k.out_bytes_per_site,
                );
                let sites_per_sec =
                    f64::from(*n_sites) * f64::from(iterations) / elapsed.as_secs_f64();
                let gflops = sites_per_sec * f64::from(k.flops_per_site) / 1e9;
                let gbps = sites_per_sec * f64::from(k.bytes_per_site) / 1e9;

                println!(
                    "  {:<20} {:>8.1}M {:>8.2} {:>8.2} {:>8}  {}",
                    k.name,
                    sites_per_sec / 1e6,
                    gflops,
                    gbps,
                    k.phase,
                    k.silicon_note,
                );

                measurements.push(PerformanceMeasurement {
                    operation: format!("{}.v{n_sites}", k.op),
                    silicon_unit: "shader_core".into(),
                    precision_mode: k.precision.into(),
                    throughput_gflops: gflops,
                    tolerance_achieved: 0.0,
                    gpu_model: gpu.adapter_name.clone(),
                    measured_by: "hotSpring/bench_qcd_silicon".into(),
                    timestamp: ts,
                });
            }
            println!();
        }

        // Silicon opportunity analysis
        println!("  ── Silicon Unit Opportunity Analysis ──\n");
        println!(
            "  {:<20} {:>8} {:>12} {:>12}  Opportunity",
            "Kernel", "F/B", "Bottleneck", "Target"
        );
        println!("  {}", "─".repeat(80));

        let all_kernels: Vec<&KernelSpec> =
            fp32_kernels.iter().chain(df64_kernels.iter()).collect();
        for k in &all_kernels {
            let intensity = f64::from(k.flops_per_site) / f64::from(k.bytes_per_site);
            let (bottleneck, target, opportunity) = classify_silicon_opportunity(k, intensity);
            println!(
                "  {:<20} {:>6.1} {:>12} {:>12}  {}",
                k.name, intensity, bottleneck, target, opportunity
            );
        }

        // HMC trajectory cost model at 32^4
        println!("\n  ── Estimated HMC Trajectory Cost (32^4, 40 MD steps) ──\n");
        print_trajectory_cost_model(&gpu.adapter_name);

        println!();
    }

    println!(
        "── Reporting {} measurements to toadStool ──\n",
        measurements.len()
    );
    toadstool_report::report_to_toadstool(&measurements);

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  QCD Silicon Benchmark v2 Complete");
    println!("═══════════════════════════════════════════════════════════");
}

fn classify_silicon_opportunity(
    k: &KernelSpec,
    intensity: f64,
) -> (&'static str, &'static str, &'static str) {
    if k.name.contains("PRNG") {
        return (
            "transcendental",
            "TMU + ALU",
            "TMU LUT for log/cos (1.9× measured on 3090)",
        );
    }
    if k.name.contains("Polyakov") {
        return (
            "latency",
            "shader_core",
            "ILP limited — serial dependency chain",
        );
    }
    if k.name.contains("CG dot") {
        return ("reduce", "shader LDS", "Workgroup shared memory throughput");
    }
    if k.name.contains("SU3 matmul") {
        return (
            "compute",
            "tensor_core",
            "MMA reshape via coralReef SASS (future)",
        );
    }
    if intensity > 3.0 {
        if k.precision == "df64" {
            (
                "compute",
                "shader_core",
                "AMD 1:16 FP64 advantage for error terms",
            )
        } else {
            ("compute", "shader_core", "Peak ALU utilization")
        }
    } else {
        ("memory", "cache/BW", "Infinity Cache advantage at ≤16^4")
    }
}

fn print_trajectory_cost_model(gpu_name: &str) {
    let n_md = 40u32;
    let n_sites = 1_048_576u64; // 32^4
    let n_links = n_sites * 4;

    // Per-step costs (FMA operations, estimated)
    let force_flops = n_links * 864;
    let link_update_flops = n_links * 400;
    let mom_update_flops = n_links * 72;
    // Omelyan: 3 force evals + 2 link updates + 3 mom updates per step
    let step_flops = 3 * force_flops + 2 * link_update_flops + 3 * mom_update_flops;
    let traj_flops = n_md as u64 * step_flops;

    // Dynamical adds CG solver (~100 CG iterations per force eval × Dirac cost)
    let cg_iters_per_force = 100u64;
    let dirac_flops = n_sites * 288;
    let cg_per_force = cg_iters_per_force * dirac_flops;
    let dynamical_overhead = 3 * n_md as u64 * cg_per_force;

    let quenched_tflops = traj_flops as f64 / 1e12;
    let dynamical_tflops = (traj_flops + dynamical_overhead) as f64 / 1e12;

    let is_3090 = gpu_name.to_lowercase().contains("3090");
    let peak_tflops = if is_3090 { 35.6 } else { 23.6 };
    let efficiency = 0.3; // 30% ALU utilization is realistic for stencil codes

    let quenched_time = quenched_tflops / (peak_tflops * efficiency);
    let dynamical_time = dynamical_tflops / (peak_tflops * efficiency);

    println!(
        "  Quenched:  {quenched_tflops:.2} TFLOP/traj → ~{quenched_time:.1}s at 30% efficiency on {gpu_name}"
    );
    println!(
        "  Dynamical: {dynamical_tflops:.2} TFLOP/traj → ~{dynamical_time:.1}s at 30% efficiency (Nf=4, ~100 CG iters)"
    );
    println!(
        "  Overnight (500 traj): ~{:.1}h quenched, ~{:.1}h dynamical",
        quenched_time * 500.0 / 3600.0,
        dynamical_time * 500.0 / 3600.0
    );
}
