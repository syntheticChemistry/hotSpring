// SPDX-License-Identifier: AGPL-3.0-or-later

//! WGSL proxy kernels and dispatch timing for QCD silicon benchmarks.

use std::time::Instant;

// ── FP32 proxy kernels ───────────────────────────────────────────────────────

pub const SHADER_FORCE_FP32: &str = r"
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

pub const SHADER_PLAQUETTE_FP32: &str = r"
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

pub const SHADER_SU3_MATMUL_FP32: &str = r"
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

pub const SHADER_LINK_UPDATE_FP32: &str = r"
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

pub const SHADER_MOM_UPDATE_FP32: &str = r"
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

pub const SHADER_CG_DOT_FP32: &str = r"
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

pub const SHADER_DIRAC_STENCIL_FP32: &str = r"
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

pub const SHADER_PSEUDOFERMION_FORCE_FP32: &str = r"
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

pub const SHADER_PRNG_BOXMULLER_FP32: &str = r"
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

pub const SHADER_POLYAKOV_FP32: &str = r"
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

pub const SHADER_GRADIENT_FLOW_FP32: &str = r"
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

pub const SHADER_FORCE_DF64: &str = r"
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

pub const SHADER_PLAQUETTE_DF64: &str = r"
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

pub const SHADER_CG_DOT_DF64: &str = r"
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

/// Time `iterations` dispatches of a WGSL compute kernel.
pub fn bench_kernel(
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
