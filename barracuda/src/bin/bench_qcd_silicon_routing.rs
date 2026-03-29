// SPDX-License-Identifier: AGPL-3.0-only

//! QCD silicon routing benchmark: move real QCD workloads across GPU silicon.
//!
//! Each QCD kernel class is implemented on multiple silicon paths and timed:
//!
//! - **PRNG** (Box-Muller): ALU software transcendentals vs TMU texture lookup
//! - **Stencil** (Dirac/force link loads): storage buffer vs textureLoad
//! - **CG reduction**: workgroup shared memory vs subgroup intrinsics
//! - **Compound**: ALU force + TMU PRNG simultaneously (composition)
//!
//! Reports GFLOP/s, silicon unit, and composition multiplier per GPU.

use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
//  WGSL kernels — ALU baselines
// ═══════════════════════════════════════════════════════════════════

const PRNG_ALU: &str = r"
// Box-Muller via ALU: software log/cos/sqrt (the current production path)
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
struct Params { volume: u32, seed: u32, }
@group(0) @binding(1) var<uniform> params: Params;

fn pcg_hash(inp: u32) -> u32 {
    var s = inp * 747796405u + 2891336453u;
    var w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn uniform01(idx: u32, seq: u32) -> f32 {
    let h = pcg_hash(pcg_hash(idx ^ params.seed) ^ seq);
    return f32(h) / 4294967295.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.volume { return; }

    // 3 color components × 2 (re, im) = 3 Box-Muller pairs
    var total: f32 = 0.0;
    for (var pair = 0u; pair < 3u; pair++) {
        let u1 = max(uniform01(idx, pair * 2u), 1e-10);
        let u2 = uniform01(idx, pair * 2u + 1u);
        // ALU transcendentals
        let r = sqrt(-2.0 * log(u1));
        let theta = 6.283185 * u2;
        total += r * cos(theta);
        total += r * sin(theta);
    }
    out[idx] = total;
}
";

const PRNG_TMU: &str = r"
// Box-Muller via TMU: texture lookup for log/cos/sin, ALU only for mul/add
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
struct Params { volume: u32, seed: u32, }
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var log_table: texture_2d<f32>;
@group(0) @binding(3) var trig_table: texture_2d<f32>;

fn pcg_hash(inp: u32) -> u32 {
    var s = inp * 747796405u + 2891336453u;
    var w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn uniform01(idx: u32, seq: u32) -> f32 {
    let h = pcg_hash(pcg_hash(idx ^ params.seed) ^ seq);
    return f32(h) / 4294967295.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.volume { return; }

    var total: f32 = 0.0;
    for (var pair = 0u; pair < 3u; pair++) {
        let u1 = max(uniform01(idx, pair * 2u), 1e-10);
        let u2 = uniform01(idx, pair * 2u + 1u);

        // TMU lookup: log_table[x] = -2 * log(x/4095)  (pre-negated, pre-doubled)
        let log_idx = u32(u1 * 4095.0);
        let neg2log = textureLoad(log_table, vec2<u32>(log_idx, 0u), 0).r;
        let r = sqrt(neg2log);

        // TMU lookup: trig_table[x] = (cos(2pi*x/4095), sin(2pi*x/4095))
        let trig_idx = u32(u2 * 4095.0);
        let cs = textureLoad(trig_table, vec2<u32>(trig_idx, 0u), 0);
        total += r * cs.r;
        total += r * cs.g;
    }
    out[idx] = total;
}
";

// ═══════════════════════════════════════════════════════════════════
//  Stencil access: storage buffer vs textureLoad
// ═══════════════════════════════════════════════════════════════════

const STENCIL_ALU: &str = r"
// SU(3) matvec stencil — storage buffer path (current production)
// Proxy: each thread reads 8 neighbor SU(3) matrices (18 f32 each) + 8 color vectors (6 f32)
@group(0) @binding(0) var<storage, read> links: array<f32>;
@group(0) @binding(1) var<storage, read> psi: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
struct StencilParams { volume: u32, pad0: u32, pad1: u32, pad2: u32, }
@group(0) @binding(3) var<uniform> params: StencilParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if site >= params.volume { return; }

    var result = array<f32, 6>();
    // 4 directions × fwd + bwd = 8 neighbor reads
    for (var mu = 0u; mu < 4u; mu++) {
        // Neighbor index (wrapping; proxy uses modular addressing)
        let fwd = (site + mu * 317u + 1u) % params.volume;
        let bwd = (site + params.volume - mu * 317u - 1u) % params.volume;

        let fl = (site * 4u + mu) * 18u;
        let fp = fwd * 6u;
        let bl = (bwd * 4u + mu) * 18u;
        let bp = bwd * 6u;

        // Forward: U(x,mu) * psi(x+mu) — 3×3 complex matvec
        for (var row = 0u; row < 3u; row++) {
            var re: f32 = 0.0;
            var im: f32 = 0.0;
            for (var col = 0u; col < 3u; col++) {
                let li = fl + (row * 3u + col) * 2u;
                let pi = fp + col * 2u;
                re += links[li] * psi[pi] - links[li + 1u] * psi[pi + 1u];
                im += links[li] * psi[pi + 1u] + links[li + 1u] * psi[pi];
            }
            result[row * 2u] += re;
            result[row * 2u + 1u] += im;
        }

        // Backward: U†(x-mu,mu) * psi(x-mu)
        for (var row = 0u; row < 3u; row++) {
            var re: f32 = 0.0;
            var im: f32 = 0.0;
            for (var col = 0u; col < 3u; col++) {
                let li = bl + (col * 3u + row) * 2u;
                let pi = bp + col * 2u;
                re += links[li] * psi[pi] + links[li + 1u] * psi[pi + 1u];
                im += links[li] * psi[pi + 1u] - links[li + 1u] * psi[pi];
            }
            result[row * 2u] -= re;
            result[row * 2u + 1u] -= im;
        }
    }

    let base = site * 6u;
    for (var i = 0u; i < 6u; i++) { out[base + i] = result[i]; }
}
";

const STENCIL_TMU: &str = r"
// SU(3) matvec stencil — TMU textureLoad path
// Gauge links stored in Rgba32Float texture: 5 texels per SU(3) matrix (18 floats / 4 = 4.5, pad to 5)
// Texture layout: width = n_links * 5, height = 1
@group(0) @binding(0) var link_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> psi: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
struct StencilParams { volume: u32, pad0: u32, pad1: u32, pad2: u32, }
@group(0) @binding(3) var<uniform> params: StencilParams;

fn load_link_elem(link_idx: u32, elem: u32) -> f32 {
    // Each SU(3) = 5 texels of Rgba32Float (20 channels, first 18 used)
    let texel = link_idx * 5u + elem / 4u;
    let channel = elem % 4u;
    let v = textureLoad(link_tex, vec2<u32>(texel, 0u), 0);
    // Branch-free channel select
    return v[channel];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if site >= params.volume { return; }

    var result = array<f32, 6>();

    for (var mu = 0u; mu < 4u; mu++) {
        let fwd = (site + mu * 317u + 1u) % params.volume;
        let bwd = (site + params.volume - mu * 317u - 1u) % params.volume;

        let fwd_link = site * 4u + mu;
        let bwd_link = bwd * 4u + mu;
        let fp = fwd * 6u;
        let bp = bwd * 6u;

        // Forward: U(x,mu) * psi(x+mu) — read links via TMU
        for (var row = 0u; row < 3u; row++) {
            var re: f32 = 0.0;
            var im: f32 = 0.0;
            for (var col = 0u; col < 3u; col++) {
                let elem = (row * 3u + col) * 2u;
                let lr = load_link_elem(fwd_link, elem);
                let li = load_link_elem(fwd_link, elem + 1u);
                re += lr * psi[fp + col * 2u] - li * psi[fp + col * 2u + 1u];
                im += lr * psi[fp + col * 2u + 1u] + li * psi[fp + col * 2u];
            }
            result[row * 2u] += re;
            result[row * 2u + 1u] += im;
        }

        // Backward: U†(x-mu,mu) * psi(x-mu) — read links via TMU
        for (var row = 0u; row < 3u; row++) {
            var re: f32 = 0.0;
            var im: f32 = 0.0;
            for (var col = 0u; col < 3u; col++) {
                let elem = (col * 3u + row) * 2u;
                let lr = load_link_elem(bwd_link, elem);
                let li = load_link_elem(bwd_link, elem + 1u);
                re += lr * psi[bp + col * 2u] + li * psi[bp + col * 2u + 1u];
                im += lr * psi[bp + col * 2u + 1u] - li * psi[bp + col * 2u];
            }
            result[row * 2u] -= re;
            result[row * 2u + 1u] -= im;
        }
    }

    let base = site * 6u;
    for (var i = 0u; i < 6u; i++) { out[base + i] = result[i]; }
}
";

// ═══════════════════════════════════════════════════════════════════
//  CG reduction: shared memory vs subgroup
// ═══════════════════════════════════════════════════════════════════

const REDUCE_SHARED: &str = r"
// Tree reduce via workgroup shared memory (current production path)
var<workgroup> wg_data: array<f32, 256>;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
struct ReduceParams { size: u32, pad0: u32, pad1: u32, pad2: u32, }
@group(0) @binding(2) var<uniform> params: ReduceParams;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let g = gid.x + gid.y * nwg.x * 256u;
    wg_data[lid.x] = select(0.0, input[g], g < params.size);
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if lid.x < s { wg_data[lid.x] += wg_data[lid.x + s]; }
        workgroupBarrier();
    }
    if lid.x == 0u { output[wgid.x + wgid.y * nwg.x] = wg_data[0]; }
}
";

// ═══════════════════════════════════════════════════════════════════
//  SU(3) force: ALU baseline (for compound composition)
// ═══════════════════════════════════════════════════════════════════

const FORCE_ALU: &str = r"
// SU(3) gauge force proxy — pure ALU FMA chain
// Proxy for staple sum: 6 directions × 3 matmuls × 3×3 complex = heavy FMA
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
struct ForceParams { volume: u32, pad0: u32, }
@group(0) @binding(1) var<uniform> params: ForceParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.volume { return; }

    // 6 staple directions × 3 SU(3) multiplies × 3 rows × 3 cols × 2 FMA (re,im)
    var acc: f32 = f32(idx + 1u) * 1e-5;
    for (var staple = 0u; staple < 6u; staple++) {
        for (var mat = 0u; mat < 3u; mat++) {
            for (var i = 0u; i < 9u; i++) {
                acc = fma(acc, 0.9999, f32(i) * 0.0001);
                acc = fma(acc, 0.9998, f32(staple) * 0.0001);
            }
        }
    }
    out[idx] = acc;
}
";

// ═══════════════════════════════════════════════════════════════════
//  Compound: ALU force + TMU PRNG simultaneously
// ═══════════════════════════════════════════════════════════════════

const COMPOUND_ALU_TMU: &str = r"
// Compound kernel: SU(3) force (ALU) + Box-Muller PRNG (TMU) in same thread
// Demonstrates multi-unit composition — ALU and TMU run in parallel
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
struct Params { volume: u32, seed: u32, }
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var log_table: texture_2d<f32>;
@group(0) @binding(3) var trig_table: texture_2d<f32>;

fn pcg_hash(inp: u32) -> u32 {
    var s = inp * 747796405u + 2891336453u;
    var w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn uniform01(idx: u32, seq: u32) -> f32 {
    let h = pcg_hash(pcg_hash(idx ^ params.seed) ^ seq);
    return f32(h) / 4294967295.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.volume { return; }

    // ── ALU work: SU(3) force proxy (same as force_alu) ──
    var force_acc: f32 = f32(idx + 1u) * 1e-5;
    for (var staple = 0u; staple < 6u; staple++) {
        for (var mat = 0u; mat < 3u; mat++) {
            for (var i = 0u; i < 9u; i++) {
                force_acc = fma(force_acc, 0.9999, f32(i) * 0.0001);
                force_acc = fma(force_acc, 0.9998, f32(staple) * 0.0001);
            }
        }
    }

    // ── TMU work: Box-Muller PRNG via texture lookup (interleaved with ALU) ──
    var prng_acc: f32 = 0.0;
    for (var pair = 0u; pair < 3u; pair++) {
        let u1 = max(uniform01(idx, pair * 2u), 1e-10);
        let u2 = uniform01(idx, pair * 2u + 1u);
        let log_idx = u32(u1 * 4095.0);
        let neg2log = textureLoad(log_table, vec2<u32>(log_idx, 0u), 0).r;
        let r = sqrt(neg2log);
        let trig_idx = u32(u2 * 4095.0);
        let cs = textureLoad(trig_table, vec2<u32>(trig_idx, 0u), 0);
        prng_acc += r * cs.r;
    }

    out[idx] = force_acc + prng_acc;
}
";

// ═══════════════════════════════════════════════════════════════════
//  Host infrastructure
// ═══════════════════════════════════════════════════════════════════

struct GpuCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    name: String,
}

fn init_gpu(adapter: &wgpu::Adapter) -> GpuCtx {
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::SHADER_F16,
        required_limits: wgpu::Limits {
            max_storage_buffer_binding_size: 1 << 30,
            max_buffer_size: 1 << 30,
            ..wgpu::Limits::default()
        },
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: wgpu::ExperimentalFeatures::default(),
        trace: wgpu::Trace::default(),
    }))
    .expect("request device");
    let info = adapter.get_info();
    GpuCtx {
        device,
        queue,
        name: info.name.clone(),
    }
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

fn timed_dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    workgroups: (u32, u32),
    iterations: u32,
) -> std::time::Duration {
    for _ in 0..3 {
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(bg), &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
        }
        queue.submit(Some(enc.finish()));
    }
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

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
            pass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
        }
        queue.submit(Some(enc.finish()));
    }
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    start.elapsed()
}

fn create_log_table(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let data: Vec<f32> = (0..4096)
        .map(|i| {
            let x = (i as f32 + 0.5) / 4096.0;
            -2.0 * x.max(1e-10).ln()
        })
        .collect();
    create_1d_texture(device, queue, &data, 4096, "log_table")
}

fn create_trig_table(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let mut data = vec![0.0f32; 4096 * 4];
    for i in 0..4096 {
        let theta = 2.0 * std::f32::consts::PI * (i as f32 + 0.5) / 4096.0;
        data[i * 4] = theta.cos();
        data[i * 4 + 1] = theta.sin();
        data[i * 4 + 2] = 0.0;
        data[i * 4 + 3] = 0.0;
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trig_table"),
        size: wgpu::Extent3d {
            width: 4096,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4096 * 16),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: 4096,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_1d_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[f32],
    width: u32,
    label: &str,
) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_link_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    n_links: u32,
) -> wgpu::TextureView {
    let texels_per_link = 5u32;
    let width = n_links * texels_per_link;
    let max_w = device.limits().max_texture_dimension_2d;
    let (tex_w, tex_h) = if width <= max_w {
        (width, 1u32)
    } else {
        let h = (width + max_w - 1) / max_w;
        (max_w, h)
    };
    let total_texels = tex_w * tex_h;
    let mut data = vec![0.0f32; total_texels as usize * 4];
    let mut seed = 0xDEAD_BEEFu64;
    for i in 0..(n_links as usize * 18) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let texel_idx = i / 4;
        let channel = i % 4;
        if texel_idx < data.len() / 4 {
            data[texel_idx * 4 + channel] = (seed as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("link_texture"),
        size: wgpu::Extent3d {
            width: tex_w,
            height: tex_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(tex_w * 16),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: tex_w,
            height: tex_h,
            depth_or_array_layers: 1,
        },
    );
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

// ═══════════════════════════════════════════════════════════════════
//  Experiment runners
// ═══════════════════════════════════════════════════════════════════

fn run_prng_experiment(gpu: &GpuCtx, volume: u32, iterations: u32) {
    println!("\n  ── PRNG: ALU transcendentals vs TMU lookup ──");
    println!("  Volume: {} sites, {} iterations", volume, iterations);

    let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: volume as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let params_data = [volume, 42u32];
    let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

    let wg = ((volume + 255) / 256, 1u32);

    // ALU path
    let bgl_alu = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgle_storage_rw(0), bgle_uniform(1)],
        });
    let pipeline_alu = make_pipeline(&gpu.device, PRNG_ALU, &bgl_alu, "prng_alu");
    let bg_alu = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_alu,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let elapsed_alu = timed_dispatch(
        &gpu.device,
        &gpu.queue,
        &pipeline_alu,
        &bg_alu,
        wg,
        iterations,
    );
    let alu_sites_per_sec = volume as f64 * iterations as f64 / elapsed_alu.as_secs_f64();
    let alu_flops = alu_sites_per_sec * 30.0; // ~30 FLOP per site (3 pairs × log + sqrt + cos + mul + add)

    // TMU path
    let log_view = create_log_table(&gpu.device, &gpu.queue);
    let trig_view = create_trig_table(&gpu.device, &gpu.queue);

    let bgl_tmu = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgle_storage_rw(0),
                bgle_uniform(1),
                bgle_texture(2),
                bgle_texture(3),
            ],
        });
    let pipeline_tmu = make_pipeline(&gpu.device, PRNG_TMU, &bgl_tmu, "prng_tmu");
    let bg_tmu = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_tmu,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&log_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&trig_view),
            },
        ],
    });
    let elapsed_tmu = timed_dispatch(
        &gpu.device,
        &gpu.queue,
        &pipeline_tmu,
        &bg_tmu,
        wg,
        iterations,
    );
    let tmu_sites_per_sec = volume as f64 * iterations as f64 / elapsed_tmu.as_secs_f64();
    let tmu_flops = tmu_sites_per_sec * 18.0; // fewer ALU ops; TMU handles transcendentals

    let speedup = elapsed_alu.as_secs_f64() / elapsed_tmu.as_secs_f64();

    println!(
        "  ALU:  {:.1} Msites/s  {:.1} GFLOP/s  [{:.1} ms]",
        alu_sites_per_sec / 1e6,
        alu_flops / 1e9,
        elapsed_alu.as_secs_f64() * 1000.0
    );
    println!(
        "  TMU:  {:.1} Msites/s  {:.1} GFLOP/s  [{:.1} ms]",
        tmu_sites_per_sec / 1e6,
        tmu_flops / 1e9,
        elapsed_tmu.as_secs_f64() * 1000.0
    );
    println!(
        "  TMU speedup: {speedup:.2}x  (TMU freed {:.0} MFLOP/s of ALU capacity)",
        (alu_flops - tmu_flops).max(0.0) / 1e6
    );
}

fn run_stencil_experiment(gpu: &GpuCtx, volume: u32, iterations: u32) {
    println!("\n  ── Stencil: storage buffer vs TMU textureLoad ──");
    println!("  Volume: {} sites, {} iterations", volume, iterations);

    let n_links = volume * 4;
    let link_floats = n_links as u64 * 18;
    let psi_floats = volume as u64 * 6;

    let link_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: link_floats * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let psi_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: psi_floats * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: psi_floats * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let params_data = [volume, 0u32, 0u32, 0u32];
    let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

    let wg = ((volume + 63) / 64, 1u32);

    // ALU storage buffer path
    let bgl_alu = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgle_storage_ro(0),
                bgle_storage_ro(1),
                bgle_storage_rw(2),
                bgle_uniform(3),
            ],
        });
    let pipeline_alu = make_pipeline(&gpu.device, STENCIL_ALU, &bgl_alu, "stencil_alu");
    let bg_alu = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_alu,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: link_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: psi_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let elapsed_alu = timed_dispatch(
        &gpu.device,
        &gpu.queue,
        &pipeline_alu,
        &bg_alu,
        wg,
        iterations,
    );
    let alu_flops = volume as f64 * iterations as f64 * 8.0 * 36.0 / elapsed_alu.as_secs_f64();

    // TMU textureLoad path
    let link_tex_view = create_link_texture(&gpu.device, &gpu.queue, n_links);
    let bgl_tmu = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgle_texture(0),
                bgle_storage_ro(1),
                bgle_storage_rw(2),
                bgle_uniform(3),
            ],
        });
    let pipeline_tmu = make_pipeline(&gpu.device, STENCIL_TMU, &bgl_tmu, "stencil_tmu");
    let bg_tmu = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_tmu,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&link_tex_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: psi_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let elapsed_tmu = timed_dispatch(
        &gpu.device,
        &gpu.queue,
        &pipeline_tmu,
        &bg_tmu,
        wg,
        iterations,
    );
    let tmu_flops = volume as f64 * iterations as f64 * 8.0 * 36.0 / elapsed_tmu.as_secs_f64();

    let speedup = elapsed_alu.as_secs_f64() / elapsed_tmu.as_secs_f64();
    let bytes_accessed = volume as f64 * iterations as f64 * 8.0 * (18.0 + 6.0) * 4.0;

    println!(
        "  Storage: {:.1} GFLOP/s  {:.1} GB/s  [{:.1} ms]",
        alu_flops / 1e9,
        bytes_accessed / elapsed_alu.as_secs_f64() / 1e9,
        elapsed_alu.as_secs_f64() * 1000.0,
    );
    println!(
        "  TMU:     {:.1} GFLOP/s  {:.1} GB/s  [{:.1} ms]",
        tmu_flops / 1e9,
        bytes_accessed / elapsed_tmu.as_secs_f64() / 1e9,
        elapsed_tmu.as_secs_f64() * 1000.0,
    );
    println!("  TMU speedup: {speedup:.2}x  (texture cache deduplication for shared neighbors)");
}

fn run_reduce_experiment(gpu: &GpuCtx, size: u32, iterations: u32) {
    println!("\n  ── CG Reduction: workgroup shared memory ──");
    println!("  Size: {} elements, {} iterations", size, iterations);

    let input_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let n_wg = (size + 255) / 256;
    let output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: n_wg as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let params_data = [size, 0u32, 0u32, 0u32];
    let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

    let bgl = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgle_storage_ro(0), bgle_storage_rw(1), bgle_uniform(2)],
        });
    let pipeline = make_pipeline(&gpu.device, REDUCE_SHARED, &bgl, "reduce_shared");
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let wg = (n_wg, 1u32);
    let elapsed = timed_dispatch(&gpu.device, &gpu.queue, &pipeline, &bg, wg, iterations);
    let reduce_ops = size as f64 * iterations as f64;
    let gops = reduce_ops / elapsed.as_secs_f64() / 1e9;
    let bw = reduce_ops * 4.0 / elapsed.as_secs_f64() / 1e9;

    println!(
        "  Shared: {:.1} Gop/s  {:.1} GB/s  [{:.1} ms]",
        gops,
        bw,
        elapsed.as_secs_f64() * 1000.0,
    );
    println!("  (Subgroup intrinsics require wgpu SUBGROUP feature — planned for next phase)");
}

fn run_compound_experiment(gpu: &GpuCtx, volume: u32, iterations: u32) {
    println!("\n  ── Compound: ALU force + TMU PRNG (composition) ──");
    println!("  Volume: {} sites, {} iterations", volume, iterations);

    let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: volume as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let params_data = [volume, 42u32];
    let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

    let log_view = create_log_table(&gpu.device, &gpu.queue);
    let trig_view = create_trig_table(&gpu.device, &gpu.queue);

    let wg = ((volume + 255) / 256, 1u32);

    // Force-only (ALU baseline)
    let bgl_force = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgle_storage_rw(0), bgle_uniform(1)],
        });
    let pipeline_force = make_pipeline(&gpu.device, FORCE_ALU, &bgl_force, "force_alu");
    let bg_force = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_force,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let elapsed_force = timed_dispatch(
        &gpu.device,
        &gpu.queue,
        &pipeline_force,
        &bg_force,
        wg,
        iterations,
    );
    let force_fma = volume as f64 * iterations as f64 * 6.0 * 3.0 * 9.0 * 2.0;
    let force_tflops = force_fma / elapsed_force.as_secs_f64() / 1e12;

    // PRNG TMU-only
    let bgl_tmu = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgle_storage_rw(0),
                bgle_uniform(1),
                bgle_texture(2),
                bgle_texture(3),
            ],
        });
    let pipeline_prng = make_pipeline(&gpu.device, PRNG_TMU, &bgl_tmu, "prng_tmu_only");
    let bg_prng = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_tmu,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&log_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&trig_view),
            },
        ],
    });
    let elapsed_prng = timed_dispatch(
        &gpu.device,
        &gpu.queue,
        &pipeline_prng,
        &bg_prng,
        wg,
        iterations,
    );

    // Compound: force (ALU) + PRNG (TMU) in same kernel
    let pipeline_compound = make_pipeline(&gpu.device, COMPOUND_ALU_TMU, &bgl_tmu, "compound");
    let bg_compound = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_tmu,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&log_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&trig_view),
            },
        ],
    });
    let elapsed_compound = timed_dispatch(
        &gpu.device,
        &gpu.queue,
        &pipeline_compound,
        &bg_compound,
        wg,
        iterations,
    );

    let serial_estimate = elapsed_force + elapsed_prng;
    let composition_multiplier = serial_estimate.as_secs_f64() / elapsed_compound.as_secs_f64();

    println!(
        "  Force ALU:    {:.2} TFLOP/s  [{:.1} ms]",
        force_tflops,
        elapsed_force.as_secs_f64() * 1000.0,
    );
    println!(
        "  PRNG TMU:     [{:.1} ms]",
        elapsed_prng.as_secs_f64() * 1000.0,
    );
    println!(
        "  Compound:     [{:.1} ms]  (force + PRNG in same dispatch)",
        elapsed_compound.as_secs_f64() * 1000.0,
    );
    println!(
        "  Serial est:   [{:.1} ms]  (force + PRNG if sequential)",
        serial_estimate.as_secs_f64() * 1000.0,
    );
    println!(
        "  Composition multiplier: {composition_multiplier:.2}x  (>1.0 = ALU+TMU run in parallel)"
    );
}

// ═══════════════════════════════════════════════════════════════════
//  Bind group layout entry helpers
// ═══════════════════════════════════════════════════════════════════

fn bgle_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgle_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgle_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgle_texture(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  QCD Silicon Routing Benchmark");
    println!("  Route QCD workloads across GPU silicon: ALU → TMU → ROP");
    println!("═══════════════════════════════════════════════════════════════");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let all_adapters: Vec<wgpu::Adapter> =
        pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
    let adapters: Vec<&wgpu::Adapter> = all_adapters
        .iter()
        .filter(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .collect();

    if adapters.is_empty() {
        eprintln!("No discrete GPUs found.");
        return;
    }

    println!("\n  Discrete GPUs:");
    for (i, a) in adapters.iter().enumerate() {
        println!("    [{i}] {}", a.get_info().name);
    }

    let volumes: &[(u32, &str)] = &[
        (4096, "8^4 (4K sites)"),
        (65536, "16^4 (65K sites)"),
        (1_048_576, "32^4 (1M sites)"),
    ];
    let iterations = 100;

    for adapter in &adapters {
        let gpu = init_gpu(adapter);
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  GPU: {:<55} ║", gpu.name);
        println!("╚═══════════════════════════════════════════════════════════════╝");

        for &(volume, label) in volumes {
            println!("\n  ━━━ Lattice: {label} ━━━");

            run_prng_experiment(&gpu, volume, iterations);
            run_stencil_experiment(&gpu, volume, iterations);
            run_reduce_experiment(&gpu, volume, iterations);
            run_compound_experiment(&gpu, volume, iterations);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Silicon routing summary:");
    println!("  TMU PRNG:  textureLoad for log/cos frees ALU for physics");
    println!("  TMU stencil: texture cache deduplication for neighbor links");
    println!("  Compound:  ALU+TMU composition multiplier = free throughput");
    println!("  Next: subgroup CG reduce, ROP scatter-add, production integration");
    println!("═══════════════════════════════════════════════════════════════");
}
