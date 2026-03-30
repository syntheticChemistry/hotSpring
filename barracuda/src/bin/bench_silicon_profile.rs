// SPDX-License-Identifier: AGPL-3.0-only

//! Silicon Profile Builder: characterize every functional unit on each GPU.
//!
//! Runs targeted saturation micro-experiments for each silicon unit, measures
//! composition multipliers, then saves a persistent `SiliconProfile` JSON
//! that production runners can load for tier-based workload routing.
//!
//! ## What it measures
//!
//! Per GPU:
//! - FP32 ALU peak (FMA chain)
//! - DF64 ALU peak (Dekker arithmetic)
//! - Memory bandwidth (sequential vec4 reads)
//! - TMU peak (textureLoad flood)
//! - ROP peak (atomicAdd throughput)
//! - Shared memory / LDS (workgroup tree reduction)
//! - ALU + TMU composition multiplier
//!
//! ## Output
//!
//! - Pretty-printed profile with tier routing table to stdout
//! - JSON saved to `profiles/silicon/<adapter_name>.json`

use hotspring_barracuda::bench::silicon_profile::{SiliconProfile, SiliconUnit, from_spec_sheet};
use hotspring_barracuda::bench::telemetry::GpuTelemetry;
use hotspring_barracuda::gpu::GpuF64;

use std::path::PathBuf;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
//  WGSL kernels (proven in bench_silicon_saturation / composition)
// ═══════════════════════════════════════════════════════════════════

const SHADER_FMA_FP32: &str = r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var a: f32 = f32(gid.x + 1u) * 1.0001;
    var b: f32 = 0.9999;
    for (var i = 0u; i < 512u; i = i + 1u) {
        a = fma(a, b, a); a = fma(a, b, a);
        a = fma(a, b, a); a = fma(a, b, a);
    }
    out[gid.x] = a;
}
";

const SHADER_FMA_DF64: &str = r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var hi: f32 = f32(gid.x + 1u) * 1.0001;
    var lo: f32 = 0.0;
    let b_hi: f32 = 0.9999;
    let b_lo: f32 = 0.00001;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let p = hi * b_hi;
        let e = fma(hi, b_hi, -p);
        lo = fma(lo, b_hi, fma(hi, b_lo, e));
        hi = p;
        let s = hi + lo;
        lo = lo - (s - hi);
        hi = s;
    }
    out[gid.x * 2u] = hi;
    out[gid.x * 2u + 1u] = lo;
}
";

const SHADER_BW_SEQ: &str = r"
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = input[gid.x];
    out[gid.x] = v.x + v.y + v.z + v.w;
}
";

const SHADER_TMU_FLOOD: &str = r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var tex: texture_2d<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var sum: f32 = 0.0;
    let base_y = gid.x / 1024u;
    let base_x = gid.x % 1024u;
    for (var i = 0u; i < 64u; i = i + 1u) {
        let coord = vec2<i32>(i32((base_x + i * 16u) % 1024u), i32(base_y % 1024u));
        sum = sum + textureLoad(tex, coord, 0).r;
    }
    out[gid.x] = sum;
}
";

const SHADER_ATOMIC_FLOOD: &str = r"
@group(0) @binding(0) var<storage, read_write> counters: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bucket = gid.x % 256u;
    for (var i = 0u; i < 64u; i = i + 1u) {
        atomicAdd(&counters[(bucket + i) % 256u], 1u);
    }
}
";

const SHADER_REDUCE: &str = r"
var<workgroup> wg_data: array<f32, 1024>;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1024)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    wg_data[lid.x] = f32(gid.x + 1u) * 0.001;
    workgroupBarrier();
    for (var s = 512u; s > 0u; s = s >> 1u) {
        if lid.x < s {
            wg_data[lid.x] = wg_data[lid.x] + wg_data[lid.x + s];
        }
        workgroupBarrier();
    }
    if lid.x == 0u { out[gid.x / 1024u] = wg_data[0]; }
}
";

const SHADER_ALU_ONLY: &str = r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var a: f32 = f32(gid.x + 1u) * 0.001;
    for (var i = 0u; i < 256u; i = i + 1u) {
        a = fma(a, 0.9999, a * 0.0001);
    }
    out[gid.x] = a;
}
";

const SHADER_TMU_ONLY: &str = r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var tex: texture_2d<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var sum: f32 = 0.0;
    for (var i = 0u; i < 32u; i = i + 1u) {
        let coord = vec2<i32>(i32((gid.x + i * 7u) % 1024u), i32((gid.x / 1024u + i) % 1024u));
        sum = sum + textureLoad(tex, coord, 0).r;
    }
    out[gid.x] = sum;
}
";

const SHADER_ALU_PLUS_TMU: &str = r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var tex: texture_2d<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var a: f32 = f32(gid.x + 1u) * 0.001;
    var sum: f32 = 0.0;
    for (var i = 0u; i < 32u; i = i + 1u) {
        a = fma(a, 0.9999, a * 0.0001);
        a = fma(a, 0.9999, a * 0.0001);
        a = fma(a, 0.9999, a * 0.0001);
        a = fma(a, 0.9999, a * 0.0001);
        a = fma(a, 0.9999, a * 0.0001);
        a = fma(a, 0.9999, a * 0.0001);
        a = fma(a, 0.9999, a * 0.0001);
        a = fma(a, 0.9999, a * 0.0001);
        let coord = vec2<i32>(i32((gid.x + i * 7u) % 1024u), i32((gid.x / 1024u + i) % 1024u));
        sum = sum + textureLoad(tex, coord, 0).r;
    }
    out[gid.x] = a + sum;
}
";

// ═══════════════════════════════════════════════════════════════════
//  Dispatch infrastructure
// ═══════════════════════════════════════════════════════════════════

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
    workgroups: u32,
    iterations: u32,
) -> std::time::Duration {
    {
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(bg), &[]);
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
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(bg), &[]);
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

fn create_1k_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("profile_tex"),
        size: wgpu::Extent3d {
            width: 1024,
            height: 1024,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let data: Vec<f32> = (0..1024 * 1024).map(|i| (i as f32) * 0.001).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(1024 * 4),
            rows_per_image: Some(1024),
        },
        wgpu::Extent3d {
            width: 1024,
            height: 1024,
            depth_or_array_layers: 1,
        },
    );
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

// ═══════════════════════════════════════════════════════════════════
//  Power snapshot helpers
// ═══════════════════════════════════════════════════════════════════

/// Let GPU settle and return average idle power over a short window.
fn snapshot_idle_watts(telem: &GpuTelemetry) -> f64 {
    std::thread::sleep(std::time::Duration::from_millis(600));
    let mut sum = 0.0;
    let n = 3;
    for _ in 0..n {
        std::thread::sleep(std::time::Duration::from_millis(200));
        sum += telem.snapshot().power_w;
    }
    sum / n as f64
}

/// Sample average GPU power during a timed closure.
/// Spawns a sampling thread that reads the telemetry snapshot at ~100ms
/// intervals while the closure runs, then returns the average.
fn measure_loaded_watts(telem: &GpuTelemetry, work: impl FnOnce()) -> f64 {
    let state = telem.snapshot(); // warm read
    let _ = state;

    let samples: std::sync::Arc<std::sync::Mutex<Vec<f64>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    let s_clone = samples.clone();
    let d_clone = done.clone();
    let telem_ptr = telem as *const GpuTelemetry;
    // SAFETY: the telem reference outlives the thread (we join before returning)
    let telem_ref: &'static GpuTelemetry = unsafe { &*telem_ptr };

    let handle = std::thread::spawn(move || {
        while !d_clone.load(std::sync::atomic::Ordering::Relaxed) {
            let snap = telem_ref.snapshot();
            if snap.power_w > 0.0 {
                s_clone.lock().unwrap().push(snap.power_w);
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });

    work();

    done.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = handle.join();

    let readings = samples.lock().unwrap();
    if readings.is_empty() {
        0.0
    } else {
        readings.iter().sum::<f64>() / readings.len() as f64
    }
}

/// Run a device+queue benchmark wrapped with idle/loaded power capture.
/// Returns (benchmark_result, idle_watts, loaded_watts).
fn bench_with_energy(
    telem: &GpuTelemetry,
    bench_fn: impl FnOnce(&wgpu::Device, &wgpu::Queue) -> f64,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (f64, f64, f64) {
    let idle = snapshot_idle_watts(telem);
    let mut result = 0.0;
    let loaded = measure_loaded_watts(telem, || {
        result = bench_fn(device, queue);
    });
    (result, idle, loaded)
}

/// Same as `bench_with_energy` but also passes a texture view.
fn bench_with_energy_tex(
    telem: &GpuTelemetry,
    bench_fn: impl FnOnce(&wgpu::Device, &wgpu::Queue, &wgpu::TextureView) -> f64,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex_view: &wgpu::TextureView,
) -> (f64, f64, f64) {
    let idle = snapshot_idle_watts(telem);
    let mut result = 0.0;
    let loaded = measure_loaded_watts(telem, || {
        result = bench_fn(device, queue, tex_view);
    });
    (result, idle, loaded)
}

// ═══════════════════════════════════════════════════════════════════
//  Measurement routines — each returns the value to set on the profile
// ═══════════════════════════════════════════════════════════════════

fn measure_fp32_alu(device: &wgpu::Device, queue: &wgpu::Queue) -> f64 {
    let n: u32 = 262_144;
    let iters: u32 = 200;
    let flops_per_thread: u64 = 512 * 4 * 2;

    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: n as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgle_storage_rw(0)],
    });
    let pl = make_pipeline(device, SHADER_FMA_FP32, &bgl, "fp32_fma");
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buf.as_entire_binding(),
        }],
    });
    let elapsed = timed_dispatch(device, queue, &pl, &bg, n / 256, iters);
    let total = flops_per_thread * n as u64 * iters as u64;
    let tflops = total as f64 / elapsed.as_secs_f64() / 1e12;
    println!(
        "    FP32 ALU:   {tflops:>8.2} TFLOPS  ({:.1} ms)",
        elapsed.as_secs_f64() * 1000.0
    );
    tflops
}

fn measure_df64_alu(device: &wgpu::Device, queue: &wgpu::Queue) -> f64 {
    let n: u32 = 262_144;
    let iters: u32 = 200;
    let flops_per_thread: u64 = 256 * 10 * 2;

    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: n as u64 * 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgle_storage_rw(0)],
    });
    let pl = make_pipeline(device, SHADER_FMA_DF64, &bgl, "df64_fma");
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buf.as_entire_binding(),
        }],
    });
    let elapsed = timed_dispatch(device, queue, &pl, &bg, n / 256, iters);
    let total = flops_per_thread * n as u64 * iters as u64;
    let tflops = total as f64 / elapsed.as_secs_f64() / 1e12;
    println!(
        "    DF64 ALU:   {tflops:>8.2} TFLOPS  ({:.1} ms)",
        elapsed.as_secs_f64() * 1000.0
    );
    tflops
}

fn measure_memory_bw(device: &wgpu::Device, queue: &wgpu::Queue) -> f64 {
    let iters: u32 = 100;
    let size_mb: u64 = 256;
    let bytes = size_mb * 1024 * 1024;
    let n_vec4 = (bytes / 16) as u32;

    let in_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: n_vec4 as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgle_storage_ro(0), bgle_storage_rw(1)],
    });
    let pl = make_pipeline(device, SHADER_BW_SEQ, &bgl, "bw_seq");
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });
    let elapsed = timed_dispatch(device, queue, &pl, &bg, n_vec4.div_ceil(256), iters);
    let gbs = (bytes * iters as u64) as f64 / elapsed.as_secs_f64() / 1e9;
    println!(
        "    Mem BW:     {gbs:>8.1} GB/s    ({:.1} ms, {size_mb} MB)",
        elapsed.as_secs_f64() * 1000.0
    );
    gbs
}

fn measure_tmu(device: &wgpu::Device, queue: &wgpu::Queue, tex_view: &wgpu::TextureView) -> f64 {
    let n: u32 = 262_144;
    let iters: u32 = 200;

    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: n as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgle_storage_rw(0), bgle_texture(1)],
    });
    let pl = make_pipeline(device, SHADER_TMU_FLOOD, &bgl, "tmu_flood");
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(tex_view),
            },
        ],
    });
    let elapsed = timed_dispatch(device, queue, &pl, &bg, n / 256, iters);
    let total_texels = 64u64 * n as u64 * iters as u64;
    let gtexels = total_texels as f64 / elapsed.as_secs_f64() / 1e9;
    println!(
        "    TMU:        {gtexels:>8.1} GT/s    ({:.1} ms)",
        elapsed.as_secs_f64() * 1000.0
    );
    gtexels
}

fn measure_rop(device: &wgpu::Device, queue: &wgpu::Queue) -> f64 {
    let n: u32 = 262_144;
    let iters: u32 = 200;

    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgle_storage_rw(0)],
    });
    let pl = make_pipeline(device, SHADER_ATOMIC_FLOOD, &bgl, "atomic_flood");
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buf.as_entire_binding(),
        }],
    });
    let elapsed = timed_dispatch(device, queue, &pl, &bg, n / 256, iters);
    let total = 64u64 * n as u64 * iters as u64;
    let gatom = total as f64 / elapsed.as_secs_f64() / 1e9;
    println!(
        "    ROP/Atom:   {gatom:>8.1} Gatom/s ({:.1} ms)",
        elapsed.as_secs_f64() * 1000.0
    );
    gatom
}

fn measure_shared_mem(device: &wgpu::Device, queue: &wgpu::Queue) -> f64 {
    let n: u32 = 262_144;
    let iters: u32 = 200;
    let wg_size: u32 = 1024;
    let wg = n / wg_size;

    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: wg as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgle_storage_rw(0)],
    });
    let pl = make_pipeline(device, SHADER_REDUCE, &bgl, "reduce");
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buf.as_entire_binding(),
        }],
    });
    let elapsed = timed_dispatch(device, queue, &pl, &bg, wg, iters);
    let total_lds_bytes = 1023u64 * 2 * 4 * wg as u64 * iters as u64;
    let gbs = total_lds_bytes as f64 / elapsed.as_secs_f64() / 1e9;
    println!(
        "    LDS/Shared: {gbs:>8.1} GB/s    ({:.1} ms)",
        elapsed.as_secs_f64() * 1000.0
    );
    gbs
}

/// Returns (alu_only_ms, tmu_only_ms, compound_ms, multiplier).
fn measure_composition(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex_view: &wgpu::TextureView,
) -> (f64, f64, f64, f64) {
    let n: u32 = 262_144;
    let iters: u32 = 200;
    let wg = n / 256;

    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: n as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // ALU only
    let bgl_alu = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgle_storage_rw(0)],
    });
    let pl_alu = make_pipeline(device, SHADER_ALU_ONLY, &bgl_alu, "alu_only");
    let bg_alu = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_alu,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buf.as_entire_binding(),
        }],
    });
    let t_alu = timed_dispatch(device, queue, &pl_alu, &bg_alu, wg, iters);

    // TMU only
    let bgl_tex = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgle_storage_rw(0), bgle_texture(1)],
    });
    let pl_tmu = make_pipeline(device, SHADER_TMU_ONLY, &bgl_tex, "tmu_only");
    let bg_tmu = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_tex,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(tex_view),
            },
        ],
    });
    let t_tmu = timed_dispatch(device, queue, &pl_tmu, &bg_tmu, wg, iters);

    // ALU + TMU compound
    let pl_both = make_pipeline(device, SHADER_ALU_PLUS_TMU, &bgl_tex, "alu_tmu");
    let bg_both = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_tex,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(tex_view),
            },
        ],
    });
    let t_both = timed_dispatch(device, queue, &pl_both, &bg_both, wg, iters);

    let serial = t_alu + t_tmu;
    let mult = serial.as_secs_f64() / t_both.as_secs_f64();

    println!("    ALU+TMU composition:");
    println!("      ALU only:  {:.1} ms", t_alu.as_secs_f64() * 1000.0);
    println!("      TMU only:  {:.1} ms", t_tmu.as_secs_f64() * 1000.0);
    println!(
        "      Compound:  {:.1} ms  → {mult:.2}x multiplier",
        t_both.as_secs_f64() * 1000.0
    );

    (
        t_alu.as_secs_f64() * 1000.0,
        t_tmu.as_secs_f64() * 1000.0,
        t_both.as_secs_f64() * 1000.0,
        mult,
    )
}

// ═══════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════

fn characterize_gpu(gpu: &GpuF64, info: &wgpu::AdapterInfo) -> SiliconProfile {
    let device = gpu.device();
    let queue = gpu.queue();
    let tex_view = create_1k_texture(device, queue);

    let mut profile = from_spec_sheet(&info.name, info.vendor);

    let telem = GpuTelemetry::start(&info.name);
    println!("  Telemetry backend: {}\n", telem.backend);

    println!("  ── Saturation experiments (with energy) ──\n");

    // FP32 ALU
    let (fp32, idle, loaded) = bench_with_energy(&telem, measure_fp32_alu, device, queue);
    profile.set_measured(SiliconUnit::Fp32Alu, fp32);
    profile.set_measured_energy(SiliconUnit::Fp32Alu, idle, loaded);
    println!(
        "      Energy: idle={:.1}W loaded={:.1}W Δ={:.1}W",
        idle,
        loaded,
        (loaded - idle).max(0.0)
    );

    // DF64 ALU (not a separate unit entry but captures df64_tflops)
    let (df64, idle, loaded) = bench_with_energy(&telem, measure_df64_alu, device, queue);
    profile.df64_tflops = df64;
    println!(
        "      Energy: idle={:.1}W loaded={:.1}W Δ={:.1}W",
        idle,
        loaded,
        (loaded - idle).max(0.0)
    );

    // Memory Bandwidth
    let (bw, idle, loaded) = bench_with_energy(&telem, measure_memory_bw, device, queue);
    profile.set_measured(SiliconUnit::MemoryBandwidth, bw);
    profile.set_measured_energy(SiliconUnit::MemoryBandwidth, idle, loaded);
    println!(
        "      Energy: idle={:.1}W loaded={:.1}W Δ={:.1}W",
        idle,
        loaded,
        (loaded - idle).max(0.0)
    );

    // TMU
    let (tmu, idle, loaded) = bench_with_energy_tex(&telem, measure_tmu, device, queue, &tex_view);
    profile.set_measured(SiliconUnit::Tmu, tmu);
    profile.set_measured_energy(SiliconUnit::Tmu, idle, loaded);
    println!(
        "      Energy: idle={:.1}W loaded={:.1}W Δ={:.1}W",
        idle,
        loaded,
        (loaded - idle).max(0.0)
    );

    // ROP / Atomics
    let (rop, idle, loaded) = bench_with_energy(&telem, measure_rop, device, queue);
    profile.set_measured(SiliconUnit::Rop, rop);
    profile.set_measured_energy(SiliconUnit::Rop, idle, loaded);
    println!(
        "      Energy: idle={:.1}W loaded={:.1}W Δ={:.1}W",
        idle,
        loaded,
        (loaded - idle).max(0.0)
    );

    // Shared Memory / LDS
    let (lds, idle, loaded) = bench_with_energy(&telem, measure_shared_mem, device, queue);
    profile.set_measured(SiliconUnit::SharedMemory, lds);
    profile.set_measured_energy(SiliconUnit::SharedMemory, idle, loaded);
    println!(
        "      Energy: idle={:.1}W loaded={:.1}W Δ={:.1}W",
        idle,
        loaded,
        (loaded - idle).max(0.0)
    );

    println!("\n  ── Composition experiments (with energy) ──\n");

    let idle = snapshot_idle_watts(&telem);
    let ((alu_ms, tmu_ms, compound_ms, _mult), comp_loaded) = {
        let d = device;
        let q = queue;
        let tv = &tex_view;
        let mut result = (0.0, 0.0, 0.0, 0.0);
        let watts = measure_loaded_watts(&telem, || {
            result = measure_composition(d, q, tv);
        });
        (result, watts)
    };
    profile.add_composition_with_energy(
        SiliconUnit::Fp32Alu,
        SiliconUnit::Tmu,
        alu_ms + tmu_ms,
        compound_ms,
        idle,
        comp_loaded,
    );
    println!(
        "      Energy: idle={:.1}W compound={:.1}W Δ={:.1}W",
        idle,
        comp_loaded,
        (comp_loaded - idle).max(0.0)
    );

    profile.stamp_now();
    profile
}

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Silicon Profile Builder");
    println!("  Characterize every functional unit on each GPU");
    println!("  Fill → Route → Save");
    println!("═══════════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    if adapters.is_empty() {
        eprintln!("No GPU adapters found.");
        std::process::exit(1);
    }

    let save_dir =
        PathBuf::from(std::env::var("HOTSPRING_ROOT").unwrap_or_else(|_| ".".to_string()))
            .join("profiles")
            .join("silicon");

    let mut profiles: Vec<SiliconProfile> = Vec::new();

    for adapter in adapters {
        let info = adapter.get_info();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  {}", info.name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let gpu = match GpuF64::from_adapter(adapter).await {
            Ok(g) => g,
            Err(e) => {
                println!("  Skip: {e}\n");
                continue;
            }
        };

        let profile = characterize_gpu(&gpu, &info);

        println!();
        profile.print_summary();

        match profile.save(Some(&save_dir)) {
            Ok(path) => println!("\n  Saved → {}", path.display()),
            Err(e) => eprintln!("\n  Save failed: {e}"),
        }

        profiles.push(profile);
        println!();
    }

    // Cross-GPU comparison
    if profiles.len() >= 2 {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  Cross-GPU Silicon Comparison");
        println!("═══════════════════════════════════════════════════════════════\n");

        let a = &profiles[0];
        let b = &profiles[1];

        println!(
            "  {:<14} {:>14} {:>14}  {:>8}",
            "Unit",
            &a.adapter_name[..a.adapter_name.len().min(14)],
            &b.adapter_name[..b.adapter_name.len().min(14)],
            "Ratio"
        );
        println!("  {}", "─".repeat(56));

        let units = [
            SiliconUnit::Fp32Alu,
            SiliconUnit::Fp64Alu,
            SiliconUnit::Tmu,
            SiliconUnit::Rop,
            SiliconUnit::MemoryBandwidth,
            SiliconUnit::SharedMemory,
            SiliconUnit::TensorCore,
        ];
        for unit in units {
            let va = a.measured(unit);
            let vb = b.measured(unit);
            if va > 0.0 || vb > 0.0 {
                let ratio = if vb > 0.0 { va / vb } else { f64::INFINITY };
                println!("  {unit:<14} {va:>14.2} {vb:>14.2}  {ratio:>7.2}x");
            }
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!(
        "  Silicon Profile Builder Complete — {} GPUs characterized",
        profiles.len()
    );
    println!("  Profiles saved to: {}", save_dir.display());
    println!("═══════════════════════════════════════════════════════════════");
}
