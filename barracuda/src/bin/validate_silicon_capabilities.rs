// SPDX-License-Identifier: AGPL-3.0-only

//! Silicon capability diagnostic: characterize precision tier behavior per GPU.
//!
//! Probes each adapter for:
//! - f32 FMA two_prod correctness (Dekker error-free product)
//! - f32 workgroup tree reduction (baseline barrier test)
//! - DF64 scalar arithmetic in storage (no workgroup memory)
//! - DF64 workgroup tree reduction with f32 storage (isolates DF64 pattern)
//! - DF64 workgroup tree reduction with f64 storage (production pattern)
//! - `ReduceScalarPipeline` end-to-end (barracuda's production path)
//!
//! Results form the empirical capability matrix that `PrecisionRoutingAdvice`
//! should be built from. No per-card if/else — just probe and route.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::toadstool_report::{self, PerformanceMeasurement};
use hotspring_barracuda::validation::ValidationHarness;

use barracuda::pipeline::ReduceScalarPipeline;

use std::sync::Arc;
use wgpu::util::DeviceExt;

// ── Inline WGSL probe shaders ────────────────────────────────────────────────

const PROBE_F32_FMA: &str = "\
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    out[0] = fma(2.0, 3.0, 1.0);

    let a: f32 = 2.0;
    let b: f32 = 3.0;
    let p = a * b;
    out[1] = fma(a, b, -p);

    let c: f32 = 1234567.0;
    let d: f32 = 7654321.0;
    let q = c * d;
    out[2] = fma(c, d, -q);
    out[3] = q;
}
";

const PROBE_F32_WORKGROUP_REDUCE: &str = "\
struct Params { size: u32, _pad1: u32, _pad2: u32, _pad3: u32, }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let tid = lid.x;
    if (gid.x < params.size) {
        wg_data[tid] = input[gid.x];
    } else {
        wg_data[tid] = 0.0;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            wg_data[tid] = wg_data[tid] + wg_data[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[wg.x] = wg_data[0];
    }
}
";

const DF64_ARITH_PREAMBLE: &str = "\
struct Df64 { hi: f32, lo: f32, }

fn two_sum(a: f32, b: f32) -> Df64 {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return Df64(s, e);
}

fn two_prod(a: f32, b: f32) -> Df64 {
    let p = a * b;
    let e = fma(a, b, -p);
    return Df64(p, e);
}

fn df64_add(a: Df64, b: Df64) -> Df64 {
    let s = two_sum(a.hi, b.hi);
    let e = a.lo + b.lo;
    let v = two_sum(s.hi, s.lo + e);
    return v;
}

fn df64_mul(a: Df64, b: Df64) -> Df64 {
    let p = two_prod(a.hi, b.hi);
    let lo = p.lo + fma(a.hi, b.lo, a.lo * b.hi);
    let r = two_sum(p.hi, lo);
    return r;
}
";

const PROBE_DF64_STORAGE_ARITH_BODY: &str = "\
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let one = Df64(1.0, 0.0);
    let sum = df64_add(one, one);
    out[0] = sum.hi;
    out[1] = sum.lo;

    let three = Df64(3.0, 0.0);
    let prod = df64_mul(sum, three);
    out[2] = prod.hi;
    out[3] = prod.lo;

    let pi_hi: f32 = 3.1415927;
    let pi_lo: f32 = -8.742278e-8;
    let pi = Df64(pi_hi, pi_lo);
    let pi_sq = df64_mul(pi, pi);
    out[4] = pi_sq.hi;
    out[5] = pi_sq.lo;
}
";

const PROBE_DF64_WORKGROUP_REDUCE_F32_BODY: &str = "\
struct Params { size: u32, _pad1: u32, _pad2: u32, _pad3: u32, }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_hi: array<f32, 256>;
var<workgroup> shared_lo: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let tid = lid.x;
    if (gid.x < params.size) {
        shared_hi[tid] = input[gid.x];
        shared_lo[tid] = 0.0;
    } else {
        shared_hi[tid] = 0.0;
        shared_lo[tid] = 0.0;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let a = Df64(shared_hi[tid], shared_lo[tid]);
            let b = Df64(shared_hi[tid + stride], shared_lo[tid + stride]);
            let s = df64_add(a, b);
            shared_hi[tid] = s.hi;
            shared_lo[tid] = s.lo;
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[0] = shared_hi[0];
        output[1] = shared_lo[0];
    }
}
";

const DF64_F64_BRIDGE: &str = "\
fn df64_from_f64(v: f64) -> Df64 {
    let hi = f32(v);
    let lo = f32(v - f64(hi));
    return Df64(hi, lo);
}

fn df64_to_f64(v: Df64) -> f64 {
    return f64(v.hi) + f64(v.lo);
}
";

const PROBE_DF64_WORKGROUP_REDUCE_F64_BODY: &str = "\
struct Params { size: u32, _pad1: u32, _pad2: u32, _pad3: u32, }

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_hi: array<f32, 256>;
var<workgroup> shared_lo: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let tid = lid.x;
    if (gid.x < params.size) {
        let v = df64_from_f64(input[gid.x]);
        shared_hi[tid] = v.hi;
        shared_lo[tid] = v.lo;
    } else {
        shared_hi[tid] = 0.0;
        shared_lo[tid] = 0.0;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let a = Df64(shared_hi[tid], shared_lo[tid]);
            let b = Df64(shared_hi[tid + stride], shared_lo[tid + stride]);
            let s = df64_add(a, b);
            shared_hi[tid] = s.hi;
            shared_lo[tid] = s.lo;
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[wg.x] = df64_to_f64(Df64(shared_hi[0], shared_lo[0]));
    }
}
";

// ── Helpers ──────────────────────────────────────────────────────────────────

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes_to_f64(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn bgl_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

async fn readback_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Option<Vec<u8>> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe_staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(std::iter::once(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    match rx.recv() {
        Ok(Ok(())) => {
            let data = slice.get_mapped_range();
            let bytes = data.to_vec();
            drop(data);
            staging.unmap();
            Some(bytes)
        }
        _ => None,
    }
}

fn check_print(
    tag: &str,
    name: &str,
    got: f64,
    expected: f64,
    tol: f64,
    harness: &mut ValidationHarness,
) {
    let label = format!("{tag}: {name}");
    let err = (got - expected).abs();
    let pass = err < tol;
    let icon = if pass { "✓" } else { "✗" };
    println!("  {icon} {name} = {got:.6} (expected {expected:.6}, err={err:.2e})");
    harness.check_abs(&label, got, expected, tol);
}

// ── Probe 1: f32 FMA ────────────────────────────────────────────────────────

async fn probe_f32_fma(gpu: &GpuF64, tag: &str, harness: &mut ValidationHarness) {
    println!("── f32 FMA (two_prod pattern) ──");
    let device = gpu.device();
    let queue = gpu.queue();

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("probe_f32_fma"),
        source: wgpu::ShaderSource::Wgsl(PROBE_F32_FMA.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgl_storage(0, false)],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe_f32_fma"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if scope.pop().await.is_some() {
        println!("  ✗ COMPILATION FAILED");
        harness.check_bool(&format!("{tag}: f32_fma_compile"), false);
        return;
    }

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fma_out"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));

    let Some(bytes) = readback_bytes(device, queue, &out_buf, 16).await else {
        println!("  ✗ READBACK FAILED");
        harness.check_bool(&format!("{tag}: f32_fma_readback"), false);
        return;
    };
    let vals = bytes_to_f32(&bytes);

    check_print(tag, "fma(2,3,1)", f64::from(vals[0]), 7.0, 1e-6, harness);
    check_print(tag, "fma(2,3,-6) [exact product]", f64::from(vals[1]), 0.0, 1e-10, harness);

    let c = 1234567.0_f32;
    let d = 7654321.0_f32;
    let q_f32 = vals[3];
    let exact_product = f64::from(c) * f64::from(d);
    let expected_error = exact_product - f64::from(q_f32);
    check_print(
        tag,
        "fma two_prod error extraction",
        f64::from(vals[2]),
        expected_error,
        1.0,
        harness,
    );
    println!();
}

// ── Probe 2: f32 workgroup reduction ─────────────────────────────────────────

async fn probe_f32_workgroup_reduce(gpu: &GpuF64, tag: &str, harness: &mut ValidationHarness) {
    println!("── f32 workgroup reduction (barrier baseline) ──");
    let device = gpu.device();
    let queue = gpu.queue();
    let n: u32 = 256;

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("probe_f32_reduce"),
        source: wgpu::ShaderSource::Wgsl(PROBE_F32_WORKGROUP_REDUCE.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgl_storage(0, true), bgl_storage(1, false), bgl_uniform(2)],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe_f32_reduce"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(err) = scope.pop().await {
        println!("  ✗ COMPILATION FAILED: {err:?}");
        harness.check_bool(&format!("{tag}: f32_reduce_compile"), false);
        return;
    }

    let input_data: Vec<f32> = vec![1.0; n as usize];
    let input_bytes: &[u8] = bytemuck::cast_slice(&input_data);
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("f32_reduce_in"),
        contents: input_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("f32_reduce_out"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_data: [u32; 4] = [n, 0, 0, 0];
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("f32_reduce_params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));

    let Some(bytes) = readback_bytes(device, queue, &out_buf, 4).await else {
        println!("  ✗ READBACK FAILED");
        harness.check_bool(&format!("{tag}: f32_reduce_readback"), false);
        return;
    };
    let vals = bytes_to_f32(&bytes);

    check_print(tag, "f32 workgroup sum(256x1.0)", f64::from(vals[0]), 256.0, 0.01, harness);
    println!();
}

// ── Probe 3: DF64 storage arithmetic ─────────────────────────────────────────

async fn probe_df64_storage_arith(gpu: &GpuF64, tag: &str, harness: &mut ValidationHarness) {
    println!("── DF64 storage arithmetic (no workgroup memory) ──");
    let device = gpu.device();
    let queue = gpu.queue();

    let source = format!("{DF64_ARITH_PREAMBLE}\n{PROBE_DF64_STORAGE_ARITH_BODY}");

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("probe_df64_arith"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgl_storage(0, false)],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe_df64_arith"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if scope.pop().await.is_some() {
        println!("  ✗ COMPILATION FAILED");
        harness.check_bool(&format!("{tag}: df64_arith_compile"), false);
        return;
    }

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("df64_arith_out"),
        size: 24,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));

    let Some(bytes) = readback_bytes(device, queue, &out_buf, 24).await else {
        println!("  ✗ READBACK FAILED");
        harness.check_bool(&format!("{tag}: df64_arith_readback"), false);
        return;
    };
    let vals = bytes_to_f32(&bytes);

    let add_result = f64::from(vals[0]) + f64::from(vals[1]);
    check_print(tag, "df64_add(1,1) = 2", add_result, 2.0, 1e-10, harness);

    let mul_result = f64::from(vals[2]) + f64::from(vals[3]);
    check_print(tag, "df64_mul(2,3) = 6", mul_result, 6.0, 1e-10, harness);

    let pi_sq = f64::from(vals[4]) + f64::from(vals[5]);
    let pi_sq_ref = std::f64::consts::PI * std::f64::consts::PI;
    check_print(tag, "df64_mul(pi,pi) ~ 9.8696", pi_sq, pi_sq_ref, 1e-6, harness);
    println!();
}

// ── Probe 4: DF64 workgroup reduce (f32 storage) ────────────────────────────

async fn probe_df64_workgroup_reduce_f32(
    gpu: &GpuF64,
    tag: &str,
    harness: &mut ValidationHarness,
) {
    println!("── DF64 workgroup reduce (f32 storage, no f64 needed) ──");
    let device = gpu.device();
    let queue = gpu.queue();
    let n: u32 = 256;

    let source = format!("{DF64_ARITH_PREAMBLE}\n{PROBE_DF64_WORKGROUP_REDUCE_F32_BODY}");

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("probe_df64_wg_f32"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgl_storage(0, true), bgl_storage(1, false), bgl_uniform(2)],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe_df64_wg_f32"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if scope.pop().await.is_some() {
        println!("  ✗ COMPILATION FAILED");
        harness.check_bool(&format!("{tag}: df64_wg_f32_compile"), false);
        return;
    }

    let input_data: Vec<f32> = vec![1.0; n as usize];
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("df64_wg_f32_in"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("df64_wg_f32_out"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_data: [u32; 4] = [n, 0, 0, 0];
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("df64_wg_f32_params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));

    let Some(bytes) = readback_bytes(device, queue, &out_buf, 8).await else {
        println!("  ✗ READBACK FAILED");
        harness.check_bool(&format!("{tag}: df64_wg_f32_readback"), false);
        return;
    };
    let vals = bytes_to_f32(&bytes);
    let result = f64::from(vals[0]) + f64::from(vals[1]);

    check_print(
        tag,
        "DF64 workgroup sum(256x1.0) [f32 storage]",
        result,
        256.0,
        1e-10,
        harness,
    );
    println!();
}

// ── Probe 5: DF64 workgroup reduce (f64 storage — production pattern) ───────

async fn probe_df64_workgroup_reduce_f64(
    gpu: &GpuF64,
    tag: &str,
    harness: &mut ValidationHarness,
) {
    println!("── DF64 workgroup reduce (f64 storage — production pattern) ──");
    let device = gpu.device();
    let queue = gpu.queue();
    let n: u32 = 256;

    let source = format!(
        "{DF64_ARITH_PREAMBLE}\n{DF64_F64_BRIDGE}\n{PROBE_DF64_WORKGROUP_REDUCE_F64_BODY}"
    );

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("probe_df64_wg_f64"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgl_storage(0, true), bgl_storage(1, false), bgl_uniform(2)],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe_df64_wg_f64"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(err) = scope.pop().await {
        println!("  ✗ COMPILATION FAILED: {err:?}");
        harness.check_bool(&format!("{tag}: df64_wg_f64_compile"), false);
        return;
    }

    let input_data: Vec<f64> = vec![1.0; n as usize];
    let input_bytes: Vec<u8> = input_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("df64_wg_f64_in"),
        contents: &input_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("df64_wg_f64_out"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_data: [u32; 4] = [n, 0, 0, 0];
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("df64_wg_f64_params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));

    let Some(bytes) = readback_bytes(device, queue, &out_buf, 8).await else {
        println!("  ✗ READBACK FAILED");
        harness.check_bool(&format!("{tag}: df64_wg_f64_readback"), false);
        return;
    };
    let vals = bytes_to_f64(&bytes);

    check_print(
        tag,
        "DF64 workgroup sum(256x1.0) [f64 storage]",
        vals[0],
        256.0,
        1e-10,
        harness,
    );
    println!();
}

// ── Probe 6: ReduceScalarPipeline end-to-end ────────────────────────────────

async fn probe_reduce_pipeline(gpu: &GpuF64, tag: &str, harness: &mut ValidationHarness) {
    println!("── ReduceScalarPipeline end-to-end (production path) ──");

    if !gpu.device().features().contains(wgpu::Features::SHADER_F64) {
        println!("  ⊘ SKIPPED (no SHADER_F64)");
        return;
    }

    let device = gpu.device();
    let wdev = gpu.to_wgpu_device();

    let n = 1024_usize;

    let reducer = match ReduceScalarPipeline::new(Arc::clone(&wdev), n) {
        Ok(r) => r,
        Err(e) => {
            println!("  ✗ PIPELINE CONSTRUCTION FAILED: {e}");
            harness.check_bool(&format!("{tag}: reduce_pipeline_new"), false);
            return;
        }
    };

    let input_data: Vec<f64> = vec![1.0; n];
    let input_bytes: Vec<u8> = input_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reduce_pipeline_in"),
        contents: &input_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    match reducer.sum_f64(&input_buf) {
        Ok(result) => {
            check_print(
                tag,
                &format!("ReduceScalarPipeline sum_f64({n}x1.0)"),
                result,
                n as f64,
                1e-6,
                harness,
            );
        }
        Err(e) => {
            println!("  ✗ sum_f64 FAILED: {e}");
            harness.check_bool(&format!("{tag}: reduce_pipeline_sum"), false);
        }
    }

    let varied: Vec<f64> = (1..=512).map(|i| i as f64).collect();
    let varied_bytes: Vec<u8> = varied.iter().flat_map(|v| v.to_le_bytes()).collect();
    let varied_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reduce_pipeline_varied"),
        contents: &varied_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let reducer_512 = match ReduceScalarPipeline::new(Arc::clone(&wdev), 512) {
        Ok(r) => r,
        Err(e) => {
            println!("  ✗ PIPELINE CONSTRUCTION (512) FAILED: {e}");
            harness.check_bool(&format!("{tag}: reduce_pipeline_512_new"), false);
            return;
        }
    };

    let gauss_expected = 512.0 * 513.0 / 2.0;
    match reducer_512.sum_f64(&varied_buf) {
        Ok(result) => {
            check_print(
                tag,
                "ReduceScalarPipeline sum(1..512) [Gauss]",
                result,
                gauss_expected,
                1e-6,
                harness,
            );
        }
        Err(e) => {
            println!("  ✗ sum_f64 (Gauss) FAILED: {e}");
            harness.check_bool(&format!("{tag}: reduce_pipeline_gauss"), false);
        }
    }
    println!();
}

// ── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Silicon Capability Diagnostic");
    println!("  ecoPrimals/hotSpring — precision tier characterization");
    println!("═══════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    if adapters.is_empty() {
        eprintln!("No GPU adapters found.");
        std::process::exit(1);
    }

    println!("Found {} adapter(s):", adapters.len());
    for a in &adapters {
        let info = a.get_info();
        let f64_mark = if a.features().contains(wgpu::Features::SHADER_F64) {
            "✓ f64"
        } else {
            "✗ f64"
        };
        println!("  [{f64_mark}] {} ({:?})", info.name, info.backend);
    }
    println!();

    let mut harness = ValidationHarness::new("silicon_capabilities");

    for adapter in adapters {
        let info = adapter.get_info();
        let tag = format!("{} ({:?})", info.name, info.backend);
        println!("━━━ {tag} ━━━\n");

        let gpu = match GpuF64::from_adapter(adapter).await {
            Ok(g) => g,
            Err(e) => {
                println!("  ⚠ Could not create device: {e}");
                harness.check_bool(&format!("{tag}: device_creation"), false);
                println!();
                continue;
            }
        };

        gpu.print_info();
        println!();

        probe_f32_fma(&gpu, &tag, &mut harness).await;
        probe_f32_workgroup_reduce(&gpu, &tag, &mut harness).await;
        probe_df64_storage_arith(&gpu, &tag, &mut harness).await;
        probe_df64_workgroup_reduce_f32(&gpu, &tag, &mut harness).await;

        if gpu.has_f64 {
            probe_df64_workgroup_reduce_f64(&gpu, &tag, &mut harness).await;
        } else {
            println!("── DF64 workgroup reduce (f64 storage) ──");
            println!("  ⊘ SKIPPED (SHADER_F64 not functional)\n");
        }

        probe_reduce_pipeline(&gpu, &tag, &mut harness).await;
    }

    // ── Report to toadStool performance surface ────────────────────────────
    println!("── Reporting to toadStool ──\n");
    let ts = toadstool_report::epoch_now();
    let mut measurements: Vec<PerformanceMeasurement> = Vec::new();

    for check in &harness.checks {
        let parts: Vec<&str> = check.label.splitn(2, ": ").collect();
        if parts.len() < 2 {
            continue;
        }
        let gpu_model = parts[0].to_string();
        let operation = parts[1].to_string();

        let precision_mode = if operation.contains("df64_workgroup") {
            "df64_workgroup"
        } else if operation.contains("df64") {
            "df64_storage"
        } else if operation.contains("ReduceScalarPipeline") {
            "df64_production"
        } else if operation.contains("f32") {
            "f32"
        } else {
            "f64"
        };

        measurements.push(PerformanceMeasurement {
            operation,
            silicon_unit: "shader_core".into(),
            precision_mode: precision_mode.into(),
            throughput_gflops: 0.0,
            tolerance_achieved: (check.observed - check.expected).abs(),
            gpu_model,
            measured_by: "hotSpring/validate_silicon_capabilities".into(),
            timestamp: ts,
        });
    }

    toadstool_report::report_to_toadstool(&measurements);
    println!();

    harness.finish();
}
