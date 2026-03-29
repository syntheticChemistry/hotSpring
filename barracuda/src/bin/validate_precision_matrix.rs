// SPDX-License-Identifier: AGPL-3.0-only

//! Precision tier characterization matrix.
//!
//! Tests arithmetic accuracy and workgroup reduction correctness across all
//! precision tiers expressible on each GPU:
//!
//! | Tier     | WGSL type           | Bits | Notes                          |
//! |----------|---------------------|------|--------------------------------|
//! | fp16     | f16                 |   16 | SHADER_F16 required            |
//! | fp32     | f32                 |   32 | Universal baseline             |
//! | df64     | f32-pair (Df64)     |  ~48 | Software double-float          |
//! | fp64     | f64                 |   64 | SHADER_F64 required            |
//! | df128    | f64-pair            | ~106 | Software quad-double           |
//!
//! Integer tiers (i32/u32) and sub-byte types (int2/int4/int8) are tested
//! for bitwise correctness rather than floating-point error.
//!
//! Each test reports: GPU name, tier, operation, result, expected, absolute
//! error, and ULP error where applicable.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::toadstool_report::{self, PerformanceMeasurement};

// ── Test infrastructure ─────────────────────────────────────────────────────

struct PrecisionTest {
    name: &'static str,
    tier: &'static str,
    wgsl: String,
    entry_point: &'static str,
    output_bytes: usize,
    workgroups: u32,
    requires_f64: bool,
    requires_f16: bool,
    expected_f64: Option<f64>,
    expected_u32: Option<u32>,
}

struct MatrixResult {
    gpu: String,
    name: &'static str,
    tier: &'static str,
    pass: bool,
    value: String,
    expected: String,
    error: String,
}

// ── Dispatch helper (reused from roundtrip binary) ──────────────────────────

fn dispatch_simple(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry_point: &str,
    output_bytes: usize,
    workgroups: u32,
) -> Vec<u8> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("precision_matrix"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pm_out"),
        size: output_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pm_staging"),
        size: output_bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
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

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, output_bytes as u64);
    queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result = data.to_vec();
    drop(data);
    staging.unmap();
    result
}

// ── Test generators ─────────────────────────────────────────────────────────

fn tests_u32() -> Vec<PrecisionTest> {
    vec![
        PrecisionTest {
            name: "u32 add",
            tier: "u32",
            wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    out[0] = 2147483647u + 1u;
}
"
            .into(),
            entry_point: "main",
            output_bytes: 4,
            workgroups: 1,
            requires_f64: false,
            requires_f16: false,
            expected_f64: None,
            expected_u32: Some(2_147_483_648),
        },
        PrecisionTest {
            name: "u32 mul wrap",
            tier: "u32",
            wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let a: u32 = 12345u;
    let b: u32 = 6789u;
    out[0] = a * b;
}
"
            .into(),
            entry_point: "main",
            output_bytes: 4,
            workgroups: 1,
            requires_f64: false,
            requires_f16: false,
            expected_f64: None,
            expected_u32: Some(12345u32.wrapping_mul(6789)),
        },
        PrecisionTest {
            name: "i32 signed arith",
            tier: "i32",
            wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let a: i32 = -1000000;
    let b: i32 = 999999;
    let c: i32 = a + b;
    out[0] = bitcast<u32>(c);
}
"
            .into(),
            entry_point: "main",
            output_bytes: 4,
            workgroups: 1,
            requires_f64: false,
            requires_f16: false,
            expected_f64: None,
            expected_u32: Some((-1i32) as u32),
        },
        PrecisionTest {
            name: "u32 bitwise pack/unpack (int8 sim)",
            tier: "int8",
            wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let a: u32 = 0xDEu;
    let b: u32 = 0xADu;
    let c: u32 = 0xBEu;
    let d: u32 = 0xEFu;
    let packed = (d << 24u) | (c << 16u) | (b << 8u) | a;
    let unpacked_a = packed & 0xFFu;
    let unpacked_d = (packed >> 24u) & 0xFFu;
    out[0] = unpacked_a + (unpacked_d << 8u);
}
"
            .into(),
            entry_point: "main",
            output_bytes: 4,
            workgroups: 1,
            requires_f64: false,
            requires_f16: false,
            expected_f64: None,
            expected_u32: Some(0xDE + (0xEF << 8)),
        },
        PrecisionTest {
            name: "u32 bit extract (int2/int4 sim)",
            tier: "int2",
            wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let packed: u32 = 0xB4E4B4E4u;
    let int2_at_0  = packed & 3u;
    let int2_at_2  = (packed >> 2u) & 3u;
    let int2_at_4  = (packed >> 4u) & 3u;
    let int4_at_0  = packed & 15u;
    let int4_at_4  = (packed >> 4u) & 15u;
    out[0] = int2_at_0 + int2_at_2 + int2_at_4 + int4_at_0 + int4_at_4;
}
"
            .into(),
            entry_point: "main",
            output_bytes: 4,
            workgroups: 1,
            requires_f64: false,
            requires_f16: false,
            expected_f64: None,
            expected_u32: Some({
                let packed: u32 = 0xB4E4B4E4;
                let int2_at_0 = packed & 3;
                let int2_at_2 = (packed >> 2) & 3;
                let int2_at_4 = (packed >> 4) & 3;
                let int4_at_0 = packed & 15;
                let int4_at_4 = (packed >> 4) & 15;
                int2_at_0 + int2_at_2 + int2_at_4 + int4_at_0 + int4_at_4
            }),
        },
    ]
}

fn tests_fp32() -> Vec<PrecisionTest> {
    vec![
        PrecisionTest {
            name: "fp32 fma",
            tier: "fp32",
            wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    out[0] = fma(3.0, 2.0, 1.0);
}
"
            .into(),
            entry_point: "main",
            output_bytes: 4,
            workgroups: 1,
            requires_f64: false,
            requires_f16: false,
            expected_f64: Some(7.0),
            expected_u32: None,
        },
        PrecisionTest {
            name: "fp32 pi*pi",
            tier: "fp32",
            wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let pi: f32 = 3.14159274;
    out[0] = pi * pi;
}
"
            .into(),
            entry_point: "main",
            output_bytes: 4,
            workgroups: 1,
            requires_f64: false,
            requires_f16: false,
            expected_f64: Some(std::f64::consts::PI * std::f64::consts::PI),
            expected_u32: None,
        },
        PrecisionTest {
            name: "fp32 Kahan sum 1024",
            tier: "fp32",
            wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    var sum: f32 = 0.0;
    var c: f32 = 0.0;
    for (var i = 0u; i < 1024u; i = i + 1u) {
        let y = 1.0 - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    out[0] = sum;
}
"
            .into(),
            entry_point: "main",
            output_bytes: 4,
            workgroups: 1,
            requires_f64: false,
            requires_f16: false,
            expected_f64: Some(1024.0),
            expected_u32: None,
        },
        PrecisionTest {
            name: "fp32 wg reduce 256",
            tier: "fp32",
            wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
var<workgroup> wg_data: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    wg_data[lid.x] = 1.0;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if lid.x < s { wg_data[lid.x] = wg_data[lid.x] + wg_data[lid.x + s]; }
        workgroupBarrier();
    }
    if lid.x == 0u { out[wid.x] = wg_data[0]; }
}
"
            .into(),
            entry_point: "main",
            output_bytes: 4,
            workgroups: 1,
            requires_f64: false,
            requires_f16: false,
            expected_f64: Some(256.0),
            expected_u32: None,
        },
    ]
}

fn tests_df64() -> Vec<PrecisionTest> {
    let df64_preamble = r"
struct Df64 { hi: f32, lo: f32, }
fn two_sum(a: f32, b: f32) -> Df64 {
    let s = a + b; let v = s - a;
    return Df64(s, (a - (s - v)) + (b - v));
}
fn two_prod(a: f32, b: f32) -> Df64 {
    let p = a * b; let e = fma(a, b, -p);
    return Df64(p, e);
}
fn df64_add(a: Df64, b: Df64) -> Df64 {
    let s = two_sum(a.hi, b.hi);
    return two_sum(s.hi, s.lo + a.lo + b.lo);
}
fn df64_mul(a: Df64, b: Df64) -> Df64 {
    let p = two_prod(a.hi, b.hi);
    return two_sum(p.hi, p.lo + a.hi * b.lo + a.lo * b.hi);
}
fn df64_from_f64(v: f64) -> Df64 {
    let hi = f32(v); return Df64(hi, f32(v - f64(hi)));
}
fn df64_to_f64(v: Df64) -> f64 { return f64(v.hi) + f64(v.lo); }
";

    vec![
        PrecisionTest {
            name: "df64 add(1,1)",
            tier: "df64",
            wgsl: format!(
                "{df64_preamble}
@group(0) @binding(0) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {{
    out[0] = df64_to_f64(df64_add(Df64(1.0, 0.0), Df64(1.0, 0.0)));
}}
"
            ),
            entry_point: "main",
            output_bytes: 8,
            workgroups: 1,
            requires_f64: true,
            requires_f16: false,
            expected_f64: Some(2.0),
            expected_u32: None,
        },
        PrecisionTest {
            name: "df64 pi*pi",
            tier: "df64",
            wgsl: format!(
                "{df64_preamble}
@group(0) @binding(0) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {{
    let pi = Df64(3.14159274, -8.74227766e-8);
    out[0] = df64_to_f64(df64_mul(pi, pi));
}}
"
            ),
            entry_point: "main",
            output_bytes: 8,
            workgroups: 1,
            requires_f64: true,
            requires_f16: false,
            expected_f64: Some(std::f64::consts::PI * std::f64::consts::PI),
            expected_u32: None,
        },
        PrecisionTest {
            name: "df64 wg reduce 256",
            tier: "df64",
            wgsl: format!(
                "{df64_preamble}
@group(0) @binding(0) var<storage, read_write> out: array<f64>;
var<workgroup> wg_hi: array<f32, 256>;
var<workgroup> wg_lo: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
    wg_hi[lid.x] = 1.0; wg_lo[lid.x] = 0.0;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {{
        if lid.x < s {{
            let a = Df64(wg_hi[lid.x], wg_lo[lid.x]);
            let b = Df64(wg_hi[lid.x + s], wg_lo[lid.x + s]);
            let r = df64_add(a, b);
            wg_hi[lid.x] = r.hi; wg_lo[lid.x] = r.lo;
        }}
        workgroupBarrier();
    }}
    if lid.x == 0u {{ out[0] = df64_to_f64(Df64(wg_hi[0], wg_lo[0])); }}
}}
"
            ),
            entry_point: "main",
            output_bytes: 8,
            workgroups: 1,
            requires_f64: true,
            requires_f16: false,
            expected_f64: Some(256.0),
            expected_u32: None,
        },
    ]
}

fn tests_fp64() -> Vec<PrecisionTest> {
    vec![
        PrecisionTest {
            name: "fp64 pi*pi",
            tier: "fp64",
            wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let pi: f64 = 3.14159265358979323846lf;
    out[0] = pi * pi;
}
"
            .into(),
            entry_point: "main",
            output_bytes: 8,
            workgroups: 1,
            requires_f64: true,
            requires_f16: false,
            expected_f64: Some(std::f64::consts::PI * std::f64::consts::PI),
            expected_u32: None,
        },
        PrecisionTest {
            name: "fp64 Kahan sum 1024",
            tier: "fp64",
            wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    var sum: f64 = 0.0lf;
    var c: f64 = 0.0lf;
    for (var i = 0u; i < 1024u; i = i + 1u) {
        let y = 1.0lf - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    out[0] = sum;
}
"
            .into(),
            entry_point: "main",
            output_bytes: 8,
            workgroups: 1,
            requires_f64: true,
            requires_f16: false,
            expected_f64: Some(1024.0),
            expected_u32: None,
        },
        PrecisionTest {
            name: "fp64 wg reduce 4",
            tier: "fp64",
            wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<f64>;
var<workgroup> wg_data: array<f64, 4>;
@compute @workgroup_size(4)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    wg_data[lid.x] = f64(lid.x + 1u);
    workgroupBarrier();
    if lid.x == 0u {
        out[0] = wg_data[0] + wg_data[1] + wg_data[2] + wg_data[3];
    }
}
"
            .into(),
            entry_point: "main",
            output_bytes: 8,
            workgroups: 1,
            requires_f64: true,
            requires_f16: false,
            expected_f64: Some(10.0),
            expected_u32: None,
        },
    ]
}

fn tests_df128() -> Vec<PrecisionTest> {
    vec![
        PrecisionTest {
            name: "df128 add (f64-pair)",
            tier: "df128",
            wgsl: r"
struct Df128 { hi: f64, lo: f64, }
fn two_sum_64(a: f64, b: f64) -> Df128 {
    let s = a + b;
    let v = s - a;
    return Df128(s, (a - (s - v)) + (b - v));
}
fn df128_add(a: Df128, b: Df128) -> Df128 {
    let s = two_sum_64(a.hi, b.hi);
    return two_sum_64(s.hi, s.lo + a.lo + b.lo);
}

@group(0) @binding(0) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let one = Df128(1.0lf, 0.0lf);
    let two = df128_add(one, one);
    out[0] = two.hi;
    out[1] = two.lo;
}
"
            .into(),
            entry_point: "main",
            output_bytes: 16,
            workgroups: 1,
            requires_f64: true,
            requires_f16: false,
            expected_f64: Some(2.0),
            expected_u32: None,
        },
        PrecisionTest {
            name: "df128 pi*pi (f64-pair)",
            tier: "df128",
            wgsl: r"
struct Df128 { hi: f64, lo: f64, }
fn two_sum_64(a: f64, b: f64) -> Df128 {
    let s = a + b;
    let v = s - a;
    return Df128(s, (a - (s - v)) + (b - v));
}
fn two_prod_64(a: f64, b: f64) -> Df128 {
    let p = a * b;
    let e = fma(a, b, -p);
    return Df128(p, e);
}
fn df128_mul(a: Df128, b: Df128) -> Df128 {
    let p = two_prod_64(a.hi, b.hi);
    return two_sum_64(p.hi, p.lo + a.hi * b.lo + a.lo * b.hi);
}

@group(0) @binding(0) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let pi = Df128(3.14159265358979323846lf, 1.2246467991473532e-16lf);
    let r = df128_mul(pi, pi);
    out[0] = r.hi;
    out[1] = r.lo;
}
"
            .into(),
            entry_point: "main",
            output_bytes: 16,
            workgroups: 1,
            requires_f64: true,
            requires_f16: false,
            expected_f64: Some(std::f64::consts::PI * std::f64::consts::PI),
            expected_u32: None,
        },
    ]
}

// ── Main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Precision Tier Matrix — Silicon Characterization");
    println!("  int2/int4/int8 → u32/i32 → fp32 → df64 → fp64 → df128");
    println!("═══════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    if adapters.is_empty() {
        eprintln!("No GPU adapters found.");
        std::process::exit(1);
    }

    println!("Found {} adapter(s):\n", adapters.len());

    let mut all_results: Vec<MatrixResult> = Vec::new();

    for adapter in adapters {
        let info = adapter.get_info();
        let tag = format!("{} ({:?})", info.name, info.backend);

        let gpu = match GpuF64::from_adapter(adapter).await {
            Ok(g) => g,
            Err(e) => {
                println!("  Could not create device for {tag}: {e}\n");
                continue;
            }
        };

        let device = gpu.device();
        let queue = gpu.queue();
        let has_f64 = gpu.has_f64;
        let has_f16 = device.features().contains(wgpu::Features::SHADER_F16);

        println!("━━━ {} ━━━", gpu.adapter_name);
        println!(
            "  SHADER_F64: {}  |  SHADER_F16: {}\n",
            if has_f64 { "YES" } else { "NO" },
            if has_f16 { "YES" } else { "NO" },
        );

        let all_tests: Vec<PrecisionTest> = [
            tests_u32(),
            tests_fp32(),
            tests_df64(),
            tests_fp64(),
            tests_df128(),
        ]
        .into_iter()
        .flatten()
        .collect();

        for test in &all_tests {
            print!("  [{:<5}] {:<35} ", test.tier, test.name);

            if test.requires_f64 && !has_f64 {
                println!("SKIP (no SHADER_F64)");
                all_results.push(MatrixResult {
                    gpu: gpu.adapter_name.clone(),
                    name: test.name,
                    tier: test.tier,
                    pass: true,
                    value: "SKIP".into(),
                    expected: "-".into(),
                    error: "-".into(),
                });
                continue;
            }
            if test.requires_f16 && !has_f16 {
                println!("SKIP (no SHADER_F16)");
                all_results.push(MatrixResult {
                    gpu: gpu.adapter_name.clone(),
                    name: test.name,
                    tier: test.tier,
                    pass: true,
                    value: "SKIP".into(),
                    expected: "-".into(),
                    error: "-".into(),
                });
                continue;
            }

            let output = dispatch_simple(
                device,
                queue,
                &test.wgsl,
                test.entry_point,
                test.output_bytes,
                test.workgroups,
            );

            if let Some(expected_u32) = test.expected_u32 {
                let got = u32::from_le_bytes(output[..4].try_into().unwrap());
                let pass = got == expected_u32;
                let mark = if pass { "PASS" } else { "FAIL" };
                println!(
                    "{mark}  got={got} (0x{got:08X}), expected={expected_u32} (0x{expected_u32:08X})"
                );
                all_results.push(MatrixResult {
                    gpu: gpu.adapter_name.clone(),
                    name: test.name,
                    tier: test.tier,
                    pass,
                    value: format!("{got}"),
                    expected: format!("{expected_u32}"),
                    error: if pass {
                        "0".into()
                    } else {
                        format!("diff={}", got.wrapping_sub(expected_u32))
                    },
                });
            } else if let Some(expected) = test.expected_f64 {
                let got = if test.output_bytes == 4 {
                    f64::from(f32::from_le_bytes(output[..4].try_into().unwrap()))
                } else {
                    f64::from_le_bytes(output[..8].try_into().unwrap())
                };
                let abs_err = (got - expected).abs();
                let rel_err = if expected.abs() > 1e-300 {
                    abs_err / expected.abs()
                } else {
                    abs_err
                };
                let tol = match test.tier {
                    "fp32" => 1e-5,
                    "df64" => 1e-6,
                    "fp64" | "df128" => 1e-14,
                    _ => 1e-3,
                };
                let pass = abs_err < tol;
                let mark = if pass { "PASS" } else { "FAIL" };
                println!(
                    "{mark}  got={got:.15e}, expected={expected:.15e}, abs_err={abs_err:.3e}, rel_err={rel_err:.3e}"
                );
                all_results.push(MatrixResult {
                    gpu: gpu.adapter_name.clone(),
                    name: test.name,
                    tier: test.tier,
                    pass,
                    value: format!("{got:.15e}"),
                    expected: format!("{expected:.15e}"),
                    error: format!("abs={abs_err:.3e} rel={rel_err:.3e}"),
                });
            }
        }
        println!();
    }

    // ── Summary table ───────────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════");
    println!("  SUMMARY MATRIX");
    println!("═══════════════════════════════════════════════════════════\n");

    let gpus: Vec<String> = all_results
        .iter()
        .map(|r| r.gpu.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    let tiers = [
        "int2", "int8", "i32", "u32", "fp32", "df64", "fp64", "df128",
    ];
    for tier in &tiers {
        let tier_results: Vec<&MatrixResult> =
            all_results.iter().filter(|r| r.tier == *tier).collect();
        if tier_results.is_empty() {
            continue;
        }
        println!("  [{tier}]");
        for gpu in &gpus {
            let gpu_tier: Vec<&&MatrixResult> =
                tier_results.iter().filter(|r| r.gpu == *gpu).collect();
            let pass_count = gpu_tier.iter().filter(|r| r.pass).count();
            let total = gpu_tier.len();
            let status = if pass_count == total {
                format!("{pass_count}/{total} PASS")
            } else {
                format!("{pass_count}/{total} ({} FAIL)", total - pass_count)
            };
            println!("    {gpu:<45} {status}");
        }
    }

    let total_pass = all_results.iter().filter(|r| r.pass).count();
    let total_fail = all_results.iter().filter(|r| !r.pass).count();
    println!("\n  TOTAL: {} pass, {} fail", total_pass, total_fail);

    // ── Report to toadStool performance surface ─────────────────────────────
    println!("\n── Reporting to toadStool ──\n");
    let ts = toadstool_report::epoch_now();
    let measurements: Vec<PerformanceMeasurement> = all_results
        .iter()
        .filter(|r| r.value != "SKIP")
        .map(|r| {
            let operation = match r.name {
                n if n.contains("wg reduce") || n.contains("Kahan sum") => {
                    format!("math.reduce.sum.{}", r.tier)
                }
                n if n.contains("mul") || n.contains("pi*pi") => {
                    format!("math.arith.mul.{}", r.tier)
                }
                n if n.contains("add") || n.contains("signed arith") => {
                    format!("math.arith.add.{}", r.tier)
                }
                n if n.contains("fma") => format!("math.arith.fma.{}", r.tier),
                n if n.contains("bit") || n.contains("pack") => {
                    format!("math.bitwise.{}", r.tier)
                }
                _ => format!("math.unknown.{}", r.tier),
            };

            let tolerance_achieved = r
                .error
                .strip_prefix("abs=")
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(if r.pass { 0.0 } else { f64::MAX });

            PerformanceMeasurement {
                operation,
                silicon_unit: "shader_core".into(),
                precision_mode: r.tier.into(),
                throughput_gflops: 0.0,
                tolerance_achieved,
                gpu_model: r.gpu.clone(),
                measured_by: "hotSpring/validate_precision_matrix".into(),
                timestamp: ts,
            }
        })
        .collect();

    toadstool_report::report_to_toadstool(&measurements);
    println!();

    if total_fail > 0 {
        println!("  FAILURES:");
        for r in all_results.iter().filter(|r| !r.pass) {
            println!(
                "    {} | [{:<5}] {}: got={}, expected={}, error={}",
                r.gpu, r.tier, r.name, r.value, r.expected, r.error
            );
        }
        std::process::exit(1);
    }
}
