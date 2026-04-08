// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sovereign compiler GPU-output round-trip verification.
//!
//! For each test shader, compiles through two paths:
//!   1. Raw WGSL  → wgpu create_shader_module → dispatch → readback
//!   2. Sovereign WGSL  → naga parse → optimize → re-emit → create_shader_module → dispatch → readback
//!
//! Asserts both paths produce identical GPU output. When they diverge, the
//! shader is a minimal reproducer for naga's WGSL round-trip bug.
//!
//! Escalates shader complexity to narrow the bug:
//!   Level 0: f32 scalar arithmetic (no workgroup memory)
//!   Level 1: f32 workgroup reduction (barriers)
//!   Level 2: f64 scalar arithmetic
//!   Level 3: f64 workgroup reduction (barriers + f64 storage)
//!   Level 4: DF64 f32-pair arithmetic (struct + helper functions)
//!   Level 5: DF64 workgroup reduction (the production pattern that broke)

use hotspring_barracuda::gpu::GpuF64;

use barracuda::device::capabilities::DeviceCapabilities;
use barracuda::shaders::sovereign::SovereignCompiler;

use wgpu::util::DeviceExt;

struct RoundtripTest {
    name: &'static str,
    wgsl: String,
    entry_point: &'static str,
    /// Number of f64 elements in the output buffer.
    output_count: usize,
    /// Number of workgroups to dispatch.
    workgroups: u32,
    /// Expected output values (from the raw WGSL path).
    /// If empty, we just compare raw vs sovereign output.
    expected: Vec<f64>,
}

// ── Level 0: f32 scalar arithmetic ──────────────────────────────────────────

fn level_0_f32_scalar() -> RoundtripTest {
    RoundtripTest {
        name: "L0: f32 scalar (no workgroup mem)",
        wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: f32 = 3.0;
    let y: f32 = x * 2.0 + 1.0;
    out[gid.x] = y;
}
"
        .into(),
        entry_point: "main",
        output_count: 1,
        workgroups: 1,
        expected: vec![7.0],
    }
}

// ── Level 1: f32 workgroup reduction ────────────────────────────────────────

fn level_1_f32_workgroup() -> RoundtripTest {
    RoundtripTest {
        name: "L1: f32 workgroup reduce (barriers)",
        wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    wg_data[tid] = 1.0;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            wg_data[tid] = wg_data[tid] + wg_data[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        out[wid.x] = wg_data[0];
    }
}
"
        .into(),
        entry_point: "main",
        output_count: 1,
        workgroups: 1,
        expected: vec![256.0],
    }
}

// ── Level 2: f64 scalar arithmetic ──────────────────────────────────────────

fn level_2_f64_scalar() -> RoundtripTest {
    RoundtripTest {
        name: "L2: f64 scalar (no workgroup mem)",
        wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<f64>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: f64 = f64(3.0);
    let y: f64 = x * f64(2.0) + f64(1.0);
    out[gid.x] = y;
}
"
        .into(),
        entry_point: "main",
        output_count: 1,
        workgroups: 1,
        expected: vec![7.0],
    }
}

// ── Level 3: f64 workgroup reduction ────────────────────────────────────────

fn level_3_f64_workgroup() -> RoundtripTest {
    RoundtripTest {
        name: "L3: f64 workgroup reduce (barriers + f64 storage)",
        wgsl: r"
@group(0) @binding(0) var<storage, read_write> out: array<f64>;

var<workgroup> wg_data: array<f64, 4>;

@compute @workgroup_size(4)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    wg_data[lid.x] = f64(lid.x + 1u);
    workgroupBarrier();

    if lid.x == 0u {
        out[0] = wg_data[0] + wg_data[1] + wg_data[2] + wg_data[3];
    }
}
"
        .into(),
        entry_point: "main",
        output_count: 1,
        workgroups: 1,
        expected: vec![10.0],
    }
}

// ── Level 4: DF64 f32-pair arithmetic ───────────────────────────────────────

fn level_4_df64_scalar() -> RoundtripTest {
    RoundtripTest {
        name: "L4: DF64 scalar arith (struct + helpers, f64 output)",
        wgsl: r"
struct Df64 { hi: f32, lo: f32, }

fn two_sum(a: f32, b: f32) -> Df64 {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return Df64(s, e);
}

fn df64_add(a: Df64, b: Df64) -> Df64 {
    let s = two_sum(a.hi, b.hi);
    let e = a.lo + b.lo;
    let v = two_sum(s.hi, s.lo + e);
    return v;
}

fn df64_to_f64(v: Df64) -> f64 {
    return f64(v.hi) + f64(v.lo);
}

@group(0) @binding(0) var<storage, read_write> out: array<f64>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) _id: vec3<u32>) {
    let one = Df64(1.0, 0.0);
    let two = df64_add(one, one);
    out[0] = df64_to_f64(two);
}
"
        .into(),
        entry_point: "main",
        output_count: 1,
        workgroups: 1,
        expected: vec![2.0],
    }
}

// ── Level 5: DF64 workgroup reduction (production pattern) ──────────────────

fn level_5_df64_workgroup() -> RoundtripTest {
    RoundtripTest {
        name: "L5: DF64 workgroup reduce (f32 shared, f64 I/O — production pattern)",
        wgsl: r"
struct Df64 { hi: f32, lo: f32, }

fn two_sum(a: f32, b: f32) -> Df64 {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return Df64(s, e);
}

fn df64_add(a: Df64, b: Df64) -> Df64 {
    let s = two_sum(a.hi, b.hi);
    let e = a.lo + b.lo;
    let v = two_sum(s.hi, s.lo + e);
    return v;
}

fn df64_from_f64(v: f64) -> Df64 {
    let hi = f32(v);
    let lo = f32(v - f64(hi));
    return Df64(hi, lo);
}

fn df64_to_f64(v: Df64) -> f64 {
    return f64(v.hi) + f64(v.lo);
}

struct ReduceParams {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: ReduceParams;

var<workgroup> shared_hi: array<f32, 256>;
var<workgroup> shared_lo: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_reduce_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        let v = df64_from_f64(input[gid]);
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
        output[workgroup_id.x] = df64_to_f64(Df64(shared_hi[0], shared_lo[0]));
    }
}
"
        .into(),
        entry_point: "sum_reduce_f64",
        output_count: 1,
        workgroups: 1,
        expected: vec![256.0],
    }
}

// ── Dispatch helper ─────────────────────────────────────────────────────────

fn dispatch_shader(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    module: &wgpu::ShaderModule,
    test: &RoundtripTest,
) -> Vec<u8> {
    let output_size = (test.output_count * 8) as u64;
    let uses_f64_output = test.wgsl.contains("array<f64>");
    let element_size: u64 = if uses_f64_output { 8 } else { 4 };
    let buf_size = test.output_count as u64 * element_size;

    let needs_input = test.wgsl.contains("var<storage, read> input");
    let needs_params = test.wgsl.contains("var<uniform> params");

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("roundtrip_out"),
        size: buf_size.max(output_size),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("roundtrip_staging"),
        size: buf_size.max(output_size),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: if needs_input {
                wgpu::BufferBindingType::Storage { read_only: true }
            } else {
                wgpu::BufferBindingType::Storage { read_only: false }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];

    let input_buf = if needs_input {
        entries[0].ty = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        let data: Vec<f64> = vec![1.0; 256];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("roundtrip_input"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE,
            }),
        )
    } else {
        None
    };

    let params_buf = if needs_params {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: if needs_input { 2 } else { 1 },
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        let params_data: [u32; 4] = [256, 0, 0, 0];
        let params_bytes: &[u8] = bytemuck::cast_slice(&params_data);
        Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("roundtrip_params"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            }),
        )
    } else {
        None
    };

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(test.name),
        layout: Some(&pl),
        module,
        entry_point: Some(test.entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let mut bg_entries: Vec<wgpu::BindGroupEntry> = Vec::new();

    if needs_input {
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: input_buf.as_ref().unwrap().as_entire_binding(),
        });
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: out_buf.as_entire_binding(),
        });
    } else {
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        });
    }

    if let Some(ref pb) = params_buf {
        bg_entries.push(wgpu::BindGroupEntry {
            binding: if needs_input { 2 } else { 1 },
            resource: pb.as_entire_binding(),
        });
    }

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &bg_entries,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(test.workgroups, 1, 1);
    }
    let copy_size = buf_size.max(output_size);
    enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, copy_size);
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

// ── Per-GPU test runner ─────────────────────────────────────────────────────

fn run_tests_on_gpu(
    gpu: &GpuF64,
    compiler: &SovereignCompiler,
    tests: &[RoundtripTest],
) -> (usize, usize, usize, Option<String>) {
    let device = gpu.device();
    let queue = gpu.queue();
    let mut pass = 0;
    let mut fail = 0;
    let mut skip = 0;
    let mut first_failure: Option<String> = None;

    for test in tests {
        print!("  {:<60} ", test.name);

        let needs_f64 = test.wgsl.contains("f64");
        if needs_f64 && !gpu.has_f64 {
            println!("SKIP (no SHADER_F64)");
            skip += 1;
            continue;
        }

        // Path 1: Raw WGSL → wgpu
        let raw_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raw"),
            source: wgpu::ShaderSource::Wgsl(test.wgsl.as_str().into()),
        });
        let raw_output = dispatch_shader(device, queue, &raw_module, test);

        // Path 2: Sovereign WGSL round-trip → wgpu
        let sovereign_result = compiler.compile_to_wgsl(&test.wgsl);
        match sovereign_result {
            Ok((sovereign_wgsl, stats)) => {
                let sovereign_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("sovereign"),
                    source: wgpu::ShaderSource::Wgsl(sovereign_wgsl.as_str().into()),
                });
                let sovereign_output = dispatch_shader(device, queue, &sovereign_module, test);

                if raw_output == sovereign_output {
                    println!(
                        "PASS  (fma={}, dead={})",
                        stats.fma_fusions, stats.dead_exprs_eliminated
                    );
                    pass += 1;

                    if !test.expected.is_empty() && test.wgsl.contains("array<f64>") {
                        let raw_f64s = bytes_to_f64s(&raw_output);
                        for (i, (got, exp)) in raw_f64s.iter().zip(&test.expected).enumerate() {
                            if (got - exp).abs() > 1e-6 {
                                println!("    WARNING: raw output[{i}]={got}, expected={exp}");
                            }
                        }
                    }
                } else {
                    println!("FAIL  (outputs differ!)");
                    println!(
                        "    raw bytes:       {:?}",
                        &raw_output[..raw_output.len().min(64)]
                    );
                    println!(
                        "    sovereign bytes: {:?}",
                        &sovereign_output[..sovereign_output.len().min(64)]
                    );

                    let raw_f64s = bytes_to_f64s(&raw_output);
                    let sov_f64s = bytes_to_f64s(&sovereign_output);
                    for (i, (r, s)) in raw_f64s.iter().zip(&sov_f64s).enumerate() {
                        if (r - s).abs() > 1e-15 {
                            println!("    output[{i}]: raw={r}, sovereign={s}");
                        }
                    }

                    fail += 1;
                    if first_failure.is_none() {
                        first_failure = Some(format!("{}: {}", gpu.adapter_name, test.name));
                    }

                    println!("\n    -- Sovereign WGSL (full) --");
                    for (i, line) in sovereign_wgsl.lines().enumerate() {
                        println!("    {i:3}| {line}");
                    }
                    println!(
                        "    -- (total {} lines, {} bytes) --\n",
                        sovereign_wgsl.lines().count(),
                        sovereign_wgsl.len()
                    );
                }
            }
            Err(e) => {
                println!("SKIP  (sovereign compile failed: {e})");
                skip += 1;
            }
        }
    }

    (pass, fail, skip, first_failure)
}

// ── Main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Sovereign Compiler GPU Round-Trip Verification");
    println!("  Compares raw WGSL vs naga round-trip output on GPU");
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
            "f64"
        } else {
            "   "
        };
        println!("  [{f64_mark}] {} ({:?})", info.name, info.backend);
    }
    println!();

    let mut total_pass = 0;
    let mut total_fail = 0;
    let mut total_skip = 0;
    let mut all_failures: Vec<String> = Vec::new();

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

        let wdev = gpu.to_wgpu_device();
        let caps = DeviceCapabilities::from_device(&wdev);
        let compiler = SovereignCompiler::new(caps);

        println!("━━━ {tag} ━━━\n");

        let tests = vec![
            level_0_f32_scalar(),
            level_1_f32_workgroup(),
            level_2_f64_scalar(),
            level_3_f64_workgroup(),
            level_4_df64_scalar(),
            level_5_df64_workgroup(),
        ];

        let (pass, fail, skip, first) = run_tests_on_gpu(&gpu, &compiler, &tests);
        total_pass += pass;
        total_fail += fail;
        total_skip += skip;
        if let Some(f) = first {
            all_failures.push(f);
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  TOTAL: {total_pass} pass, {total_fail} fail, {total_skip} skip");
    println!("═══════════════════════════════════════════════════════════");

    if !all_failures.is_empty() {
        println!("\nFailures (minimal reproducers for naga round-trip bug):");
        for f in &all_failures {
            println!("  - {f}");
        }
    }

    if total_fail > 0 {
        std::process::exit(1);
    }
}

fn bytes_to_f64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
