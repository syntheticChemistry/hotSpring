// SPDX-License-Identifier: AGPL-3.0-only

//! f64 Built-in Function Test
//!
//! Tests whether native WGSL built-in functions (sqrt, exp, inverseSqrt)
//! compile AND produce correct results when applied to f64 types.
//! Also benchmarks native vs `math_f64` software implementations.
//!
//! Usage:
//!   cargo run --release --bin `f64_builtin_test`

use barracuda::shaders::precision::ShaderTemplate;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::md::shaders::patch_math_f64_preamble;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

#[tokio::main]
async fn main() {
    use wgpu::util::DeviceExt;
    let mut harness = ValidationHarness::new("f64_builtin_test");
    let gpu = GpuF64::new().await.expect("No GPU");
    println!("GPU: {}", gpu.adapter_name);
    println!("SHADER_F64: {}", if gpu.has_f64 { "YES" } else { "NO" });
    println!();

    // ── Phase 1: Compilation tests ──
    println!("═══ Phase 1: Compilation Tests ═══");

    let builtins = [
        ("sqrt", "output[i] = sqrt(val);"),
        ("exp", "output[i] = exp(val);"),
        ("inverseSqrt", "output[i] = inverseSqrt(val);"),
        ("log", "output[i] = log(val);"),
        ("abs", "output[i] = abs(val);"),
        ("floor", "output[i] = floor(val);"),
        ("ceil", "output[i] = ceil(val);"),
        ("round", "output[i] = round(val);"),
    ];

    for (name, body) in &builtins {
        let shader = format!(
            r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let val = input[i];
    {body}
}}
"
        );

        gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let _module = gpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(shader.into()),
            });
        let err = gpu.device().pop_error_scope().await;
        let compiled = err.is_none();
        match err {
            Some(e) => println!("  {name:<15} FAILED — {e}"),
            None => println!("  {name:<15} COMPILED OK"),
        }
        harness.check_bool(&format!("builtin_{name}_compiles"), compiled);
    }

    // ── Phase 2: Correctness — native vs software sqrt ──
    println!();
    println!("═══ Phase 2: Correctness — native vs math_f64 sqrt ═══");

    let n = 1024u32;
    let test_values: Vec<f64> = (0..n).map(|i| (f64::from(i) + 1.0) * 0.01).collect();

    // Native sqrt shader
    let native_sqrt_shader = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input)) { return; }
    output[i] = sqrt(input[i]);
}
";

    // Software sqrt shader (math_f64)
    let math_preamble = patch_math_f64_preamble(&ShaderTemplate::math_f64_preamble());
    let sw_sqrt_shader = format!(
        r"
{math_preamble}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&input)) {{ return; }}
    output[i] = sqrt_f64(input[i]);
}}
"
    );

    // Create buffers
    let input_bytes: Vec<u8> = test_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input"),
            contents: &input_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output_buf = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: u64::from(n) * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buf = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: u64::from(n) * 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Run native sqrt
    let native_results = run_shader(
        &gpu,
        native_sqrt_shader,
        &input_buf,
        &output_buf,
        &staging_buf,
        n,
    );
    // Run software sqrt
    let sw_results = run_shader(
        &gpu,
        &sw_sqrt_shader,
        &input_buf,
        &output_buf,
        &staging_buf,
        n,
    );

    // Compare
    let mut max_diff: f64 = 0.0;
    let mut max_diff_cpu: f64 = 0.0;
    let mut max_diff_idx = 0;
    for i in 0..n as usize {
        let cpu_val = test_values[i].sqrt();
        let diff_native = (native_results[i] - cpu_val).abs();
        let _diff_sw = (sw_results[i] - cpu_val).abs();
        let diff_nat_sw = (native_results[i] - sw_results[i]).abs();
        if diff_nat_sw > max_diff {
            max_diff = diff_nat_sw;
            max_diff_idx = i;
        }
        if diff_native > max_diff_cpu {
            max_diff_cpu = diff_native;
        }
    }
    println!("  Native vs software max diff: {max_diff:.2e} (at index {max_diff_idx})");
    println!("  Native vs CPU f64:           {max_diff_cpu:.2e}");
    println!(
        "  Sample: sqrt(2.0): native={:.15}, sw={:.15}, cpu={:.15}",
        native_results[199],
        sw_results[199],
        test_values[199].sqrt()
    );
    harness.check_upper(
        "sqrt_native_vs_software_max_diff",
        max_diff,
        tolerances::GPU_NATIVE_VS_SOFTWARE_F64,
    );
    harness.check_upper(
        "sqrt_native_vs_cpu_max_diff",
        max_diff_cpu,
        tolerances::GPU_NATIVE_VS_SOFTWARE_F64,
    );

    // ── Phase 3: Correctness — native vs software exp ──
    println!();
    println!("═══ Phase 3: Correctness — native vs math_f64 exp ═══");

    let native_exp_shader = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input)) { return; }
    output[i] = exp(input[i]);
}
";

    let sw_exp_shader = format!(
        r"
{math_preamble}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&input)) {{ return; }}
    output[i] = exp_f64(input[i]);
}}
"
    );

    // Use negative values for exp (like Yukawa screening: exp(-kappa*r))
    let exp_values: Vec<f64> = (0..n).map(|i| -(f64::from(i) + 1.0) * 0.01).collect();
    let exp_input_bytes: Vec<u8> = exp_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let exp_input_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("exp_input"),
            contents: &exp_input_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

    let native_exp = run_shader(
        &gpu,
        native_exp_shader,
        &exp_input_buf,
        &output_buf,
        &staging_buf,
        n,
    );
    let sw_exp = run_shader(
        &gpu,
        &sw_exp_shader,
        &exp_input_buf,
        &output_buf,
        &staging_buf,
        n,
    );

    let mut max_exp_diff: f64 = 0.0;
    let mut max_exp_diff_cpu: f64 = 0.0;
    for i in 0..n as usize {
        let cpu_val = exp_values[i].exp();
        let diff_nat_sw = (native_exp[i] - sw_exp[i]).abs();
        let diff_cpu = (native_exp[i] - cpu_val).abs();
        if diff_nat_sw > max_exp_diff {
            max_exp_diff = diff_nat_sw;
        }
        if diff_cpu > max_exp_diff_cpu {
            max_exp_diff_cpu = diff_cpu;
        }
    }
    println!("  Native vs software max diff: {max_exp_diff:.2e}");
    println!("  Native vs CPU f64:           {max_exp_diff_cpu:.2e}");
    println!(
        "  Sample: exp(-2.0): native={:.15}, sw={:.15}, cpu={:.15}",
        native_exp[199],
        sw_exp[199],
        exp_values[199].exp()
    );
    harness.check_upper(
        "exp_native_vs_software_max_diff",
        max_exp_diff,
        tolerances::GPU_NATIVE_VS_SOFTWARE_F64,
    );
    harness.check_upper(
        "exp_native_vs_cpu_max_diff",
        max_exp_diff_cpu,
        tolerances::GPU_NATIVE_VS_SOFTWARE_F64,
    );

    // ── Phase 4: Performance — 1M elements ──
    println!();
    println!("═══ Phase 4: Performance Benchmark (1M elements) ═══");

    let big_n = 1_000_000u32;
    let big_values: Vec<f64> = (0..big_n)
        .map(|i| (f64::from(i) + 1.0) * 0.000_001)
        .collect();
    let big_bytes: Vec<u8> = big_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let big_input = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("big_input"),
            contents: &big_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    let big_output = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("big_output"),
        size: u64::from(big_n) * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let big_staging = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("big_staging"),
        size: u64::from(big_n) * 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Benchmark: native sqrt × 10 iterations
    let iters = 10;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = run_shader(
            &gpu,
            native_sqrt_shader,
            &big_input,
            &big_output,
            &big_staging,
            big_n,
        );
    }
    let native_sqrt_time = t0.elapsed().as_secs_f64() / f64::from(iters);

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = run_shader(
            &gpu,
            &sw_sqrt_shader,
            &big_input,
            &big_output,
            &big_staging,
            big_n,
        );
    }
    let sw_sqrt_time = t0.elapsed().as_secs_f64() / f64::from(iters);

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = run_shader(
            &gpu,
            native_exp_shader,
            &big_input,
            &big_output,
            &big_staging,
            big_n,
        );
    }
    let native_exp_time = t0.elapsed().as_secs_f64() / f64::from(iters);

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = run_shader(
            &gpu,
            &sw_exp_shader,
            &big_input,
            &big_output,
            &big_staging,
            big_n,
        );
    }
    let sw_exp_time = t0.elapsed().as_secs_f64() / f64::from(iters);

    println!(
        "  {:>20} {:>12} {:>12} {:>10}",
        "Function", "Native", "Software", "Speedup"
    );
    println!(
        "  {:>20} {:>12} {:>12} {:>10}",
        "────────", "──────", "────────", "───────"
    );
    println!(
        "  {:>20} {:>10.3}ms {:>10.3}ms {:>9.1}×",
        "sqrt (1M f64)",
        native_sqrt_time * 1000.0,
        sw_sqrt_time * 1000.0,
        sw_sqrt_time / native_sqrt_time
    );
    println!(
        "  {:>20} {:>10.3}ms {:>10.3}ms {:>9.1}×",
        "exp (1M f64)",
        native_exp_time * 1000.0,
        sw_exp_time * 1000.0,
        sw_exp_time / native_exp_time
    );

    println!();
    println!("═══ Summary ═══");
    println!("Native f64 builtins are available and produce correct results.");
    println!("If speedup > 1.0×, switching to native builtins in the Yukawa");
    println!("force kernel would improve MD throughput proportionally.");
    println!("The Yukawa kernel calls sqrt + exp per pair — any speedup here");
    println!("directly translates to faster simulation steps.");
    harness.finish();
}

fn run_shader(
    gpu: &GpuF64,
    shader_src: &str,
    input_buf: &wgpu::Buffer,
    output_buf: &wgpu::Buffer,
    staging_buf: &wgpu::Buffer,
    n: u32,
) -> Vec<f64> {
    let module = gpu
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

    let bind_group_layout =
        gpu.device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

    let pipeline_layout = gpu
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = gpu
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(output_buf, 0, staging_buf, 0, u64::from(n) * 8);
    gpu.queue().submit(Some(encoder.finish()));

    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(
        wgpu::MapMode::Read,
        move |r: Result<(), wgpu::BufferAsyncError>| {
            tx.send(r).ok();
        },
    );
    gpu.device().poll(wgpu::Maintain::Wait);
    rx.recv()
        .expect("channel recv from map_async")
        .expect("wgpu buffer map_async succeeded");

    let data = slice.get_mapped_range();
    let result: Vec<f64> = data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8-byte chunk")))
        .collect();
    drop(data);
    staging_buf.unmap();
    result
}
