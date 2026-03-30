// SPDX-License-Identifier: AGPL-3.0-only

//! Subgroup f64 diagnostic: isolate why `subgroupAdd(f64)` produces zero.
//!
//! Tests:
//! 1. subgroupAdd(f32) — baseline: does subgroup reduce work at all?
//! 2. subgroupAdd(f64) — the failing case
//! 3. Manual f64 reduce via subgroupShuffle on u32 pairs (bitcast workaround)
//! 4. Shared-memory f64 reduce (known-good reference)
//!
//! Each test sums 256 copies of 1.0 → expected result is 256.0.

use hotspring_barracuda::gpu::GpuF64;
use wgpu::PipelineCompilationOptions;
use wgpu::util::DeviceExt;

const N: usize = 256;

// ── Test 1: subgroupAdd on f32 (baseline) ──────────────────────────────────

const SHADER_SUBGROUP_F32: &str = "
enable subgroups;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> wg_partial: array<f32, 8>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let tid = local_id.x;
    let val = input[tid];

    let sg_sum = subgroupAdd(val);

    let sg_idx = tid / sg_size;
    if sg_id == 0u {
        wg_partial[sg_idx] = sg_sum;
    }
    workgroupBarrier();

    let n_subgroups = 256u / sg_size;
    var v2: f32 = 0.0;
    if tid < n_subgroups {
        v2 = wg_partial[tid];
    }

    if tid < sg_size {
        let final_sum = subgroupAdd(v2);
        if tid == 0u {
            output[0] = final_sum;
        }
    }
}
";

// ── Test 2: subgroupAdd on f64 (the suspect) ──────────────────────────────

const SHADER_SUBGROUP_F64: &str = "
enable subgroups;

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

var<workgroup> wg_partial: array<f64, 8>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let tid = local_id.x;
    let val = input[tid];

    let sg_sum = subgroupAdd(val);

    let sg_idx = tid / sg_size;
    if sg_id == 0u {
        wg_partial[sg_idx] = sg_sum;
    }
    workgroupBarrier();

    let n_subgroups = 256u / sg_size;
    var v2: f64 = f64(0.0);
    if tid < n_subgroups {
        v2 = wg_partial[tid];
    }

    if tid < sg_size {
        let final_sum = subgroupAdd(v2);
        if tid == 0u {
            output[0] = final_sum;
        }
    }
}
";

// ── Test 3: Manual f64 reduce via subgroupShuffle on u32 bitcast pairs ─────

const SHADER_SUBGROUP_F64_SHUFFLE: &str = "
enable subgroups;

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

var<workgroup> wg_partial: array<f64, 8>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let tid = local_id.x;
    var acc: f64 = input[tid];

    // Manual butterfly reduce: transport f64 as u32 pair via subgroupShuffleXor
    for (var offset = sg_size >> 1u; offset > 0u; offset = offset >> 1u) {
        let bits = bitcast<vec2<u32>>(acc);
        let other_lo = subgroupShuffleXor(bits.x, offset);
        let other_hi = subgroupShuffleXor(bits.y, offset);
        acc = acc + bitcast<f64>(vec2<u32>(other_lo, other_hi));
    }

    let sg_idx = tid / sg_size;
    if sg_id == 0u {
        wg_partial[sg_idx] = acc;
    }
    workgroupBarrier();

    let n_subgroups = 256u / sg_size;
    if tid < n_subgroups {
        acc = wg_partial[tid];
    } else {
        acc = f64(0.0);
    }

    if tid < sg_size {
        for (var offset = sg_size >> 1u; offset > 0u; offset = offset >> 1u) {
            let bits = bitcast<vec2<u32>>(acc);
            let other_lo = subgroupShuffleXor(bits.x, offset);
            let other_hi = subgroupShuffleXor(bits.y, offset);
            acc = acc + bitcast<f64>(vec2<u32>(other_lo, other_hi));
        }
        if tid == 0u {
            output[0] = acc;
        }
    }
}
";

// ── Test 4: Shared-memory reduce (known-good reference) ────────────────────

const SHADER_SHARED_MEM_F64: &str = "
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

var<workgroup> shared_data: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    shared_data[tid] = input[tid];
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if tid < stride {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    if tid == 0u {
        output[0] = shared_data[0];
    }
}
";

// ── Test 6: subgroupAdd(f32) WITHOUT enable subgroups; directive ────────────

const SHADER_SUBGROUP_F32_NO_ENABLE: &str = "
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> wg_partial: array<f32, 8>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let tid = local_id.x;
    let val = input[tid];

    let sg_sum = subgroupAdd(val);

    let sg_idx = tid / sg_size;
    if sg_id == 0u {
        wg_partial[sg_idx] = sg_sum;
    }
    workgroupBarrier();

    let n_subgroups = 256u / sg_size;
    var v2: f32 = 0.0;
    if tid < n_subgroups {
        v2 = wg_partial[tid];
    }

    if tid < sg_size {
        let final_sum = subgroupAdd(v2);
        if tid == 0u {
            output[0] = final_sum;
        }
    }
}
";

// ── Test 7: subgroupAdd(f64) WITHOUT enable subgroups; directive ────────────

const SHADER_SUBGROUP_F64_NO_ENABLE: &str = "
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

var<workgroup> wg_partial: array<f64, 8>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let tid = local_id.x;
    let val = input[tid];

    let sg_sum = subgroupAdd(val);

    let sg_idx = tid / sg_size;
    if sg_id == 0u {
        wg_partial[sg_idx] = sg_sum;
    }
    workgroupBarrier();

    let n_subgroups = 256u / sg_size;
    var v2: f64 = f64(0.0);
    if tid < n_subgroups {
        v2 = wg_partial[tid];
    }

    if tid < sg_size {
        let final_sum = subgroupAdd(v2);
        if tid == 0u {
            output[0] = final_sum;
        }
    }
}
";

// ── Test 5: subgroupAdd(f32) to reduce f64 via hi/lo split ─────────────────

const SHADER_SUBGROUP_F64_VIA_F32: &str = "
enable subgroups;

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

var<workgroup> wg_partial_hi: array<f32, 8>;
var<workgroup> wg_partial_lo: array<f32, 8>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let tid = local_id.x;
    let val = input[tid];

    // Dekker split: f64 → (hi: f32, lo: f32) where hi + lo ≈ val
    let hi = f32(val);
    let lo = f32(val - f64(hi));

    // subgroupAdd on f32 (known working)
    let sg_hi = subgroupAdd(hi);
    let sg_lo = subgroupAdd(lo);

    let sg_idx = tid / sg_size;
    if sg_id == 0u {
        wg_partial_hi[sg_idx] = sg_hi;
        wg_partial_lo[sg_idx] = sg_lo;
    }
    workgroupBarrier();

    let n_subgroups = 256u / sg_size;
    var h2: f32 = 0.0;
    var l2: f32 = 0.0;
    if tid < n_subgroups {
        h2 = wg_partial_hi[tid];
        l2 = wg_partial_lo[tid];
    }

    if tid < sg_size {
        let final_hi = subgroupAdd(h2);
        let final_lo = subgroupAdd(l2);
        if tid == 0u {
            output[0] = f64(final_hi) + f64(final_lo);
        }
    }
}
";

fn main() {
    eprintln!("=== Subgroup f64 Diagnostic ===\n");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU init");
    eprintln!("Adapter: {}", gpu.adapter_name);
    eprintln!("has_subgroups: {}", gpu.has_subgroups);
    eprintln!("has_f64: {}", gpu.has_f64);
    eprintln!();

    if !gpu.has_subgroups {
        eprintln!("SKIP: subgroups not available on this adapter");
        return;
    }

    // Test 4 first (shared memory reference — no subgroups needed)
    run_f64_test(
        &gpu,
        "Test 4: shared-memory f64 reduce (reference)",
        SHADER_SHARED_MEM_F64,
        false,
    );

    // Test 1: f32 subgroup
    run_f32_test(&gpu, "Test 1: subgroupAdd(f32)", SHADER_SUBGROUP_F32);

    // Test 2: f64 subgroup (the suspect)
    run_f64_test(
        &gpu,
        "Test 2: subgroupAdd(f64) — THE SUSPECT",
        SHADER_SUBGROUP_F64,
        true,
    );

    // Test 3: f64 via u32 bitcast shuffle
    run_f64_test(
        &gpu,
        "Test 3: f64 via subgroupShuffleXor(u32) bitcast",
        SHADER_SUBGROUP_F64_SHUFFLE,
        true,
    );

    // Test 5: f64 via f32 Dekker split
    run_f64_test(
        &gpu,
        "Test 5: f64 via subgroupAdd(f32) Dekker split",
        SHADER_SUBGROUP_F64_VIA_F32,
        true,
    );

    // Test 6: f32 subgroup WITHOUT enable subgroups;
    run_f32_test(
        &gpu,
        "Test 6: subgroupAdd(f32) NO 'enable subgroups;'",
        SHADER_SUBGROUP_F32_NO_ENABLE,
    );

    // Test 7: f64 subgroup WITHOUT enable subgroups;
    run_f64_test(
        &gpu,
        "Test 7: subgroupAdd(f64) NO 'enable subgroups;'",
        SHADER_SUBGROUP_F64_NO_ENABLE,
        true,
    );

    eprintln!("\n=== Diagnostic Complete ===");
}

fn run_f32_test(gpu: &GpuF64, label: &str, shader: &str) {
    eprintln!("--- {label} ---");

    let input_data: Vec<f32> = vec![1.0_f32; N];

    let input_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output_buf = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = gpu.create_staging_buffer(4, "staging_f32");

    let module = gpu
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
    let pipeline = gpu
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

    let bg = gpu.create_bind_group(&pipeline, &[&input_buf, &output_buf]);

    let mut enc = gpu.begin_encoder("test");
    GpuF64::encode_pass(&mut enc, &pipeline, &bg, 1);
    enc.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, 4);
    gpu.submit_encoder(enc);

    let result = {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        let _ = gpu.device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let vals: &[f32] = bytemuck::cast_slice(&data);
        vals[0]
    };

    let expected = N as f32;
    let pass = (result - expected).abs() < 0.01;
    eprintln!("  Result:   {result}");
    eprintln!("  Expected: {expected}");
    eprintln!(
        "  Status:   {}\n",
        if pass { "PASS" } else { "*** FAIL ***" }
    );
}

fn run_f64_test(gpu: &GpuF64, label: &str, shader: &str, needs_subgroups: bool) {
    eprintln!("--- {label} ---");

    if needs_subgroups && !gpu.has_subgroups {
        eprintln!("  SKIP (no subgroup support)\n");
        return;
    }

    let input_data: Vec<f64> = vec![1.0_f64; N];

    let input_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output_buf = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = gpu.create_staging_buffer(8, "staging_f64");

    let pipeline = if needs_subgroups {
        // Subgroup shaders: compile raw (enable subgroups; already in source)
        // but strip enable f64; same as create_pipeline_f64 does
        let patched = shader.replace("enable f64;", "");
        let module = gpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(patched.into()),
            });
        gpu.device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
    } else {
        // Non-subgroup: use standard f64 pipeline
        gpu.create_pipeline_f64(shader, label)
    };

    let bg = gpu.create_bind_group(&pipeline, &[&input_buf, &output_buf]);

    let mut enc = gpu.begin_encoder("test");
    GpuF64::encode_pass(&mut enc, &pipeline, &bg, 1);
    enc.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, 8);
    gpu.submit_encoder(enc);

    let result = gpu.read_staging_f64(&staging);
    match result {
        Ok(vals) if !vals.is_empty() => {
            let val = vals[0];
            let expected = N as f64;
            let pass = (val - expected).abs() < 0.01;
            eprintln!("  Result:   {val}");
            eprintln!("  Expected: {expected}");
            eprintln!(
                "  Status:   {}\n",
                if pass { "PASS" } else { "*** FAIL ***" }
            );
        }
        Ok(_) => eprintln!("  Status:   *** FAIL *** (empty result)\n"),
        Err(e) => eprintln!("  Status:   *** FAIL *** (readback error: {e})\n"),
    }
}
