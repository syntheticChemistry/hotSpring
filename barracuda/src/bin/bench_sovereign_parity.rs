// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign compute parity benchmark: compare coral-reef SASS dispatch
//! against vendor wgpu/Vulkan dispatch on the same WGSL QCD shaders.
//!
//! Dual-path comparison for each kernel:
//! 1. **Sovereign**: coral-reef compile (WGSL → native ISA) + coral-driver UVM dispatch
//! 2. **Vendor**: wgpu/Vulkan (NVIDIA proprietary Vulkan driver compiles the WGSL)
//!
//! Outputs per-kernel correctness (ULP-bounded parity) and wall-clock timing.
//!
//! ## Required feature
//!
//! `sovereign-dispatch` — pulls in `coral-gpu` (coral-reef compiler + coral-driver).
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin bench_sovereign_parity --features sovereign-dispatch
//! ```

use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use coral_gpu::{BufferHandle, GpuContext, GpuTarget, NvArch};
use hotspring_barracuda::gpu::GpuF64;

// ── Shader sources (loaded from lattice/shaders/) ────────────────────────────

const WILSON_PLAQUETTE_WGSL: &str =
    include_str!("../lattice/shaders/wilson_plaquette_f64.wgsl");
const SUM_REDUCE_WGSL: &str =
    include_str!("../lattice/shaders/sum_reduce_f64.wgsl");
const CG_COMPUTE_ALPHA_WGSL: &str =
    include_str!("../lattice/shaders/cg_compute_alpha_f64.wgsl");
const SU3_GAUGE_FORCE_WGSL: &str =
    include_str!("../lattice/shaders/su3_gauge_force_f64.wgsl");
const METROPOLIS_WGSL: &str =
    include_str!("../lattice/shaders/metropolis_f64.wgsl");
const DIRAC_STAGGERED_WGSL: &str =
    include_str!("../lattice/shaders/dirac_staggered_f64.wgsl");
const STAGGERED_FERMION_FORCE_WGSL: &str =
    include_str!("../lattice/shaders/staggered_fermion_force_f64.wgsl");
const FERMION_ACTION_SUM_WGSL: &str =
    include_str!("../lattice/shaders/fermion_action_sum_f64.wgsl");
const HAMILTONIAN_ASSEMBLY_WGSL: &str =
    include_str!("../lattice/shaders/hamiltonian_assembly_f64.wgsl");
const CG_KERNELS_WGSL: &str =
    include_str!("../lattice/shaders/cg_kernels_f64.wgsl");

// ── Kernel descriptors ───────────────────────────────────────────────────────

struct KernelSpec {
    name: &'static str,
    wgsl: &'static str,
    setup: SetupFn,
}

type SetupFn = fn(volume: u32) -> KernelInputs;

struct KernelInputs {
    uniform_data: Vec<u8>,
    storage_buffers: Vec<Vec<u8>>,
    output_size_bytes: usize,
    workgroups: [u32; 3],
}

// ── Buffer layout helpers ────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlaqParams {
    volume: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReduceParams {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

fn gen_unit_links(volume: u32) -> Vec<u8> {
    let n_links = volume as usize * 4;
    let n_f64 = n_links * 18;
    let mut data = vec![0.0_f64; n_f64];
    for link in 0..n_links {
        for i in 0..3_usize {
            let idx = link * 18 + (i * 3 + i) * 2;
            data[idx] = 1.0;
        }
    }
    bytemuck::cast_slice(&data).to_vec()
}

fn gen_trivial_nbr(volume: u32) -> Vec<u8> {
    let n_entries = volume as usize * 8;
    let mut nbr = vec![0u32; n_entries];
    for site in 0..volume as usize {
        for mu in 0..4_usize {
            nbr[site * 8 + mu * 2] = (site + 1).min(volume as usize - 1) as u32;
            nbr[site * 8 + mu * 2 + 1] = site.saturating_sub(1) as u32;
        }
    }
    bytemuck::cast_slice(&nbr).to_vec()
}

fn setup_wilson_plaquette(volume: u32) -> KernelInputs {
    let params = PlaqParams {
        volume,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let links = gen_unit_links(volume);
    let nbr = gen_trivial_nbr(volume);
    let out_size = volume as usize * 8;

    let wg_x = (volume + 63) / 64;
    KernelInputs {
        uniform_data: bytemuck::bytes_of(&params).to_vec(),
        storage_buffers: vec![links, nbr, vec![0u8; out_size]],
        output_size_bytes: out_size,
        workgroups: [wg_x, 1, 1],
    }
}

fn setup_sum_reduce(volume: u32) -> KernelInputs {
    let n_f64 = volume as usize;
    let mut input = vec![0.0_f64; n_f64];
    for (i, v) in input.iter_mut().enumerate() {
        *v = 1.0 + (i as f64) * 0.001;
    }
    let input_bytes = bytemuck::cast_slice(&input).to_vec();

    let n_workgroups = (volume + 255) / 256;
    let out_size = n_workgroups as usize * 8;

    let params = ReduceParams {
        size: volume,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };
    KernelInputs {
        uniform_data: bytemuck::bytes_of(&params).to_vec(),
        storage_buffers: vec![input_bytes, vec![0u8; out_size]],
        output_size_bytes: out_size,
        workgroups: [n_workgroups, 1, 1],
    }
}

const KERNELS: &[KernelSpec] = &[
    KernelSpec {
        name: "wilson_plaquette_f64",
        wgsl: WILSON_PLAQUETTE_WGSL,
        setup: setup_wilson_plaquette,
    },
    KernelSpec {
        name: "sum_reduce_f64",
        wgsl: SUM_REDUCE_WGSL,
        setup: setup_sum_reduce,
    },
];

// ── Vendor path (wgpu) ──────────────────────────────────────────────────────

fn run_vendor(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    kernel: &KernelSpec,
    volume: u32,
    warmup: usize,
    iters: usize,
) -> Result<(Vec<u8>, f64), String> {
    let inputs = (kernel.setup)(volume);

    let module = device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(kernel.name),
            source: wgpu::ShaderSource::Wgsl(kernel.wgsl.into()),
        });

    let mut bind_group_entries = Vec::new();
    let mut buffers: Vec<wgpu::Buffer> = Vec::new();

    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform"),
        size: inputs.uniform_data.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buf, 0, &inputs.uniform_data);
    buffers.push(uniform_buf);

    for (i, data) in inputs.storage_buffers.iter().enumerate() {
        let is_output = i == inputs.storage_buffers.len() - 1;
        let usage = if is_output {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
        } else {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
        };
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("storage_{i}")),
            size: data.len().max(4) as u64,
            usage,
            mapped_at_creation: false,
        });
        if !data.is_empty() && !is_output {
            queue.write_buffer(&buf, 0, data);
        }
        buffers.push(buf);
    }

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("layout"),
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(i, _)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if i == 0 {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                } else {
                    let read_only = i < buffers.len() - 1;
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                },
                count: None,
            })
            .collect::<Vec<_>>(),
    });

    for (i, buf) in buffers.iter().enumerate() {
        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buf.as_entire_binding(),
        });
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(kernel.name),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: inputs.output_size_bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let output_buf_idx = buffers.len() - 1;

    let dispatch = |device: &wgpu::Device, queue: &wgpu::Queue| {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(
                inputs.workgroups[0],
                inputs.workgroups[1],
                inputs.workgroups[2],
            );
        }
        queue.submit(Some(encoder.finish()));
    };

    for _ in 0..warmup {
        dispatch(device, queue);
    }
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    let t0 = Instant::now();
    for _ in 0..iters {
        dispatch(device, queue);
    }
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let elapsed = t0.elapsed().as_secs_f64() / iters as f64;

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(
        &buffers[output_buf_idx],
        0,
        &readback_buf,
        0,
        inputs.output_size_bytes as u64,
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv()
        .map_err(|e| format!("readback channel: {e}"))?
        .map_err(|e| format!("readback map: {e}"))?;

    let data = slice.get_mapped_range().to_vec();
    readback_buf.unmap();

    Ok((data, elapsed))
}

// ── Sovereign path (coral-gpu) ──────────────────────────────────────────────

fn run_sovereign(
    ctx: &mut GpuContext,
    kernel: &KernelSpec,
    volume: u32,
    warmup: usize,
    iters: usize,
) -> Result<(Vec<u8>, f64, f64), String> {
    let t_compile = Instant::now();
    let compiled = ctx
        .compile_wgsl(kernel.wgsl)
        .map_err(|e| format!("sovereign compile: {e}"))?;
    let compile_time = t_compile.elapsed().as_secs_f64();

    let inputs = (kernel.setup)(volume);

    let uniform_handle = ctx
        .alloc(inputs.uniform_data.len() as u64)
        .map_err(|e| format!("alloc uniform: {e}"))?;
    ctx.upload(uniform_handle, &inputs.uniform_data)
        .map_err(|e| format!("upload uniform: {e}"))?;

    let mut storage_handles: Vec<BufferHandle> = Vec::new();
    for (i, data) in inputs.storage_buffers.iter().enumerate() {
        let size = data.len().max(4) as u64;
        let h = ctx
            .alloc(size)
            .map_err(|e| format!("alloc storage_{i}: {e}"))?;
        if !data.is_empty() {
            ctx.upload(h, data)
                .map_err(|e| format!("upload storage_{i}: {e}"))?;
        }
        storage_handles.push(h);
    }

    let mut all_handles = vec![uniform_handle];
    all_handles.extend_from_slice(&storage_handles);

    let wg = inputs.workgroups;

    for _ in 0..warmup {
        ctx.dispatch(&compiled, &all_handles, wg)
            .map_err(|e| format!("sovereign dispatch warmup: {e}"))?;
        ctx.sync()
            .map_err(|e| format!("sovereign sync warmup: {e}"))?;
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        ctx.dispatch(&compiled, &all_handles, wg)
            .map_err(|e| format!("sovereign dispatch: {e}"))?;
        ctx.sync()
            .map_err(|e| format!("sovereign sync: {e}"))?;
    }
    let elapsed = t0.elapsed().as_secs_f64() / iters as f64;

    let output_handle = *storage_handles.last().ok_or("no output buffer")?;
    let result = ctx
        .readback(output_handle, inputs.output_size_bytes)
        .map_err(|e| format!("sovereign readback: {e}"))?;

    for h in &storage_handles {
        let _ = ctx.free(*h);
    }
    let _ = ctx.free(uniform_handle);

    Ok((result, elapsed, compile_time))
}

// ── Parity comparison ────────────────────────────────────────────────────────

fn compare_f64_results(vendor: &[u8], sovereign: &[u8], max_ulp: u64) -> (bool, f64) {
    if vendor.len() != sovereign.len() {
        return (false, f64::INFINITY);
    }

    let vendor_f64: &[f64] = bytemuck::cast_slice(vendor);
    let sovereign_f64: &[f64] = bytemuck::cast_slice(sovereign);

    let mut max_ulp_diff = 0u64;
    for (v, s) in vendor_f64.iter().zip(sovereign_f64.iter()) {
        if v.is_nan() && s.is_nan() {
            continue;
        }
        let vi = v.to_bits() as i64;
        let si = s.to_bits() as i64;
        let diff = vi.abs_diff(si);
        max_ulp_diff = max_ulp_diff.max(diff);
    }

    (max_ulp_diff <= max_ulp, max_ulp_diff as f64)
}

// ── JSON result output ───────────────────────────────────────────────────────

fn emit_result(
    kernel_name: &str,
    volume: u32,
    vendor_time: f64,
    sovereign_time: f64,
    compile_time: f64,
    parity_ok: bool,
    max_ulp: f64,
    gpu_name: &str,
) {
    let result = serde_json::json!({
        "kernel": kernel_name,
        "volume": volume,
        "gpu": gpu_name,
        "vendor_dispatch_ms": vendor_time * 1000.0,
        "sovereign_dispatch_ms": sovereign_time * 1000.0,
        "sovereign_compile_ms": compile_time * 1000.0,
        "speedup": vendor_time / sovereign_time.max(1e-12),
        "parity": parity_ok,
        "max_ulp_diff": max_ulp,
    });
    println!("{result}");
}

// ── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  Sovereign Compute Parity Benchmark");
    eprintln!("  coral-reef SASS vs wgpu/Vulkan on QCD WGSL shaders");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    let volumes: &[(u32, &str)] = &[
        (256, "4^4"),
        (4096, "8^4"),
    ];
    let warmup = 2;
    let iters = 5;
    let max_ulp_tolerance = 4;

    // ── Sovereign context ────────────────────────────────────────────────────
    let mut sovereign_ctx = match GpuContext::auto() {
        Ok(ctx) => {
            eprintln!("  Sovereign: {}", ctx.target());
            Some(ctx)
        }
        Err(e) => {
            eprintln!("  Sovereign: unavailable ({e})");
            None
        }
    };

    // ── Vendor context (wgpu) ────────────────────────────────────────────────
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance
        .enumerate_adapters(wgpu::Backends::all())
        .await;

    if adapters.is_empty() {
        eprintln!("  No wgpu GPU adapters found.");
        std::process::exit(1);
    }

    let adapter = adapters.into_iter().next().unwrap();
    let gpu_name = adapter.get_info().name.clone();
    eprintln!("  Vendor:    {gpu_name} (wgpu/Vulkan)\n");

    let gpu = match GpuF64::from_adapter(adapter).await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("  Failed to create wgpu device: {e}");
            std::process::exit(1);
        }
    };
    let device = gpu.device();
    let queue = gpu.queue();

    let mut any_failure = false;

    for kernel in KERNELS {
        for &(volume, vol_name) in volumes {
            eprintln!("━━━ {}: {} ({} sites) ━━━", kernel.name, vol_name, volume);

            // Vendor dispatch
            let (vendor_data, vendor_time) = match run_vendor(
                device, queue, kernel, volume, warmup, iters,
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  vendor FAIL: {e}");
                    any_failure = true;
                    continue;
                }
            };
            eprintln!(
                "  vendor:    {:.3} ms/dispatch",
                vendor_time * 1000.0,
            );

            // Sovereign dispatch
            if let Some(ref mut ctx) = sovereign_ctx {
                let t_compile = Instant::now();
                let compile_result = ctx.compile_wgsl(kernel.wgsl);
                let compile_time = t_compile.elapsed().as_secs_f64();

                match compile_result {
                    Ok(_) => {
                        eprintln!(
                            "  sovereign: compile OK ({:.1} ms, {} bytes SASS)",
                            compile_time * 1000.0,
                            compile_result.as_ref().map_or(0, |k| k.binary.len()),
                        );
                    }
                    Err(ref e) => {
                        eprintln!("  sovereign: compile FAIL: {e}");
                    }
                }

                match run_sovereign(ctx, kernel, volume, warmup, iters) {
                    Ok((sovereign_data, sovereign_time, _)) => {
                        let (parity_ok, max_ulp) = compare_f64_results(
                            &vendor_data,
                            &sovereign_data,
                            max_ulp_tolerance,
                        );
                        eprintln!(
                            "  sovereign: {:.3} ms/dispatch, parity={}, max_ulp={}",
                            sovereign_time * 1000.0,
                            if parity_ok { "PASS" } else { "FAIL" },
                            max_ulp,
                        );
                        if !parity_ok {
                            any_failure = true;
                        }
                        emit_result(
                            kernel.name,
                            volume,
                            vendor_time,
                            sovereign_time,
                            compile_time,
                            parity_ok,
                            max_ulp,
                            &gpu_name,
                        );
                    }
                    Err(e) => {
                        eprintln!("  sovereign: dispatch FAIL: {e}");
                        emit_result(
                            kernel.name,
                            volume,
                            vendor_time,
                            0.0,
                            compile_time,
                            false,
                            f64::NAN,
                            &gpu_name,
                        );
                        any_failure = true;
                    }
                }
            } else {
                eprintln!("  sovereign: SKIPPED (no context)");
                emit_result(
                    kernel.name,
                    volume,
                    vendor_time,
                    0.0,
                    0.0,
                    false,
                    f64::NAN,
                    &gpu_name,
                );
            }
            eprintln!();
        }
    }

    // ── Cross-generation compile validation (full HMC pipeline) ────────────
    eprintln!("\n━━━ Cross-Generation Compile Validation (full HMC pipeline) ━━━\n");

    struct CompileTarget {
        name: &'static str,
        wgsl: &'static str,
    }
    let hmc_shaders: &[CompileTarget] = &[
        CompileTarget { name: "wilson_plaquette_f64", wgsl: WILSON_PLAQUETTE_WGSL },
        CompileTarget { name: "sum_reduce_f64", wgsl: SUM_REDUCE_WGSL },
        CompileTarget { name: "cg_compute_alpha_f64", wgsl: CG_COMPUTE_ALPHA_WGSL },
        CompileTarget { name: "su3_gauge_force_f64", wgsl: SU3_GAUGE_FORCE_WGSL },
        CompileTarget { name: "metropolis_f64", wgsl: METROPOLIS_WGSL },
        CompileTarget { name: "dirac_staggered_f64", wgsl: DIRAC_STAGGERED_WGSL },
        CompileTarget { name: "staggered_fermion_force_f64", wgsl: STAGGERED_FERMION_FORCE_WGSL },
        CompileTarget { name: "fermion_action_sum_f64", wgsl: FERMION_ACTION_SUM_WGSL },
        CompileTarget { name: "hamiltonian_assembly_f64", wgsl: HAMILTONIAN_ASSEMBLY_WGSL },
        CompileTarget { name: "cg_kernels_f64", wgsl: CG_KERNELS_WGSL },
    ];

    let compile_targets: &[(&str, NvArch)] = &[
        ("SM 35 (Kepler/K80)", NvArch::Sm35),
        ("SM 70 (Volta/Titan V)", NvArch::Sm70),
        ("SM 120 (Blackwell/5060)", NvArch::Sm120),
    ];

    for (label, arch) in compile_targets {
        let target = GpuTarget::Nvidia(*arch);
        match GpuContext::new(target) {
            Ok(ctx) => {
                for shader in hmc_shaders {
                    let wgsl = shader.wgsl;
                    let ctx_ref = std::panic::AssertUnwindSafe(&ctx);
                    let result = std::panic::catch_unwind(move || {
                        let t0 = Instant::now();
                        let r = ctx_ref.compile_wgsl(wgsl);
                        (t0.elapsed().as_secs_f64(), r)
                    });
                    match result {
                        Ok((secs, Ok(k))) => {
                            eprintln!(
                                "  {label:30} {name:30} → {size:>6} bytes  ({ms:.0} ms)",
                                name = shader.name,
                                size = k.binary.len(),
                                ms = secs * 1000.0,
                            );
                        }
                        Ok((_, Err(e))) => {
                            eprintln!("  {label:30} {name:30} → FAIL: {e}", name = shader.name);
                        }
                        Err(_) => {
                            eprintln!(
                                "  {label:30} {name:30} → PANIC (ISA limitation)",
                                name = shader.name,
                            );
                        }
                    }
                }
                eprintln!();
            }
            Err(e) => {
                eprintln!("  {label:30} → context FAIL: {e}");
            }
        }
    }

    eprintln!("\n═══════════════════════════════════════════════════════════");
    if any_failure {
        eprintln!("  RESULT: Some kernels FAILED parity or dispatch");
        std::process::exit(1);
    } else if sovereign_ctx.is_some() {
        eprintln!("  RESULT: All kernels PASSED parity check");
    } else {
        eprintln!("  RESULT: Vendor-only run (sovereign unavailable)");
    }
}
