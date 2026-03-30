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

const PRNG_ALU: &str = include_str!("shaders/qcd_silicon_routing/prng_alu.wgsl");
const PRNG_TMU: &str = include_str!("shaders/qcd_silicon_routing/prng_tmu.wgsl");

// ═══════════════════════════════════════════════════════════════════
//  Stencil access: storage buffer vs textureLoad
// ═══════════════════════════════════════════════════════════════════

const STENCIL_ALU: &str = include_str!("shaders/qcd_silicon_routing/stencil_storage.wgsl");
const STENCIL_TMU: &str = include_str!("shaders/qcd_silicon_routing/stencil_texture.wgsl");

// ═══════════════════════════════════════════════════════════════════
//  CG reduction: shared memory vs subgroup
// ═══════════════════════════════════════════════════════════════════

const REDUCE_SHARED: &str = include_str!("shaders/qcd_silicon_routing/reduce_shared.wgsl");

// ═══════════════════════════════════════════════════════════════════
//  SU(3) force: ALU baseline (for compound composition)
// ═══════════════════════════════════════════════════════════════════

const FORCE_ALU: &str = include_str!("shaders/qcd_silicon_routing/force_alu.wgsl");

// ═══════════════════════════════════════════════════════════════════
//  Compound: ALU force + TMU PRNG simultaneously
// ═══════════════════════════════════════════════════════════════════

const COMPOUND_ALU_TMU: &str = include_str!("shaders/qcd_silicon_routing/compound_alu_tmu.wgsl");

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
        name: info.name,
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
        let h = width.div_ceil(max_w);
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
    println!("  Volume: {volume} sites, {iterations} iterations");

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

    let wg = (volume.div_ceil(256), 1u32);

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
    println!("  Volume: {volume} sites, {iterations} iterations");

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

    let wg = (volume.div_ceil(64), 1u32);

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
    println!("  Size: {size} elements, {iterations} iterations");

    let input_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let n_wg = size.div_ceil(256);
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
    println!("  Volume: {volume} sites, {iterations} iterations");

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

    let wg = (volume.div_ceil(256), 1u32);

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
