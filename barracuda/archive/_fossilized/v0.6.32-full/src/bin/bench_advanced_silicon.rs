// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark: Advanced compute silicon paths.
//!
//! Exercises silicon units that go beyond basic compute shaders:
//! 1. Subgroup operations (warp/wave shuffle — hardware reduction)
//! 2. Texture sampling interpolation (TMU bilinear — hardware multigrid)
//! 3. Indirect dispatch (GPU self-scheduling — adaptive algorithms)
//! 4. Timestamp queries (hardware cycle counter — zero-overhead profiling)
//!
//! These leverage dedicated hardware that compute shaders normally leave idle.

use hotspring_barracuda::gpu::GpuF64;
use std::time::Instant;

const SHADER_SUBGROUP: &str = include_str!("shaders/silicon_science/subgroup_reduce.wgsl");
const SHADER_TEXTURE: &str = include_str!("shaders/silicon_science/texture_interpolate.wgsl");
const SHADER_INDIRECT: &str = include_str!("shaders/silicon_science/indirect_dispatch.wgsl");

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Advanced Compute Silicon Experiments");
    println!("  Warp intrinsics, TMU interpolation, GPU self-dispatch");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    if adapters.is_empty() {
        eprintln!("No GPU adapters found.");
        std::process::exit(1);
    }

    for adapter in adapters {
        let info = adapter.get_info();
        if info.device_type != wgpu::DeviceType::DiscreteGpu {
            continue;
        }
        let gpu_name = info.name.clone();

        let gpu = match GpuF64::from_adapter(adapter).await {
            Ok(g) => g,
            Err(e) => {
                println!("  Skip {gpu_name}: {e}\n");
                continue;
            }
        };

        let device = gpu.device();
        let queue = gpu.queue();

        println!("━━━ {} ━━━", gpu.adapter_name);
        println!("  Features: {:?}", gpu.device().features() & wgpu::Features::SUBGROUP);
        println!();

        exp1_subgroup_reduce(device, queue);
        exp2_texture_interpolation(device, queue);
        exp3_indirect_dispatch(device, queue);
        exp4_timestamp_profiling(device, queue);

        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  Advanced Compute Silicon Experiments Complete");
    println!("═══════════════════════════════════════════════════════════");
}

fn exp1_subgroup_reduce(device: &wgpu::Device, queue: &wgpu::Queue) {
    println!("  ── Experiment 1: Subgroup Reduction (warp shuffle) ──");
    println!("  QCD analog: dot products, trace sums without shared memory");
    println!();

    let n: u32 = 1 << 20; // 1M elements

    let features = device.features();
    let has_subgroup = features.contains(wgpu::Features::SUBGROUP);

    if !has_subgroup {
        println!("  ⚠ SUBGROUP feature not enabled — falling back to shared-memory reduce");
        println!("  (Feature requires wgpu feature flag at device creation)");
        println!();

        // Still benchmark a workgroup reduce for comparison
        bench_workgroup_reduce(device, queue, n);
        return;
    }

    let input_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    let n_workgroups = (n + 255) / 256;

    let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sg_input"),
        size: (n as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buf, 0, bytemuck::cast_slice(&input_data));

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sg_output"),
        size: (n_workgroups as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params: [u32; 4] = [n, 0, 0, 0];
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sg_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("subgroup_reduce"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SUBGROUP.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            bgl_entry(0, wgpu::ShaderStages::COMPUTE, true, true),
            bgl_entry(1, wgpu::ShaderStages::COMPUTE, true, false),
            bgl_entry(2, wgpu::ShaderStages::COMPUTE, false, false),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sg_reduce"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("reduce_subgroup"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
        ],
    });

    // Warmup
    for _ in 0..5 {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_workgroups, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    let iterations = 100u32;
    let t0 = Instant::now();
    for _ in 0..iterations {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_workgroups, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    let elapsed = t0.elapsed();
    let ms_per = elapsed.as_secs_f64() * 1000.0 / f64::from(iterations);
    let elements_per_sec = f64::from(n) * f64::from(iterations) / elapsed.as_secs_f64();

    println!("  Elements: {} (1M f32)", n);
    println!("  Time: {ms_per:.4} ms/reduce ({elements_per_sec:.2e} elements/s)");
    println!("  Mechanism: subgroupAdd → workgroup reduce (2-tier, zero shared memory in tier 1)");
    println!("  Status: SUBGROUP REDUCTION — SILICON PATH ACTIVATED");
    println!();
}

fn bench_workgroup_reduce(device: &wgpu::Device, queue: &wgpu::Queue, n: u32) {
    // Fallback: measure basic compute throughput for comparison
    let shader_src = r#"
        struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32, }
        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<storage, read> input: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;

        var<workgroup> shared: array<f32, 256>;

        @compute @workgroup_size(256)
        fn main(
            @builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32,
            @builtin(workgroup_id) wg_id: vec3<u32>,
        ) {
            shared[lid] = select(0.0, input[gid.x], gid.x < params.n);
            workgroupBarrier();

            for (var s = 128u; s > 0u; s >>= 1u) {
                if lid < s { shared[lid] += shared[lid + s]; }
                workgroupBarrier();
            }

            if lid == 0u { output[wg_id.x] = shared[0]; }
        }
    "#;

    let n_workgroups = (n + 255) / 256;

    let input_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();

    let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wg_input"),
        size: (n as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buf, 0, bytemuck::cast_slice(&input_data));

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wg_output"),
        size: (n_workgroups as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params: [u32; 4] = [n, 0, 0, 0];
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wg_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wg_reduce"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            bgl_entry(0, wgpu::ShaderStages::COMPUTE, true, true),
            bgl_entry(1, wgpu::ShaderStages::COMPUTE, true, false),
            bgl_entry(2, wgpu::ShaderStages::COMPUTE, false, false),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wg_reduce_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
        ],
    });

    for _ in 0..5 {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_workgroups, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    let iterations = 100u32;
    let t0 = Instant::now();
    for _ in 0..iterations {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_workgroups, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    let elapsed = t0.elapsed();
    let ms_per = elapsed.as_secs_f64() * 1000.0 / f64::from(iterations);
    let elements_per_sec = f64::from(n) * f64::from(iterations) / elapsed.as_secs_f64();
    let bandwidth_gb = (n as f64 * 4.0) / elapsed.as_secs_f64() * f64::from(iterations) / 1e9;

    println!("  Elements: {} (1M f32)", n);
    println!("  Time: {ms_per:.4} ms/reduce ({elements_per_sec:.2e} elements/s)");
    println!("  Effective bandwidth: {bandwidth_gb:.1} GB/s");
    println!("  Mechanism: shared memory tree reduce (8 barriers per dispatch)");
    println!("  Status: WORKGROUP REDUCTION — BASELINE MEASURED");
    println!();
}

fn exp2_texture_interpolation(device: &wgpu::Device, queue: &wgpu::Queue) {
    println!("  ── Experiment 2: Texture Sampling Interpolation (TMU) ──");
    println!("  QCD analog: multigrid prolongation, field smearing at hardware speed");
    println!();

    let grid_dim = 256u32;
    let n_samples = 65536u32;
    let mip_levels = 4u32;

    // Create a 2D texture with mipmaps (simulating lattice field at multiple resolutions)
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("field_texture"),
        size: wgpu::Extent3d {
            width: grid_dim,
            height: grid_dim,
            depth_or_array_layers: 1,
        },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Fill base level with field data
    let field_data: Vec<f32> = (0..grid_dim * grid_dim * 4)
        .map(|i| {
            let x = (i / 4) % grid_dim;
            let y = (i / 4) / grid_dim;
            let comp = i % 4;
            match comp {
                0 => (x as f32 * 0.1).sin() * (y as f32 * 0.1).cos(),
                1 => (x as f32 * 0.05).cos(),
                2 => (y as f32 * 0.07).sin(),
                _ => 1.0,
            }
        })
        .collect();

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&field_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(grid_dim * 16),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: grid_dim,
            height: grid_dim,
            depth_or_array_layers: 1,
        },
    );

    // Fill mip levels with averaged data
    for mip in 1..mip_levels {
        let mip_dim = grid_dim >> mip;
        let mip_data: Vec<f32> = (0..mip_dim * mip_dim * 4)
            .map(|i| {
                let x = (i / 4) % mip_dim;
                let y = (i / 4) / mip_dim;
                let comp = i % 4;
                let scale = (1 << mip) as f32;
                match comp {
                    0 => (x as f32 * 0.1 * scale).sin() * (y as f32 * 0.1 * scale).cos() / scale,
                    1 => (x as f32 * 0.05 * scale).cos() / scale,
                    2 => (y as f32 * 0.07 * scale).sin() / scale,
                    _ => 1.0,
                }
            })
            .collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&mip_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(mip_dim * 16),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: mip_dim,
                height: mip_dim,
                depth_or_array_layers: 1,
            },
        );
    }

    let tex_view = tex.create_view(&wgpu::TextureViewDescriptor::default());

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("bilinear"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    // Random sample coordinates (sub-pixel positions for interpolation test)
    let coords: Vec<f32> = (0..n_samples * 2)
        .map(|i| {
            let seed = (i * 17 + 31) % 10000;
            seed as f32 / 10000.0
        })
        .collect();

    let coords_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("coords"),
        size: (coords.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&coords_buf, 0, bytemuck::cast_slice(&coords));

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("interpolated"),
        size: (n_samples as u64) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params: [u32; 4] = [grid_dim, n_samples, 0, 0];
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tex_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tex_interp"),
        source: wgpu::ShaderSource::Wgsl(SHADER_TEXTURE.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });

    let pipeline_interp = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("interpolate"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("interpolate_field"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let pipeline_mg = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("multigrid"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("multigrid_restrict"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&tex_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: coords_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: output_buf.as_entire_binding() },
        ],
    });

    let n_wg = (n_samples + 63) / 64;

    // Benchmark bilinear interpolation
    for _ in 0..5 {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline_interp);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    let iterations = 100u32;
    let t0 = Instant::now();
    for _ in 0..iterations {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline_interp);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    let elapsed = t0.elapsed();
    let ms_interp = elapsed.as_secs_f64() * 1000.0 / f64::from(iterations);
    let samples_per_sec = f64::from(n_samples) * f64::from(iterations) / elapsed.as_secs_f64();

    // Benchmark multigrid (3 texture samples per thread)
    for _ in 0..5 {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline_mg);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    let t0 = Instant::now();
    for _ in 0..iterations {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline_mg);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    let elapsed = t0.elapsed();
    let ms_mg = elapsed.as_secs_f64() * 1000.0 / f64::from(iterations);
    let mg_samples_per_sec = f64::from(n_samples) * f64::from(iterations) / elapsed.as_secs_f64();

    println!("  Texture: {}×{} Rgba32Float, {} mip levels", grid_dim, grid_dim, mip_levels);
    println!("  Samples: {} random sub-pixel positions", n_samples);
    println!("  Bilinear interp: {ms_interp:.4} ms ({samples_per_sec:.2e} samples/s)");
    println!("  Multigrid (3 LODs): {ms_mg:.4} ms ({mg_samples_per_sec:.2e} samples/s)");
    println!("  Mechanism: TMU hardware bilinear filter + mipmap LOD selection");
    println!("  Status: TEXTURE INTERPOLATION — SILICON PATH ACTIVATED");
    println!();
}

fn exp3_indirect_dispatch(device: &wgpu::Device, queue: &wgpu::Queue) {
    println!("  ── Experiment 3: Indirect Dispatch (GPU self-scheduling) ──");
    println!("  QCD analog: adaptive CG, error-driven refinement without CPU roundtrip");
    println!();

    let n: u32 = 1 << 16; // 64K sites
    let threshold = 0.3f32;

    // Simulate residuals — some above threshold, some below
    let residuals: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.01).sin() * 0.5))
        .collect();
    let n_active_expected = residuals.iter().filter(|&&r| r.abs() > threshold).count();

    let residual_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("residuals"),
        size: (n as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&residual_buf, 0, bytemuck::cast_slice(&residuals));

    // Indirect dispatch buffer (also used as atomic counter)
    let indirect_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("indirect_args"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let active_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("active_sites"),
        size: (n as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params_data: [u32; 4] = [threshold.to_bits(), n, 0, 0];
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("indirect_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("indirect"),
        source: wgpu::ShaderSource::Wgsl(SHADER_INDIRECT.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            bgl_entry(0, wgpu::ShaderStages::COMPUTE, true, true),
            bgl_entry(1, wgpu::ShaderStages::COMPUTE, true, false),
            bgl_entry(2, wgpu::ShaderStages::COMPUTE, false, false),
            bgl_entry(3, wgpu::ShaderStages::COMPUTE, false, false),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });

    let compact_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compact"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("compact_active_sites"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let process_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("process"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("process_active"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: residual_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: indirect_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: active_buf.as_entire_binding() },
        ],
    });

    let n_wg_compact = (n + 255) / 256;

    // Benchmark the full compact+indirect_dispatch cycle
    let iterations = 50u32;
    let t0 = Instant::now();
    for _ in 0..iterations {
        // Zero the counter
        queue.write_buffer(&indirect_buf, 0, &[0u8; 16]);

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            // Phase 1: compact active sites (writes dispatch count)
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&compact_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_wg_compact, 1, 1);
        }
        {
            // Phase 2: process only active sites (GPU reads its own dispatch count)
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&process_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups_indirect(&indirect_buf, 0);
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    let elapsed = t0.elapsed();
    let ms_per = elapsed.as_secs_f64() * 1000.0 / f64::from(iterations);
    let speedup = n as f64 / n_active_expected as f64;

    println!("  Sites: {} total, ~{} active (threshold={threshold})", n, n_active_expected);
    println!("  Time: {ms_per:.4} ms/cycle (compact + indirect dispatch)");
    println!("  Work reduction: {speedup:.1}x (only process {:.0}% of sites)", 100.0 / speedup);
    println!("  Mechanism: GPU writes dispatch_workgroups_indirect args — zero CPU roundtrip");
    println!("  Status: INDIRECT DISPATCH — SILICON PATH ACTIVATED");
    println!();
}

fn exp4_timestamp_profiling(device: &wgpu::Device, queue: &wgpu::Queue) {
    println!("  ── Experiment 4: Timestamp Query (hardware cycle counter) ──");
    println!("  QCD analog: zero-overhead kernel profiling for silicon routing decisions");
    println!();

    let features = device.features();
    let has_timestamps = features.contains(wgpu::Features::TIMESTAMP_QUERY);

    if !has_timestamps {
        println!("  ⚠ TIMESTAMP_QUERY not enabled");
        println!("  Status: TIMESTAMP PROFILING — NEEDS FEATURE FLAG");
        println!();
        return;
    }

    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: 4,
    });

    let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts_resolve"),
        size: 32,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts_readback"),
        size: 32,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Simple workload to timestamp
    let shader_src = r#"
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let idx = gid.x;
            var val = data[idx];
            for (var i = 0u; i < 100u; i++) {
                val = val * 1.00001 + 0.00001;
            }
            data[idx] = val;
        }
    "#;

    let n = 65536u32;
    let data_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts_data"),
        size: (n as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let zeros = vec![0u8; (n * 4) as usize];
    queue.write_buffer(&data_buf, 0, &zeros);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ts_workload"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[bgl_entry(0, wgpu::ShaderStages::COMPUTE, false, false)],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ts_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: data_buf.as_entire_binding() }],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.write_timestamp(&query_set, 0);
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(n / 64, 1, 1);
    }
    enc.write_timestamp(&query_set, 1);
    enc.resolve_query_set(&query_set, 0..2, &resolve_buf, 0);
    enc.copy_buffer_to_buffer(&resolve_buf, 0, &readback_buf, 0, 16);
    queue.submit(std::iter::once(enc.finish()));
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

    let slice = readback_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

    let data = slice.get_mapped_range();
    let timestamps: &[u64] = bytemuck::cast_slice(&data);
    let t0_ns = timestamps[0];
    let t1_ns = timestamps[1];
    let delta_ns = t1_ns.saturating_sub(t0_ns);
    let period = queue.get_timestamp_period();
    let kernel_us = delta_ns as f64 * period as f64 / 1000.0;
    drop(data);
    readback_buf.unmap();

    println!("  Timestamps: t0={t0_ns}, t1={t1_ns}");
    println!("  Period: {period} ns/tick");
    println!("  Kernel time: {kernel_us:.2} µs (65K threads × 100 FMA)");
    println!("  Mechanism: hardware cycle counter — zero overhead, sub-µs resolution");
    println!("  Status: TIMESTAMP PROFILING — SILICON PATH ACTIVATED");
    println!();
}

fn bgl_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    read_only: bool,
    is_uniform: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: if is_uniform {
                wgpu::BufferBindingType::Uniform
            } else {
                wgpu::BufferBindingType::Storage { read_only }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
