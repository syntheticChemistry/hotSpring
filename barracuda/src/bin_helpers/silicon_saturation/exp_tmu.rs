// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 4: TMU saturation — texture unit peak (textureLoad throughput).

use hotspring_barracuda::toadstool_report::PerformanceMeasurement;

use super::dispatch::{create_compute_pipeline, timed_dispatch};

const SHADER_TMU_FLOOD: &str =
    include_str!("../../bin/shaders/silicon_saturation/tmu_flood.wgsl");

pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 4: TMU Saturation (texture unit throughput) ──\n");

    let n_threads: u32 = 262_144;
    let iterations: u32 = 200;
    let workgroups = n_threads / 256;
    let tex_dim: u32 = 1024;

    let tex_data: Vec<f32> = (0..tex_dim * tex_dim)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let tex_bytes: Vec<u8> = tex_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tmu_flood"),
        size: wgpu::Extent3d {
            width: tex_dim,
            height: tex_dim,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &tex_bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(tex_dim * 4),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: tex_dim,
            height: tex_dim,
            depth_or_array_layers: 1,
        },
    );
    let tex_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n_threads as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let pipeline = create_compute_pipeline(device, SHADER_TMU_FLOOD, &bgl, "tmu_flood");

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&tex_view),
            },
        ],
    });

    let elapsed = timed_dispatch(device, queue, &pipeline, &bg, workgroups, iterations);
    let total_texels = 64u64 * n_threads as u64 * iterations as u64;
    let gtexels = total_texels as f64 / elapsed.as_secs_f64() / 1e9;

    println!(
        "  TMU textureLoad:  {:.1} GT/s  ({:.1} ms, {}M texels)",
        gtexels,
        elapsed.as_secs_f64() * 1000.0,
        total_texels / 1_000_000
    );

    measurements.push(PerformanceMeasurement {
        operation: "saturation.tmu.textureload".into(),
        silicon_unit: "texture_unit".into(),
        precision_mode: "r32float".into(),
        throughput_gflops: gtexels,
        tolerance_achieved: 0.0,
        gpu_model: gpu_name.into(),
        measured_by: "hotSpring/bench_silicon_saturation".into(),
        timestamp: ts,
    });
    println!();
}
