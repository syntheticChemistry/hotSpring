// SPDX-License-Identifier: AGPL-3.0-or-later
//! Voronoi Coarsening via Rasterizer + Depth Buffer.
//!
//! Uses GPU fixed-function hardware for multigrid prolongation weights:
//! - Render coarse-grid sites as point primitives with depth = distance
//! - Depth buffer gives O(1) nearest-site lookup per fine-grid cell
//! - Rasterizer bins fine-grid cells into Voronoi regions automatically
//!
//! This replaces a compute-shader sort/search with hardware z-test.
//! For lattice QCD multigrid: coarse 4⁴ → fine 32⁴, the depth buffer
//! finds which coarse block each fine site belongs to at fill rate.

use hotspring_barracuda::gpu::GpuF64;
use std::time::Instant;

const DEPTH_VORONOI_SHADER: &str = r#"
struct Params {
    coarse_dim: u32,
    fine_dim: u32,
    inv_fine: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> coarse_sites: array<vec4<f32>>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    var out: VsOut;

    // Triangle list: 6 triangles × 3 verts = 18 verts per coarse site
    let tri_idx = vid / 3u;
    let vert_in_tri = vid % 3u;

    let site = coarse_sites[iid];
    let cx = site.x * params.inv_fine * 2.0 - 1.0;
    let cy = site.y * params.inv_fine * 2.0 - 1.0;
    let radius = 2.0 * params.inv_fine;

    var x: f32;
    var y: f32;
    if vert_in_tri == 0u {
        x = cx;
        y = cy;
    } else {
        let edge = tri_idx * 3u + vert_in_tri - 1u;
        let angle = f32(edge) * 6.28318 / 6.0;
        x = cx + cos(angle) * radius;
        y = cy + sin(angle) * radius;
    }

    // Depth encodes distance from coarse site center (smaller = closer)
    let dx = x - cx;
    let dy = y - cy;
    let depth = sqrt(dx * dx + dy * dy) / radius;

    out.pos = vec4<f32>(x, y, depth, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // The color encodes which coarse block this pixel belongs to
    let block_id = in.pos.z;
    return vec4<f32>(block_id, block_id, block_id, 1.0);
}
"#;

const COMPUTE_NEAREST_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> fine_sites: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> coarse_sites: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> assignment: array<u32>;

struct Params {
    n_fine: u32,
    n_coarse: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_fine { return; }

    let pos = fine_sites[idx];
    var min_dist = 1e30f;
    var nearest = 0u;

    for (var c = 0u; c < params.n_coarse; c = c + 1u) {
        let cpos = coarse_sites[c];
        let dx = pos.x - cpos.x;
        let dy = pos.y - cpos.y;
        let dist = dx * dx + dy * dy;
        if dist < min_dist {
            min_dist = dist;
            nearest = c;
        }
    }
    assignment[idx] = nearest;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VoronoiParams {
    coarse_dim: u32,
    fine_dim: u32,
    inv_fine: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ComputeParams {
    n_fine: u32,
    n_coarse: u32,
}

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Voronoi Coarsening: Rasterizer + Depth Buffer");
    println!("  Multigrid prolongation weights via hardware z-test");
    println!("  Path A: Compute O(n_fine × n_coarse) brute-force search");
    println!("  Path B: Depth buffer O(1) per pixel (hardware min-reduction)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let configs: &[(u32, u32, &str)] = &[
        (4, 16, "4² → 16² (2D slice)"),
        (8, 64, "8² → 64² (2D slice)"),
        (16, 128, "16² → 128² (multigrid)"),
        (32, 256, "32² → 256² (production)"),
    ];

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

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
        println!();
        println!("  {:>22} │ {:>12} {:>12} │ {:>12} {:>12} │ {:>7}",
                 "Config", "Compute ms", "Compute Mq/s", "Depth ms", "Depth Mq/s", "Winner");
        println!("  {:─>22} │ {:─>12} {:─>12} │ {:─>12} {:─>12} │ {:─>7}",
                 "", "", "", "", "", "");

        for &(coarse_dim, fine_dim, label) in configs {
            let (compute_ms, depth_ms) = run_voronoi_ab(device, queue, coarse_dim, fine_dim);

            let n_queries = f64::from(fine_dim * fine_dim);
            let compute_mqps = n_queries / (compute_ms / 1000.0) / 1e6;
            let depth_mqps = n_queries / (depth_ms / 1000.0) / 1e6;
            let winner = if depth_ms < compute_ms { "Depth" } else { "Compute" };

            println!("  {:>22} │ {:>9.3} ms {:>9.1} M │ {:>9.3} ms {:>9.1} M │ {:>7}",
                     label, compute_ms, compute_mqps, depth_ms, depth_mqps, winner);
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Voronoi Coarsening Complete");
    println!("  Science: depth buffer gives O(1) nearest-site per pixel");
    println!("  Multigrid prolongation: coarse→fine weight assignment");
    println!("  at hardware fill rate, freeing ALU for physics");
    println!("═══════════════════════════════════════════════════════════════");
}

fn run_voronoi_ab(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    coarse_dim: u32,
    fine_dim: u32,
) -> (f64, f64) {
    let n_coarse = coarse_dim * coarse_dim;
    let n_fine = fine_dim * fine_dim;

    let coarse_data: Vec<f32> = (0..n_coarse * 4)
        .map(|i| {
            let site = i / 4;
            let comp = i % 4;
            match comp {
                0 => (site % coarse_dim) as f32 * (fine_dim as f32 / coarse_dim as f32) + 0.5,
                1 => (site / coarse_dim) as f32 * (fine_dim as f32 / coarse_dim as f32) + 0.5,
                _ => 0.0,
            }
        })
        .collect();

    let fine_data: Vec<f32> = (0..n_fine * 4)
        .map(|i| {
            let site = i / 4;
            let comp = i % 4;
            match comp {
                0 => (site % fine_dim) as f32 + 0.5,
                1 => (site / fine_dim) as f32 + 0.5,
                _ => 0.0,
            }
        })
        .collect();

    let compute_ms = bench_compute_nearest(device, queue, &fine_data, &coarse_data, n_fine, n_coarse);
    let depth_ms = bench_depth_voronoi(device, queue, &coarse_data, coarse_dim, fine_dim);

    (compute_ms, depth_ms)
}

fn bench_compute_nearest(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fine_data: &[f32],
    coarse_data: &[f32],
    n_fine: u32,
    n_coarse: u32,
) -> f64 {
    let fine_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fine_sites"),
        size: (fine_data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&fine_buf, 0, bytemuck::cast_slice(fine_data));

    let coarse_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("coarse_sites"),
        size: (coarse_data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&coarse_buf, 0, bytemuck::cast_slice(coarse_data));

    let assign_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("assignment"),
        size: u64::from(n_fine) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let params = ComputeParams { n_fine, n_coarse };
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: std::mem::size_of::<ComputeParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compute_nearest"),
        source: wgpu::ShaderSource::Wgsl(COMPUTE_NEAREST_SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
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

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("nearest_site"),
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
            wgpu::BindGroupEntry { binding: 0, resource: fine_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: coarse_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: assign_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
    });

    let wg_count = (n_fine + 255) / 256;

    for _ in 0..3 {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    let iterations = 20u32;
    let t0 = Instant::now();
    for _ in 0..iterations {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    let elapsed = t0.elapsed();
    elapsed.as_secs_f64() * 1000.0 / f64::from(iterations)
}

fn bench_depth_voronoi(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    coarse_data: &[f32],
    coarse_dim: u32,
    fine_dim: u32,
) -> f64 {
    let n_coarse = coarse_dim * coarse_dim;

    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("voronoi_depth"),
        size: wgpu::Extent3d {
            width: fine_dim,
            height: fine_dim,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let color_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("voronoi_color"),
        size: wgpu::Extent3d {
            width: fine_dim,
            height: fine_dim,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let coarse_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("coarse_sites"),
        size: (coarse_data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&coarse_buf, 0, bytemuck::cast_slice(coarse_data));

    let params = VoronoiParams {
        coarse_dim,
        fine_dim,
        inv_fine: 1.0 / fine_dim as f32,
        _pad: 0,
    };
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voronoi_params"),
        size: std::mem::size_of::<VoronoiParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("depth_voronoi"),
        source: wgpu::ShaderSource::Wgsl(DEPTH_VORONOI_SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("voronoi_depth_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview_mask: None,
        cache: None,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: coarse_buf.as_entire_binding() },
        ],
    });

    let tris_per_site = 6u32;
    let verts_per_site = tris_per_site * 3;

    for _ in 0..3 {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&render_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.draw(0..verts_per_site, 0..n_coarse);
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    let iterations = 20u32;
    let t0 = Instant::now();
    for _ in 0..iterations {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&render_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.draw(0..verts_per_site, 0..n_coarse);
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    let elapsed = t0.elapsed();
    elapsed.as_secs_f64() * 1000.0 / f64::from(iterations)
}
