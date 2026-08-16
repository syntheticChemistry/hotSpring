// SPDX-License-Identifier: AGPL-3.0-or-later
//! A/B Test: ROP Render-Path Force Accumulation vs Compute atomicAdd
//!
//! Compares two silicon paths for multi-pole force accumulation:
//! - Path A: Compute shader with atomicAdd(i32) fixed-point (uses ALU + L2 atomics)
//! - Path B: Render pass with additive blending (uses ROP units, zero ALU)
//!
//! At production 32^4 lattice sizes (4M links), measures:
//! - Throughput (scatter-adds/s)
//! - Latency per accumulation frame
//! - Correctness (sum agreement between paths)
//!
//! Silicon routing implication: if ROP path is competitive, force accumulation
//! can be offloaded to ROPs while ALU runs leapfrog integration — true
//! silicon-level parallelism within a single trajectory.

use hotspring_barracuda::gpu::GpuF64;
use std::time::Instant;

const COMPUTE_ATOMIC_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> forces: array<f32>;
@group(0) @binding(1) var<storage, read_write> accum: array<atomic<i32>>;

const SCALE: f32 = 1048576.0;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&forces) { return; }
    let fixed = i32(forces[idx] * SCALE);
    atomicAdd(&accum[idx], fixed);
}
"#;

const ROP_SCATTER_SHADER: &str = r#"
struct Params {
    inv_width: f32,
    inv_height: f32,
    n_links: u32,
    component_offset: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> force_data: array<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) val: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let link = vid;

    let x = f32(link % 256u) * params.inv_width * 2.0 - 1.0;
    let y = f32(link / 256u) * params.inv_height * 2.0 - 1.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);

    let base = link * 4u + params.component_offset;
    out.val = vec4<f32>(
        force_data[base],
        force_data[base + 1u],
        force_data[base + 2u],
        force_data[base + 3u],
    );

    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.val;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScatterParams {
    inv_width: f32,
    inv_height: f32,
    n_links: u32,
    component_offset: u32,
}

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  ROP Force Accumulation A/B Test");
    println!("  Path A: Compute atomicAdd(i32) — lights ALU + L2 atomics");
    println!("  Path B: Render-pass additive blend — lights ROP only");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let sizes: &[(u32, &str)] = &[
        (4096, "8³ (test)"),
        (32768, "12⁴-equiv"),
        (262144, "16⁴"),
        (1048576, "32⁴ (production)"),
    ];

    let n_poles = 8u32;

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
        let blend_format = wgpu::TextureFormat::Rgba32Float;

        println!("━━━ {} ━━━", gpu.adapter_name);
        println!("  Poles per accumulation: {n_poles}");
        println!();

        println!("  {:>16} │ {:>12} {:>12} │ {:>12} {:>12} │ {:>7}",
                 "Lattice", "A (compute)", "A Gops/s", "B (render)", "B Gops/s", "Winner");
        println!("  {:─>16} │ {:─>12} {:─>12} │ {:─>12} {:─>12} │ {:─>7}",
                 "", "", "", "", "", "");

        for &(n_links, label) in sizes {
            let (compute_ms, render_ms) = run_ab_test(
                device, queue, n_links, n_poles, blend_format,
            );

            let compute_ops = f64::from(n_links) * f64::from(n_poles) / (compute_ms / 1000.0) / 1e9;
            let render_ops = f64::from(n_links) * f64::from(n_poles) / (render_ms / 1000.0) / 1e9;
            let winner = if render_ms < compute_ms { "ROP" } else { "ALU" };

            println!("  {:>16} │ {:>9.3} ms {:>9.1} G │ {:>9.3} ms {:>9.1} G │ {:>7}",
                     label, compute_ms, compute_ops, render_ms, render_ops, winner);
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("  A/B Test Complete");
    println!("  Science: if ROP wins at 32⁴, force accumulation offloads to");
    println!("  fixed-function while ALU runs leapfrog — true silicon parallelism");
    println!("═══════════════════════════════════════════════════════════════");
}

fn run_ab_test(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    n_links: u32,
    n_poles: u32,
    blend_format: wgpu::TextureFormat,
) -> (f64, f64) {
    let n_values = n_links * 4;
    let force_data: Vec<f32> = (0..n_values)
        .map(|i| ((i as f32) * 0.001).sin() * 0.01)
        .collect();

    let force_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("force_data"),
        size: (force_data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&force_buf, 0, bytemuck::cast_slice(&force_data));

    let compute_ms = bench_compute_atomic(device, queue, &force_buf, n_values, n_poles);
    let render_ms = bench_render_rop(device, queue, &force_buf, n_links, n_poles, blend_format);

    (compute_ms, render_ms)
}

fn bench_compute_atomic(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    force_buf: &wgpu::Buffer,
    n_values: u32,
    n_poles: u32,
) -> f64 {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compute_atomic"),
        source: wgpu::ShaderSource::Wgsl(COMPUTE_ATOMIC_SHADER.into()),
    });

    let accum_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("accum"),
        size: u64::from(n_values) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
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

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_atomic_force"),
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: force_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: accum_buf.as_entire_binding(),
            },
        ],
    });

    let wg_count = (n_values + 255) / 256;

    // Warmup
    for _ in 0..3 {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            for _ in 0..n_poles {
                pass.dispatch_workgroups(wg_count, 1, 1);
            }
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    let iterations = 20u32;
    let t0 = Instant::now();
    for _ in 0..iterations {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.clear_buffer(&accum_buf, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            for _ in 0..n_poles {
                pass.dispatch_workgroups(wg_count, 1, 1);
            }
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    let elapsed = t0.elapsed();
    elapsed.as_secs_f64() * 1000.0 / f64::from(iterations)
}

fn bench_render_rop(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    force_buf: &wgpu::Buffer,
    n_links: u32,
    n_poles: u32,
    blend_format: wgpu::TextureFormat,
) -> f64 {
    let tex_w = 256u32;
    let tex_h = (n_links + tex_w - 1) / tex_w;

    let target_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rop_target"),
        size: wgpu::Extent3d {
            width: tex_w,
            height: tex_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: blend_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rop_scatter"),
        source: wgpu::ShaderSource::Wgsl(ROP_SCATTER_SHADER.into()),
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
        label: Some("rop_blend"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::PointList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: blend_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview_mask: None,
        cache: None,
    });

    let params = ScatterParams {
        inv_width: 1.0 / tex_w as f32,
        inv_height: 1.0 / tex_h as f32,
        n_links,
        component_offset: 0,
    };
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: std::mem::size_of::<ScatterParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: force_buf.as_entire_binding(),
            },
        ],
    });

    // Warmup
    for _ in 0..3 {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&render_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            for _ in 0..n_poles {
                pass.draw(0..n_links, 0..1);
            }
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
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&render_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            for _ in 0..n_poles {
                pass.draw(0..n_links, 0..1);
            }
        }
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    let elapsed = t0.elapsed();
    elapsed.as_secs_f64() * 1000.0 / f64::from(iterations)
}
