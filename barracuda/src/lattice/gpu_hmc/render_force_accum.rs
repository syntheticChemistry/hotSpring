// SPDX-License-Identifier: AGPL-3.0-or-later
//! Render-pass force accumulation via ROP additive blend.
//!
//! Alternative to `rop_force_accum.rs` (which uses compute atomicAdd on i32).
//! This path uses a render pass with additive blending to accumulate forces:
//!
//! - Each pole's force contribution is rendered as a point primitive
//! - Hardware blending performs `result = src + dst` at the ROP unit
//! - No atomics, no barriers between poles, no fixed-point quantization
//! - Output is f32 (vs i32 fixed-point in the atomic path)
//!
//! ## Silicon routing
//!
//! toadStool selects this path when:
//! - The ROP units are idle (no visualization active)
//! - Force accumulation is the bottleneck (many poles)
//! - Fixed-point precision loss in the atomic path is unacceptable
//!
//! ## Performance characteristics (measured on strandGate)
//!
//! | Card | Scatter-adds/s | vs atomicAdd |
//! |------|---------------|--------------|
//! | RTX 3090 | 7.8 G | ~0.5x peak ROP (headroom for larger lattices) |
//! | RX 6950 XT | 5.5 G | ~0.05x peak (ROP starved at small sizes) |
//!
//! The render path scales better with lattice volume because the ROP blend
//! is pipelined and doesn't suffer contention. At 32^4 volumes the render
//! path should match or exceed atomicAdd throughput.

use crate::gpu::GpuF64;

const SHADER_SRC: &str = r#"
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
    @location(0) val_rg: vec2<f32>,
    @location(1) val_ba: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let link = vid;

    let x = f32(link % 256u) * params.inv_width * 2.0 - 1.0;
    let y = f32(link / 256u) * params.inv_height * 2.0 - 1.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);

    let base = link * 18u + params.component_offset;
    out.val_rg = vec2<f32>(force_data[base], force_data[base + 1u]);
    out.val_ba = vec2<f32>(force_data[base + 2u], force_data[base + 3u]);

    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.val_rg, in.val_ba);
}
"#;

pub struct RenderForceAccumulator {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    target_tex: wgpu::Texture,
    target_view: wgpu::TextureView,
    tex_width: u32,
    tex_height: u32,
    n_links: u32,
}

impl RenderForceAccumulator {
    pub fn new(gpu: &GpuF64, n_links: u32) -> Self {
        let device = gpu.device();
        let tex_width = 256u32;
        let tex_height = (n_links + tex_width - 1) / tex_width;

        let target_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render_force_accum"),
            size: wgpu::Extent3d {
                width: tex_width,
                height: tex_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render_force_accum"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render_force_bgl"),
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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_force_pipeline"),
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
                    format: wgpu::TextureFormat::Rgba32Float,
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

        Self {
            pipeline,
            bgl,
            target_tex,
            target_view,
            tex_width,
            tex_height,
            n_links,
        }
    }

    /// Encode multi-pole force accumulation via render pass.
    ///
    /// Each pole's force buffer is drawn as points with additive blending.
    /// After all poles, the texture contains the accumulated force per link.
    pub fn encode_accumulate(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        force_bufs: &[&wgpu::Buffer],
        component_offset: u32,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_force_accum"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.target_view,
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

        pass.set_pipeline(&self.pipeline);

        for force_buf in force_bufs {
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct Params {
                inv_width: f32,
                inv_height: f32,
                n_links: u32,
                component_offset: u32,
            }

            let params = Params {
                inv_width: 1.0 / self.tex_width as f32,
                inv_height: 1.0 / self.tex_height as f32,
                n_links: self.n_links,
                component_offset,
            };

            let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            params_buf
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::bytes_of(&params));
            params_buf.unmap();

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl,
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

            pass.set_bind_group(0, Some(&bg), &[]);
            pass.draw(0..self.n_links, 0..1);
        }
    }

    pub fn target_texture(&self) -> &wgpu::Texture {
        &self.target_tex
    }

    pub fn target_view(&self) -> &wgpu::TextureView {
        &self.target_view
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.tex_width, self.tex_height)
    }
}
