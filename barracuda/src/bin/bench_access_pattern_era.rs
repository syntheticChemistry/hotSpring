// SPDX-License-Identifier: AGPL-3.0-or-later
//! Access Pattern vs Generation — proving the IC theory.
//!
//! The genealogy profiler showed NVIDIA wins at linear copies (1.5×),
//! but AMD wins at HMC (20×). This experiment isolates the reason:
//! strided lattice-like access patterns vs linear sequential access.
//!
//! If the Infinity Cache theory is correct:
//! - Linear access: NVIDIA should win (GDDR6X bandwidth dominates)
//! - Strided access (QCD-like): AMD should win (IC absorbs random)
//! - The crossover should occur near the IC boundary (~64 MB)

use hotspring_barracuda::gpu::GpuF64;
use std::time::Instant;

const SHADER: &str = include_str!("shaders/silicon_genealogy/strided_vs_linear.wgsl");

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Access Pattern × Generation — IC Theory Proof               ║");
    println!("║     Linear vs Strided at various working set sizes              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let discrete: Vec<_> = adapters
        .into_iter()
        .filter(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .collect();

    // Working set sizes to test (in vec4 elements = ×16 bytes)
    let sizes: &[(u32, &str)] = &[
        (256 * 1024, "4 MB"),      // Well within both caches
        (1024 * 1024, "16 MB"),    // Within IC, within NVIDIA VRAM sweet spot
        (4 * 1024 * 1024, "64 MB"),  // Near IC boundary
        (8 * 1024 * 1024, "128 MB"), // Exceeds IC for copy pattern
    ];

    let mut results: Vec<(String, Vec<(f64, f64)>)> = Vec::new();

    for adapter in discrete {
        let info = adapter.get_info();
        let name = info.name.clone();
        println!("━━━ {} ━━━", name);

        let gpu = match GpuF64::from_adapter(adapter).await {
            Ok(g) => g,
            Err(e) => {
                println!("  SKIP: {e}\n");
                continue;
            }
        };

        let device = gpu.device();
        let queue = gpu.queue();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
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
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let linear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("linear"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("linear_access"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let strided_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("strided"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("strided_access"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        println!("  {:>8}  {:>12}  {:>12}  {:>10}", "Size", "Linear GB/s", "Strided GB/s", "Ratio S/L");
        println!("  {:>8}  {:>12}  {:>12}  {:>10}", "─".repeat(8), "─".repeat(12), "─".repeat(12), "─".repeat(10));

        let mut card_results: Vec<(f64, f64)> = Vec::new();
        let limit = device.limits().max_buffer_size as u64;

        for &(n_elements, label) in sizes {
            let buf_size = n_elements as u64 * 16;
            if buf_size > limit / 2 {
                println!("  {:>8}  (exceeds buffer limit)", label);
                card_results.push((0.0, 0.0));
                continue;
            }

            // Strides mimic a 4D lattice: L = (n_elements)^(1/4), strides = 1, L, L², L³
            let l = (n_elements as f64).powf(0.25) as u32;
            let stride_x = 1u32;
            let stride_y = l;
            let stride_z = l * l;

            let params: [u32; 4] = [n_elements, stride_x, stride_y, stride_z];
            let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

            let src = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: buf_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let dst = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: buf_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
                ],
            });

            let wgs = (n_elements + 255) / 256;

            let bench = |pipeline: &wgpu::ComputePipeline| -> f64 {
                // Warmup
                for _ in 0..3 {
                    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default()); p.set_pipeline(pipeline); p.set_bind_group(0, Some(&bg), &[]); p.dispatch_workgroups(wgs, 1, 1); }
                    queue.submit(std::iter::once(enc.finish()));
                    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                }

                let iters = 30u32;
                let t0 = Instant::now();
                for _ in 0..iters {
                    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default()); p.set_pipeline(pipeline); p.set_bind_group(0, Some(&bg), &[]); p.dispatch_workgroups(wgs, 1, 1); }
                    queue.submit(std::iter::once(enc.finish()));
                    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                }
                let elapsed = t0.elapsed().as_secs_f64();
                let bytes = buf_size as f64 * 2.0 * iters as f64;
                bytes / elapsed / 1e9
            };

            let linear_bw = bench(&linear_pipeline);
            let strided_bw = bench(&strided_pipeline);
            let ratio = strided_bw / linear_bw;

            println!("  {:>8}  {:>10.1} GB/s  {:>10.1} GB/s  {:>8.3}×", label, linear_bw, strided_bw, ratio);
            card_results.push((linear_bw, strided_bw));
        }

        println!();
        results.push((name, card_results));
    }

    // Cross-card comparison
    if results.len() >= 2 {
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║     Cross-Generation Access Pattern Analysis                    ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();
        println!("  {:>8}  {:>18}  {:>18}  {:>12}", "Size", &results[0].0[..20.min(results[0].0.len())], &results[1].0[..20.min(results[1].0.len())], "Winner");
        println!();

        println!("  LINEAR access (sequential — favors raw bandwidth):");
        for (i, &(_, label)) in sizes.iter().enumerate() {
            let nv = results[0].1[i].0;
            let amd = results[1].1[i].0;
            if nv == 0.0 || amd == 0.0 { continue; }
            let winner = if nv > amd { &results[0].0 } else { &results[1].0 };
            let ratio = nv.max(amd) / nv.min(amd);
            println!("    {:>8}: NV {:.1}, AMD {:.1} → {} ({:.2}×)", label, nv, amd, &winner[..18.min(winner.len())], ratio);
        }
        println!();

        println!("  STRIDED access (lattice-like — favors cache hierarchy):");
        for (i, &(_, label)) in sizes.iter().enumerate() {
            let nv = results[0].1[i].1;
            let amd = results[1].1[i].1;
            if nv == 0.0 || amd == 0.0 { continue; }
            let winner = if nv > amd { &results[0].0 } else { &results[1].0 };
            let ratio = nv.max(amd) / nv.min(amd);
            println!("    {:>8}: NV {:.1}, AMD {:.1} → {} ({:.2}×)", label, nv, amd, &winner[..18.min(winner.len())], ratio);
        }
        println!();

        println!("  INTERPRETATION:");
        println!("  If NVIDIA wins LINEAR but AMD wins STRIDED at larger sizes,");
        println!("  the Infinity Cache theory is confirmed: IC absorbs strided patterns");
        println!("  that NVIDIA's L2 cannot. This is WHY AMD is 20× faster at HMC.");
        println!("  A card with >128 MB L2/L3 (Ada 96 MB, future arch) could close this.");
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Access Pattern × Generation Experiment Complete              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}
