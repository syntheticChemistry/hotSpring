// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 3: Cache hierarchy — L2/Infinity Cache boundary detection.

use hotspring_barracuda::toadstool_report::PerformanceMeasurement;

use super::dispatch::{create_compute_pipeline, timed_dispatch};

const SHADER_CACHE_PROBE: &str =
    include_str!("../../bin/shaders/silicon_saturation/cache_probe.wgsl");

pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 3: Cache Hierarchy (L2 / Infinity Cache boundary) ──\n");

    let iterations: u32 = 100;
    let n_threads: u32 = 65_536;
    let workgroups = n_threads / 256;
    let sizes_kb: &[(u64, &str)] = &[
        (64, "64 KB"),
        (256, "256 KB"),
        (1024, "1 MB"),
        (4096, "4 MB"),
        (8192, "8 MB"),
        (32768, "32 MB"),
        (65536, "64 MB"),
        (131072, "128 MB"),
        (262144, "256 MB"),
    ];

    let max_bytes = 256 * 1024 * 1024u64;
    let max_elems = max_bytes / 4;

    let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: max_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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

    let pipeline = create_compute_pipeline(device, SHADER_CACHE_PROBE, &bgl, "cache_probe");

    println!("  {:<12} {:>10} {:>10}", "Working set", "GB/s", "Note");
    println!("  {}", "─".repeat(36));

    let mut prev_gbs = 0.0f64;

    for (size_kb, label) in sizes_kb {
        let size_bytes = size_kb * 1024;
        let size_elems = (size_bytes / 4).min(max_elems) as u32;
        let params_data = [size_elems, 0u32, 0u32, 0u32];

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let elapsed = timed_dispatch(device, queue, &pipeline, &bg, workgroups, iterations);
        let total_bytes_read = 256u64 * n_threads as u64 * iterations as u64;
        let gbs = total_bytes_read as f64 / elapsed.as_secs_f64() / 1e9;

        let note = if prev_gbs > 0.0 && gbs < prev_gbs * 0.7 {
            "← CACHE BOUNDARY"
        } else {
            ""
        };

        println!("  {label:<12} {gbs:>8.1} {note:>10}");
        prev_gbs = gbs;

        measurements.push(PerformanceMeasurement {
            operation: format!("saturation.cache.{size_kb}kb"),
            silicon_unit: "memory".into(),
            precision_mode: "cache_probe".into(),
            throughput_gflops: gbs,
            tolerance_achieved: 0.0,
            gpu_model: gpu_name.into(),
            measured_by: "hotSpring/bench_silicon_saturation".into(),
            timestamp: ts,
        });
    }
    println!();
}
