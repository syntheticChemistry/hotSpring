// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 2: Bandwidth sweep — memory controller saturation (sequential vec4 read).

use hotspring_barracuda::toadstool_report::PerformanceMeasurement;

use super::dispatch::{create_compute_pipeline, timed_dispatch};

const SHADER_BANDWIDTH_SEQ: &str =
    include_str!("../../bin/shaders/silicon_saturation/bandwidth_seq.wgsl");

pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 2: Bandwidth Sweep (memory controller saturation) ──\n");

    let iterations: u32 = 100;
    let sizes_mb: &[u64] = &[16, 64, 256, 512];

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

    let pipeline = create_compute_pipeline(device, SHADER_BANDWIDTH_SEQ, &bgl, "bw_seq");

    println!("  {:<12} {:>10} {:>10}", "Size (MB)", "GB/s", "Efficiency");
    println!("  {}", "─".repeat(36));

    for size_mb in sizes_mb {
        let bytes = size_mb * 1024 * 1024;
        let n_vec4 = bytes / 16;
        let n_threads = n_vec4 as u32;
        let workgroups = n_threads.div_ceil(256);

        let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: n_threads as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

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
            ],
        });

        let elapsed = timed_dispatch(device, queue, &pipeline, &bg, workgroups, iterations);
        let total_bytes_read = bytes * iterations as u64;
        let gbs = total_bytes_read as f64 / elapsed.as_secs_f64() / 1e9;

        println!("  {:<12} {:>8.1} {:>10}", size_mb, gbs, "—");

        measurements.push(PerformanceMeasurement {
            operation: format!("saturation.memory.bw_seq_{size_mb}mb"),
            silicon_unit: "memory".into(),
            precision_mode: "fp32_vec4".into(),
            throughput_gflops: gbs,
            tolerance_achieved: 0.0,
            gpu_model: gpu_name.into(),
            measured_by: "hotSpring/bench_silicon_saturation".into(),
            timestamp: ts,
        });
    }
    println!();
}
