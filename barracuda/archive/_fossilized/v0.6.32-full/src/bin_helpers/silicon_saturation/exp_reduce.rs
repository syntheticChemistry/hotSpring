// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 5: Workgroup reduce — shared memory / LDS bandwidth.

use hotspring_barracuda::toadstool_report::PerformanceMeasurement;

use super::dispatch::{create_compute_pipeline, timed_dispatch};

const SHADER_REDUCE_SWEEP: &str =
    include_str!("../../bin/shaders/silicon_saturation/reduce_sweep.wgsl");

pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 5: Workgroup Reduce (shared memory / LDS) ──\n");

    let n_threads: u32 = 262_144;
    let iterations: u32 = 200;
    let wg_size: u32 = 1024;
    let workgroups = n_threads / wg_size;

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (workgroups as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pipeline = create_compute_pipeline(device, SHADER_REDUCE_SWEEP, &bgl, "reduce_sweep");
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });

    let elapsed = timed_dispatch(device, queue, &pipeline, &bg, workgroups, iterations);
    let total_adds = 1023u64 * workgroups as u64 * iterations as u64;
    let gops = total_adds as f64 / elapsed.as_secs_f64() / 1e9;
    let total_lds_bytes = 1023u64 * 2 * 4 * workgroups as u64 * iterations as u64;
    let lds_gbs = total_lds_bytes as f64 / elapsed.as_secs_f64() / 1e9;

    println!(
        "  Workgroup reduce:  {:.1} Gop/s  LDS: {:.1} GB/s  ({:.1} ms)",
        gops,
        lds_gbs,
        elapsed.as_secs_f64() * 1000.0,
    );

    measurements.push(PerformanceMeasurement {
        operation: "saturation.shared_mem.reduce".into(),
        silicon_unit: "shader_core".into(),
        precision_mode: "fp32_lds".into(),
        throughput_gflops: gops,
        tolerance_achieved: 0.0,
        gpu_model: gpu_name.into(),
        measured_by: "hotSpring/bench_silicon_saturation".into(),
        timestamp: ts,
    });
    println!();
}
