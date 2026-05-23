// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 6: Atomic contention — global atomicAdd throughput.

use hotspring_barracuda::toadstool_report::PerformanceMeasurement;

use super::dispatch::{create_compute_pipeline, timed_dispatch};

const SHADER_ATOMIC_FLOOD: &str =
    include_str!("../../bin/shaders/silicon_saturation/atomic_flood.wgsl");

pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 6: Atomic Contention (global atomicAdd) ──\n");

    let n_threads: u32 = 262_144;
    let iterations: u32 = 200;
    let workgroups = n_threads / 256;

    let counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 4,
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

    let pipeline = create_compute_pipeline(device, SHADER_ATOMIC_FLOOD, &bgl, "atomic_flood");
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: counter_buf.as_entire_binding(),
        }],
    });

    let elapsed = timed_dispatch(device, queue, &pipeline, &bg, workgroups, iterations);
    let total_atomics = 64u64 * n_threads as u64 * iterations as u64;
    let gatom = total_atomics as f64 / elapsed.as_secs_f64() / 1e9;

    println!(
        "  Atomic throughput: {:.1} Gatom/s  ({:.1} ms, {}M ops)",
        gatom,
        elapsed.as_secs_f64() * 1000.0,
        total_atomics / 1_000_000
    );

    measurements.push(PerformanceMeasurement {
        operation: "saturation.atomic.global_add".into(),
        silicon_unit: "shader_core".into(),
        precision_mode: "u32_atomic".into(),
        throughput_gflops: gatom,
        tolerance_achieved: 0.0,
        gpu_model: gpu_name.into(),
        measured_by: "hotSpring/bench_silicon_saturation".into(),
        timestamp: ts,
    });
    println!();
}
