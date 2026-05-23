// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 1: Pure FMA chain — shader ALU saturation (FP32 and DF64).

use hotspring_barracuda::toadstool_report::PerformanceMeasurement;

use super::dispatch::{create_compute_pipeline, timed_dispatch};

const SHADER_FMA_CHAIN_FP32: &str =
    include_str!("../../bin/shaders/silicon_saturation/fma_chain_fp32.wgsl");
const SHADER_FMA_CHAIN_DF64: &str =
    include_str!("../../bin/shaders/silicon_saturation/fma_chain_df64.wgsl");

pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_name: &str,
    measurements: &mut Vec<PerformanceMeasurement>,
    ts: u64,
) {
    println!("  ── Experiment 1: Pure FMA Chain (shader ALU saturation) ──\n");

    let n_threads: u32 = 262_144;
    let iterations: u32 = 200;
    let wg_size: u32 = 256;
    let workgroups = n_threads / wg_size;

    // FP32: 512 iterations × 4 FMA × 2 FLOP = 4096 FLOP per thread
    let fp32_flops_per_thread: u64 = 512 * 4 * 2;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n_threads as u64) * 4,
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

    let pipeline_fp32 = create_compute_pipeline(device, SHADER_FMA_CHAIN_FP32, &bgl, "fma_fp32");
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });

    let elapsed_fp32 = timed_dispatch(device, queue, &pipeline_fp32, &bg, workgroups, iterations);
    let total_fp32_flops = fp32_flops_per_thread * n_threads as u64 * iterations as u64;
    let fp32_tflops = total_fp32_flops as f64 / elapsed_fp32.as_secs_f64() / 1e12;

    println!(
        "  FP32 FMA chain:  {:.2} TFLOPS  ({:.1} ms, {} GFLOP)",
        fp32_tflops,
        elapsed_fp32.as_secs_f64() * 1000.0,
        total_fp32_flops / 1_000_000_000
    );

    measurements.push(PerformanceMeasurement {
        operation: "saturation.shader_core.fp32_fma".into(),
        silicon_unit: "shader_core".into(),
        precision_mode: "fp32".into(),
        throughput_gflops: fp32_tflops * 1000.0,
        tolerance_achieved: 0.0,
        gpu_model: gpu_name.into(),
        measured_by: "hotSpring/bench_silicon_saturation".into(),
        timestamp: ts,
    });

    // DF64: 256 iterations × ~10 FMA-equivalent ops (two_prod + two_sum + accumulate)
    let df64_flops_per_thread: u64 = 256 * 10 * 2;

    let out_buf_df64 = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n_threads as u64) * 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bg_df64 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf_df64.as_entire_binding(),
        }],
    });

    let pipeline_df64 = create_compute_pipeline(device, SHADER_FMA_CHAIN_DF64, &bgl, "fma_df64");
    let elapsed_df64 = timed_dispatch(
        device,
        queue,
        &pipeline_df64,
        &bg_df64,
        workgroups,
        iterations,
    );
    let total_df64_flops = df64_flops_per_thread * n_threads as u64 * iterations as u64;
    let df64_tflops = total_df64_flops as f64 / elapsed_df64.as_secs_f64() / 1e12;

    println!(
        "  DF64 Dekker chain: {:.2} TFLOPS  ({:.1} ms, {} GFLOP)",
        df64_tflops,
        elapsed_df64.as_secs_f64() * 1000.0,
        total_df64_flops / 1_000_000_000
    );

    measurements.push(PerformanceMeasurement {
        operation: "saturation.shader_core.df64_fma".into(),
        silicon_unit: "shader_core".into(),
        precision_mode: "df64".into(),
        throughput_gflops: df64_tflops * 1000.0,
        tolerance_achieved: 0.0,
        gpu_model: gpu_name.into(),
        measured_by: "hotSpring/bench_silicon_saturation".into(),
        timestamp: ts,
    });

    println!();
}
