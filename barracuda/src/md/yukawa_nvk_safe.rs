// SPDX-License-Identifier: AGPL-3.0-only

//! NVK-safe Yukawa force dispatch.
//!
//! hotSpring local workaround for barracuda `YukawaForceF64` which compiles
//! shaders via `ShaderTemplate::with_math_f64()` — missing the NVK `exp(f64)`
//! crash workaround. This module uses `ShaderTemplate::for_device_auto()` which
//! auto-patches `exp()` → `exp_f64()` on NVK/nouveau drivers.
//!
//! **For toadstool to absorb**: change barracuda's `yukawa_f64.rs` line 129
//! from `with_math_f64` to `for_device` (or `for_device_auto`). Same fix
//! needed for `yukawa_celllist_f64.rs`, `erfc_forces.rs`, `greens_apply.rs`.

use barracuda::device::WgpuDevice;
use barracuda::shaders::precision::ShaderTemplate;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const YUKAWA_SHADER: &str = include_str!("shaders/yukawa_force_f64_nvk_safe.wgsl");

/// Execute Yukawa f64 force computation with NVK-safe shader compilation.
///
/// Works on both proprietary NVIDIA and NVK/nouveau drivers.
/// Takes raw f64 slices (not `Tensor`) to avoid barracuda's `pub(crate)` buffer API.
///
/// Returns `(forces_flat [n*3], pe [n])`.
///
/// # Errors
/// Returns error string if GPU dispatch fails.
pub fn yukawa_force_f64_nvk_safe(
    device: &Arc<WgpuDevice>,
    positions: &[f64],
    n: usize,
    kappa: f64,
    prefactor: f64,
    cutoff: f64,
    box_side: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let raw = device.device();
    let queue = device.queue();

    let pos_bytes: Vec<u8> = positions.iter().flat_map(|v| v.to_le_bytes()).collect();
    let pos_buf = raw.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Yukawa pos"),
        contents: &pos_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let forces_size = (n * 3 * size_of::<f64>()) as u64;
    let forces_buf = raw.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Yukawa forces"),
        size: forces_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let pe_size = (n * size_of::<f64>()) as u64;
    let pe_buf = raw.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Yukawa PE"),
        size: pe_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params: [f64; 8] = [
        n as f64,
        kappa,
        prefactor,
        cutoff * cutoff,
        box_side,
        box_side,
        box_side,
        0.0,
    ];
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    let params_buf = raw.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Yukawa params"),
        contents: &params_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let shader_src = ShaderTemplate::for_device_auto(YUKAWA_SHADER, device);
    let shader = raw.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Yukawa NVK-safe"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let bgl = raw.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            entry(0, true),
            entry(1, false),
            entry(2, false),
            entry(3, true),
        ],
    });

    let pl = raw.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = raw.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Yukawa F64 Pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bg = raw.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pos_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: forces_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pe_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = raw.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }

    let forces_staging = raw.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: forces_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let pe_staging = raw.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: pe_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&forces_buf, 0, &forces_staging, 0, forces_size);
    encoder.copy_buffer_to_buffer(&pe_buf, 0, &pe_staging, 0, pe_size);

    queue.submit(Some(encoder.finish()));

    let forces = read_f64_buffer(raw, &forces_staging, n * 3)?;
    let pe = read_f64_buffer(raw, &pe_staging, n)?;

    Ok((forces, pe))
}

fn read_f64_buffer(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<f64>, String> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("map recv: {e}"))?
        .map_err(|e| format!("map: {e}"))?;
    let data = slice.get_mapped_range();
    let result: Vec<f64> = bytemuck::cast_slice(&data)
        .iter()
        .copied()
        .take(count)
        .collect();
    drop(data);
    staging.unmap();
    Ok(result)
}
