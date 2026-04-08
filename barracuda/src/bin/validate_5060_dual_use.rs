// SPDX-License-Identifier: AGPL-3.0-or-later
//! Proof-of-concept: dispatch CUDA compute on the RTX 5060 while it serves as
//! the active display GPU.  Validates dual-use (display + compute) without
//! displacing the nvidia driver or disrupting DRM.
//!
//! The kernel is a simple SAXPY (y = α·x + y) over 1M f32 elements, enough to
//! exercise the SM pipeline, prove memory access, and verify correctness.

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use hotspring_barracuda::tolerances;
use std::sync::Arc;
use std::time::Instant;

const N: usize = 1 << 20; // 1M elements
const ALPHA: f32 = 2.5;
const BLOCK: u32 = 256;

static SAXPY_PTX: &str = include_str!("saxpy.ptx");

fn find_5060_ordinal() -> Option<usize> {
    let count = CudaContext::device_count().ok()? as usize;
    for i in 0..count {
        let ctx = CudaContext::new(i).ok()?;
        let name = ctx.name().ok()?;
        if name.contains("5060") || name.contains("GB206") {
            return Some(i);
        }
    }
    None
}

fn main() {
    println!("=== RTX 5060 Dual-Use CUDA Validation ===\n");

    let ordinal = match find_5060_ordinal() {
        Some(o) => o,
        None => {
            let count = CudaContext::device_count().unwrap_or(0);
            eprintln!("WARNING: RTX 5060 not found by name among {count} CUDA devices.");
            eprintln!(
                "         Falling back to device 0 — results may not reflect 5060 behaviour."
            );
            0
        }
    };

    let ctx = CudaContext::new(ordinal).expect("failed to create CUDA context");
    let name = ctx.name().unwrap_or_else(|_| "unknown".into());
    let (cc_major, cc_minor) = ctx.compute_capability().unwrap_or((0, 0));
    println!("Device {ordinal}: {name} (sm_{cc_major}{cc_minor})");

    let stream: Arc<cudarc::driver::CudaStream> =
        ctx.new_stream().expect("failed to create stream");

    let ptx = Ptx::from_src(SAXPY_PTX);
    let module = ctx.load_module(ptx).expect("module load failed");
    let f = module.load_function("saxpy").expect("kernel lookup failed");

    let x_host: Vec<f32> = (0..N).map(|i| (i as f32) * 0.001).collect();
    let y_host: Vec<f32> = (0..N).map(|i| 1.0 + (i as f32) * 0.0005).collect();
    let expected: Vec<f32> = x_host
        .iter()
        .zip(y_host.iter())
        .map(|(xi, yi)| ALPHA * xi + yi)
        .collect();

    let x_dev: CudaSlice<f32> = stream.clone_htod(&x_host).expect("htod x");
    let mut y_dev: CudaSlice<f32> = stream.clone_htod(&y_host).expect("htod y");

    let grid = ((N as u32 + BLOCK - 1) / BLOCK, 1, 1);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let n_param: i32 = N as i32;
    let t0 = Instant::now();
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&ALPHA)
            .arg(&x_dev)
            .arg(&mut y_dev)
            .arg(&n_param)
            .launch(cfg)
            .expect("kernel launch failed");
    }
    ctx.synchronize().expect("sync failed");
    let elapsed = t0.elapsed();

    let mut y_result = vec![0.0f32; N];
    stream.memcpy_dtoh(&y_dev, &mut y_result).expect("dtoh y");

    let mut max_err: f64 = 0.0;
    for i in 0..N {
        let err = (y_result[i] as f64 - expected[i] as f64).abs();
        if err > max_err {
            max_err = err;
        }
    }

    println!("\nSAXPY N={N} (α={ALPHA})");
    println!("  Kernel time: {:.3} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Max error:   {max_err:.2e}");
    println!(
        "  Throughput:  {:.2} GB/s",
        (3.0 * N as f64 * 4.0) / elapsed.as_secs_f64() / 1e9
    );

    let ok = max_err < tolerances::GLOWPLUG_F32_SAXPY_MAX_ABS; // f32 SAXPY on 1M elements; FMA precision is ~2.4e-4
    println!(
        "\n{}\n",
        if ok {
            "PASS: RTX 5060 dual-use compute validated — display + CUDA coexist"
        } else {
            "FAIL: SAXPY results diverged"
        }
    );

    if !ok {
        std::process::exit(1);
    }
}
