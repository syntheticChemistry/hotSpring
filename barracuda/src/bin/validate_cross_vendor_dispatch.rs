// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-vendor dispatch validation: submit SAXPY via coralReef RPC to every
//! available CUDA-capable GPU, proving the same kernel runs interchangeably
//! through the daemon pipeline.
//!
//! No direct device access or privileges required — all work flows through
//! the glowplug daemon's `device.dispatch` RPC.

use std::path::Path;
use std::time::Instant;

use hotspring_barracuda::glowplug_client::{GlowplugClient, GlowplugDispatchOptions};
use hotspring_barracuda::tolerances;

const N: usize = 1 << 12; // 4K f32 elements — fits RPC line limit comfortably
const ALPHA: f32 = 2.5;

fn glowplug_socket() -> String {
    if let Ok(p) = std::env::var("CORALREEF_GLOWPLUG_SOCKET") {
        return p;
    }
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
    let family = std::env::var("CORALREEF_FAMILY_ID")
        .or_else(|_| std::env::var("FAMILY_ID"))
        .unwrap_or_else(|_| "default".into());
    format!("{runtime_dir}/biomeos/coral-glowplug-{family}.sock")
}

/// Build PTX for SAXPY: `out[i] = alpha*x[i] + y[i]`
///
/// Three buffer params: param_x (input), param_y (input), param_out (output).
/// Alpha and N are baked into the PTX as immediate values.
/// Targets sm_70 for Volta compatibility (JIT-compiles to higher SM at runtime).
fn build_saxpy_ptx() -> String {
    let alpha_hex = format!("{:08X}", ALPHA.to_bits());
    format!(
        r"
.version 7.0
.target sm_70
.address_size 64

.visible .entry main_kernel(
    .param .u64 param_x,
    .param .u64 param_y,
    .param .u64 param_out
)
{{
    .reg .pred %p<2>;
    .reg .f32 %f<5>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;

    mov.u32     %r3, %ctaid.x;
    mov.u32     %r4, %ntid.x;
    mov.u32     %r5, %tid.x;
    mad.lo.s32  %r1, %r3, %r4, %r5;
    setp.ge.s32 %p1, %r1, {N};
    @%p1 bra    $done;

    ld.param.u64 %rd1, [param_x];
    ld.param.u64 %rd2, [param_y];
    ld.param.u64 %rd3, [param_out];

    mul.wide.s32 %rd5, %r1, 4;
    add.s64      %rd6, %rd1, %rd5;
    add.s64      %rd7, %rd2, %rd5;
    add.s64      %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];   // x[i]
    ld.global.f32 %f2, [%rd7];   // y[i]
    fma.rn.f32    %f3, %f1, 0f{alpha_hex}, %f2;
    st.global.f32 [%rd8], %f3;   // out[i]

$done:
    ret;
}}
",
    )
}

fn main() {
    println!("=== Cross-Vendor Dispatch Validation ===");
    println!("Kernel: SAXPY out[i] = {ALPHA}*x[i] + y[i], N={N}\n");

    let client = GlowplugClient::from_socket(Path::new(&glowplug_socket()));

    let devices = match client.list_devices() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("device.list failed: {e}");
            eprintln!(
                "Is coral-glowplug running?  systemctl status coral-glowplug  (socket: {})",
                glowplug_socket()
            );
            std::process::exit(1);
        }
    };

    if devices.is_empty() {
        println!("No managed devices found. Is coral-glowplug running with managed GPUs?");
        std::process::exit(1);
    }

    // Prepare host data
    let x_host: Vec<f32> = (0..N).map(|i| (i as f32) * 0.001).collect();
    let y_host: Vec<f32> = (0..N).map(|i| 1.0 + (i as f32) * 0.0005).collect();
    let expected: Vec<f32> = x_host
        .iter()
        .zip(y_host.iter())
        .map(|(xi, yi)| ALPHA * xi + yi)
        .collect();

    let x_bytes: Vec<u8> = x_host.iter().flat_map(|v| v.to_le_bytes()).collect();
    let y_bytes: Vec<u8> = y_host.iter().flat_map(|v| v.to_le_bytes()).collect();

    let ptx = build_saxpy_ptx();
    let input_bufs = vec![x_bytes, y_bytes];
    let output_bytes = (N * 4) as u64;

    let mut total = 0;
    let mut passed = 0;

    for dev in &devices {
        let bdf = dev.bdf.as_str();
        let name = dev
            .name
            .as_deref()
            .unwrap_or("unknown");
        let personality = dev.personality.as_str();
        let protected = dev.protected;

        if !personality.contains("nvidia") && !personality.contains("cuda") {
            println!("{bdf}  {name:<30} ({personality:<10})  SKIP (no CUDA driver)");
            continue;
        }

        total += 1;
        print!("{bdf}  {name:<30} ");

        let block = 256u32;
        let grid = (N as u32).div_ceil(block);

        let opts = GlowplugDispatchOptions {
            dims: [grid, 1, 1],
            workgroup: [block, 1, 1],
            kernel_name: "main_kernel".to_string(),
            shared_mem: 0,
        };

        let t0 = Instant::now();
        let dispatch_result = client.dispatch_with_options(
            bdf,
            ptx.as_bytes(),
            &input_bufs,
            std::slice::from_ref(&output_bytes),
            &opts,
        );
        let elapsed = t0.elapsed();

        let outputs = match dispatch_result {
            Ok(o) => o,
            Err(e) => {
                println!("FAIL ({e})");
                continue;
            }
        };

        let out_bytes = match outputs.first() {
            Some(b) if !b.is_empty() => b.as_slice(),
            _ => {
                println!("FAIL (empty output)");
                continue;
            }
        };

        let out_f32: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut max_err: f64 = 0.0;
        let mut first_bad = None;
        for i in 0..N.min(out_f32.len()) {
            let err = (out_f32[i] as f64 - expected[i] as f64).abs();
            if err > max_err {
                max_err = err;
                if err > tolerances::GLOWPLUG_F32_SAXPY_MAX_ABS && first_bad.is_none() {
                    first_bad = Some(i);
                }
            }
        }

        let throughput_gbs = (3.0 * N as f64 * 4.0) / elapsed.as_secs_f64() / 1e9;

        let ok = max_err < tolerances::GLOWPLUG_F32_SAXPY_MAX_ABS;
        if ok {
            passed += 1;
            println!(
                "PASS  {:.3}ms  max_err={:.2e}  {:.1} GB/s{}",
                elapsed.as_secs_f64() * 1000.0,
                max_err,
                throughput_gbs,
                if protected { "  [PROTECTED]" } else { "" },
            );
        } else {
            println!(
                "FAIL  max_err={:.2e} (threshold {:.2e})  first_bad={}",
                max_err,
                tolerances::GLOWPLUG_F32_SAXPY_MAX_ABS,
                first_bad.map_or_else(|| "?".to_string(), |i| format!("[{i}]")),
            );
        }
    }

    println!("\n{passed}/{total} CUDA-capable GPUs passed SAXPY validation via RPC");

    if total == 0 {
        println!("No CUDA-capable GPUs were dispatched.");
        std::process::exit(1);
    }
    if passed < total {
        std::process::exit(1);
    }
    println!(
        "\nAll dispatches succeeded through the daemon pipeline — pkexec-free, vendor-agnostic."
    );
}
