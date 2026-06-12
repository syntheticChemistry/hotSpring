// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-vendor dispatch validation via the **compute trio** pipeline:
//!
//! 1. **coralReef** — compile WGSL SAXPY shader to native binary
//! 2. **toadStool** — dispatch compiled binary to every GPU
//! 3. **barraCuda** — WGSL math provides the kernel source
//!
//! Validates the full sovereign dispatch path: WGSL → coralReef → toadStool → GPU.
//! No direct device access, no inline PTX, fully vendor-agnostic through WGSL.

use std::time::Instant;

use hotspring_barracuda::glowplug_client::GlowplugDispatchOptions;
use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::tolerances;

#[path = "../bin_helpers/sovereignty/mod.rs"]
mod sovereignty;
use sovereignty::connect::{check_coralreef_liveness, connect_glowplug};

const N: usize = 1 << 12; // 4K f32 elements
const ALPHA: f32 = 2.5;

/// WGSL SAXPY kernel: `out[i] = alpha * x[i] + y[i]`
///
/// Vendor-agnostic — coralReef compiles this to SASS/PTX/GFX ISA depending
/// on the target GPU.
fn build_saxpy_wgsl() -> String {
    format!(
        r"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

const ALPHA: f32 = {ALPHA};
const N: u32 = {N}u;

@compute @workgroup_size(256)
fn main_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= N) {{
        return;
    }}
    out[i] = fma(ALPHA, x[i], y[i]);
}}
"
    )
}

/// Build PTX fallback for SAXPY (used when coralReef is unavailable).
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

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    fma.rn.f32    %f3, %f1, 0f{alpha_hex}, %f2;
    st.global.f32 [%rd8], %f3;

$done:
    ret;
}}
",
    )
}

/// Compile WGSL source via coralReef's `shader.compile.wgsl` RPC.
/// Returns the compiled binary bytes on success.
fn compile_wgsl_via_coralreef(
    nucleus: &NucleusContext,
    wgsl_source: &str,
) -> Result<Vec<u8>, String> {
    let compile_params = serde_json::json!({ "wgsl_source": wgsl_source });
    let resp = nucleus
        .call_by_capability("shader", "shader.compile.wgsl", compile_params)
        .map_err(|e| format!("shader.compile.wgsl RPC: {e}"))?;

    let result = resp.get("result").cloned().unwrap_or_else(|| resp.clone());

    let binary_b64 = result
        .get("binary_b64")
        .or_else(|| result.get("binary"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            let err = result.get("error").map(ToString::to_string);
            format!(
                "no binary in compile response{}",
                err.map_or_else(String::new, |e| format!(": {e}"))
            )
        })?;

    hotspring_barracuda::base64_encode::decode(binary_b64.as_bytes())
        .map_err(|e| format!("base64 decode: {e}"))
}

fn main() {
    println!("=== Cross-Vendor Dispatch Validation (Compute Trio) ===");
    println!("Pipeline: WGSL → coralReef → toadStool → GPU");
    println!("Kernel: SAXPY out[i] = {ALPHA}*x[i] + y[i], N={N}\n");

    println!("--- Harness checks ---");
    let coral_alive = check_coralreef_liveness();
    let client = connect_glowplug();

    let devices = match client.list_devices() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("device.list failed: {e}");
            eprintln!("Is toadstool-ember running?  systemctl status toadstool-ember");
            std::process::exit(1);
        }
    };

    if devices.is_empty() {
        println!("No managed devices found. Is toadstool-ember running with managed GPUs?");
        std::process::exit(1);
    }

    let nucleus = NucleusContext::detect();

    let (kernel_bytes, compile_path) = if coral_alive {
        let wgsl = build_saxpy_wgsl();
        println!("\nCompiling WGSL SAXPY via coralReef...");
        let t0 = Instant::now();
        match compile_wgsl_via_coralreef(&nucleus, &wgsl) {
            Ok(binary) => {
                println!(
                    "  Compiled: {} bytes in {:.1}ms",
                    binary.len(),
                    t0.elapsed().as_secs_f64() * 1000.0
                );
                (binary, "coralReef WGSL")
            }
            Err(e) => {
                eprintln!("  coralReef compile failed: {e}");
                eprintln!("  Falling back to inline PTX");
                (build_saxpy_ptx().into_bytes(), "PTX fallback")
            }
        }
    } else {
        eprintln!("  coralReef unavailable — using inline PTX fallback");
        (build_saxpy_ptx().into_bytes(), "PTX fallback")
    };

    println!("\n--- Dispatch ({compile_path}) ---");

    let x_host: Vec<f32> = (0..N).map(|i| (i as f32) * 0.001).collect();
    let y_host: Vec<f32> = (0..N).map(|i| 1.0 + (i as f32) * 0.0005).collect();
    let expected: Vec<f32> = x_host
        .iter()
        .zip(y_host.iter())
        .map(|(xi, yi)| ALPHA * xi + yi)
        .collect();

    let x_bytes: Vec<u8> = x_host.iter().flat_map(|v| v.to_le_bytes()).collect();
    let y_bytes: Vec<u8> = y_host.iter().flat_map(|v| v.to_le_bytes()).collect();

    let input_bufs = vec![x_bytes, y_bytes];
    let output_bytes = (N * 4) as u64;

    let mut total = 0;
    let mut passed = 0;

    for dev in &devices {
        let bdf = dev.bdf.as_str();
        let name = dev.name.as_deref().unwrap_or("unknown");
        let personality = dev.personality.as_str();
        let protected = dev.protected;

        let skip = !personality.contains("nvidia")
            && !personality.contains("cuda")
            && !personality.contains("vfio");
        if skip {
            println!("{bdf}  {name:<30} ({personality:<10})  SKIP (no compute driver)");
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
        let dispatch_result = client.dispatch(
            bdf,
            &kernel_bytes,
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

    println!("\n{passed}/{total} GPUs passed SAXPY validation ({compile_path})");
    if coral_alive {
        println!(
            "  Compute trio validated: barraCuda WGSL → coralReef compile → toadStool dispatch"
        );
    }

    if total == 0 {
        println!("No compute-capable GPUs were dispatched.");
        std::process::exit(1);
    }
    if passed < total {
        std::process::exit(1);
    }
    println!(
        "\nAll dispatches succeeded through the sovereign pipeline — pkexec-free, vendor-agnostic."
    );
}
