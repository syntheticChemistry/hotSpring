// SPDX-License-Identifier: AGPL-3.0-or-later

//! VFIO Sovereign Dispatch Validation
//!
//! Exercises the sovereign GPU compute path on VFIO-bound GPUs:
//!   WGSL → coralReef compiler → native SASS → toadStool VFIO dispatch → readback
//!
//! No wgpu. No Vulkan. No vendor drivers. Pure Rust through VFIO to hardware.
//!
//! **NOTE (May 2026):** `coral-gpu` was excised from coralReef Sprint 9. This
//! binary requires the `sovereign-dispatch` feature and will not compile until
//! toadStool Phase C provides the equivalent `GpuContext` API.
//!
//! Default mode is **warm** — expects GPUs were warm-caught via
//! `coralctl warm-catch <BDF>`. Pass `--cold` for cold-init path.
//!
//! Target hardware (biomeGate):
//!   - Titan V (GV100, SM70) at BDF 02:00.0
//!   - Tesla K80 (GK210, SM37) at BDF 4b:00.0 / 4c:00.0
//!
//! Prerequisites:
//!   - GPUs bound to vfio-pci
//!   - Warm-caught via `coralctl warm-catch <BDF>` (FECS running)
//!   - VFIO group permissions (/dev/vfio/* readable)
//!
//! Usage:
//!   cargo run --release --features sovereign-dispatch --bin validate_vfio_sovereign
//!   cargo run --release --features sovereign-dispatch --bin validate_vfio_sovereign -- --cold
//!   cargo run --release --features sovereign-dispatch --bin validate_vfio_sovereign -- --bdf 02:00.0
//!   cargo run --release --features sovereign-dispatch --bin validate_vfio_sovereign -- --bdf 4b:00.0 --sm 37

use coral_gpu::GpuContext;

const WRITE_CONSTANT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = 42u;
}
"#;

#[derive(Clone, Copy, PartialEq, Eq)]
enum OpenMode {
    Warm,
    WarmLegacy,
    Cold,
}

struct GpuInfo {
    name: &'static str,
    bdf: String,
    sm: u32,
    open_mode: OpenMode,
}

fn known_gpus(cold: bool) -> Vec<GpuInfo> {
    if cold {
        return vec![
            GpuInfo {
                name: "Titan V (GV100)",
                bdf: "0000:02:00.0".into(),
                sm: 70,
                open_mode: OpenMode::Cold,
            },
            GpuInfo {
                name: "K80 die0 (GK210)",
                bdf: "0000:4b:00.0".into(),
                sm: 37,
                open_mode: OpenMode::Cold,
            },
            GpuInfo {
                name: "K80 die1 (GK210)",
                bdf: "0000:4c:00.0".into(),
                sm: 37,
                open_mode: OpenMode::Cold,
            },
        ];
    }
    vec![
        GpuInfo {
            name: "Titan V (GV100)",
            bdf: "0000:02:00.0".into(),
            sm: 70,
            open_mode: OpenMode::Warm,
        },
        GpuInfo {
            name: "K80 die0 (GK210)",
            bdf: "0000:4b:00.0".into(),
            sm: 37,
            open_mode: OpenMode::WarmLegacy,
        },
        GpuInfo {
            name: "K80 die1 (GK210)",
            bdf: "0000:4c:00.0".into(),
            sm: 37,
            open_mode: OpenMode::WarmLegacy,
        },
    ]
}

fn open_gpu(gpu: &GpuInfo) -> Result<GpuContext, coral_gpu::GpuError> {
    match gpu.open_mode {
        OpenMode::Warm => {
            if gpu.sm > 0 {
                GpuContext::from_vfio_warm_with_sm(&gpu.bdf, gpu.sm)
            } else {
                GpuContext::from_vfio_warm(&gpu.bdf)
            }
        }
        OpenMode::WarmLegacy => GpuContext::from_vfio_warm_legacy(&gpu.bdf, gpu.sm),
        OpenMode::Cold => {
            if gpu.sm > 0 {
                GpuContext::from_vfio_with_sm(&gpu.bdf, gpu.sm)
            } else {
                GpuContext::from_vfio(&gpu.bdf)
            }
        }
    }
}

fn mode_label(mode: OpenMode) -> &'static str {
    match mode {
        OpenMode::Warm => "warm",
        OpenMode::WarmLegacy => "warm-legacy",
        OpenMode::Cold => "cold",
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  VFIO Sovereign Dispatch Validation                        ║");
    println!("║  No wgpu · No Vulkan · No vendor driver · VFIO → Hardware  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let args: Vec<String> = std::env::args().collect();
    let cold_mode = args.iter().any(|a| a == "--cold");
    let target_bdf = args.iter().position(|a| a == "--bdf").map(|i| &args[i + 1]);
    let explicit_sm = args
        .iter()
        .position(|a| a == "--sm")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u32>().ok());
    let legacy = args.iter().any(|a| a == "--legacy");

    if cold_mode {
        println!("  Mode: COLD (full cold init — will reset warm state)\n");
    } else {
        println!("  Mode: WARM (preserves warm-catch state from coralctl)\n");
    }

    // --- Phase 1: VFIO GPU Discovery ---
    println!("━━━ Phase 1: VFIO GPU Discovery ━━━\n");

    let vfio_dir = std::path::Path::new("/sys/bus/pci/drivers/vfio-pci");
    if vfio_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(vfio_dir) {
            let mut found = 0u32;
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.contains(':') {
                    continue;
                }
                let vendor_path = format!("/sys/bus/pci/devices/{name_str}/vendor");
                if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                    if vendor.trim() == "0x10de" {
                        let device_path = format!("/sys/bus/pci/devices/{name_str}/device");
                        let device_id = std::fs::read_to_string(&device_path).unwrap_or_default();
                        println!("  VFIO NVIDIA: {name_str} (device {})", device_id.trim());
                        found += 1;
                    }
                }
            }
            if found == 0 {
                println!("  No NVIDIA GPUs found on vfio-pci driver");
            }
        }
    } else {
        println!("  /sys/bus/pci/drivers/vfio-pci not found");
    }

    for gpu in &known_gpus(false) {
        let bound =
            std::path::Path::new(&format!("/sys/bus/pci/devices/{}/driver", gpu.bdf)).exists();
        println!(
            "  {} ({}) — SM{} (sysfs present: {bound})",
            gpu.name, gpu.bdf, gpu.sm
        );
    }
    println!();

    // --- Phase 2: Select target GPUs ---
    let targets: Vec<GpuInfo> = if let Some(bdf) = target_bdf {
        let full_bdf = if bdf.contains(':') && !bdf.starts_with("0000:") {
            format!("0000:{bdf}")
        } else {
            bdf.to_string()
        };
        let sm = explicit_sm.unwrap_or(0);
        let open_mode = if cold_mode {
            OpenMode::Cold
        } else if legacy {
            OpenMode::WarmLegacy
        } else {
            OpenMode::Warm
        };
        vec![GpuInfo {
            name: "user-specified",
            bdf: full_bdf,
            sm,
            open_mode,
        }]
    } else {
        known_gpus(cold_mode)
    };

    let mut total_pass = 0u32;
    let mut total_fail = 0u32;
    let mut total_skip = 0u32;

    for gpu in &targets {
        println!(
            "━━━ {} (BDF={}, SM{}, mode={}) ━━━\n",
            gpu.name,
            gpu.bdf,
            gpu.sm,
            mode_label(gpu.open_mode)
        );

        let mut ctx = match open_gpu(gpu) {
            Ok(ctx) => {
                let target = ctx.target();
                println!(
                    "  VFIO open ({}): OK — {target:?}",
                    mode_label(gpu.open_mode)
                );
                ctx
            }
            Err(e) => {
                println!("  VFIO open ({}): FAILED — {e}", mode_label(gpu.open_mode));
                println!("  SKIP (GPU not accessible via VFIO)\n");
                total_skip += 1;
                continue;
            }
        };

        // Test 1: Compile WGSL → native binary
        print!("  Test 1: WGSL compile (write_constant)... ");
        match ctx.compile_wgsl(WRITE_CONSTANT_WGSL) {
            Ok(kernel) => {
                println!(
                    "PASS ({} bytes, {} GPRs)",
                    kernel.binary.len(),
                    kernel.gpr_count
                );
                total_pass += 1;

                // Test 2: Dispatch + readback
                print!("  Test 2: dispatch + readback... ");
                match try_dispatch(&mut ctx, &kernel) {
                    Ok(val) => {
                        if val == 42 {
                            println!("PASS (read back {val}, expected 42)");
                            total_pass += 1;
                        } else {
                            println!("FAIL (read back {val}, expected 42)");
                            total_fail += 1;
                        }
                    }
                    Err(e) => {
                        println!("FAIL ({e})");
                        total_fail += 1;
                    }
                }
            }
            Err(e) => {
                println!("FAIL ({e})");
                total_fail += 1;
            }
        }

        // Test 3: QCD shader compilation (wilson_plaquette)
        print!("  Test 3: QCD shader compile (wilson_plaquette_f64)... ");
        let qcd_source = std::fs::read_to_string("src/lattice/shaders/wilson_plaquette_f64.wgsl");
        match qcd_source {
            Ok(src) => match ctx.compile_wgsl(&src) {
                Ok(kernel) => {
                    println!(
                        "PASS ({} bytes, {} GPRs)",
                        kernel.binary.len(),
                        kernel.gpr_count
                    );
                    total_pass += 1;
                }
                Err(e) => {
                    println!("FAIL ({e})");
                    total_fail += 1;
                }
            },
            Err(e) => {
                println!("SKIP (shader file not found: {e})");
                total_skip += 1;
            }
        }

        // Test 4: QCD shader compilation (su3_gauge_force)
        print!("  Test 4: QCD shader compile (su3_gauge_force_f64)... ");
        let su3_source = std::fs::read_to_string("src/lattice/shaders/su3_gauge_force_f64.wgsl");
        match su3_source {
            Ok(src) => match ctx.compile_wgsl(&src) {
                Ok(kernel) => {
                    println!(
                        "PASS ({} bytes, {} GPRs)",
                        kernel.binary.len(),
                        kernel.gpr_count
                    );
                    total_pass += 1;
                }
                Err(e) => {
                    println!("FAIL ({e})");
                    total_fail += 1;
                }
            },
            Err(e) => {
                println!("SKIP (shader file not found: {e})");
                total_skip += 1;
            }
        }

        println!();
    }

    // --- Summary ---
    println!("━━━ Summary ━━━\n");
    println!("  PASS: {total_pass}");
    println!("  FAIL: {total_fail}");
    println!("  SKIP: {total_skip}");
    println!();

    if total_fail > 0 {
        std::process::exit(1);
    }
}

fn try_dispatch(ctx: &mut GpuContext, kernel: &coral_gpu::CompiledKernel) -> Result<u32, String> {
    let buf = ctx.alloc(4096).map_err(|e| format!("alloc: {e}"))?;

    let sentinel: u32 = 0xDEAD_BEEF;
    let mut init_data = vec![0u8; 4096];
    for chunk in init_data[..16].chunks_exact_mut(4) {
        chunk.copy_from_slice(&sentinel.to_le_bytes());
    }
    ctx.upload(buf, &init_data)
        .map_err(|e| format!("upload: {e}"))?;

    ctx.dispatch(kernel, &[buf], [1, 1, 1])
        .map_err(|e| format!("dispatch: {e}"))?;

    let readback = ctx
        .readback(buf, 16)
        .map_err(|e| format!("readback: {e}"))?;
    let vals: &[u32] = bytemuck::cast_slice(&readback[..4]);
    Ok(vals[0])
}
