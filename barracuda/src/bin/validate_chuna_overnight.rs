// SPDX-License-Identifier: AGPL-3.0-or-later

//! Chuna overnight validation — all Paper 43/44/45 systems in one run.
//!
//! Hardware-agnostic: discovers all f64-capable GPUs at runtime, profiles
//! each via `HardwareCalibration + PrecisionBrain`, sizes workloads to
//! available VRAM, and validates on every target substrate. Any GPU with
//! SHADER_F64 or DF64 fallback is a science device at 14-digit precision.
//!
//! **Paper 43** (Gradient flow integrators) — **GPU streaming HMC**:
//!   - Convergence sweep: ε = 0.02→0.001 for W6/W7/CK4 (GPU streaming)
//!   - Production flow: symmetric N⁴ and asymmetric Ns³×Nt geometries
//!     at β = {5.9, 6.0, 6.2} with `gpu_hmc_trajectory_streaming`
//!   - Includes 8³×16 (T≈125 MeV) and 16³×32 (T≈62 MeV) finite-T lattices
//!   - Dynamical fermion extension via `gpu_dynamical_hmc_trajectory`
//!
//! **Paper 44** (Conservative BGK dielectric):
//!   - Standard + completed Mermin (CPU + GPU)
//!   - Multi-component Mermin (electron-ion, CPU + GPU)
//!
//! **Paper 45** (Multi-species kinetic-fluid coupling):
//!   - GPU BGK relaxation, Euler/Sod shock tube, coupled kinetic-fluid
//!
//! Usage:
//!   cargo run --release --bin validate_chuna_overnight              # auto-select best GPU
//!   cargo run --release --bin validate_chuna_overnight -- --all-gpus # validate on every f64 GPU
//!   cargo run --release --bin validate_chuna_overnight -- --gpu 3090 # target specific GPU

use hotspring_barracuda::bin_helpers::chuna_overnight::{
    SubstrateResults, max_lattice_l, paper_43_convergence, paper_43_dynamical, paper_43_production,
    paper_44_cpu, paper_44_gpu, paper_44_multicomponent_cpu, paper_44_multicomponent_gpu,
    paper_45_gpu_bgk, paper_45_gpu_coupled, paper_45_gpu_euler,
};
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::precision_brain::PrecisionBrain;
use hotspring_barracuda::validation::{TelemetryWriter, ValidationHarness};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let skip_to_dynamical = args.iter().any(|a| a == "--dynamical-only");
    let all_gpus = args.iter().any(|a| a == "--all-gpus");
    let gpu_token: Option<String> = args
        .iter()
        .position(|a| a == "--gpu")
        .and_then(|i| args.get(i + 1).cloned());

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Chuna Overnight Validation — Papers 43 / 44 / 45         ║");
    println!("║  Hardware-agnostic: discover, profile, validate            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    if skip_to_dynamical {
        println!("  [--dynamical-only] Skipping quenched/GPU sections, jumping to dynamical\n");
    }

    let total_start = Instant::now();

    // ═══ Phase 1: Discover GPU substrates ═══
    let adapters = GpuF64::enumerate_adapters();
    let f64_adapters: Vec<_> = adapters.iter().filter(|a| a.has_f64).collect();

    println!(
        "  Substrate inventory ({} adapters, {} with f64):",
        adapters.len(),
        f64_adapters.len()
    );
    for a in &adapters {
        let tag = if a.has_f64 { "f64" } else { "f32" };
        let mem_gb = a.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "    [{}] {} ({}, {tag}, {mem_gb:.1} GB)",
            a.index, a.name, a.driver
        );
    }
    println!();

    let targets: Vec<String> = if all_gpus {
        f64_adapters.iter().map(|a| a.index.to_string()).collect()
    } else if let Some(token) = gpu_token {
        vec![token]
    } else {
        vec!["auto".to_string()]
    };

    if targets.is_empty() {
        eprintln!("  FATAL: no f64-capable GPUs discovered");
        std::process::exit(1);
    }

    let mut harness = ValidationHarness::new("chuna_overnight");
    let mut all_results: Vec<SubstrateResults> = Vec::new();

    for (gpu_idx, token) in targets.iter().enumerate() {
        let gpu_header = if targets.len() > 1 {
            format!("GPU {}/{}", gpu_idx + 1, targets.len())
        } else {
            "GPU".to_string()
        };

        // ═══ Phase 2: Open + Profile ═══
        let gpu = match rt.block_on(GpuF64::with_adapter(token)) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("  [{gpu_header}] Failed to open GPU '{token}': {e}");
                continue;
            }
        };

        let max_l = max_lattice_l(gpu.device().limits().max_buffer_size);
        let vram_gb = gpu.device().limits().max_buffer_size as f64 / (1024.0 * 1024.0 * 1024.0);

        let brain = PrecisionBrain::new(&gpu);
        let cal = &brain.calibration;

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "  [{gpu_header}] {} (f64={}, df64={}, f16={}, subgroups={})",
            gpu.adapter_name, gpu.has_f64, gpu.full_df64_mode, gpu.has_f16, gpu.has_subgroups
        );
        println!("  [{gpu_header}] {cal}");
        println!("  [{gpu_header}] VRAM: {vram_gb:.1} GB → max lattice L={max_l}");

        let telem_filename = if targets.len() > 1 {
            let safe_name: String = gpu
                .adapter_name
                .chars()
                .map(|c| if c.is_alphanumeric() { c } else { '_' })
                .collect();
            format!("chuna_overnight_{safe_name}.jsonl")
        } else {
            "chuna_overnight_telemetry.jsonl".to_string()
        };
        let mut telem =
            TelemetryWriter::discover(&telem_filename).with_substrate(gpu.adapter_name.clone());

        telem.log_map(
            "hardware_profile",
            &[
                ("has_f64", f64::from(u8::from(gpu.has_f64))),
                ("full_df64_mode", f64::from(u8::from(gpu.full_df64_mode))),
                ("has_f16", f64::from(u8::from(gpu.has_f16))),
                ("has_subgroups", f64::from(u8::from(gpu.has_subgroups))),
                ("has_timestamps", f64::from(u8::from(gpu.has_timestamps))),
                ("vram_gb", vram_gb),
                ("max_lattice_l", max_l as f64),
            ],
        );
        for tier in &cal.tiers {
            telem.log_map(
                &format!("tier_{:?}", tier.tier),
                &[
                    ("compiles", f64::from(u8::from(tier.compiles))),
                    ("dispatches", f64::from(u8::from(tier.dispatches))),
                    (
                        "transcendentals_safe",
                        f64::from(u8::from(tier.transcendentals_safe)),
                    ),
                    ("compile_us", tier.compile_us),
                    ("dispatch_us", tier.dispatch_us),
                    (
                        "probe_ulp",
                        if tier.probe_ulp.is_finite() {
                            tier.probe_ulp
                        } else {
                            -1.0
                        },
                    ),
                ],
            );
        }

        harness.set_gpu(&gpu.adapter_name);
        harness.set_substrate(&gpu.adapter_name);

        let gpu_start = Instant::now();
        let mut results = SubstrateResults {
            adapter_name: gpu.adapter_name.clone(),
            ..Default::default()
        };

        // ═══ Phase 3: Validate ═══
        if !skip_to_dynamical {
            println!("\n━━━ Paper 43: Gradient Flow Integrators ━━━\n");
            paper_43_convergence(&mut harness, &gpu, &mut telem);
            paper_43_production(&mut harness, &gpu, &mut telem, &mut results);

            println!("\n━━━ Paper 44: Conservative BGK Dielectric ━━━\n");
            paper_44_cpu(&mut harness, &mut telem);
            paper_44_multicomponent_cpu(&mut harness, &mut telem);
            paper_44_gpu(&mut harness, &gpu, &mut telem);
            paper_44_multicomponent_gpu(&mut harness, &gpu, &mut telem);

            println!("\n━━━ Paper 45: Kinetic-Fluid Coupling ━━━\n");
            paper_45_gpu_bgk(&mut harness, &gpu, &mut telem);
            paper_45_gpu_euler(&mut harness, &gpu, &mut telem);
            paper_45_gpu_coupled(&mut harness, &gpu, &mut telem);
        }

        println!("\n━━━ Paper 43: Dynamical Extension (warm-start, GPU streaming) ━━━\n");
        paper_43_dynamical(&mut harness, &gpu, &mut telem, &mut results);

        results.wall_seconds = gpu_start.elapsed().as_secs_f64();
        telem.log("substrate_summary", "wall_seconds", results.wall_seconds);
        println!(
            "\n  [{gpu_header}] {} done in {:.1}s\n",
            gpu.adapter_name, results.wall_seconds
        );
        all_results.push(results);
    }

    // ═══ Phase 4: Cross-GPU Comparison ═══
    if all_results.len() > 1 {
        println!(
            "━━━ Cross-Substrate Comparison ({} GPUs) ━━━\n",
            all_results.len()
        );
        harness.clear_substrate();

        for i in 0..all_results.len() {
            for j in (i + 1)..all_results.len() {
                let a = &all_results[i];
                let b = &all_results[j];
                println!("  {} vs {}:", a.adapter_name, b.adapter_name);

                for (label_a, plaq_a) in &a.plaquettes {
                    if let Some((_, plaq_b)) = b.plaquettes.iter().find(|(l, _)| l == label_a) {
                        let rel = if plaq_a.abs() > 1e-15 {
                            (plaq_a - plaq_b).abs() / plaq_a.abs()
                        } else {
                            (plaq_a - plaq_b).abs()
                        };
                        let agree = rel < 0.05;
                        println!(
                            "    {label_a}: {plaq_a:.6} vs {plaq_b:.6} (rel={rel:.4e}) {}",
                            if agree { "OK" } else { "CHECK" }
                        );
                        harness.check_upper(
                            &format!("xgpu_{label_a}_{}_{}", a.adapter_name, b.adapter_name),
                            rel,
                            0.05,
                        );
                    }
                }

                if let (Some(w0_a), Some(w0_b)) = (a.w0, b.w0) {
                    let rel = (w0_a - w0_b).abs() / w0_a.abs().max(1e-15);
                    println!("    w0: {w0_a:.4} vs {w0_b:.4} (rel={rel:.4e})");
                    harness.check_upper(
                        &format!("xgpu_w0_{}_{}", a.adapter_name, b.adapter_name),
                        rel,
                        0.1,
                    );
                }
                if let (Some(t0_a), Some(t0_b)) = (a.t0, b.t0) {
                    let rel = (t0_a - t0_b).abs() / t0_a.abs().max(1e-15);
                    println!("    t0: {t0_a:.4} vs {t0_b:.4} (rel={rel:.4e})");
                    harness.check_upper(
                        &format!("xgpu_t0_{}_{}", a.adapter_name, b.adapter_name),
                        rel,
                        0.1,
                    );
                }
                println!();
            }
        }

        println!("  Wall times:");
        for r in &all_results {
            println!("    {}: {:.1}s", r.adapter_name, r.wall_seconds);
        }
        println!();
    }

    let total = total_start.elapsed();
    println!(
        "\n  Total wall time: {:.1}s ({} substrate{})",
        total.as_secs_f64(),
        all_results.len(),
        if all_results.len() == 1 { "" } else { "s" }
    );
    harness.finish();
}
