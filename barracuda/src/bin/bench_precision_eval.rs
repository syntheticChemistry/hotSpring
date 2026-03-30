// SPDX-License-Identifier: AGPL-3.0-only

//! Master precision evaluation benchmark.
//!
//! Runs all evaluation phases with safe hardware probing:
//! 0. Hardware calibration (probe each tier — cannot poison the device)
//! 1. Transfer profiles per card
//! 2. Per-shader precision/throughput matrix (only safe tiers)
//! 3. Full physics pipeline end-to-end
//! 4. Dual-card cooperative patterns
//! 5. Brain routing table

use hotspring_barracuda::device_pair::DevicePair;
use hotspring_barracuda::dual_pipeline_eval::DualPipelineEval;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::pipeline_eval::PipelineEval;
use hotspring_barracuda::precision_brain::PrecisionBrain;
use hotspring_barracuda::precision_eval::{
    EVAL_SHADER_EXP_LOG, EVAL_SHADER_KAHAN_SUM, EVAL_SHADER_SQUARE_PLUS_ONE, PrecisionEval,
};
use hotspring_barracuda::transfer_eval::TransferEval;
use std::time::Instant;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(run());
}

async fn run() {
    let t_total = Instant::now();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  3-Tier Precision Evaluation: Heterogeneous GPU Pipeline");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // ── Discover GPUs ────────────────────────────────────────────────
    println!("── GPU Discovery ───────────────────────────────────────────");
    GpuF64::print_available_adapters();
    println!();

    let pair_result = DevicePair::discover().await;

    let (single_gpu, pair): (Option<GpuF64>, Option<DevicePair>) = if let Ok(p) = pair_result {
        println!("  Device pair discovered:");
        println!("    Precise:    {}", p.profile.precise_name);
        println!("    Throughput: {}", p.profile.throughput_name);
        println!(
            "    Bridge:     {:.1} GB/s, {:.0}us latency",
            p.profile.bridge_bandwidth_gbps, p.profile.bridge_latency_us
        );
        println!();
        (None, Some(p))
    } else {
        println!("  Single-GPU mode (no device pair found)");
        let gpu = GpuF64::new().await.expect("at least one GPU required");
        println!("  Using: {}", gpu.adapter_name);
        println!();
        (Some(gpu), None)
    };

    let gpus: Vec<&GpuF64> = if let Some(ref p) = pair {
        vec![&p.precise, &p.throughput]
    } else if let Some(ref g) = single_gpu {
        vec![g]
    } else {
        vec![]
    };

    // ── Phase 0: Hardware Calibration (safe probing) ─────────────────
    println!("── Hardware Calibration ─────────────────────────────────────");
    let mut brains: Vec<PrecisionBrain> = Vec::new();
    for gpu in &gpus {
        let brain = PrecisionBrain::new(gpu);
        println!("  {}", brain.calibration);
        println!("  {brain}");
        brains.push(brain);
    }
    println!();

    // ── Phase 1: Transfer Profiles ───────────────────────────────────
    println!("── Transfer Profiles ────────────────────────────────────────");
    for gpu in &gpus {
        let t_eval = TransferEval::new(gpu);
        let profile = t_eval.profile();
        profile.print_report();
        println!();
    }

    // ── Phase 2: Shader Tier Matrix (only safe tiers) ────────────────
    println!("── Shader Tier Matrix ───────────────────────────────────────");

    let test_shaders: Vec<(&str, &str)> = vec![
        ("square_plus_one", EVAL_SHADER_SQUARE_PLUS_ONE),
        ("kahan_sum", EVAL_SHADER_KAHAN_SUM),
        ("exp_log", EVAL_SHADER_EXP_LOG),
    ];
    let n_elements = 1024;
    let input: Vec<f64> = (0..n_elements).map(|i| (i as f64 + 1.0) * 0.01).collect();
    let workgroups = (n_elements as u32).div_ceil(64);

    println!(
        "  {:<24} {:<28} {:>10} {:>10} {:>10} {:>10}",
        "Shader", "Card", "F32", "F64", "DF64", "Precise"
    );
    println!("  {}", "─".repeat(104));

    for (gpu, brain) in gpus.iter().zip(brains.iter()) {
        let safe: Vec<hotspring_barracuda::precision_routing::PrecisionTier> = [
            hotspring_barracuda::precision_routing::PrecisionTier::F32,
            hotspring_barracuda::precision_routing::PrecisionTier::F64,
            hotspring_barracuda::precision_routing::PrecisionTier::DF64,
            hotspring_barracuda::precision_routing::PrecisionTier::F64Precise,
        ]
        .into_iter()
        .filter(|&t| brain.tier_safe(t))
        .collect();

        let p_eval = PrecisionEval::new(gpu);
        for &(name, source) in &test_shaders {
            if name == "exp_log" && brain.calibration.nvvm_transcendental_risk {
                print!("  {:<24} {:<28}", name, gpu.adapter_name);
                print!(
                    " {:>10} {:>10} {:>10} {:>10}",
                    "NVVM⚠", "NVVM⚠", "NVVM⚠", "NVVM⚠"
                );
                println!();
                continue;
            }
            let result =
                p_eval.eval_shader_tiers(name, source, &input, n_elements, workgroups, &safe);
            print!("  {:<24} {:<28}", result.shader_name, gpu.adapter_name);
            for tier in &result.tiers {
                if tier.compiled {
                    print!(" {:>9.1}ms", tier.dispatch_us / 1000.0);
                } else {
                    print!(" {:>10}", "—");
                }
            }
            println!();
        }
    }
    println!();

    // ── Numerical Accuracy ───────────────────────────────────────────
    println!("── Numerical Accuracy ───────────────────────────────────────");
    println!(
        "  {:<24} {:<28} {:>18} {:>18}",
        "Shader", "Card", "DF64 vs F64 (ULP)", "F32 vs F64 (ULP)"
    );
    println!("  {}", "─".repeat(90));

    for (gpu, brain) in gpus.iter().zip(brains.iter()) {
        let safe: Vec<hotspring_barracuda::precision_routing::PrecisionTier> = [
            hotspring_barracuda::precision_routing::PrecisionTier::F32,
            hotspring_barracuda::precision_routing::PrecisionTier::F64,
            hotspring_barracuda::precision_routing::PrecisionTier::DF64,
            hotspring_barracuda::precision_routing::PrecisionTier::F64Precise,
        ]
        .into_iter()
        .filter(|&t| brain.tier_safe(t))
        .collect();

        let p_eval = PrecisionEval::new(gpu);
        for &(name, source) in &test_shaders {
            if name == "exp_log" && brain.calibration.nvvm_transcendental_risk {
                println!(
                    "  {:<24} {:<28} {:>18} {:>18}",
                    name, gpu.adapter_name, "NVVM⚠", "NVVM⚠"
                );
                continue;
            }
            let result =
                p_eval.eval_shader_tiers(name, source, &input, n_elements, workgroups, &safe);
            let format_ulp =
                |tier: hotspring_barracuda::precision_routing::PrecisionTier| -> String {
                    result.tiers.iter().find(|t| t.tier == tier).map_or_else(
                        || "—".to_string(),
                        |t| {
                            if t.compiled {
                                format!("{:.1}", t.max_ulp_error)
                            } else {
                                "—".to_string()
                            }
                        },
                    )
                };
            let df64_ulp = format_ulp(hotspring_barracuda::precision_routing::PrecisionTier::DF64);
            let f32_ulp = format_ulp(hotspring_barracuda::precision_routing::PrecisionTier::F32);
            println!(
                "  {:<24} {:<28} {:>18} {:>18}",
                name, gpu.adapter_name, df64_ulp, f32_ulp
            );
        }
    }
    println!();

    // ── Phase 3: Pipeline End-to-End ─────────────────────────────────
    println!("── Pipeline End-to-End ──────────────────────────────────────");
    println!(
        "  {:<16} {:<28} {:<10} {:>10} Accuracy",
        "Pipeline", "Card", "Tier", "Wall(ms)"
    );
    println!("  {}", "─".repeat(92));

    for gpu in &gpus {
        let p_eval = PipelineEval::new(gpu);
        let results = p_eval.run_all();
        for r in &results {
            println!("{}", r.report_line());
        }
    }
    println!();

    // ── Phase 4: Dual-Card Cooperative ───────────────────────────────
    if let Some(ref p) = pair {
        println!("── Dual-Card Cooperative ────────────────────────────────────");
        println!(
            "  {:<28} {:>10} {:>14} Details",
            "Pattern", "Wall(ms)", "vs Single"
        );
        println!("  {}", "─".repeat(80));

        let dual_eval = DualPipelineEval::new(p);
        let results = dual_eval.run_all();
        for r in &results {
            println!("{}", r.report_line());
        }
        println!();
    } else {
        println!("── Dual-Card Cooperative ────────────────────────────────────");
        println!("  (skipped — single GPU mode)");
        println!();
    }

    // ── Phase 5: Brain Routing Summary ───────────────────────────────
    println!("── Brain Routing Table ─────────────────────────────────────");
    for brain in &brains {
        println!("  {brain}");
    }

    // ── Summary ──────────────────────────────────────────────────────
    let total_s = t_total.elapsed().as_secs_f64();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Total evaluation time: {total_s:.1}s");
    println!("═══════════════════════════════════════════════════════════════");
}
