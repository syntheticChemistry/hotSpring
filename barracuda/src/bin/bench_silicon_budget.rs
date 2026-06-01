// SPDX-License-Identifier: AGPL-3.0-or-later

//! Silicon budget calculator: theoretical peak throughput per GPU per silicon unit.

use hotspring_barracuda::bin_helpers::silicon_budget::{
    classify_vendor, lookup_silicon_specs, print_budget, print_compound_budget,
    print_precision_tier_analysis, print_working_set_analysis, GpuSiliconBudget,
};
use hotspring_barracuda::toadstool_report::{self, PerformanceMeasurement};

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Silicon Budget Calculator");
    println!("  Theoretical peak throughput per GPU per silicon unit");
    println!("═══════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    if adapters.is_empty() {
        eprintln!("No GPU adapters found.");
        std::process::exit(1);
    }

    let mut measurements: Vec<PerformanceMeasurement> = Vec::new();
    let ts = toadstool_report::epoch_now();

    for adapter in &adapters {
        let info = adapter.get_info();
        let vendor = classify_vendor(&info);
        let budget = lookup_silicon_specs(&info.name, vendor);

        println!("━━━ {} ━━━\n", info.name);

        print_budget(&budget);
        print_compound_budget(&budget);
        print_working_set_analysis(&budget);
        print_precision_tier_analysis(&budget);

        let units = [
            ("theoretical.shader_core.fp32", budget.fp32_tflops),
            ("theoretical.shader_core.df64", budget.df64_tflops),
            ("theoretical.shader_core.fp64", budget.fp64_tflops),
            ("theoretical.memory.bandwidth_gbs", budget.memory_bw_gbs),
            ("theoretical.tmu.gtexels", budget.tmu_gtexels),
            ("theoretical.rop.gpixels", budget.rop_gpixels),
            ("theoretical.tensor.fp16", budget.tensor_fp16_tflops),
            ("theoretical.tensor.tf32", budget.tensor_tf32_tflops),
        ];

        for (op, value) in &units {
            if *value > 0.0 {
                measurements.push(PerformanceMeasurement {
                    operation: (*op).to_string(),
                    silicon_unit: op.split('.').nth(1).unwrap_or("unknown").to_string(),
                    precision_mode: "theoretical_peak".into(),
                    throughput_gflops: *value * 1000.0,
                    tolerance_achieved: 0.0,
                    gpu_model: info.name.clone(),
                    measured_by: "hotSpring/bench_silicon_budget".into(),
                    timestamp: ts,
                });
            }
        }

        println!();
    }

    if adapters.len() >= 2 {
        print_cross_gpu_comparison(&adapters);
    }

    println!(
        "\n── Reporting {} measurements to toadStool ──\n",
        measurements.len()
    );
    toadstool_report::report_to_toadstool(&measurements);

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Silicon Budget Calculator Complete");
    println!("═══════════════════════════════════════════════════════════");
}

fn print_cross_gpu_comparison(adapters: &[wgpu::Adapter]) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Cross-GPU Comparison");
    println!("═══════════════════════════════════════════════════════════\n");

    let budgets: Vec<GpuSiliconBudget> = adapters
        .iter()
        .map(|a| {
            let info = a.get_info();
            lookup_silicon_specs(&info.name, classify_vendor(&info))
        })
        .filter(|b| b.fp32_tflops > 0.0)
        .collect();

    if budgets.len() < 2 {
        return;
    }

    println!(
        "  {:<28} {:>10} {:>10} {:>10} {:>10}",
        "Metric",
        &budgets[0].name[..budgets[0].name.len().min(10)],
        &budgets[1].name[..budgets[1].name.len().min(10)],
        "Ratio",
        "Advantage"
    );
    println!("  {}", "─".repeat(72));

    let comparisons: &[(&str, f64, f64)] = &[
        (
            "FP32 TFLOPS",
            budgets[0].fp32_tflops,
            budgets[1].fp32_tflops,
        ),
        (
            "DF64 TFLOPS",
            budgets[0].df64_tflops,
            budgets[1].df64_tflops,
        ),
        (
            "FP64 TFLOPS",
            budgets[0].fp64_tflops,
            budgets[1].fp64_tflops,
        ),
        (
            "Memory GB/s",
            budgets[0].memory_bw_gbs,
            budgets[1].memory_bw_gbs,
        ),
        ("TMU GT/s", budgets[0].tmu_gtexels, budgets[1].tmu_gtexels),
        ("ROP GP/s", budgets[0].rop_gpixels, budgets[1].rop_gpixels),
        (
            "L2+IC (MB)",
            (budgets[0].l2_bytes + budgets[0].infinity_cache_bytes) as f64 / 1_048_576.0,
            (budgets[1].l2_bytes + budgets[1].infinity_cache_bytes) as f64 / 1_048_576.0,
        ),
        (
            "VRAM (GB)",
            budgets[0].vram_bytes as f64 / 1e9,
            budgets[1].vram_bytes as f64 / 1e9,
        ),
    ];

    for (metric, a, b) in comparisons {
        if *a > 0.0 && *b > 0.0 {
            let ratio = a / b;
            let advantage = if ratio > 1.05 {
                &budgets[0].name
            } else if ratio < 0.95 {
                &budgets[1].name
            } else {
                "~parity"
            };
            println!("  {metric:<28} {a:>10.2} {b:>10.2} {ratio:>10.2}x  {advantage}");
        }
    }
}
