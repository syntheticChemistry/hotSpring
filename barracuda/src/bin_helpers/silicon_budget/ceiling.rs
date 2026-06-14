// SPDX-License-Identifier: AGPL-3.0-or-later

//! Compound ceiling math for parallel silicon units on QCD workloads.

use super::specs::GpuSiliconBudget;

pub fn print_compound_budget(budget: &GpuSiliconBudget) {
    println!("\n  ── Compound Budget (parallel sub-problems) ──\n");

    let shader_equiv = budget.fp32_tflops;
    let tensor_equiv = if budget.tensor_tf32_tflops > 0.0 {
        budget.tensor_tf32_tflops * 0.3
    } else {
        0.0
    };
    let tmu_equiv = budget.tmu_gtexels * 2.0 / 1000.0;
    let rop_equiv = budget.rop_gpixels / 1000.0;

    let compound_low = shader_equiv + tmu_equiv;
    let compound_high = shader_equiv + tensor_equiv + tmu_equiv + rop_equiv;

    println!("  Shader cores alone:         {shader_equiv:>8.2} TFLOPS (FP32)");
    println!(
        "  + TMU table lookups:        {:>8.2} TFLOPS equiv ({:.1} GT/s × 2 FLOP/texel)",
        tmu_equiv, budget.tmu_gtexels
    );
    if tensor_equiv > 0.0 {
        println!(
            "  + Tensor MMA (30% util):    {:>8.2} TFLOPS equiv (of {:.1} TF32 peak)",
            tensor_equiv, budget.tensor_tf32_tflops
        );
    }
    println!(
        "  + ROP atomic blend:         {:>8.2} TFLOPS equiv ({:.1} GP/s)",
        rop_equiv, budget.rop_gpixels
    );
    println!();
    println!("  Conservative compound:      {compound_low:>8.2} TFLOPS (shader + TMU)");
    println!("  Optimistic compound:        {compound_high:>8.2} TFLOPS (all units parallel)");
    println!(
        "  Multiplier over shader:     {:.2}x – {:.2}x",
        compound_low / shader_equiv.max(0.001),
        compound_high / shader_equiv.max(0.001)
    );
}
