// SPDX-License-Identifier: AGPL-3.0-or-later

//! Production silicon-instrumented QCD runner.
//!
//! Runs the full QCD flavor ladder (quenched, Nf=1,2,2+1,3,4,2+1+1) at specified
//! lattice sizes with per-trajectory timing, FLOP counting, energy
//! accounting (RAPL + nvidia-smi/amdgpu), and QUDA comparison columns.
//!
//! # Usage
//!
//! ```bash
//! # Quenched 8^4 beta-scan on default GPU:
//! cargo run --release --bin production_silicon_qcd -- \
//!   --mode=quenched --lattice=8 --betas=5.5,5.69,6.0,6.2 --therm=200 --meas=100
//!
//! # Nf=2 RHMC at 16^4:
//! cargo run --release --bin production_silicon_qcd -- \
//!   --mode=nf2 --lattice=16 --betas=5.5,6.0 --mass=0.1 --therm=100 --meas=50
//!
//! # Nf=2+1 at 16^4 with gradient flow:
//! cargo run --release --bin production_silicon_qcd -- \
//!   --mode=nf2+1 --lattice=16 --betas=6.0 --mass=0.05 --strange-mass=0.5 \
//!   --therm=100 --meas=50 --flow --flow-configs=10
//!
//! # 32^4 quenched with specific GPU:
//! HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin production_silicon_qcd -- \
//!   --mode=quenched --lattice=32 --betas=5.5,6.0 --therm=200 --meas=100
//! ```

use hotspring_barracuda::bench::GpuTelemetry;
use hotspring_barracuda::bin_helpers::silicon_qcd::{
    AmdGpuPower, BetaSummary, lookup_budget, mode_label, parse_args, run_beta_point,
};
use hotspring_barracuda::gpu::GpuF64;

use std::io::Write;
use std::time::Instant;

fn main() {
    let args = parse_args();
    let l = args.lattice;
    let dims = [l, l, l, l];
    let vol = l * l * l * l;
    let label = mode_label(&args);

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  Production Silicon-Instrumented QCD                       ║");
    eprintln!("║  ecoPrimals · Exp 105 · Silicon Revalidation Campaign      ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  Mode:        {label}");
    eprintln!("  Lattice:     {l}^4 ({vol} sites)");
    eprintln!("  β points:    {:?}", args.betas);
    eprintln!("  MD:          {} steps × dt={}", args.n_md_steps, args.dt);
    eprintln!(
        "  Therm:       {} (+ {} quenched pre-therm)",
        args.n_therm, args.n_quenched_pretherm
    );
    eprintln!("  Meas:        {} per β", args.n_meas);
    if args.flow {
        eprintln!(
            "  Flow:        {} configs, skip={}, ε={}, t_max={}",
            args.flow_configs, args.flow_skip, args.flow_epsilon, args.flow_t_max
        );
    }
    eprintln!();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("  GPU not available: {e}");
            std::process::exit(1);
        }
    };

    let budget = lookup_budget(&gpu.adapter_name);
    let amd_power = AmdGpuPower::detect();
    let has_amd = amd_power.hwmon_path.is_some();
    let telemetry = GpuTelemetry::start(&gpu.adapter_name);

    eprintln!("  GPU:         {}", gpu.adapter_name);
    eprintln!("  DF64:        {:.2} TFLOPS", budget.df64_tflops);
    eprintln!("  Mem BW:      {:.0} GB/s", budget.memory_bw_gbs);
    eprintln!(
        "  TMU:         {} units × {:.2} GHz = {:.0} GTexels/s",
        budget.tmu_count,
        budget.boost_ghz,
        budget.tmu_count as f64 * budget.boost_ghz
    );
    eprintln!(
        "  VRAM:        {:.1} GB",
        budget.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    eprintln!(
        "  QUDA ref:    {:.1} TFLOPS (FP64 ALU only)",
        budget.quda_style_tflops
    );
    eprintln!(
        "  Total cap:   {:.0} TFLOPS (all silicon)",
        budget.total_tflops
    );
    eprintln!(
        "  Telemetry:   {} ({})",
        telemetry.backend,
        if has_amd { "AMD sysfs" } else { "nvidia-smi" }
    );
    eprintln!();

    let mut out_file = args.output.as_ref().map(|path| {
        let f = std::fs::File::create(path).expect("create output file");
        std::io::BufWriter::new(f)
    });

    let csv_header = "mode,beta,lattice,gpu,traj,accepted,delta_h,plaquette,cg_iters,\
                      wall_ms,est_gflops,est_gflop_per_s,est_gb_s,phase";
    if let Some(f) = &mut out_file {
        writeln!(f, "{csv_header}").ok();
    }
    println!("{csv_header}");

    let run_start = Instant::now();
    let mut all_summaries: Vec<BetaSummary> = Vec::new();
    let mut seed = args.seed;

    for &beta in &args.betas {
        let summary = run_beta_point(
            &gpu,
            &budget,
            &amd_power,
            &telemetry,
            &args,
            beta,
            dims,
            vol,
            &mut seed,
            &mut out_file,
        );
        all_summaries.push(summary);
    }

    let total_time = run_start.elapsed();

    // ── Grand summary ──
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  SILICON UTILIZATION SUMMARY                               ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!(
        "  {:<8} {:<6} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "β", "acc%", "ms/traj", "GFLOP/s", "eco%", "QUDA%", "J/traj", "⟨P⟩", "σ(P)"
    );
    eprintln!("  {}", "─".repeat(88));

    for s in &all_summaries {
        eprintln!(
            "  {:<8.4} {:<6.1} {:<10.1} {:<10.1} {:<10.2} {:<10.2} {:<10.2} {:<10.6} {:<10.2e}",
            s.beta,
            s.acceptance_pct,
            s.mean_ms_per_traj,
            s.mean_gflop_per_s * 1000.0,
            s.silicon_util_pct,
            s.quda_util_pct,
            s.joules_per_traj,
            s.mean_plaq,
            s.std_plaq
        );
    }

    eprintln!();
    eprintln!("  Legend: eco% = ecoPrimals sustained / total silicon capacity");
    eprintln!("          QUDA% = QUDA-style sustained / total silicon capacity");
    eprintln!();

    // QUDA comparison table
    eprintln!("  ┌──────────────────────────────────────────────────────────────┐");
    eprintln!("  │  QUDA vs ecoPrimals Comparison                             │");
    eprintln!("  ├──────────────────────────────────────────────────────────────┤");

    let mean_eco_tflops = if all_summaries.is_empty() {
        0.0
    } else {
        all_summaries.iter().map(|s| s.eco_tflops).sum::<f64>() / all_summaries.len() as f64
    };

    eprintln!("  │  GPU:            {:<43}│", budget.name);
    eprintln!(
        "  │  Total silicon:  {:<6.0} TFLOPS{:>31}│",
        budget.total_tflops, ""
    );
    eprintln!(
        "  │  QUDA sustained: {:<6.2} TFLOPS ({:.2}% util){:>19}│",
        budget.quda_style_tflops,
        budget.quda_style_tflops / budget.total_tflops * 100.0,
        ""
    );
    eprintln!(
        "  │  ecoPrimals:     {:<6.4} TFLOPS ({:.2}% util){:>19}│",
        mean_eco_tflops,
        mean_eco_tflops / budget.total_tflops * 100.0,
        ""
    );
    eprintln!(
        "  │  Speedup:        {:<6.2}x vs QUDA-style{:>23}│",
        if budget.quda_style_tflops > 0.0 {
            mean_eco_tflops / budget.quda_style_tflops
        } else {
            0.0
        },
        ""
    );
    eprintln!("  └──────────────────────────────────────────────────────────────┘");

    // Flow results if any
    let has_flow = all_summaries.iter().any(|s| s.flow_results.is_some());
    if has_flow {
        eprintln!();
        eprintln!("  ┌──────────────────────────────────────────────────────────────┐");
        eprintln!("  │  Gradient Flow Scale Setting                                │");
        eprintln!("  ├──────────────────────────────────────────────────────────────┤");
        for s in &all_summaries {
            if let Some(ref f) = s.flow_results {
                let t0_str = f.mean_t0.map_or_else(
                    || "N/A".to_string(),
                    |v| format!("{:.4} ± {:.4}", v, f.std_t0.unwrap_or(0.0)),
                );
                let w0_str = f.mean_w0.map_or_else(
                    || "N/A".to_string(),
                    |v| format!("{:.4} ± {:.4}", v, f.std_w0.unwrap_or(0.0)),
                );
                eprintln!(
                    "  │  β={:.4}: t₀ = {:<20} w₀ = {:<18}│",
                    s.beta, t0_str, w0_str
                );
            }
        }
        eprintln!("  └──────────────────────────────────────────────────────────────┘");
    }

    // Energy summary
    let total_gpu_j: f64 = all_summaries.iter().map(|s| s.energy.gpu_joules).sum();
    let total_cpu_j: f64 = all_summaries.iter().map(|s| s.energy.cpu_joules).sum();
    let total_traj: usize = all_summaries.iter().map(|s| s.n_meas).sum();

    eprintln!();
    eprintln!(
        "  Energy: GPU {:.1} J + CPU {:.1} J = {:.1} J total ({} measurement trajectories)",
        total_gpu_j,
        total_cpu_j,
        total_gpu_j + total_cpu_j,
        total_traj
    );
    if total_traj > 0 {
        eprintln!(
            "  Average: {:.2} J/trajectory",
            (total_gpu_j + total_cpu_j) / total_traj as f64
        );
    }
    eprintln!(
        "  Total wall: {:.1}s ({:.2}h)",
        total_time.as_secs_f64(),
        total_time.as_secs_f64() / 3600.0
    );
    eprintln!();
}
