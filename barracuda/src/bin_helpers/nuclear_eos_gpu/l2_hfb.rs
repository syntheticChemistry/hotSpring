// SPDX-License-Identifier: AGPL-3.0-or-later

//! L2 HFB baseline, DirectSampler optimization, and summary reporting.

use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;
use barracuda::sample::direct::{DirectSamplerConfig, direct_sampler};
use hotspring_barracuda::bench::{BenchReport, PhaseResult, PowerMonitor, peak_rss_mb};
use hotspring_barracuda::data;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::nuclear_eos_helpers::l2_objective_nmp_exp_data;
use hotspring_barracuda::physics::{binding_energy_l2, nuclear_matter_properties};
use hotspring_barracuda::provenance;
use hotspring_barracuda::tolerances;

pub struct L2HfbResult {
    pub l2_chi2: f64,
    pub l2_count: usize,
    pub l2_converged: usize,
    pub l2_time: f64,
    pub l2_opt_chi2: f64,
}

/// L2 HFB: CPU evaluation with SLy4 baseline.
pub fn run_l2_hfb_baseline(
    report: &mut BenchReport,
    sorted_nuclei: &[((usize, usize), (f64, f64))],
) -> L2HfbResult {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  L2 HFB: CPU (SLy4) — baseline for GPU comparison");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let pmon_l2_sly4 = PowerMonitor::start();
    let t_l2 = Instant::now();
    let mut l2_chi2 = 0.0f64;
    let mut l2_count = 0usize;
    let mut l2_converged = 0usize;

    for &((z, n), (b_exp, _)) in sorted_nuclei {
        let a = z + n;
        if (56..=132).contains(&a) {
            let (b_calc, converged) =
                binding_energy_l2(z, n, &provenance::SLY4_PARAMS).expect("HFB solve");
            if b_calc > 0.0 {
                let sigma_theo = tolerances::sigma_theo(b_exp);
                l2_chi2 += ((b_calc - b_exp) / sigma_theo).powi(2);
                l2_count += 1;
                if converged {
                    l2_converged += 1;
                }
            }
        }
    }
    if l2_count > 0 {
        l2_chi2 /= l2_count as f64;
    }
    let l2_time = t_l2.elapsed().as_secs_f64();
    let energy_l2_sly4 = pmon_l2_sly4.stop();

    report.add_phase(PhaseResult {
        phase: "L2 HFB SLy4".into(),
        substrate: "BarraCuda CPU".into(),
        wall_time_s: l2_time,
        per_eval_us: l2_time * 1e6 / l2_count.max(1) as f64,
        n_evals: l2_count,
        energy: energy_l2_sly4,
        peak_rss_mb: peak_rss_mb(),
        chi2: l2_chi2,
        precision_mev: 0.0,
        notes: format!("{l2_converged}/{l2_count} converged"),
    });

    println!("  L2 CPU (SLy4):");
    println!("    chi2/datum = {l2_chi2:.2}");
    println!("    Nuclei: {l2_converged}/{l2_count} converged");
    println!(
        "    Time: {:.1}s ({:.1}s/nucleus avg)",
        l2_time,
        l2_time / l2_count.max(1) as f64
    );

    L2HfbResult {
        l2_chi2,
        l2_count,
        l2_converged,
        l2_time,
        l2_opt_chi2: 0.0,
    }
}

/// L2 DirectSampler optimization (CPU + rayon parallelism).
pub fn run_l2_direct_sampler(
    report: &mut BenchReport,
    ctx: &data::EosContext,
    bounds: &[(f64, f64)],
    device: Arc<WgpuDevice>,
    l2_count: usize,
) -> f64 {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  L2 Optimization: DirectSampler (CPU + rayon)");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let exp_data_arc = ctx.exp_data.clone();
    let exp_data_l2 = exp_data_arc;

    let l2_objective = move |x: &[f64]| -> f64 { l2_objective_nmp_exp_data(x, &exp_data_l2, 0.1) };

    let pmon_l2_opt = PowerMonitor::start();
    let t_l2_opt = Instant::now();
    let l2_config = DirectSamplerConfig::new(42)
        .with_rounds(3)
        .with_solvers(4)
        .with_eval_budget(30)
        .with_patience(2);

    println!(
        "  Config: {} rounds × {} solvers × {} evals (patience={})",
        l2_config.n_rounds, l2_config.n_solvers, l2_config.max_eval_per_solver, l2_config.patience
    );
    println!("  Running... (each eval = full HFB SCF for ~{l2_count} nuclei)");

    let result_l2 =
        direct_sampler(device, l2_objective, bounds, &l2_config).expect("L2 DirectSampler failed");
    let l2_opt_time = t_l2_opt.elapsed().as_secs_f64();
    let energy_l2_opt = pmon_l2_opt.stop();
    let l2_opt_chi2 = result_l2.f_best.exp_m1();

    report.add_phase(PhaseResult {
        phase: "L2 DirectSampler".into(),
        substrate: "BarraCuda CPU".into(),
        wall_time_s: l2_opt_time,
        per_eval_us: l2_opt_time * 1e6 / result_l2.cache.len().max(1) as f64,
        n_evals: result_l2.cache.len(),
        energy: energy_l2_opt,
        peak_rss_mb: peak_rss_mb(),
        chi2: l2_opt_chi2,
        precision_mev: 0.0,
        notes: format!("{} evals, rayon parallel HFB", result_l2.cache.len()),
    });

    println!();
    println!("  L2 DirectSampler result:");
    println!(
        "    chi2/datum = {:.2} ({} evals in {:.1}s)",
        l2_opt_chi2,
        result_l2.cache.len(),
        l2_opt_time
    );
    println!(
        "    Time per eval: {:.1}s",
        l2_opt_time / result_l2.cache.len().max(1) as f64
    );

    if let Some(nmp_l2) = nuclear_matter_properties(&result_l2.x_best) {
        println!(
            "    NMP chi2: {:.4}",
            provenance::nmp_chi2_from_props(&nmp_l2) / 5.0
        );
    }

    l2_opt_chi2
}

/// Print three-way comparison summary and GPU validation status.
pub fn print_summary_table(
    cpu_full_chi2: f64,
    gpu_full_chi2: f64,
    l2_opt_chi2: f64,
    max_diff: f64,
    gpu_arc: &Arc<GpuF64>,
) {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  SUMMARY: Three-Way Comparison");
    println!("══════════════════════════════════════════════════════════════");
    println!();
    println!("  ┌─────────┬────────────────┬────────────────┬────────────────┐");
    println!("  │ Level   │ Python/SciPy   │ BarraCuda CPU  │ BarraCuda GPU  │");
    println!("  ├─────────┼────────────────┼────────────────┼────────────────┤");
    println!(
        "  │ L1 SEMF │ chi2 ~{:.2}     │ chi2 = {cpu_full_chi2:<7.2} │ chi2 = {gpu_full_chi2:<7.2} │",
        provenance::L1_PYTHON_CHI2.value
    );
    println!(
        "  │ L2 HFB  │ chi2 ~{:.2}    │ chi2 = {l2_opt_chi2:<7.2} │ (SCF on CPU)   │",
        provenance::L2_PYTHON_CHI2.value
    );
    println!("  └─────────┴────────────────┴────────────────┴────────────────┘");
    println!();
    println!("  GPU FP64 Validation:");
    println!("    SEMF precision: max |delta| = {max_diff:.2e} MeV");
    println!("    IEEE 754 compliance: 0 ULP (verified by fp64_validation)");
    println!("    Adapter: {}", gpu_arc.adapter_name);
    println!("    SHADER_F64: ENABLED");
    println!();
    println!("  GPU evolution roadmap:");
    println!("    [x] L1 SEMF — batched f64 compute shader");
    println!("    [x] L1 chi2 — batched f64 reduction shader");
    println!("    [ ] L2 density accumulation — batched across nuclei on GPU");
    println!("    [ ] L2 Skyrme potential — element-wise f64 on GPU");
    println!("    [ ] L2 Coulomb — prefix-sum f64 on GPU");
    println!("    [ ] L2 eigh_f64 — batched eigendecomposition shader");
    println!("    [ ] L3 2D grid operations — f64 on GPU");
}
