// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nuclear EOS Level 2 — Heterogeneous Pipeline
//!
//! Solves the L2 `SparsitySampler` quality problem using:
//!   1. L1 data to warm-start the search (previous work informs later)
//!   2. Three-tier pre-screening cascade to filter before expensive HFB
//!   3. NPU/CPU classifier trained on accumulated data
//!   4. Modified sampling loop where surrogate sees ONLY real HFB values
//!
//! **Heterogeneous compute architecture:**
//!   - CPU: NMP pre-screening (Tier 1, ~1μs), L1 proxy (Tier 2, ~100μs)
//!   - NPU/CPU: Learned classifier (Tier 3, ~1μs)
//!   - CPU parallel (rayon): HFB evaluation (Tier 4, ~200ms) — only for survivors
//!   - GPU (barracuda WGSL): RBF surrogate training (cdist shader)
//!
//! **Comparison modes:**
//!   - `--mode=plain`  — standard `SparsitySampler` (baseline, for comparison)
//!   - `--mode=hetero` — heterogeneous pipeline (default)

use hotspring_barracuda::bin_helpers::nuclear_eos_gpu::{
    parse_cli, run_direct_l2, run_hetero_mode, run_plain_l2, run_screen_l2,
};
use hotspring_barracuda::data;
use hotspring_barracuda::nuclear_eos_helpers::save_results;
use hotspring_barracuda::physics::nuclear_matter_properties;
use hotspring_barracuda::provenance;

use hotspring_barracuda as barracuda;

use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Nuclear EOS L2 — Heterogeneous Pipeline                   ║");
    println!("║  Architecture: L1 warm-start → cascade filter → HFB       ║");
    println!("║  Compute: CPU(NMP) → NPU/CPU(clf) → CPU∥(HFB) → GPU(RBF) ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let discovered = rt
        .block_on(barracuda::device::Auto::new())
        .expect("GPU device required for RBF surrogate (barracuda::Auto)");
    let device = discovered
        .wgpu_device()
        .expect("RBF surrogate requires local wgpu device, not sovereign IPC")
        .clone();

    let cli = parse_cli();
    let mode = cli.mode.as_str();

    let ctx = data::load_eos_context().expect("Failed to load EOS context");
    let base = &ctx.base;
    let exp_data = &*ctx.exp_data;
    let bounds = &ctx.bounds;

    println!("  Experimental nuclei: {}", exp_data.len());
    println!("  Mode: {mode}");
    println!("  Rayon threads: {}", rayon::current_num_threads());
    println!();

    let total_t0 = Instant::now();

    match mode {
        "both" | "compare" => {
            println!("═══════════════════════════════════════════════════════════");
            println!("  HETEROGENEOUS PIPELINE");
            println!("═══════════════════════════════════════════════════════════");

            let hetero = run_hetero_mode(device.clone(), &cli, bounds, exp_data);
            let hetero_time = total_t0.elapsed().as_secs_f64();
            let hetero_chi2 = hetero.best_f.exp_m1();

            println!();
            hetero.stats.print_summary();
            println!();

            println!("═══════════════════════════════════════════════════════════");
            println!("  PLAIN SPARSITYSAMPLER (COMPARISON)");
            println!("═══════════════════════════════════════════════════════════");

            let (plain_chi2, plain_time, plain_evals) =
                run_plain_l2(device, bounds, exp_data, cli.n_rounds);

            println!();
            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║  COMPARISON                                                ║");
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║  Heterogeneous:                                            ║");
            println!("║    χ²/datum = {hetero_chi2:12.2}                                  ║");
            println!("║    Time:      {hetero_time:6.1}s                                      ║");
            println!(
                "║    HFB evals: {:6}                                        ║",
                hetero.cache.len()
            );
            println!(
                "║    Cascade pass rate: {:.1}%                                ║",
                hetero.stats.pass_rate() * 100.0
            );
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║  Plain SparsitySampler:                                    ║");
            println!("║    χ²/datum = {plain_chi2:12.2}                                  ║");
            println!("║    Time:      {plain_time:6.1}s                                      ║");
            println!("║    HFB evals: {plain_evals:6}                                        ║");
            println!("╚══════════════════════════════════════════════════════════════╝");

            save_results(
                base,
                &hetero.best_params,
                hetero.best_f,
                &hetero.cache,
                &hetero.stats,
                hetero_time,
                Some(plain_chi2),
                Some(plain_time),
                Some(plain_evals),
            );
        }

        "screen" => {
            let args: Vec<String> = std::env::args().collect();
            let n_l1 = data::parse_cli_usize(&args, "--l1-samples", 10_000);
            let n_eval = data::parse_cli_usize(&args, "--l2-evals", 200);
            let (best_params, best_f, total_time, total_evals) =
                run_screen_l2(bounds, exp_data, n_l1, n_eval);
            let chi2 = best_f.exp_m1();
            let result_json = serde_json::json!({
                "level": 2,
                "engine": "barracuda::l1_screen_l2_eval",
                "chi2_per_datum": chi2,
                "log_chi2": best_f,
                "total_l2_evals": total_evals,
                "l1_samples": n_l1,
                "time_seconds": total_time,
                "best_params": best_params,
            });
            data::save_json_to_results(base, "barracuda_screen_l2.json", &result_json);
        }

        "direct" => {
            let args: Vec<String> = std::env::args().collect();
            let n_starts = data::parse_cli_usize(&args, "--starts", 10);
            let max_evals_per = data::parse_cli_usize(&args, "--evals", 200);
            let (best_params, best_f, total_time, total_evals) =
                run_direct_l2(bounds, exp_data, n_starts, max_evals_per);
            let chi2 = best_f.exp_m1();
            let result_json = serde_json::json!({
                "level": 2,
                "engine": "barracuda::direct_multi_start_nm",
                "chi2_per_datum": chi2,
                "log_chi2": best_f,
                "total_evals": total_evals,
                "n_starts": n_starts,
                "max_evals_per_start": max_evals_per,
                "time_seconds": total_time,
                "best_params": best_params,
            });
            data::save_json_to_results(base, "barracuda_direct_l2.json", &result_json);
        }

        _ => {
            let hetero = run_hetero_mode(device, &cli, bounds, exp_data);
            let total_time = total_t0.elapsed().as_secs_f64();
            let chi2 = hetero.best_f.exp_m1();
            println!();
            data::print_l2_result_box(
                "Heterogeneous L2 Results",
                chi2,
                hetero.best_f,
                hetero.cache.len(),
                total_time,
                hetero.cache.len() as f64 / total_time,
            );
            println!();

            hetero.stats.print_summary();

            if let Some(nmp) = nuclear_matter_properties(&hetero.best_params) {
                println!();
                provenance::print_nmp_analysis(&nmp);
            }

            save_results(
                base,
                &hetero.best_params,
                hetero.best_f,
                &hetero.cache,
                &hetero.stats,
                total_time,
                None,
                None,
                None,
            );
        }
    }
}
