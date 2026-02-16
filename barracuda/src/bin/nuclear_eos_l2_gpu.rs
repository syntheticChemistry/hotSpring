// SPDX-License-Identifier: AGPL-3.0-only

//! Nuclear EOS Level 2 — GPU-Batched HFB Pipeline
//!
//! Uses `BatchedEighGpu` from toadstool for O(1) GPU dispatches per SCF iteration,
//! replacing `O(N_nuclei)` sequential eigensolves.
//!
//! Architecture:
//!   1. L1 LHS screening with NMP constraint (CPU, fast)
//!   2. L2 GPU-batched HFB evaluation per optimizer step
//!   3. Statistical analysis and comparison
//!
//! Run: cargo run --release --bin `nuclear_eos_l2_gpu` [--nuclei=full] [--seed=42]
//!      [--max-iter=200] [--tol=0.05] [--mixing=0.3]

use hotspring_barracuda::data;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::physics::{
    binding_energies_l2_gpu, binding_energies_l2_gpu_resident, binding_energy_l2,
    nuclear_matter_properties, NuclearMatterProps,
};
use hotspring_barracuda::provenance;
use hotspring_barracuda::tolerances;

use barracuda::sample::latin_hypercube;

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
// NMP chi2 (uses provenance::NMP_TARGETS)
// ═══════════════════════════════════════════════════════════════════

fn nmp_chi2_per_datum(nmp: &NuclearMatterProps) -> f64 {
    provenance::nmp_chi2_from_props(nmp) / 5.0
}

// ═══════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════

struct CliArgs {
    seed: u64,
    max_iter: usize,
    tol: f64,
    mixing: f64,
    n_lhs: usize,
    phase1_only: bool,
    gpu_resident: bool,
    cpu_only: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let get = |prefix: &str| -> Option<String> {
        args.iter()
            .find(|a| a.starts_with(prefix))
            .map(|a| a[prefix.len()..].to_string())
    };
    let has = |flag: &str| -> bool { args.iter().any(|a| a == flag) };

    CliArgs {
        seed: get("--seed=").and_then(|s| s.parse().ok()).unwrap_or(42),
        max_iter: get("--max-iter=")
            .and_then(|s| s.parse().ok())
            .unwrap_or(200),
        tol: get("--tol=").and_then(|s| s.parse().ok()).unwrap_or(0.05),
        mixing: get("--mixing=").and_then(|s| s.parse().ok()).unwrap_or(0.3),
        n_lhs: get("--n-lhs=").and_then(|s| s.parse().ok()).unwrap_or(64),
        phase1_only: has("--phase1-only"),
        gpu_resident: has("--gpu-resident"),
        cpu_only: has("--cpu-only"),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let cli = parse_args();

    if cli.cpu_only {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  Nuclear EOS L2 — CPU-ONLY BASELINE                        ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
    } else if cli.gpu_resident {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  Nuclear EOS L2 — GPU-RESIDENT HFB (Experiment 005b)       ║");
        println!("║  Potentials + H-build on GPU, zero CPU↔GPU H round-trips   ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
    } else {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  Nuclear EOS L2 — GPU-Batched HFB (BatchedEighGpu)         ║");
        println!("║  toadstool BatchedEighGpu → O(1) dispatches per SCF iter   ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
    println!();

    // ── Load data (shared by CPU-only and GPU paths) ─────────────────
    let ctx = data::load_eos_context();

    // ── CPU-only fast path (no GPU needed) ─────────────────────────
    if cli.cpu_only {
        let mut sorted_nuclei: Vec<((usize, usize), (f64, f64))> =
            ctx.exp_data.iter().map(|(&k, &v)| (k, v)).collect();
        sorted_nuclei.sort_by_key(|&((z, n), _)| (z, n));
        let nuclei_zn: Vec<(usize, usize)> =
            sorted_nuclei.iter().map(|&((z, n), _)| (z, n)).collect();
        let b_exp_map: HashMap<(usize, usize), f64> = sorted_nuclei
            .iter()
            .map(|&((z, n), (b, _))| ((z, n), b))
            .collect();

        println!("  Nuclei: {} total", nuclei_zn.len());
        println!();
        let t_cpu = Instant::now();
        let mut chi2_sum = 0.0;
        let mut n_valid = 0usize;
        let mut n_conv = 0usize;
        let mut n_hfb = 0usize;
        for &(z, n) in &nuclei_zn {
            let (b_calc, conv) = binding_energy_l2(z, n, &provenance::SLY4_PARAMS);
            let a = z + n;
            if (56..=132).contains(&a) {
                n_hfb += 1;
            }
            if b_calc > 0.0 {
                if let Some(&b_exp) = b_exp_map.get(&(z, n)) {
                    let sigma = tolerances::sigma_theo(b_exp);
                    chi2_sum += ((b_calc - b_exp) / sigma).powi(2);
                    n_valid += 1;
                    if conv {
                        n_conv += 1;
                    }
                }
            }
        }
        let cpu_time = t_cpu.elapsed().as_secs_f64();
        let chi2_d = if n_valid > 0 {
            chi2_sum / n_valid as f64
        } else {
            1e10
        };
        println!("  CPU-Only SLy4:");
        println!("    chi2/datum:  {chi2_d:.4}");
        println!("    Converged:   {n_conv}/{n_hfb} HFB nuclei");
        println!(
            "    Wall time:   {:.2}s ({:.1}ms per HFB nucleus)",
            cpu_time,
            if n_hfb > 0 {
                cpu_time * 1000.0 / n_hfb as f64
            } else {
                0.0
            }
        );
        return;
    }

    // ── Initialize GPU ──────────────────────────────────────────────
    let rt = tokio::runtime::Runtime::new().unwrap();
    let gpu = rt
        .block_on(GpuF64::new())
        .expect("Failed to create GPU device");
    print!("  ");
    gpu.print_info();
    if !gpu.has_f64 {
        println!("\n  SHADER_F64 not supported — cannot run FP64 BatchedEighGpu.");
        return;
    }

    // Bridge to toadstool WgpuDevice
    let wgpu_device = gpu.to_wgpu_device();
    println!("  WgpuDevice bridge: OK");
    println!();

    // ── Use loaded data ──────────────────────────────────────────────
    let base = &ctx.base;
    let exp_data = &*ctx.exp_data;
    let bounds = &ctx.bounds;

    // Sort nuclei for deterministic ordering
    let mut sorted_nuclei: Vec<((usize, usize), (f64, f64))> =
        exp_data.iter().map(|(&k, &v)| (k, v)).collect();
    sorted_nuclei.sort_by_key(|&((z, n), _)| (z, n));

    let nuclei_zn: Vec<(usize, usize)> = sorted_nuclei.iter().map(|&((z, n), _)| (z, n)).collect();
    let b_exp_map: HashMap<(usize, usize), f64> = sorted_nuclei
        .iter()
        .map(|&((z, n), (b, _))| ((z, n), b))
        .collect();

    // Count HFB-range nuclei
    let n_hfb_range = nuclei_zn
        .iter()
        .filter(|(z, n)| {
            let a = z + n;
            (56..=132).contains(&a)
        })
        .count();

    println!(
        "  Nuclei:      {} total ({} in HFB range A=56-132)",
        nuclei_zn.len(),
        n_hfb_range
    );
    println!("  Parameters:  {} dimensions (Skyrme)", bounds.len());
    println!(
        "  SCF config:  max_iter={}, tol={:.3}, mixing={:.2}",
        cli.max_iter, cli.tol, cli.mixing
    );
    println!("  LHS samples: {}", cli.n_lhs);
    println!();

    // ═══════════════════════════════════════════════════════════════
    //  PHASE 1: SLy4 baseline — CPU vs GPU comparison
    // ═══════════════════════════════════════════════════════════════
    println!("══════════════════════════════════════════════════════════════");
    println!("  PHASE 1: SLy4 Baseline — GPU-Batched HFB");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let t_gpu = Instant::now();

    // Choose pipeline: GPU-resident (shaders for H-build) or mega-batch (CPU H-build)
    let (gpu_results_vec, gpu_disp, n_hfb_out, n_semf_out, total_disp) = if cli.gpu_resident {
        let r = binding_energies_l2_gpu_resident(
            wgpu_device.clone(),
            &nuclei_zn,
            &provenance::SLY4_PARAMS,
            cli.max_iter,
            cli.tol,
            cli.mixing,
        );
        let d = r.gpu_dispatches;
        let td = r.total_gpu_dispatches;
        let nh = r.n_hfb;
        let ns = r.n_semf;
        (r.results, d, nh, ns, td)
    } else {
        let r = binding_energies_l2_gpu(
            wgpu_device.clone(),
            &nuclei_zn,
            &provenance::SLY4_PARAMS,
            cli.max_iter,
            cli.tol,
            cli.mixing,
        );
        let d = r.gpu_dispatches;
        let nh = r.n_hfb;
        let ns = r.n_semf;
        (r.results, d, nh, ns, d)
    };
    let gpu_time = t_gpu.elapsed().as_secs_f64();

    // Compute chi2
    let mut chi2_sum = 0.0;
    let mut n_valid = 0usize;
    let mut n_converged = 0usize;

    for &(z, n, b_calc, converged) in &gpu_results_vec {
        if b_calc > 0.0 {
            if let Some(&b_exp) = b_exp_map.get(&(z, n)) {
                let sigma = tolerances::sigma_theo(b_exp);
                chi2_sum += ((b_calc - b_exp) / sigma).powi(2);
                n_valid += 1;
                if converged {
                    n_converged += 1;
                }
            }
        }
    }
    let chi2_datum = if n_valid > 0 {
        chi2_sum / n_valid as f64
    } else {
        1e10
    };

    let engine_label = if cli.gpu_resident {
        "GPU-Resident HFB"
    } else {
        "GPU-Batched HFB"
    };
    println!("  {engine_label} (SLy4):");
    println!("    chi2/datum:     {chi2_datum:.4}");
    println!("    Converged:      {n_converged}/{n_hfb_out} HFB nuclei");
    println!("    SEMF fallback:  {n_semf_out} nuclei");
    println!("    GPU dispatches: {total_disp} (eigh: {gpu_disp}, total: {total_disp})");
    println!(
        "    Wall time:      {:.2}s ({:.1}ms per HFB nucleus)",
        gpu_time,
        if n_hfb_out > 0 {
            gpu_time * 1000.0 / n_hfb_out as f64
        } else {
            0.0
        }
    );

    let sly4_nmp = nuclear_matter_properties(&provenance::SLY4_PARAMS);
    if let Some(ref nmp) = sly4_nmp {
        let chi2_nmp = nmp_chi2_per_datum(nmp);
        println!("    NMP chi2/datum: {chi2_nmp:.4}");
    }
    println!();

    // ── Save Phase 1 results to JSON (incremental) ─────────────────
    let results_dir = base.join("results");
    std::fs::create_dir_all(&results_dir).ok();

    // Per-nucleus Phase 1 detail
    let phase1_nuclei: Vec<serde_json::Value> = gpu_results_vec
        .iter()
        .map(|&(z, n, b_calc, conv)| {
            let b_exp = b_exp_map.get(&(z, n)).copied().unwrap_or(0.0);
            serde_json::json!({
                "Z": z, "N": n, "A": z + n,
                "B_exp": b_exp, "B_calc": b_calc,
                "converged": conv,
                "error_MeV": b_calc - b_exp,
            })
        })
        .collect();

    let phase1_json = serde_json::json!({
        "level": 2,
        "engine": if cli.gpu_resident { "barracuda::l2_gpu_resident" } else { "barracuda::l2_gpu_batched_hfb" },
        "phase": "sly4_baseline",
        "n_nuclei": nuclei_zn.len(),
        "n_hfb": n_hfb_out,
        "n_semf": n_semf_out,
        "chi2_per_datum": chi2_datum,
        "n_converged": n_converged,
        "gpu_dispatches": total_disp,
        "wall_time_s": gpu_time,
        "scf_config": {
            "max_iter": cli.max_iter,
            "tol": cli.tol,
            "mixing": cli.mixing,
        },
        "nmp": sly4_nmp.as_ref().map(|n| serde_json::json!({
            "rho0_fm3": n.rho0_fm3, "e_a_mev": n.e_a_mev,
            "k_inf_mev": n.k_inf_mev, "m_eff_ratio": n.m_eff_ratio, "j_mev": n.j_mev,
        })),
        "nuclei": phase1_nuclei,
    });
    let p1_path = results_dir.join("barracuda_l2_gpu_phase1.json");
    std::fs::write(
        &p1_path,
        serde_json::to_string_pretty(&phase1_json).unwrap(),
    )
    .ok();
    println!("  Phase 1 results saved: {}", p1_path.display());
    println!();

    if cli.phase1_only {
        println!("  --phase1-only: skipping LHS sweep.");
        return;
    }

    // ═══════════════════════════════════════════════════════════════
    //  PHASE 2: LHS sweep with GPU-batched HFB objective
    // ═══════════════════════════════════════════════════════════════
    println!("══════════════════════════════════════════════════════════════");
    println!(
        "  PHASE 2: LHS Sweep with GPU-Batched HFB ({} samples)",
        cli.n_lhs
    );
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let samples = latin_hypercube(cli.n_lhs, bounds, cli.seed).expect("LHS failed");

    let t_sweep = Instant::now();
    let mut best_chi2 = f64::MAX;
    let mut best_idx = 0usize;
    let mut best_params_sweep: Option<Vec<f64>> = None;
    let mut total_dispatches = 0usize;
    let mut n_evaluated = 0usize;
    let mut sweep_results: Vec<serde_json::Value> = Vec::new();

    for (i, params) in samples.iter().enumerate() {
        // Quick NMP pre-screen
        if params[8] <= 0.01 || params[8] > 1.0 {
            continue;
        }
        let nmp = match nuclear_matter_properties(params) {
            Some(n) => n,
            None => continue,
        };
        if nmp.rho0_fm3 < 0.05 || nmp.rho0_fm3 > 0.30 || nmp.e_a_mev > 0.0 {
            continue;
        }

        let t_eval = Instant::now();
        let result = binding_energies_l2_gpu(
            wgpu_device.clone(),
            &nuclei_zn,
            params,
            cli.max_iter,
            cli.tol,
            cli.mixing,
        );
        let eval_time = t_eval.elapsed().as_secs_f64();
        total_dispatches += result.gpu_dispatches;
        n_evaluated += 1;

        // chi2
        let mut chi2 = 0.0;
        let mut cnt = 0;
        for &(z, n, b_calc, _) in &result.results {
            if b_calc > 0.0 {
                if let Some(&b_exp) = b_exp_map.get(&(z, n)) {
                    let sigma = tolerances::sigma_theo(b_exp);
                    chi2 += ((b_calc - b_exp) / sigma).powi(2);
                    cnt += 1;
                }
            }
        }
        if cnt > 0 {
            let chi2_d = chi2 / f64::from(cnt);
            let nmp_penalty = nmp_chi2_per_datum(&nmp) * 0.1;
            let total = chi2_d + nmp_penalty;

            sweep_results.push(serde_json::json!({
                "sample": i,
                "chi2_be": chi2_d,
                "chi2_nmp": nmp_penalty,
                "chi2_total": total,
                "n_converged": result.results.iter().filter(|(_, _, _, c)| *c).count(),
                "gpu_dispatches": result.gpu_dispatches,
                "wall_time_s": eval_time,
                "nmp": { "rho0": nmp.rho0_fm3, "e_a": nmp.e_a_mev, "k_inf": nmp.k_inf_mev, "m_eff": nmp.m_eff_ratio, "j": nmp.j_mev },
            }));

            if total < best_chi2 {
                best_chi2 = total;
                best_idx = i;
                best_params_sweep = Some(params.clone());
                println!(
                    "    [{:>3}] chi2_BE={:.4}  NMP={:.4}  total={:.4}  ({:.0}s, dispatches={})",
                    i, chi2_d, nmp_penalty, total, eval_time, result.gpu_dispatches
                );
                let _ = std::io::stdout().flush();
            }
        }
    }

    let sweep_time = t_sweep.elapsed().as_secs_f64();
    println!();
    println!("  LHS Sweep complete:");
    println!("    Best total chi2: {best_chi2:.4} (sample {best_idx})");
    println!(
        "    Evaluated:       {}/{} (passed NMP screen)",
        n_evaluated, cli.n_lhs
    );
    println!(
        "    Wall time:       {:.1}s ({:.2}s per eval avg)",
        sweep_time,
        if n_evaluated > 0 {
            sweep_time / n_evaluated as f64
        } else {
            0.0
        }
    );
    println!("    GPU dispatches:  {total_dispatches} total");
    println!();

    // ═══════════════════════════════════════════════════════════════
    //  SUMMARY
    // ═══════════════════════════════════════════════════════════════
    println!("══════════════════════════════════════════════════════════════");
    println!("  SUMMARY: GPU-Batched L2 HFB");
    println!("══════════════════════════════════════════════════════════════");
    println!();
    println!("  GPU:               {}", gpu.adapter_name);
    println!("  SHADER_F64:        YES");
    println!("  BatchedEighGpu:    ACTIVE");
    println!(
        "  Nuclei:            {} ({} HFB, {} SEMF)",
        nuclei_zn.len(),
        n_hfb_range,
        nuclei_zn.len() - n_hfb_range
    );
    println!("  SLy4 chi2/datum:   {chi2_datum:.4}");
    println!("  Best sweep chi2:   {best_chi2:.4}");
    println!();
    println!("  Architecture advantage:");
    println!(
        "    CPU (sequential eigh_f64): {} eigensolves × {} SCF iters = {} calls",
        n_hfb_range * 2,
        cli.max_iter,
        n_hfb_range * 2 * cli.max_iter
    );
    println!(
        "    GPU (BatchedEighGpu):      ~5 groups × {} iters = ~{} dispatches",
        cli.max_iter,
        5 * cli.max_iter
    );
    println!(
        "    Reduction: ~{:.0}x fewer eigensolve calls",
        (n_hfb_range * 2 * cli.max_iter) as f64 / (5 * cli.max_iter) as f64
    );
    println!();

    let names = [
        "t0", "t1", "t2", "t3", "x0", "x1", "x2", "x3", "alpha", "W0",
    ];

    // Best params
    if let Some(ref best_params) = best_params_sweep {
        println!("  Best parameters:");
        for (i, &v) in best_params.iter().enumerate() {
            println!("    {:6} = {:>12.4}", names.get(i).unwrap_or(&"?"), v);
        }

        if let Some(nmp) = nuclear_matter_properties(best_params) {
            println!();
            println!("  Best NMP:");
            let vals = [
                nmp.rho0_fm3,
                nmp.e_a_mev,
                nmp.k_inf_mev,
                nmp.m_eff_ratio,
                nmp.j_mev,
            ];
            let targets = provenance::NMP_TARGETS.values();
            let sigmas = provenance::NMP_TARGETS.sigmas();
            let names = ["rho0", "E/A", "K_inf", "m*/m", "J"];
            for (i, &v) in vals.iter().enumerate() {
                let pull = (v - targets[i]) / sigmas[i];
                println!(
                    "    {:>6} = {:>10.4}  (pull: {:>+.2}sigma)",
                    names[i], v, pull
                );
            }
        }
    }

    // ── Save full results to JSON ──────────────────────────────────
    let sweep_json = serde_json::json!({
        "level": 2,
        "engine": "barracuda::l2_gpu_batched_hfb",
        "n_nuclei": nuclei_zn.len(),
        "n_hfb": n_hfb_range,
        "n_semf": nuclei_zn.len() - n_hfb_range,
        "seed": cli.seed,
        "scf_config": {
            "max_iter": cli.max_iter,
            "tol": cli.tol,
            "mixing": cli.mixing,
        },
        "sly4_chi2_per_datum": chi2_datum,
        "sly4_wall_time_s": gpu_time,
        "n_lhs_requested": cli.n_lhs,
        "n_lhs_evaluated": n_evaluated,
        "sweep_wall_time_s": sweep_time,
        "total_gpu_dispatches": total_dispatches,
        "best_sample": best_idx,
        "best_chi2_total": best_chi2,
        "best_params": best_params_sweep.as_ref().map(|p| {
            let mut map = serde_json::Map::new();
            for (i, &v) in p.iter().enumerate() {
                map.insert(names.get(i).unwrap_or(&"?").to_string(), serde_json::json!(v));
            }
            serde_json::Value::Object(map)
        }),
        "sweep_results": sweep_results,
    });
    let result_path = results_dir.join("barracuda_l2_gpu_sweep.json");
    std::fs::write(
        &result_path,
        serde_json::to_string_pretty(&sweep_json).unwrap(),
    )
    .ok();
    println!();
    println!("  Results saved: {}", result_path.display());
}
