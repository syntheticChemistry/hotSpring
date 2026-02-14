//! Nuclear EOS Level 2 — GPU-Batched HFB Pipeline
//!
//! Uses BatchedEighGpu from toadstool for O(1) GPU dispatches per SCF iteration,
//! replacing O(N_nuclei) sequential eigensolves.
//!
//! Architecture:
//!   1. L1 LHS screening with NMP constraint (CPU, fast)
//!   2. L2 GPU-batched HFB evaluation per optimizer step
//!   3. Statistical analysis and comparison
//!
//! Run: cargo run --release --bin nuclear_eos_l2_gpu [--nuclei=full] [--seed=42]
//!      [--max-iter=200] [--tol=0.05] [--mixing=0.3]

use hotspring_barracuda::data;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::physics::{
    nuclear_matter_properties, NuclearMatterProps,
    binding_energies_l2_gpu,
};

use barracuda::sample::latin_hypercube;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
// NMP targets
// ═══════════════════════════════════════════════════════════════════

struct NmpTarget {
    name: &'static str,
    target: f64,
    sigma: f64,
}

const NMP_TARGETS: [NmpTarget; 5] = [
    NmpTarget { name: "rho0",  target: 0.16,   sigma: 0.005 },
    NmpTarget { name: "E/A",   target: -15.97,  sigma: 0.5   },
    NmpTarget { name: "K_inf", target: 230.0,   sigma: 20.0  },
    NmpTarget { name: "m*/m",  target: 0.69,    sigma: 0.1   },
    NmpTarget { name: "J",     target: 32.0,    sigma: 2.0   },
];

fn nmp_chi2(nmp: &NuclearMatterProps) -> f64 {
    let vals = [nmp.rho0_fm3, nmp.e_a_mev, nmp.k_inf_mev, nmp.m_eff_ratio, nmp.j_mev];
    let mut chi2 = 0.0;
    for (i, &val) in vals.iter().enumerate() {
        chi2 += ((val - NMP_TARGETS[i].target) / NMP_TARGETS[i].sigma).powi(2);
    }
    chi2 / 5.0
}

const SLY4_PARAMS: [f64; 10] = [
    -2488.91, 486.82, -546.39, 13777.0,
    0.834, -0.344, -1.0, 1.354, 0.1667, 123.0,
];

// ═══════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════

struct CliArgs {
    seed: u64,
    max_iter: usize,
    tol: f64,
    mixing: f64,
    n_lhs: usize,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let get = |prefix: &str| -> Option<String> {
        args.iter().find(|a| a.starts_with(prefix)).map(|a| a[prefix.len()..].to_string())
    };

    CliArgs {
        seed: get("--seed=").and_then(|s| s.parse().ok()).unwrap_or(42),
        max_iter: get("--max-iter=").and_then(|s| s.parse().ok()).unwrap_or(200),
        tol: get("--tol=").and_then(|s| s.parse().ok()).unwrap_or(0.05),
        mixing: get("--mixing=").and_then(|s| s.parse().ok()).unwrap_or(0.3),
        n_lhs: get("--n-lhs=").and_then(|s| s.parse().ok()).unwrap_or(64),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let cli = parse_args();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Nuclear EOS L2 — GPU-Batched HFB (BatchedEighGpu)         ║");
    println!("║  toadstool BatchedEighGpu → O(1) dispatches per SCF iter   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Initialize GPU ──────────────────────────────────────────────
    let rt = tokio::runtime::Runtime::new().unwrap();
    let gpu = rt.block_on(GpuF64::new()).expect("Failed to create GPU device");
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

    // ── Load data ───────────────────────────────────────────────────
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("control/surrogate/nuclear-eos");

    let nuclei_set = data::parse_nuclei_set_from_args();
    let exp_data: HashMap<(usize, usize), (f64, f64)> =
        data::load_nuclei(&base, nuclei_set)
            .expect("Failed to load experimental data");
    let bounds =
        data::load_bounds(&base.join("wrapper/skyrme_bounds.json"))
            .expect("Failed to load parameter bounds");

    // Sort nuclei for deterministic ordering
    let mut sorted_nuclei: Vec<((usize, usize), (f64, f64))> =
        exp_data.iter().map(|(&k, &v)| (k, v)).collect();
    sorted_nuclei.sort_by_key(|&((z, n), _)| (z, n));

    let nuclei_zn: Vec<(usize, usize)> =
        sorted_nuclei.iter().map(|&((z, n), _)| (z, n)).collect();
    let b_exp_map: HashMap<(usize, usize), f64> =
        sorted_nuclei.iter().map(|&((z, n), (b, _))| ((z, n), b)).collect();

    // Count HFB-range nuclei
    let n_hfb_range = nuclei_zn.iter().filter(|(z, n)| {
        let a = z + n;
        a >= 56 && a <= 132
    }).count();

    println!("  Nuclei:      {} total ({} in HFB range A=56-132)", nuclei_zn.len(), n_hfb_range);
    println!("  Parameters:  {} dimensions (Skyrme)", bounds.len());
    println!("  SCF config:  max_iter={}, tol={:.3}, mixing={:.2}", cli.max_iter, cli.tol, cli.mixing);
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
    let gpu_result = binding_energies_l2_gpu(
        wgpu_device.clone(),
        &nuclei_zn,
        &SLY4_PARAMS,
        cli.max_iter,
        cli.tol,
        cli.mixing,
    );
    let gpu_time = t_gpu.elapsed().as_secs_f64();

    // Compute chi2
    let mut chi2_sum = 0.0;
    let mut n_valid = 0usize;
    let mut n_converged = 0usize;

    for &(z, n, b_calc, converged) in &gpu_result.results {
        if b_calc > 0.0 {
            if let Some(&b_exp) = b_exp_map.get(&(z, n)) {
                let sigma = (0.01 * b_exp).max(2.0);
                chi2_sum += ((b_calc - b_exp) / sigma).powi(2);
                n_valid += 1;
                if converged {
                    n_converged += 1;
                }
            }
        }
    }
    let chi2_datum = if n_valid > 0 { chi2_sum / n_valid as f64 } else { 1e10 };

    println!("  GPU-Batched HFB (SLy4):");
    println!("    chi2/datum:     {:.4}", chi2_datum);
    println!("    Converged:      {}/{} HFB nuclei", n_converged, gpu_result.n_hfb);
    println!("    SEMF fallback:  {} nuclei", gpu_result.n_semf);
    println!("    GPU dispatches: {}", gpu_result.gpu_dispatches);
    println!("    Wall time:      {:.2}s ({:.1}ms per HFB nucleus)",
        gpu_time, if gpu_result.n_hfb > 0 { gpu_time * 1000.0 / gpu_result.n_hfb as f64 } else { 0.0 });

    if let Some(nmp) = nuclear_matter_properties(&SLY4_PARAMS) {
        let chi2_nmp = nmp_chi2(&nmp);
        println!("    NMP chi2/datum: {:.4}", chi2_nmp);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    //  PHASE 2: LHS sweep with GPU-batched HFB objective
    // ═══════════════════════════════════════════════════════════════
    println!("══════════════════════════════════════════════════════════════");
    println!("  PHASE 2: LHS Sweep with GPU-Batched HFB ({} samples)", cli.n_lhs);
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let samples = latin_hypercube(cli.n_lhs, &bounds, cli.seed)
        .expect("LHS failed");

    let t_sweep = Instant::now();
    let mut best_chi2 = f64::MAX;
    let mut best_idx = 0usize;
    let mut total_dispatches = 0usize;

    for (i, params) in samples.iter().enumerate() {
        // Quick NMP pre-screen
        if params[8] <= 0.01 || params[8] > 1.0 { continue; }
        let nmp = match nuclear_matter_properties(params) {
            Some(n) => n,
            None => continue,
        };
        if nmp.rho0_fm3 < 0.05 || nmp.rho0_fm3 > 0.30 || nmp.e_a_mev > 0.0 { continue; }

        let result = binding_energies_l2_gpu(
            wgpu_device.clone(),
            &nuclei_zn,
            params,
            cli.max_iter,
            cli.tol,
            cli.mixing,
        );
        total_dispatches += result.gpu_dispatches;

        // chi2
        let mut chi2 = 0.0;
        let mut cnt = 0;
        for &(z, n, b_calc, _) in &result.results {
            if b_calc > 0.0 {
                if let Some(&b_exp) = b_exp_map.get(&(z, n)) {
                    let sigma = (0.01 * b_exp).max(2.0);
                    chi2 += ((b_calc - b_exp) / sigma).powi(2);
                    cnt += 1;
                }
            }
        }
        if cnt > 0 {
            let chi2_d = chi2 / cnt as f64;
            let nmp_penalty = nmp_chi2(&nmp) * 0.1;
            let total = chi2_d + nmp_penalty;
            if total < best_chi2 {
                best_chi2 = total;
                best_idx = i;
                println!("    [{:>3}] chi2_BE={:.4}  NMP={:.4}  total={:.4}  (dispatches={})",
                    i, chi2_d, nmp_penalty, total, result.gpu_dispatches);
            }
        }
    }

    let sweep_time = t_sweep.elapsed().as_secs_f64();
    println!();
    println!("  LHS Sweep complete:");
    println!("    Best total chi2: {:.4} (sample {})", best_chi2, best_idx);
    println!("    Wall time:       {:.1}s ({:.2}s per eval avg)",
        sweep_time, sweep_time / cli.n_lhs as f64);
    println!("    GPU dispatches:  {} total", total_dispatches);
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
    println!("  Nuclei:            {} ({} HFB, {} SEMF)",
        nuclei_zn.len(), n_hfb_range, nuclei_zn.len() - n_hfb_range);
    println!("  SLy4 chi2/datum:   {:.4}", chi2_datum);
    println!("  Best sweep chi2:   {:.4}", best_chi2);
    println!();
    println!("  Architecture advantage:");
    println!("    CPU (sequential eigh_f64): {} eigensolves × 200 SCF iters = {} calls",
        n_hfb_range * 2, n_hfb_range * 2 * 200);
    println!("    GPU (BatchedEighGpu):      ~5 groups × 200 iters = ~1000 dispatches");
    println!("    Reduction: ~{:.0}x fewer eigensolve calls", (n_hfb_range * 2 * 200) as f64 / 1000.0);
    println!();

    // Best params
    if best_idx < samples.len() {
        let best_params = &samples[best_idx];
        println!("  Best parameters:");
        let names = ["t0", "t1", "t2", "t3", "x0", "x1", "x2", "x3", "alpha", "W0"];
        for (i, &v) in best_params.iter().enumerate() {
            println!("    {:6} = {:>12.4}", names.get(i).unwrap_or(&"?"), v);
        }

        if let Some(nmp) = nuclear_matter_properties(best_params) {
            println!();
            println!("  Best NMP:");
            let vals = [nmp.rho0_fm3, nmp.e_a_mev, nmp.k_inf_mev, nmp.m_eff_ratio, nmp.j_mev];
            for (i, &v) in vals.iter().enumerate() {
                let pull = (v - NMP_TARGETS[i].target) / NMP_TARGETS[i].sigma;
                println!("    {:>6} = {:>10.4}  (pull: {:>+.2}sigma)", NMP_TARGETS[i].name, v, pull);
            }
        }
    }
}
