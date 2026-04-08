// SPDX-License-Identifier: AGPL-3.0-or-later

//! Production finite-temperature β-scan with multiple `N_t` values.
//!
//! Runs asymmetric `N_s³` × `N_t` lattices at β values near the deconfinement
//! transition `β_c(N_t)`. Measures plaquette, Polyakov loop magnitude and phase,
//! susceptibility, and action density. Designed for overnight/weekend runs.
//!
//! Literature `β_c` values (quenched SU(3)):
//!   `N_t=4`:  `β_c` ≈ 5.692
//!   `N_t=6`:  `β_c` ≈ 5.894
//!   `N_t=8`:  `β_c` ≈ 6.062
//!   `N_t=12`: `β_c` ≈ 6.338
//!
//! # Usage
//!
//! ```bash
//! # Single N_t scan:
//! cargo run --release --bin production_finite_temp -- \
//!   --ns=32 --nt=8 --therm=100 --meas=500
//!
//! # Resume from β index 2 (skip first 2 completed β points):
//! cargo run --release --bin production_finite_temp -- \
//!   --ns=64 --nt=8 --therm=50 --meas=100 --resume-from=2
//!
//! # Override dt for manual tuning:
//! cargo run --release --bin production_finite_temp -- \
//!   --ns=64 --nt=8 --dt=0.006 --therm=50 --meas=100
//!
//! # Multi-N_t for continuum extrapolation:
//! cargo run --release --bin production_finite_temp -- \
//!   --ns=32 --nts=4,6,8 --therm=100 --meas=500
//! ```

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming, gpu_links_to_lattice,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;

use std::io::Write;
use std::time::Instant;

fn beta_c_approx(nt: usize) -> f64 {
    match nt {
        4 => 5.692,
        6 => 5.894,
        8 => 6.062,
        10 => 6.20,
        12 => 6.338,
        16 => 6.55,
        _ => 0.95f64.mul_add((nt as f64 / 4.0).ln(), 5.692),
    }
}

fn beta_scan_points(nt: usize, n_points: usize) -> Vec<f64> {
    let bc = beta_c_approx(nt);
    let half_width = 0.15;
    let step = 2.0 * half_width / (n_points - 1).max(1) as f64;
    (0..n_points)
        .map(|i| bc - half_width + step * i as f64)
        .map(|b| (b * 1000.0).round() / 1000.0)
        .collect()
}

fn vram_estimate_gb(ns: usize, nt: usize) -> f64 {
    let vol = ns * ns * ns * nt;
    vol as f64 * 4.0 * 18.0 * 8.0 * 3.0 / 1e9
}

fn main() {
    let mut ns = 32usize;
    let mut nts: Vec<usize> = vec![8];
    let mut n_therm = 100;
    let mut n_meas = 500;
    let mut seed = 42u64;
    let mut n_beta_points = 8;
    let mut output_dir: Option<String> = None;
    let mut resume_from_beta_idx = 0usize;
    let mut dt_override: Option<f64> = None;
    let mut adaptive_dt = true;

    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--ns=") {
            ns = val.parse().expect("--ns=N");
        } else if let Some(val) = arg.strip_prefix("--nt=") {
            nts = vec![val.parse().expect("--nt=N")];
        } else if let Some(val) = arg.strip_prefix("--nts=") {
            nts = val
                .split(',')
                .map(|s| s.parse().expect("--nts=N1,N2,..."))
                .collect();
        } else if let Some(val) = arg.strip_prefix("--therm=") {
            n_therm = val.parse().expect("--therm=N");
        } else if let Some(val) = arg.strip_prefix("--meas=") {
            n_meas = val.parse().expect("--meas=N");
        } else if let Some(val) = arg.strip_prefix("--seed=") {
            seed = val.parse().expect("--seed=N");
        } else if let Some(val) = arg.strip_prefix("--nbeta=") {
            n_beta_points = val.parse().expect("--nbeta=N");
        } else if let Some(val) = arg.strip_prefix("--output-dir=") {
            output_dir = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--resume-from=") {
            resume_from_beta_idx = val.parse().expect("--resume-from=INDEX");
        } else if let Some(val) = arg.strip_prefix("--dt=") {
            dt_override = Some(val.parse().expect("--dt=0.008"));
        } else if arg == "--no-adaptive" {
            adaptive_dt = false;
        }
    }

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Production Finite-Temperature β-Scan — SU(3) Deconfinement    ║");
    println!("║  GPU Streaming HMC with DF64 Precision                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  N_s = {ns}");
    println!("  N_t values: {nts:?}");
    for &nt in &nts {
        let vol = ns * ns * ns * nt;
        println!(
            "    {}³×{}: {} sites, {:.2} GB VRAM, β_c ≈ {:.3}",
            ns,
            nt,
            vol,
            vram_estimate_gb(ns, nt),
            beta_c_approx(nt)
        );
    }
    println!("  Therm: {n_therm}, Meas: {n_meas}, β points: {n_beta_points}");
    println!();

    let total_traj: usize = nts.len() * n_beta_points * (n_therm + n_meas);
    println!("  Total trajectories: {total_traj}");

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("  GPU not available: {e}");
            std::process::exit(1);
        }
    };
    println!("  GPU: {}", gpu.adapter_name);
    println!();

    let total_start = Instant::now();
    let pipelines = GpuHmcStreamingPipelines::new_with_tmu(&gpu);

    for &nt in &nts {
        let dims = [ns, ns, ns, nt];
        let vol = ns * ns * ns * nt;
        let betas = beta_scan_points(nt, n_beta_points);

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "  {}³×{} (V={}, {:.2} GB), β_c ≈ {:.3}",
            ns,
            nt,
            vol,
            vram_estimate_gb(ns, nt),
            beta_c_approx(nt)
        );
        println!("  β scan: {betas:?}");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Omelyan HMC tuning: ΔH ∝ dt⁴ × V, target <|ΔH|> ≈ 0.5 → ~68% acceptance.
        // Calibrated from 32³×8 (V=262K): dt=0.008 gives <|ΔH|> ≈ 0.78 at β=6.0.
        let vol_f = vol as f64;
        let ref_vol = 262144.0_f64;
        let ref_dt = 0.008_f64;
        let auto_dt = (ref_dt * (ref_vol / vol_f).powf(0.25)).clamp(0.002, 0.02);
        let mut dt = dt_override.unwrap_or(auto_dt);
        let mut n_md = ((1.0 / dt).round() as usize).max(10);
        let tau = dt * n_md as f64;
        println!("  HMC: dt={dt:.6}, n_md={n_md}, τ={tau:.3}");

        let traj_log_path = output_dir
            .as_ref()
            .map(|d| format!("{d}/traj_{ns}x{nt}_nt{nt}.jsonl"));

        let mut traj_writer: Option<std::io::BufWriter<std::fs::File>> =
            traj_log_path.as_ref().map(|path| {
                let f = std::fs::File::create(path)
                    .unwrap_or_else(|e| panic!("Cannot create {path}: {e}"));
                std::io::BufWriter::new(f)
            });

        let mut nt_results: Vec<(f64, f64, f64, f64, f64, f64, f64)> = Vec::new();

        for (bi, &beta) in betas.iter().enumerate() {
            if bi < resume_from_beta_idx {
                println!(
                    "  ── β = {beta:.4} ({}/{}) ── SKIPPED (resume) ──",
                    bi + 1,
                    betas.len()
                );
                continue;
            }
            let beta_start = Instant::now();
            println!();
            println!(
                "  ── β = {beta:.4} ({}/{}) ── dt={dt:.6} n_md={n_md} ──",
                bi + 1,
                betas.len()
            );

            let mut lat = Lattice::hot_start(dims, beta, seed + bi as u64);

            let cpu_therm = if vol <= 65536 { 5 } else { 0 };
            if cpu_therm > 0 {
                let mut cfg = HmcConfig {
                    n_md_steps: n_md,
                    dt,
                    seed: seed + bi as u64 * 1000,
                    integrator: IntegratorType::Omelyan,
                };
                for _ in 0..cpu_therm {
                    hmc::hmc_trajectory(&mut lat, &mut cfg);
                }
            }

            let state = GpuHmcState::from_lattice(&gpu, &lat, beta);
            let mut rng_seed = seed * 100 + bi as u64;

            print!("    Therm ({n_therm})...");
            std::io::stdout().flush().ok();
            for i in 0..n_therm {
                let traj_start = Instant::now();
                let r = gpu_hmc_trajectory_streaming(
                    &gpu,
                    &pipelines,
                    &state,
                    n_md,
                    dt,
                    i as u32,
                    &mut rng_seed,
                )
                .expect("streaming HMC trajectory");
                let wall_us = traj_start.elapsed().as_micros() as u64;

                if let Some(ref mut w) = traj_writer {
                    let line = serde_json::json!({
                        "nt": nt, "ns": ns, "beta": beta,
                        "traj_idx": i, "is_therm": true,
                        "accepted": r.accepted, "plaquette": r.plaquette,
                        "delta_h": r.delta_h, "wall_us": wall_us,
                    });
                    writeln!(w, "{line}").ok();
                }

                if (i + 1) % 25 == 0 {
                    print!(" {}", i + 1);
                    std::io::stdout().flush().ok();
                }
            }
            println!(" done");

            let mut plaq_vals = Vec::with_capacity(n_meas);
            let mut poly_vals = Vec::with_capacity(n_meas / 10);
            let mut accepted = 0usize;

            print!("    Meas ({n_meas})...");
            std::io::stdout().flush().ok();
            for i in 0..n_meas {
                let traj_start = Instant::now();
                let r = gpu_hmc_trajectory_streaming(
                    &gpu,
                    &pipelines,
                    &state,
                    n_md,
                    dt,
                    (n_therm + i) as u32,
                    &mut rng_seed,
                )
                .expect("streaming HMC trajectory");
                let wall_us = traj_start.elapsed().as_micros() as u64;
                plaq_vals.push(r.plaquette);
                if r.accepted {
                    accepted += 1;
                }

                let do_poly = (i + 1) % 10 == 0;
                let mut poly_mag = 0.0;
                let mut poly_phase = 0.0;
                if do_poly {
                    gpu_links_to_lattice(&gpu, &state, &mut lat);
                    let (re, im) = lat.complex_polyakov_average();
                    poly_mag = re.hypot(im);
                    poly_phase = im.atan2(re);
                    poly_vals.push(poly_mag);
                }

                if let Some(ref mut w) = traj_writer {
                    let line = serde_json::json!({
                        "nt": nt, "ns": ns, "beta": beta,
                        "traj_idx": n_therm + i, "is_therm": false,
                        "accepted": r.accepted, "plaquette": r.plaquette,
                        "polyakov_mag": poly_mag, "polyakov_phase": poly_phase,
                        "delta_h": r.delta_h, "wall_us": wall_us,
                    });
                    writeln!(w, "{line}").ok();
                }

                if (i + 1) % 100 == 0 {
                    print!(" {}", i + 1);
                    std::io::stdout().flush().ok();
                }
            }
            println!(" done");

            if let Some(ref mut w) = traj_writer {
                w.flush().ok();
            }

            let mean_plaq = plaq_vals.iter().sum::<f64>() / plaq_vals.len() as f64;
            let var_plaq = plaq_vals
                .iter()
                .map(|p| (p - mean_plaq).powi(2))
                .sum::<f64>()
                / (plaq_vals.len() - 1).max(1) as f64;
            let mean_poly = if poly_vals.is_empty() {
                0.0
            } else {
                poly_vals.iter().sum::<f64>() / poly_vals.len() as f64
            };
            let susc = var_plaq * vol as f64;
            let acc_rate = accepted as f64 / n_meas as f64;
            let wall_s = beta_start.elapsed().as_secs_f64();

            nt_results.push((
                beta,
                mean_plaq,
                var_plaq.sqrt(),
                mean_poly,
                susc,
                acc_rate,
                wall_s,
            ));

            println!(
                "    ⟨P⟩={:.6}±{:.6} |L|={:.4} χ={:.4} acc={:.0}% ({:.1}s) dt={:.6}",
                mean_plaq,
                var_plaq.sqrt(),
                mean_poly,
                susc,
                acc_rate * 100.0,
                wall_s,
                dt,
            );

            if adaptive_dt && dt_override.is_none() {
                let target_acc = 0.67;
                if acc_rate < 0.40 {
                    let old_dt = dt;
                    dt *= 0.75;
                    n_md = ((1.0 / dt).round() as usize).max(10);
                    println!(
                        "    [adaptive] acc={:.0}% < 40% → dt {:.6} → {:.6}, n_md={}",
                        acc_rate * 100.0,
                        old_dt,
                        dt,
                        n_md
                    );
                } else if acc_rate < 0.55 {
                    let old_dt = dt;
                    dt *= 0.88;
                    n_md = ((1.0 / dt).round() as usize).max(10);
                    println!(
                        "    [adaptive] acc={:.0}% < 55% → dt {:.6} → {:.6}, n_md={}",
                        acc_rate * 100.0,
                        old_dt,
                        dt,
                        n_md
                    );
                } else if acc_rate > 0.85 && dt < 0.02 {
                    let old_dt = dt;
                    dt *= 1.15;
                    dt = dt.min(0.02);
                    n_md = ((1.0 / dt).round() as usize).max(10);
                    println!(
                        "    [adaptive] acc={:.0}% > 85% → dt {:.6} → {:.6}, n_md={}",
                        acc_rate * 100.0,
                        old_dt,
                        dt,
                        n_md
                    );
                }
                let _ = target_acc;
            }
        }

        println!();
        println!("  ── {ns}³×{nt} Summary ──");
        println!(
            "  {:>8} {:>10} {:>10} {:>10} {:>10} {:>8}",
            "β", "⟨P⟩", "σ(P)", "|L|", "χ", "acc%"
        );
        for &(beta, mp, sp, poly, chi, acc, _) in &nt_results {
            println!(
                "  {:>8.4} {:>10.6} {:>10.6} {:>10.4} {:>10.4} {:>7.1}%",
                beta,
                mp,
                sp,
                poly,
                chi,
                acc * 100.0
            );
        }

        if let Some(ref dir) = output_dir {
            let path = format!("{dir}/results_{ns}x{nt}.json");
            let json = serde_json::json!({
                "ns": ns, "nt": nt,
                "dims": [ns, ns, ns, nt],
                "volume": vol,
                "beta_c_lit": beta_c_approx(nt),
                "gpu": gpu.adapter_name,
                "n_therm": n_therm, "n_meas": n_meas,
                "points": nt_results.iter().map(|&(b, mp, sp, poly, chi, acc, ws)| {
                    serde_json::json!({
                        "beta": b, "mean_plaquette": mp, "std_plaquette": sp,
                        "polyakov_mag": poly, "susceptibility": chi,
                        "acceptance": acc, "wall_s": ws,
                    })
                }).collect::<Vec<_>>(),
            });
            std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
                .unwrap_or_else(|e| eprintln!("Failed to write {path}: {e}"));
            println!("  Saved: {path}");
        }
        println!();
    }

    let total_wall = total_start.elapsed().as_secs_f64();
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  Total wall time: {:.1}s ({:.1} min, {:.1} hours)",
        total_wall,
        total_wall / 60.0,
        total_wall / 3600.0,
    );
    println!("  GPU: {}", gpu.adapter_name);
}
