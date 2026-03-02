// SPDX-License-Identifier: AGPL-3.0-only

//! HMC Parameter Sweep — Exp 024
//!
//! Systematically sweeps (L, β, m, dt) to map acceptance surfaces and
//! generate NPU training data for parameter suggestion (Head 6).
//!
//! ```bash
//! cargo run --release --bin production_dynamical_sweep -- \
//!   --output=results/exp024_sweep.jsonl
//! ```

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::dynamical::GpuDynHmcState;
use hotspring_barracuda::lattice::gpu_hmc::resident_cg::{
    gpu_dynamical_hmc_trajectory_resident, GpuResidentCgBuffers, GpuResidentCgPipelines,
};
use hotspring_barracuda::lattice::gpu_hmc::streaming::GpuDynHmcStreamingPipelines;
use hotspring_barracuda::lattice::wilson::Lattice;
use std::io::Write;
use std::time::Instant;

fn main() {
    let mut output_path = "results/exp024_sweep.jsonl".to_string();
    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--output=") {
            output_path = val.to_string();
        }
    }

    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&output_path).expect("cannot create output file"),
    );

    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU required");
    let adapter_name = gpu.adapter_name.clone();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Exp 024: HMC Parameter Sweep — NPU Training Data     ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("  GPU: {adapter_name}");
    println!("  Output: {output_path}");
    println!();

    let lattices: Vec<usize> = vec![4, 8];
    let betas: Vec<f64> = vec![5.0, 5.5, 5.69, 6.0];
    let masses: Vec<f64> = vec![0.05, 0.1, 0.2, 0.5];
    let dts: Vec<f64> = vec![0.002, 0.005, 0.01, 0.02, 0.05];

    let n_therm = 5;
    let n_meas = 10;
    let cg_tol = 1e-8;
    let cg_max_iter = 5000;
    let check_interval = 10;
    let base_seed = 42u64;

    let total = lattices.len() * betas.len() * masses.len() * dts.len();
    println!(
        "  Grid: {} lattices × {} betas × {} masses × {} dts = {} points",
        lattices.len(),
        betas.len(),
        masses.len(),
        dts.len(),
        total
    );
    println!("  Trajectories per point: {n_therm} therm + {n_meas} meas");
    println!();

    let sweep_start = Instant::now();
    let mut point_idx = 0usize;

    for &l in &lattices {
        let dims = [l, l, l, l];
        let vol: usize = dims.iter().product();

        for &beta in &betas {
            for &mass in &masses {
                for &dt in &dts {
                    point_idx += 1;
                    let n_md = ((1.0 / dt).round() as usize).max(10);
                    let traj_length = dt * n_md as f64;

                    print!("[{point_idx:3}/{total}] L={l} β={beta:.2} m={mass:.2} dt={dt:.4} n_md={n_md}...");
                    std::io::stdout().flush().ok();

                    let point_start = Instant::now();
                    let mut seed = base_seed + point_idx as u64 * 1000;
                    let lat = Lattice::hot_start(dims, beta, seed);

                    // Skip if volume is too large for a quick sweep
                    if vol > 65536 {
                        println!(" SKIP (vol too large)");
                        continue;
                    }

                    let streaming = GpuDynHmcStreamingPipelines::new(&gpu);
                    let dyn_state =
                        GpuDynHmcState::from_lattice(&gpu, &lat, beta, mass, cg_tol, cg_max_iter);
                    let resident = GpuResidentCgPipelines::new(&gpu);
                    let cg_bufs =
                        GpuResidentCgBuffers::new(&gpu, &streaming.dyn_hmc, &resident, &dyn_state);

                    // Thermalization
                    for i in 0..n_therm {
                        let _r = gpu_dynamical_hmc_trajectory_resident(
                            &gpu,
                            &streaming,
                            &resident,
                            &dyn_state,
                            &cg_bufs,
                            n_md,
                            dt,
                            i as u32,
                            &mut seed,
                            check_interval,
                        );
                    }

                    // Measurement
                    let mut delta_hs = Vec::with_capacity(n_meas);
                    let mut acceptances = Vec::with_capacity(n_meas);
                    let mut cg_iters_list = Vec::with_capacity(n_meas);
                    let mut plaquettes = Vec::with_capacity(n_meas);
                    let mut traj_times = Vec::with_capacity(n_meas);

                    for i in 0..n_meas {
                        let traj_start = Instant::now();
                        let r = gpu_dynamical_hmc_trajectory_resident(
                            &gpu,
                            &streaming,
                            &resident,
                            &dyn_state,
                            &cg_bufs,
                            n_md,
                            dt,
                            (n_therm + i) as u32,
                            &mut seed,
                            check_interval,
                        );
                        let traj_wall = traj_start.elapsed().as_secs_f64();

                        delta_hs.push(r.delta_h);
                        acceptances.push(r.accepted);
                        cg_iters_list.push(r.cg_iterations);
                        plaquettes.push(r.plaquette);
                        traj_times.push(traj_wall);
                    }

                    let wall_s = point_start.elapsed().as_secs_f64();
                    let n_acc = acceptances.iter().filter(|&&a| a).count();
                    let acc_rate = n_acc as f64 / n_meas as f64;
                    let mean_dh: f64 = delta_hs.iter().sum::<f64>() / n_meas as f64;
                    let abs_mean_dh: f64 =
                        delta_hs.iter().map(|d| d.abs()).sum::<f64>() / n_meas as f64;
                    let std_dh = {
                        let var = delta_hs.iter().map(|d| (d - mean_dh).powi(2)).sum::<f64>()
                            / (n_meas as f64 - 1.0).max(1.0);
                        var.sqrt()
                    };
                    let mean_cg: f64 = cg_iters_list.iter().sum::<usize>() as f64 / n_meas as f64;
                    let mean_plaq: f64 = plaquettes.iter().sum::<f64>() / n_meas as f64;
                    let mean_traj_time: f64 = traj_times.iter().sum::<f64>() / n_meas as f64;
                    let throughput = if mean_traj_time > 0.0 {
                        1.0 / mean_traj_time
                    } else {
                        0.0
                    };
                    let effective_throughput = acc_rate * throughput;

                    let result = serde_json::json!({
                        "lattice": l,
                        "volume": vol,
                        "beta": beta,
                        "mass": mass,
                        "dt": dt,
                        "n_md": n_md,
                        "traj_length": traj_length,
                        "cg_tol": cg_tol,
                        "cg_max_iter": cg_max_iter,
                        "check_interval": check_interval,
                        "n_therm": n_therm,
                        "n_meas": n_meas,
                        "acceptance_rate": acc_rate,
                        "mean_delta_h": mean_dh,
                        "abs_mean_delta_h": abs_mean_dh,
                        "std_delta_h": std_dh,
                        "mean_cg_iters": mean_cg,
                        "mean_plaquette": mean_plaq,
                        "mean_traj_wall_s": mean_traj_time,
                        "throughput_traj_per_s": throughput,
                        "effective_throughput": effective_throughput,
                        "total_wall_s": wall_s,
                        "gpu": &adapter_name,
                    });

                    writeln!(out, "{}", serde_json::to_string(&result).unwrap()).ok();
                    out.flush().ok();

                    let status = if acc_rate > 0.6 {
                        "✓"
                    } else if acc_rate > 0.0 {
                        "~"
                    } else {
                        "✗"
                    };

                    println!(
                        " {status} acc={:.0}% |ΔH|={abs_mean_dh:.1} CG={mean_cg:.0} ({wall_s:.1}s)",
                        acc_rate * 100.0
                    );
                }
            }
        }
    }

    out.flush().ok();
    let total_wall = sweep_start.elapsed().as_secs_f64();

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  Sweep complete: {point_idx} points in {:.0}s ({:.1} min)",
        total_wall,
        total_wall / 60.0
    );
    println!("  Output: {output_path}");
    println!("═══════════════════════════════════════════════════════════");
}
