// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quick smoke test for streaming encoder at 32⁴.
//! Runs 10 trajectories, reports plaquette and timing.
//! Expected: P evolves from hot start, acceptance ~80-100%, no hang.

use barracuda::device::WgpuDevice;
use barracuda::ops::lattice::gpu_hmc_types::GpuHmcConfig;
use hotspring_barracuda::node_atomic::NodeAtomicQcd;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let n_traj: usize = std::env::var("N_TRAJ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let beta: f64 = std::env::var("BETA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.9);

    println!("═══ Streaming Encoder Smoke Test (32⁴) ═══");
    println!("  β = {beta}, {n_traj} trajectories, hot start ε=3.0");
    println!("  Omelyan 2MN, n_md=40, dt=0.0025, τ=0.1");
    println!();

    let device = Arc::new(
        hotspring_barracuda::block_on::block_on(WgpuDevice::from_env())
            .expect("GPU device creation failed"),
    );
    let gpu_name = device.adapter_info().name.clone();
    println!("  GPU: {gpu_name}");
    println!();

    let config = GpuHmcConfig {
        nt: 32,
        nx: 32,
        ny: 32,
        nz: 32,
        beta,
        mass: 1.0,
        n_md_steps: 40,
        dt: 0.0025,
        cg_tol: 1e-10,
        cg_max_iter: 1000,
        n_flavors_over_4: 0,
    };

    let qcd = NodeAtomicQcd::with_device(device, config, 42)
        .expect("NodeAtomicQcd init failed");

    qcd.upload_topology();
    qcd.seed_rng(42);
    qcd.hot_start(3.0).expect("hot_start failed");

    let mut accepted = 0usize;
    let mut plaquettes = Vec::with_capacity(n_traj);
    let mut times = Vec::with_capacity(n_traj);

    let total_start = Instant::now();

    for i in 0..n_traj {
        let t0 = Instant::now();
        let result = qcd.run_trajectory().expect("trajectory failed");
        let dt_s = t0.elapsed().as_secs_f64();

        if result.accepted {
            accepted += 1;
        }

        let plaq = result.gauge_action / (beta * 6.0 * (32u64 * 32 * 32 * 32) as f64);
        let plaq = 1.0 - plaq;
        plaquettes.push(plaq);
        times.push(dt_s);

        println!(
            "  traj {:>2}: P={:.6}, ΔH={:+.4e}, acc={}, {:.1}s",
            i + 1,
            plaq,
            result.delta_h,
            if result.accepted { "✓" } else { "✗" },
            dt_s
        );
    }

    let total_time = total_start.elapsed().as_secs_f64();
    let mean_plaq: f64 = plaquettes.iter().sum::<f64>() / plaquettes.len() as f64;
    let mean_time: f64 = times.iter().sum::<f64>() / times.len() as f64;
    let first_time = times[0];
    let rest_mean: f64 = if times.len() > 1 {
        times[1..].iter().sum::<f64>() / (times.len() - 1) as f64
    } else {
        first_time
    };

    println!();
    println!("═══ Results ═══");
    println!("  ⟨P⟩ = {mean_plaq:.6}");
    println!("  Acceptance: {accepted}/{n_traj} ({:.0}%)", 100.0 * accepted as f64 / n_traj as f64);
    println!("  First trajectory: {first_time:.1}s (includes pipeline warm-up)");
    println!("  Subsequent mean:  {rest_mean:.1}s");
    println!("  Overall mean:     {mean_time:.1}s/traj");
    println!("  Total wall time:  {total_time:.1}s");
    println!();

    if plaquettes.last().unwrap_or(&0.0) > &0.3 && (accepted as f64 / n_traj as f64) > 0.5 {
        println!("  ✓ PASS — lattice evolving, acceptance sane");
    } else {
        println!("  ✗ FAIL — plaquette stuck or acceptance too low");
        std::process::exit(1);
    }
}
